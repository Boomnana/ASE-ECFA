

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import pandas as pd
import jieba
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv


BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parents[1]
dotenv_path = REPO_ROOT / ".env"
load_dotenv(dotenv_path=dotenv_path if dotenv_path.exists() else None)


PROJECT_ROOT = BASE_DIR
OUTPUT_DIR = PROJECT_ROOT / "output"
GRAPH_OUTPUT_DIR = PROJECT_ROOT / "graph_output"


DEFAULT_CONFIG = {
    "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
    "INPUT_FILE": OUTPUT_DIR / "standardized_triples.xlsx",
    "GRAPH_NODES_FILE": GRAPH_OUTPUT_DIR / "nodes.xlsx",
    "GRAPH_EDGES_FILE": GRAPH_OUTPUT_DIR / "edges.xlsx",
    "OUTPUT_DIR": OUTPUT_DIR,
    "ERROR_FILE": "error_triples.xlsx",
    "STOPWORDS_FILE": (REPO_ROOT / "stopwords.txt") if (REPO_ROOT / "stopwords.txt").exists() else (PROJECT_ROOT / "stopwords.txt"),
    "ALLOWED_TYPES": {"功能模块", "影响元素", "用户感知现象", "用户操作", "系统诊断信息", "问题陈述"},
    "LOG_LEVEL": logging.INFO,
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 1000)),
}


if not DEFAULT_CONFIG["NEO4J_PASSWORD"]:
    raise ValueError("❌ NEO4J_PASSWORD 必须通过环境变量设置！")


logging.basicConfig(
    level=DEFAULT_CONFIG["LOG_LEVEL"],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Neo4jImporter")


def load_stopwords(stopwords_path: Path) -> set:

    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        logger.warning(f"停用词表未找到: {stopwords_path}，将使用空停用词表")
        return set()


def escape_label(label: str) -> str:


    if not isinstance(label, str) or not label.strip():
        raise ValueError("标签不能为空或非字符串")

    safe_label = label.strip().replace("`", "\\`")
    return safe_label


def tokenize(text: str, stopwords: set) -> List[str]:

    if not isinstance(text, str):
        return []
    words = jieba.lcut(text)
    return [w for w in words if w.strip() and w not in stopwords]


def chunked(iterable: List[Any], size: int) -> Iterator[List[Any]]:

    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


class Neo4jImporter:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._validate_config()
        self.stopwords = load_stopwords(self.config["STOPWORDS_FILE"])
        self.error_triples_list: List[Dict] = []
        self.driver = None

    def _validate_config(self):

        required = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
        for key in required:
            if not self.config.get(key):
                raise ValueError(f"配置缺失: {key}")


        path_keys = ["INPUT_FILE", "GRAPH_NODES_FILE", "GRAPH_EDGES_FILE", "STOPWORDS_FILE", "OUTPUT_DIR"]
        for k in path_keys:
            if isinstance(self.config[k], str):
                self.config[k] = Path(self.config[k])

    def _connect(self):

        try:
            self.driver = GraphDatabase.driver(
                self.config["NEO4J_URI"],
                auth=(self.config["NEO4J_USER"], self.config["NEO4J_PASSWORD"])
            )
            self.driver.verify_connectivity()
            logger.info("✅ 成功连接到 Neo4j 数据库")
        except ServiceUnavailable as e:
            logger.error(f"❌ 无法连接 Neo4j: {e}")
            raise RuntimeError("请确保 Neo4j 已启动并可访问") from e

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("🔌 Neo4j 连接已关闭")


    def clear_database(self):

        with self.driver.session() as session:
            session.run("MATCH ()-[r]->() DELETE r")
            session.run("MATCH (n) DELETE n")
        logger.info("✅ 数据库已清空")

    def create_indexes(self):

        with self.driver.session() as session:
            logger.info("🔧 创建索引和约束...")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Text) ON (t.value)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (w:Word) ON (w.value)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:缺陷报告) ON (d.name)")

            for entity_type in self.config["ALLOWED_TYPES"]:
                label = escape_label(entity_type)
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (e:`{label}`) REQUIRE e.id IS UNIQUE")
                session.run(f"CREATE INDEX IF NOT EXISTS FOR (e:`{label}`) ON (e.name)")
        logger.info("✅ 索引和约束创建完成")


    def import_graph_nodes(self, nodes_file_path: Path):
        df = pd.read_excel(
            nodes_file_path,
            dtype={
                "id": "Int64",
                "name": "string",
                "type": "string",
                "source_row_index": "string"
            },
            keep_default_na=True,
            na_values=["", "N/A", "NULL", "#N/A"]
        )
        required_cols = {"id", "name", "type", "source_row_index"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"nodes.xlsx 缺少列: {sorted(missing)}")


        df = df.dropna(subset=["id", "type"])

        invalid_types = set(df["type"].dropna().unique()) - self.config["ALLOWED_TYPES"]
        if invalid_types:
            logger.warning(f"跳过非法实体类型: {sorted(invalid_types)}")
            df = df[df["type"].isin(self.config["ALLOWED_TYPES"])]

        logger.info(f"📥 准备导入 {len(df)} 个节点")

        with self.driver.session() as session:
            for entity_type, group in df.groupby("type"):
                label = escape_label(entity_type)
                rows = [
                    {
                        "id": int(r["id"]),
                        "name": str(r["name"]) if pd.notna(r["name"]) else "",
                        "source_row_index": None if pd.isna(r["source_row_index"]) else str(r["source_row_index"])
                    }
                    for _, r in group.iterrows()
                ]

                query = f"""
                UNWIND $rows AS row
                MERGE (n:`{label}` {{id: row.id}})
                SET n.name = row.name,
                    n.source_row_index = row.source_row_index
                """
                session.run(query, rows=rows)

        logger.info("✅ 节点导入完成")

    def import_graph_edges(self, edges_file_path: Path):
        df = pd.read_excel(
            edges_file_path,
            dtype={
                "id": "Int64",
                "source_id": "Int64",
                "target_id": "Int64",
                "relation": "string",
                "source_type": "string",
                "target_type": "string",
                "source_row_index": "Int64"
            },
            keep_default_na=True,
            na_values=["", "N/A", "NULL", "#N/A"]
        )
        required_cols = {"id", "source_id", "target_id", "relation", "source_type", "target_type", "source_row_index"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"edges.xlsx 缺少列: {sorted(missing)}")

        df = df.dropna(subset=["id", "source_id", "target_id", "relation", "source_type", "target_type"])

        mask = (
            df["source_type"].isin(self.config["ALLOWED_TYPES"]) &
            df["target_type"].isin(self.config["ALLOWED_TYPES"])
        )
        invalid_count = (~mask).sum()
        if invalid_count > 0:
            logger.warning(f"跳过 {invalid_count} 条含非法类型的边")
            df = df[mask].copy()

        logger.info(f"📥 准备导入 {len(df)} 条边")

        with self.driver.session() as session:
            for (relation, source_type, target_type), group in df.groupby(["relation", "source_type", "target_type"]):
                rel_type = escape_label(relation)
                source_label = escape_label(source_type)
                target_label = escape_label(target_type)

                rows = [
                    {
                        "edge_id": int(r["id"]),
                        "source_id": int(r["source_id"]),
                        "target_id": int(r["target_id"]),
                        "source_row_index": r["source_row_index"]
                    }
                    for _, r in group.iterrows()
                ]

                query = f"""
                UNWIND $rows AS row
                MATCH (a:`{source_label}` {{id: row.source_id}})
                MATCH (b:`{target_label}` {{id: row.target_id}})
                MERGE (a)-[r:`{rel_type}` {{id: row.edge_id}}]->(b)
                SET r.source_row_index = row.source_row_index
                """
                session.run(query, rows=rows)

        with self.driver.session() as session:
            count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        logger.info(f"✅ 边导入完成 (总数: {count})")

    def export_nodes_and_edges(self):
        logger.info("📤 开始导出图数据...")
        self.config["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

        with self.driver.session() as session:
            nodes = session.run("""
            MATCH (n)
            RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS props
            """).data()
            nodes_df = pd.json_normalize(nodes)

            edges = session.run("""
            MATCH (a)-[r]->(b)
            RETURN
                elementId(a) AS src_id, elementId(b) AS tgt_id,
                type(r) AS rel_type, properties(r) AS props,
                labels(a) AS src_labels, labels(b) AS tgt_labels
            """).data()
            edges_df = pd.json_normalize(edges)

        nodes_path = self.config["OUTPUT_DIR"] / "exported_nodes.xlsx"
        edges_path = self.config["OUTPUT_DIR"] / "exported_edges.xlsx"
        nodes_df.to_excel(nodes_path, index=False)
        edges_df.to_excel(edges_path, index=False)
        logger.info(f"✅ 节点 ({len(nodes_df)}) 和边 ({len(edges_df)}) 已导出")

    def run_graph_output_import(self, clear_db: bool = True, export_data: bool = True):
        if clear_db:
            self.clear_database()
        self.create_indexes()
        self.import_graph_nodes(self.config["GRAPH_NODES_FILE"])
        self.import_graph_edges(self.config["GRAPH_EDGES_FILE"])
        if export_data:
            self.export_nodes_and_edges()
        logger.info("🎉 graph_output 导入完成！")

def main():

    try:
        with Neo4jImporter() as importer:

            importer.run_graph_output_import(clear_db=True, export_data=True)


        print("✅ 执行成功！")
    except Exception as e:
        logger.exception("💥 程序异常终止")
        raise


if __name__ == "__main__":
    main()
