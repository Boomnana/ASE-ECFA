import pandas as pd
import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TRIPLETS_FILE = BASE_DIR / "output" / "triples.xlsx"
CLUSTERS_FILE = BASE_DIR / "output" / "aligned_clusters.xlsx"
OPTIMIZED_FILE = BASE_DIR / "output" / "optimized_triplets.xlsx"
OUTPUT_DIR = BASE_DIR / "graph_output"

parser = argparse.ArgumentParser()
parser.add_argument("--allowed-types", default="功能模块,用户操作,用户感知现象,系统诊断信息,影响元素,问题陈述")
parser.add_argument("--node-id-prefix", default="1011")
parser.add_argument("--node-id-width", type=int, default=6)

args = parser.parse_args()


ALLOWED_TYPES = set([s.strip() for s in args.allowed_types.split(",") if s.strip()])
NODE_ID_PREFIX = args.node_id_prefix
NODE_ID_WIDTH = args.node_id_width
OPTIMIZED_FILE.parent.mkdir(parents=True, exist_ok=True)

triplets = pd.read_excel(TRIPLETS_FILE)
clusters = pd.read_excel(CLUSTERS_FILE)


cluster_map = {}
for _, row in clusters.iterrows():
    cluster = row["cluster"]
    entity = row["entity"]
    if cluster not in cluster_map:
        cluster_map[cluster] = entity


entity_to_rep = {}
for _, row in clusters.iterrows():
    entity_to_rep[row["entity"]] = cluster_map[row["cluster"]]


triplets["subject"] = triplets["subject"].map(lambda x: entity_to_rep.get(x, x))

triplets["object"] = triplets["object"].map(lambda x: entity_to_rep.get(x, x))


triplets.to_excel(OPTIMIZED_FILE, index=False)


shared_output_dir = REPO_ROOT / "shared_output"
shared_output_dir.mkdir(parents=True, exist_ok=True)
triplets.to_excel(shared_output_dir / "optimized_triplets.xlsx", index=False)
print(f"✅ Optimized triplets also saved to {shared_output_dir / 'optimized_triplets.xlsx'}")

print("✅ 优化完成！已生成 optimized_triplets.xlsx")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


triplets = pd.read_excel(OPTIMIZED_FILE)
print(f"✅ 已加载 {len(triplets)} 条三元组数据")


node_source_map = {}

for _, row in triplets.iterrows():

    subj_key = (row["subject"], row["subject_type"])
    if row["subject_type"] in ALLOWED_TYPES:
        if subj_key not in node_source_map:
            node_source_map[subj_key] = []
        node_source_map[subj_key].append(str(row["source_row_index"]))


    obj_key = (row["object"], row["object_type"])
    if row["object_type"] in ALLOWED_TYPES:
        if obj_key not in node_source_map:
            node_source_map[obj_key] = []
        node_source_map[obj_key].append(str(row["source_row_index"]))


node_id_map = {}
nodes = []
node_counter = 0

for (name, node_type), indices in node_source_map.items():

    node_id = f"{NODE_ID_PREFIX}{node_counter:0{NODE_ID_WIDTH}d}"
    node_counter += 1


    source_row_index = "|".join(sorted(set(indices)))


    node_id_map[(name, node_type)] = node_id

    nodes.append({
        "id": node_id,
        "name": name,
        "type": node_type,
        "source_row_index": source_row_index
    })


edges = []
edge_counter = 0

for _, row in triplets.iterrows():

    source_key = (row["subject"], row["subject_type"])
    target_key = (row["object"], row["object_type"])

    source_id = node_id_map.get(source_key)
    target_id = node_id_map.get(target_key)


    if (row["subject_type"] in ALLOWED_TYPES and
        row["object_type"] in ALLOWED_TYPES and
        source_id is not None and
        target_id is not None):
        edges.append({
            "id": edge_counter,
            "source_id": source_id,
            "target_id": target_id,
            "relation": row["relation"],
            "source_type": row["subject_type"],
            "target_type": row["object_type"],
            "source_row_index": str(row["source_row_index"])
        })
        edge_counter += 1


nodes_df = pd.DataFrame(nodes)
edges_df = pd.DataFrame(edges)

nodes_path = OUTPUT_DIR / "nodes.xlsx"
edges_path = OUTPUT_DIR / "edges.xlsx"
with pd.ExcelWriter(str(nodes_path), engine="openpyxl") as writer:
    nodes_df.to_excel(writer, sheet_name="nodes", index=False)
with pd.ExcelWriter(str(edges_path), engine="openpyxl") as writer:
    edges_df.to_excel(writer, sheet_name="edges", index=False)


shared_output_dir = REPO_ROOT / "shared_output"
shared_output_dir.mkdir(parents=True, exist_ok=True)
nodes_df.to_excel(shared_output_dir / "nodes.xlsx", index=False)
edges_df.to_excel(shared_output_dir / "edges.xlsx", index=False)
print(f"✅ Nodes and edges also saved to {shared_output_dir}")

print(f"🎉 节点文件: {str(nodes_path)}")
print(f"🎉 边文件: {str(edges_path)}")
print("✅ 所有节点ID已生成10位格式，边ID已正确关联（已过滤类型）！")
