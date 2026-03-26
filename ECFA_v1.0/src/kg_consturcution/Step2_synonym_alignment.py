import asyncio
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from llm_client import LLMClient, LLMSettings
from utils.llm_conversation_logger import setup_llm_logger
from entity_alignment.common import read_entities_from_file, load_fewshot
from entity_alignment.stage1_bucketing import semantic_bucketing
from entity_alignment.stage2_merging import merge_buckets
from entity_alignment.stage3_alignment import intra_bucket_alignment
from entity_alignment.stage4_clustering import closure_and_clusters


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
run_dir = BASE_DIR / "output" / f"alignment_files_{timestamp}"
run_dir.mkdir(parents=True, exist_ok=True)


setup_llm_logger(str(run_dir))

parser = argparse.ArgumentParser(description="实体对齐流水线")
parser.add_argument("--llm-config", default=str(REPO_ROOT / "LLM-config.yaml"), help="LLM 配置文件路径")


parser.add_argument("--input-file", default=str(BASE_DIR / "output" / "triples.xlsx"), help="输入实体文件路径")
parser.add_argument("--output-buckets", default=str(run_dir / "semantic_buckets.json"), help="阶段1输出：语义分桶结果")
parser.add_argument("--output-pairs", default=str(run_dir / "alignment_pairs.json"), help="阶段3输出：对齐对结果")
parser.add_argument("--output-clusters", default=str(run_dir / "aligned_clusters.xlsx"), help="阶段4输出：最终聚类结果")


parser.add_argument("--coarse-size", type=int, default=200, help="阶段1粗分块大小（每批实体数）")


parser.add_argument("--bucket-limit", type=int, default=150, help="阶段3单桶内最大实体数")


parser.add_argument("--batch-validation-size", type=int, default=20, help="阶段3批量验证对数")


parser.add_argument("--entity-type-filter", default="", help="逗号分隔的类型过滤（如: 功能模块,影响元素）")


parser.add_argument("--prefix-len", type=int, default=3, help="粗分桶时使用的前缀长度")


parser.add_argument("--fewshot-file", default="", help="可选，提供 few-shot 示例以强化 LLM 判断标准")

parser.add_argument("--stage2-iterative", action="store_true", help="是否在阶段2启用多轮迭代合并（默认关闭）")
parser.add_argument("--stage2-max-rounds", type=int, default=3, help="阶段2最大迭代轮数（仅当 --stage2-iterative 启用时生效）")
args = parser.parse_args()

settings = LLMSettings.from_yaml(args.llm_config)
llm = LLMClient.from_yaml(args.llm_config).build()
semaphore = asyncio.Semaphore(settings.concurrency)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EntityAlignment")


async def run_pipeline() -> Dict[str, Any]:

    input_filter = args.entity_type_filter.strip()
    issue_type = "问题陈述"


    default_std_types = ["功能模块", "用户操作", "用户感知现象", "系统诊断信息", "影响元素"]

    std_filter = ""
    issue_filter = ""

    if not input_filter:

        std_filter = ",".join(default_std_types)
        issue_filter = issue_type
    else:

        types = [t.strip() for t in input_filter.split(",") if t.strip()]
        std_types = [t for t in types if t != issue_type]
        has_issue = issue_type in types

        std_filter = ",".join(std_types)
        if has_issue:
            issue_filter = issue_type


    entities_std = []
    if std_filter:
        try:
            entities_std = read_entities_from_file(args.input_file, std_filter)
        except Exception:
            logger.warning("读取标准类型实体为空或失败")

    entities_issue = []
    if issue_filter:
        try:
            entities_issue = read_entities_from_file(args.input_file, issue_filter)
        except Exception:
             logger.warning("读取问题陈述实体为空或失败")


    all_entities = sorted(list(set(entities_std + entities_issue)))

    logger.info(f"启动实体对齐流水线，总实体: {len(all_entities)} (标准: {len(entities_std)}, 问题: {len(entities_issue)})")
    logger.info(f"配置: 粗分块大小={args.coarse_size}, 桶内上限={args.bucket_limit}")

    fewshot = load_fewshot(args.fewshot_file)
    type_hint = args.entity_type_filter if args.entity_type_filter else None

    buckets: Dict[str, List[str]] = {}
    ambiguous: Dict[str, List[str]] = {}
    pairs: List[Tuple[str, str]] = []


    buckets_std = {}
    ambiguous_std = {}
    buckets_by_id_total = {}
    concept_to_ids_total = {}

    if entities_std:
        logger.info(">>> 开始处理标准类型实体分桶...")
        buckets_std, ambiguous_std, aux_std = await semantic_bucketing(
            entities=entities_std,
            coarse_size=args.coarse_size,
            prefix_len=args.prefix_len,
            llm=llm,
            semaphore=semaphore,
            type_hint=std_filter,
            fewshot_concept=fewshot.get("concept_examples"),
            logger=logger,
            id_prefix="std_"
        )
        buckets.update(buckets_std)
        ambiguous.update(ambiguous_std)
        buckets_by_id_total.update(aux_std.get("buckets_by_id", {}))
        concept_to_ids_total.update(aux_std.get("concept_to_ids", {}))


    if entities_issue:
        logger.info(">>> 开始处理问题陈述实体分桶...")
        buckets_issue, ambiguous_issue, aux_issue = await semantic_bucketing(
            entities=entities_issue,
            coarse_size=args.coarse_size,
            prefix_len=args.prefix_len,
            llm=llm,
            semaphore=semaphore,
            type_hint=issue_type,
            fewshot_concept=fewshot.get("concept_examples"),
            logger=logger,
            id_prefix="issue_"
        )

        for k, v in buckets_issue.items():
            if k in buckets:
                buckets[k] = sorted(list(set(buckets[k] + v)))
            else:
                buckets[k] = v

        ambiguous.update(ambiguous_issue)
        buckets_by_id_total.update(aux_issue.get("buckets_by_id", {}))

        for k, v in aux_issue.get("concept_to_ids", {}).items():
            if k in concept_to_ids_total:
                concept_to_ids_total[k].extend(v)
            else:
                concept_to_ids_total[k] = v

    out_buckets = Path(args.output_buckets)
    out_buckets.parent.mkdir(parents=True, exist_ok=True)
    Path(out_buckets).write_text(json.dumps({"buckets": buckets, "ambiguous": ambiguous}, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        stage1_batches_path = Path(run_dir) / "stage1_batches.json"
        stage1_batches_path.parent.mkdir(parents=True, exist_ok=True)

        buckets_by_id = buckets_by_id_total
        stage1_batches_path.write_text(json.dumps(buckets_by_id, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"保存 stage1_batches.json 失败: {e}")

    logger.info(f"阶段1完成：概念桶数量 {len(buckets)}，歧义实体数量 {len(ambiguous)}，结果已写入 {out_buckets}")


    if buckets_by_id:
        logger.info(f"阶段2开始：共 {len(buckets_by_id)} 个原始桶，准备进行语义合并...")
        buckets, merged_buckets_by_id = await merge_buckets(
            buckets_by_id=buckets_by_id,
            llm=llm,
            semaphore=semaphore,
            iterative=args.stage2_iterative,
            max_rounds=args.stage2_max_rounds,
            max_context_tokens=settings.max_tokens or 8000
        )
        logger.info(f"阶段2完成：合并后剩余 {len(buckets)} 个概念桶。")
        stage2_output = run_dir / "stage2_merged_buckets.json"
        with open(stage2_output, "w", encoding="utf-8") as f:
            json.dump(buckets, f, ensure_ascii=False, indent=2)
        logger.info(f"阶段2合并结果已额外保存至: {stage2_output}")
    else:
        logger.info("阶段2跳过：未找到 buckets_by_id，无法执行桶间合并。")


    tasks = [
        intra_bucket_alignment(
            concept=concept,
            entities=ents,
            limit=args.bucket_limit,
            llm=llm,
            semaphore=semaphore,
            type_hint=type_hint,
            fewshot_pairs=fewshot.get("pair_examples"),
            batch_validation_size=args.batch_validation_size,
        )
        for concept, ents in buckets.items()
    ]
    results = await asyncio.gather(*tasks)
    for sub in results:
        pairs.extend(sub)

    out_pairs = Path(args.output_pairs)
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    Path(out_pairs).write_text(json.dumps([[a, b] for a, b in pairs], ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"阶段3完成：生成等价对 {len(pairs)} 对，结果已写入 {out_pairs}")


    clusters: Dict[str, List[str]] = closure_and_clusters(pairs, all_entities=all_entities)
    rows = []
    for std, members in clusters.items():
        for m in members:
            rows.append({"cluster": std, "entity": m})
    df_out = pd.DataFrame(rows)
    out_clusters = Path(args.output_clusters)
    out_clusters.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(str(out_clusters), index=False)

    fixed_output_path = BASE_DIR / "output" / "aligned_clusters.xlsx"
    fixed_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(str(fixed_output_path), index=False)
    logger.info(f"阶段4聚类结果已额外保存至根 output 目录: {fixed_output_path}")

    stage4_output = run_dir / "stage4_final_clusters.json"
    with open(stage4_output, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    logger.info(f"阶段4聚类结果已额外保存至: {stage4_output}")
    logger.info(f"阶段4完成：形成 {len(clusters)} 个最终聚类，结果已写入 {out_clusters}")

    return {
        "bucket_count": len(buckets),
        "pair_count": len(pairs),
        "cluster_count": len(clusters)
    }


def main():
    stats = asyncio.run(run_pipeline())
    print(f"最终结果 → 分桶数: {stats['bucket_count']} | 对齐对数: {stats['pair_count']} | 聚类数: {stats['cluster_count']}")


if __name__ == "__main__":
    main()
