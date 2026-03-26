import numpy as np
from typing import Dict, Set, Any, Optional, List, Tuple
from collections import defaultdict
import logging
import re
import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.llm_conversation_logger import get_llm_log_dir

logger = logging.getLogger("EntityAlignment")

def build_bucket_signature(bucket: Dict[str, Any]) -> str:

    concept = bucket.get("concept", "").strip()
    definition = bucket.get("definition", "").strip()
    examples = bucket.get("examples", [])


    real_examples = [
        ex for ex in examples
        if not (
            (ex.startswith("(") and ex.endswith(")")) or
            "未明确" in ex or
            "未提及" in ex or
            "未具体" in ex or
            ex.strip() == ""
        )
    ]

    parts = []
    if concept:
        parts.append(concept)
    if definition and not any(phrase in definition for phrase in ["未明确", "未提及", "未具体"]):
        parts.append(definition)
    if real_examples:

        parts.append("；".join(real_examples[:3]))

    signature = "。".join(parts)
    return signature if signature else concept

def find_semantic_cross_groups(
    buckets_by_id: Dict[str, Dict[str, Any]],
    similarity_threshold: float,
    min_group_size: int,
    max_group_size: Optional[int] = None,
) -> Dict[str, Set[str]]:


    if len(buckets_by_id) < 2:
        return {}

    logger.info("🔍 开始语义分组（使用 TF-IDF + 层次聚类）...")


    valid_ids = []
    signatures = []

    for bid, bucket in buckets_by_id.items():
        sig = build_bucket_signature(bucket)
        if sig.strip():
            valid_ids.append(bid)
            signatures.append(sig)

    if len(signatures) < min_group_size:
        return {}


    try:
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 2),
            min_df=1,
            norm='l2'
        )
        tfidf_matrix = vectorizer.fit_transform(signatures)
        vectors = tfidf_matrix.toarray()
    except Exception as e:
        logger.warning(f"TF-IDF 向量化失败: {e}")
        return {}


    distance_matrix = cosine_distances(vectors)


    np.clip(distance_matrix, 0.0, 2.0, out=distance_matrix)


    try:
        rows = []
        n = len(valid_ids)
        sim_matrix = 1.0 - distance_matrix

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= similarity_threshold:
                     rows.append({
                         "id1": valid_ids[i],
                         "sig1": signatures[i],
                         "id2": valid_ids[j],
                         "sig2": signatures[j],
                         "similarity": sim
                     })

        if rows:
            df = pd.DataFrame(rows)
            out_dir = get_llm_log_dir() or os.path.join("output")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "stage2_semantic_similarity.xlsx")
            df.to_excel(out_path, index=False)
            logger.info(f"📝 语义相似度审查文件已写出：{out_path}（记录 {len(rows)} 对，阈值>={similarity_threshold}）")
        else:
            logger.info("📝 无满足阈值的语义相似对，Excel不写出。")
    except Exception as e:
        logger.warning(f"写出语义相似度Excel失败：{e}")


    distance_threshold = 1.0 - similarity_threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='precomputed',
        linkage='complete'
    )
    labels = clustering.fit_predict(distance_matrix)


    groups = defaultdict(set)
    for bid, label in zip(valid_ids, labels):
        groups[label].add(bid)


    semantic_groups = {}
    group_idx = 0
    for members in groups.values():
        if len(members) < min_group_size:
            continue
        if max_group_size and len(members) > max_group_size:


            continue

        semantic_groups[f"__SEMANTIC_GROUP_{group_idx}__"] = members
        group_idx += 1

    logger.info(f"📦 层次聚类生成 {len(semantic_groups)} 个语义组。")
    if len(semantic_groups) == 0:
        logger.info("🔍 未发现满足相似度阈值的语义组。")
    return semantic_groups
