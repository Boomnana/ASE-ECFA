

import unicodedata
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd

def normalize_text(s: str) -> str:


    return unicodedata.normalize("NFKC", str(s)).strip()

def read_entities_from_file(path: str, entity_type_filter: str = "") -> List[str]:
    p = Path(path)
    if p.suffix.lower() not in [".xlsx", ".xls"]:
        raise ValueError("输入文件必须为 Excel（.xlsx/.xls）")
    df = pd.read_excel(str(p))
    subj_col = next((c for c in df.columns if str(c).lower() == "subject"), None)
    obj_col = next((c for c in df.columns if str(c).lower() == "object"), None)
    subj_t_col = next((c for c in df.columns if str(c).lower() == "subject_type"), None)
    obj_t_col = next((c for c in df.columns if str(c).lower() == "object_type"), None)
    if not all([subj_col, obj_col, subj_t_col, obj_t_col]):
        raise ValueError("Excel 缺少必需列：subject、object、subject_type、object_type")
    type_filter = [t.strip() for t in str(entity_type_filter).split(",") if t.strip()]
    s_vals = df[subj_col].astype(str)
    o_vals = df[obj_col].astype(str)
    s_types = df[subj_t_col].astype(str)
    o_types = df[obj_t_col].astype(str)
    items: List[str] = []
    if not type_filter:
        items.extend(s_vals.tolist())
        items.extend(o_vals.tolist())
    else:
        for i in range(len(df)):
            if s_types.iloc[i] in type_filter:
                items.append(s_vals.iloc[i])
            if o_types.iloc[i] in type_filter:
                items.append(o_vals.iloc[i])
    series = pd.Series(items, dtype="object").astype(str).map(normalize_text)
    series = series[series != ""]
    entities = series.drop_duplicates().tolist()
    return sorted(entities, key=lambda s: normalize_text(s).lower())

def coarse_chunks(entities: List[str], prefix_len: int, size: int) -> List[List[str]]:


    buckets: Dict[str, List[str]] = {}
    for e in entities:
        k = normalize_text(e).lower()
        if k.startswith("选择"):
            tail = k[2:]
            key = ("选择" + tail[: max(0, prefix_len - 2)]) if tail else "选择"
        else:
            key = k[: prefix_len] if len(k) >= prefix_len else k
        buckets.setdefault(key, []).append(e)
    chunks: List[List[str]] = []
    for _, items in buckets.items():
        sub = []
        for e in items:
            sub.append(e)
            if len(sub) >= size:
                chunks.append(sub)
                sub = []
        if sub:
            chunks.append(sub)
    if not chunks:
        res = [entities[i:i+size] for i in range(0, len(entities), size)]
        single_count = sum(1 for c in res if len(c) == 1)
        print(f"[coarse_chunks] 总分块数: {len(res)}, 单实体块数: {single_count}")
        return res

    single_count = sum(1 for c in chunks if len(c) == 1)
    print(f"[coarse_chunks] 总分块数: {len(chunks)}, 单实体块数: {single_count}")
    return chunks

def _ngrams(s: str, n: int) -> set:
    s = normalize_text(s).lower()
    length = len(s)
    if length == 0:
        return set()
    grams = set()
    end = max(1, length - n + 1)
    for i in range(0, end):
        grams.add(s[i:i+n])
    return grams

def coarse_chunks_jaccard(entities: List[str], size: int, ngram_size: int = 2, threshold: float = 0.5) -> List[List[str]]:
    buckets: List[List[str]] = []
    reps: List[set] = []
    for e in entities:
        g = _ngrams(e, ngram_size)
        best_idx = -1
        best_score = -1.0
        for i, r in enumerate(reps):
            inter = len(g & r)
            union = len(g | r)
            j = (inter / union) if union else 0.0
            if j > best_score:
                best_score = j
                best_idx = i
        if best_score >= threshold and best_idx >= 0:
            buckets[best_idx].append(e)
            reps[best_idx] = reps[best_idx] | g
        else:
            buckets.append([e])
            reps.append(g)
    chunks: List[List[str]] = []
    for items in buckets:
        sub: List[str] = []
        for e in items:
            sub.append(e)
            if len(sub) >= size:
                chunks.append(sub)
                sub = []
        if sub:
            chunks.append(sub)
    if not chunks:
        res = [entities[i:i+size] for i in range(0, len(entities), size)]
        single_count = sum(1 for c in res if len(c) == 1)
        print(f"[coarse_chunks_jaccard] 总分块数: {len(res)}, 单实体块数: {single_count}")
        return res

    single_count = sum(1 for c in chunks if len(c) == 1)
    print(f"[coarse_chunks_jaccard] 总分块数: {len(chunks)}, 单实体块数: {single_count}")
    return chunks

def _merge_buckets_by_similarity(buckets: List[List[str]], ngram_size: int = 2, threshold: float = 0.6) -> List[List[str]]:
    merged_buckets: List[List[str]] = []
    merged_signatures: List[set] = []
    for items in buckets:
        sig_i: set = set()
        for e in items:
            sig_i |= _ngrams(e, ngram_size)
        best_idx = -1
        best_score = -1.0
        for j, sig_j in enumerate(merged_signatures):
            inter = len(sig_i & sig_j)
            union = len(sig_i | sig_j)
            jacc = (inter / union) if union else 0.0
            if jacc > best_score:
                best_score = jacc
                best_idx = j
        if best_score >= threshold and best_idx >= 0:
            merged_buckets[best_idx].extend(items)
            merged_signatures[best_idx] |= sig_i
        else:
            merged_buckets.append(list(items))
            merged_signatures.append(sig_i)
    return merged_buckets

def _attach_small_buckets(buckets: List[List[str]], ngram_size: int = 2, attach_threshold: float = 0.4, min_size: int = 2) -> List[List[str]]:
    if not buckets:
        return buckets
    big_idxs: List[int] = []
    small_idxs: List[int] = []
    for i, b in enumerate(buckets):
        if len(b) >= min_size:
            big_idxs.append(i)
        else:
            small_idxs.append(i)
    if not small_idxs or not big_idxs:
        return buckets
    big_sigs: List[set] = []
    for i in big_idxs:
        sig = set()
        for e in buckets[i]:
            sig |= _ngrams(e, ngram_size)
        big_sigs.append(sig)
    used_small = set()
    for si in small_idxs:
        sig_s = set()
        for e in buckets[si]:
            sig_s |= _ngrams(e, ngram_size)
        best_j = -1.0
        best_bi = -1
        for idx, sig_b in zip(big_idxs, big_sigs):
            inter = len(sig_s & sig_b)
            union = len(sig_s | sig_b)
            j = (inter / union) if union else 0.0
            if j > best_j:
                best_j = j
                best_bi = idx
        if best_j >= attach_threshold and best_bi >= 0:
            buckets[best_bi].extend(buckets[si])
            used_small.add(si)
    res: List[List[str]] = []
    for i, b in enumerate(buckets):
        if i not in used_small:
            res.append(b)
    return res

def coarse_chunks_jaccard_merged(
    entities: List[str],
    size: int,
    ngram_size: int = 2,
    threshold: float = 0.5,
    merge_threshold: float = 0.6,
    attach_threshold: float = 0.4,
    min_bucket_size: int = 2,
) -> List[List[str]]:
    initial = coarse_chunks_jaccard(entities, size=max(1, size), ngram_size=ngram_size, threshold=threshold)
    merged = _merge_buckets_by_similarity(initial, ngram_size=ngram_size, threshold=merge_threshold)
    merged = _attach_small_buckets(merged, ngram_size=ngram_size, attach_threshold=attach_threshold, min_size=min_bucket_size)
    chunks: List[List[str]] = []
    for items in merged:
        sub: List[str] = []
        for e in items:
            sub.append(e)
            if len(sub) >= size:
                chunks.append(sub)
                sub = []
        if sub:
            chunks.append(sub)
    if not chunks:
        res = [entities[i:i+size] for i in range(0, len(entities), size)]
        single_count = sum(1 for c in res if len(c) == 1)
        print(f"[coarse_chunks_jaccard_merged] 总分块数: {len(res)}, 单实体块数: {single_count}")
        return res

    single_count = sum(1 for c in chunks if len(c) == 1)
    print(f"[coarse_chunks_jaccard_merged] 总分块数: {len(chunks)}, 单实体块数: {single_count}")
    return chunks

def load_fewshot(path: str) -> Dict:


    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
