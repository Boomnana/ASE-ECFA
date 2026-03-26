from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .utils import normalize_report_id, safe_str


def build_crossrel_map(df_reports: pd.DataFrame, clusters_df: pd.DataFrame) -> tuple[dict[tuple[str, str], list[str]], pd.DataFrame]:
    df = df_reports.copy()
    df['app'] = df['app'].map(safe_str)
    df['id'] = df['id'].map(normalize_report_id)
    df['issue_key'] = df['issue_key'].map(safe_str)
    cl = clusters_df[['app', 'id', 'cluster_id']].copy()
    cl['app'] = cl['app'].map(safe_str)
    cl['id'] = cl['id'].map(normalize_report_id)
    cl['cluster_id'] = cl['cluster_id'].map(safe_str)
    merged = df.merge(cl, on=['app', 'id'], how='left', validate='one_to_one')
    if merged['cluster_id'].isna().any():
        miss = merged[merged['cluster_id'].isna()][['app', 'id']].head(10).to_dict('records')
        raise ValueError(f'clusters_df missing some (app,id) assignments. Examples: {miss}')
    crossrel: Dict[Tuple[str, str], list[str]] = {}
    rows: list[dict] = []
    for (app, issue_key), g in merged.groupby(['app', 'issue_key']):
        ids = g['id'].astype(str).tolist()
        if len(ids) <= 1:
            continue
        cid_to_ids = {str(cid): gg['id'].astype(str).tolist() for cid, gg in g.groupby('cluster_id')}
        for _, r in g.iterrows():
            qid = str(r['id'])
            cq = str(r['cluster_id'])
            rel = [x for x in ids if x != qid]
            cross = [x for x in rel if x not in cid_to_ids.get(cq, [])]
            crossrel[(str(app), qid)] = cross
            rows.append({'app': str(app), 'query_id': qid, 'issue_key': str(issue_key), 'cluster_id': cq, 'n_rel': int(len(rel)), 'n_cross_rel': int(len(cross)), 'is_cross_query': int(len(cross) > 0)})
    perq = pd.DataFrame(rows).sort_values(['app', 'issue_key', 'query_id']).reset_index(drop=True)
    return crossrel, perq


def _first_relevant_rank(ranked: list[str], rel_set: set[str], k: int) -> Optional[int]:
    for i, rid in enumerate(ranked[: int(k)], start=1):
        if rid in rel_set:
            return i
    return None


def _ndcg_binary(ranked: list[str], rel_set: set[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(ranked[: int(k)], start=1):
        if rid in rel_set:
            dcg += 1.0 / math.log2(i + 1)
    m = min(len(rel_set), int(k))
    if m <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, m + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def transform_ranked(
    ranked_ids: list[str],
    *,
    app: str,
    qid: str,
    item_to_rel: Optional[Callable[[str, str], str]] = None,
    drop_same_as_query: Optional[str] = None,
    limit: int = 200,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    qid_n = normalize_report_id(qid)
    for rid in ranked_ids:
        rid_n = normalize_report_id(rid)
        if not rid_n or rid_n == qid_n:
            continue
        item = item_to_rel(app, rid_n) if item_to_rel is not None else rid_n
        item = safe_str(item)
        if not item:
            continue
        if drop_same_as_query and item == drop_same_as_query:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= int(limit):
            break
    return out


def evaluate_rankings_from_relmap(
    *,
    rankings_by_method: dict[str, dict[tuple[str, str], list[str]]],
    rel_map: dict[tuple[str, str], Iterable[str]],
    boundary: str,
    eval_keys: list[tuple[str, str]],
    success_ks: tuple[int, ...] = (10, 20),
    recall_ks: tuple[int, ...] = (20, 50),
    mrr_k: int = 20,
    ndcg_k: int = 20,
    max_rank_cap: int = 200,
    item_to_rel: Optional[Callable[[str, str], str]] = None,
    query_drop_rel: Optional[dict[tuple[str, str], str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_query_rows: list[dict] = []
    query_drop_rel = query_drop_rel or {}
    for method, rmap in rankings_by_method.items():
        for app, qid in eval_keys:
            rel = {safe_str(x) for x in rel_map.get((app, qid), []) if safe_str(x)}
            if not rel:
                continue
            ranked = transform_ranked(rmap.get((app, qid), []), app=app, qid=qid, item_to_rel=item_to_rel, drop_same_as_query=query_drop_rel.get((app, qid)), limit=max_rank_cap)
            row = {'boundary': boundary, 'method': method, 'app': app, 'query_id': qid, 'n_cross_rel': len(rel), 'ranked_len': len(ranked)}
            for k in success_ks:
                rank = _first_relevant_rank(ranked, rel, int(k))
                row[f'success_at_{int(k)}'] = float(rank is not None)
            rr_rank = _first_relevant_rank(ranked, rel, int(mrr_k))
            row[f'mrr_at_{int(mrr_k)}'] = (1.0 / rr_rank) if rr_rank is not None else 0.0
            row[f'first_rel_rank_at_{int(mrr_k)}'] = int(rr_rank) if rr_rank is not None else ''
            for k in recall_ks:
                top = ranked[: int(k)]
                row[f'recall_at_{int(k)}'] = len([x for x in top if x in rel]) / max(len(rel), 1)
            row[f'ndcg_at_{int(ndcg_k)}'] = _ndcg_binary(ranked, rel, int(ndcg_k))
            per_query_rows.append(row)
    per_query = pd.DataFrame(per_query_rows)
    summary_rows: list[dict] = []
    by_app_rows: list[dict] = []
    if len(per_query):
        metric_cols = [c for c in per_query.columns if c.startswith(('success_at_', 'mrr_at_', 'recall_at_', 'ndcg_at_'))]
        for method, g in per_query.groupby('method'):
            base = {'boundary': boundary, 'method': method, 'n_eval_queries': int(g.shape[0])}
            for c in metric_cols:
                base[c] = float(g[c].mean())
            summary_rows.append(base)
        for (app, method), g in per_query.groupby(['app', 'method']):
            base = {'boundary': boundary, 'app': app, 'method': method, 'n_eval_queries': int(g.shape[0])}
            for c in metric_cols:
                base[c] = float(g[c].mean())
            by_app_rows.append(base)
    summary = pd.DataFrame(summary_rows).sort_values('method').reset_index(drop=True) if summary_rows else pd.DataFrame()
    by_app = pd.DataFrame(by_app_rows).sort_values(['app', 'method']).reset_index(drop=True) if by_app_rows else pd.DataFrame()
    return summary, per_query, by_app


def evaluate_rankings(
    *,
    df_reports: pd.DataFrame,
    clusters_df: pd.DataFrame,
    rankings_by_method: dict[str, dict[tuple[str, str], list[str]]],
    boundary: str,
    success_ks: tuple[int, ...] = (10, 20),
    recall_ks: tuple[int, ...] = (20, 50),
    mrr_k: int = 20,
    ndcg_k: int = 20,
    max_rank_cap: int = 200,
    eval_keys: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    crossrel_map, _ = build_crossrel_map(df_reports, clusters_df)
    eval_keys = eval_keys or [k for k, v in crossrel_map.items() if len(v) > 0]
    eval_keys = [k for k in eval_keys if len(crossrel_map.get(k, [])) > 0]
    return evaluate_rankings_from_relmap(
        rankings_by_method=rankings_by_method,
        rel_map=crossrel_map,
        boundary=boundary,
        eval_keys=eval_keys,
        success_ks=tuple(success_ks),
        recall_ks=tuple(recall_ks),
        mrr_k=int(mrr_k),
        ndcg_k=int(ndcg_k),
        max_rank_cap=int(max_rank_cap),
    )
