from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .utils import dedup_keep_order, normalize_report_id, safe_str
from .representations import RepresentationStore


@dataclass(frozen=True)
class QueryBundle:
    app: str
    query_id: str


def _cluster_lookup(clusters_df: pd.DataFrame) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], list[str]], dict[str, list[str]]]:
    id_to_cluster: dict[tuple[str, str], str] = {}
    cluster_members: dict[tuple[str, str], list[str]] = {}
    app_cluster_ids: dict[str, list[str]] = {}
    for (app, cid), g in clusters_df.groupby(['app', 'cluster_id']):
        app = safe_str(app)
        cid = safe_str(cid)
        ids = [normalize_report_id(x) for x in g['id'].astype(str).tolist()]
        cluster_members[(app, cid)] = ids
        app_cluster_ids.setdefault(app, []).append(cid)
        for rid in ids:
            id_to_cluster[(app, rid)] = cid
    return id_to_cluster, cluster_members, app_cluster_ids


def rank_text_baselines(
    *,
    df_reports: pd.DataFrame,
    clusters_df: pd.DataFrame,
    rep_store: RepresentationStore,
    q_keys: list[tuple[str, str]],
    multi_Ls: list[int],
    max_rank_cap: int,
) -> dict[str, dict[tuple[str, str], list[str]]]:
    id_to_cluster, cluster_members, app_cluster_ids = _cluster_lookup(clusters_df)
    by_app_queries: dict[str, list[str]] = {}
    for app, qid in q_keys:
        by_app_queries.setdefault(str(app), []).append(str(qid))

    rankings_by_method: dict[str, dict[tuple[str, str], list[str]]] = {'cluster_only': {}, 'global': {}}
    for L in multi_Ls:
        rankings_by_method[f'multi_L{int(L)}'] = {}

    for app, qids in by_app_queries.items():
        idx = rep_store.by_app[app]
        scores, qids_valid, q_rows = idx.query_scores(qids)
        if len(qids_valid) == 0:
            continue
        app_ids = np.asarray(idx.ids, dtype=object)
        cluster_ids = app_cluster_ids.get(app, [])
        for row_idx, qid in enumerate(qids_valid):
            q_scores = scores[row_idx].copy()
            if qid in idx.id_to_row:
                q_scores[idx.id_to_row[qid]] = -np.inf
            order_global = np.argsort(-q_scores, kind='mergesort')
            rankings_by_method['global'][(app, qid)] = dedup_keep_order(app_ids[order_global].tolist(), drop=qid, limit=max_rank_cap)

            q_cluster = id_to_cluster.get((app, qid))
            if q_cluster is None:
                continue
            mask = np.full_like(q_scores, -np.inf)
            rows = idx.rows(cluster_members.get((app, q_cluster), []))
            mask[rows] = q_scores[rows]
            order_cluster = np.argsort(-mask, kind='mergesort')
            rankings_by_method['cluster_only'][(app, qid)] = dedup_keep_order(app_ids[order_cluster].tolist(), drop=qid, limit=max_rank_cap)

            if cluster_ids:
                sims = idx.centroid_scores(qid, cluster_ids)
                c_order = np.argsort(-sims, kind='mergesort')
            else:
                c_order = np.asarray([], dtype=np.int32)
            for L in multi_Ls:
                top_clusters = [cluster_ids[i] for i in c_order[: int(L)] if i < len(cluster_ids)]
                member_ids: list[str] = []
                for cid in top_clusters:
                    member_ids.extend(cluster_members.get((app, cid), []))
                rows = idx.rows(member_ids)
                mask = np.full_like(q_scores, -np.inf)
                if len(rows):
                    mask[rows] = q_scores[rows]
                order = np.argsort(-mask, kind='mergesort')
                rankings_by_method[f'multi_L{int(L)}'][(app, qid)] = dedup_keep_order(app_ids[order].tolist(), drop=qid, limit=max_rank_cap)
    return rankings_by_method
