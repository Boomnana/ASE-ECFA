from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

from .ecfa_legacy import _resolve_app_dir, load_nodes_xlsx, load_edges_xlsx, build_graph_index
from .metrics import build_crossrel_map
from .representations import RepresentationStore
from .utils import normalize_report_id, safe_str


@dataclass(frozen=True)
class SubsetSpec:
    name: str
    eval_keys: list[tuple[str, str]]
    rel_mode: str = "id"


def canonical_text_key(text: object) -> str:
    s = safe_str(text).lower()
    if not s:
        return ""

    out = []
    prev_sep = False
    for ch in s:
        ok = ch.isalnum() or ('\u4e00' <= ch <= '\u9fff')
        if ok:
            out.append(ch)
            prev_sep = False
        else:
            if not prev_sep:
                out.append(' ')
            prev_sep = True
    return ' '.join(''.join(out).split())


def build_text_key_map(df_reports: pd.DataFrame) -> pd.DataFrame:
    df = df_reports[['app', 'id', 'issue_key', 'text']].copy()
    df['app'] = df['app'].map(safe_str)
    df['id'] = df['id'].map(normalize_report_id)
    df['issue_key'] = df['issue_key'].map(safe_str)
    df['text_key'] = df['text'].map(canonical_text_key)
    return df[['app', 'id', 'issue_key', 'text_key']]


def build_report_anchor_inventory(df_reports: pd.DataFrame, input_root: str | Path) -> pd.DataFrame:
    root = Path(input_root).resolve()
    rows: list[dict] = []
    for app, g in df_reports.groupby('app'):
        app = str(app)
        app_dir = _resolve_app_dir(root, app)
        nodes_path = app_dir / 'unify' / 'nodes.xlsx'
        edges_path = app_dir / 'unify' / 'edges.xlsx'
        if not nodes_path.exists() or not edges_path.exists():
            for rid in g['id'].astype(str).tolist():
                rows.append({'app': app, 'id': normalize_report_id(rid), 'has_anchor': 0})
            continue
        nodes_df = load_nodes_xlsx(nodes_path)
        edges_df = load_edges_xlsx(edges_path)
        idx = build_graph_index(nodes_df, edges_df, backfill_node_reports_from_edges=False)
        covered = {normalize_report_id(rid) for rid, nodes in idx.report_to_nodes.items() if nodes}
        for rid in g['id'].astype(str).tolist():
            rid_n = normalize_report_id(rid)
            rows.append({'app': app, 'id': rid_n, 'has_anchor': int(rid_n in covered)})
    return pd.DataFrame(rows).drop_duplicates(subset=['app', 'id']).reset_index(drop=True)


def _max_cross_similarity(rep_store: RepresentationStore, crossrel_map: dict[tuple[str, str], list[str]], eval_keys: list[tuple[str, str]]) -> pd.DataFrame:
    out: list[dict] = []
    by_app: dict[str, list[str]] = {}
    for app, qid in eval_keys:
        by_app.setdefault(str(app), []).append(str(qid))
    for app, qids in by_app.items():
        idx = rep_store.by_app.get(app)
        if idx is None:
            continue
        for qid in qids:
            rel = [rid for rid in crossrel_map.get((app, qid), []) if idx.has(rid)]
            if not idx.has(qid) or not rel:
                out.append({'app': app, 'query_id': qid, 'max_cross_sim': np.nan})
                continue
            q_row = idx.id_to_row[qid]
            rel_rows = idx.rows(rel)
            if len(rel_rows) == 0:
                out.append({'app': app, 'query_id': qid, 'max_cross_sim': np.nan})
                continue
            if sparse.issparse(idx.X):
                sims = linear_kernel(idx.X[q_row], idx.X[rel_rows]).ravel()
            else:
                sims = idx.X[q_row] @ idx.X[rel_rows].T
            out.append({'app': app, 'query_id': qid, 'max_cross_sim': float(np.max(sims)) if len(sims) else np.nan})
    return pd.DataFrame(out)


def build_subset_protocol(
    *,
    df_reports: pd.DataFrame,
    clusters_df: pd.DataFrame,
    rep_store: RepresentationStore,
    report_anchor_df: Optional[pd.DataFrame] = None,
) -> tuple[dict[str, SubsetSpec], pd.DataFrame, dict[tuple[str, str], set[str]], dict[tuple[str, str], str]]:
    crossrel_map, perq = build_crossrel_map(df_reports, clusters_df)
    perq = perq.copy()
    if len(perq) == 0:
        return {'all_cross': SubsetSpec('all_cross', [])}, perq, {}, {}

    all_cross_keys = [(str(a), str(q)) for a, q in perq.loc[perq['is_cross_query'] == 1, ['app', 'query_id']].itertuples(index=False)]

    text_keys = build_text_key_map(df_reports)
    perq = perq.merge(text_keys[['app', 'id', 'text_key']].rename(columns={'id': 'query_id', 'text_key': 'query_text_key'}), on=['app', 'query_id'], how='left')
    id_to_text_key = {(str(r.app), normalize_report_id(r.id)): str(r.text_key) for r in text_keys.itertuples(index=False)}

    dedup_rel_map: dict[tuple[str, str], set[str]] = {}
    for key in all_cross_keys:
        app, qid = key
        q_text_key = id_to_text_key.get((app, qid), '')
        rel_texts = {id_to_text_key.get((app, rid), '') for rid in crossrel_map.get(key, [])}
        rel_texts.discard('')
        if q_text_key:
            rel_texts.discard(q_text_key)
        dedup_rel_map[key] = rel_texts

    if report_anchor_df is not None and len(report_anchor_df):
        a = report_anchor_df.copy()
        a['app'] = a['app'].map(safe_str)
        a['id'] = a['id'].map(normalize_report_id)
        a['has_anchor'] = a['has_anchor'].fillna(0).astype(int)
        anchor_map = {(str(r.app), normalize_report_id(r.id)): int(r.has_anchor) for r in a.itertuples(index=False)}
        perq['query_has_anchor'] = perq.apply(lambda r: int(anchor_map.get((str(r['app']), str(r['query_id'])), 0)), axis=1)
        n_cross_anchor = []
        for r in perq.itertuples(index=False):
            key = (str(r.app), str(r.query_id))
            n_cross_anchor.append(int(sum(anchor_map.get((str(r.app), rid), 0) for rid in crossrel_map.get(key, []))))
        perq['n_cross_rel_anchor_covered'] = n_cross_anchor
        perq['anchor_eligible'] = ((perq['is_cross_query'] == 1) & (perq['query_has_anchor'] == 1) & (perq['n_cross_rel_anchor_covered'] > 0)).astype(int)
    else:
        perq['query_has_anchor'] = np.nan
        perq['n_cross_rel_anchor_covered'] = np.nan
        perq['anchor_eligible'] = 0

    anchor_keys = [(str(a), str(q)) for a, q in perq.loc[perq['anchor_eligible'] == 1, ['app', 'query_id']].itertuples(index=False)]

    sim_df = _max_cross_similarity(rep_store, crossrel_map, anchor_keys if anchor_keys else all_cross_keys)
    perq = perq.merge(sim_df, on=['app', 'query_id'], how='left')
    anchor_mask = perq['anchor_eligible'] == 1 if 'anchor_eligible' in perq.columns else pd.Series(False, index=perq.index)
    valid_sims = perq.loc[anchor_mask & perq['max_cross_sim'].notna(), 'max_cross_sim']
    hard_thr = float(valid_sims.median()) if len(valid_sims) else np.nan
    perq['text_hard'] = ((perq['anchor_eligible'] == 1) & perq['max_cross_sim'].notna() & (perq['max_cross_sim'] <= hard_thr)).astype(int) if len(valid_sims) else 0


    perq['n_dedup_cross_rel'] = [int(len(dedup_rel_map.get((str(r.app), str(r.query_id)), set()))) for r in perq.itertuples(index=False)]
    perq['dedup_subset'] = ((perq['anchor_eligible'] == 1) & (perq['n_dedup_cross_rel'] > 0)).astype(int)

    specs = {
        'all_cross': SubsetSpec('all_cross', all_cross_keys, rel_mode='id'),
        'anchor_eligible': SubsetSpec('anchor_eligible', [(str(a), str(q)) for a, q in perq.loc[perq['anchor_eligible'] == 1, ['app', 'query_id']].itertuples(index=False)], rel_mode='id'),
        'text_hard': SubsetSpec('text_hard', [(str(a), str(q)) for a, q in perq.loc[perq['text_hard'] == 1, ['app', 'query_id']].itertuples(index=False)], rel_mode='id'),
        'dedup': SubsetSpec('dedup', [(str(a), str(q)) for a, q in perq.loc[perq['dedup_subset'] == 1, ['app', 'query_id']].itertuples(index=False)], rel_mode='text_key'),
    }
    return specs, perq, dedup_rel_map, id_to_text_key
