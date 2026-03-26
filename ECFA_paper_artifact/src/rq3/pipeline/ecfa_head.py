from __future__ import annotations

"""Head-optimized ECFA retrieval for RQ3 engineering comparisons.

Design goals:
- Keep ECFA candidate generation structural (views/subgraphs + anchors)
- Improve top-20 / head metrics via wider candidate recall + stronger rerank
- Preserve the same per_query_K20_ecfa.csv schema so downstream evaluation stays unchanged

This module intentionally exposes a *tuned engineering variant*.
Use it when you want a practical, head-oriented ECFA retrieval stack rather than
reproducing the legacy paper-aligned defaults.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from collections import defaultdict
import math
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import normalize_report_id, ensure_dir, write_csv, read_clusters_csv
from .ecfa_legacy import (
    PER_QUERY_K20_ECFA_COLUMNS,
    EcfaAppIndex,
    build_ecfa_index_for_app,
)


@dataclass(frozen=True)
class EcfaHeadRunConfig:
    xlsx: Path
    input_root: Path
    out_dir: Path
    n_clusters: int = 20


    retr_view_pool: str = "merged"
    ecfa_scope: str = "pool"
    ecfa_budget: int = 200
    ecfa_r_max: int = 0
    merge_sim_thresh: float = 0.6
    backfill_node_reports_from_edges: bool = False


    topk_list: Sequence[int] = (20, 50, 100)
    max_rank_cap: int = 200
    retr_top_views: int = 0
    pre_refine_cap: int = 600
    candidate_ranker: str = "hybrid"
    retr_shared_weight: float = 1.0


    refine: bool = True
    refine_tfidf_analyzer: str = "char"
    refine_ngram_min: int = 2
    refine_ngram_max: int = 4
    refine_max_features: int = 20000


    wt_text: float = 0.58
    wt_support: float = 0.18
    wt_shared: float = 0.18
    wt_cost: float = 0.06


@dataclass
class _RefineIndex:
    vectorizer: TfidfVectorizer
    X: "np.ndarray | object"
    id_to_row: Dict[str, int]


def _build_refine_index_for_app(df_app: pd.DataFrame, cfg: EcfaHeadRunConfig) -> _RefineIndex:
    texts = df_app["text"].astype(str).tolist()
    ids = [normalize_report_id(x) for x in df_app["id"].astype(str).tolist()]
    id_to_row = {rid: i for i, rid in enumerate(ids)}

    vec = TfidfVectorizer(
        analyzer=str(cfg.refine_tfidf_analyzer),
        ngram_range=(int(cfg.refine_ngram_min), int(cfg.refine_ngram_max)),
        min_df=2,
        max_df=0.95,
        max_features=int(cfg.refine_max_features),
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        vec = TfidfVectorizer(
            analyzer=str(cfg.refine_tfidf_analyzer),
            ngram_range=(int(cfg.refine_ngram_min), int(cfg.refine_ngram_max)),
            min_df=1,
            max_df=0.95,
            max_features=int(cfg.refine_max_features),
        )
        X = vec.fit_transform(texts)
    return _RefineIndex(vectorizer=vec, X=X, id_to_row=id_to_row)


def _build_rel_map(df_reports: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
    rel: Dict[Tuple[str, str], List[str]] = {}
    for (app, issue), g in df_reports.groupby(["app", "issue_key"]):
        ids = [normalize_report_id(x) for x in g["id"].tolist()]
        if len(ids) < 2:
            continue
        for q in ids:
            rel[(str(app), str(q))] = [x for x in ids if x != q]
    return rel


def _build_cluster_map(clusters_df: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    mp: Dict[Tuple[str, str], str] = {}
    for r in clusters_df.itertuples(index=False):
        mp[(str(r.app), normalize_report_id(r.id))] = str(r.cluster_id)
    return mp


def _auto_find_clusters_csv(out_dir: Path, *, K: int) -> Optional[Path]:
    out_dir = Path(out_dir)
    run_dir = out_dir
    for _ in range(6):
        if (run_dir / "clusters").exists():
            break
        if run_dir.parent == run_dir:
            break
        run_dir = run_dir.parent
    clusters_dir = run_dir / "clusters"
    if not clusters_dir.exists():
        return None
    cand: List[Path] = list(clusters_dir.glob(f"clusters_*_K{int(K)}.csv"))
    if not cand:
        cand = list(clusters_dir.glob("clusters_*.csv"))
    if not cand:
        return None

    def _prio(pp: Path) -> int:
        s = pp.name.lower()
        if "sbert" in s:
            return 3
        if "tfidf" in s:
            return 2
        if "llm" in s:
            return 1
        return 0

    return sorted(cand, key=lambda pp: (_prio(pp), pp.name), reverse=True)[0]


def _candidate_rank_key(item: Tuple[str, dict], ranker: str) -> tuple:
    rid, feat = item
    support = float(feat.get("support", 0.0))
    shared = float(feat.get("shared", 0.0))
    score = float(feat.get("base_score", 0.0))
    cost = float(feat.get("min_cost", 1e9))
    if ranker == "support":
        return (-score, cost, rid)
    if ranker == "cost":
        return (cost, -score, rid)
    return (-(score / (cost + 1e-9)), -score, cost, rid)


def ecfa_retrieve_ranked_head(
    qid: str,
    app_idx: EcfaAppIndex,
    *,
    candidate_ranker: str = "hybrid",
    retr_top_views: int = 0,
    retr_shared_weight: float = 1.0,
    pre_refine_cap: int = 600,
) -> Tuple[List[str], Dict[str, int], Dict[str, dict]]:
    qid = normalize_report_id(qid)
    anchors = set(app_idx.report_to_nodes.get(qid, set()))
    if not anchors:
        return [], {"has_anchor": 0, "matched_views": 0}, {}

    view_hits: Dict[int, int] = defaultdict(int)
    for nid in anchors:
        for vid in app_idx.node_to_view_ids.get(str(nid), []):
            view_hits[int(vid)] += 1
    if not view_hits:
        return [], {"has_anchor": 1, "matched_views": 0}, {}

    sorted_views = sorted(view_hits.items(), key=lambda x: (-x[1], app_idx.views[x[0]].cost, x[0]))
    if int(retr_top_views) > 0:
        use_views = [vid for vid, _ in sorted_views[: int(retr_top_views)]]
    else:
        use_views = [vid for vid, _ in sorted_views]

    features: Dict[str, dict] = {}
    for vid in use_views:
        sg = app_idx.views[int(vid)]
        cost = float(sg.cost)
        for rid0 in sg.report_ids:
            rid = normalize_report_id(rid0)
            if not rid or rid == qid:
                continue
            feat = features.setdefault(
                rid,
                {"support": 0, "shared": 0, "min_cost": cost, "view_count": 0, "base_score": 0.0},
            )
            feat["support"] += 1
            feat["view_count"] += 1
            if cost < float(feat["min_cost"]):
                feat["min_cost"] = cost

    if not features:
        return [], {"has_anchor": 1, "matched_views": int(len(view_hits))}, {}

    for rid, feat in features.items():
        r_nodes = app_idx.report_to_nodes.get(rid, set())
        shared = len(anchors.intersection(r_nodes)) if r_nodes else 0
        feat["shared"] = int(shared)
        feat["base_score"] = float(feat["support"]) + float(retr_shared_weight) * float(shared)

    ranked_items = sorted(features.items(), key=lambda x: _candidate_rank_key(x, candidate_ranker))
    if int(pre_refine_cap) > 0:
        ranked_items = ranked_items[: int(pre_refine_cap)]
    ranked_ids = [rid for rid, _ in ranked_items]
    return ranked_ids, {"has_anchor": 1, "matched_views": int(len(view_hits))}, features


def _norm01(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    x = (float(v) - float(lo)) / (float(hi) - float(lo))
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _refine_rank_head(
    qid: str,
    ranked: List[str],
    feature_map: Dict[str, dict],
    refine_idx: _RefineIndex,
    *,
    wt_text: float,
    wt_support: float,
    wt_shared: float,
    wt_cost: float,
) -> List[str]:
    if not ranked:
        return ranked
    qid = normalize_report_id(qid)
    if qid not in refine_idx.id_to_row:
        return ranked

    cand = [rid for rid in ranked if rid in refine_idx.id_to_row]
    if not cand:
        return ranked

    q_row = refine_idx.id_to_row[qid]
    q_vec = refine_idx.X[q_row]
    c_rows = [refine_idx.id_to_row[rid] for rid in cand]
    C = refine_idx.X[c_rows]
    sims = cosine_similarity(q_vec, C).ravel().tolist()

    supports = [float(feature_map.get(rid, {}).get("support", 0.0)) for rid in cand]
    shareds = [float(feature_map.get(rid, {}).get("shared", 0.0)) for rid in cand]
    costs = [float(feature_map.get(rid, {}).get("min_cost", 1e9)) for rid in cand]

    max_support = max(supports) if supports else 1.0
    max_shared = max(shareds) if shareds else 1.0
    min_cost = min(costs) if costs else 0.0
    max_cost = max(costs) if costs else 1.0

    scored: List[Tuple[str, float, float, float, float, float]] = []
    for rid, sim in zip(cand, sims):
        feat = feature_map.get(rid, {})
        support = float(feat.get("support", 0.0)) / max(max_support, 1e-9)
        shared = float(feat.get("shared", 0.0)) / max(max_shared, 1e-9) if max_shared > 0 else 0.0

        inv_cost = 1.0 - _norm01(float(feat.get("min_cost", max_cost)), min_cost, max_cost)
        final = float(wt_text) * float(sim) + float(wt_support) * support + float(wt_shared) * shared + float(wt_cost) * inv_cost
        scored.append((rid, float(final), float(sim), shared, support, float(feat.get("min_cost", 1e9))))

    scored_sorted = sorted(scored, key=lambda x: (-x[1], -x[2], -x[3], -x[4], x[5], x[0]))
    refined = [rid for rid, *_ in scored_sorted]
    seen = set(refined)
    tail = [rid for rid in ranked if rid not in seen]
    return refined + tail


def run_ecfa_head_and_export_per_query(
    cfg: EcfaHeadRunConfig,
    *,
    df_reports: pd.DataFrame,
    clusters_csv: Optional[str | Path] = None,
) -> Path:
    if df_reports is None or len(df_reports) == 0:
        raise ValueError("df_reports is empty")

    ensure_dir(cfg.out_dir)
    df = df_reports.copy()
    df["app"] = df["app"].astype(str)
    df["id"] = df["id"].map(normalize_report_id)
    df["issue_key"] = df["issue_key"].astype(str)
    if "text" not in df.columns:
        raise ValueError("df_reports must contain 'text' column for refine.")
    df["text"] = df["text"].astype(str)

    if clusters_csv is not None:
        clusters_path = Path(clusters_csv)
    else:
        clusters_path = _auto_find_clusters_csv(Path(cfg.out_dir), K=int(cfg.n_clusters))
    if clusters_path is None or not clusters_path.exists():
        raise FileNotFoundError("clusters_csv is required (or must be discoverable under <run_dir>/clusters/)")

    clusters_df = read_clusters_csv(clusters_path)
    cluster_map = _build_cluster_map(clusters_df)
    rel_map = _build_rel_map(df)

    app_to_corpus: Dict[str, Set[str]] = defaultdict(set)
    for r in df.itertuples(index=False):
        app_to_corpus[str(r.app)].add(str(r.id))

    app_indices: Dict[str, EcfaAppIndex] = {}
    root = Path(cfg.input_root).resolve()
    for app in sorted(df["app"].unique().tolist()):
        allowed = app_to_corpus[app] if str(cfg.ecfa_scope).lower() == "pool" else None
        app_indices[app] = build_ecfa_index_for_app(
            app,
            input_root=root,
            unify_subdir="unify",
            ecfa_k_hop=2,
            ecfa_r_max=int(cfg.ecfa_r_max),
            merge_sim_thresh=float(cfg.merge_sim_thresh),
            ecfa_budget_k=int(cfg.ecfa_budget),
            alpha=1.0,
            beta=0.02,
            gamma=0.01,
            backfill_node_reports_from_edges=bool(cfg.backfill_node_reports_from_edges),
            retr_view_pool=str(cfg.retr_view_pool),
            allowed_reports=allowed,
        )

    refine_idx_by_app: Dict[str, _RefineIndex] = {}
    if bool(cfg.refine):
        for app, g in df.groupby("app"):
            refine_idx_by_app[str(app)] = _build_refine_index_for_app(g.reset_index(drop=True), cfg)

    topk_list = sorted({int(x) for x in (cfg.topk_list or []) if int(x) > 0}) or [20, 50, 100]
    maxk_dump = int(max(topk_list))
    rows: List[dict] = []

    for (app, qid), rel in rel_map.items():
        app = str(app)
        qid = normalize_report_id(qid)
        if app not in app_indices:
            continue
        if (app, qid) not in cluster_map:
            raise ValueError(f"Missing cluster assignment for query ({app},{qid}) from clusters csv: {clusters_path}")

        rel = [normalize_report_id(x) for x in rel if (app, normalize_report_id(x)) in cluster_map]
        if not rel:
            continue
        cq = cluster_map[(app, qid)]
        cross_rel = [r for r in rel if cluster_map[(app, r)] != cq]
        is_cross_query = int(len(cross_rel) > 0)

        ranked, diag, feature_map = ecfa_retrieve_ranked_head(
            qid,
            app_indices[app],
            candidate_ranker=str(cfg.candidate_ranker),
            retr_top_views=int(cfg.retr_top_views),
            retr_shared_weight=float(cfg.retr_shared_weight),
            pre_refine_cap=int(cfg.pre_refine_cap),
        )

        if bool(cfg.refine) and ranked and app in refine_idx_by_app:
            ranked = _refine_rank_head(
                qid,
                ranked,
                feature_map,
                refine_idx_by_app[app],
                wt_text=float(cfg.wt_text),
                wt_support=float(cfg.wt_support),
                wt_shared=float(cfg.wt_shared),
                wt_cost=float(cfg.wt_cost),
            )

        ranked = [normalize_report_id(x) for x in ranked if normalize_report_id(x)]
        ranked = [x for x in ranked if x != qid]
        seen: Set[str] = set()
        deduped: List[str] = []
        for x in ranked:
            if x in seen:
                continue
            seen.add(x)
            deduped.append(x)
            if len(deduped) >= int(cfg.max_rank_cap):
                break
        ranked = deduped

        has_anchor = int(diag.get("has_anchor", 0))
        matched_views = int(diag.get("matched_views", 0))
        retrieved_total = int(len(ranked))
        corpus_ids = app_to_corpus.get(app, set())
        ranked_in_pool = [x for x in ranked if x in corpus_ids]
        retrieved_in_pool = int(len(ranked_in_pool))

        first_rank = None
        if is_cross_query and ranked:
            pos = {rid: i for i, rid in enumerate(ranked, start=1)}
            first_rank = min([pos[r] for r in cross_rel if r in pos], default=None)

        issue_key = df.loc[(df["app"] == app) & (df["id"] == qid), "issue_key"].iloc[0]
        rows.append(
            {
                "app": app,
                "query_id": qid,
                "cluster_id": cq,
                "issue_key": str(issue_key),
                "n_rel": int(len(rel)),
                "n_cross_rel": int(len(cross_rel)),
                "is_cross_query": int(is_cross_query),
                "has_anchor": int(has_anchor),
                "matched_views": int(matched_views),
                "retrieved_total": int(retrieved_total),
                "retrieved_in_pool": int(retrieved_in_pool),
                "first_cross_rank": (int(first_rank) if first_rank is not None else ""),
                "maxk_dump": int(maxk_dump),
                "top_retrieved_50": "|".join(ranked[:50]),
                "top_retrieved_inpool_50": "|".join(ranked_in_pool[:50]),
                "top_retrieved_maxk": "|".join(ranked[: int(maxk_dump)]),
                "top_retrieved_inpool_maxk": "|".join(ranked_in_pool[: int(maxk_dump)]),
            }
        )

    per_query = pd.DataFrame(rows)
    missing = [c for c in PER_QUERY_K20_ECFA_COLUMNS if c not in per_query.columns]
    if missing:
        raise ValueError(f"per_query missing columns: {missing}")
    per_query = per_query[PER_QUERY_K20_ECFA_COLUMNS].copy()

    out_csv = Path(cfg.out_dir) / "per_query_K20_ecfa.csv"
    write_csv(per_query, out_csv)
    return out_csv


def load_ecfa_rankings(
    per_query_csv: str | Path,
    *,
    max_rank_cap: int = 200,
    prefer_inpool: bool = True,
) -> dict[tuple[str, str], list[str]]:
    p = Path(per_query_csv)
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)

    cols = [c for c in df.columns if c.startswith("top_retrieved")]
    if not cols:
        raise ValueError(f"No top list columns found in {p}")

    maxk_fallback = int(df["maxk_dump"].max()) if "maxk_dump" in df.columns and len(df) else int(max_rank_cap)

    def _col_key(c: str) -> tuple[int, int]:
        inpool = 1 if "inpool" in c else 0
        if c.endswith("_maxk"):
            k = maxk_fallback
            return (inpool, k)
        m = re.search(r"(\d+)$", c)
        k = int(m.group(1)) if m else 0
        return (inpool, k)

    if prefer_inpool:
        best = sorted(cols, key=_col_key, reverse=True)[0]
        if "inpool" not in best:
            best = sorted(cols, key=_col_key, reverse=True)[0]
    else:
        best = sorted(cols, key=_col_key, reverse=True)[0]

    out: dict[tuple[str, str], list[str]] = {}
    for _, r in df.iterrows():
        app = str(r.get("app", "")).strip()
        qid = normalize_report_id(r.get("query_id"))
        raw_list = str(r.get(best, "") or "")
        parts = [pp for pp in raw_list.split("|") if pp and str(pp).strip()]
        ranked: List[str] = []
        seen: Set[str] = set()
        for x in parts:
            rid = normalize_report_id(x)
            if not rid or rid == qid or rid in seen:
                continue
            seen.add(rid)
            ranked.append(rid)
            if len(ranked) >= int(max_rank_cap):
                break
        out[(app, qid)] = ranked
    return out
