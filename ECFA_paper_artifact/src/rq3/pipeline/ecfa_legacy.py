from __future__ import annotations

"""
ECFA-Retr (v3a) implementation for RQ3.

This module is a drop-in replacement for `core/retrieval/ecfa_retr.py`.

Key guarantees:
- Produces `per_query_K20_ecfa.csv` with STRICT columns & order (PER_QUERY_K20_ECFA_COLUMNS).
- Computes *real* values for: n_cross_rel/is_cross_query/has_anchor/matched_views/first_cross_rank/maxk_dump/...
- NO fallback-to-global retrieval when ECFA has no anchor or no matched views (returns empty ranking).
- Default retrieval = ECFA + refine (candidate set from ECFA views, then TFIDF similarity rerank).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Literal
from collections import defaultdict
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None

from .utils import normalize_report_id, split_report_ids, ensure_dir, write_csv, read_clusters_csv


PER_QUERY_K20_ECFA_COLUMNS: List[str] = [
    "app",
    "query_id",
    "cluster_id",
    "issue_key",
    "n_rel",
    "n_cross_rel",
    "is_cross_query",
    "has_anchor",
    "matched_views",
    "retrieved_total",
    "retrieved_in_pool",
    "first_cross_rank",
    "maxk_dump",
    "top_retrieved_50",
    "top_retrieved_inpool_50",
    "top_retrieved_maxk",
    "top_retrieved_inpool_maxk",
]


@dataclass(frozen=True)
class EcfaRunConfig:

    xlsx: Path
    input_root: Path
    out_dir: Path
    n_clusters: int = 20


    retr_view_pool: str = "selected"
    ecfa_scope: str = "pool"
    ecfa_budget: int = 50
    ecfa_r_max: int = 0


    topk_list: Sequence[int] = (20, 50, 100)
    max_rank_cap: int = 200
    retr_ranker: str = "support"
    retr_top_views: int = 5
    retr_shared_weight: float = 1.0


    refine: bool = True
    refine_tfidf_analyzer: str = "char"
    refine_ngram_min: int = 2
    refine_ngram_max: int = 4
    refine_max_candidates: int = 500


REQUIRED_NODE_COLS = {"id", "name", "type", "source_row_index"}
REQUIRED_EDGE_COLS = {"id", "source_id", "target_id", "relation", "source_type", "target_type", "source_row_index"}

DEFAULT_SEED_TYPES = {"功能模块", "影响元素", "用户感知现象", "用户操作", "系统诊断信息", "问题陈述"}


def _read_excel_str(path: Path, sheet_name=0) -> pd.DataFrame:

    return pd.read_excel(path, sheet_name=sheet_name, dtype=str)


def load_nodes_xlsx(path: str | Path, sheet_name=0) -> pd.DataFrame:
    p = Path(path)
    df = _read_excel_str(p, sheet_name=sheet_name)
    miss = REQUIRED_NODE_COLS - set(df.columns)
    if miss:
        raise ValueError(f"nodes.xlsx missing columns: {sorted(miss)}; got={list(df.columns)}")
    df = df[list(REQUIRED_NODE_COLS)].copy()
    for c in ["id", "name", "type"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["source_row_index"] = df["source_row_index"].fillna("").astype(str)
    df["report_ids"] = df["source_row_index"].map(split_report_ids)
    return df


def load_edges_xlsx(path: str | Path, sheet_name=0) -> pd.DataFrame:
    p = Path(path)
    df = _read_excel_str(p, sheet_name=sheet_name)
    miss = REQUIRED_EDGE_COLS - set(df.columns)
    if miss:
        raise ValueError(f"edges.xlsx missing columns: {sorted(miss)}; got={list(df.columns)}")
    df = df[list(REQUIRED_EDGE_COLS)].copy()
    for c in ["id", "source_id", "target_id", "relation", "source_type", "target_type"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["source_row_index"] = df["source_row_index"].fillna("").astype(str)
    df["report_ids"] = df["source_row_index"].map(split_report_ids)
    return df


@dataclass
class GraphIndex:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_to_reports: Dict[str, Set[str]]
    report_to_nodes: Dict[str, Set[str]]
    report_to_edges: Dict[str, Set[str]]
    edge_id_to_row: Dict[str, dict]
    node_id_to_row: Dict[str, dict]


def build_graph_index(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    backfill_node_reports_from_edges: bool = False,
) -> GraphIndex:
    node_to_reports: Dict[str, Set[str]] = {}
    report_to_nodes: Dict[str, Set[str]] = {}


    for row in nodes.itertuples(index=False):
        nid = str(getattr(row, "id"))
        rids = list(getattr(row, "report_ids"))
        node_to_reports[nid] = set(map(str, rids))
        for rid in rids:
            rid = normalize_report_id(rid)
            if not rid:
                continue
            report_to_nodes.setdefault(rid, set()).add(nid)

    report_to_edges: Dict[str, Set[str]] = {}
    edge_id_to_row: Dict[str, dict] = {}
    for row in edges.to_dict(orient="records"):
        eid = str(row["id"])
        edge_id_to_row[eid] = row
        src_id = str(row.get("source_id", "") or "")
        tgt_id = str(row.get("target_id", "") or "")
        rids = row.get("report_ids", []) or []
        for rid0 in rids:
            rid = normalize_report_id(rid0)
            if not rid:
                continue
            report_to_edges.setdefault(rid, set()).add(eid)
            if backfill_node_reports_from_edges:
                if src_id:
                    report_to_nodes.setdefault(rid, set()).add(src_id)
                    node_to_reports.setdefault(src_id, set()).add(rid)
                if tgt_id:
                    report_to_nodes.setdefault(rid, set()).add(tgt_id)
                    node_to_reports.setdefault(tgt_id, set()).add(rid)

    node_id_to_row = {str(r["id"]): r for r in nodes.to_dict(orient="records")}
    return GraphIndex(
        nodes=nodes,
        edges=edges,
        node_to_reports=node_to_reports,
        report_to_nodes=report_to_nodes,
        report_to_edges=report_to_edges,
        edge_id_to_row=edge_id_to_row,
        node_id_to_row=node_id_to_row,
    )


@dataclass
class ExpansionResult:
    seed_node: str
    hop_k: int
    report_ids: Set[str]
    node_ids: Set[str]
    stopped_reason: str


def expand_reports_from_seed(
    idx: GraphIndex,
    seed_node: str,
    *,
    k: int = 2,
    r_max: int = 200,
    v_max: int = 2000,
    strict_r_max: bool = True,
) -> ExpansionResult:
    seed_node = str(seed_node)
    R = set(map(str, idx.node_to_reports.get(seed_node, set())))
    V = {seed_node}
    if not R:
        return ExpansionResult(seed_node, 0, set(), {seed_node}, "seed-has-no-reports")

    report_depth: Dict[str, int] = {str(rid): 0 for rid in R}

    def _nodes_from_reports(report_ids: Set[str]) -> Set[str]:
        out = {seed_node}
        for rid in report_ids:
            out |= set(idx.report_to_nodes.get(rid, set()))
        return out

    def _truncate_reports(report_ids: Set[str]) -> Set[str]:
        if r_max <= 0 or len(report_ids) <= r_max:
            return set(report_ids)
        ranked = sorted(
            map(str, report_ids),
            key=lambda rid: (int(report_depth.get(str(rid), 10**9)), int(rid) if rid.isdigit() else 10**9, rid),
        )
        return set(ranked[: int(r_max)])

    if r_max > 0 and strict_r_max and len(R) > r_max:
        R = _truncate_reports(R)
        V = _nodes_from_reports(R)
        stopped = "r-max-truncated-initial"
        if v_max > 0 and len(V) >= v_max:
            stopped += "+v-max"
        return ExpansionResult(seed_node, 0, R, V, stopped)

    stopped = "k-reached"
    hop = 0
    for hop in range(1, k + 1):
        V_new: Set[str] = set()
        for rid in R:
            V_new |= idx.report_to_nodes.get(rid, set())
        V2 = V | V_new

        R_new: Set[str] = set()
        for nid in V2:
            R_new |= idx.node_to_reports.get(nid, set())
        R2 = set(R) | set(map(str, R_new))

        for rid in (R2 - R):
            report_depth.setdefault(str(rid), int(hop))

        if r_max > 0 and strict_r_max and len(R2) > r_max:
            R2 = _truncate_reports(R2)
            V2 = _nodes_from_reports(R2)
            stopped = "r-max-truncated"
            if v_max > 0 and len(V2) >= v_max:
                stopped = "r-max-truncated+v-max"
            V, R = V2, R2
            break

        if len(V2) == len(V) and len(R2) == len(R):
            stopped = "no-growth"
            V, R = V2, R2
            break

        V, R = V2, R2

        if len(V) >= v_max:
            stopped = "v-max"
            break

    return ExpansionResult(seed_node, hop, set(map(str, R)), set(map(str, V)), stopped)


@dataclass
class Subgraph:
    subgraph_id: str
    seed_node: str
    report_ids: Set[str]
    node_ids: Set[str]
    edge_ids: Set[str]
    cost: float


def induce_subgraph_from_reports(idx: GraphIndex, report_ids: Set[str]) -> Tuple[Set[str], Set[str]]:
    node_ids: Set[str] = set()
    for rid in report_ids:
        node_ids |= idx.report_to_nodes.get(rid, set())
    edge_ids: Set[str] = set()
    for rid in report_ids:
        edge_ids |= idx.report_to_edges.get(rid, set())
    return set(map(str, node_ids)), set(map(str, edge_ids))


def compute_cost(
    report_ids: Set[str], node_ids: Set[str], edge_ids: Set[str], *, alpha: float, beta: float, gamma: float
) -> float:
    return float(alpha) * len(report_ids) + float(beta) * len(node_ids) + float(gamma) * len(edge_ids)


def make_subgraph(
    idx: GraphIndex,
    subgraph_id: str,
    seed_node: str,
    report_ids: Set[str],
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> Subgraph:
    node_ids, edge_ids = induce_subgraph_from_reports(idx, report_ids)
    cost = compute_cost(report_ids, node_ids, edge_ids, alpha=alpha, beta=beta, gamma=gamma)
    return Subgraph(
        subgraph_id=str(subgraph_id),
        seed_node=str(seed_node),
        report_ids=set(map(str, report_ids)),
        node_ids=set(map(str, node_ids)),
        edge_ids=set(map(str, edge_ids)),
        cost=float(cost),
    )


def jaccard(a: set, b: set, *, empty_both: float = 1.0) -> float:
    if not a and not b:
        return float(empty_both)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _signature_sets(
    node_ids: Iterable[str], node_type_by_id: Dict[str, str], type_alias: Dict[str, str]
) -> Dict[str, Set[str]]:
    sig: Dict[str, Set[str]] = {"ISSUE": set(), "PHEN": set(), "DIAG": set(), "OP": set(), "ELEM": set(), "MOD": set()}
    for nid in node_ids:
        t_raw = node_type_by_id.get(str(nid))
        if not t_raw:
            continue
        t = type_alias.get(t_raw, t_raw)
        if t in sig:
            sig[t].add(str(nid))
    return sig


def weighted_jaccard(
    parts_a: Dict[str, Set[str]], parts_b: Dict[str, Set[str]], weights: Dict[str, float], *, empty_both: float = 1.0
) -> float:
    score = 0.0
    wsum = 0.0
    for k, w in weights.items():
        if w <= 0:
            continue
        score += float(w) * jaccard(parts_a.get(k, set()), parts_b.get(k, set()), empty_both=empty_both)
        wsum += float(w)
    return (score / wsum) if wsum > 0 else 0.0


def merge_similar_subgraphs_signature(
    subgraphs: List[Subgraph],
    *,
    node_type_by_id: Dict[str, str],
    thresh: float = 0.6,
    weights: Optional[Dict[str, float]] = None,
    type_alias: Optional[Dict[str, str]] = None,
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.01,
) -> List[Subgraph]:
    if weights is None:
        weights = {"ISSUE": 0.45, "PHEN": 0.30, "DIAG": 0.20, "OP": 0.05}
    if type_alias is None:
        type_alias = {
            "问题陈述": "ISSUE",
            "用户感知现象": "PHEN",
            "系统诊断信息": "DIAG",
            "用户操作": "OP",
            "影响元素": "ELEM",
            "功能模块": "MOD",
        }

    remaining = subgraphs[:]
    sigs = [_signature_sets(sg.node_ids, node_type_by_id=node_type_by_id, type_alias=type_alias) for sg in remaining]

    paired = sorted(
        zip(remaining, sigs),
        key=lambda p: (
            len(p[1].get("ISSUE", set())),
            len(p[1].get("PHEN", set())) + len(p[1].get("DIAG", set())),
            len(p[0].report_ids),
            p[0].subgraph_id,
        ),
        reverse=True,
    )
    remaining = [p[0] for p in paired]
    sigs = [p[1] for p in paired]

    used = [False] * len(remaining)
    merged: List[Subgraph] = []

    for i, sg in enumerate(remaining):
        if used[i]:
            continue
        rep_union = set(sg.report_ids)
        node_union = set(sg.node_ids)
        edge_union = set(sg.edge_ids)
        seeds = [sg.seed_node]
        sig_union = {k: set(v) for k, v in sigs[i].items()}

        changed = True
        while changed:
            changed = False
            for j in range(i + 1, len(remaining)):
                if used[j]:
                    continue
                sg2 = remaining[j]
                sig2 = sigs[j]
                sim = weighted_jaccard(sig_union, sig2, weights, empty_both=0.0)
                if sim < float(thresh):
                    continue

                used[j] = True
                rep_union |= set(sg2.report_ids)
                node_union |= set(sg2.node_ids)
                edge_union |= set(sg2.edge_ids)
                seeds.append(sg2.seed_node)
                for k in sig_union.keys():
                    sig_union[k] |= sig2.get(k, set())
                changed = True

        used[i] = True
        new_id = sg.subgraph_id if len(seeds) == 1 else f"{sg.subgraph_id}_m{len(seeds)}"
        cost = compute_cost(rep_union, node_union, edge_union, alpha=alpha, beta=beta, gamma=gamma)
        merged.append(
            Subgraph(
                subgraph_id=new_id,
                seed_node=seeds[0],
                report_ids=rep_union,
                node_ids=node_union,
                edge_ids=edge_union,
                cost=cost,
            )
        )
    return merged


def greedy_select_views(
    subgraphs: List[Subgraph],
    *,
    budget_k: int,
    cost_budget: Optional[float] = None,
    warmup_frac: float = 0.25,
    cost_power_start: float = 1.0,
    gain_power: float = 1.0,
    fixed_cost: float = 0.0,
) -> List[Subgraph]:
    K = int(budget_k)
    B = float(cost_budget) if (cost_budget is not None and float(cost_budget) > 0) else None

    selected: List[Subgraph] = []
    covered_reports: Set[str] = set()
    remaining = list(subgraphs)
    cum_cost = 0.0

    def _gain(sg: Subgraph) -> int:
        return int(len(set(sg.report_ids) - covered_reports))

    while remaining and len(selected) < K:
        t = len(selected)
        warmup_steps = max(1, int(K * float(warmup_frac)))
        progress = min(1.0, t / float(warmup_steps))
        q = float(cost_power_start) + (1.0 - float(cost_power_start)) * progress

        best = None
        best_score = -1.0
        for sg in remaining:
            g = _gain(sg)
            if g <= 0:
                continue
            c = float(sg.cost) + float(fixed_cost)
            c = max(c, 1e-9)
            if B is not None and (cum_cost + float(sg.cost)) > B + 1e-12:
                continue
            score = (float(g) ** float(gain_power)) / (c**q)
            if score > best_score:
                best_score = score
                best = sg

        if best is None:
            break

        selected.append(best)
        covered_reports |= set(best.report_ids)
        cum_cost += float(best.cost)
        remaining = [sg for sg in remaining if sg.subgraph_id != best.subgraph_id]

    return selected


@dataclass
class EcfaAppIndex:
    app: str
    report_to_nodes: Dict[str, Set[str]]
    views: List[Subgraph]
    node_to_view_ids: Dict[str, List[int]]
    n_reports_in_graph: int
    n_reports_with_anchors: int
    union_view_reports: Set[str]


def _encode_unicode_path_name(name: str) -> str:
    out = []
    for ch in str(name):
        if ord(ch) < 128:
            out.append(ch)
        else:
            out.append(f"#U{ord(ch):04x}")
    return "".join(out)


def _resolve_app_dir(input_root: Path, app: str) -> Path:
    app = str(app)
    direct = input_root / app
    if direct.exists():
        return direct
    enc = input_root / _encode_unicode_path_name(app)
    if enc.exists():
        return enc

    for p in input_root.iterdir():
        if p.name == app or p.name == _encode_unicode_path_name(app):
            return p
    return direct


def build_ecfa_index_for_app(
    app: str,
    *,
    input_root: Path,
    unify_subdir: str = "unify",
    ecfa_k_hop: int = 2,
    ecfa_r_max: int = 0,
    merge_sim_thresh: float = 0.6,
    ecfa_budget_k: int = 50,
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.01,
    backfill_node_reports_from_edges: bool = False,
    retr_view_pool: str = "selected",
    allowed_reports: Optional[Set[str]] = None,
) -> EcfaAppIndex:
    app = str(app)
    app_dir = _resolve_app_dir(input_root, app)
    nodes_path = app_dir / unify_subdir / "nodes.xlsx"
    edges_path = app_dir / unify_subdir / "edges.xlsx"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Missing unify files for app={app}: {nodes_path} / {edges_path}")

    nodes_df = load_nodes_xlsx(nodes_path)
    edges_df = load_edges_xlsx(edges_path)
    idx = build_graph_index(nodes_df, edges_df, backfill_node_reports_from_edges=bool(backfill_node_reports_from_edges))

    if allowed_reports is not None:
        allowed_reports = set(map(normalize_report_id, allowed_reports))

        for nid in list(idx.node_to_reports.keys()):
            idx.node_to_reports[nid] = {r for r in idx.node_to_reports[nid] if normalize_report_id(r) in allowed_reports}

        idx.report_to_nodes = {normalize_report_id(rid): ns for rid, ns in idx.report_to_nodes.items() if normalize_report_id(rid) in allowed_reports}
        idx.report_to_edges = {normalize_report_id(rid): es for rid, es in idx.report_to_edges.items() if normalize_report_id(rid) in allowed_reports}

    node_type_by_id = {str(r["id"]): str(r["type"]) for r in nodes_df.to_dict(orient="records")}

    seeds = nodes_df.loc[nodes_df["type"].isin(DEFAULT_SEED_TYPES), "id"].astype(str).tolist()
    if not seeds:
        seeds = nodes_df["id"].astype(str).tolist()

    raw_subgraphs: List[Subgraph] = []
    for i, seed in enumerate(seeds):
        ex = expand_reports_from_seed(
            idx,
            seed_node=str(seed),
            k=int(ecfa_k_hop),
            r_max=int(ecfa_r_max),
            strict_r_max=True,
        )
        if not ex.report_ids:
            continue
        sg = make_subgraph(
            idx,
            subgraph_id=f"sg_{i:05d}",
            seed_node=str(seed),
            report_ids=set(ex.report_ids),
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
        )
        raw_subgraphs.append(sg)

    merged = merge_similar_subgraphs_signature(
        raw_subgraphs,
        node_type_by_id=node_type_by_id,
        thresh=float(merge_sim_thresh),
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
    )

    pool = merged if retr_view_pool in {"selected", "merged"} else raw_subgraphs
    if retr_view_pool == "selected":
        views = greedy_select_views(pool, budget_k=int(ecfa_budget_k))
    else:
        views = list(pool)

    node_to_view_ids: Dict[str, List[int]] = defaultdict(list)
    union_view_reports: Set[str] = set()
    for vid, sg in enumerate(views):
        union_view_reports |= set(sg.report_ids)
        for nid in sg.node_ids:
            node_to_view_ids[str(nid)].append(int(vid))

    all_reports_in_graph = set(idx.report_to_nodes.keys()) | set(idx.report_to_edges.keys())
    n_reports_in_graph = len(all_reports_in_graph)
    n_reports_with_anchors = sum(1 for _, ns in idx.report_to_nodes.items() if ns)

    return EcfaAppIndex(
        app=app,
        report_to_nodes={normalize_report_id(k): set(map(str, v)) for k, v in idx.report_to_nodes.items()},
        views=views,
        node_to_view_ids=dict(node_to_view_ids),
        n_reports_in_graph=int(n_reports_in_graph),
        n_reports_with_anchors=int(n_reports_with_anchors),
        union_view_reports=set(map(normalize_report_id, union_view_reports)),
    )


@dataclass
class _RefineIndex:
    vectorizer: TfidfVectorizer
    X: "np.ndarray | object"
    id_to_row: Dict[str, int]
    id_set: Set[str]


def _build_refine_index_for_app(df_app: pd.DataFrame, cfg: EcfaRunConfig) -> _RefineIndex:
    texts = df_app["text"].astype(str).tolist()
    ids = [normalize_report_id(x) for x in df_app["id"].astype(str).tolist()]
    id_to_row = {rid: i for i, rid in enumerate(ids)}


    vec = TfidfVectorizer(
        analyzer=str(cfg.refine_tfidf_analyzer),
        ngram_range=(int(cfg.refine_ngram_min), int(cfg.refine_ngram_max)),
        min_df=2,
        max_df=0.95,
        max_features=20000,
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        vec = TfidfVectorizer(
            analyzer=str(cfg.refine_tfidf_analyzer),
            ngram_range=(int(cfg.refine_ngram_min), int(cfg.refine_ngram_max)),
            min_df=1,
            max_df=0.95,
            max_features=20000,
        )
        X = vec.fit_transform(texts)

    return _RefineIndex(vectorizer=vec, X=X, id_to_row=id_to_row, id_set=set(ids))


def ecfa_retrieve_ranked(
    qid: str,
    app_idx: EcfaAppIndex,
    *,
    exclude_self: bool = True,
    retr_ranker: str = "support",
    retr_top_views: int = 20,
    retr_shared_weight: float = 0.5,
) -> Tuple[List[str], Dict[str, int], Dict[str, Tuple[float, float]]]:


    qid = normalize_report_id(qid)
    anchors = set(app_idx.report_to_nodes.get(qid, set()))
    if not anchors:
        return [], {"has_anchor": 0, "matched_views": 0}, {}

    view_hits = defaultdict(int)
    for nid in anchors:
        for vid in app_idx.node_to_view_ids.get(str(nid), []):
            view_hits[int(vid)] += 1
    if not view_hits:
        return [], {"has_anchor": 1, "matched_views": 0}, {}

    sorted_views = sorted(view_hits.items(), key=lambda x: x[1], reverse=True)
    if retr_top_views > 0:
        use_views = [v for v, _ in sorted_views[: int(retr_top_views)]]
    else:
        use_views = [v for v, _ in sorted_views]

    min_cost: Dict[str, float] = {}
    support_cnt: Dict[str, int] = defaultdict(int)
    shared_cnt: Dict[str, int] = defaultdict(int)

    for vid in use_views:
        sg = app_idx.views[vid]
        cost = float(sg.cost)
        for rid0 in sg.report_ids:
            rid = normalize_report_id(rid0)
            if not rid:
                continue
            if exclude_self and rid == qid:
                continue
            support_cnt[rid] += 1
            if rid not in min_cost or cost < min_cost[rid]:
                min_cost[rid] = cost


    if retr_shared_weight > 0:
        for rid in support_cnt.keys():
            r_nodes = app_idx.report_to_nodes.get(rid, set())
            if r_nodes:
                shared = len(anchors.intersection(r_nodes))
                shared_cnt[rid] = int(shared)

    candidate_scores: List[Tuple[str, float, float]] = []
    score_map: Dict[str, Tuple[float, float]] = {}
    for rid in support_cnt:
        supp = float(support_cnt[rid])
        shared_bonus = float(shared_cnt.get(rid, 0)) * float(retr_shared_weight)
        score = supp + shared_bonus
        cost = float(min_cost.get(rid, 1e9))
        candidate_scores.append((rid, score, cost))
        score_map[rid] = (score, cost)

    if retr_ranker == "support":
        ranked_tuples = sorted(candidate_scores, key=lambda x: (-x[1], x[2], x[0]))
    elif retr_ranker == "hybrid":
        ranked_tuples = sorted(candidate_scores, key=lambda x: (-(x[1] / (x[2] + 1e-9)), -x[1], x[2], x[0]))
    else:
        ranked_tuples = sorted(candidate_scores, key=lambda x: (x[2], -x[1], x[0]))

    ranked_ids = [r for r, _, _ in ranked_tuples]
    return ranked_ids, {"has_anchor": 1, "matched_views": int(len(view_hits))}, score_map


def _refine_rank(
    qid: str,
    ranked: List[str],
    score_map: Dict[str, Tuple[float, float]],
    refine_idx: _RefineIndex,
    *,
    max_candidates: int,
) -> List[str]:


    if not ranked:
        return ranked
    if cosine_similarity is None:

        return ranked

    qid = normalize_report_id(qid)
    if qid not in refine_idx.id_to_row:
        return ranked


    cand = ranked[: int(max_candidates)] if max_candidates > 0 else ranked[:]
    cand = [rid for rid in cand if rid in refine_idx.id_to_row]

    if not cand:
        return ranked

    q_row = refine_idx.id_to_row[qid]
    q_vec = refine_idx.X[q_row]
    c_rows = [refine_idx.id_to_row[rid] for rid in cand]
    C = refine_idx.X[c_rows]

    sims = cosine_similarity(q_vec, C).ravel().tolist()


    scored: List[Tuple[str, float, float, float]] = []
    for rid, sim in zip(cand, sims):
        ecfa_score, ecfa_cost = score_map.get(rid, (0.0, 1e9))
        scored.append((rid, float(sim), float(ecfa_score), float(ecfa_cost)))

    scored_sorted = sorted(scored, key=lambda x: (-x[1], -x[2], x[3], x[0]))
    refined_top = [rid for rid, *_ in scored_sorted]


    seen = set(refined_top)
    tail = [rid for rid in ranked if rid not in seen]
    return refined_top + tail


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

    cand = sorted(cand, key=lambda pp: (_prio(pp), pp.name), reverse=True)
    return cand[0]


def run_ecfa_v3a_and_export_per_query(
    cfg: EcfaRunConfig,
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


    clusters_path: Optional[Path]
    if clusters_csv is not None:
        clusters_path = Path(clusters_csv)
    else:
        clusters_path = _auto_find_clusters_csv(Path(cfg.out_dir), K=int(cfg.n_clusters))

    if clusters_path is None or not clusters_path.exists():
        raise FileNotFoundError(
            "clusters_csv is required (or must be discoverable under <run_dir>/clusters/). "
            "Pass clusters_csv=<path to clusters_*_K20.csv> when exporting ECFA per-query."
        )

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
            merge_sim_thresh=0.6,
            ecfa_budget_k=int(cfg.ecfa_budget),
            alpha=1.0,
            beta=0.02,
            gamma=0.01,
            backfill_node_reports_from_edges=False,
            retr_view_pool=str(cfg.retr_view_pool),
            allowed_reports=allowed,
        )


    refine_idx_by_app: Dict[str, _RefineIndex] = {}
    if bool(cfg.refine):
        for app, g in df.groupby("app"):
            refine_idx_by_app[str(app)] = _build_refine_index_for_app(g.reset_index(drop=True), cfg)

    topk_list = sorted({int(x) for x in (cfg.topk_list or []) if int(x) > 0})
    if not topk_list:
        topk_list = [20, 50, 100]
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

        ranked, diag, score_map = ecfa_retrieve_ranked(
            qid,
            app_indices[app],
            retr_ranker=str(cfg.retr_ranker),
            retr_top_views=int(cfg.retr_top_views),
            retr_shared_weight=float(cfg.retr_shared_weight),
        )


        if bool(cfg.refine) and ranked:
            ranked = _refine_rank(
                qid,
                ranked,
                score_map=score_map,
                refine_idx=refine_idx_by_app.get(app),
                max_candidates=int(cfg.refine_max_candidates),
            ) if app in refine_idx_by_app else ranked


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

        row = {
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
        rows.append(row)

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
