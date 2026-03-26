

from __future__ import annotations

import argparse
import re
import random
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Callable, Any, Literal, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


METHOD_ID_TO_NAME: Dict[str, str] = {
    "ECFA": "ECFA",
    "ECFA_NoMerge": "ECFA (No Merge)",
    "ECFA_NoCost": "ECFA w/o cost",
    "Random": "Random-k",
    "TopUni": "TopUniSize-k",
    "TopRaw": "TopRawSize-k",
}

def parse_method_ids(s: str) -> Set[str]:
    s = (s or "").strip()
    if not s:
        return set()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 1 and parts[0].lower() == "all":
        return set(METHOD_ID_TO_NAME.keys())
    unknown = [p for p in parts if p not in METHOD_ID_TO_NAME]
    if unknown:
        raise ValueError(
            f"Unknown method ids in --cap_methods: {unknown}. "
            f"Allowed: {sorted(METHOD_ID_TO_NAME.keys())} or 'all'."
        )
    return set(parts)


REQUIRED_NODE_COLS = {"id", "name", "type", "source_row_index"}
REQUIRED_EDGE_COLS = {"id", "source_id", "target_id", "relation", "source_type", "target_type", "source_row_index"}


def normalize_report_id(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return x


    if re.fullmatch(r"\d+\.0+", x):
        return x.split(".", 1)[0]


    if re.fullmatch(r"\d+(\.\d+)?[eE]\+?\d+", x):
        try:
            return str(int(float(x)))
        except Exception:
            return x


    if re.fullmatch(r"\d+\.\d+", x):
        try:
            f = float(x)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
    return x


def split_report_ids(s: object) -> list[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    tokens = [t for t in re.split(r"[|,;\s]+", s) if t]
    out: list[str] = []
    seen = set()
    for t in tokens:
        t = normalize_report_id(t)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def load_nodes_xlsx(path: str | Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
    missing = REQUIRED_NODE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"nodes file missing columns: {sorted(missing)}; got {list(df.columns)}")
    df = df[list(REQUIRED_NODE_COLS)].copy()
    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].astype(str)
    df["type"] = df["type"].astype(str)
    df["source_row_index"] = df["source_row_index"].fillna("").astype(str)
    df["report_ids"] = df["source_row_index"].map(split_report_ids)
    return df


def load_edges_xlsx(path: str | Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
    missing = REQUIRED_EDGE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"edges file missing columns: {sorted(missing)}; got {list(df.columns)}")
    df = df[list(REQUIRED_EDGE_COLS)].copy()
    for c in ["id", "source_id", "target_id"]:
        df[c] = df[c].astype(str)
    for c in ["relation", "source_type", "target_type"]:
        df[c] = df[c].astype(str)
    df["source_row_index"] = df["source_row_index"].fillna("").astype(str)
    df["report_ids"] = df["source_row_index"].map(split_report_ids)
    return df


def load_defects_optional(path: Optional[str | Path], sheet_name=0) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, sheet_name=sheet_name, dtype=str)
    elif p.suffix.lower() in [".csv"]:
        df = pd.read_csv(p, dtype=str)
    elif p.suffix.lower() in [".tsv"]:
        df = pd.read_csv(p, sep="\t", dtype=str)
    elif p.suffix.lower() in [".jsonl"]:
        import json
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"unsupported defect file suffix: {p.suffix}")
    if not {"id", "description"} <= set(df.columns):
        raise ValueError(f"defects file must have columns: id, description; got {list(df.columns)}")
    df = df[["id", "description"]].copy()
    df["id"] = df["id"].astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    return df


@dataclass
class GraphIndex:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    defects: Optional[pd.DataFrame] = None
    node_to_reports: Dict[str, Set[str]] = None
    report_to_nodes: Dict[str, Set[str]] = None
    report_to_edges: Dict[str, Set[str]] = None
    edge_id_to_row: Dict[str, dict] = None
    node_id_to_row: Dict[str, dict] = None
    defect_id_to_text: Dict[str, str] = None


def build_graph_index(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    defects: Optional[pd.DataFrame] = None,
    *,
    backfill_node_reports_from_edges: bool = False,
) -> GraphIndex:
    node_to_reports: Dict[str, Set[str]] = {}
    report_to_nodes: Dict[str, Set[str]] = {}

    for row in nodes.itertuples(index=False):
        nid = str(getattr(row, "id"))
        rids = list(getattr(row, "report_ids"))
        rs = set(map(str, rids))
        node_to_reports[nid] = rs
        for rid in rs:
            report_to_nodes.setdefault(rid, set()).add(nid)

    report_to_edges: Dict[str, Set[str]] = {}
    edge_id_to_row: Dict[str, dict] = {}

    for row in edges.to_dict(orient="records"):
        eid = str(row["id"])
        edge_id_to_row[eid] = row
        src_id = str(row.get("source_id", "") or "")
        tgt_id = str(row.get("target_id", "") or "")
        for rid in row.get("report_ids", []) or []:
            rid = str(rid)
            report_to_edges.setdefault(rid, set()).add(eid)

            if backfill_node_reports_from_edges:
                if src_id:
                    report_to_nodes.setdefault(rid, set()).add(src_id)
                    node_to_reports.setdefault(src_id, set()).add(rid)
                if tgt_id:
                    report_to_nodes.setdefault(rid, set()).add(tgt_id)
                    node_to_reports.setdefault(tgt_id, set()).add(rid)

    node_id_to_row = {str(r["id"]): r for r in nodes.to_dict(orient="records")}
    defect_id_to_text = None
    if defects is not None:
        defect_id_to_text = {str(r["id"]): str(r.get("description", "")) for r in defects.to_dict(orient="records")}

    return GraphIndex(
        nodes=nodes,
        edges=edges,
        defects=defects,
        node_to_reports=node_to_reports,
        report_to_nodes=report_to_nodes,
        report_to_edges=report_to_edges,
        edge_id_to_row=edge_id_to_row,
        node_id_to_row=node_id_to_row,
        defect_id_to_text=defect_id_to_text,
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

    report_depth: dict[str, int] = {str(rid): 0 for rid in R}

    def _nodes_from_reports(report_ids: Set[str]) -> Set[str]:
        out = {seed_node}
        for rid in report_ids:
            out |= idx.report_to_nodes.get(rid, set())
        return out

    def _truncate_reports(report_ids: Set[str]) -> Set[str]:
        if r_max <= 0 or len(report_ids) <= r_max:
            return set(report_ids)
        ranked = sorted(
            map(str, report_ids),
            key=lambda rid: (int(report_depth.get(str(rid), 10**9)), rid),
        )
        return set(ranked[: int(r_max)])

    if r_max > 0 and strict_r_max and len(R) > r_max:
        R = _truncate_reports(R)
        V = _nodes_from_reports(R)
        if v_max > 0 and len(V) >= v_max:
            return ExpansionResult(seed_node, 0, R, V, "r-max-truncated-initial+v-max")
        return ExpansionResult(seed_node, 0, R, V, "r-max-truncated-initial")

    stopped = "k-reached"
    hop = 0
    for hop in range(1, k + 1):

        V_new = set()
        for rid in R:
            V_new |= idx.report_to_nodes.get(rid, set())
        V2 = V | V_new


        R_new = set()
        for nid in V2:
            R_new |= idx.node_to_reports.get(nid, set())
        R2 = set(map(str, R_new)) | set(R)

        for rid in (R2 - R):
            report_depth.setdefault(str(rid), int(hop))

        if r_max > 0 and strict_r_max and len(R2) > r_max:
            R2 = _truncate_reports(R2)
            V2 = _nodes_from_reports(R2)
            stopped = "r-max-truncated+v-max" if (v_max > 0 and len(V2) >= v_max) else "r-max-truncated"
            V, R = V2, R2
            break

        if len(V2) == len(V) and len(R2) == len(R):
            stopped = "no-growth"
            V, R = V2, R2
            break

        V, R = V2, R2

        if (not strict_r_max) and r_max > 0 and len(R) >= r_max:
            stopped = "r-max-soft"
            break
        if len(V) >= v_max:
            stopped = "v-max"
            break

    return ExpansionResult(seed_node, hop, R, V, stopped)


@dataclass
class Subgraph:
    subgraph_id: str
    seed_node: str
    report_ids: Set[str]
    covered_report_ids: Set[str]
    node_ids: Set[str]
    edge_ids: Set[str]
    covered_universe: Set[str]
    cost: float


def induce_subgraph_from_reports(idx: GraphIndex, report_ids: Set[str]) -> tuple[Set[str], Set[str]]:
    node_ids: Set[str] = set()
    edge_ids: Set[str] = set()
    for rid in report_ids:
        node_ids |= idx.report_to_nodes.get(rid, set())
        edge_ids |= idx.report_to_edges.get(rid, set())
    return node_ids, edge_ids


def compute_cost(
    report_ids: Set[str],
    node_ids: Set[str],
    edge_ids: Set[str],
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.01,
) -> float:
    return alpha * len(report_ids) + beta * len(node_ids) + gamma * len(edge_ids)


def make_subgraph(
    idx: GraphIndex,
    subgraph_id: str,
    seed_node: str,
    report_ids: Set[str],
    universe_nodes: Set[str],
    universe_reports: Optional[Set[str]] = None,
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.01,
) -> Subgraph:
    node_ids, edge_ids = induce_subgraph_from_reports(idx, report_ids)
    covered_universe = set(node_ids) & set(universe_nodes)

    if universe_reports is not None:
        covered_report_ids = set(map(str, report_ids)) & set(map(str, universe_reports))
    else:
        covered_report_ids = set(map(str, report_ids))

    cost = compute_cost(covered_report_ids, node_ids, edge_ids, alpha=alpha, beta=beta, gamma=gamma)

    return Subgraph(
        subgraph_id=str(subgraph_id),
        seed_node=str(seed_node),
        report_ids=set(map(str, report_ids)),
        covered_report_ids=set(map(str, covered_report_ids)),
        node_ids=set(map(str, node_ids)),
        edge_ids=set(map(str, edge_ids)),
        covered_universe=set(map(str, covered_universe)),
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


def weighted_jaccard(
    parts_a: Dict[str, Set[str]],
    parts_b: Dict[str, Set[str]],
    weights: Dict[str, float],
    *,
    empty_both: float = 1.0,
) -> float:
    score = 0.0
    wsum = 0.0
    for k, w in weights.items():
        if w <= 0:
            continue
        score += w * jaccard(parts_a.get(k, set()), parts_b.get(k, set()), empty_both=empty_both)
        wsum += w
    if wsum <= 0:
        raise ValueError("weights must have at least one positive entry")
    return score / wsum


def _signature_sets(
    node_ids: Iterable[str],
    node_type_by_id: Dict[str, str],
    type_alias: Dict[str, str],
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


def merge_similar_subgraphs(
    subgraphs: List[Subgraph],
    thresh: float = 0.55,
    *,
    merge_mode: Optional[str] = None,
    node_type_by_id: Optional[Dict[str, str]] = None,
    weights: Optional[Dict[str, float]] = None,
    min_issue_jaccard: float = 0.2,
    min_phen_diag_jaccard_when_no_issue: float = 0.5,
    min_report_jaccard_when_no_issue: float = 0.0,
    max_nodes_after_merge: Optional[int] = None,
    max_reports_after_merge: Optional[int] = None,
    max_universe_after_merge: Optional[int] = None,
    type_alias: Optional[Dict[str, str]] = None,
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.01,
) -> List[Subgraph]:
    if merge_mode is None:
        mode = "report" if node_type_by_id is None else "signature"
    else:
        mode = str(merge_mode).strip().lower()
        if mode not in {"report", "signature"}:
            raise ValueError("merge_mode must be one of: report, signature")

    if mode == "report":
        remaining = sorted(
            subgraphs,
            key=lambda sg: (len(sg.report_ids), len(sg.covered_universe), len(sg.node_ids), sg.subgraph_id, sg.seed_node),
            reverse=True,
        )
        merged: List[Subgraph] = []
        used = [False] * len(remaining)

        for i, sg in enumerate(remaining):
            if used[i]:
                continue
            rep_union = set(sg.report_ids)
            cov_rep_union = set(sg.covered_report_ids)
            node_union = set(sg.node_ids)
            edge_union = set(sg.edge_ids)
            covered_union = set(sg.covered_universe)
            seeds = [sg.seed_node]

            changed = True
            while changed:
                changed = False
                for j in range(i + 1, len(remaining)):
                    if used[j]:
                        continue
                    sg2 = remaining[j]
                    if jaccard(rep_union, sg2.report_ids) >= thresh:
                        if max_nodes_after_merge is not None and len(node_union | sg2.node_ids) > max_nodes_after_merge:
                            continue
                        if max_reports_after_merge is not None and len(rep_union | sg2.report_ids) > max_reports_after_merge:
                            continue
                        if max_universe_after_merge is not None and len(covered_union | sg2.covered_universe) > max_universe_after_merge:
                            continue
                        used[j] = True
                        rep_union |= sg2.report_ids
                        cov_rep_union |= sg2.covered_report_ids
                        node_union |= sg2.node_ids
                        edge_union |= sg2.edge_ids
                        covered_union |= sg2.covered_universe
                        seeds.append(sg2.seed_node)
                        changed = True

            used[i] = True
            new_id = sg.subgraph_id if len(seeds) == 1 else f"{sg.subgraph_id}_m{len(seeds)}"
            cost = compute_cost(cov_rep_union, node_union, edge_union, alpha=alpha, beta=beta, gamma=gamma)
            merged.append(
                Subgraph(
                    subgraph_id=new_id,
                    seed_node=seeds[0],
                    report_ids=rep_union,
                    covered_report_ids=cov_rep_union,
                    node_ids=node_union,
                    edge_ids=edge_union,
                    covered_universe=covered_union,
                    cost=cost,
                )
            )
        return merged


    if node_type_by_id is None:
        raise ValueError("node_type_by_id is required when merge_mode='signature'")

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
            len(p[0].covered_report_ids),
            len(p[0].covered_universe),
            len(p[0].node_ids),
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
        cov_rep_union = set(sg.covered_report_ids)
        node_union = set(sg.node_ids)
        edge_union = set(sg.edge_ids)
        covered_union = set(sg.covered_universe)
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

                issue_j = jaccard(sig_union["ISSUE"], sig2["ISSUE"], empty_both=0.0)
                phen_diag_union = sig_union["PHEN"] | sig_union["DIAG"]
                phen_diag_2 = sig2["PHEN"] | sig2["DIAG"]
                phen_diag_j = jaccard(phen_diag_union, phen_diag_2, empty_both=0.0)

                if issue_j <= 1e-12 and phen_diag_j < min_phen_diag_jaccard_when_no_issue:
                    continue
                if issue_j <= 1e-12 and min_report_jaccard_when_no_issue > 0:
                    rep_j = jaccard(rep_union, sg2.report_ids)
                    if rep_j < min_report_jaccard_when_no_issue:
                        continue
                if issue_j > 1e-12 and issue_j < min_issue_jaccard and phen_diag_j < 0.35:
                    continue

                sim = weighted_jaccard(sig_union, sig2, weights, empty_both=0.0)
                if sim < thresh:
                    continue

                if max_nodes_after_merge is not None and len(node_union | sg2.node_ids) > max_nodes_after_merge:
                    continue
                if max_reports_after_merge is not None and len(rep_union | sg2.report_ids) > max_reports_after_merge:
                    continue
                if max_universe_after_merge is not None and len(covered_union | sg2.covered_universe) > max_universe_after_merge:
                    continue

                used[j] = True
                rep_union |= sg2.report_ids
                cov_rep_union |= sg2.covered_report_ids
                node_union |= sg2.node_ids
                edge_union |= sg2.edge_ids
                covered_union |= sg2.covered_universe
                seeds.append(sg2.seed_node)
                for k in sig_union.keys():
                    sig_union[k] |= sig2.get(k, set())
                changed = True

        used[i] = True
        new_id = sg.subgraph_id if len(seeds) == 1 else f"{sg.subgraph_id}_m{len(seeds)}"
        cost = compute_cost(cov_rep_union, node_union, edge_union, alpha=alpha, beta=beta, gamma=gamma)
        merged.append(
            Subgraph(
                subgraph_id=new_id,
                seed_node=seeds[0],
                report_ids=rep_union,
                covered_report_ids=cov_rep_union,
                node_ids=node_union,
                edge_ids=edge_union,
                covered_universe=covered_union,
                cost=cost,
            )
        )

    return merged


@dataclass
class CoverResult:
    selected: List[Subgraph]
    covered: Set[str]
    uncovered: Set[str]

Objective = Literal["report", "entity", "hybrid"]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _anneal_q(t: int, K: int, warmup_frac: float, cost_power_start: float) -> float:


    K = max(1, int(K))
    warmup_steps = max(1, int(K * float(warmup_frac)))
    start = _clamp01(float(cost_power_start))
    progress = min(1.0, float(t) / float(warmup_steps))
    return start + (1.0 - start) * progress


def lazy_greedy_set_cover(
    subgraphs: List[Subgraph],
    universe: Set[str],
    universe_reports: Optional[Set[str]] = None,
    budget: int = 50,
    min_gain: float = 1.0,
    unit_cost: bool = False,
    objective: Objective = "report",
    w_report: float = 1.0,
    w_entity: float = 1.0,
    warmup_frac: float = 0.25,
    cost_power_start: float = 1.0,
    gain_power: float = 1.0,
    fixed_cost: float = 0.0,
) -> CoverResult:


    U = set(map(str, universe))
    U_reports = set(map(str, universe_reports)) if universe_reports is not None else None

    covered_entities: Set[str] = set()
    covered_reports: Set[str] = set()
    selected: List[Subgraph] = []


    remaining: Dict[str, Subgraph] = {sg.subgraph_id: sg for sg in subgraphs}

    def _gain_nr_ne(sg: Subgraph) -> Tuple[float, int, int]:

        ne = 0
        for v in sg.covered_universe:
            if v not in covered_entities:
                ne += 1


        eff_reports = set(sg.covered_report_ids)
        if U_reports is not None:
            eff_reports &= U_reports

        nr = 0
        for r in eff_reports:
            if r not in covered_reports:
                nr += 1

        if objective == "report":
            gain = float(nr)
        elif objective == "entity":
            gain = float(ne)
        else:
            gain = float(w_report) * float(nr) + float(w_entity) * float(ne)
        return gain, nr, ne

    def _score(sg: Subgraph, q: float) -> Tuple[float, float, int, int]:
        gain, nr, ne = _gain_nr_ne(sg)
        if gain < float(min_gain):
            return -1.0, gain, nr, ne

        if unit_cost:

            score = float(gain) ** float(gain_power)
        else:
            eff_cost = max(float(sg.cost) + float(fixed_cost), 1e-9)
            score = (float(gain) ** float(gain_power)) / (eff_cost ** float(q))
        return score, gain, nr, ne


    import heapq
    heap = []
    K = int(budget)


    q0 = _anneal_q(0, K, warmup_frac, cost_power_start)

    for sg in remaining.values():
        sc, gain, nr, ne = _score(sg, q0)

        heapq.heappush(heap, (-sc, -gain, -ne, -nr, sg.subgraph_id, 0))

    while len(selected) < K and heap:
        t = len(selected)
        q = _anneal_q(t, K, warmup_frac, cost_power_start)


        while heap:
            neg_ub, neg_gain, neg_ne, neg_nr, sid, stamp = heapq.heappop(heap)
            sg = remaining.get(sid)
            if sg is None:
                continue


            sc, gain, nr, ne = _score(sg, q)
            if sc < 0:

                remaining.pop(sid, None)
                continue


            if heap:
                best_other_ub = -heap[0][0]
            else:
                best_other_ub = -1.0

            if sc + 1e-12 >= best_other_ub:

                selected.append(sg)
                remaining.pop(sid, None)

                covered_entities |= set(sg.covered_universe)

                eff_reports = set(sg.covered_report_ids)
                if U_reports is not None:
                    eff_reports &= U_reports
                covered_reports |= eff_reports


                if objective == "entity":
                    if covered_entities == U:
                        heap = []
                        break
                else:
                    if covered_entities == U and (U_reports is None or covered_reports == U_reports):
                        heap = []
                        break
                    if objective in ("report", "hybrid") and U_reports is not None and covered_reports == U_reports:
                        heap = []
                        break

                break
            else:

                heapq.heappush(heap, (-sc, -gain, -ne, -nr, sid, t))

        else:
            break

    uncovered = U - covered_entities
    return CoverResult(selected=selected, covered=covered_entities, uncovered=uncovered)


def top_k_selection(
    subgraphs: List[Subgraph],
    universe: Set[str],
    k: int = 50,
    score_fn: Optional[Callable[[Subgraph], float]] = None,
) -> CoverResult:
    if score_fn is None:
        score_fn = lambda sg: len(sg.covered_report_ids)
    sorted_subgraphs = sorted(subgraphs, key=score_fn, reverse=True)
    selected = sorted_subgraphs[: int(k)]
    U = set(map(str, universe))
    covered = set()
    for sg in selected:
        covered |= set(sg.covered_universe)
    uncovered = U - covered
    return CoverResult(selected=selected, covered=covered, uncovered=uncovered)


def random_selection(
    subgraphs: List[Subgraph],
    universe: Set[str],
    k: int = 50,
    seed: Optional[int] = None,
) -> CoverResult:
    rng = random.Random(seed) if seed is not None else random
    if len(subgraphs) <= int(k):
        selected = list(subgraphs)
    else:
        selected = rng.sample(list(subgraphs), int(k))
    U = set(map(str, universe))
    covered = set()
    for sg in selected:
        covered |= set(sg.covered_universe)
    uncovered = U - covered
    return CoverResult(selected=selected, covered=covered, uncovered=uncovered)


DEFAULT_UNIVERSE_TYPES = {"功能模块", "影响元素", "用户感知现象", "用户操作", "系统诊断信息", "问题陈述"}
DEFAULTS = {
    "seed_types": "功能模块,影响元素,用户感知现象,用户操作,系统诊断信息,问题陈述",
    "universe_types": "用户感知现象,系统诊断信息,问题陈述",
}

def build_universe(nodes_df: pd.DataFrame, types: Set[str]) -> Set[str]:
    return set(nodes_df.loc[nodes_df["type"].isin(types), "id"].astype(str).tolist())

def pick_seeds(nodes_df: pd.DataFrame, seed_types: Set[str]) -> List[str]:
    return nodes_df.loc[nodes_df["type"].isin(seed_types), "id"].astype(str).tolist()

@dataclass
class BaselineResult:
    name: str
    cover_results: List[CoverResult]
    metrics: dict

@dataclass
class PipelineResult:
    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    defects_df: Optional[pd.DataFrame]
    idx: GraphIndex
    universe: Set[str]
    universe_reports: Set[str]
    seeds: List[str]
    raw_subgraphs: List[Subgraph]
    merged_subgraphs: List[Subgraph]
    cover: CoverResult
    baselines: List[BaselineResult]
    seed_stats: List[dict]
    skipped: List[dict]
    report_counts: List[int]
    maybe_truncated_cnt: int
    node_type_by_id: dict[str, str]
    edge_only_nodes: Set[str]
    original_node_to_reports: dict[str, set[str]]
    total_reports: int
    meta_extra: dict


def calculate_nauc(values: List[float], k: int) -> float:

    if not values or k <= 0:
        return 0.0
    return float(sum(values[:k]) / float(k))


def compute_metrics(
    cover_result: CoverResult,
    total_reports_count: int,
    total_universe_count: int,
    k_limit: int,
    universe_reports_set: Optional[Set[str]] = None,
) -> dict:
    report_coverage_curve = []
    entity_coverage_curve = []
    current_reports = set()
    current_universe = set()

    selected = cover_result.selected[: int(k_limit)]
    for sg in selected:
        rep_ids = set(getattr(sg, "covered_report_ids", getattr(sg, "report_ids", set())))
        if universe_reports_set is not None:
            rep_ids &= set(universe_reports_set)

        current_reports.update(rep_ids)
        current_universe.update(set(sg.covered_universe))

        report_coverage_curve.append(len(current_reports) / max(1, int(total_reports_count)))
        entity_coverage_curve.append(len(current_universe) / max(1, int(total_universe_count)))

    while len(report_coverage_curve) < int(k_limit):
        if report_coverage_curve:
            report_coverage_curve.append(report_coverage_curve[-1])
            entity_coverage_curve.append(entity_coverage_curve[-1])
        else:
            report_coverage_curve.append(0.0)
            entity_coverage_curve.append(0.0)

    nauc_report = calculate_nauc(report_coverage_curve, int(k_limit))
    nauc_entity = calculate_nauc(entity_coverage_curve, int(k_limit))

    return {
        "report_coverage_curve": report_coverage_curve,
        "entity_coverage_curve": entity_coverage_curve,
        "nAUC@K_report": nauc_report,
        "nAUC@K_entity": nauc_entity,
        "final_report_coverage": report_coverage_curve[-1] if report_coverage_curve else 0.0,
        "final_entity_coverage": entity_coverage_curve[-1] if entity_coverage_curve else 0.0,
    }


def run_subgraph_cover(
    *,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    defects_df: Optional[pd.DataFrame] = None,
    seed_types: Optional[Set[str]] = None,
    universe_types: Optional[Set[str]] = None,
    k: int = 2,
    r_max: int = 100,
    strict_r_max: bool = True,
    merge_sim_thresh: float = 0.6,
    budget: int = 100,
    max_nodes_after_merge: int = 800,
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.01,
    unit_cost: bool = False,
    objective: Objective = "report",
    w_report: float = 1.0,
    w_entity: float = 1.0,
    backfill_node_reports_from_edges: bool = False,
    warmup_frac: float = 0.25,
    cost_power_start: float = 1.0,
    gain_power: float = 1.0,
    fixed_cost: float = 0.0,

    max_view_report_ratio: float = 1.0,
    cap_methods: Optional[Set[str]] = None,

    random_runs: int = 30,
    random_seed_base: int = 0,

    include_no_cost_baseline: bool = True,
) -> PipelineResult:
    idx = build_graph_index(nodes_df, edges_df, defects_df, backfill_node_reports_from_edges=backfill_node_reports_from_edges)
    node_type_by_id = {str(r["id"]): str(r["type"]) for _, r in nodes_df.iterrows()}

    nodes_id_set = set(nodes_df["id"].astype(str).tolist())
    endpoint_nodes = set(edges_df["source_id"].astype(str).tolist()) | set(edges_df["target_id"].astype(str).tolist())
    edge_only_nodes = endpoint_nodes - nodes_id_set
    original_node_to_reports = {str(r.id): set(map(str, r.report_ids)) for r in nodes_df.itertuples(index=False)}

    if seed_types is None:
        seed_types = set(DEFAULT_UNIVERSE_TYPES)
    if universe_types is None:
        universe_types = {"用户感知现象", "系统诊断信息", "问题陈述"}

    universe = build_universe(nodes_df, set(universe_types))
    seeds = pick_seeds(nodes_df, set(seed_types))

    universe_reports: Set[str] = set()
    for nid in universe:
        universe_reports |= set(map(str, idx.node_to_reports.get(str(nid), set())))
    total_reports = len(universe_reports)

    raw_subgraphs: List[Subgraph] = []
    seed_stats: List[dict] = []
    skipped: List[dict] = []
    report_counts: List[int] = []
    maybe_truncated_cnt = 0

    for i, seed in enumerate(seeds):
        ex = expand_reports_from_seed(
            idx,
            seed_node=str(seed),
            k=int(k),
            r_max=int(r_max),
            strict_r_max=bool(strict_r_max),
        )
        rc = len(ex.report_ids)
        report_counts.append(int(rc))

        maybe_trunc = bool(ex.stopped_reason.startswith("r-max") or (r_max > 0 and rc >= int(r_max)))
        if maybe_trunc:
            maybe_truncated_cnt += 1

        if rc == 0:
            skipped.append({"seed": str(seed), "reason": str(ex.stopped_reason), "hop_k": int(ex.hop_k)})
            continue

        sg = make_subgraph(
            idx,
            subgraph_id=f"sg_{i:05d}",
            seed_node=str(seed),
            report_ids=set(ex.report_ids),
            universe_nodes=universe,
            universe_reports=universe_reports,
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
        )
        raw_subgraphs.append(sg)
        seed_stats.append(
            {
                "seed": str(seed),
                "reports": int(rc),
                "maybe_truncated": bool(maybe_trunc),
                "stopped_reason": str(ex.stopped_reason),
                "hop_k": int(ex.hop_k),
                "nodes": int(len(sg.node_ids)),
                "edges": int(len(sg.edge_ids)),
                "covered_universe": int(len(sg.covered_universe)),
                "gamma_eff": int(len(sg.covered_report_ids)),
                "cost": float(sg.cost),
            }
        )

    merged = merge_similar_subgraphs(
        raw_subgraphs,
        thresh=float(merge_sim_thresh),
        node_type_by_id=node_type_by_id,
        max_nodes_after_merge=int(max_nodes_after_merge) if max_nodes_after_merge is not None else None,
        max_reports_after_merge=int(max(1, total_reports * 0.32)) if total_reports > 0 else None,
        max_universe_after_merge=int(max(1, len(universe) * 0.32)) if len(universe) > 0 else None,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
    )


    cap_methods = set(cap_methods or [])
    cap_ratio = float(max_view_report_ratio)

    merged_all = merged
    raw_all = raw_subgraphs

    cap_n = None
    if cap_ratio < 1.0 and universe_reports:
        cap_n = max(1, int(len(universe_reports) * cap_ratio))
        merged_cap = [sg for sg in merged_all if len(sg.covered_report_ids) <= cap_n]
        raw_cap = [sg for sg in raw_all if len(sg.covered_report_ids) <= cap_n]
    else:
        merged_cap = merged_all
        raw_cap = raw_all

    def _pool(method_id: str, kind: str) -> List[Subgraph]:
        use_cap = (method_id in cap_methods)
        if kind == "merged":
            return merged_cap if use_cap else merged_all
        return raw_cap if use_cap else raw_all


    cover = lazy_greedy_set_cover(
        _pool("ECFA", "merged"),
        universe=universe,
        universe_reports=universe_reports,
        budget=int(budget),
        min_gain=1.0,
        unit_cost=bool(unit_cost),
        objective=str(objective),
        w_report=float(w_report),
        w_entity=float(w_entity),
        warmup_frac=float(warmup_frac),
        cost_power_start=float(cost_power_start),
        gain_power=float(gain_power),
        fixed_cost=float(fixed_cost),
    )

    cover_wo_merge = lazy_greedy_set_cover(
        _pool("ECFA_NoMerge", "raw"),
        universe=universe,
        universe_reports=universe_reports,
        budget=int(budget),
        min_gain=1.0,
        unit_cost=bool(unit_cost),
        objective=str(objective),
        w_report=float(w_report),
        w_entity=float(w_entity),
        warmup_frac=float(warmup_frac),
        cost_power_start=float(cost_power_start),
        gain_power=float(gain_power),
        fixed_cost=float(fixed_cost),
    )

    baselines: List[BaselineResult] = []


    ecfa_metrics = compute_metrics(cover, total_reports, len(universe), int(budget), universe_reports_set=universe_reports)
    baselines.append(BaselineResult(name=METHOD_ID_TO_NAME["ECFA"], cover_results=[cover], metrics=ecfa_metrics))


    wo_merge_metrics = compute_metrics(cover_wo_merge, total_reports, len(universe), int(budget), universe_reports_set=universe_reports)
    baselines.append(BaselineResult(name=METHOD_ID_TO_NAME["ECFA_NoMerge"], cover_results=[cover_wo_merge], metrics=wo_merge_metrics))


    if include_no_cost_baseline:
        cover_wo_cost = lazy_greedy_set_cover(
            _pool("ECFA_NoCost", "merged"),
            universe=universe,
            universe_reports=universe_reports,
            budget=int(budget),
            min_gain=1.0,
            unit_cost=True,
            objective=str(objective),
            w_report=float(w_report),
            w_entity=float(w_entity),
            warmup_frac=float(warmup_frac),
            cost_power_start=float(cost_power_start),
            gain_power=float(gain_power),
            fixed_cost=float(fixed_cost),
        )
        metrics_wo_cost = compute_metrics(cover_wo_cost, total_reports, len(universe), int(budget), universe_reports_set=universe_reports)
        baselines.append(BaselineResult(name=METHOD_ID_TO_NAME["ECFA_NoCost"], cover_results=[cover_wo_cost], metrics=metrics_wo_cost))


    random_covs: List[CoverResult] = []
    pool_random = _pool("Random", "merged")
    for i in range(int(random_runs)):
        random_covs.append(random_selection(pool_random, universe=universe, k=int(budget), seed=int(random_seed_base) + int(i)))

    random_metrics_list = [compute_metrics(r, total_reports, len(universe), int(budget), universe_reports_set=universe_reports) for r in random_covs]
    curves_r = np.array([m["report_coverage_curve"] for m in random_metrics_list], dtype=float)
    curves_e = np.array([m["entity_coverage_curve"] for m in random_metrics_list], dtype=float)

    avg_report_curve = curves_r.mean(axis=0).tolist()
    std_report_curve = curves_r.std(axis=0, ddof=0).tolist()
    avg_entity_curve = curves_e.mean(axis=0).tolist()
    std_entity_curve = curves_e.std(axis=0, ddof=0).tolist()

    auc_r_list = [m["nAUC@K_report"] for m in random_metrics_list]
    auc_e_list = [m["nAUC@K_entity"] for m in random_metrics_list]
    fin_r_list = [m["final_report_coverage"] for m in random_metrics_list]
    fin_e_list = [m["final_entity_coverage"] for m in random_metrics_list]

    def _mean(xs): return float(sum(xs) / len(xs)) if xs else 0.0
    def _std(xs): return float(statistics.pstdev(xs)) if xs and len(xs) > 1 else 0.0

    avg_random_metrics = {
        "report_coverage_curve": avg_report_curve,
        "entity_coverage_curve": avg_entity_curve,
        "report_coverage_std": std_report_curve,
        "entity_coverage_std": std_entity_curve,
        "nAUC@K_report": calculate_nauc(avg_report_curve, int(budget)),
        "nAUC@K_entity": calculate_nauc(avg_entity_curve, int(budget)),
        "final_report_coverage": _mean(fin_r_list),
        "final_entity_coverage": _mean(fin_e_list),
        "std_nAUC@K_report": _std(auc_r_list),
        "std_nAUC@K_entity": _std(auc_e_list),
        "std_final_report_coverage": _std(fin_r_list),
        "std_final_entity_coverage": _std(fin_e_list),
    }
    baselines.append(BaselineResult(name=METHOD_ID_TO_NAME["Random"], cover_results=random_covs, metrics=avg_random_metrics))


    top_uni_size_res = top_k_selection(_pool("TopUni", "merged"), universe=universe, k=int(budget), score_fn=lambda sg: len(sg.covered_universe))
    baselines.append(BaselineResult(
        name=METHOD_ID_TO_NAME["TopUni"],
        cover_results=[top_uni_size_res],
        metrics=compute_metrics(top_uni_size_res, total_reports, len(universe), int(budget), universe_reports_set=universe_reports),
    ))


    top_raw_size_res = top_k_selection(_pool("TopRaw", "raw"), universe=universe, k=int(budget), score_fn=lambda sg: (len(sg.node_ids) + len(sg.edge_ids)))
    baselines.append(BaselineResult(
        name=METHOD_ID_TO_NAME["TopRaw"],
        cover_results=[top_raw_size_res],
        metrics=compute_metrics(top_raw_size_res, total_reports, len(universe), int(budget), universe_reports_set=universe_reports),
    ))

    meta_extra = {
        "cap_ratio": float(cap_ratio),
        "cap_n_reports": int(cap_n) if cap_n is not None else -1,
        "cap_methods": ",".join(sorted(cap_methods)) if cap_methods else "",
        "random_runs": int(random_runs),
        "random_seed_base": int(random_seed_base),
    }

    return PipelineResult(
        nodes_df=nodes_df,
        edges_df=edges_df,
        defects_df=defects_df,
        idx=idx,
        universe=universe,
        universe_reports=universe_reports,
        seeds=seeds,
        raw_subgraphs=raw_subgraphs,
        merged_subgraphs=merged,
        cover=cover,
        baselines=baselines,
        seed_stats=seed_stats,
        skipped=skipped,
        report_counts=report_counts,
        maybe_truncated_cnt=int(maybe_truncated_cnt),
        node_type_by_id=node_type_by_id,
        edge_only_nodes=edge_only_nodes,
        original_node_to_reports=original_node_to_reports,
        total_reports=total_reports,
        meta_extra=meta_extra,
    )


def _safe_sheet_name(name: str) -> str:
    bad = set("[]:*?/\\")
    name = "".join([c if c not in bad else "_" for c in str(name)])
    name = name.strip() or "sheet"
    return name[:31]


def _count_reports_from_source_row_index(x) -> int:
    return len(split_report_ids(x))


def build_excel_sheets(
    *,
    nodes_df: pd.DataFrame,
    idx: GraphIndex,
    universe: set[str],
    universe_reports: set[str],
    merged_all: list[Subgraph],
    raw_all: list[Subgraph],
    cover: CoverResult,
    baselines: List[Any],
    seed_stats: list[dict],
    skipped: list[dict],
    original_node_to_reports: dict[str, set[str]],
    edge_only_nodes: set[str],
    maybe_truncated_cnt: int = 0,
    n_seeds_total: int = 0,
    eps_overlap: float = 0.1,
    meta_extra: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 1.0

    def _pairwise_overlap_stats(sets: List[set], eps: float = 0.1) -> dict:
        n = len(sets)
        if n < 2:
            return {"avg": 0.0, "max": 0.0, "avg_nz": 0.0, "frac_pos0": 0.0, "frac_poseps": 0.0}
        vals: List[float] = []
        pos0: List[float] = []
        pose: List[float] = []
        for i in range(n):
            si = set(sets[i])
            for j in range(i + 1, n):
                v = _jaccard(si, set(sets[j]))
                vals.append(v)
                if v > 0.0:
                    pos0.append(v)
                if v > float(eps):
                    pose.append(v)
        avg = (sum(vals) / len(vals)) if vals else 0.0
        mx = max(vals) if vals else 0.0
        frac_pos0 = (len(pos0) / len(vals)) if vals else 0.0
        frac_poseps = (len(pose) / len(vals)) if vals else 0.0
        avg_nz = (sum(pos0) / len(pos0)) if pos0 else 0.0
        return {"avg": float(avg), "max": float(mx), "avg_nz": float(avg_nz), "frac_pos0": float(frac_pos0), "frac_poseps": float(frac_poseps)}

    id_str = nodes_df["id"].astype(str)
    uncovered_ids = set(map(str, cover.uncovered))

    cand_cov = set()
    for sg in merged_all:
        cand_cov |= set(map(str, sg.covered_universe))


    backfilled_pairs = 0
    backfilled_nodes = 0
    for nid, orig in original_node_to_reports.items():
        after = set(idx.node_to_reports.get(str(nid), set()))
        inc = after - set(orig)
        if inc:
            backfilled_nodes += 1
            backfilled_pairs += len(inc)

    u_df = nodes_df.loc[id_str.isin(universe)].copy()
    if "source_row_index" in u_df.columns:
        src = u_df["source_row_index"]
        empty_src = int(src.isna().sum() + (src.astype(str).str.strip() == "").sum())
    else:
        empty_src = 0

    seed_stats_cols = ["seed", "reports", "maybe_truncated", "stopped_reason", "hop_k", "nodes", "edges", "covered_universe", "gamma_eff", "cost"]
    skipped_cols = ["seed", "reason", "hop_k"]
    df_seed_stats = pd.DataFrame(seed_stats, columns=seed_stats_cols)
    df_skipped = pd.DataFrame(skipped, columns=skipped_cols)

    uncovered_df = nodes_df.loc[id_str.isin(uncovered_ids)].copy()
    if not uncovered_df.empty:
        uncovered_df["id"] = uncovered_df["id"].astype(str)
        if "source_row_index" not in uncovered_df.columns:
            uncovered_df["source_row_index"] = ""
        uncovered_df["n_reports_in_source"] = uncovered_df["source_row_index"].apply(_count_reports_from_source_row_index)
        uncovered_df["ever_in_any_candidate"] = uncovered_df["id"].apply(lambda x: x in cand_cov)
        uncovered_df.sort_values(["ever_in_any_candidate", "n_reports_in_source"], ascending=[True, False], inplace=True)


    metrics_rows = []
    for b in baselines:
        row = {
            "Method": b.name,
            "nAUC@K (Report)": b.metrics.get("nAUC@K_report", 0.0),
            "nAUC@K (Entity)": b.metrics.get("nAUC@K_entity", 0.0),
            "Coverage@K (Report)": b.metrics.get("final_report_coverage", 0.0),
            "Coverage@K (Entity)": b.metrics.get("final_entity_coverage", 0.0),
        }
        if b.name == METHOD_ID_TO_NAME["Random"]:
            row["Std nAUC@K (Report)"] = b.metrics.get("std_nAUC@K_report", 0.0)
            row["Std nAUC@K (Entity)"] = b.metrics.get("std_nAUC@K_entity", 0.0)
            row["Std Coverage@K (Report)"] = b.metrics.get("std_final_report_coverage", 0.0)
            row["Std Coverage@K (Entity)"] = b.metrics.get("std_final_entity_coverage", 0.0)
        metrics_rows.append(row)
    df_metrics = pd.DataFrame(metrics_rows)


    def _build_curve_df(curve_key: str, std_key: str, std_col_suffix: str) -> pd.DataFrame:
        max_k = 0
        curve_data = {}
        std_data = {}
        for b in baselines:
            curve = b.metrics.get(curve_key, [])
            if curve:
                curve_data[b.name] = list(curve)
                max_k = max(max_k, len(curve))
            if b.name == METHOD_ID_TO_NAME["Random"] and b.metrics.get(std_key):
                std_data[b.name] = list(b.metrics[std_key])

        df = pd.DataFrame({"k": range(1, max_k + 1)})
        for name, curve in curve_data.items():
            if len(curve) < max_k and curve:
                curve = curve + [curve[-1]] * (max_k - len(curve))
            df[name] = curve
        if METHOD_ID_TO_NAME["Random"] in std_data:
            std = std_data[METHOD_ID_TO_NAME["Random"]]
            if len(std) < max_k and std:
                std = std + [std[-1]] * (max_k - len(std))
            df[f"{METHOD_ID_TO_NAME['Random']}_{std_col_suffix}"] = std
        return df

    df_curve_report = _build_curve_df("report_coverage_curve", "report_coverage_std", "std")
    df_curve_entity = _build_curve_df("entity_coverage_curve", "entity_coverage_std", "std")


    def _mean_std(xs: List[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        if len(xs) == 1:
            return float(xs[0]), 0.0
        return float(statistics.mean(xs)), float(statistics.pstdev(xs))

    def _fmt_mean_std(mean: float, std: float, *, digits: int, show_pm: bool) -> str:
        return f"{mean:.{digits}f} ± {std:.{digits}f}" if show_pm else f"{mean:.{digits}f}"

    redundancy_rows_report = []
    redundancy_rows_entity = []

    for b in baselines:
        avg_r_vals: List[float] = []
        max_r_vals: List[float] = []
        avg_r_nz_vals: List[float] = []
        frac_r_pos0_vals: List[float] = []
        frac_r_poseps_vals: List[float] = []

        avg_e_vals: List[float] = []
        max_e_vals: List[float] = []
        avg_e_nz_vals: List[float] = []
        frac_e_pos0_vals: List[float] = []
        frac_e_poseps_vals: List[float] = []

        k_used = 0
        for cr in getattr(b, "cover_results", []) or []:
            selected_sgs = list(getattr(cr, "selected", []) or [])
            k_run = len(selected_sgs)
            k_used = max(k_used, k_run)

            report_sets = []
            for sg in selected_sgs:
                reps = set(getattr(sg, "covered_report_ids", getattr(sg, "report_ids", set())))
                reps &= set(universe_reports)
                if reps:
                    report_sets.append(reps)

            entity_sets = [set(sg.covered_universe) for sg in selected_sgs if sg.covered_universe]

            s_r = _pairwise_overlap_stats(report_sets, eps=eps_overlap)
            s_e = _pairwise_overlap_stats(entity_sets, eps=eps_overlap)

            avg_r_vals.append(s_r["avg"])
            max_r_vals.append(s_r["max"])
            avg_r_nz_vals.append(s_r["avg_nz"])
            frac_r_pos0_vals.append(s_r["frac_pos0"])
            frac_r_poseps_vals.append(s_r["frac_poseps"])

            avg_e_vals.append(s_e["avg"])
            max_e_vals.append(s_e["max"])
            avg_e_nz_vals.append(s_e["avg_nz"])
            frac_e_pos0_vals.append(s_e["frac_pos0"])
            frac_e_poseps_vals.append(s_e["frac_poseps"])

        avg_r_mean, avg_r_std = _mean_std(avg_r_vals)
        max_r_mean, max_r_std = _mean_std(max_r_vals)
        avg_r_nz_mean, avg_r_nz_std = _mean_std(avg_r_nz_vals)
        frac_r_pos0_mean, frac_r_pos0_std = _mean_std(frac_r_pos0_vals)
        frac_r_poseps_mean, frac_r_poseps_std = _mean_std(frac_r_poseps_vals)

        avg_e_mean, avg_e_std = _mean_std(avg_e_vals)
        max_e_mean, max_e_std = _mean_std(max_e_vals)
        avg_e_nz_mean, avg_e_nz_std = _mean_std(avg_e_nz_vals)
        frac_e_pos0_mean, frac_e_pos0_std = _mean_std(frac_e_pos0_vals)
        frac_e_poseps_mean, frac_e_poseps_std = _mean_std(frac_e_poseps_vals)

        is_random = str(b.name) == METHOD_ID_TO_NAME["Random"]

        redundancy_rows_report.append(
            {
                "Method": b.name,
                "K": int(k_used),
                "AvgOverlap ↓": _fmt_mean_std(avg_r_mean, avg_r_std, digits=4, show_pm=is_random),
                "MaxOverlap ↓": _fmt_mean_std(max_r_mean, max_r_std, digits=4, show_pm=is_random),
                "AvgOverlap_nz ↓": _fmt_mean_std(avg_r_nz_mean, avg_r_nz_std, digits=4, show_pm=is_random),
                "Frac(J>0) ↓": _fmt_mean_std(frac_r_pos0_mean, frac_r_pos0_std, digits=3, show_pm=is_random),
                f"Frac(J>{eps_overlap}) ↓": _fmt_mean_std(frac_r_poseps_mean, frac_r_poseps_std, digits=3, show_pm=is_random),
            }
        )
        redundancy_rows_entity.append(
            {
                "Method": b.name,
                "K": int(k_used),
                "AvgOverlap ↓": _fmt_mean_std(avg_e_mean, avg_e_std, digits=4, show_pm=is_random),
                "MaxOverlap ↓": _fmt_mean_std(max_e_mean, max_e_std, digits=4, show_pm=is_random),
                "AvgOverlap_nz ↓": _fmt_mean_std(avg_e_nz_mean, avg_e_nz_std, digits=4, show_pm=is_random),
                "Frac(J>0) ↓": _fmt_mean_std(frac_e_pos0_mean, frac_e_pos0_std, digits=3, show_pm=is_random),
                f"Frac(J>{eps_overlap}) ↓": _fmt_mean_std(frac_e_poseps_mean, frac_e_poseps_std, digits=3, show_pm=is_random),
            }
        )

    df_redundancy_report = pd.DataFrame(redundancy_rows_report)
    df_redundancy_entity = pd.DataFrame(redundancy_rows_entity)


    def _collect_selected_views_stats(method_name: str, cr: CoverResult) -> List[dict]:
        rows = []
        for rank, sg in enumerate(list(getattr(cr, "selected", []) or []), start=1):
            rows.append({
                "Method": str(method_name),
                "rank": int(rank),
                "subgraph_id": str(sg.subgraph_id),
                "seed_node": str(sg.seed_node),
                "|Gamma_eff|": int(len(getattr(sg, "covered_report_ids", set()) or [])),
                "|Gamma_all|": int(len(getattr(sg, "report_ids", set()) or [])),
                "|V|": int(len(getattr(sg, "node_ids", set()) or [])),
                "|E|": int(len(getattr(sg, "edge_ids", set()) or [])),
                "|CoveredUniverse|": int(len(getattr(sg, "covered_universe", set()) or [])),
                "cost": float(getattr(sg, "cost", 0.0)),
            })
        return rows

    sv_rows = []
    for b in baselines or []:
        for cr in getattr(b, "cover_results", []) or []:
            sv_rows.extend(_collect_selected_views_stats(b.name, cr))
    df_selected_views_stats = pd.DataFrame(sv_rows)


    merged_rep_counts = [len(getattr(sg, "covered_report_ids", [])) for sg in merged_all]
    merged_rep_counts.sort()
    n_merged = len(merged_rep_counts)
    max_cov_rep = merged_rep_counts[-1] if n_merged else 0
    p95_cov_rep = merged_rep_counts[int(n_merged * 0.95)] if n_merged else 0

    raw_rep_counts = [len(getattr(sg, "covered_report_ids", [])) for sg in raw_all]
    raw_rep_counts.sort()
    n_raw = len(raw_rep_counts)

    base_meta_rows = [
        {"key": "edge_only_nodes", "value": int(len(edge_only_nodes))},
        {"key": "index_backfilled_nodes", "value": int(backfilled_nodes)},
        {"key": "index_backfilled_report_links", "value": int(backfilled_pairs)},
        {"key": "universe_nodes_with_empty_source_row_index", "value": int(empty_src)},
        {"key": "n_reports_universe", "value": int(len(universe_reports))},
        {"key": "n_candidates_raw", "value": int(n_raw)},
        {"key": "n_candidates_merged", "value": int(n_merged)},
        {"key": "truncated_seed_ratio", "value": float((maybe_truncated_cnt / n_seeds_total) if n_seeds_total else 0.0)},
        {"key": "max_covered_reports_candidate", "value": int(max_cov_rep)},
        {"key": "p95_covered_reports_candidate", "value": int(p95_cov_rep)},
        {"key": "eps_overlap", "value": float(eps_overlap)},
    ]
    if meta_extra:
        for k, v in meta_extra.items():
            base_meta_rows.append({"key": str(k), "value": v})

    df_meta = pd.DataFrame(base_meta_rows)

    sheets = {
        "Metrics": df_metrics,
        "CoverageCurve_Report": df_curve_report,
        "CoverageCurve_Entity": df_curve_entity,
        "Redundancy_Report": df_redundancy_report,
        "Redundancy_Entity": df_redundancy_entity,
        "SelectedViews_Stats_AllMethods": df_selected_views_stats,
        "SeedStats": df_seed_stats,
        "SkippedSeeds": df_skipped,
        "UncoveredNodes": uncovered_df,
        "meta": df_meta,
    }
    return sheets


def write_excel_report(out_path: str | Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as xw:
        for name, df in sheets.items():
            if df is None or (not isinstance(df, pd.DataFrame)):
                continue
            df.to_excel(xw, sheet_name=_safe_sheet_name(name), index=False)
    return out_path


def plot_coverage_curves(
    baselines: List[Any],
    out_base: Path,
    *,
    metric: Literal["report", "entity"] = "report",
):


    styles = {
        METHOD_ID_TO_NAME["ECFA"]: {"marker": "o", "linewidth": 2, "label": "ECFA (Full)"},
        METHOD_ID_TO_NAME["ECFA_NoCost"]: {"marker": "x", "linewidth": 2, "label": "ECFA (No Cost)"},
        METHOD_ID_TO_NAME["ECFA_NoMerge"]: {"marker": "s", "linewidth": 2, "label": "ECFA (No Merge)"},
        METHOD_ID_TO_NAME["Random"]: {"linestyle": "--", "label": "Random-k"},
        METHOD_ID_TO_NAME["TopUni"]: {"linestyle": "-.", "label": "TopUniSize-k"},
        METHOD_ID_TO_NAME["TopRaw"]: {"linestyle": ":", "label": "TopRawSize-k"},
    }

    curve_key = "report_coverage_curve" if metric == "report" else "entity_coverage_curve"
    std_key = "report_coverage_std" if metric == "report" else "entity_coverage_std"

    max_k = 0
    valid = []
    for b in baselines or []:
        curve = list(b.metrics.get(curve_key, []))
        if not curve:
            continue
        valid.append(b)
        max_k = max(max_k, len(curve))
    if not valid:
        return None

    plt.figure(figsize=(8, 5), dpi=140)
    xs = list(range(1, max_k + 1))

    for b in valid:
        name = b.name
        curve = list(b.metrics.get(curve_key, []))
        if len(curve) < max_k and curve:
            curve = curve + [curve[-1]] * (max_k - len(curve))

        style = styles.get(name, {"label": name})

        if name == METHOD_ID_TO_NAME["Random"] and b.metrics.get(std_key):
            std = list(b.metrics[std_key])
            if len(std) < max_k and std:
                std = std + [std[-1]] * (max_k - len(std))
            means = np.array(curve, dtype=float)
            stds = np.array(std, dtype=float)
            plt.plot(xs, curve, label=style["label"], linestyle=style.get("linestyle"))
            plt.fill_between(xs, means - stds, means + stds, alpha=0.2)
        else:
            plt.plot(
                xs, curve,
                label=style.get("label", name),
                linestyle=style.get("linestyle"),
                marker=style.get("marker"),
                linewidth=style.get("linewidth", 1.5),
            )

    plt.xlabel("Budget (k)")
    plt.ylabel("Coverage" if metric == "report" else "Entity Coverage")
    plt.title("Report Coverage @ k" if metric == "report" else "Entity Coverage @ k")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = out_base.with_suffix(".png")
    svg_path = out_base.with_suffix(".svg")
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.close()
    print(f"[Plot] Saved {metric} curve to: {png_path} and {svg_path}")
    return png_path, svg_path


def _parse_type_list(s: str) -> set[str]:
    return {x.strip() for x in str(s).split(",") if x.strip()}

def _list_apps_by_sheets(dataset_xlsx: Path, *, include_deprecated: bool) -> list[str]:
    xl = pd.ExcelFile(dataset_xlsx)
    names = [str(n) for n in xl.sheet_names]
    if include_deprecated:
        return names
    return [n for n in names if not n.strip().startswith("!")]

def main() -> int:
    base_dir = Path(__file__).resolve().parents[0]
    default_input_root = base_dir / "ROOT_glm4.7"
    default_dataset_xlsx = base_dir / "RQ2根数据集.xlsx"
    default_output_root = base_dir / "shared_output"

    ap = argparse.ArgumentParser(description="Batch generate RQ2 outputs per APP (xlsx + plots) with baselines and ablations.")
    ap.add_argument("--input_root", default=str(default_input_root))
    ap.add_argument("--dataset_xlsx", default=str(default_dataset_xlsx))
    ap.add_argument("--output_root", default=str(default_output_root))
    ap.add_argument("--apps", default="")
    ap.add_argument("--include_deprecated", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")


    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--r_max", type=int, default=200)
    ap.add_argument("--soft_r_max", action="store_true")
    ap.add_argument("--merge_sim_thresh", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--eps_overlap", type=float, default=0.1)


    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=0.4)


    ap.add_argument("--objective", choices=["report", "entity", "hybrid"], default="report")
    ap.add_argument("--w_report", type=float, default=1.0)
    ap.add_argument("--w_entity", type=float, default=1.0)


    ap.add_argument("--backfill_node_reports_from_edges", action="store_true", default=False)


    ap.add_argument("--warmup_frac", type=float, default=0.25)
    ap.add_argument("--cost_power_start", type=float, default=1.0)
    ap.add_argument("--gain_power", type=float, default=1.0)
    ap.add_argument("--fixed_cost", type=float, default=0.0)


    ap.add_argument("--max_view_report_ratio", type=float, default=1.0,
                    help="Cap a view by |Gamma_eff| <= ratio * |universe_reports| (only applied to methods in --cap_methods).")
    ap.add_argument("--cap_methods", type=str, default="",
                    help="Comma-separated method IDs to apply cap. e.g., ECFA,ECFA_NoCost. Use 'all' for all methods. "
                         f"Allowed: {','.join(sorted(METHOD_ID_TO_NAME.keys()))}")


    ap.add_argument("--random_runs", type=int, default=30)
    ap.add_argument("--random_seed_base", type=int, default=0)


    ap.add_argument("--curve_metric", choices=["auto", "report", "entity", "both"], default="auto",
                    help="Which coverage curve to plot/export. 'auto': follow objective (entity->entity else report).")

    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    dataset_xlsx = Path(args.dataset_xlsx).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if str(args.apps).strip():
        apps = [a.strip() for a in str(args.apps).split(",") if a.strip()]
    else:
        apps = _list_apps_by_sheets(dataset_xlsx, include_deprecated=bool(args.include_deprecated))

    seed_types = _parse_type_list(DEFAULTS["seed_types"])
    universe_types = _parse_type_list(DEFAULTS["universe_types"])
    strict_r_max = not bool(args.soft_r_max)
    cap_methods = parse_method_ids(args.cap_methods)


    if args.curve_metric == "auto":
        curve_metric = "entity" if str(args.objective) == "entity" else "report"
    else:
        curve_metric = args.curve_metric

    print(f"[Start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[InputRoot] {input_root.as_posix()}")
    print(f"[Dataset] {dataset_xlsx.as_posix()}")
    print(f"[OutputRoot] {output_root.as_posix()}")
    print(f"[Apps] {len(apps)}")
    print(f"[Objective] {args.objective}")
    print(f"[ProvenanceBackfill] {args.backfill_node_reports_from_edges}")
    print(f"[Cap] ratio={args.max_view_report_ratio} methods={','.join(sorted(cap_methods)) if cap_methods else '(none)'}")
    print(f"[CurveMetric] {curve_metric}")

    ok = 0
    skipped = 0
    failed = 0

    for app in apps:
        app_dir = input_root / app
        unify_dir = app_dir / "unify"
        nodes_path = unify_dir / "nodes.xlsx"
        edges_path = unify_dir / "edges.xlsx"

        out_dir = output_root / app
        out_xlsx = out_dir / "rq2_report.xlsx"

        if args.skip_existing and out_xlsx.exists():
            print(f"[Skip] {app} exists: {out_xlsx.as_posix()}")
            skipped += 1
            continue

        if not nodes_path.exists() or not edges_path.exists():
            print(f"[Fail] {app} missing unify files: nodes={nodes_path.exists()} edges={edges_path.exists()}")
            failed += 1
            continue

        try:
            print(f"\n=== {app} ===")
            out_dir.mkdir(parents=True, exist_ok=True)

            nodes_df = load_nodes_xlsx(nodes_path)
            edges_df = load_edges_xlsx(edges_path)
            defects_df = load_defects_optional(dataset_xlsx, sheet_name=app)

            res = run_subgraph_cover(
                nodes_df=nodes_df,
                edges_df=edges_df,
                defects_df=defects_df,
                seed_types=seed_types,
                universe_types=universe_types,
                k=int(args.k),
                r_max=int(args.r_max),
                strict_r_max=bool(strict_r_max),
                merge_sim_thresh=float(args.merge_sim_thresh),
                budget=int(args.budget),
                alpha=float(args.alpha),
                beta=float(args.beta),
                gamma=float(args.gamma),
                unit_cost=False,
                objective=str(args.objective),
                w_report=float(args.w_report),
                w_entity=float(args.w_entity),
                backfill_node_reports_from_edges=bool(args.backfill_node_reports_from_edges),
                warmup_frac=float(args.warmup_frac),
                cost_power_start=float(args.cost_power_start),
                gain_power=float(args.gain_power),
                fixed_cost=float(args.fixed_cost),
                max_view_report_ratio=float(args.max_view_report_ratio),
                cap_methods=cap_methods,
                random_runs=int(args.random_runs),
                random_seed_base=int(args.random_seed_base),
                include_no_cost_baseline=True,
            )

            sheets = build_excel_sheets(
                nodes_df=res.nodes_df,
                idx=res.idx,
                universe=res.universe,
                universe_reports=res.universe_reports,
                merged_all=res.merged_subgraphs,
                raw_all=res.raw_subgraphs,
                cover=res.cover,
                baselines=res.baselines,
                seed_stats=res.seed_stats,
                skipped=res.skipped,
                original_node_to_reports=res.original_node_to_reports,
                edge_only_nodes=res.edge_only_nodes,
                maybe_truncated_cnt=res.maybe_truncated_cnt,
                n_seeds_total=len(res.seeds),
                eps_overlap=float(args.eps_overlap),
                meta_extra=res.meta_extra,
            )
            write_excel_report(out_xlsx, sheets)


            if curve_metric in ("report", "entity"):
                plot_coverage_curves(res.baselines, out_dir / f"coverage_curve_{curve_metric}", metric=curve_metric)
            else:
                plot_coverage_curves(res.baselines, out_dir / "coverage_curve_report", metric="report")
                plot_coverage_curves(res.baselines, out_dir / "coverage_curve_entity", metric="entity")

            print(f"[OK] {app} -> {out_dir.as_posix()}")
            ok += 1

        except Exception as e:
            print(f"[Fail] {app}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n=== Summary ===")
    print(f"OK: {ok}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"[End] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
