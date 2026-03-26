from __future__ import annotations

import argparse
import re
import random
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Callable, Any, Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    try:
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
            print(f"[Warn] defects file suffix not supported: {p.suffix}. Ignore defects.")
            return None
    except Exception as e:
        print(f"[Warn] failed to load defects for sheet={sheet_name}: {e}. Ignore defects.")
        return None

    cols = set(df.columns)
    if not {"id", "description"} <= cols:
        print(f"[Warn] defects sheet '{sheet_name}' missing required columns {{'id','description'}}, got={list(df.columns)}. Ignore defects.")
        return None

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
        node_to_reports[nid] = set(map(str, rids))
        for rid in rids:
            report_to_nodes.setdefault(str(rid), set()).add(nid)

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

    def _report_sort_key(x: str):
        xs = str(x)
        return (0, int(xs)) if xs.isdigit() else (1, xs)

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
            key=lambda rid: (
                int(report_depth.get(str(rid), 10**9)),
                (0, int(str(rid))) if str(rid).isdigit() else (1, str(rid)),
            ),
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
        for rid in sorted(R, key=_report_sort_key):
            V_new |= idx.report_to_nodes.get(rid, set())
        V2 = V | V_new

        R_new = set()
        for nid in sorted(V2):
            R_new |= idx.node_to_reports.get(nid, set())
        R_new = set(map(str, R_new))
        R2 = set(R) | set(R_new)

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
    for rid in report_ids:
        node_ids |= idx.report_to_nodes.get(rid, set())

    edge_ids: Set[str] = set()
    for rid in report_ids:
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


    cost = compute_cost(report_ids, node_ids, edge_ids, alpha=alpha, beta=beta, gamma=gamma)

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


def weighted_jaccard(parts_a: Dict[str, Set[str]], parts_b: Dict[str, Set[str]], weights: Dict[str, float], *, empty_both: float = 1.0) -> float:
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


def greedy_set_cover(
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
    remaining = list(subgraphs)

    def _score(sg: Subgraph) -> tuple[float, int, int]:
        new_entities = len(set(sg.covered_universe) - covered_entities)

        eff_reports = set(sg.covered_report_ids)
        if U_reports is not None:
            eff_reports = eff_reports & U_reports
        new_reports = len(eff_reports - covered_reports)

        if objective == "report":
            gain = float(new_reports)
        elif objective == "entity":
            gain = float(new_entities)
        else:
            gain = float(w_report) * float(new_reports) + float(w_entity) * float(new_entities)

        return gain, new_reports, new_entities

    while len(selected) < int(budget):
        best = None
        best_score = -1.0
        best_gain = -1.0
        best_new_reports = -1
        best_new_entities = -1

        t = len(selected)
        K = int(budget)
        warmup_steps = max(1, int(K * warmup_frac))
        progress = min(1.0, t / float(warmup_steps))
        q = float(cost_power_start) + (1.0 - float(cost_power_start)) * progress

        for sg in remaining:
            gain, nr, ne = _score(sg)
            if gain < float(min_gain):
                continue

            if unit_cost:
                score = gain
            else:
                eff_cost = max(float(sg.cost) + float(fixed_cost), 1e-9)
                score = (float(gain) ** float(gain_power)) / (eff_cost ** q)

            if (score > best_score) or (
                abs(score - best_score) <= 1e-12 and (
                    gain > best_gain or (
                        abs(gain - best_gain) <= 1e-12 and (
                            ne > best_new_entities or (
                                ne == best_new_entities and (
                                    nr > best_new_reports or (nr == best_new_reports and sg.subgraph_id < (best.subgraph_id if best else "~~~"))
                                )
                            )
                        )
                    )
                )
            ):
                best = sg
                best_score = score
                best_gain = gain
                best_new_reports = nr
                best_new_entities = ne

        if best is None:
            break

        selected.append(best)
        covered_entities |= set(best.covered_universe)

        eff_reports = set(best.covered_report_ids)
        if U_reports is not None:
            eff_reports = eff_reports & U_reports
        covered_reports |= eff_reports

        remaining = [sg for sg in remaining if sg.subgraph_id != best.subgraph_id]

        if covered_entities == U:
            if objective == "entity":
                break
            if U_reports is None or covered_reports == U_reports:
                break
        if objective in ("report", "hybrid") and U_reports is not None and covered_reports == U_reports:
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


BudgetMode = Literal["count", "cost"]


def _norm_budget_mode(x: str) -> BudgetMode:
    s = str(x or "").strip().lower()
    return "cost" if s == "cost" else "count"


def _cover_from_selected(selected: List[Subgraph], universe: Set[str]) -> CoverResult:
    U = set(map(str, universe))
    covered: Set[str] = set()
    for sg in selected:
        covered |= set(map(str, getattr(sg, "covered_universe", set())))
    uncovered = U - covered
    return CoverResult(selected=list(selected), covered=covered, uncovered=uncovered)


def _feasible_subset_from_order(
    order: List[Subgraph],
    *,
    B_cost_ref: Optional[float],
    k_limit: int,
    skip_infeasible: bool = True,
) -> List[Subgraph]:
    if B_cost_ref is None or float(B_cost_ref) <= 0:
        return list(order[: int(k_limit)])

    B = float(B_cost_ref)
    selected: List[Subgraph] = []
    cum_cost = 0.0
    for sg in order:
        if len(selected) >= int(k_limit):
            break
        c = float(getattr(sg, "cost", 0.0))
        if cum_cost + c <= B + 1e-12:
            selected.append(sg)
            cum_cost += c
        else:
            if not skip_infeasible:
                break
            continue
    return selected


def greedy_set_cover_cost_budget(
    subgraphs: List[Subgraph],
    universe: Set[str],
    universe_reports: Optional[Set[str]] = None,
    *,
    k_limit: int = 50,
    budget_cost_ref: Optional[float] = None,
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
    B = float(budget_cost_ref) if (budget_cost_ref is not None and float(budget_cost_ref) > 0) else None
    if B is None:
        return greedy_set_cover(
            subgraphs=subgraphs,
            universe=universe,
            universe_reports=universe_reports,
            budget=int(k_limit),
            min_gain=min_gain,
            unit_cost=unit_cost,
            objective=objective,
            w_report=w_report,
            w_entity=w_entity,
            warmup_frac=warmup_frac,
            cost_power_start=cost_power_start,
            gain_power=gain_power,
            fixed_cost=fixed_cost,
        )

    U = set(map(str, universe))
    U_reports = set(map(str, universe_reports)) if universe_reports is not None else None

    covered_entities_full: Set[str] = set()
    covered_reports_full: Set[str] = set()
    selected: List[Subgraph] = []
    selected_full: List[Subgraph] = []
    remaining = list(subgraphs)

    cum_cost = 0.0

    def _score_gain(sg: Subgraph) -> tuple[float, int, int]:
        new_entities = len(set(sg.covered_universe) - covered_entities_full)

        eff_reports = set(sg.covered_report_ids)
        if U_reports is not None:
            eff_reports = eff_reports & U_reports
        new_reports = len(eff_reports - covered_reports_full)

        if objective == "report":
            gain = float(new_reports)
        elif objective == "entity":
            gain = float(new_entities)
        else:
            gain = float(w_report) * float(new_reports) + float(w_entity) * float(new_entities)
        return gain, new_reports, new_entities

    def _pick_best(remaining_list: List[Subgraph], budget_left: Optional[float]) -> Optional[Subgraph]:
        best = None
        best_score = -1.0
        best_gain = -1.0
        best_new_reports = -1
        best_new_entities = -1

        t = len(selected_full)
        K = int(k_limit)
        warmup_steps = max(1, int(K * warmup_frac))
        progress = min(1.0, t / float(warmup_steps))
        q = float(cost_power_start) + (1.0 - float(cost_power_start)) * progress

        for sg in remaining_list:
            c_true = float(getattr(sg, "cost", 0.0))
            if budget_left is not None and c_true > float(budget_left) + 1e-12:
                continue

            gain, nr, ne = _score_gain(sg)
            if gain < float(min_gain):
                continue

            if unit_cost:
                score = gain
            else:
                eff_cost = max(float(c_true) + float(fixed_cost), 1e-9)
                score = (float(gain) ** float(gain_power)) / (eff_cost ** q)

            if (score > best_score) or (
                abs(score - best_score) <= 1e-12 and (
                    gain > best_gain or (
                        abs(gain - best_gain) <= 1e-12 and (
                            ne > best_new_entities or (
                                ne == best_new_entities and (
                                    nr > best_new_reports or (nr == best_new_reports and sg.subgraph_id < (best.subgraph_id if best else "~~~"))
                                )
                            )
                        )
                    )
                )
            ):
                best = sg
                best_score = score
                best_gain = gain
                best_new_reports = nr
                best_new_entities = ne
        return best

    while len(selected_full) < int(k_limit):
        remaining_budget = B - cum_cost
        budget_left = remaining_budget if remaining_budget > 1e-12 else None

        best = _pick_best(remaining, budget_left)
        if best is None and budget_left is not None:
            best = _pick_best(remaining, None)
        if best is None:
            break

        selected_full.append(best)
        c_true = float(getattr(best, "cost", 0.0))
        if budget_left is not None and c_true <= float(remaining_budget) + 1e-12:
            selected.append(best)
            cum_cost += c_true

        covered_entities_full |= set(best.covered_universe)
        eff_reports = set(best.covered_report_ids)
        if U_reports is not None:
            eff_reports = eff_reports & U_reports
        covered_reports_full |= eff_reports

        remaining = [sg for sg in remaining if sg.subgraph_id != best.subgraph_id]

        if covered_entities_full == U:
            if objective == "entity":
                break
            if U_reports is None or covered_reports_full == U_reports:
                break
        if objective in ("report", "hybrid") and U_reports is not None and covered_reports_full == U_reports:
            break

    cover = _cover_from_selected(selected, universe)
    cover.selected_full = list(selected_full)
    return cover


def selector_random_AB(
    subgraphs: List[Subgraph],
    universe: Set[str],
    *,
    budget_mode: BudgetMode,
    k_limit: int,
    B_cost_ref: Optional[float],
    seed: int,
) -> CoverResult:
    order = list(subgraphs)
    rng = random.Random(int(seed))
    rng.shuffle(order)

    if budget_mode == "count":
        selected = order[: int(k_limit)]
    else:
        selected = _feasible_subset_from_order(order, B_cost_ref=B_cost_ref, k_limit=int(k_limit), skip_infeasible=True)

    cover = _cover_from_selected(selected, universe)
    cover.selected_full = list(order[: int(k_limit)])
    return cover


def selector_topscore_AB(
    subgraphs: List[Subgraph],
    universe: Set[str],
    *,
    budget_mode: BudgetMode,
    k_limit: int,
    B_cost_ref: Optional[float],
    score_fn: Callable[[Subgraph], float],
) -> CoverResult:
    order = sorted(list(subgraphs), key=score_fn, reverse=True)

    if budget_mode == "count":
        selected = order[: int(k_limit)]
    else:
        selected = _feasible_subset_from_order(order, B_cost_ref=B_cost_ref, k_limit=int(k_limit), skip_infeasible=True)

    cover = _cover_from_selected(selected, universe)
    cover.selected_full = list(order[: int(k_limit)])
    return cover


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
    total_reports_all: int


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
    *,
    budget_cost_ref: Optional[float] = None,
) -> dict:
    report_coverage_curve: List[float] = []
    entity_coverage_curve: List[float] = []
    cum_cost_curve: List[float] = []
    rho_curve: List[float] = []

    current_reports: Set[str] = set()
    current_universe: Set[str] = set()
    cum_cost = 0.0

    selected = list(cover_result.selected[: int(k_limit)])

    B = float(budget_cost_ref) if (budget_cost_ref is not None and budget_cost_ref > 0) else None

    for sg in selected:
        rep_ids = set(getattr(sg, "covered_report_ids", getattr(sg, "report_ids", set())))
        if universe_reports_set is not None:
            rep_ids = rep_ids & set(universe_reports_set)

        current_reports.update(rep_ids)
        current_universe.update(set(getattr(sg, "covered_universe", set())))

        cum_cost += float(getattr(sg, "cost", 0.0))
        cum_cost_curve.append(cum_cost)

        r_cov = len(current_reports) / max(1, int(total_reports_count))
        e_cov = len(current_universe) / max(1, int(total_universe_count))
        report_coverage_curve.append(float(r_cov))
        entity_coverage_curve.append(float(e_cov))

        if B is None:
            rho_curve.append(cum_cost)
        else:
            rho_curve.append(min(1.0, cum_cost / B))

    def _pad(xs: List[float], fill: float) -> List[float]:
        if len(xs) >= int(k_limit):
            return xs[: int(k_limit)]
        if not xs:
            return [fill] * int(k_limit)
        return xs + [xs[-1]] * (int(k_limit) - len(xs))

    report_coverage_curve = _pad(report_coverage_curve, 0.0)
    entity_coverage_curve = _pad(entity_coverage_curve, 0.0)
    cum_cost_curve = _pad(cum_cost_curve, 0.0)
    rho_curve = _pad(rho_curve, 0.0)

    if B is None:
        B_self = float(cum_cost_curve[-1]) if cum_cost_curve else 0.0
        if B_self <= 0:
            rho_curve = [0.0] * int(k_limit)
        else:
            rho_curve = [min(1.0, float(x) / B_self) for x in rho_curve]

    def _nauc_over_rho(rhos: List[float], covs: List[float]) -> float:
        area = 0.0
        prev_r = 0.0
        prev_c = 0.0
        for r, c in zip(rhos, covs):
            r = float(max(0.0, min(1.0, r)))
            if r < prev_r:
                continue
            area += prev_c * (r - prev_r)
            prev_r = r
            prev_c = float(c)
            if prev_r >= 1.0:
                break
        if prev_r < 1.0:
            area += prev_c * (1.0 - prev_r)
        return float(area)

    naucB_report = _nauc_over_rho(rho_curve, report_coverage_curve)
    naucB_entity = _nauc_over_rho(rho_curve, entity_coverage_curve)

    def _cov_at_frac(rhos: List[float], covs: List[float], p: float) -> float:
        p = float(max(0.0, min(1.0, p)))
        best = 0.0
        for r, c in zip(rhos, covs):
            if float(r) <= p + 1e-12:
                best = float(c)
            else:
                break
        return float(best)

    covR_025 = _cov_at_frac(rho_curve, report_coverage_curve, 0.25)
    covR_050 = _cov_at_frac(rho_curve, report_coverage_curve, 0.50)
    covR_075 = _cov_at_frac(rho_curve, report_coverage_curve, 0.75)

    covE_025 = _cov_at_frac(rho_curve, entity_coverage_curve, 0.25)
    covE_050 = _cov_at_frac(rho_curve, entity_coverage_curve, 0.50)
    covE_075 = _cov_at_frac(rho_curve, entity_coverage_curve, 0.75)

    nauc_k_report = calculate_nauc(report_coverage_curve, int(k_limit))
    nauc_k_entity = calculate_nauc(entity_coverage_curve, int(k_limit))

    return {
        "report_coverage_curve": report_coverage_curve,
        "entity_coverage_curve": entity_coverage_curve,
        "sum@k_report": nauc_k_report,
        "sum@k_entity": nauc_k_entity,
        "final_report_coverage": report_coverage_curve[-1] if report_coverage_curve else 0.0,
        "final_entity_coverage": entity_coverage_curve[-1] if entity_coverage_curve else 0.0,
        "cum_cost_curve": cum_cost_curve,
        "rho_curve": rho_curve,
        "nAUC@B_report": naucB_report,
        "nAUC@B_entity": naucB_entity,
        "Cov@0.25B_report": covR_025,
        "Cov@0.50B_report": covR_050,
        "Cov@0.75B_report": covR_075,
        "Cov@0.25B_entity": covE_025,
        "Cov@0.50B_entity": covE_050,
        "Cov@0.75B_entity": covE_075,
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
    cap_methods: str = "",
    budget_mode: str = "count",
    cost_budget: float = 0.0,
) -> PipelineResult:
    bm: BudgetMode = _norm_budget_mode(budget_mode)
    K = int(budget)

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
    all_reports = set(idx.report_to_nodes.keys()) | set(idx.report_to_edges.keys())
    total_reports_all = len(all_reports)

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
                "covered": int(len(sg.covered_universe)),
            }
        )

    merged = merge_similar_subgraphs(
        raw_subgraphs,
        thresh=float(merge_sim_thresh),
        node_type_by_id=node_type_by_id,
        max_nodes_after_merge=int(max_nodes_after_merge) if max_nodes_after_merge is not None else None,
        max_reports_after_merge=int(max(1, total_reports_all * 0.32)) if total_reports_all > 0 else None,
        max_universe_after_merge=int(max(1, len(universe) * 0.32)) if len(universe) > 0 else None,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
    )

    cap_methods_set = set()
    if cap_methods:
        wanted = [m.strip() for m in cap_methods.split(",") if m.strip()]
        name_map = {
            "ECFA": "ECFA",
            "ECFA_NoCost": "ECFA w/o cost",
            "ECFA_NoMerge": "ECFA (No Merge)",
            "Random": "Random-k",
            "TopUniSize": "TopUniSize-k",
            "TopRawSize": "TopRawSize-k",
        }
        for w in wanted:
            cap_methods_set.add(name_map.get(w, w))

    merged_full = merged
    raw_full = raw_subgraphs

    merged_cap = merged_full
    raw_cap = raw_full

    cap_ratio = float(max_view_report_ratio)
    if cap_ratio < 1.0 and total_reports_all > 0:
        cap = int(total_reports_all * cap_ratio)
        merged_cap = [sg for sg in merged_full if len(sg.report_ids) <= cap]
        raw_cap = [sg for sg in raw_full if len(sg.report_ids) <= cap]

    def _pool(method_name: str, kind: str) -> List[Subgraph]:
        use_cap = (method_name in cap_methods_set)
        if kind == "merged":
            return merged_cap if use_cap else merged_full
        else:
            return raw_cap if use_cap else raw_full

    cover_topk_for_Bref = greedy_set_cover(
        _pool("ECFA", "merged"),
        universe=universe,
        universe_reports=universe_reports,
        budget=K,
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

    if float(cost_budget) > 0:
        B_cost_ref: Optional[float] = float(cost_budget)
    else:
        B_cost_ref = float(sum(float(getattr(sg, "cost", 0.0)) for sg in cover_topk_for_Bref.selected[:K]))
        if B_cost_ref <= 0:
            B_cost_ref = None

    if bm == "count":
        cover = cover_topk_for_Bref
    else:
        cover = greedy_set_cover_cost_budget(
            _pool("ECFA", "merged"),
            universe=universe,
            universe_reports=universe_reports,
            k_limit=K,
            budget_cost_ref=B_cost_ref,
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

    if bm == "count":
        cover_wo_merge = greedy_set_cover(
            _pool("ECFA (No Merge)", "raw"),
            universe=universe,
            universe_reports=universe_reports,
            budget=K,
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
    else:
        cover_wo_merge = greedy_set_cover_cost_budget(
            _pool("ECFA (No Merge)", "raw"),
            universe=universe,
            universe_reports=universe_reports,
            k_limit=K,
            budget_cost_ref=B_cost_ref,
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

    baselines.append(
        BaselineResult(
            name="ECFA",
            cover_results=[cover],
            metrics=compute_metrics(cover, total_reports, len(universe), K, universe_reports_set=universe_reports, budget_cost_ref=B_cost_ref),
        )
    )

    baselines.append(
        BaselineResult(
            name="ECFA (No Merge)",
            cover_results=[cover_wo_merge],
            metrics=compute_metrics(cover_wo_merge, total_reports, len(universe), K, universe_reports_set=universe_reports, budget_cost_ref=B_cost_ref),
        )
    )

    random_runs: List[CoverResult] = []
    pool_random = _pool("Random-k", "merged")
    for i in range(30):
        if bm == "count":
            random_runs.append(random_selection(pool_random, universe=universe, k=K, seed=int(i)))
        else:
            random_runs.append(selector_random_AB(pool_random, universe, budget_mode=bm, k_limit=K, B_cost_ref=B_cost_ref, seed=int(i)))

    random_metrics_list = [
        compute_metrics(r, total_reports, len(universe), K, universe_reports_set=universe_reports, budget_cost_ref=B_cost_ref)
        for r in random_runs
    ]

    curves_r = np.array([m.get("report_coverage_curve", [0.0] * K) for m in random_metrics_list], dtype=float)
    curves_e = np.array([m.get("entity_coverage_curve", [0.0] * K) for m in random_metrics_list], dtype=float)
    curves_rho = np.array([m.get("rho_curve", [0.0] * K) for m in random_metrics_list], dtype=float)
    curves_cum = np.array([m.get("cum_cost_curve", [0.0] * K) for m in random_metrics_list], dtype=float)

    avg_report_curve = curves_r.mean(axis=0).tolist()
    std_report_curve = curves_r.std(axis=0, ddof=0).tolist()
    avg_entity_curve = curves_e.mean(axis=0).tolist()
    std_entity_curve = curves_e.std(axis=0, ddof=0).tolist()

    avg_rho_curve = curves_rho.mean(axis=0).tolist()
    avg_cum_curve = curves_cum.mean(axis=0).tolist()

    def _mean(xs): return float(sum(xs) / len(xs)) if xs else 0.0
    def _std(xs): return float(statistics.pstdev(xs)) if xs and len(xs) > 1 else 0.0

    auc_r_list = [float(m.get("sum@k_report", 0.0)) for m in random_metrics_list]
    auc_e_list = [float(m.get("sum@k_entity", 0.0)) for m in random_metrics_list]
    fin_r_list = [float(m.get("final_report_coverage", 0.0)) for m in random_metrics_list]
    fin_e_list = [float(m.get("final_entity_coverage", 0.0)) for m in random_metrics_list]

    naucB_r_list = [float(m.get("nAUC@B_report", 0.0)) for m in random_metrics_list]
    naucB_e_list = [float(m.get("nAUC@B_entity", 0.0)) for m in random_metrics_list]

    cov025_r_list = [float(m.get("Cov@0.25B_report", 0.0)) for m in random_metrics_list]
    cov050_r_list = [float(m.get("Cov@0.50B_report", 0.0)) for m in random_metrics_list]
    cov075_r_list = [float(m.get("Cov@0.75B_report", 0.0)) for m in random_metrics_list]
    cov025_e_list = [float(m.get("Cov@0.25B_entity", 0.0)) for m in random_metrics_list]
    cov050_e_list = [float(m.get("Cov@0.50B_entity", 0.0)) for m in random_metrics_list]
    cov075_e_list = [float(m.get("Cov@0.75B_entity", 0.0)) for m in random_metrics_list]

    avg_random_metrics = {
        "report_coverage_curve": avg_report_curve,
        "entity_coverage_curve": avg_entity_curve,
        "report_coverage_std": std_report_curve,
        "entity_coverage_std": std_entity_curve,
        "rho_curve": avg_rho_curve,
        "cum_cost_curve": avg_cum_curve,

        "sum@k_report": calculate_nauc(avg_report_curve, K),
        "sum@k_entity": calculate_nauc(avg_entity_curve, K),
        "final_report_coverage": _mean(fin_r_list),
        "final_entity_coverage": _mean(fin_e_list),
        "std_sum@k_report": _std(auc_r_list),
        "std_sum@k_entity": _std(auc_e_list),
        "std_final_report_coverage": _std(fin_r_list),
        "std_final_entity_coverage": _std(fin_e_list),

        "nAUC@B_report": _mean(naucB_r_list),
        "nAUC@B_entity": _mean(naucB_e_list),
        "Cov@0.25B_report": _mean(cov025_r_list),
        "Cov@0.50B_report": _mean(cov050_r_list),
        "Cov@0.75B_report": _mean(cov075_r_list),
        "Cov@0.25B_entity": _mean(cov025_e_list),
        "Cov@0.50B_entity": _mean(cov050_e_list),
        "Cov@0.75B_entity": _mean(cov075_e_list),
    }
    baselines.append(BaselineResult(name="Random-k", cover_results=random_runs, metrics=avg_random_metrics))

    pool_top_uni = _pool("TopUniSize-k", "merged")
    if bm == "count":
        top_uni_size_res = top_k_selection(pool_top_uni, universe=universe, k=K, score_fn=lambda sg: len(sg.covered_universe))
    else:
        top_uni_size_res = selector_topscore_AB(pool_top_uni, universe, budget_mode=bm, k_limit=K, B_cost_ref=B_cost_ref, score_fn=lambda sg: len(sg.covered_universe))

    baselines.append(
        BaselineResult(
            name="TopUniSize-k",
            cover_results=[top_uni_size_res],
            metrics=compute_metrics(top_uni_size_res, total_reports, len(universe), K, universe_reports_set=universe_reports, budget_cost_ref=B_cost_ref),
        )
    )

    pool_top_raw = _pool("TopRawSize-k", "raw")
    if bm == "count":
        top_raw_size_res = top_k_selection(pool_top_raw, universe=universe, k=K, score_fn=lambda sg: (len(sg.node_ids) + len(sg.edge_ids)))
    else:
        top_raw_size_res = selector_topscore_AB(pool_top_raw, universe, budget_mode=bm, k_limit=K, B_cost_ref=B_cost_ref, score_fn=lambda sg: (len(sg.node_ids) + len(sg.edge_ids)))

    baselines.append(
        BaselineResult(
            name="TopRawSize-k",
            cover_results=[top_raw_size_res],
            metrics=compute_metrics(top_raw_size_res, total_reports, len(universe), K, universe_reports_set=universe_reports, budget_cost_ref=B_cost_ref),
        )
    )

    res = PipelineResult(
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
        total_reports_all=total_reports_all,
    )

    res.B_cost_ref = B_cost_ref
    res.budget_mode = bm
    res.cover_topk_for_Bref = cover_topk_for_Bref
    return res


def _safe_sheet_name(name: str) -> str:
    bad = set("[]:*?/\\")
    name = "".join([c if c not in bad else "_" for c in str(name)])
    name = name.strip() or "sheet"
    return name[:31]


def _count_reports_from_source_row_index(x) -> int:
    return len(split_report_ids(x))


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    arr = np.array([float(x) for x in xs], dtype=float)
    return float(np.percentile(arr, float(p)))


def _median_iqr(xs: List[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    arr = np.array([float(x) for x in xs], dtype=float)
    med = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return med, float(q3 - q1)


def _top1_share(costs: List[float]) -> float:
    if not costs:
        return 0.0
    tot = float(sum(float(c) for c in costs))
    if tot <= 0:
        return 0.0
    return float(max(float(c) for c in costs) / tot)


def _hhi(costs: List[float]) -> float:
    if not costs:
        return 0.0
    tot = float(sum(float(c) for c in costs))
    if tot <= 0:
        return 0.0
    shares = [float(c) / tot for c in costs]
    return float(sum(s * s for s in shares))


def portfolio_metrics_from_coverresult(cr: "CoverResult") -> Dict[str, float]:
    selected = list(getattr(cr, "selected", []) or [])
    costs = [float(getattr(sg, "cost", 0.0)) for sg in selected]
    gammas = [float(len(getattr(sg, "covered_report_ids", getattr(sg, "report_ids", []) ) or [])) for sg in selected]

    nSel = int(len(selected))
    total_cost = float(sum(costs)) if costs else 0.0
    avg_cost = float(total_cost / nSel) if nSel > 0 else 0.0

    med_c, iqr_c = _median_iqr(costs)
    med_g, iqr_g = _median_iqr(gammas)

    return {
        "nSel": float(nSel),
        "TotalCost": float(total_cost),
        "AvgCostPerView": float(avg_cost),
        "Top1CostShare": float(_top1_share(costs)),
        "HHI_Cost": float(_hhi(costs)),
        "MedianCost": float(med_c),
        "IQR_Cost": float(iqr_c),
        "P95Cost": float(_percentile(costs, 95)),
        "MedianGammaEff": float(med_g),
        "IQR_GammaEff": float(iqr_g),
        "P95GammaEff": float(_percentile(gammas, 95)),
    }


def build_excel_sheets(
    *,
    nodes_df: pd.DataFrame,
    idx: GraphIndex,
    universe: set[str],
    universe_reports: set[str],
    total_reports_all: int,
    merged: list[Subgraph],
    raw_subgraphs: list[Subgraph],
    cover: CoverResult,
    baselines: List[Any],
    seed_stats: list[dict],
    skipped: list[dict],
    original_node_to_reports: dict[str, set[str]],
    edge_only_nodes: set[str],
    n_candidates_raw: int = 0,
    maybe_truncated_cnt: int = 0,
    n_seeds_total: int = 0,
    eps_overlap: float = 0.1,
    run_config: Optional[Dict[str, Any]] = None,
    budget_cost_ref: Optional[float] = None,
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
            for j in range(i + 1, len(sets)):
                v = _jaccard(set(sets[i]), set(sets[j]))
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
    for sg in merged:
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

    seed_stats_cols = ["seed", "reports", "maybe_truncated", "stopped_reason", "hop_k", "nodes", "edges", "covered"]
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
            "nAUC@K (Report)": b.metrics.get("sum@k_report", 0.0),
            "nAUC@K (Entity)": b.metrics.get("sum@k_entity", 0.0),
            "Coverage@K (Report)": b.metrics.get("final_report_coverage", 0.0),
            "Coverage@K (Entity)": b.metrics.get("final_entity_coverage", 0.0),
        }
        row.update({
            "nAUC@B (Report)": b.metrics.get("nAUC@B_report", 0.0),
            "nAUC@B (Entity)": b.metrics.get("nAUC@B_entity", 0.0),
            "Cov@0.25B (Report)": b.metrics.get("Cov@0.25B_report", 0.0),
            "Cov@0.50B (Report)": b.metrics.get("Cov@0.50B_report", 0.0),
            "Cov@0.75B (Report)": b.metrics.get("Cov@0.75B_report", 0.0),
            "Cov@0.25B (Entity)": b.metrics.get("Cov@0.25B_entity", 0.0),
            "Cov@0.50B (Entity)": b.metrics.get("Cov@0.50B_entity", 0.0),
            "Cov@0.75B (Entity)": b.metrics.get("Cov@0.75B_entity", 0.0),
        })
        if b.name == "Random-k":
            row["Std nAUC@K (Report)"] = b.metrics.get("std_sum@k_report", 0.0)
            row["Std nAUC@K (Entity)"] = b.metrics.get("std_sum@k_entity", 0.0)
            row["Std Coverage@K (Report)"] = b.metrics.get("std_final_report_coverage", 0.0)
            row["Std Coverage@K (Entity)"] = b.metrics.get("std_final_entity_coverage", 0.0)
        metrics_rows.append(row)
    df_metrics = pd.DataFrame(metrics_rows)


    df_random_runs_metrics = pd.DataFrame()
    try:
        K_budget = int(run_config.get("budget", 50)) if run_config else 50
        if K_budget <= 0:
            K_budget = 50
    except Exception:
        K_budget = 50

    random_rows = []
    random_baseline = None
    for b in baselines or []:
        if str(b.name) == "Random-k":
            random_baseline = b
            break

    if random_baseline is not None:
        for rid, cr in enumerate(getattr(random_baseline, "cover_results", []) or []):
            m = compute_metrics(
                cover_result=cr,
                total_reports_count=int(len(universe_reports)),
                total_universe_count=int(len(universe)),
                k_limit=int(K_budget),
                universe_reports_set=universe_reports,
                budget_cost_ref=budget_cost_ref,
            )
            sel = list(getattr(cr, "selected", []) or [])
            random_rows.append({
                "run_id": int(rid),
                "nSel": int(len(sel)),
                "TotalCost": float(sum(float(getattr(sg, "cost", 0.0)) for sg in sel)),
                "Coverage@K (Report)": float(m.get("final_report_coverage", 0.0)),
                "Coverage@K (Entity)": float(m.get("final_entity_coverage", 0.0)),
                "nAUC@B (Report)": float(m.get("nAUC@B_report", 0.0)),
                "nAUC@B (Entity)": float(m.get("nAUC@B_entity", 0.0)),
                "Cov@0.25B (Report)": float(m.get("Cov@0.25B_report", 0.0)),
                "Cov@0.50B (Report)": float(m.get("Cov@0.50B_report", 0.0)),
                "Cov@0.75B (Report)": float(m.get("Cov@0.75B_report", 0.0)),
                "Cov@0.25B (Entity)": float(m.get("Cov@0.25B_entity", 0.0)),
                "Cov@0.50B (Entity)": float(m.get("Cov@0.50B_entity", 0.0)),
                "Cov@0.75B (Entity)": float(m.get("Cov@0.75B_entity", 0.0)),
            })
    df_random_runs_metrics = pd.DataFrame(random_rows)


    max_k = 0
    curve_data = {}
    std_data = {}
    for b in baselines:
        curve = b.metrics.get("report_coverage_curve", [])
        if curve:
            curve_data[b.name] = curve
            max_k = max(max_k, len(curve))
        if b.name == "Random-k" and b.metrics.get("report_coverage_std"):
            std_data[b.name] = b.metrics["report_coverage_std"]

    df_curves = pd.DataFrame({"k": range(1, max_k + 1)})
    for name, curve in curve_data.items():
        padded = curve + ([curve[-1]] * (max_k - len(curve)) if curve else [0.0] * (max_k - len(curve)))
        df_curves[name] = padded
    if "Random-k" in std_data:
        std = std_data["Random-k"]
        std = std + ([std[-1]] * (max_k - len(std)) if std else [0.0] * (max_k - len(std)))
        df_curves["Random-k_std"] = std

    cost_rows = []
    total_reports_count = int(len(universe_reports))
    total_universe_count = int(len(universe))
    budget_ref = float(budget_cost_ref) if (budget_cost_ref is not None and float(budget_cost_ref) > 0) else None

    def _curves_from_selected_list(selected_list: List[Subgraph]) -> tuple[list[float], list[float], list[float], list[float]]:
        current_reports: Set[str] = set()
        current_universe: Set[str] = set()
        cum_cost = 0.0
        cum_cost_curve: list[float] = []
        rho_curve: list[float] = []
        cov_report: list[float] = []
        cov_entity: list[float] = []

        for sg in selected_list:
            rep_ids = set(getattr(sg, "covered_report_ids", getattr(sg, "report_ids", set())))
            rep_ids = rep_ids & set(universe_reports)
            current_reports.update(rep_ids)
            current_universe.update(set(getattr(sg, "covered_universe", set())))

            cum_cost += float(getattr(sg, "cost", 0.0))
            cum_cost_curve.append(cum_cost)

            cov_report.append(len(current_reports) / max(1, total_reports_count))
            cov_entity.append(len(current_universe) / max(1, total_universe_count))

        if budget_ref is not None:
            rho_curve = [min(1.0, float(x) / float(budget_ref)) for x in cum_cost_curve]
        else:
            total_cost = float(cum_cost_curve[-1]) if cum_cost_curve else 0.0
            if total_cost <= 0:
                rho_curve = [0.0] * len(cum_cost_curve)
            else:
                rho_curve = [min(1.0, float(x) / total_cost) for x in rho_curve]

        return rho_curve, cum_cost_curve, cov_report, cov_entity

    def _pad_to(xs: list[float], n: int) -> list[float]:
        if len(xs) >= n:
            return xs[:n]
        if not xs:
            return [0.0] * n
        return xs + [xs[-1]] * (n - len(xs))

    for b in baselines:
        if str(b.name) == "Random-k" and getattr(b, "cover_results", None):
            curves_r: list[list[float]] = []
            curves_e: list[list[float]] = []
            curves_rho: list[list[float]] = []
            curves_cum: list[list[float]] = []
            max_len = 0
            for cr in getattr(b, "cover_results", []) or []:
                selected_list = list(getattr(cr, "selected_full", None) or getattr(cr, "selected", []) or [])
                rhos, cumc, covr, cove = _curves_from_selected_list(selected_list)
                max_len = max(max_len, len(rhos), len(cumc), len(covr), len(cove))
                curves_r.append(covr)
                curves_e.append(cove)
                curves_rho.append(rhos)
                curves_cum.append(cumc)
            if max_len <= 0:
                continue
            arr_r = np.array([_pad_to(x, max_len) for x in curves_r], dtype=float)
            arr_e = np.array([_pad_to(x, max_len) for x in curves_e], dtype=float)
            arr_rho = np.array([_pad_to(x, max_len) for x in curves_rho], dtype=float)
            arr_cum = np.array([_pad_to(x, max_len) for x in curves_cum], dtype=float)

            avg_r = arr_r.mean(axis=0).tolist()
            std_r = arr_r.std(axis=0, ddof=0).tolist()
            avg_e = arr_e.mean(axis=0).tolist()
            std_e = arr_e.std(axis=0, ddof=0).tolist()
            avg_rho = arr_rho.mean(axis=0).tolist()
            avg_cum = arr_cum.mean(axis=0).tolist()

            for i in range(max_len):
                cost_rows.append(
                    {
                        "Method": str(b.name),
                        "step": int(i + 1),
                        "cum_cost": float(avg_cum[i]),
                        "rho": float(avg_rho[i]),
                        "cov_report": float(avg_r[i]),
                        "cov_entity": float(avg_e[i]),
                        "cov_report_std": float(std_r[i]),
                        "cov_entity_std": float(std_e[i]),
                    }
                )
        else:
            cr0 = (getattr(b, "cover_results", []) or [None])[0]
            selected_list = list(getattr(cr0, "selected_full", None) or getattr(cr0, "selected", []) or [])
            rhos, cumc, covr, cove = _curves_from_selected_list(selected_list)
            k_len = max(len(rhos), len(cumc), len(covr), len(cove))
            for i in range(k_len):
                cost_rows.append(
                    {
                        "Method": str(b.name),
                        "step": int(i + 1),
                        "cum_cost": float(cumc[i]) if i < len(cumc) else float(cumc[-1] if cumc else 0.0),
                        "rho": float(rhos[i]) if i < len(rhos) else float(rhos[-1] if rhos else 0.0),
                        "cov_report": float(covr[i]) if i < len(covr) else float(covr[-1] if covr else 0.0),
                        "cov_entity": float(cove[i]) if i < len(cove) else float(cove[-1] if cove else 0.0),
                    }
                )
    df_cost_curve_long = pd.DataFrame(cost_rows)


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
            k_run = min(int(max_k), len(selected_sgs)) if max_k else len(selected_sgs)
            k_used = max(k_used, k_run)

            report_sets = []
            for sg in selected_sgs[:k_run]:
                reps = set(getattr(sg, "covered_report_ids", getattr(sg, "report_ids", set())))
                reps = reps & set(universe_reports)
                if reps:
                    report_sets.append(reps)

            entity_sets = [set(sg.covered_universe) for sg in selected_sgs[:k_run] if sg.covered_universe]

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

        is_random = str(b.name) == "Random-k"

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


    def _collect_selected_views_stats(method_name: str, cr: CoverResult, run_id: int = 0) -> List[dict]:
        rows = []
        for rank, sg in enumerate(list(getattr(cr, "selected", []) or []), start=1):
            rows.append({
                "Method": str(method_name),
                "run_id": int(run_id),
                "rank": int(rank),
                "subgraph_id": str(sg.subgraph_id),
                "seed_node": str(sg.seed_node),
                "|Gamma_eff|": int(len(getattr(sg, "covered_report_ids", getattr(sg, "report_ids", set())) or [])),
                "|Gamma_all|": int(len(getattr(sg, "report_ids", set()) or [])),
                "|V|": int(len(getattr(sg, "node_ids", set()) or [])),
                "|E|": int(len(getattr(sg, "edge_ids", set()) or [])),
                "|CoveredUniverse|": int(len(getattr(sg, "covered_universe", set()) or [])),
                "cost": float(getattr(sg, "cost", 0.0)),
            })
        return rows

    sv_rows = []
    for b in baselines or []:
        if str(b.name) == "Random-k":
            for rid, cr in enumerate(getattr(b, "cover_results", []) or []):
                sv_rows.extend(_collect_selected_views_stats(b.name, cr, run_id=rid))
        else:
            for cr in getattr(b, "cover_results", []) or []:
                sv_rows.extend(_collect_selected_views_stats(b.name, cr, run_id=0))
    df_selected_views_stats = pd.DataFrame(sv_rows)


    portfolio_rows = []

    def _mean_std2(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        if len(xs) == 1:
            return float(xs[0]), 0.0
        return float(statistics.mean(xs)), float(statistics.pstdev(xs))

    for b in baselines or []:
        per_run = []
        for cr in getattr(b, "cover_results", []) or []:
            per_run.append(portfolio_metrics_from_coverresult(cr))

        def _col(name: str) -> list[float]:
            return [float(r.get(name, 0.0)) for r in per_run]

        row = {"Method": str(b.name)}
        for key in ["nSel", "TotalCost", "AvgCostPerView",
                    "Top1CostShare", "HHI_Cost",
                    "MedianCost", "IQR_Cost", "P95Cost",
                    "MedianGammaEff", "IQR_GammaEff", "P95GammaEff"]:
            vals = _col(key)
            m, s = _mean_std2(vals)
            row[key] = m
            if str(b.name) == "Random-k":
                row[f"Std_{key}"] = s
        portfolio_rows.append(row)

    df_portfolio = pd.DataFrame(portfolio_rows)


    df_candidates_by_method = pd.DataFrame()

    def _cap_methods_set_from_run_config(rc: Optional[Dict[str, Any]]) -> set[str]:
        if not rc:
            return set()
        cap_methods = str(rc.get("cap_methods", "") or "").strip()
        if not cap_methods:
            return set()
        wanted = [m.strip() for m in cap_methods.split(",") if m.strip()]
        name_map = {
            "ECFA": "ECFA",
            "ECFA_NoCost": "ECFA w/o cost",
            "ECFA_NoMerge": "ECFA (No Merge)",
            "Random": "Random-k",
            "TopUniSize": "TopUniSize-k",
            "TopRawSize": "TopRawSize-k",
        }
        return set(name_map.get(w, w) for w in wanted)

    cap_methods_set = _cap_methods_set_from_run_config(run_config)

    cap_ratio = 1.0
    if run_config and str(run_config.get("max_view_report_ratio", "")).strip():
        try:
            cap_ratio = float(run_config.get("max_view_report_ratio"))
        except Exception:
            cap_ratio = 1.0

    cap_threshold = None
    if cap_ratio < 1.0 and int(total_reports_all) > 0:
        cap_threshold = int(int(total_reports_all) * float(cap_ratio))
        if cap_threshold <= 0:
            cap_threshold = None

    MERGED_POOL_METHODS = {"ECFA", "ECFA w/o cost", "Random-k", "TopUniSize-k"}
    RAW_POOL_METHODS = {"ECFA (No Merge)", "TopRawSize-k"}

    present_methods = [str(b.name) for b in (baselines or [])]
    cand_rows = []
    cand_map = {}

    for m in present_methods:
        if m in RAW_POOL_METHODS:
            base_pool = list(raw_subgraphs)
            pool_kind = "raw"
        else:
            base_pool = list(merged)
            pool_kind = "merged"

        use_cap = (m in cap_methods_set)
        if use_cap and cap_threshold is not None:
            used_pool = [sg for sg in base_pool if int(len(getattr(sg, "report_ids", []) or [])) <= int(cap_threshold)]
            source = "cap"
        else:
            used_pool = base_pool
            source = "full"

        n_used = int(len(used_pool))
        cand_map[m] = n_used

        cand_rows.append({
            "Method": str(m),
            "PoolKind": str(pool_kind),
            "PoolSource": str(source),
            "CapUsed": bool(use_cap),
            "CapThreshold(|Gamma_all|<=)": int(cap_threshold) if cap_threshold is not None else "",
            "#Candidates": int(n_used),
            "#Candidates_full": int(len(base_pool)),
            "#Candidates_cap": int(len([sg for sg in base_pool if (cap_threshold is not None and int(len(getattr(sg, "report_ids", []) or [])) <= int(cap_threshold))])) if cap_threshold is not None else "",
        })

    df_candidates_by_method = pd.DataFrame(cand_rows)

    if "Method" in df_metrics.columns:
        df_metrics["#Candidates"] = df_metrics["Method"].map(cand_map)
    if "Method" in df_portfolio.columns:
        df_portfolio["#Candidates"] = df_portfolio["Method"].map(cand_map)


    merged_rep_counts = [len(getattr(sg, "covered_report_ids", [])) for sg in merged]
    merged_rep_counts.sort()
    n_merged = len(merged_rep_counts)
    max_cov_rep = merged_rep_counts[-1] if n_merged else 0
    p95_cov_rep = merged_rep_counts[int(n_merged * 0.95)] if n_merged else 0

    df_meta = pd.DataFrame(
        [
            {"key": "edge_only_nodes", "value": int(len(edge_only_nodes))},
            {"key": "index_backfilled_nodes", "value": int(backfilled_nodes)},
            {"key": "index_backfilled_report_links", "value": int(backfilled_pairs)},
            {"key": "universe_nodes_with_empty_source_row_index", "value": int(empty_src)},
            {"key": "n_reports_all", "value": int(total_reports_all)},
            {"key": "n_reports_universe", "value": int(len(universe_reports))},
            {"key": "n_candidates_raw", "value": int(n_candidates_raw)},
            {"key": "n_candidates_merged", "value": int(n_merged)},
            {"key": "truncated_seed_ratio", "value": float((maybe_truncated_cnt / n_seeds_total) if n_seeds_total else 0.0)},
            {"key": "max_covered_reports_candidate", "value": int(max_cov_rep)},
            {"key": "p95_covered_reports_candidate", "value": int(p95_cov_rep)},
            {"key": "eps_overlap", "value": float(eps_overlap)},
        ]
    )


    rc_rows = []
    if run_config:
        for k, v in sorted(run_config.items(), key=lambda kv: str(kv[0])):
            if isinstance(v, (list, tuple, set)):
                vv = ",".join(map(str, v))
            elif isinstance(v, dict):
                vv = str(v)
            else:
                vv = "" if v is None else str(v)
            rc_rows.append({"key": str(k), "value": vv})
    df_run_config = pd.DataFrame(rc_rows, columns=["key", "value"])

    sheets = {
        "Metrics": df_metrics,
        "CoverageCurve": df_curves,
        "CostCurve_Long": df_cost_curve_long,
        "Portfolio": df_portfolio,
        "Candidates_ByMethod": df_candidates_by_method,
        "RandomRuns_Metrics": df_random_runs_metrics,
        "Redundancy_Report": df_redundancy_report,
        "Redundancy_Entity": df_redundancy_entity,
        "SelectedViews_Stats_AllMethods": df_selected_views_stats,
        "SeedStats": df_seed_stats,
        "SkippedSeeds": df_skipped,
        "UncoveredNodes": uncovered_df,
        "meta": df_meta,
        "RunConfig": df_run_config,
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


def plot_coverage_curves(baselines: List[Any], out_base: Path, *, metric: Literal["report", "entity"] = "report"):
    curve_kind = "report" if metric == "report" else "entity"
    curve_key = "report_coverage_curve" if curve_kind == "report" else "entity_coverage_curve"
    valid = [b for b in (baselines or []) if b.metrics.get(curve_key)]
    if not valid:
        return None
    max_k = max(len(b.metrics[curve_key]) for b in valid)
    xs = list(range(1, max_k + 1))
    plt.figure(figsize=(8, 5), dpi=140)
    styles_map = {
        "ECFA": {"color": "red", "marker": "o", "linewidth": 2, "label": "ECFA (Full)"},
        "ECFA w/o cost": {"color": "orange", "marker": "x", "linewidth": 2, "label": "ECFA (No Cost)"},
        "ECFA (No Merge)": {"color": "purple", "marker": "s", "linewidth": 2, "label": "ECFA (No Merge)"},
        "Random-k": {"color": "gray", "linestyle": "--", "label": "Random-k"},
        "TopUniSize-k": {"color": "blue", "linestyle": "-.", "label": "TopUniSize-k"},
        "TopRawSize-k": {"color": "green", "linestyle": ":", "label": "TopRawSize-k"},
    }
    for b in valid:
        name = b.name
        curve = list(b.metrics[curve_key])
        if len(curve) < max_k:
            curve += [curve[-1]] * (max_k - len(curve))
        style = styles_map.get(name, {"label": name})
        if name == "Random-k" and curve_kind == "report" and b.metrics.get("report_coverage_std"):
            std = list(b.metrics["report_coverage_std"])
            if len(std) < max_k:
                std += [std[-1]] * (max_k - len(std))
            means = np.array(curve, dtype=float)
            stds = np.array(std, dtype=float)
            plt.plot(xs, curve, label=style["label"], color=style.get("color"), linestyle=style.get("linestyle"))
            plt.fill_between(xs, means - stds, means + stds, color=style.get("color", "gray"), alpha=0.2)
        else:
            plt.plot(
                xs, curve,
                label=style.get("label", name),
                color=style.get("color"),
                linestyle=style.get("linestyle"),
                marker=style.get("marker"),
                linewidth=style.get("linewidth", 1.5),
            )
    ylabel = "Report Coverage" if curve_kind == "report" else "Entity Coverage"
    title = "Report Coverage @ k" if curve_kind == "report" else "Entity Coverage @ k"
    plt.xlabel("Budget (k)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"))
    plt.savefig(out_base.with_suffix(".svg"))
    plt.close()
    print(f"[Plot] Saved {curve_kind} curve to: {out_base.with_suffix('.png')} and {out_base.with_suffix('.svg')}")
    return out_base.with_suffix(".png"), out_base.with_suffix(".svg")


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

    ap = argparse.ArgumentParser(description="Batch generate RQ2 outputs per APP (xlsx + png) with baselines and ablations.")
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
    ap.add_argument("--budget_mode", choices=["count", "cost"], default="cost",
                    help="cost: selection constrained by B_cost_ref; count: Top-K selection (protocol-A).")
    ap.add_argument("--cost_budget", type=float, default=0.0,
                    help="B_cost_ref. If <=0, inferred per-app from ECFA Top-K total cost.")

    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--eps_overlap", type=float, default=0.1)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=0.01)

    ap.add_argument("--unit_cost", action="store_true", default=False)
    ap.add_argument("--no_unit_cost", action="store_false", dest="unit_cost")

    ap.add_argument("--objective", choices=["report", "entity", "hybrid"], default="report")
    ap.add_argument("--w_report", type=float, default=1.0)
    ap.add_argument("--w_entity", type=float, default=1.0)

    ap.add_argument("--backfill_node_reports_from_edges", action="store_true", default=False)

    ap.add_argument("--max_view_report_ratio", type=float, default=1.0,
                    help="Hard cap: disallow any view whose |Gamma_eff| exceeds ratio * |universe_reports|")

    ap.add_argument("--cap_methods", type=str, default="", help="Comma-separated list of methods to include (e.g. ECFA,ECFA_NoCost)")

    ap.add_argument("--warmup_frac", type=float, default=0.25)
    ap.add_argument("--cost_power_start", type=float, default=1.0)
    ap.add_argument("--gain_power", type=float, default=1.0)
    ap.add_argument("--fixed_cost", type=float, default=0.0)

    ap.add_argument("--run_tag", type=str, default="")
    ap.add_argument("--plot_curves", choices=["auto", "report", "entity", "both"], default="auto")

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

    print(f"[Start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[InputRoot] {input_root.as_posix()}")
    print(f"[Dataset] {dataset_xlsx.as_posix()}")
    print(f"[OutputRoot] {output_root.as_posix()}")
    print(f"[Apps] {len(apps)}")
    print(f"[Objective] {args.objective}  unit_cost={args.unit_cost}")
    print(f"[ProvenanceBackfill] {args.backfill_node_reports_from_edges}")

    ok = 0
    skipped = 0
    failed = 0

    for app in apps:
        app_dir = input_root / app
        unify_dir = app_dir / "unify"
        nodes_path = unify_dir / "nodes.xlsx"
        edges_path = unify_dir / "edges.xlsx"

        run_tag = str(args.run_tag).strip()
        if not run_tag:
            run_tag = f"obj-{args.objective}_B{args.budget}_k{args.k}_rmax{args.r_max}_m{args.merge_sim_thresh}_capR{args.max_view_report_ratio}"
            run_tag = run_tag.replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_dir = output_root / app
        out_xlsx = out_dir / f"rq2_report_{run_tag}.xlsx"

        if args.skip_existing and out_xlsx.exists():
            print(f"[Skip] {app} exists: {out_xlsx.as_posix()}")
            skipped += 1
            continue

        if not nodes_path.exists() or not edges_path.exists():
            print(f"[Fail] {app} missing unify files: {nodes_path.exists()=} {edges_path.exists()=}")
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
                unit_cost=bool(args.unit_cost),
                objective=str(args.objective),
                w_report=float(args.w_report),
                w_entity=float(args.w_entity),
                backfill_node_reports_from_edges=bool(args.backfill_node_reports_from_edges),
                warmup_frac=float(args.warmup_frac),
                cost_power_start=float(args.cost_power_start),
                gain_power=float(args.gain_power),
                fixed_cost=float(args.fixed_cost),
                max_view_report_ratio=float(args.max_view_report_ratio),
                cap_methods=str(args.cap_methods),
                budget_mode=str(args.budget_mode),
                cost_budget=float(args.cost_budget),
            )


            if not args.unit_cost:
                print("Running ECFA w/o cost (unit_cost=True)...")

                bm = getattr(res, "budget_mode", _norm_budget_mode(args.budget_mode))
                B_cost_ref = getattr(res, "B_cost_ref", None)

                use_cap = False
                if args.cap_methods:
                    wanted = [m.strip() for m in str(args.cap_methods).split(",") if m.strip()]
                    name_map = {
                        "ECFA": "ECFA",
                        "ECFA_NoCost": "ECFA w/o cost",
                        "ECFA_NoMerge": "ECFA (No Merge)",
                        "Random": "Random-k",
                        "TopUniSize": "TopUniSize-k",
                        "TopRawSize": "TopRawSize-k",
                    }
                    cap_methods_set = set(name_map.get(w, w) for w in wanted)
                    use_cap = ("ECFA w/o cost" in cap_methods_set)

                pool_wo_cost = res.merged_subgraphs
                if use_cap:
                    cap_ratio = float(args.max_view_report_ratio)
                    if cap_ratio < 1.0 and getattr(res, "total_reports_all", 0) > 0:
                        cap = int(int(getattr(res, "total_reports_all", 0)) * cap_ratio)
                        pool_wo_cost = [sg for sg in res.merged_subgraphs if len(sg.report_ids) <= cap]

                if bm == "count":
                    cover_wo_cost = greedy_set_cover(
                        subgraphs=pool_wo_cost,
                        universe=res.universe,
                        universe_reports=res.universe_reports,
                        budget=int(args.budget),
                        min_gain=1.0,
                        unit_cost=True,
                        objective=str(args.objective),
                        w_report=float(args.w_report),
                        w_entity=float(args.w_entity),
                        warmup_frac=float(args.warmup_frac),
                        cost_power_start=float(args.cost_power_start),
                        gain_power=float(args.gain_power),
                        fixed_cost=float(args.fixed_cost),
                    )
                else:
                    cover_wo_cost = greedy_set_cover_cost_budget(
                        subgraphs=pool_wo_cost,
                        universe=res.universe,
                        universe_reports=res.universe_reports,
                        k_limit=int(args.budget),
                        budget_cost_ref=B_cost_ref,
                        min_gain=1.0,
                        unit_cost=True,
                        objective=str(args.objective),
                        w_report=float(args.w_report),
                        w_entity=float(args.w_entity),
                        warmup_frac=float(args.warmup_frac),
                        cost_power_start=float(args.cost_power_start),
                        gain_power=float(args.gain_power),
                        fixed_cost=float(args.fixed_cost),
                    )

                metrics_wo_cost = compute_metrics(
                    cover_result=cover_wo_cost,
                    total_reports_count=res.total_reports,
                    total_universe_count=len(res.universe),
                    k_limit=int(args.budget),
                    universe_reports_set=res.universe_reports,
                    budget_cost_ref=B_cost_ref,
                )

                res.baselines.insert(1, BaselineResult(
                    name="ECFA w/o cost",
                    cover_results=[cover_wo_cost],
                    metrics=metrics_wo_cost
                ))

            sheets = build_excel_sheets(
                nodes_df=res.nodes_df,
                idx=res.idx,
                universe=res.universe,
                universe_reports=res.universe_reports,
                total_reports_all=int(getattr(res, "total_reports_all", 0)),
                merged=res.merged_subgraphs,
                raw_subgraphs=res.raw_subgraphs,
                cover=res.cover,
                baselines=res.baselines,
                seed_stats=res.seed_stats,
                skipped=res.skipped,
                original_node_to_reports=res.original_node_to_reports,
                edge_only_nodes=res.edge_only_nodes,
                n_candidates_raw=len(res.raw_subgraphs),
                maybe_truncated_cnt=res.maybe_truncated_cnt,
                n_seeds_total=len(res.seeds),
                eps_overlap=float(args.eps_overlap),
                run_config=dict(args.__dict__),
                budget_cost_ref=getattr(res, "B_cost_ref", None),
            )
            write_excel_report(out_xlsx, sheets)

            pc = str(args.plot_curves)
            if pc == "auto":
                if str(args.objective) == "entity":
                    kinds = ["entity"]
                elif str(args.objective) == "hybrid":
                    kinds = ["report", "entity"]
                else:
                    kinds = ["report"]
            elif pc == "both":
                kinds = ["report", "entity"]
            else:
                kinds = [pc]
            for ck in kinds:
                plot_coverage_curves(res.baselines, out_dir / f"coverage_curve_{run_tag}_{ck}", metric=("report" if ck == "report" else "entity"))

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
