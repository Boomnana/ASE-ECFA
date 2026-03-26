from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


REQUIRED_NODE_COLS = {"id", "name", "type", "source_row_index"}
REQUIRED_EDGE_COLS = {"id", "source_id", "target_id", "relation", "source_type", "target_type", "source_row_index"}


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
        t = t.strip()
        if not t:
            continue
        if t in seen:
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


def build_graph_index(nodes: pd.DataFrame, edges: pd.DataFrame, defects: Optional[pd.DataFrame] = None) -> GraphIndex:
    node_to_reports: Dict[str, Set[str]] = {}
    report_to_nodes: Dict[str, Set[str]] = {}
    for row in nodes.itertuples(index=False):
        nid = str(getattr(row, "id"))
        rids = list(getattr(row, "report_ids"))
        node_to_reports[nid] = set(rids)
        for rid in rids:
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

    def _report_sort_key(x: str):
        xs = str(x)
        return (0, int(xs)) if xs.isdigit() else (1, xs)

    report_depth: dict[str, int] = {str(rid): 0 for rid in R}

    def _nodes_from_reports(report_ids: Set[str]) -> Set[str]:
        out = {seed_node}
        for rid in report_ids:
            out |= set(idx.report_to_nodes.get(rid, set()))
        return out

    def _truncate_reports(report_ids: Set[str]) -> Set[str]:
        if r_max <= 0:
            return set(report_ids)
        if len(report_ids) <= r_max:
            return set(report_ids)
        ranked = sorted(
            map(str, report_ids),
            key=lambda rid: (int(report_depth.get(str(rid), 10**9)), _report_sort_key(str(rid))),
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
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.01,
) -> Subgraph:
    node_ids, edge_ids = induce_subgraph_from_reports(idx, report_ids)
    covered = set(node_ids) & set(universe_nodes)
    cost = compute_cost(report_ids, node_ids, edge_ids, alpha=alpha, beta=beta, gamma=gamma)
    return Subgraph(
        subgraph_id=subgraph_id,
        seed_node=str(seed_node),
        report_ids=set(report_ids),
        node_ids=set(node_ids),
        edge_ids=set(edge_ids),
        covered_universe=covered,
        cost=float(cost),
    )


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def weighted_jaccard(parts_a: Dict[str, Set[str]], parts_b: Dict[str, Set[str]], weights: Dict[str, float]) -> float:
    score = 0.0
    wsum = 0.0
    for k, w in weights.items():
        if w <= 0:
            continue
        score += w * jaccard(parts_a.get(k, set()), parts_b.get(k, set()))
        wsum += w
    if wsum <= 0:
        raise ValueError("weights must have at least one positive entry")
    return score / wsum


def _signature_sets(
    node_ids: Iterable[str],
    node_type_by_id: Dict[str, str],
    type_alias: Dict[str, str],
) -> Dict[str, Set[str]]:
    sig: Dict[str, Set[str]] = {
        "ISSUE": set(),
        "PHEN": set(),
        "DIAG": set(),
        "OP": set(),
        "ELEM": set(),
        "MOD": set(),
    }
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
        def _report_sort_key(sg: Subgraph) -> tuple:
            return (
                len(sg.report_ids),
                len(sg.covered_universe),
                len(sg.node_ids),
                str(sg.subgraph_id),
                str(sg.seed_node),
            )

        remaining = sorted(subgraphs, key=_report_sort_key, reverse=True)
        merged: List[Subgraph] = []
        used = [False] * len(remaining)
        for i, sg in enumerate(remaining):
            if used[i]:
                continue
            rep_union = set(sg.report_ids)
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
                        if max_nodes_after_merge is not None:
                            if len(node_union | set(sg2.node_ids)) > max_nodes_after_merge:
                                continue
                        used[j] = True
                        rep_union |= sg2.report_ids
                        node_union |= sg2.node_ids
                        edge_union |= sg2.edge_ids
                        covered_union |= sg2.covered_universe
                        seeds.append(sg2.seed_node)
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

    def _signature_sort_key(sg_sig: tuple) -> tuple:
        sg, sig = sg_sig
        issue_n = len(sig.get("ISSUE", set()))
        phen_diag_n = len(sig.get("PHEN", set())) + len(sig.get("DIAG", set()))
        return (
            issue_n,
            phen_diag_n,
            len(sg.report_ids),
            len(sg.covered_universe),
            len(sg.node_ids),
            str(sg.subgraph_id),
            str(sg.seed_node),
        )

    paired = sorted(zip(remaining, sigs), key=_signature_sort_key, reverse=True)
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

                issue_j = jaccard(sig_union["ISSUE"], sig2["ISSUE"])
                phen_diag_union = sig_union["PHEN"] | sig_union["DIAG"]
                phen_diag_2 = sig2["PHEN"] | sig2["DIAG"]
                phen_diag_j = jaccard(phen_diag_union, phen_diag_2)

                if issue_j <= 1e-12 and phen_diag_j < min_phen_diag_jaccard_when_no_issue:
                    continue
                if issue_j <= 1e-12 and min_report_jaccard_when_no_issue > 0:
                    rep_j = jaccard(rep_union, sg2.report_ids)
                    if rep_j < min_report_jaccard_when_no_issue:
                        continue
                if issue_j > 1e-12 and issue_j < min_issue_jaccard and phen_diag_j < 0.35:
                    continue

                sim = weighted_jaccard(sig_union, sig2, weights)
                if sim < thresh:
                    continue

                if max_nodes_after_merge is not None:
                    if len(node_union | set(sg2.node_ids)) > max_nodes_after_merge:
                        continue

                used[j] = True
                rep_union |= set(sg2.report_ids)
                node_union |= set(sg2.node_ids)
                edge_union |= set(sg2.edge_ids)
                covered_union |= set(sg2.covered_universe)
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


def greedy_set_cover(
    subgraphs: List[Subgraph],
    universe: Set[str],
    budget: int = 50,
    min_gain: int = 1,
) -> CoverResult:
    U = set(universe)
    covered: Set[str] = set()
    selected: List[Subgraph] = []
    remaining = subgraphs[:]

    while len(selected) < budget:
        best = None
        best_score = -1.0
        for sg in remaining:
            gain_set = sg.covered_universe - covered
            gain = len(gain_set)
            if gain < min_gain:
                continue
            score = gain / max(sg.cost, 1e-9)
            if score > best_score:
                best = sg
                best_score = score
        if best is None:
            break
        selected.append(best)
        covered |= best.covered_universe
        remaining = [sg for sg in remaining if sg.subgraph_id != best.subgraph_id]
        if covered == U:
            break

    uncovered = U - covered
    return CoverResult(selected=selected, covered=covered, uncovered=uncovered)


def stable_id_key(x: str) -> Tuple[int, int, str]:
    s = str(x)
    if s.isdigit():
        return (0, int(s), "")
    return (1, 0, s)


def _node_label(idx: GraphIndex, nid: str) -> str:
    r = idx.node_id_to_row.get(str(nid), {}) or {}
    name = (r.get("name") or "").strip()
    return name if name else str(nid)


def _node_type(idx: GraphIndex, nid: str) -> str:
    r = idx.node_id_to_row.get(str(nid), {}) or {}
    return str(r.get("type", "") or "")


def _edge_relation(idx: GraphIndex, eid: str) -> str:
    r = idx.edge_id_to_row.get(str(eid), {}) or {}
    return str(r.get("relation", "") or "")


def _edge_uv(idx: GraphIndex, eid: str) -> Tuple[str, str]:
    r = idx.edge_id_to_row.get(str(eid), {}) or {}
    return str(r.get("source_id", "") or ""), str(r.get("target_id", "") or "")


def _build_subgraph_adjacency(idx: GraphIndex, sg: Subgraph):
    node_set = set(map(str, sg.node_ids))
    adj = defaultdict(list)
    radj = defaultdict(list)

    for eid in sorted(map(str, sg.edge_ids), key=stable_id_key):
        u, v = _edge_uv(idx, eid)
        if not u or not v:
            continue
        if u not in node_set or v not in node_set:
            continue
        adj[u].append((v, eid))
        radj[v].append((u, eid))

    for u in list(adj.keys()):
        adj[u].sort(key=lambda t: (stable_id_key(t[0]), stable_id_key(t[1])))
    for v in list(radj.keys()):
        radj[v].sort(key=lambda t: (stable_id_key(t[0]), stable_id_key(t[1])))

    return adj, radj


def _pick_start_nodes(idx: GraphIndex, sg: Subgraph, *, only_core: bool = True) -> List[str]:
    core_types = {"问题陈述", "用户感知现象", "系统诊断信息", "用户操作"}
    nodes = sorted(list(map(str, sg.node_ids)), key=stable_id_key)
    if not only_core:
        return nodes
    core = [nid for nid in nodes if _node_type(idx, nid) in core_types]
    return core if core else nodes


def _path_score(idx: GraphIndex, node_path: List[str], eid_path: List[str]) -> Tuple[int, int, int]:
    preferred_types = {"问题陈述", "用户感知现象", "系统诊断信息", "用户操作"}
    pref_nodes = sum(1 for nid in node_path if _node_type(idx, nid) in preferred_types)
    return (len(eid_path), pref_nodes, len(node_path))


def _is_subsegment_nodes(small_nodes: List[str], big_nodes: List[str]) -> bool:
    n = len(small_nodes)
    m = len(big_nodes)
    if n > m:
        return False
    for i in range(m - n + 1):
        if big_nodes[i : i + n] == small_nodes:
            return True
    return False


def _canonical_path_key(node_path: List[str], eid_path: List[str]) -> Tuple:
    return (
        tuple(map(str, node_path)),
        tuple(map(str, eid_path)),
    )


def _extract_top_paths(
    idx: GraphIndex,
    sg: Subgraph,
    *,
    max_len: int = 4,
    top_k: int = 30,
    per_start_cap: int = 80,
    only_core_starts: bool = True,
    require_core_in_path: bool = True,
) -> List[List[Tuple[str, str, str]]]:
    adj, _ = _build_subgraph_adjacency(idx, sg)
    starts = _pick_start_nodes(idx, sg, only_core=only_core_starts)

    core_types = {"问题陈述", "用户感知现象", "系统诊断信息", "用户操作"}

    candidates: List[Tuple[List[str], List[str]]] = []

    for s in starts:
        q = deque()
        q.append((s, [s], []))
        expanded = 0

        while q and expanded < per_start_cap:
            u, node_path, eid_path = q.popleft()
            expanded += 1

            if eid_path:
                if require_core_in_path:
                    if any(_node_type(idx, nid) in core_types for nid in node_path):
                        candidates.append((node_path, eid_path))
                else:
                    candidates.append((node_path, eid_path))

            if len(eid_path) >= max_len:
                continue

            for v, eid in adj.get(u, []):
                if v in node_path:
                    continue
                q.append((v, node_path + [v], eid_path + [eid]))

    candidates.sort(
        key=lambda x: (
            _path_score(idx, x[0], x[1]),
            _canonical_path_key(x[0], x[1]),
        ),
        reverse=True,
    )

    kept: List[Tuple[List[str], List[str]]] = []
    for np, ep in candidates:
        redundant = False
        for knp, _ in kept:
            if _is_subsegment_nodes(np, knp):
                redundant = True
                break
        if not redundant:
            kept.append((np, ep))
            if len(kept) >= top_k:
                break

    out: List[List[Tuple[str, str, str]]] = []
    for np, ep in kept:
        triples: List[Tuple[str, str, str]] = []
        for i, eid in enumerate(ep):
            u = np[i]
            v = np[i + 1]
            rel = _edge_relation(idx, eid) or ""
            triples.append((u, rel, v))
        if triples:
            out.append(triples)

    return out


def _group_star_paths(idx: GraphIndex, paths_triples: List[List[Tuple[str, str, str]]]):
    len1, len_gt1 = [], []
    for t in paths_triples:
        if t:
            (len1 if len(t) == 1 else len_gt1).append(t)

    star_map = defaultdict(set)
    for t in len1:
        u, rel, v = t[0]
        star_map[(u, rel)].add(v)

    star_groups = []
    singles = []
    for (u, rel), vs in star_map.items():
        vs = sorted(list(vs), key=stable_id_key)
        if len(vs) > 1:
            star_groups.append((u, rel, vs))
        else:
            singles.append((u, rel, vs[0]))

    rev_map = defaultdict(set)
    for u, rel, v in singles:
        rev_map[(v, rel)].add(u)

    reverse_star_groups = []
    for (v, rel), us in rev_map.items():
        us = sorted(list(us), key=stable_id_key)
        if len(us) > 1:
            reverse_star_groups.append((us, rel, v))
        else:
            star_groups.append((us[0], rel, [v]))

    chain_map = defaultdict(set)
    for chain in len_gt1:
        prefix = tuple(chain[:-1])
        last_u, last_rel, last_v = chain[-1]
        key = (prefix, last_u, last_rel)
        chain_map[key].add(last_v)

    chain_star_groups = []
    simple_chains = []

    for (prefix, last_u, last_rel), vs in chain_map.items():
        vs = sorted(list(vs), key=stable_id_key)
        if len(vs) > 1:
            chain_star_groups.append((list(prefix), last_u, last_rel, vs))
        else:
            full = list(prefix) + [(last_u, last_rel, vs[0])]
            simple_chains.append(full)

    star_groups.sort(key=lambda x: (len(x[2]), stable_id_key(x[0])), reverse=True)
    reverse_star_groups.sort(key=lambda x: (len(x[0]), stable_id_key(x[2])), reverse=True)
    chain_star_groups.sort(key=lambda x: len(x[3]), reverse=True)
    simple_chains.sort(
        key=lambda ch: (
            _path_score(idx, [ch[0][0]] + [e[2] for e in ch], [e[1] for e in ch]),
            _canonical_path_key([ch[0][0]] + [e[2] for e in ch], [e[1] for e in ch]),
        ),
        reverse=True,
    )

    return star_groups, reverse_star_groups, chain_star_groups, simple_chains


def _format_star(idx: GraphIndex, u: str, rel: str, vs: List[str]) -> str:
    src = _node_label(idx, u)
    targets = ", ".join([_node_label(idx, v) for v in vs])
    return f"{src} --({rel})--> [{targets}]"


def _format_reverse_star(idx: GraphIndex, us: List[str], rel: str, v: str) -> str:
    srcs = ", ".join([_node_label(idx, u) for u in us])
    target = _node_label(idx, v)
    return f"[{srcs}] --({rel})--> {target}"


def _format_chain(idx: GraphIndex, triples: List[Tuple[str, str, str]]) -> str:
    if not triples:
        return ""
    u0, rel0, v0 = triples[0]
    s = f"{_node_label(idx, u0)} --({rel0})--> {_node_label(idx, v0)}"
    for (_, rel, v) in triples[1:]:
        s += f" --({rel})--> {_node_label(idx, v)}"
    return s


def _format_chain_star(
    idx: GraphIndex, prefix: List[Tuple[str, str, str]], last_u: str, last_rel: str, vs: List[str]
) -> str:
    base = _format_chain(idx, prefix) if prefix else _node_label(idx, last_u)
    targets = ", ".join([_node_label(idx, v) for v in vs])
    return f"{base} --({last_rel})--> [{targets}]"


def build_paths_descriptions(
    idx: GraphIndex,
    sg: Subgraph,
    *,
    max_len: int = 4,
    top_k: int = 30,
    max_output: int = 20,
    per_start_cap: int = 80,
    only_core_starts: bool = True,
) -> List[str]:
    raw = _extract_top_paths(
        idx,
        sg,
        max_len=max_len,
        top_k=top_k,
        per_start_cap=per_start_cap,
        only_core_starts=only_core_starts,
        require_core_in_path=True,
    )

    star_groups, reverse_star_groups, chain_star_groups, simple_chains = _group_star_paths(idx, raw)

    out: List[str] = []
    seen = set()

    for u, rel, vs in star_groups:
        if len(out) >= max_output:
            break
        desc = _format_star(idx, u, rel, vs)
        if desc not in seen:
            out.append(desc)
            seen.add(desc)

    for us, rel, v in reverse_star_groups:
        if len(out) >= max_output:
            break
        desc = _format_reverse_star(idx, us, rel, v)
        if desc not in seen:
            out.append(desc)
            seen.add(desc)

    for prefix, last_u, last_rel, vs in chain_star_groups:
        if len(out) >= max_output:
            break
        desc = _format_chain_star(idx, prefix, last_u, last_rel, vs)
        if desc not in seen:
            out.append(desc)
            seen.add(desc)

    for triples in simple_chains:
        if len(out) >= max_output:
            break
        desc = _format_chain(idx, triples)
        if desc not in seen:
            out.append(desc)
            seen.add(desc)

    if not out:
        src_rel_to_vs = defaultdict(list)
        for eid in sorted(map(str, sg.edge_ids), key=stable_id_key):
            u, v = _edge_uv(idx, eid)
            if not u or not v:
                continue
            rel = _edge_relation(idx, eid)
            src_rel_to_vs[(u, rel)].append(v)

        items = []
        for (u, rel), vs in src_rel_to_vs.items():
            dedup = []
            seen_v = set()
            for v in sorted(map(str, vs), key=stable_id_key):
                if v not in seen_v:
                    seen_v.add(v)
                    dedup.append(v)
            items.append((u, rel, dedup))

        items.sort(key=lambda x: (len(x[2]), stable_id_key(x[0]), x[1]), reverse=True)
        for u, rel, vs in items[:max_output]:
            out.append(_format_star(idx, u, rel, vs))

    return out


def _node_payload(idx: GraphIndex, nid: str) -> dict:
    r = idx.node_id_to_row.get(str(nid), {})
    return {
        "id": str(nid),
        "name": r.get("name", ""),
        "type": r.get("type", ""),
        "source_row_index": r.get("source_row_index", ""),
    }


def _edge_payload(idx: GraphIndex, eid: str) -> dict:
    r = idx.edge_id_to_row.get(str(eid), {})
    return {
        "id": str(eid),
        "source_id": r.get("source_id", ""),
        "target_id": r.get("target_id", ""),
        "relation": r.get("relation", ""),
        "source_type": r.get("source_type", ""),
        "target_type": r.get("target_type", ""),
        "source_row_index": r.get("source_row_index", ""),
    }


def export_selected_subgraphs_brief_dir(
    idx: GraphIndex,
    selected: List[Subgraph],
    out_dir: str | Path,
    paths_max_len: int = 4,
    paths_top_k: int = 30,
    paths_max_output: int = 20,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sg in selected:
        reports = sorted(list(map(str, sg.report_ids)))
        defects = []
        if idx.defect_id_to_text is not None:
            for rid in reports:
                defects.append({"id": str(rid), "description": idx.defect_id_to_text.get(str(rid), "")})
        else:
            for rid in reports:
                defects.append({"id": str(rid), "description": ""})

        path_descs = build_paths_descriptions(
            idx,
            sg,
            max_len=paths_max_len,
            top_k=paths_top_k,
            max_output=paths_max_output,
            per_start_cap=80,
            only_core_starts=True,
        )
        paths = [{"id": i + 1, "description": d} for i, d in enumerate(path_descs)]

        obj = {
            "subgraph_id": sg.subgraph_id,
            "seed_node": str(sg.seed_node),
            "defects": defects,
            "paths": paths,
        }

        p = out_dir / f"{sg.subgraph_id}.brief.json"
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

        full_obj = {
            "subgraph_id": sg.subgraph_id,
            "seed_node": str(sg.seed_node),
            "reports": reports,
            "nodes": [_node_payload(idx, nid) for nid in sorted(sg.node_ids, key=stable_id_key)],
            "edges": [_edge_payload(idx, eid) for eid in sorted(sg.edge_ids, key=stable_id_key)],
            "cost": sg.cost,
            "covered_universe": sorted(list(sg.covered_universe), key=stable_id_key),
            "defects": defects,
        }
        p_full = out_dir / f"{sg.subgraph_id}.json"
        p_full.write_text(json.dumps(full_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_dir


def export_selected_subgraphs_jsonl(
    idx: GraphIndex,
    selected: List[Subgraph],
    out_path: str | Path,
    include_defects: bool = True,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for sg in selected:
            reports = sorted(list(map(str, sg.report_ids)), key=stable_id_key)
            obj = {
                "subgraph_id": sg.subgraph_id,
                "seed_node": sg.seed_node,
                "reports": reports,
                "nodes": [_node_payload(idx, nid) for nid in sorted(sg.node_ids, key=stable_id_key)],
                "edges": [_edge_payload(idx, eid) for eid in sorted(sg.edge_ids, key=stable_id_key)],
                "cost": sg.cost,
                "covered_universe": sorted(list(sg.covered_universe), key=stable_id_key),
            }
            if include_defects and idx.defect_id_to_text is not None:
                obj["defects"] = [{"id": rid, "description": idx.defect_id_to_text.get(rid, "")} for rid in reports]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out_path


def export_selected_subgraphs_dir(
    idx: GraphIndex,
    selected: List[Subgraph],
    out_dir: str | Path,
    include_defects: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sg in selected:
        p = out_dir / f"{sg.subgraph_id}.json"
        reports = sorted(list(map(str, sg.report_ids)), key=stable_id_key)
        obj = {
            "subgraph_id": sg.subgraph_id,
            "seed_node": sg.seed_node,
            "reports": reports,
            "nodes": [_node_payload(idx, nid) for nid in sorted(sg.node_ids, key=stable_id_key)],
            "edges": [_edge_payload(idx, eid) for eid in sorted(sg.edge_ids, key=stable_id_key)],
            "cost": sg.cost,
            "covered_universe": sorted(list(sg.covered_universe), key=stable_id_key),
        }
        if include_defects and idx.defect_id_to_text is not None:
            obj["defects"] = [{"id": rid, "description": idx.defect_id_to_text.get(rid, "")} for rid in reports]
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_dir


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
    merged: list[Subgraph],
    cover: CoverResult,
    seed_stats: list[dict],
    skipped: list[dict],
    original_node_to_reports: dict[str, set[str]],
    edge_only_nodes: set[str],
) -> Dict[str, pd.DataFrame]:
    id_str = nodes_df["id"].astype(str)
    covered_ids = set(map(str, cover.covered))
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
        sort_cols = ["ever_in_any_candidate", "n_reports_in_source"]
        uncovered_df.sort_values(sort_cols, ascending=[True, False], inplace=True)

    sel_cov = set(map(str, cover.covered))
    unreachable_ids = sorted(list(set(map(str, universe)) - cand_cov))
    missed_ids = sorted(list(cand_cov - sel_cov))

    df_unreachable = nodes_df.loc[id_str.isin(unreachable_ids)].copy()
    if not df_unreachable.empty:
        df_unreachable["id"] = df_unreachable["id"].astype(str)
        if "source_row_index" not in df_unreachable.columns:
            df_unreachable["source_row_index"] = ""
        df_unreachable["n_reports_in_source"] = df_unreachable["source_row_index"].apply(_count_reports_from_source_row_index)

    df_missed = nodes_df.loc[id_str.isin(missed_ids)].copy()
    if not df_missed.empty:
        df_missed["id"] = df_missed["id"].astype(str)
        if "source_row_index" not in df_missed.columns:
            df_missed["source_row_index"] = ""
        df_missed["n_reports_in_source"] = df_missed["source_row_index"].apply(_count_reports_from_source_row_index)

    node_to_candidates = {}
    for sg in merged:
        sg_cost = float(sg.cost)
        sg_reports = int(len(sg.report_ids))
        sg_nodes = int(len(sg.node_ids))
        sg_edges = int(len(sg.edge_ids))
        sg_cov = int(len(sg.covered_universe))
        for nid in sg.covered_universe:
            nid = str(nid)
            node_to_candidates.setdefault(nid, []).append(
                {
                    "subgraph_id": sg.subgraph_id,
                    "cost": sg_cost,
                    "n_reports": sg_reports,
                    "n_nodes": sg_nodes,
                    "n_edges": sg_edges,
                    "n_covered": sg_cov,
                }
            )
    for nid, lst in node_to_candidates.items():
        lst.sort(key=lambda r: (r["cost"], r["subgraph_id"]))

    if not df_missed.empty:
        df_missed["candidate_count"] = df_missed["id"].apply(lambda x: len(node_to_candidates.get(str(x), [])))
        df_missed["best_candidate"] = df_missed["id"].apply(
            lambda x: (node_to_candidates.get(str(x), [{}])[0].get("subgraph_id") if node_to_candidates.get(str(x)) else "")
        )
        df_missed["best_candidate_cost"] = df_missed["id"].apply(
            lambda x: (node_to_candidates.get(str(x), [{}])[0].get("cost") if node_to_candidates.get(str(x)) else "")
        )
        df_missed["top_candidates"] = df_missed["id"].apply(
            lambda x: "|".join([r["subgraph_id"] for r in node_to_candidates.get(str(x), [])[:5]])
        )

    df_selected = pd.DataFrame(
        [
            {
                "subgraph_id": sg.subgraph_id,
                "seed_node": sg.seed_node,
                "cost": float(sg.cost),
                "n_reports": int(len(sg.report_ids)),
                "n_nodes": int(len(sg.node_ids)),
                "n_edges": int(len(sg.edge_ids)),
                "n_covered_universe": int(len(sg.covered_universe)),
            }
            for sg in cover.selected
        ]
    ).sort_values(["cost", "subgraph_id"], ascending=[True, True])

    node_rows = []
    edge_rows = []
    report_rows = []
    defect_rows = []
    for sg in cover.selected:
        for rid in sorted(sg.report_ids):
            report_rows.append({"subgraph_id": sg.subgraph_id, "report_id": str(rid)})
            if idx.defect_id_to_text is not None:
                defect_rows.append({"report_id": str(rid), "description": idx.defect_id_to_text.get(str(rid), "")})
        for nid in sorted(sg.node_ids):
            r = idx.node_id_to_row.get(str(nid), {})
            node_rows.append(
                {
                    "subgraph_id": sg.subgraph_id,
                    "id": str(nid),
                    "name": r.get("name", ""),
                    "type": r.get("type", ""),
                    "source_row_index": r.get("source_row_index", ""),
                }
            )
        for eid in sorted(sg.edge_ids):
            r = idx.edge_id_to_row.get(str(eid), {})
            edge_rows.append(
                {
                    "subgraph_id": sg.subgraph_id,
                    "id": str(eid),
                    "source_id": r.get("source_id", ""),
                    "target_id": r.get("target_id", ""),
                    "relation": r.get("relation", ""),
                    "source_type": r.get("source_type", ""),
                    "target_type": r.get("target_type", ""),
                    "source_row_index": r.get("source_row_index", ""),
                }
            )

    df_sel_nodes = pd.DataFrame(node_rows)
    df_sel_edges = pd.DataFrame(edge_rows)
    df_sel_reports = pd.DataFrame(report_rows)
    df_sel_defects = (
        pd.DataFrame(defect_rows).drop_duplicates(subset=["report_id"])
        if defect_rows
        else pd.DataFrame(columns=["report_id", "description"])
    )

    df_type_universe = (
        nodes_df.loc[id_str.isin(universe), ["type"]].value_counts().reset_index(name="count")
        if "type" in nodes_df.columns
        else pd.DataFrame()
    )
    df_type_covered = (
        nodes_df.loc[id_str.isin(covered_ids), ["type"]].value_counts().reset_index(name="count")
        if "type" in nodes_df.columns
        else pd.DataFrame()
    )
    df_type_uncovered = (
        nodes_df.loc[id_str.isin(uncovered_ids), ["type"]].value_counts().reset_index(name="count")
        if "type" in nodes_df.columns
        else pd.DataFrame()
    )

    df_meta = pd.DataFrame(
        [
            {"key": "edge_only_nodes", "value": int(len(edge_only_nodes))},
            {"key": "index_backfilled_nodes", "value": int(backfilled_nodes)},
            {"key": "index_backfilled_report_links", "value": int(backfilled_pairs)},
            {"key": "universe_nodes_with_empty_source_row_index", "value": int(empty_src)},
        ]
    )

    sheets = {
        "selected": df_selected,
        "sel_reports": df_sel_reports,
        "sel_defects": df_sel_defects,
        "sel_nodes": df_sel_nodes,
        "sel_edges": df_sel_edges,
        "seed_stats": df_seed_stats,
        "skipped_seeds": df_skipped,
        "uncovered_nodes": uncovered_df,
        "unreachable_nodes": df_unreachable,
        "missed_nodes": df_missed,
        "type_universe": df_type_universe,
        "type_covered": df_type_covered,
        "type_uncovered": df_type_uncovered,
        "meta": df_meta,
    }
    return sheets


def write_excel_report(out_path: str | Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as xw:
        for name, df in sheets.items():
            if df is None:
                continue
            if not isinstance(df, pd.DataFrame):
                continue
            df.to_excel(xw, sheet_name=_safe_sheet_name(name), index=False)
    return out_path


DEFAULT_UNIVERSE_TYPES = {"功能模块", "影响元素", "用户感知现象", "用户操作", "系统诊断信息", "问题陈述"}


def build_universe(nodes_df: pd.DataFrame, types: Set[str]) -> Set[str]:
    return set(nodes_df.loc[nodes_df["type"].isin(types), "id"].astype(str).tolist())


def pick_seeds(nodes_df: pd.DataFrame, seed_types: Set[str]) -> List[str]:
    return nodes_df.loc[nodes_df["type"].isin(seed_types), "id"].astype(str).tolist()


@dataclass
class PipelineResult:
    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    defects_df: Optional[pd.DataFrame]
    idx: GraphIndex
    universe: Set[str]
    seeds: List[str]
    raw_subgraphs: List[Subgraph]
    merged_subgraphs: List[Subgraph]
    cover: CoverResult
    seed_stats: List[dict]
    skipped: List[dict]
    report_counts: List[int]
    maybe_truncated_cnt: int
    node_type_by_id: dict[str, str]
    edge_only_nodes: Set[str]
    original_node_to_reports: dict[str, set[str]]


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
    merge_jaccard: float = 0.6,
    budget: int = 100,
    max_nodes_after_merge: int = 800,
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.01,
) -> PipelineResult:
    idx = build_graph_index(nodes_df, edges_df, defects_df)
    node_type_by_id = {str(r["id"]): str(r["type"]) for _, r in nodes_df.iterrows()}
    nodes_id_set = set(nodes_df["id"].astype(str).tolist())
    endpoint_nodes = set(edges_df["source_id"].astype(str).tolist()) | set(edges_df["target_id"].astype(str).tolist())
    edge_only_nodes = endpoint_nodes - nodes_id_set
    original_node_to_reports = {str(r.id): set(r.report_ids) for r in nodes_df.itertuples(index=False)}

    if seed_types is None:
        seed_types = set(DEFAULT_UNIVERSE_TYPES)
    if universe_types is None:
        universe_types = {"用户感知现象", "系统诊断信息", "问题陈述"}

    universe = build_universe(nodes_df, universe_types)
    seeds = pick_seeds(nodes_df, seed_types)

    raw_subgraphs: List[Subgraph] = []
    seed_stats: List[dict] = []
    skipped: List[dict] = []
    report_counts: List[int] = []
    maybe_truncated_cnt = 0

    for i, seed in enumerate(seeds):
        ex = expand_reports_from_seed(
            idx,
            seed_node=seed,
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
            seed_node=seed,
            report_ids=ex.report_ids,
            universe_nodes=universe,
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
        thresh=float(merge_jaccard),
        node_type_by_id=node_type_by_id,
        max_nodes_after_merge=int(max_nodes_after_merge) if max_nodes_after_merge is not None else None,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
    )

    cover = greedy_set_cover(merged, universe=universe, budget=int(budget), min_gain=1)

    return PipelineResult(
        nodes_df=nodes_df,
        edges_df=edges_df,
        defects_df=defects_df,
        idx=idx,
        universe=universe,
        seeds=seeds,
        raw_subgraphs=raw_subgraphs,
        merged_subgraphs=merged,
        cover=cover,
        seed_stats=seed_stats,
        skipped=skipped,
        report_counts=report_counts,
        maybe_truncated_cnt=int(maybe_truncated_cnt),
        node_type_by_id=node_type_by_id,
        edge_only_nodes=edge_only_nodes,
        original_node_to_reports=original_node_to_reports,
    )


def _default_repo_root_from_this_file() -> Path:
    base_dir = Path(__file__).resolve().parent
    return base_dir.parents[1]


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = _default_repo_root_from_this_file()
    shared_output = repo_root / "shared_output"
    input_dir = repo_root / "input"

    defaults = {
        "nodes_path": str(shared_output / "nodes.xlsx"),
        "edges": str(shared_output / "edges.xlsx"),
        "defects": str(input_dir / "测试雅思.xlsx"),
        "k": 2,
        "r_max": 100,
        "strict_r_max": True,
        "merge_jaccard": 0.6,
        "budget": 100,
        "check_cost": False,
        "out": r"src\defect_subgraph_cover\subgraph_cover_report.xlsx",
        "brief_dir": r"src\defect_subgraph_cover\brief_json_test",
        "paths_max_len": 4,
        "paths_top_k": 30,
        "paths_max_output": 30,
        "seed_types": "功能模块, 影响元素, 用户感知现象,用户操作, 系统诊断信息, 问题陈述",
        "universe_types": "用户感知现象, 系统诊断信息, 问题陈述",
        "alpha": 1.0,
        "beta": 0.02,
        "gamma": 0.01,
    }

    if argv is None:
        argv = sys.argv[1:]

    print(f"[Start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ap = argparse.ArgumentParser(description="Report-induced subgraph cover (single-file).")
    ap.add_argument("nodes_path", nargs="?", default=argparse.SUPPRESS, help=f"Path to nodes.xlsx (default: {defaults['nodes_path']})")
    ap.add_argument("--nodes", dest="nodes_path", default=argparse.SUPPRESS, help=f"Path to nodes.xlsx (default: {defaults['nodes_path']})")
    ap.add_argument("--edges", default=argparse.SUPPRESS, help=f"Path to edges.xlsx (default: {defaults['edges']})")
    ap.add_argument("--defects", default=argparse.SUPPRESS, help=f"Optional defects file (default: {defaults['defects']})")
    ap.add_argument("--k", type=int, default=argparse.SUPPRESS, help=f"Hop rounds for expansion (default: {defaults['k']})")
    ap.add_argument("--r_max", type=int, default=argparse.SUPPRESS, help=f"Max reports during expansion per seed (default: {defaults['r_max']})")
    ap.add_argument("--soft_r_max", action="store_true", default=argparse.SUPPRESS, help="Use soft r_max (stop when reached, without truncation)")
    ap.add_argument("--check_cost", action="store_true", default=argparse.SUPPRESS, help="Validate cost consistency after merge")
    ap.add_argument("--merge_jaccard", type=float, default=argparse.SUPPRESS, help=f"Threshold for merging subgraphs (default: {defaults['merge_jaccard']})")
    ap.add_argument("--budget", type=int, default=argparse.SUPPRESS, help=f"Max selected subgraphs for set cover (default: {defaults['budget']})")
    ap.add_argument("--out", default=argparse.SUPPRESS, help=f"Output excel path (.xlsx). If you pass .jsonl, it will be normalized to .xlsx (default: {defaults['out']})")
    ap.add_argument("--brief_dir", default=argparse.SUPPRESS, help="Optional: output per-subgraph brief jsons into this directory (default: empty)")
    ap.add_argument("--paths_max_len", type=int, default=argparse.SUPPRESS, help=f"Max edges per path for brief export (default: {defaults['paths_max_len']})")
    ap.add_argument("--paths_top_k", type=int, default=argparse.SUPPRESS, help=f"TopK raw paths before compression (default: {defaults['paths_top_k']})")
    ap.add_argument("--paths_max_output", type=int, default=argparse.SUPPRESS, help=f"Max path lines in brief output (default: {defaults['paths_max_output']})")
    ap.add_argument("--seed_types", default=argparse.SUPPRESS, help=f"Comma-separated seed node types (default: {defaults['seed_types']})")
    ap.add_argument("--universe_types", default=argparse.SUPPRESS, help=f"Comma-separated universe node types (default: {defaults['universe_types']})")
    ap.add_argument("--alpha", type=float, default=argparse.SUPPRESS, help=f"Cost weight for #reports (default: {defaults['alpha']})")
    ap.add_argument("--beta", type=float, default=argparse.SUPPRESS, help=f"Cost weight for #nodes (default: {defaults['beta']})")
    ap.add_argument("--gamma", type=float, default=argparse.SUPPRESS, help=f"Cost weight for #edges (default: {defaults['gamma']})")

    args = ap.parse_args(argv)
    provided = vars(args)
    cfg = dict(defaults)
    cfg.update(provided)
    if cfg.get("soft_r_max"):
        cfg["strict_r_max"] = False

    nodes_path = cfg["nodes_path"]
    edges_path = cfg["edges"]
    defects_path = cfg["defects"]

    out_path_input = Path(str(cfg["out"])).resolve()
    out_path = out_path_input
    if out_path.suffix.lower() != ".xlsx":
        out_path = out_path.with_suffix(".xlsx")

    provided_keys = sorted(list(provided.keys()))
    defaulted_keys = sorted([k for k in defaults.keys() if k not in provided])
    if provided_keys:
        print("[Args] provided:")
        for k in provided_keys:
            print(f"  - {k}={cfg.get(k)}")
    if defaulted_keys:
        print("[Args] defaulted:")
        for k in defaulted_keys:
            print(f"  - {k}={cfg.get(k)}")
    if out_path != out_path_input:
        print(f"[Args] out normalized: {out_path_input.as_posix()} -> {out_path.as_posix()}")

    print("Loading data...")
    nodes_df = load_nodes_xlsx(nodes_path)
    edges_df = load_edges_xlsx(edges_path)
    defects_df = load_defects_optional(defects_path) if defects_path else None

    seed_types = {s.strip() for s in str(cfg["seed_types"]).split(",") if s.strip()}
    universe_types = {s.strip() for s in str(cfg["universe_types"]).split(",") if s.strip()}

    print("Running pipeline...")
    res = run_subgraph_cover(
        nodes_df=nodes_df,
        edges_df=edges_df,
        defects_df=defects_df,
        seed_types=seed_types,
        universe_types=universe_types,
        k=int(cfg["k"]),
        r_max=int(cfg["r_max"]),
        strict_r_max=bool(cfg["strict_r_max"]),
        merge_jaccard=float(cfg["merge_jaccard"]),
        budget=int(cfg["budget"]),
        alpha=float(cfg["alpha"]),
        beta=float(cfg["beta"]),
        gamma=float(cfg["gamma"]),
    )

    if res.report_counts:
        s = pd.Series(res.report_counts)
        p50 = float(s.quantile(0.5))
        p90 = float(s.quantile(0.9))
        mx = int(s.max())
        print(
            f"[Expand] seeds={len(res.seeds)} built={len(res.raw_subgraphs)} skipped={len(res.skipped)} ({(100.0*len(res.skipped)/max(1,len(res.seeds))):.1f}%)"
        )
        print(f"[Expand] report_ids per seed: p50={p50:.0f} p90={p90:.0f} max={mx} r_max={int(cfg['r_max'])}")
        print(
            f"[Expand] maybe_truncated_by_rmax={res.maybe_truncated_cnt} ({(100.0*res.maybe_truncated_cnt/max(1,len(res.seeds))):.1f}%)"
        )
        print(f"[Expand] r_max_mode={'strict' if bool(cfg['strict_r_max']) else 'soft'}")

    if res.skipped:
        reason_counts = Counter([r.get("reason") for r in res.skipped])
        reason_rows = [{"reason": k, "count": int(v)} for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], str(x[0])))]
        print("[Expand] skipped reasons:", ", ".join([f"{r['reason']}={r['count']}" for r in reason_rows[:12]]))

    print(f"Seeds: {len(res.seeds)}")
    print(f"Raw subgraphs: {len(res.raw_subgraphs)}")
    print(f"Merged subgraphs: {len(res.merged_subgraphs)}")
    print(f"Selected: {len(res.cover.selected)}")
    print(f"Universe size: {len(res.universe)} Covered: {len(res.cover.covered)} Uncovered: {len(res.cover.uncovered)}")
    print(f"[IndexClosure] edge_only_nodes={len(res.edge_only_nodes)}")

    backfilled_pairs = 0
    backfilled_nodes = 0
    for nid, orig in res.original_node_to_reports.items():
        after = set(res.idx.node_to_reports.get(str(nid), set()))
        inc = after - set(orig)
        if inc:
            backfilled_nodes += 1
            backfilled_pairs += len(inc)
    print(f"[IndexClosure] backfilled_nodes={backfilled_nodes} backfilled_report_links={backfilled_pairs}")

    if len(res.cover.uncovered) > 0:
        print("Uncovered node ids (first 50):", list(sorted(res.cover.uncovered))[:50])

    id_str = nodes_df["id"].astype(str)
    if "type" in nodes_df.columns:
        u_types = nodes_df.loc[id_str.isin(res.universe), "type"].value_counts()
        print("[Universe] by type:\n" + u_types.to_string())

    covered_ids = set(map(str, res.cover.covered))
    uncovered_ids = set(map(str, res.cover.uncovered))
    if "type" in nodes_df.columns:
        cov_types = nodes_df.loc[id_str.isin(covered_ids), "type"].value_counts()
        uncov_types = nodes_df.loc[id_str.isin(uncovered_ids), "type"].value_counts()
        print("[Covered] by type:\n" + cov_types.to_string())
        print("[Uncovered] by type:\n" + uncov_types.to_string())

    cand_cov = set()
    for sg in res.merged_subgraphs:
        cand_cov |= set(map(str, sg.covered_universe))
    print(f"[CandidateCoverage] union_covered={len(cand_cov)} / universe={len(res.universe)} ({len(cand_cov)/max(1,len(res.universe)):.3f})")

    if bool(cfg.get("check_cost")):
        alpha = float(cfg["alpha"])
        beta = float(cfg["beta"])
        gamma = float(cfg["gamma"])
        for sg in res.merged_subgraphs:
            expected = alpha * len(sg.report_ids) + beta * len(sg.node_ids) + gamma * len(sg.edge_ids)
            if abs(float(sg.cost) - float(expected)) > 1e-9:
                raise ValueError(f"cost mismatch: {sg.subgraph_id} cost={sg.cost} expected={expected}")

    print("Building excel sheets...")
    sheets = build_excel_sheets(
        nodes_df=res.nodes_df,
        idx=res.idx,
        universe=res.universe,
        merged=res.merged_subgraphs,
        cover=res.cover,
        seed_stats=res.seed_stats,
        skipped=res.skipped,
        original_node_to_reports=res.original_node_to_reports,
        edge_only_nodes=res.edge_only_nodes,
    )
    write_excel_report(out_path, sheets)
    print(f"[OK] wrote excel: {out_path.as_posix()}")

    if cfg.get("brief_dir"):
        base_brief_dir = Path(str(cfg["brief_dir"]))
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_brief_dir = base_brief_dir / timestamp_str

        export_selected_subgraphs_brief_dir(
            res.idx,
            res.cover.selected,
            final_brief_dir,
            paths_max_len=int(cfg["paths_max_len"]),
            paths_top_k=int(cfg["paths_top_k"]),
            paths_max_output=int(cfg["paths_max_output"]),
        )
        print(f"[Export] brief_dir: {final_brief_dir.resolve().as_posix()}")

    return 0


__all__ = [
    "load_nodes_xlsx",
    "load_edges_xlsx",
    "load_defects_optional",
    "split_report_ids",
    "GraphIndex",
    "build_graph_index",
    "ExpansionResult",
    "expand_reports_from_seed",
    "Subgraph",
    "induce_subgraph_from_reports",
    "compute_cost",
    "make_subgraph",
    "merge_similar_subgraphs",
    "CoverResult",
    "greedy_set_cover",
    "export_selected_subgraphs_jsonl",
    "export_selected_subgraphs_brief_dir",
    "export_selected_subgraphs_dir",
    "PipelineResult",
    "run_subgraph_cover",
    "build_excel_sheets",
    "write_excel_report",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
