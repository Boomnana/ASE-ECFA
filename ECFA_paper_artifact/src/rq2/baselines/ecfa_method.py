from __future__ import annotations

"""
ECFA core algorithm (method-only).

This file contains ONLY the ECFA method components, independent of any
experimental harness (no baselines, no ablations, no plotting, no CLI).

It intentionally exposes:
- data-facing interfaces (EvidenceGraph, GraphIndex) so experiments can feed data
- parameter interface (ECFAParams) so experiments can configure the method

Algorithmic stages included:
  1) Index (provenance-aware graph indexing)
  2) Induce (candidate view induction via node↔report alternating expansion)
  3) Merge (candidate compression via clue-signature similarity; evidence-first merge)
  4) Select (budget-aware greedy selection using marginal gain / unit cost)
  5) (Optional) Retrieval (evidence-based cross-boundary retrieval over selected views)

All dataset I/O is delegated to `ecfa_data_io.py`.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple, Literal

import pandas as pd
import heapq


NodeId = str
EdgeId = str
ReportId = str
EntityType = str

Objective = Literal["clue", "report", "hybrid"]
StopReason = Literal["converged", "r-max-truncated", "k-reached", "no-evidence"]


@dataclass(frozen=True)
class ECFAParams:


    seed_types: Set[EntityType] = field(default_factory=lambda: {"ISSUE", "PHEN", "DIAG"})
    universe_types: Set[EntityType] = field(default_factory=lambda: {"ISSUE", "PHEN", "DIAG"})


    k: int = 2
    r_max: int = 200
    strict_r_max: bool = True
    hop_radius: Optional[int] = None


    merge_threshold: float = 0.55
    merge_weights: Dict[EntityType, float] = field(default_factory=lambda: {
        "ISSUE": 0.45,
        "PHEN": 0.30,
        "DIAG": 0.20,
        "OP": 0.05,
    })
    min_phen_diag_when_no_issue: float = 0.50
    max_nodes_after_merge: Optional[int] = None
    max_reports_after_merge: Optional[int] = None
    max_universe_after_merge: Optional[int] = None
    reinduce_from_union_evidence: bool = True


    lam: float = 0.01


    objective: Objective = "clue"
    w_clue: float = 1.0
    w_report: float = 1.0
    min_gain: float = 1e-12
    gain_only: bool = False

    @staticmethod
    def from_dict(d: Dict) -> "ECFAParams":

        dd = dict(d)
        if "seed_types" in dd and not isinstance(dd["seed_types"], set):
            dd["seed_types"] = set(dd["seed_types"])
        if "universe_types" in dd and not isinstance(dd["universe_types"], set):
            dd["universe_types"] = set(dd["universe_types"])
        return ECFAParams(**dd)


@dataclass
class EvidenceGraph:


    nodes: pd.DataFrame
    edges: pd.DataFrame
    defects: Optional[pd.DataFrame] = None


@dataclass
class GraphIndex:

    graph: EvidenceGraph

    node_to_reports: Dict[NodeId, Set[ReportId]]
    report_to_nodes: Dict[ReportId, Set[NodeId]]

    report_to_edges: Dict[ReportId, Set[EdgeId]]
    edge_id_to_row: Dict[EdgeId, dict]
    node_id_to_row: Dict[NodeId, dict]

    node_type_by_id: Dict[NodeId, EntityType]


    neighbors: Dict[NodeId, Set[NodeId]]


@dataclass
class View:


    view_id: str
    seed: Optional[NodeId]

    node_ids: Set[NodeId]
    edge_ids: Set[EdgeId]
    report_ids: Set[ReportId]

    covered_universe: Set[NodeId]
    covered_report_ids: Set[ReportId]

    cost: float
    meta: Dict[str, object] = field(default_factory=dict)


def build_graph_index(
    graph: EvidenceGraph,
    *,
    backfill_node_reports_from_edges: bool = True,
) -> GraphIndex:


    nodes = graph.nodes
    edges = graph.edges

    node_to_reports: Dict[NodeId, Set[ReportId]] = {}
    report_to_nodes: Dict[ReportId, Set[NodeId]] = {}
    node_id_to_row: Dict[NodeId, dict] = {}
    node_type_by_id: Dict[NodeId, EntityType] = {}

    for row in nodes.to_dict(orient="records"):
        nid = str(row["id"])
        node_id_to_row[nid] = row
        node_type_by_id[nid] = str(row.get("type", ""))
        rids = row.get("report_ids", []) or []
        rset = set(map(str, rids))
        node_to_reports[nid] = rset
        for rid in rset:
            report_to_nodes.setdefault(rid, set()).add(nid)

    report_to_edges: Dict[ReportId, Set[EdgeId]] = {}
    edge_id_to_row: Dict[EdgeId, dict] = {}

    neighbors: Dict[NodeId, Set[NodeId]] = {}

    for row in edges.to_dict(orient="records"):
        eid = str(row["id"])
        edge_id_to_row[eid] = row
        src_id = str(row.get("source_id", "") or "")
        tgt_id = str(row.get("target_id", "") or "")

        if src_id and tgt_id:
            neighbors.setdefault(src_id, set()).add(tgt_id)
            neighbors.setdefault(tgt_id, set()).add(src_id)

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

    return GraphIndex(
        graph=graph,
        node_to_reports=node_to_reports,
        report_to_nodes=report_to_nodes,
        report_to_edges=report_to_edges,
        edge_id_to_row=edge_id_to_row,
        node_id_to_row=node_id_to_row,
        node_type_by_id=node_type_by_id,
        neighbors=neighbors,
    )


def jaccard(a: Set, b: Set, *, empty_both: float = 1.0) -> float:
    if not a and not b:
        return float(empty_both)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


def _weighted_jaccard(sig1: Dict[str, Set[str]], sig2: Dict[str, Set[str]], weights: Dict[str, float]) -> float:
    total_w = 0.0
    s = 0.0
    for t, w in weights.items():
        total_w += float(w)
        s += float(w) * jaccard(sig1.get(t, set()), sig2.get(t, set()), empty_both=0.0)
    return (s / total_w) if total_w > 0 else 0.0


def _bfs_within_hops(neighbors: Dict[NodeId, Set[NodeId]], anchors: Set[NodeId], h: int) -> Set[NodeId]:

    if h < 0:
        return set()
    visited = set(anchors)
    frontier = set(anchors)
    for _ in range(h):
        nxt = set()
        for u in frontier:
            for v in neighbors.get(u, set()):
                if v not in visited:
                    visited.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    return visited


@dataclass
class ExpansionResult:
    seed_node: NodeId
    report_ids: Set[ReportId]
    node_ids: Set[NodeId]
    stopped_reason: StopReason


def expand_from_seed(
    idx: GraphIndex,
    seed_node: NodeId,
    *,
    k: int,
    r_max: int,
    strict_r_max: bool = True,
    hop_radius: Optional[int] = None,
) -> ExpansionResult:


    R: Set[ReportId] = set(idx.node_to_reports.get(seed_node, set()))
    if not R:
        return ExpansionResult(seed_node=seed_node, report_ids=set(), node_ids={seed_node}, stopped_reason="no-evidence")

    V: Set[NodeId] = {seed_node}
    stopped: StopReason = "k-reached"

    for _step in range(1, k + 1):

        V_new: Set[NodeId] = set()
        for rid in R:
            V_new |= idx.report_to_nodes.get(rid, set())
        V2 = V | V_new


        if hop_radius is not None:
            allowed = _bfs_within_hops(idx.neighbors, anchors={seed_node}, h=int(hop_radius))
            V2 = {v for v in V2 if v in allowed}


        R_new: Set[ReportId] = set()
        for nid in V2:
            R_new |= idx.node_to_reports.get(nid, set())
        R2 = R | R_new


        if V2 == V and R2 == R:
            stopped = "converged"
            V, R = V2, R2
            break


        if strict_r_max and len(R2) > r_max:
            kept = set(sorted(R2)[: int(r_max)])
            R2 = kept


            V2 = {seed_node}
            for rid in R2:
                V2 |= idx.report_to_nodes.get(rid, set())

            if hop_radius is not None:
                allowed = _bfs_within_hops(idx.neighbors, anchors={seed_node}, h=int(hop_radius))
                V2 = {v for v in V2 if v in allowed}

            stopped = "r-max-truncated"
            V, R = V2, R2
            break

        V, R = V2, R2

    return ExpansionResult(seed_node=seed_node, report_ids=R, node_ids=V, stopped_reason=stopped)


def induce_subgraph_from_reports(idx: GraphIndex, report_ids: Set[ReportId]) -> Tuple[Set[NodeId], Set[EdgeId]]:

    node_ids: Set[NodeId] = set()
    edge_ids: Set[EdgeId] = set()
    for rid in report_ids:
        node_ids |= idx.report_to_nodes.get(rid, set())
        edge_ids |= idx.report_to_edges.get(rid, set())
    return node_ids, edge_ids


def make_view(
    idx: GraphIndex,
    params: ECFAParams,
    *,
    view_id: str,
    seed: Optional[NodeId],
    report_ids: Set[ReportId],
    universe_nodes: Set[NodeId],
    universe_reports: Optional[Set[ReportId]] = None,
) -> View:


    node_ids, edge_ids = induce_subgraph_from_reports(idx, report_ids)
    covered_universe = set(node_ids) & set(universe_nodes)

    if universe_reports is not None:
        covered_report_ids = set(report_ids) & set(universe_reports)
    else:
        covered_report_ids = set(report_ids)

    cost = float(len(report_ids)) + float(params.lam) * (float(len(node_ids)) + float(len(edge_ids)))

    return View(
        view_id=view_id,
        seed=seed,
        node_ids=node_ids,
        edge_ids=edge_ids,
        report_ids=set(report_ids),
        covered_universe=covered_universe,
        covered_report_ids=covered_report_ids,
        cost=cost,
    )


def induce_candidates(
    idx: GraphIndex,
    params: ECFAParams,
    *,
    seed_node_ids: Optional[Iterable[NodeId]] = None,
    universe_report_ids: Optional[Set[ReportId]] = None,
) -> List[View]:


    if seed_node_ids is None:
        seed_node_ids = [nid for nid, t in idx.node_type_by_id.items() if t in params.seed_types]

    universe_nodes = {nid for nid, t in idx.node_type_by_id.items() if t in params.universe_types}

    out: List[View] = []
    for s in seed_node_ids:
        exp = expand_from_seed(
            idx,
            seed_node=str(s),
            k=int(params.k),
            r_max=int(params.r_max),
            strict_r_max=bool(params.strict_r_max),
            hop_radius=params.hop_radius,
        )
        if exp.stopped_reason == "no-evidence" or not exp.report_ids:
            continue
        out.append(
            make_view(
                idx,
                params,
                view_id=f"seed:{s}",
                seed=str(s),
                report_ids=set(exp.report_ids),
                universe_nodes=universe_nodes,
                universe_reports=universe_report_ids,
            )
        )
    return out


def _signature_sets(node_ids: Set[NodeId], node_type_by_id: Dict[NodeId, EntityType]) -> Dict[EntityType, Set[NodeId]]:
    sig: Dict[EntityType, Set[NodeId]] = {}
    for nid in node_ids:
        t = node_type_by_id.get(nid, "")
        sig.setdefault(t, set()).add(nid)
    return sig


def compress_candidates(
    idx: GraphIndex,
    params: ECFAParams,
    candidates: List[View],
    *,
    universe_report_ids: Optional[Set[ReportId]] = None,
) -> List[View]:


    if not candidates:
        return []

    remaining = list(candidates)
    n_candidates = len(remaining)
    used = [False] * n_candidates
    sigs = [_signature_sets(v.node_ids, idx.node_type_by_id) for v in remaining]


    node_to_indices: Dict[NodeId, List[int]] = {}
    for i in range(n_candidates):
        nodes_in_sig = set()
        for t, s in sigs[i].items():
             if params.merge_weights.get(t, 0.0) > 0:
                 nodes_in_sig.update(s)
        for nid in nodes_in_sig:
            if nid not in node_to_indices:
                node_to_indices[nid] = []
            node_to_indices[nid].append(i)

    universe_nodes = {nid for nid, t in idx.node_type_by_id.items() if t in params.universe_types}

    merged: List[View] = []
    group_id = 0

    for i in range(n_candidates):
        if used[i]:
            continue
        used[i] = True
        group = [i]

        rep_union = set(remaining[i].report_ids)
        clue_union = set(remaining[i].covered_universe)
        sig_union = {k: set(v) for k, v in sigs[i].items()}


        potential_indices = set()
        for t, s in sigs[i].items():
            if params.merge_weights.get(t, 0.0) > 0:
                for nid in s:
                    potential_indices.update(node_to_indices.get(nid, []))

        sorted_candidates = sorted([x for x in potential_indices if x > i and not used[x]])

        k_idx = 0
        while k_idx < len(sorted_candidates):
            j = sorted_candidates[k_idx]
            k_idx += 1

            if used[j]:
                continue

            vj = remaining[j]
            sigj = sigs[j]

            issue_j = jaccard(sig_union.get("ISSUE", set()), sigj.get("ISSUE", set()), empty_both=0.0)
            phen_diag_j = jaccard(
                (sig_union.get("PHEN", set()) | sig_union.get("DIAG", set())),
                (sigj.get("PHEN", set()) | sigj.get("DIAG", set())),
                empty_both=0.0,
            )
            if issue_j <= 1e-12 and phen_diag_j < float(params.min_phen_diag_when_no_issue):
                continue

            sim = _weighted_jaccard(sig_union, sigj, params.merge_weights)
            if sim < float(params.merge_threshold):
                continue


            if params.max_reports_after_merge is not None and len(rep_union | vj.report_ids) > int(params.max_reports_after_merge):
                continue
            if params.max_universe_after_merge is not None and len(clue_union | vj.covered_universe) > int(params.max_universe_after_merge):
                continue

            used[j] = True
            group.append(j)
            rep_union |= vj.report_ids
            clue_union |= vj.covered_universe
            for t, s in sigj.items():
                sig_union.setdefault(t, set()).update(s)


            new_pots = set()
            for t, s in sigj.items():
                if params.merge_weights.get(t, 0.0) > 0:
                    for nid in s:
                        new_pots.update(node_to_indices.get(nid, []))

            to_add = []
            for p in new_pots:
                if p > j and not used[p] and p not in potential_indices:
                     to_add.append(p)
                     potential_indices.add(p)

            if to_add:

                current_remainder = sorted_candidates[k_idx:]
                new_remainder = sorted(current_remainder + to_add)
                sorted_candidates = sorted_candidates[:k_idx] + new_remainder

        group_id += 1

        if params.reinduce_from_union_evidence:
            v_new = make_view(
                idx,
                params,
                view_id=f"merge:{group_id}",
                seed=None,
                report_ids=rep_union,
                universe_nodes=universe_nodes,
                universe_reports=universe_report_ids,
            )
        else:
            node_union: Set[NodeId] = set()
            edge_union: Set[EdgeId] = set()
            for g in group:
                node_union |= remaining[g].node_ids
                edge_union |= remaining[g].edge_ids
            cost = float(len(rep_union)) + float(params.lam) * (float(len(node_union)) + float(len(edge_union)))
            v_new = View(
                view_id=f"merge:{group_id}",
                seed=None,
                node_ids=node_union,
                edge_ids=edge_union,
                report_ids=rep_union,
                covered_universe=set(node_union) & universe_nodes,
                covered_report_ids=set(rep_union) if universe_report_ids is None else set(rep_union) & set(universe_report_ids),
                cost=cost,
            )

        v_new.meta["merge_group_size"] = len(group)
        v_new.meta["merge_members"] = [remaining[g].view_id for g in group]

        if params.max_nodes_after_merge is not None and len(v_new.node_ids) > int(params.max_nodes_after_merge):
            continue

        merged.append(v_new)

    return merged


@dataclass
class SelectionResult:
    selected: List[View]
    total_cost: float
    covered_universe: Set[NodeId]
    covered_reports: Set[ReportId]


def select_views(
    params: ECFAParams,
    candidates: List[View],
    *,
    budget: float,
    universe_report_ids: Optional[Set[ReportId]] = None,
) -> SelectionResult:


    remaining = list(candidates)
    selected: List[View] = []

    covered_clues: Set[NodeId] = set()
    covered_reports: Set[ReportId] = set()
    total_cost = 0.0

    U_reports = universe_report_ids


    pq = []


    for i, v in enumerate(remaining):
        c = float(v.cost)
        if c > float(budget):
            continue

        if params.objective == "clue":
            gain = float(len(v.covered_universe))
        elif params.objective == "report":
            if U_reports is not None:
                gain = float(len(v.report_ids & U_reports))
            else:
                gain = float(len(v.report_ids))
        else:
            clue_gain = float(len(v.covered_universe))
            if U_reports is not None:
                rep_gain = float(len(v.report_ids & U_reports))
            else:
                rep_gain = float(len(v.report_ids))
            gain = float(params.w_clue) * clue_gain + float(params.w_report) * rep_gain

        if gain < float(params.min_gain):
            continue

        score = gain if bool(params.gain_only) else (gain / max(c, 1e-9))
        heapq.heappush(pq, (-score, i))

    while pq:
        neg_score, idx = heapq.heappop(pq)
        v = remaining[idx]


        if total_cost + float(v.cost) > float(budget):
            continue


        new_clues = len(v.covered_universe - covered_clues)
        if U_reports is not None:
            new_reports = len((v.report_ids & U_reports) - covered_reports)
        else:
            new_reports = len(v.report_ids - covered_reports)

        if params.objective == "clue":
            gain = float(new_clues)
        elif params.objective == "report":
            gain = float(new_reports)
        else:
            gain = float(params.w_clue) * float(new_clues) + float(params.w_report) * float(new_reports)

        if gain < float(params.min_gain):
            continue

        c = float(v.cost)
        new_score = gain if bool(params.gain_only) else (gain / max(c, 1e-9))


        if not pq:
            selected.append(v)
            total_cost += c
            covered_clues |= v.covered_universe
            if U_reports is not None:
                covered_reports |= (v.report_ids & U_reports)
            else:
                covered_reports |= v.report_ids
            break


        next_best_score = -pq[0][0]

        if new_score >= next_best_score:

            selected.append(v)
            total_cost += c
            covered_clues |= v.covered_universe
            if U_reports is not None:
                covered_reports |= (v.report_ids & U_reports)
            else:
                covered_reports |= v.report_ids
        else:

            heapq.heappush(pq, (-new_score, idx))

    return SelectionResult(
        selected=selected,
        total_cost=float(total_cost),
        covered_universe=covered_clues,
        covered_reports=covered_reports,
    )


def retrieve_related_reports(
    selected_views: List[View],
    *,
    query_anchor_nodes: Set[NodeId],
    exclude_report_id: Optional[ReportId] = None,
) -> Set[ReportId]:


    out: Set[ReportId] = set()
    if not query_anchor_nodes:
        return out
    for v in selected_views:
        if v.node_ids & query_anchor_nodes:
            out |= v.report_ids
    if exclude_report_id is not None:
        out.discard(str(exclude_report_id))
    return out


@dataclass
class ECFAOutput:

    candidates: List[View]
    compressed: List[View]
    selection: SelectionResult


def ecfa(
    graph: EvidenceGraph,
    params: ECFAParams,
    *,
    budget: float,
    backfill_node_reports_from_edges: bool = True,
    universe_report_ids: Optional[Set[ReportId]] = None,
) -> ECFAOutput:


    idx = build_graph_index(graph, backfill_node_reports_from_edges=backfill_node_reports_from_edges)
    candidates = induce_candidates(idx, params, universe_report_ids=universe_report_ids)
    compressed = compress_candidates(idx, params, candidates, universe_report_ids=universe_report_ids)
    selection = select_views(params, compressed, budget=float(budget), universe_report_ids=universe_report_ids)
    return ECFAOutput(candidates=candidates, compressed=compressed, selection=selection)
