from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from .index import GraphIndex


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
