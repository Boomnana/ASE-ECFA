from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from .index import GraphIndex


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
