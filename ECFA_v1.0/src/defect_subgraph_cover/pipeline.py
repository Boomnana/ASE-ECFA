from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Optional

import pandas as pd

from .core import (
    GraphIndex,
    build_graph_index,
    expand_reports_from_seed,
    make_subgraph,
    merge_similar_subgraphs,
    greedy_set_cover,
    CoverResult,
    Subgraph,
)


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
