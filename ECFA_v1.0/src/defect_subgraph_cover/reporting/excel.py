from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ..adapters.io import split_report_ids
from ..core.index import GraphIndex
from ..core.set_cover import CoverResult
from ..core.subgraph import Subgraph


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
