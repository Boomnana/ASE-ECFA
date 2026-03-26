from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Optional

import pandas as pd


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
