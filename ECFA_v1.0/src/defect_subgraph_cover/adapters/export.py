from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..core.index import GraphIndex
from ..core.subgraph import Subgraph
from .paths import stable_id_key, build_paths_descriptions


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
