from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Set, List
import pandas as pd
from datetime import datetime

try:
    from . import (
        load_nodes_xlsx,
        load_edges_xlsx,
        load_defects_optional,
        run_subgraph_cover,
        build_excel_sheets,
        write_excel_report,
        export_selected_subgraphs_brief_dir,
    )
except ImportError:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from defect_subgraph_cover import (
        load_nodes_xlsx,
        load_edges_xlsx,
        load_defects_optional,
        run_subgraph_cover,
        build_excel_sheets,
        write_excel_report,
        export_selected_subgraphs_brief_dir,
    )

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SHARED_OUTPUT = REPO_ROOT / "shared_output"
INPUT_DIR = REPO_ROOT / "input"

DEFAULTS = {
    "nodes_path": str(SHARED_OUTPUT / "nodes.xlsx"),
    "edges": str(SHARED_OUTPUT / "edges.xlsx"),
    "defects": str(INPUT_DIR / "测试雅思.xlsx"),
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

def main():
    print(f"[Start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ap = argparse.ArgumentParser(description="Report-induced subgraph cover (xlsx nodes/edges).")
    ap.add_argument("nodes_path", nargs="?", default=argparse.SUPPRESS, help=f"Path to nodes.xlsx (default: {DEFAULTS['nodes_path']})")
    ap.add_argument("--nodes", dest="nodes_path", default=argparse.SUPPRESS, help=f"Path to nodes.xlsx (default: {DEFAULTS['nodes_path']})")
    ap.add_argument("--edges", default=argparse.SUPPRESS, help=f"Path to edges.xlsx (default: {DEFAULTS['edges']})")
    ap.add_argument("--defects", default=argparse.SUPPRESS, help=f"Optional defects file (xlsx/csv/jsonl) with columns id,description (default: {DEFAULTS['defects']})")
    ap.add_argument("--k", type=int, default=argparse.SUPPRESS, help=f"Hop rounds for expansion (default: {DEFAULTS['k']})")
    ap.add_argument("--r_max", type=int, default=argparse.SUPPRESS, help=f"Max reports during expansion per seed (default: {DEFAULTS['r_max']})")
    ap.add_argument("--soft_r_max", action="store_true", default=argparse.SUPPRESS, help="Use soft r_max (stop when reached, without truncation)")
    ap.add_argument("--check_cost", action="store_true", default=argparse.SUPPRESS, help="Validate cost consistency after merge")
    ap.add_argument("--merge_jaccard", type=float, default=argparse.SUPPRESS, help=f"Threshold for merging subgraphs (default: {DEFAULTS['merge_jaccard']})")
    ap.add_argument("--budget", type=int, default=argparse.SUPPRESS, help=f"Max selected subgraphs for set cover (default: {DEFAULTS['budget']})")
    ap.add_argument("--out", default=argparse.SUPPRESS, help=f"Output excel path (.xlsx). If you pass .jsonl, it will be normalized to .xlsx (default: {DEFAULTS['out']})")
    ap.add_argument("--brief_dir", default=argparse.SUPPRESS, help="Optional: output per-subgraph brief jsons into this directory (default: empty)")
    ap.add_argument("--paths_max_len", type=int, default=argparse.SUPPRESS, help=f"Max edges per path for brief export (default: {DEFAULTS['paths_max_len']})")
    ap.add_argument("--paths_top_k", type=int, default=argparse.SUPPRESS, help=f"TopK raw paths before compression (default: {DEFAULTS['paths_top_k']})")
    ap.add_argument("--paths_max_output", type=int, default=argparse.SUPPRESS, help=f"Max path lines in brief output (default: {DEFAULTS['paths_max_output']})")
    ap.add_argument("--seed_types", default=argparse.SUPPRESS, help=f"Comma-separated seed node types (default: {DEFAULTS['seed_types']})")
    ap.add_argument("--universe_types", default=argparse.SUPPRESS, help=f"Comma-separated universe node types (default: {DEFAULTS['universe_types']})")
    ap.add_argument("--alpha", type=float, default=argparse.SUPPRESS, help=f"Cost weight for #reports (default: {DEFAULTS['alpha']})")
    ap.add_argument("--beta", type=float, default=argparse.SUPPRESS, help=f"Cost weight for #nodes (default: {DEFAULTS['beta']})")
    ap.add_argument("--gamma", type=float, default=argparse.SUPPRESS, help=f"Cost weight for #edges (default: {DEFAULTS['gamma']})")

    args = ap.parse_args()
    provided = vars(args)
    cfg = dict(DEFAULTS)
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
    defaulted_keys = sorted([k for k in DEFAULTS.keys() if k not in provided])
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

if __name__ == "__main__":
    main()
