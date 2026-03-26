#!/usr/bin/env python3


from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ecfa_data_io import XlsxEvidenceGraphProvider
from ecfa_method import (
    ECFAParams,
    EvidenceGraph,
    build_graph_index,
    induce_candidates,
    compress_candidates,
    View,
)


TYPE_MAPPING = {

    "ISSUE": "ISSUE",
    "PHEN": "PHEN",
    "DIAG": "DIAG",

    "问题": "ISSUE",
    "故障": "ISSUE",
    "缺陷": "ISSUE",
    "现象": "PHEN",
    "症状": "PHEN",
    "诊断": "DIAG",
    "日志": "DIAG",
    "堆栈": "DIAG",
}

def normalize_entity_type(t: object) -> str:


    s0 = "" if t is None else str(t)
    s0 = s0.strip()
    if not s0:
        return s0
    if s0 in TYPE_MAPPING:
        return TYPE_MAPPING[s0]

    s = s0.strip().upper()
    if s in TYPE_MAPPING:
        return TYPE_MAPPING[s]


    if ("问题" in s0) or ("缺陷" in s0) or ("故障" in s0) or ("ISSUE" in s) or ("BUG" in s) or ("ERROR" in s):
        return "ISSUE"
    if ("现象" in s0) or ("症状" in s0) or ("PHEN" in s) or ("SYMPTOM" in s):
        return "PHEN"
    if ("诊断" in s0) or ("日志" in s0) or ("堆栈" in s0) or ("DIAG" in s) or ("LOG" in s) or ("TRACE" in s) or ("STACK" in s):
        return "DIAG"


    return s


def comb2(n: int) -> int:
    return n * (n - 1) // 2 if n >= 2 else 0


def jaccard_size(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    inter = 0
    for x in a:
        if x in b:
            inter += 1
    uni = len(a) + len(b) - inter
    return inter / uni if uni else 0.0


def summarize_pair_overlaps(
    clue_sets: List[Set[str]],
    *,
    eps: float = 0.01,
    topk_pairs: int = 30,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    n = len(clue_sets)
    total_pairs = comb2(n)
    if total_pairs == 0:
        summary = {
            "SelPairsTotal": int(total_pairs),
            "SelCluePosPairs(J>eps)": 0,
            "SelClueFrac(J>eps)": 0.0,
            "SelClueAvgOverlap_nz(J>eps)": 0.0,
            "SelClueMaxOverlap(J)": 0.0,
        }
        return summary, pd.DataFrame(columns=["i", "j", "J", "Ci", "Cj", "inter", "union"])

    pos = 0
    s_pos = 0.0
    max_j = 0.0

    import heapq
    heap: List[Tuple[float, int, int]] = []

    for i in range(n):
        Ci = clue_sets[i]
        for j in range(i + 1, n):
            Cj = clue_sets[j]
            J = jaccard_size(Ci, Cj)
            if J > max_j:
                max_j = J
            if J > eps:
                pos += 1
                s_pos += J
                if topk_pairs > 0:
                    if len(heap) < topk_pairs:
                        heapq.heappush(heap, (J, i, j))
                    elif J > heap[0][0]:
                        heapq.heapreplace(heap, (J, i, j))

    avg_nz = (s_pos / pos) if pos > 0 else 0.0
    frac = (pos / total_pairs) if total_pairs > 0 else 0.0

    summary = {
        "SelPairsTotal": int(total_pairs),
        "SelCluePosPairs(J>eps)": int(pos),
        "SelClueFrac(J>eps)": float(frac),
        "SelClueAvgOverlap_nz(J>eps)": float(avg_nz),
        "SelClueMaxOverlap(J)": float(max_j),
    }

    rows = []
    for (J, i, j) in sorted(heap, key=lambda t: t[0], reverse=True):
        Ci, Cj = clue_sets[i], clue_sets[j]
        if len(Ci) > len(Cj):
            small, big = Cj, Ci
        else:
            small, big = Ci, Cj
        inter = sum(1 for x in small if x in big)
        uni = len(Ci) + len(Cj) - inter
        rows.append({"i": i, "j": j, "J": float(J), "Ci": len(Ci), "Cj": len(Cj), "inter": int(inter), "union": int(uni)})
    return summary, pd.DataFrame(rows)


def greedy_select_exact_k(
    candidates: List[View],
    *,
    K: int,
    objective: str = "clue",
    w_clue: float = 1.0,
    w_report: float = 1.0,
    gain_only: bool = False,
    min_gain: float = 1e-12,
    report_universe: Optional[Set[str]] = None,
) -> Tuple[List[View], float, List[Dict[str, float]]]:
    remaining = list(candidates)
    selected: List[View] = []
    covered_clues: Set[str] = set()
    covered_reports: Set[str] = set()
    total_cost = 0.0
    trace: List[Dict[str, float]] = []

    U_reports = report_universe

    for step in range(1, K + 1):
        best_idx = None
        best_score = -1.0
        best_gain = 0.0

        for i, v in enumerate(remaining):
            new_clues = len(v.covered_universe - covered_clues)
            if U_reports is not None:
                new_reports = len((v.report_ids & U_reports) - covered_reports)
            else:
                new_reports = len(v.report_ids - covered_reports)

            if objective == "clue":
                gain = float(new_clues)
            elif objective == "report":
                gain = float(new_reports)
            else:
                gain = float(w_clue) * float(new_clues) + float(w_report) * float(new_reports)

            if gain < float(min_gain):
                continue

            c = float(v.cost)
            score = gain if gain_only else (gain / max(c, 1e-9))

            if score > best_score:
                best_score = score
                best_idx = i
                best_gain = gain

        if best_idx is None:
            break

        v = remaining.pop(best_idx)
        selected.append(v)
        total_cost += float(v.cost)

        covered_clues |= v.covered_universe
        if U_reports is not None:
            covered_reports |= (v.report_ids & U_reports)
        else:
            covered_reports |= v.report_ids

        trace.append({
            "step": step,
            "gain": float(best_gain),
            "score": float(best_score),
            "view_cost": float(v.cost),
            "cum_cost": float(total_cost),
            "cov_clue_cnt": float(len(covered_clues)),
            "cov_report_cnt": float(len(covered_reports)),
        })

    return selected, float(total_cost), trace


def nAUC_stepwise(trace: List[Dict[str, float]], *, budget: float, cov_key: str) -> float:
    if budget <= 0 or not trace:
        return 0.0
    area = 0.0
    prev_c = 0.0
    prev_cov = 0.0
    for row in trace:
        c = float(row["cum_cost"])
        cov = float(row[cov_key])
        c2 = min(c, budget)
        if c2 > prev_c:
            area += prev_cov * (c2 - prev_c)
            prev_c = c2
        prev_cov = cov
        if prev_c >= budget:
            break
    if prev_c < budget:
        area += prev_cov * (budget - prev_c)
    return area / float(budget)


def choose_data_dir(app_root: Path) -> Path:
    if (app_root / "unify" / "nodes.xlsx").exists() and (app_root / "unify" / "edges.xlsx").exists():
        return app_root / "unify"
    if (app_root / "no_unify" / "nodes.xlsx").exists() and (app_root / "no_unify" / "edges.xlsx").exists():
        return app_root / "no_unify"
    return app_root


def run_app_worker(
    *,
    input_root: str,
    app: str,
    lam_list: List[float],
    K_list: List[int],
    objective: str,
    w_clue: float,
    w_report: float,
    k: int,
    r_max: int,
    merge_threshold: float,
    red_eps: float,
    topk_pairs: int,
) -> Dict[str, pd.DataFrame]:
    input_root_p = Path(input_root)
    app_root = input_root_p / app
    data_dir = choose_data_dir(app_root)

    provider = XlsxEvidenceGraphProvider(
        nodes_path=data_dir / "nodes.xlsx",
        edges_path=data_dir / "edges.xlsx",
        defects_path=(app_root / "defects.xlsx") if (app_root / "defects.xlsx").exists() else None,
    )


    nodes_df = provider.load_nodes()


    if "type" in nodes_df.columns:
        nodes_df["type"] = nodes_df["type"].apply(normalize_entity_type)


    use_source_row_as_reports = False
    if "source_row_index" in nodes_df.columns:
        if "report_ids" not in nodes_df.columns:
            use_source_row_as_reports = True
        else:
            def is_empty_reports(x):
                if isinstance(x, list): return len(x) == 0
                return pd.isna(x) or str(x).strip() == ""
            if nodes_df["report_ids"].apply(is_empty_reports).all():
                use_source_row_as_reports = True

    if use_source_row_as_reports:
        print(f"[WARN] {app}: Using source_row_index as report_ids for nodes")

        nodes_df["report_ids"] = nodes_df["source_row_index"].apply(lambda x: [str(x)])

    edges_df = provider.load_edges()


    use_source_row_as_reports_edges = False
    if "source_row_index" in edges_df.columns:
        if "report_ids" not in edges_df.columns:
            use_source_row_as_reports_edges = True
        else:
            def is_empty_reports(x):
                if isinstance(x, list): return len(x) == 0
                return pd.isna(x) or str(x).strip() == ""
            if edges_df["report_ids"].apply(is_empty_reports).all():
                use_source_row_as_reports_edges = True

    if use_source_row_as_reports_edges:

        edges_df["report_ids"] = edges_df["source_row_index"].apply(lambda x: [str(x)])

    defects_df = None
    if provider.defects_path and provider.defects_path.exists():


        pass


    graph = EvidenceGraph(nodes=nodes_df, edges=edges_df, defects=defects_df)


    idx = build_graph_index(graph, backfill_node_reports_from_edges=True)


    universe_nodes = {nid for nid, t in idx.node_type_by_id.items() if t in ECFAParams().universe_types}
    universe_reports = set(idx.report_to_nodes.keys())

    if len(universe_nodes) == 0:
        seen = sorted(set(idx.node_type_by_id.values()))
        print(f"[WARN] App={app}: universe_nodes empty. Check type mapping. Seen types (head): {seen[:25]}")

    seed_node_ids = {nid for nid, t in idx.node_type_by_id.items() if t in ECFAParams().seed_types}
    if len(seed_node_ids) == 0:
        seen = sorted(set(idx.node_type_by_id.values()))
        print(f"[WARN] App={app}: seed_node_ids empty. No seeds of types {sorted(ECFAParams().seed_types)}. Seen types (head): {seen[:25]}")

    rows_pool: List[Dict] = []
    rows_summary: List[Dict] = []
    rows_step: List[Dict] = []
    rows_view: List[Dict] = []
    top_pairs_tables: List[pd.DataFrame] = []

    for lam in lam_list:
        params = ECFAParams(
            k=int(k),
            r_max=int(r_max),
            merge_threshold=float(merge_threshold),
            lam=float(lam),
            objective=str(objective),
            w_clue=float(w_clue),
            w_report=float(w_report),
        )

        candidates = induce_candidates(idx, params, universe_report_ids=None)
        compressed = compress_candidates(idx, params, candidates, universe_report_ids=None)

        rows_pool.append({
            "App": app,
            "Lambda": lam,
            "n_candidates": len(candidates),
            "n_compressed": len(compressed),
            "avg_cost_candidate": float(np.mean([v.cost for v in candidates])) if candidates else 0.0,
            "avg_cost_compressed": float(np.mean([v.cost for v in compressed])) if compressed else 0.0,
            "median_merge_group_size": float(np.median([v.meta.get("merge_group_size", 1) for v in compressed])) if compressed else 0.0,
            "mean_merge_group_size": float(np.mean([v.meta.get("merge_group_size", 1) for v in compressed])) if compressed else 0.0,
        })

        for K in K_list:
            selected, total_cost, trace = greedy_select_exact_k(
                compressed,
                K=int(K),
                objective=str(objective),
                w_clue=float(w_clue),
                w_report=float(w_report),
                gain_only=bool(params.gain_only),
                min_gain=float(params.min_gain),
            )

            nsel = len(selected)
            covered_clues = set().union(*(v.covered_universe for v in selected)) if selected else set()
            covered_reports = set().union(*(v.report_ids for v in selected)) if selected else set()

            cov_clue = (len(covered_clues) / len(universe_nodes)) if universe_nodes else 0.0
            cov_rep = (len(covered_reports) / len(universe_reports)) if universe_reports else 0.0

            nAUC_clue = nAUC_stepwise(trace, budget=max(total_cost, 1e-9), cov_key="cov_clue_cnt")
            nAUC_rep = nAUC_stepwise(trace, budget=max(total_cost, 1e-9), cov_key="cov_report_cnt")

            clue_sets = [set(v.covered_universe) for v in selected]
            comp_summary, top_pairs = summarize_pair_overlaps(clue_sets, eps=float(red_eps), topk_pairs=int(topk_pairs))
            top_pairs.insert(0, "K", K)
            top_pairs.insert(0, "Lambda", lam)
            top_pairs.insert(0, "App", app)
            top_pairs_tables.append(top_pairs)

            rows_summary.append({
                "App": app,
                "Lambda": lam,
                "K": K,
                "Objective": str(objective),
                "#Candidates": len(candidates),
                "#Compressed": len(compressed),
                "#Selected": nsel,
                "TotalCostSelected": float(total_cost),
                "AvgCostPerView": float(total_cost / max(nsel, 1)),
                "Coverage@K (Clue)": float(cov_clue),
                "Coverage@K (Report)": float(cov_rep),
                "nAUC@K (Clue over cost)": float(nAUC_clue),
                "nAUC@K (Report over cost)": float(nAUC_rep),
                **comp_summary,
            })

            for i, row in enumerate(trace, start=1):
                v = selected[i - 1]
                rows_step.append({
                    "App": app,
                    "Lambda": lam,
                    "K": K,
                    "step": int(row["step"]),
                    "view_id": v.view_id,
                    "seed": v.seed,
                    "view_cost": float(row["view_cost"]),
                    "cum_cost": float(row["cum_cost"]),
                    "gain": float(row["gain"]),
                    "score": float(row["score"]),
                    "cov_clue_cnt": float(row["cov_clue_cnt"]),
                    "cov_report_cnt": float(row["cov_report_cnt"]),
                    "cov_clue": float(row["cov_clue_cnt"] / max(len(universe_nodes), 1)),
                    "cov_report": float(row["cov_report_cnt"] / max(len(universe_reports), 1)),
                })

            for rank, v in enumerate(selected, start=1):
                rows_view.append({
                    "App": app,
                    "Lambda": lam,
                    "K": K,
                    "rank": rank,
                    "view_id": v.view_id,
                    "seed": v.seed,
                    "|Gamma|": len(v.report_ids),
                    "|V|": len(v.node_ids),
                    "|E|": len(v.edge_ids),
                    "|Clues|": len(v.covered_universe),
                    "cost": float(v.cost),
                    "merge_group_size": int(v.meta.get("merge_group_size", 1)),
                })

    df_pool = pd.DataFrame(rows_pool)
    df_summary = pd.DataFrame(rows_summary)
    df_steps = pd.DataFrame(rows_step)
    df_views = pd.DataFrame(rows_view)
    df_top_pairs = pd.concat(top_pairs_tables, ignore_index=True) if top_pairs_tables else pd.DataFrame(
        columns=["App", "Lambda", "K", "i", "j", "J", "Ci", "Cj", "inter", "union"]
    )

    if not df_summary.empty and "SelCluePosPairs(J>eps)" in df_summary.columns:
        df_summary["AvgOverlapsPerView(J>eps)"] = 2.0 * df_summary["SelCluePosPairs(J>eps)"] / df_summary["#Selected"].clip(lower=1)
        df_summary["OverlapRate%(J>eps)"] = 100.0 * df_summary["SelClueFrac(J>eps)"]

    return {
        "PoolStats_PerApp_Lambda": df_pool,
        "Summary_PerApp_LambdaK": df_summary,
        "SelectionTrace_Steps": df_steps,
        "SelectedViews_Stats": df_views,
        "TopOverlapPairs_Clue": df_top_pairs,
    }


def resolve_apps(input_root: Path, apps_csv: str) -> List[str]:
    if apps_csv.strip():
        return [a.strip() for a in apps_csv.split(",") if a.strip()]
    out = []
    for p in sorted(input_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "unify" / "nodes.xlsx").exists() and (p / "unify" / "edges.xlsx").exists():
            out.append(p.name)
        elif (p / "no_unify" / "nodes.xlsx").exists() and (p / "no_unify" / "edges.xlsx").exists():
            out.append(p.name)
        elif (p / "nodes.xlsx").exists() and (p / "edges.xlsx").exists():
            out.append(p.name)
    if not out:
        raise FileNotFoundError(f"No apps found under {input_root}. Provide --apps or ensure artifacts exist.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--apps", default="")
    ap.add_argument("--lambda_list", default="0,0.002,0.005,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--K_list", default="50")
    ap.add_argument("--objective", default="clue", choices=["clue", "report", "hybrid"])
    ap.add_argument("--w_clue", type=float, default=1.0)
    ap.add_argument("--w_report", type=float, default=1.0)

    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--r_max", type=int, default=200)
    ap.add_argument("--merge_threshold", type=float, default=0.55)
    ap.add_argument("--red_eps", type=float, default=0.01)
    ap.add_argument("--topk_pairs", type=int, default=30)

    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out_xlsx", required=True)
    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    apps = resolve_apps(input_root, args.apps)

    lam_list = [float(x.strip()) for x in args.lambda_list.split(",") if x.strip()]
    K_list = [int(x.strip()) for x in args.K_list.split(",") if x.strip()]

    if args.workers <= 0:
        workers = min(len(apps), os.cpu_count() or 1)
    else:
        workers = max(1, int(args.workers))

    results: List[Dict[str, pd.DataFrame]] = []
    if workers == 1 or len(apps) == 1:
        for app in apps:
            results.append(run_app_worker(
                input_root=str(input_root),
                app=app,
                lam_list=lam_list,
                K_list=K_list,
                objective=str(args.objective),
                w_clue=float(args.w_clue),
                w_report=float(args.w_report),
                k=int(args.k),
                r_max=int(args.r_max),
                merge_threshold=float(args.merge_threshold),
                red_eps=float(args.red_eps),
                topk_pairs=int(args.topk_pairs),
            ))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = []
            for app in apps:
                futs.append(ex.submit(
                    run_app_worker,
                    input_root=str(input_root),
                    app=app,
                    lam_list=lam_list,
                    K_list=K_list,
                    objective=str(args.objective),
                    w_clue=float(args.w_clue),
                    w_report=float(args.w_report),
                    k=int(args.k),
                    r_max=int(args.r_max),
                    merge_threshold=float(args.merge_threshold),
                    red_eps=float(args.red_eps),
                    topk_pairs=int(args.topk_pairs),
                ))
            for fut in as_completed(futs):
                results.append(fut.result())

    def cat(sheet: str) -> pd.DataFrame:
        dfs = [r[sheet] for r in results if sheet in r and r[sheet] is not None and not r[sheet].empty]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    df_pool = cat("PoolStats_PerApp_Lambda")
    df_summary = cat("Summary_PerApp_LambdaK")
    df_steps = cat("SelectionTrace_Steps")
    df_views = cat("SelectedViews_Stats")
    df_top_pairs = cat("TopOverlapPairs_Clue")

    out_xlsx = Path(args.out_xlsx).resolve()
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        pd.DataFrame([{
            "input_root": str(input_root),
            "apps": ",".join(apps),
            "lambda_list": ",".join(map(str, lam_list)),
            "K_list": ",".join(map(str, K_list)),
            "objective": args.objective,
            "w_clue": args.w_clue,
            "w_report": args.w_report,
            "k": args.k,
            "r_max": args.r_max,
            "merge_threshold": args.merge_threshold,
            "red_eps": args.red_eps,
            "topk_pairs": args.topk_pairs,
            "workers": workers,
        }]).to_excel(xw, sheet_name="Config", index=False)

        df_pool.to_excel(xw, sheet_name="PoolStats_PerApp_Lambda", index=False)
        df_summary.to_excel(xw, sheet_name="Summary_PerApp_LambdaK", index=False)
        df_steps.to_excel(xw, sheet_name="SelectionTrace_Steps", index=False)
        df_views.to_excel(xw, sheet_name="SelectedViews_Stats", index=False)
        df_top_pairs.to_excel(xw, sheet_name="TopOverlapPairs_Clue", index=False)

    print(f"[OK] Wrote plotting workbook: {out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
