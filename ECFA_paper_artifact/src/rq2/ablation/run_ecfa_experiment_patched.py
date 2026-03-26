

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
import math

import pandas as pd
import numpy as np


from ecfa_data_io import XlsxEvidenceGraphProvider, load_defects_optional
from ecfa_method import (
    ECFAParams,
    EvidenceGraph,
    View,
    build_graph_index,
    induce_candidates,
    compress_candidates,
    select_views,
    jaccard
)


DEFAULTS = {
    "seed_types": "ISSUE,PHEN,DIAG",
    "universe_types": "ISSUE,PHEN,DIAG"
}

TYPE_MAPPING = {
    "问题陈述": "ISSUE",
    "用户感知现象": "PHEN",
    "系统诊断信息": "DIAG",
    "用户操作": "OP",
    "功能模块": "MOD",
    "影响元素": "ELEM"
}

def _safe_filename(name: str) -> str:
    s = str(name)

    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        s = s.replace(ch, '_')
    s = s.strip()
    return s or "app"

def _list_apps_by_sheets(xlsx_path: Path, include_deprecated: bool = False) -> List[str]:
    xl = pd.ExcelFile(xlsx_path)
    sheets = xl.sheet_names


    apps = [s for s in sheets if not s.lower().startswith("meta") and not s.lower().startswith("sheet")]
    return sorted(apps)

def _parse_type_list(s: str) -> Set[str]:
    return {t.strip() for t in s.split(",") if t.strip()}

def _fmt(val: float, std: float, digits: int = 4) -> str:
    return f"{val:.{digits}f}±{std:.{digits}f}"

def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0

def _std(xs: List[float]) -> float:
    return float(np.std(xs)) if xs else 0.0

def norm_set_report_ids(ids: Any) -> Set[str]:
    if pd.isna(ids) or ids is None:
        return set()
    if isinstance(ids, (int, float, np.integer, np.floating)):
        return {str(int(ids))}
    if isinstance(ids, str):
        return {x.strip() for x in ids.split(",") if x.strip()}
    if isinstance(ids, (set, list, tuple, np.ndarray)):
        return {str(x) for x in ids if str(x)}
    return {str(ids)}

def _pairwise_redundancy_stats(
    views: List[View],
    set_getter: Callable[[View], Set[Any]],
    *,
    eps: float = 1e-6,
    max_pairs: Optional[int] = 50000,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:


    n = len(views)
    if n <= 1:
        return {"n_pairs": 0, "avg": 0.0, "max": 0.0, "avg_pos": 0.0, "frac_pos": 0.0, "n_pos_pairs": 0, "ov_deg": 0.0}

    total_possible = n * (n - 1) // 2
    use_all = (max_pairs is None) or (total_possible <= int(max_pairs))

    pairs: List[Tuple[int, int]] = []
    if use_all:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
    else:

        r = rng if rng is not None else random.Random()
        k = int(max_pairs)
        for _ in range(k):
            i = r.randint(0, n - 2)
            j = r.randint(i + 1, n - 1)
            pairs.append((i, j))

    if not pairs:
        return {"n_pairs": 0, "avg": 0.0, "max": 0.0, "avg_pos": 0.0, "frac_pos": 0.0, "n_pos_pairs": 0, "ov_deg": 0.0}

    sims: List[float] = []
    max_sim = 0.0
    for i, j in pairs:
        s1 = set_getter(views[i]) or set()
        s2 = set_getter(views[j]) or set()
        sim = jaccard(set(s1), set(s2), empty_both=0.0)
        sims.append(sim)
        if sim > max_sim:
            max_sim = sim

    sims_arr = np.asarray(sims, dtype=float)
    avg_val = float(np.mean(sims_arr)) if sims_arr.size else 0.0

    pos_mask = sims_arr > float(eps)
    n_pos = int(np.sum(pos_mask))
    avg_pos = float(np.mean(sims_arr[pos_mask])) if n_pos > 0 else 0.0
    frac_pos = (n_pos / float(sims_arr.size)) if sims_arr.size else 0.0


    ov_deg = frac_pos * float(n - 1) if n > 1 else 0.0

    return {
        "n_pairs": int(sims_arr.size),
        "avg": avg_val,
        "max": float(max_sim),
        "avg_pos": avg_pos,
        "frac_pos": frac_pos,
        "n_pos_pairs": n_pos,
        "ov_deg": ov_deg,
    }


def calculate_pool_redundancy(
    candidates: List[View],
    *,
    eps: float = 1e-6,
    max_pairs: int = 50000,
    rng: Optional[random.Random] = None,
    mode: str = "clue",
) -> Dict[str, Any]:


    if mode == "node":
        getter = lambda v: v.node_ids
    elif mode == "report":
        getter = lambda v: v.covered_report_ids
    else:
        getter = lambda v: v.covered_universe

    return _pairwise_redundancy_stats(
        candidates, getter, eps=float(eps), max_pairs=int(max_pairs), rng=rng
    )


def calculate_curve_and_metrics(
    selected_views: List[View],
    total_budget: float,
    curve_points: int,
    universe_items: Set[str],
    item_getter: Callable[[View], Set[str]]
) -> Tuple[Dict[str, List[float]], float, float]:


    xs = [0.0]
    ys = [0.0]

    current_cost = 0.0
    covered_items = set()


    for v in selected_views:
        current_cost += v.cost
        covered_items |= (item_getter(v) & universe_items)

        xs.append(current_cost)
        ys.append(len(covered_items))


    max_y = len(universe_items) if universe_items else 1.0
    norm_ys = [y / max_y if max_y > 0 else 0.0 for y in ys]


    norm_xs = [min(x / total_budget, 1.0) if total_budget > 0 else 0.0 for x in xs]


    target_xs = np.linspace(0.0, 1.0, curve_points)
    target_ys = np.interp(target_xs, norm_xs, norm_ys)

    curve = {
        "x": target_xs.tolist(),
        "y": target_ys.tolist()
    }


    nauc = np.trapz(target_ys, target_xs)

    final_cov = norm_ys[-1] if norm_ys else 0.0

    return curve, nauc, final_cov


def build_raw_candidates(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    defects_df: Optional[pd.DataFrame],
    seed_types: Set[str],
    universe_types: Set[str],
    k: int,
    r_max: int,
    strict_r_max: bool,
    lam: float,
    backfill_node_reports_from_edges: bool
) -> Dict[str, Any]:


    if "id" in nodes_df.columns:
        nodes_df["id"] = nodes_df["id"].astype(str)
    if "id" in edges_df.columns:
        edges_df["id"] = edges_df["id"].astype(str)
        edges_df["source_id"] = edges_df["source_id"].astype(str)
        edges_df["target_id"] = edges_df["target_id"].astype(str)


    if "report_ids" in nodes_df.columns:
        nodes_df["report_ids"] = nodes_df["report_ids"].apply(lambda x: list(norm_set_report_ids(x)))
    if "report_ids" in edges_df.columns:
        edges_df["report_ids"] = edges_df["report_ids"].apply(lambda x: list(norm_set_report_ids(x)))

    graph = EvidenceGraph(nodes=nodes_df, edges=edges_df, defects=defects_df)


    idx = build_graph_index(graph, backfill_node_reports_from_edges=backfill_node_reports_from_edges)


    params = ECFAParams(
        seed_types=seed_types,
        universe_types=universe_types,
        k=k,
        r_max=r_max,
        strict_r_max=strict_r_max,
        lam=lam
    )


    universe_report_ids = set()
    for nid, rids in idx.node_to_reports.items():
        if idx.node_type_by_id.get(nid) in universe_types:
            universe_report_ids.update(rids)

    raw_subgraphs = induce_candidates(idx, params, universe_report_ids=universe_report_ids)


    universe_nodes = {nid for nid, t in idx.node_type_by_id.items() if t in universe_types}

    return {
        "raw_subgraphs": raw_subgraphs,
        "node_type_by_id": idx.node_type_by_id,
        "universe": universe_nodes,
        "universe_reports": universe_report_ids,
        "maybe_truncated_cnt": sum(1 for v in raw_subgraphs if len(v.report_ids) >= r_max),
        "seeds": [v.seed for v in raw_subgraphs if v.seed],
        "idx": idx
    }

def run_one_variant(
    name: str,
    raw_subgraphs: List[View],
    idx: Any,
    universe: Set[str],
    universe_reports: Set[str],
    budget_mode: str,
    k_budget: int,
    cost_budget: float,
    merge_sim_thresh: float,
    do_merge: bool,
    gain_only: bool,
    objective: str,
    w_report: float,
    w_entity: float,
    lam: float,
    max_nodes_after_merge: int,
    max_reports_after_merge_ratio: float,
    max_universe_after_merge_ratio: float,
    min_gain: float,
    red_eps: float,
    curve_points: int,
    pool_pairs: int,
    rng: random.Random
) -> Dict[str, Any]:


    params = ECFAParams(
        merge_threshold=merge_sim_thresh,
        lam=lam,
        objective=objective,
        w_report=w_report,
        w_clue=w_entity,
        min_gain=min_gain,
        gain_only=bool(gain_only),
        max_nodes_after_merge=max_nodes_after_merge,
        max_reports_after_merge=int(len(universe_reports) * max_reports_after_merge_ratio),
        max_universe_after_merge=int(len(universe) * max_universe_after_merge_ratio),
        reinduce_from_union_evidence=True
    )


    if do_merge:
        candidates = compress_candidates(idx, params, raw_subgraphs, universe_report_ids=universe_reports)
    else:
        candidates = list(raw_subgraphs)


    if budget_mode == "count":


        effective_budget = float("inf")
    else:
        effective_budget = cost_budget

    selection_res = select_views(params, candidates, budget=effective_budget, universe_report_ids=universe_reports)

    selected_views = selection_res.selected


    if budget_mode == "count":
        selected_views = selected_views[:k_budget]

        total_cost_selected = sum(v.cost for v in selected_views)
    elif budget_mode == "cost":

         total_cost_selected = selection_res.total_cost
    else:
        total_cost_selected = selection_res.total_cost


    pool_redundancy_clue = calculate_pool_redundancy(
        candidates, eps=red_eps, max_pairs=pool_pairs, rng=rng, mode="clue"
    )
    pool_redundancy_node = calculate_pool_redundancy(
        candidates, eps=red_eps, max_pairs=pool_pairs, rng=rng, mode="node"
    )


    sel_redundancy_clue = _pairwise_redundancy_stats(
        selected_views, lambda v: v.covered_universe, eps=red_eps, max_pairs=None
    )
    sel_redundancy_report = _pairwise_redundancy_stats(
        selected_views, lambda v: v.covered_report_ids, eps=red_eps, max_pairs=None
    )

    curve_rep, nauc_rep, final_cov_rep = calculate_curve_and_metrics(
        selected_views,
        total_budget=cost_budget if cost_budget > 0 else total_cost_selected,
        curve_points=curve_points,
        universe_items=universe_reports,
        item_getter=lambda v: v.report_ids
    )

    curve_clue, nauc_clue, final_cov_clue = calculate_curve_and_metrics(
        selected_views,
        total_budget=cost_budget if cost_budget > 0 else total_cost_selected,
        curve_points=curve_points,
        universe_items=universe,
        item_getter=lambda v: v.covered_universe
    )

    eff_rep = final_cov_rep / total_cost_selected if total_cost_selected > 1e-9 else 0.0
    eff_clue = final_cov_clue / total_cost_selected if total_cost_selected > 1e-9 else 0.0
    avg_cost_per_view = (total_cost_selected / len(selected_views)) if selected_views else 0.0

    return {

        "n_candidates": len(candidates),
        "n_selected": len(selected_views),
        "total_cost_selected": total_cost_selected,
        "pool_redundancy_entity_j": pool_redundancy_clue,
        "pool_redundancy_node_j": pool_redundancy_node,
        "sel_redundancy_clue_j": sel_redundancy_clue,
        "sel_redundancy_report_j": sel_redundancy_report,
        "avg_cost_per_view": avg_cost_per_view,
        "curve_report": curve_rep,
        "curve_clue": curve_clue,
        "Coverage@Budget (Report)": final_cov_rep,
        "nAUC@Budget (Report)": nauc_rep,
        "Coverage@Budget (Clue)": final_cov_clue,
        "nAUC@Budget (Clue)": nauc_clue,
        "Eff_CovR_perCost": eff_rep,
        "Eff_CovClue_perCost": eff_clue,
        "budget_protocol": f"{budget_mode}@{k_budget if budget_mode=='count' else cost_budget}",
        "B_cost": cost_budget if budget_mode == "cost" else total_cost_selected
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    rng = random.Random(int(args.rand_seed))

    input_root = Path(args.input_root).resolve()
    dataset_xlsx = Path(args.dataset_xlsx).resolve()
    output_xlsx = Path(args.output_xlsx).resolve()
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root.as_posix()}")
    if not dataset_xlsx.exists():
        raise FileNotFoundError(f"dataset_xlsx not found: {dataset_xlsx.as_posix()}")

    if str(args.apps).strip():
        apps = [a.strip() for a in str(args.apps).split(",") if a.strip()]
    else:
        apps = _list_apps_by_sheets(dataset_xlsx, include_deprecated=bool(args.include_deprecated))

    seed_types = _parse_type_list(DEFAULTS.get("seed_types", "ISSUE,PHEN,DIAG"))
    universe_types = _parse_type_list(DEFAULTS.get("universe_types", "ISSUE,PHEN,DIAG"))
    strict_r_max = not bool(args.soft_r_max)
    backfill = not bool(args.no_backfill)

    print(f"[Start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Apps] {len(apps)} objective={args.objective} budget_mode={args.budget_mode} K={args.K} red_eps={args.red_eps}")
    print(f"[Types] seed_types={sorted(seed_types)} universe_types={sorted(universe_types)} backfill={backfill}")


    lam = float(args.beta)

    methods = [
        ("ECFA (Full)", dict(do_merge=True, gain_only=False)),
        ("ECFA w/o Merge", dict(do_merge=False, gain_only=False)),
        ("ECFA w/o Cost", dict(do_merge=True, gain_only=True)),


    ]


    per_app_pool_rows: List[dict] = []
    per_app_selected_rows: List[dict] = []
    curve_rows: List[dict] = []

    inferred_cost_budget_by_app: Dict[str, float] = {}

    for app in apps:
        app_dir = input_root / app / "unify"
        nodes_path = app_dir / "nodes.xlsx"
        edges_path = app_dir / "edges.xlsx"

        if not nodes_path.exists() or not edges_path.exists():
            print(f"[Skip] {app}: missing unify files")
            continue

        try:
            provider = XlsxEvidenceGraphProvider(nodes_path=nodes_path, edges_path=edges_path)
            nodes_df = provider.load_nodes()


            if "type" in nodes_df.columns:
                nodes_df["type"] = nodes_df["type"].apply(lambda x: TYPE_MAPPING.get(str(x).strip(), str(x).strip()))


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
                print(f"[{app}] Using source_row_index as report_ids for nodes")
                nodes_df["report_ids"] = nodes_df["source_row_index"]

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

                 edges_df["report_ids"] = edges_df["source_row_index"]

            defects_df = None
            xl = pd.ExcelFile(dataset_xlsx)
            if bool(args.use_defects) and (app in xl.sheet_names):
                defects_df = load_defects_optional(dataset_xlsx, sheet_name=app)


            raw_pack = build_raw_candidates(
                nodes_df=nodes_df,
                edges_df=edges_df,
                defects_df=defects_df,
                seed_types=seed_types,
                universe_types=universe_types,
                k=int(args.k),
                r_max=int(args.r_max),
                strict_r_max=bool(strict_r_max),
                lam=lam,
                backfill_node_reports_from_edges=bool(backfill),
            )

            n_universe_nodes = int(len(raw_pack["universe"]))
            n_universe_reports = int(len(raw_pack["universe_reports"]))
            truncated_seed_ratio = (
                float(raw_pack["maybe_truncated_cnt"]) / float(len(raw_pack["seeds"])) if raw_pack["seeds"] else 0.0
            )


            inferred_cost_budget = float(args.cost_budget)
            if str(args.budget_mode) == "cost" and inferred_cost_budget <= 0:

                tmp = run_one_variant(
                    name="tmp_full_for_budget",
                    raw_subgraphs=raw_pack["raw_subgraphs"],
                    idx=raw_pack["idx"],
                    universe=raw_pack["universe"],
                    universe_reports=raw_pack["universe_reports"],
                    budget_mode="count",
                    k_budget=int(args.K),
                    cost_budget=0.0,
                    merge_sim_thresh=float(args.merge_sim_thresh),
                    do_merge=True,
                    gain_only=False,
                    objective=str(args.objective),
                    w_report=float(args.w_report),
                    w_entity=float(args.w_entity),
                    lam=lam,
                    max_nodes_after_merge=int(args.max_nodes_after_merge),
                    max_reports_after_merge_ratio=float(args.max_reports_after_merge_ratio),
                    max_universe_after_merge_ratio=float(args.max_universe_after_merge_ratio),
                    min_gain=float(args.min_gain),
                    red_eps=float(args.red_eps),
                    curve_points=int(args.curve_points),
                    pool_pairs=int(args.pool_pairs),
                    rng=rng,
                )
                inferred_cost_budget = float(tmp["total_cost_selected"])
                print(f"[Budget] {app}: inferred B_cost from Full(count@{args.K}) = {inferred_cost_budget:.6f}")

            inferred_cost_budget_by_app[str(app)] = float(inferred_cost_budget)


            for method_name, cfg in methods:


                out = run_one_variant(
                    name=method_name,
                    raw_subgraphs=raw_pack["raw_subgraphs"],
                    idx=raw_pack["idx"],
                    universe=raw_pack["universe"],
                    universe_reports=raw_pack["universe_reports"],
                    budget_mode=str(args.budget_mode),
                    k_budget=int(args.K),
                    cost_budget=float(inferred_cost_budget),
                    merge_sim_thresh=float(args.merge_sim_thresh),
                    do_merge=bool(cfg["do_merge"]),
                    gain_only=bool(cfg["gain_only"]),
                    objective=str(args.objective),
                    w_report=float(args.w_report),
                    w_entity=float(args.w_entity),
                    lam=lam,
                    max_nodes_after_merge=int(args.max_nodes_after_merge),
                    max_reports_after_merge_ratio=float(args.max_reports_after_merge_ratio),
                    max_universe_after_merge_ratio=float(args.max_universe_after_merge_ratio),
                    min_gain=float(args.min_gain),
                    red_eps=float(args.red_eps),
                    curve_points=int(args.curve_points),
                    pool_pairs=int(args.pool_pairs),
                    rng=rng,
                )

                prj = out["pool_redundancy_entity_j"]

                per_app_pool_rows.append({
                    "App": app,
                    "Method": method_name,
                    "merge_sim_thresh": float(args.merge_sim_thresh),
                    "red_eps": float(args.red_eps),
                    "pool_pairs_used": int(prj["n_pairs"]),
                    "#Candidates": int(out["n_candidates"]),
                    "Pool_AvgOverlap(J incl 0)": float(prj["avg"]),
                    "Pool_MaxOverlap(J)": float(prj["max"]),
                    "Pool_AvgOverlap_nz(J>eps)": float(prj["avg_pos"]),
                    "Pool_Frac(J>eps)": float(prj["frac_pos"]),
                    "Pool_OvDeg(J>eps)": float(prj["ov_deg"]),
                    "#PosPairs(J>eps)": int(prj["n_pos_pairs"]),

                    "PoolNode_AvgOverlap(J incl 0)": float(out["pool_redundancy_node_j"]["avg"]),
                    "PoolNode_MaxOverlap(J)": float(out["pool_redundancy_node_j"]["max"]),
                    "PoolNode_AvgOverlap_nz(J>eps)": float(out["pool_redundancy_node_j"]["avg_pos"]),
                    "PoolNode_Frac(J>eps)": float(out["pool_redundancy_node_j"]["frac_pos"]),
                    "n_universe_reports": int(n_universe_reports),
                    "n_universe_nodes": int(n_universe_nodes),
                    "truncated_seed_ratio": float(truncated_seed_ratio),
                })

                per_app_selected_rows.append({
                    "App": app,
                    "Method": method_name,
                    "budget_mode": str(args.budget_mode),
                    "budget_protocol": out["budget_protocol"],
                    "B_cost": float(out["B_cost"]),
                    "K": int(args.K),
                    "#Candidates": int(out["n_candidates"]),
                    "#Selected": int(out["n_selected"]),
                    "TotalCostSelected": float(out["total_cost_selected"]),
                    "Coverage@Budget (Report)": float(out["Coverage@Budget (Report)"]),
                    "nAUC@Budget (Report)": float(out["nAUC@Budget (Report)"]),
                    "Coverage@Budget (Clue)": float(out["Coverage@Budget (Clue)"]),
                    "nAUC@Budget (Clue)": float(out["nAUC@Budget (Clue)"]),
                    "Eff_CovR_perCost": float(out["Eff_CovR_perCost"]),
                    "Eff_CovClue_perCost": float(out["Eff_CovClue_perCost"]),
                    "AvgCostPerView": float(out["avg_cost_per_view"]),

                    "SelClueFrac(J>eps)": float(out["sel_redundancy_clue_j"]["frac_pos"]),
                    "SelClueAvgOverlap_nz(J>eps)": float(out["sel_redundancy_clue_j"]["avg_pos"]),
                    "SelClueMaxOverlap(J)": float(out["sel_redundancy_clue_j"]["max"]),
                    "SelClueOvDeg(J>eps)": float(out["sel_redundancy_clue_j"]["ov_deg"]),

                    "SelReportFrac(J>eps)": float(out["sel_redundancy_report_j"]["frac_pos"]),
                    "SelReportAvgOverlap_nz(J>eps)": float(out["sel_redundancy_report_j"]["avg_pos"]),
                    "SelReportMaxOverlap(J)": float(out["sel_redundancy_report_j"]["max"]),
                    "SelReportOvDeg(J>eps)": float(out["sel_redundancy_report_j"]["ov_deg"]),
                })

                c_rep = out["curve_report"]
                c_clue = out["curve_clue"]
                xs = c_rep["x"]
                yr = c_rep["y"]
                yc = c_clue["y"]
                for i in range(len(xs)):
                    curve_rows.append({
                        "App": app,
                        "Method": method_name,
                        "x_cost_norm": float(xs[i]),
                        "CoverageCurve(Report)": float(yr[i]),
                        "CoverageCurve(Clue)": float(yc[i]),
                    })

            print(f"[OK] {app}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Fail] {app}: {type(e).__name__}: {e}")

    df_pool = pd.DataFrame(per_app_pool_rows)
    if df_pool.empty:
        raise RuntimeError("No results produced (df_pool is empty).")

    df_sel = pd.DataFrame(per_app_selected_rows)
    df_curve = pd.DataFrame(curve_rows)

    def summarize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        rows = []
        method_names = df["Method"].unique()
        for method_name in method_names:
            sub = df[df["Method"] == method_name]
            row = {"Method": method_name, "Apps_N": int(len(sub))}
            for c in cols:
                xs = sub[c].tolist()
                row[c] = _fmt(_mean(xs), _std(xs), digits=4)
            rows.append(row)
        return pd.DataFrame(rows)

    df_main = summarize(
        df_pool,
        [
            "#Candidates",
            "Pool_Frac(J>eps)",
            "Pool_AvgOverlap_nz(J>eps)",
            "Pool_MaxOverlap(J)",
            "Pool_OvDeg(J>eps)",
        ],
    ).rename(columns={
        "#Candidates": "#Candidates ↓",
        "Pool_Frac(J>eps)": "PoolFrac(J>ε) ↓",
        "Pool_AvgOverlap_nz(J>eps)": "PoolAvgOverlap_nz(J>ε) ↓",
        "Pool_MaxOverlap(J)": "PoolMaxOverlap ↓",
        "Pool_OvDeg(J>eps)": "PoolOvDeg(J>ε) ↓",
    })

    df_appendix_sel = summarize(
        df_sel,
        [
            "Coverage@Budget (Report)",
            "nAUC@Budget (Report)",
            "Coverage@Budget (Clue)",
            "nAUC@Budget (Clue)",
            "#Selected",
            "AvgCostPerView",

            "SelClueFrac(J>eps)",
            "SelClueAvgOverlap_nz(J>eps)",
            "SelClueMaxOverlap(J)",
            "SelClueOvDeg(J>eps)",

            "SelReportFrac(J>eps)",
            "SelReportAvgOverlap_nz(J>eps)",
            "SelReportMaxOverlap(J)",
            "SelReportOvDeg(J>eps)",

            "#Candidates",
            "TotalCostSelected",
            "Eff_CovR_perCost",
            "Eff_CovClue_perCost",
        ],
    ).rename(columns={
        "Coverage@Budget (Report)": "Cov(Rep)@B ↑",
        "nAUC@Budget (Report)": "nAUC(Rep)@B ↑",
        "Coverage@Budget (Clue)": "Cov(Clue)@B ↑",
        "nAUC@Budget (Clue)": "nAUC(Clue)@B ↑",
        "#Selected": "#Selected@Budget ↑",
        "AvgCostPerView": "AvgCost/View ↓",

        "SelClueFrac(J>eps)": "SelClueFrac(J>ε) ↓",
        "SelClueAvgOverlap_nz(J>eps)": "SelClueAvgOverlap_nz(J>ε) ↓",
        "SelClueMaxOverlap(J)": "SelClueMaxOverlap ↓",
        "SelClueOvDeg(J>eps)": "SelClueOvDeg(J>ε) ↓",

        "SelReportFrac(J>eps)": "SelRepFrac(J>ε) ↓",
        "SelReportAvgOverlap_nz(J>eps)": "SelRepAvgOverlap_nz(J>ε) ↓",
        "SelReportMaxOverlap(J)": "SelRepMaxOverlap ↓",
        "SelReportOvDeg(J>eps)": "SelRepOvDeg(J>ε) ↓",

        "#Candidates": "#Candidates ↓",
        "TotalCostSelected": "TotalCostSelected",
        "Eff_CovR_perCost": "Eff(CovR/Cost) ↑",
        "Eff_CovClue_perCost": "Eff(CovClue/Cost) ↑",
    })

    df_meta = pd.DataFrame([{
        "budget_mode": str(args.budget_mode),
        "K": int(args.K),
        "merge_sim_thresh": float(args.merge_sim_thresh),
        "objective": str(args.objective),
        "w_report": float(args.w_report),
        "w_entity": float(args.w_entity),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "lam": lam,
        "k_expand": int(args.k),
        "r_max": int(args.r_max),
        "strict_r_max": bool(strict_r_max),
        "red_eps": float(args.red_eps),
        "curve_points": int(args.curve_points),
        "pool_pairs": int(args.pool_pairs),
        "rand_seed": int(args.rand_seed),
        "backfill_node_reports_from_edges": bool(backfill),
        "input_root": str(input_root),
        "dataset_xlsx": str(dataset_xlsx),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "apps_N": int(df_pool["App"].nunique()),
    }])


    df_paper = df_main.merge(df_appendix_sel, on=["Method", "Apps_N"], how="left")
    preferred_cols = [
        "Method", "Apps_N",
        "#Candidates ↓",
        "PoolFrac(J>ε) ↓", "PoolAvgOverlap_nz(J>ε) ↓", "PoolMaxOverlap ↓", "PoolOvDeg(J>ε) ↓",
        "Cov(Clue)@B ↑", "nAUC(Clue)@B ↑",
        "Cov(Rep)@B ↑", "nAUC(Rep)@B ↑",
        "#Selected@Budget ↑", "AvgCost/View ↓",
        "SelClueFrac(J>ε) ↓", "SelClueAvgOverlap_nz(J>ε) ↓", "SelClueMaxOverlap ↓", "SelClueOvDeg(J>ε) ↓",
        "SelRepFrac(J>ε) ↓", "SelRepAvgOverlap_nz(J>ε) ↓", "SelRepMaxOverlap ↓", "SelRepOvDeg(J>ε) ↓",
        "TotalCostSelected", "Eff(CovClue/Cost) ↑",
    ]
    cols = [c for c in preferred_cols if c in df_paper.columns]
    df_paper = df_paper[cols]


    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_xlsx) as xw:
        df_main.to_excel(xw, sheet_name="RQ2_MainTable", index=False)
        df_paper.to_excel(xw, sheet_name="RQ2_PaperTable", index=False)
        df_pool.to_excel(xw, sheet_name="RQ2_PoolRedundancy_PerApp", index=False)
        df_appendix_sel.to_excel(xw, sheet_name="RQ2_Appendix_Selected_Summary", index=False)
        df_sel.to_excel(xw, sheet_name="RQ2_Appendix_Selected_PerApp", index=False)
        df_curve.to_excel(xw, sheet_name="RQ2_CostCurve_Long", index=False)
        df_meta.to_excel(xw, sheet_name="Meta", index=False)

    print(f"[Done] -> {output_xlsx.as_posix()}")
    return inferred_cost_budget_by_app

def main():
    default_input_root = r"ROOT_glm4.7"
    default_dataset_xlsx = r"RQ2根数据集.xlsx"
    default_output_xlsx = r"out\rq2_publishable_poolMain_Patched.xlsx"

    ap = argparse.ArgumentParser(description="Batch XLSX generator (Replication)")
    ap.add_argument("--input_root", default=str(default_input_root))
    ap.add_argument("--dataset_xlsx", default=str(default_dataset_xlsx))
    ap.add_argument("--output_xlsx", default=str(default_output_xlsx))
    ap.add_argument("--apps", default="", help="Comma-separated APP names; default: all sheets.")
    ap.add_argument("--include_deprecated", action="store_true")
    ap.add_argument("--use_defects", action="store_true")

    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--r_max", type=int, default=200)
    ap.add_argument("--soft_r_max", action="store_true")
    ap.add_argument("--no_backfill", action="store_true")

    ap.add_argument("--merge_sim_thresh", type=float, default=0.6)
    ap.add_argument("--budget_mode", choices=["count", "cost"], default="cost")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--cost_budget", type=float, default=0.0)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=0.01)

    ap.add_argument("--objective", choices=["report", "entity", "hybrid"], default="entity")
    ap.add_argument("--w_report", type=float, default=1.0)
    ap.add_argument("--w_entity", type=float, default=1.0)

    ap.add_argument("--max_nodes_after_merge", type=int, default=800)
    ap.add_argument("--max_reports_after_merge_ratio", type=float, default=0.32)
    ap.add_argument("--max_universe_after_merge_ratio", type=float, default=1.0)
    ap.add_argument("--min_gain", type=float, default=1e-9)
    ap.add_argument("--red_eps", type=float, default=0.01)
    ap.add_argument("--curve_points", type=int, default=100)
    ap.add_argument("--pool_pairs", type=int, default=10000)
    ap.add_argument("--rand_seed", type=int, default=42)


    ap.add_argument("--detail_dir", default="", help="Ignored in this version")

    args = ap.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main()
