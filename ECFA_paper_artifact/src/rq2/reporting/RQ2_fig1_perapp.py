from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})


METHODS = [
    "ECFA",
    "ECFA w/o cost",
    "ECFA (No Merge)",
    "Random-k",
]

LABEL = {
    "ECFA": "ECFA (Full)",
    "ECFA w/o cost": "ECFA (No Cost)",
    "ECFA (No Merge)": "ECFA (No Merge)",
    "Random-k": "Random-k",
}

APP_NAME_MAP = {
    "Slife": "Slife",
    "jayme": "JayMe",
    "华为运动": "Huawei Health",
    "沪江英语": "Hujiang English",
    "途牛": "Tuniu",
    "雅思听力": "IELTS Listening",
}

STYLE = {
    "ECFA":            dict(color="red",    ls="-",  marker="o"),
    "ECFA w/o cost":   dict(color="orange", ls="-",  marker="x"),
    "ECFA (No Merge)": dict(color="purple", ls="-",  marker="s"),
    "Random-k":        dict(color="0.5",    ls="--", marker=None),
}

TRUNC_MARKER = dict(marker="v", color="black")
FADE_ALPHA = 0.13
LINE_W = 2.8
POST_LW = LINE_W * 0.85
MARKER_SZ = 7.2
RAND_SHADE_ALPHA = 0.18

RANDOM_TRIALS_TO_USE = 30


def style_axes(ax):
    ax.grid(True, alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

def app_name_from_filename(fp: Path) -> str:
    m = re.match(r"rq2_report_(.+)\.xlsx$", fp.name)
    return m.group(1) if m else fp.stem

def pick_xcol(df: pd.DataFrame) -> str:
    candidates = ["countk", "k", "step", "t", "iter", "rank"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find count-k column. Available: {list(df.columns)}")

def pick_ycol(df: pd.DataFrame, metric: str) -> str:
    if metric == "report":
        for c in ["cov_report", "coverage_report", "report_coverage", "covR"]:
            if c in df.columns:
                return c
    else:
        for c in ["cov_entity", "coverage_entity", "entity_coverage", "covE"]:
            if c in df.columns:
                return c
    raise ValueError(f"Cannot find coverage column for metric={metric}. Available: {list(df.columns)}")

def safe_read_meta_budget_cost(meta: pd.DataFrame) -> Optional[float]:
    if meta is None or meta.empty:
        return None
    if "key" not in meta.columns or "value" not in meta.columns:
        return None
    for k in ["B_cost", "budget_cost", "B_cost_shared", "B_cost_selected", "B_cost_ref"]:
        hit = meta.loc[meta["key"].astype(str) == k, "value"]
        if len(hit) > 0:
            try:
                return float(hit.iloc[0])
            except Exception:
                pass
    return None

def safe_read_nsel(portfolio: pd.DataFrame, method: str) -> Optional[int]:
    if portfolio is None or portfolio.empty or "Method" not in portfolio.columns:
        return None
    nsel_col = None
    for c in ["nSel", "n_selected", "Selected", "NumSelected"]:
        if c in portfolio.columns:
            nsel_col = c
            break
    if nsel_col is None:
        return None
    row = portfolio[portfolio["Method"].astype(str) == method]
    if row.empty:
        return None
    try:
        v = row.iloc[0][nsel_col]
        nsel = int(round(float(v)))
        return nsel if nsel > 0 else None
    except Exception:
        return None

def find_budget_end_index_cost(
    sub_cost: pd.DataFrame,
    *,
    xcol_cost: str,
    nSel: Optional[int],
    B_cost: Optional[float],
) -> int:


    sub_cost = sub_cost.sort_values(xcol_cost).reset_index(drop=True)
    last = int(len(sub_cost) - 1)

    if nSel is not None and nSel >= 1:
        return int(min(nSel - 1, last))

    if "rho" in sub_cost.columns:
        rho = sub_cost["rho"].astype(float).to_numpy()
        hit = np.where(rho >= 1.0 - 1e-12)[0]
        if len(hit) > 0:
            return int(hit[0])

    cum_candidates = ["cum_cost", "cost_cum", "cumCost", "TotalCostSoFar", "cost_sum"]
    cum_col = next((c for c in cum_candidates if c in sub_cost.columns), None)
    if cum_col is not None and B_cost is not None:
        cc = sub_cost[cum_col].astype(float).to_numpy()
        hit = np.where(cc >= float(B_cost) - 1e-12)[0]
        if len(hit) > 0:
            return int(hit[0])

    return last

def _load_curve_sheets(xlsx_path: Path) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    xl = pd.ExcelFile(xlsx_path)
    curve = xl.parse("CostCurve_Long") if "CostCurve_Long" in xl.sheet_names else None
    meta = xl.parse("meta") if "meta" in xl.sheet_names else None
    port = xl.parse("Portfolio") if "Portfolio" in xl.sheet_names else None
    if curve is None:
        raise ValueError(f"{xlsx_path.name} missing sheet: CostCurve_Long")
    return curve, meta, port

def _trial_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["trial", "seed", "run", "repeat", "rep", "trial_id"]:
        if c in df.columns:
            return c
    return None

def _split_trials_by_method_prefix(df: pd.DataFrame, base_method: str) -> Dict[str, pd.DataFrame]:


    df = df.copy()
    df["Method"] = df["Method"].astype(str)

    tcol = _trial_col(df)
    if tcol is not None:
        sub = df[df["Method"] == base_method].copy()
        if sub.empty:
            return {}
        out = {}
        for k, g in sub.groupby(tcol):
            out[str(k)] = g.copy()
        return out


    sub = df[df["Method"].str.startswith(base_method)].copy()
    if sub.empty:
        return {}
    out = {}
    for mname, g in sub.groupby("Method"):
        out[str(mname)] = g.copy()
    return out

def _aggregate_random_curve(
    curve_cnt: pd.DataFrame,
    xcol_cnt: str,
    ycol_cnt: str,
    *,
    base_method: str = "Random-k",
    n_trials: int = 30,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:


    df = curve_cnt.copy()
    df["Method"] = df["Method"].astype(str)


    std_col = f"{ycol_cnt}_std"
    sub = df[df["Method"] == base_method].copy()
    if (not sub.empty) and (std_col in sub.columns):
        sub = sub.sort_values(xcol_cnt).reset_index(drop=True)
        x = sub[xcol_cnt].astype(int).to_numpy()
        y_mean = sub[ycol_cnt].astype(float).to_numpy()
        y_std  = sub[std_col].astype(float).to_numpy()

        y_mean = np.maximum.accumulate(y_mean)
        return x, y_mean, y_std


    trials = _split_trials_by_method_prefix(df, base_method)
    if not trials:
        return None

    keys = sorted(trials.keys())
    if len(keys) >= n_trials:
        keys = keys[:n_trials]

    series_list = []
    for k in keys:
        sub = trials[k].sort_values(xcol_cnt).reset_index(drop=True)
        x = sub[xcol_cnt].astype(int).to_numpy()
        y = sub[ycol_cnt].astype(float).to_numpy()
        y = np.maximum.accumulate(y)
        series_list.append(pd.Series(y, index=x, name=k))

    mat = pd.concat(series_list, axis=1).sort_index().astype(float)
    if mat.shape[1] <= 1:
        return None

    x_steps = mat.index.to_numpy(dtype=int)
    y_mean = mat.mean(axis=1).to_numpy(dtype=float)
    y_std  = mat.std(axis=1, ddof=0).to_numpy(dtype=float)
    return x_steps, y_mean, y_std


def _median_trunc_step_for_random(
    curve_cost: pd.DataFrame,
    *,
    xcol_cost: str,
    port_cost: Optional[pd.DataFrame],
    B_cost: Optional[float],
    base_method: str = "Random-k",
    n_trials: int = 30,
) -> Optional[int]:


    trials = _split_trials_by_method_prefix(curve_cost, base_method)
    if not trials:
        return None

    keys = sorted(trials.keys())
    if len(keys) >= n_trials:
        keys = keys[:n_trials]

    steps = []
    for k in keys:
        sub_cost = trials[k].sort_values(xcol_cost).reset_index(drop=True)

        nSel = None
        end_idx_cost = find_budget_end_index_cost(sub_cost, xcol_cost=xcol_cost, nSel=nSel, B_cost=B_cost)
        try:
            steps.append(int(sub_cost.loc[end_idx_cost, xcol_cost]))
        except Exception:
            continue

    if not steps:
        return None
    return int(np.median(np.array(steps, dtype=int)))


def plot_one_app_pair(
    fp_count: Path,
    fp_cost: Path,
    out_dir: Path,
    *,
    metric: str = "report",
    title_prefix: str = "",
):

    curve_cnt, _, _ = _load_curve_sheets(fp_count)
    curve_cnt["Method"] = curve_cnt["Method"].astype(str)
    xcol_cnt = pick_xcol(curve_cnt)
    ycol_cnt = pick_ycol(curve_cnt, metric)


    curve_cost, meta_cost, port_cost = _load_curve_sheets(fp_cost)
    curve_cost["Method"] = curve_cost["Method"].astype(str)
    xcol_cost = pick_xcol(curve_cost)
    B_cost = safe_read_meta_budget_cost(meta_cost) if meta_cost is not None else None

    app_raw = app_name_from_filename(fp_count)
    app = APP_NAME_MAP.get(app_raw, app_raw)

    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    style_axes(ax)


    rand_agg = _aggregate_random_curve(
        curve_cnt, xcol_cnt, ycol_cnt,
        base_method="Random-k",
        n_trials=RANDOM_TRIALS_TO_USE,
    )
    rand_trunc_step = _median_trunc_step_for_random(
        curve_cost,
        xcol_cost=xcol_cost,
        port_cost=port_cost,
        B_cost=B_cost,
        base_method="Random-k",
        n_trials=RANDOM_TRIALS_TO_USE,
    )

    if rand_agg is not None:
        x_rand, y_rand_mean, y_rand_std = rand_agg
        if len(y_rand_mean) > 0:

            y_rand_mean = np.clip(y_rand_mean, 0.0, 1.0)
    else:
        x_rand = y_rand_mean = y_rand_std = None


    plot_items = []

    for m in METHODS:

        if m == "Random-k" and rand_agg is not None:

            if rand_trunc_step is None:
                sub_cost = curve_cost[curve_cost["Method"].str.startswith("Random-k")].copy()
                if sub_cost.empty:
                    trunc_step_val = int(x_rand[-1])
                else:
                    sub_cost = sub_cost.sort_values(xcol_cost).reset_index(drop=True)
                    end_idx_cost = find_budget_end_index_cost(sub_cost, xcol_cost=xcol_cost, nSel=None, B_cost=B_cost)
                    trunc_step_val = int(sub_cost.loc[end_idx_cost, xcol_cost])
            else:
                trunc_step_val = int(rand_trunc_step)

            plot_items.append({
                "type": "random_agg",
                "method": m,
                "sort_key": trunc_step_val,
                "trunc_val": trunc_step_val,
                "data": (x_rand, y_rand_mean, y_rand_std)
            })
            continue


        sub_cnt = curve_cnt[curve_cnt["Method"] == m].copy()
        sub_cost = curve_cost[curve_cost["Method"] == m].copy()

        if sub_cnt.empty or sub_cost.empty:
            continue

        sub_cnt = sub_cnt.sort_values(xcol_cnt).reset_index(drop=True)
        sub_cost = sub_cost.sort_values(xcol_cost).reset_index(drop=True)

        x = sub_cnt[xcol_cnt].astype(int).to_numpy()
        y = sub_cnt[ycol_cnt].astype(float).to_numpy()
        y = np.maximum.accumulate(y)

        K_cnt = int(len(x))
        if K_cnt <= 0:
            continue

        nSel = safe_read_nsel(port_cost, m) if port_cost is not None else None
        end_idx_cost = find_budget_end_index_cost(sub_cost, xcol_cost=xcol_cost, nSel=nSel, B_cost=B_cost)
        trunc_step_val = int(sub_cost.loc[end_idx_cost, xcol_cost])

        plot_items.append({
            "type": "single",
            "method": m,
            "sort_key": trunc_step_val,
            "trunc_val": trunc_step_val,
            "data": (x, y)
        })

        if m == "Random-k":
             print(f"[Warn] {app}: Random-k has no multiple trials; plotted single curve (no shadow).")


    plot_items.sort(key=lambda item: item["sort_key"])

    method_handles = {}


    for item in plot_items:
        m = item["method"]
        trunc_val = item["trunc_val"]
        st = STYLE[m]
        color, ls = st["color"], st["ls"]

        if item["type"] == "random_agg":
            x, y_mean, y_std = item["data"]
            y_lo = np.maximum.accumulate(np.clip(y_mean - y_std, 0.0, 1.0))
            y_hi = np.maximum.accumulate(np.clip(y_mean + y_std, 0.0, 1.0))

            idx_candidates = np.where(x <= trunc_val)[0]
            end_idx = int(idx_candidates[-1]) if len(idx_candidates) else 0
            end_idx = max(0, min(end_idx, len(x) - 1))

            x_pre = x[: end_idx + 1]
            x_post = x[end_idx:]


            line = ax.plot(x_pre, y_mean[: end_idx + 1], color=color, linestyle=ls, linewidth=LINE_W, label=LABEL[m])[0]
            method_handles[m] = line


            ax.fill_between(
                x_pre, y_lo[: end_idx + 1], y_hi[: end_idx + 1],
                color=color, alpha=RAND_SHADE_ALPHA, linewidth=0,
            )

            if len(x_post) >= 2:
                ax.fill_between(
                    x_post, y_lo[end_idx:], y_hi[end_idx:],
                    color=color, alpha=RAND_SHADE_ALPHA * 0.3, linewidth=0,
                )

            ax.plot([x[end_idx]], [y_mean[end_idx]], linestyle="None",
                    marker=TRUNC_MARKER["marker"], color=TRUNC_MARKER["color"],
                    markersize=10.5, zorder=10)

            if len(x_post) >= 2:
                ax.plot(x_post, y_mean[end_idx:], color=color, linestyle=ls, linewidth=POST_LW, alpha=FADE_ALPHA)

        else:
            x, y = item["data"]
            idx_candidates = np.where(x <= trunc_val)[0]
            end_idx = int(idx_candidates[-1]) if len(idx_candidates) else 0
            end_idx = max(0, min(end_idx, len(x) - 1))

            x_pre, y_pre = x[: end_idx + 1], y[: end_idx + 1]
            x_post, y_post = x[end_idx:], y[end_idx:]

            mk = st.get("marker", None)
            line = ax.plot(
                x_pre, y_pre,
                color=color, linestyle=ls, linewidth=LINE_W,
                marker=mk, markersize=MARKER_SZ if mk else 0,
                markevery=2,
                markeredgecolor="black", markeredgewidth=0.6,
                label=LABEL.get(m, m),
            )[0]
            method_handles[m] = line


            ax.plot([x[end_idx]], [y[end_idx]], linestyle="None",
                    marker=TRUNC_MARKER["marker"], color=TRUNC_MARKER["color"],
                    markersize=10.5, zorder=10)

            if len(x_post) >= 2:
                ax.plot(x_post, y_post, color=color, linestyle=ls, linewidth=POST_LW, alpha=FADE_ALPHA)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax.set_xlabel("Selection step (count-k)")
    ax.set_ylabel("Report Coverage" if metric == "report" else "Entity Coverage")

    ax.set_title(f"{title_prefix}{app}: {'Report' if metric=='report' else 'Entity'} coverage vs. selection step")


    handles = []
    labels = []
    for m in METHODS:
        if m in method_handles:
            handles.append(method_handles[m])
            labels.append(LABEL.get(m, m))

    trunc_handle = Line2D(
        [0], [0],
        marker=TRUNC_MARKER["marker"],
        color="none",
        markerfacecolor=TRUNC_MARKER["color"],
        markeredgecolor=TRUNC_MARKER["color"],
        markersize=10.5,
        linestyle="None",
        label="Budget exhausted (truncation)",
    )
    handles.append(trunc_handle)
    labels.append("Budget exhausted (truncation)")
    ax.legend(handles, labels, frameon=True, loc="lower right")

    out_svg = out_dir / f"{app}_CoverageAtK_{metric}.svg"
    out_pdf = out_dir / f"{app}_CoverageAtK_{metric}.pdf"

    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir_cost",
        default=r"C:\Users\27826\Desktop\workspace\0 20250520 知识图谱+聚类\TOSEM\实验相关\RQ2预算下子图选择机制是否更高覆盖效率、且冗余更低？\RQ2实验重构\RQ2fig_table_gen\data\result_cost",
        type=str,
        help="Folder containing cost-mode rq2_report_*.xlsx (for truncation)",
    )
    ap.add_argument(
        "--in_dir_count",
        default=r"C:\Users\27826\Desktop\workspace\0 20250520 知识图谱+聚类\TOSEM\实验相关\RQ2预算下子图选择机制是否更高覆盖效率、且冗余更低？\RQ2实验重构\RQ2fig_table_gen\data\result_count",
        type=str,
        help="Folder containing count-mode rq2_report_*.xlsx (for full growth curves)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="out/fig1_perapp",
        help="Output folder (default: out/fig1_perapp)",
    )
    ap.add_argument("--metric", default="report", choices=["entity", "report"])
    ap.add_argument("--title_prefix", type=str, default="")
    args = ap.parse_args()

    in_cost = Path(args.in_dir_cost).resolve()
    in_cnt = Path(args.in_dir_count).resolve()
    if not in_cost.exists():
        raise FileNotFoundError(f"in_dir_cost not found: {in_cost}")
    if not in_cnt.exists():
        raise FileNotFoundError(f"in_dir_count not found: {in_cnt}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files_cnt = sorted(in_cnt.glob("rq2_report_*.xlsx"))
    if not files_cnt:
        raise FileNotFoundError(f"No rq2_report_*.xlsx found in {in_cnt}")

    missing_cost = 0
    for fp_cnt in files_cnt:
        fp_cost = in_cost / fp_cnt.name
        if not fp_cost.exists():
            print(f"[Warn] cost file missing for {fp_cnt.name}: {fp_cost}")
            missing_cost += 1
            continue
        plot_one_app_pair(fp_cnt, fp_cost, out_dir, metric=args.metric, title_prefix=args.title_prefix)

    print(f"[Done] Saved per-app k-plots to: {out_dir}")
    if missing_cost:
        print(f"[Warn] {missing_cost} apps skipped due to missing cost files.")

if __name__ == "__main__":
    main()
