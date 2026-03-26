

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


METHOD_ORDER = [
    "ECFA",
    "ECFA w/o cost",
    "ECFA (No Merge)",
    "Random-k",
]

METHOD_LABEL = {
    "ECFA": "ECFA",
    "ECFA w/o cost": "w/o cost",
    "ECFA (No Merge)": "w/o merge",
    "Random-k": "Random",
}


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def app_name_from_filename(fp: Path) -> str:
    m = re.match(r"rq2_report_(.+)\.xlsx$", fp.name)
    return m.group(1) if m else fp.stem


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_fig(fig, out_png: Path, out_pdf: Optional[Path] = None, dpi: int = 300) -> None:
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    out_svg = out_png.with_suffix(".svg")
    fig.savefig(out_svg)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def normalize_method_name(s: str) -> str:
    s0 = str(s).strip()
    sl = s0.lower()

    if sl in {"ecfa (full)", "ecfa full"}:
        return "ECFA"
    if s0 == "ECFA":
        return "ECFA"

    if sl.startswith("ecfa"):
        if ("w/o" in sl and "cost" in sl) or ("no cost" in sl):
            return "ECFA w/o cost"
        if ("no merge" in sl) or ("w/o merge" in sl):
            return "ECFA (No Merge)"
        return "ECFA"

    if sl.startswith("random"):
        return "Random-k"


    if sl.startswith("topunisize"):
        return "TopUniSize-k"
    if sl.startswith("toprawsize"):
        return "TopRawSize-k"

    return s0


def first_existing_sheet(xl: pd.ExcelFile, candidates: List[str]) -> Optional[str]:
    for s in candidates:
        if s in xl.sheet_names:
            return s
    return None


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {str(k).strip().lower(): k for k in cols}
    for c in candidates:
        key = str(c).strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def load_selected_views(fp: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(fp)
    sheet = first_existing_sheet(xl, ["SelectedViews_Stats_AllMethods", "SelectedViews_Stats"])
    if not sheet:
        raise ValueError(f"{fp.name} missing selected-views sheet")

    df = xl.parse(sheet)

    col_m = pick_col(df, ["Method"])
    col_c = pick_col(df, ["cost", "Cost", "view_cost", "ViewCost"])
    if col_m is None or col_c is None:
        raise ValueError(f"{fp.name}:{sheet} must contain Method & cost columns. Got={list(df.columns)}")

    out = df[[col_m, col_c]].rename(columns={col_m: "Method", col_c: "cost"}).copy()
    out["Method"] = out["Method"].astype(str).map(normalize_method_name)
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")


    out.loc[out["cost"] <= 0, "cost"] = np.nan
    out = out.dropna(subset=["cost"])


    out = out[out["Method"].isin(METHOD_ORDER)].copy()
    return out


def load_nsel_from_portfolio_or_red(fp: Path) -> pd.DataFrame:


    xl = pd.ExcelFile(fp)


    sheet_p = first_existing_sheet(xl, ["Portfolio"])
    if sheet_p:
        dfp = xl.parse(sheet_p)
        col_m = pick_col(dfp, ["Method"])
        col_n = pick_col(dfp, ["nSel", "n_selected", "Selected", "#Selected", "K"])
        if col_m is not None and col_n is not None:
            out = dfp[[col_m, col_n]].rename(columns={col_m: "Method", col_n: "nSel"}).copy()
            out["Method"] = out["Method"].astype(str).map(normalize_method_name)
            out["nSel"] = pd.to_numeric(out["nSel"], errors="coerce")
            out = out.dropna(subset=["nSel"])
            out = out[out["Method"].isin(METHOD_ORDER)].copy()
            return out


    sheet_r = first_existing_sheet(xl, ["Redundancy_Report", "Redundancy"])
    if sheet_r:
        dfr = xl.parse(sheet_r)
        col_m = pick_col(dfr, ["Method"])
        col_k = pick_col(dfr, ["K", "nSel", "#Selected", "Selected"])
        if col_m is not None and col_k is not None:
            out = dfr[[col_m, col_k]].rename(columns={col_m: "Method", col_k: "nSel"}).copy()
            out["Method"] = out["Method"].astype(str).map(normalize_method_name)
            out["nSel"] = pd.to_numeric(out["nSel"], errors="coerce")
            out = out.dropna(subset=["nSel"])
            out = out[out["Method"].isin(METHOD_ORDER)].copy()
            return out

    return pd.DataFrame(columns=["Method", "nSel"])


def app_balanced_sample(df: pd.DataFrame, cap_per_app: int, seed: int) -> pd.DataFrame:


    rng = np.random.default_rng(seed)
    parts = []
    for (app, method), g in df.groupby(["App", "Method"], dropna=False):
        if len(g) <= cap_per_app:
            parts.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=cap_per_app, replace=False)
            parts.append(g.loc[idx])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def compute_tick_nsel(
    nsel_table: pd.DataFrame,
    selected_views: pd.DataFrame,
    methods: List[str],
    *,
    agg: str = "median",
) -> Dict[str, float]:


    assert agg in {"median", "mean", "total"}
    out: Dict[str, float] = {}

    for m in methods:
        vals: np.ndarray
        sub = nsel_table[nsel_table["Method"] == m] if not nsel_table.empty else pd.DataFrame()
        if not sub.empty:
            vals = sub["nSel"].to_numpy(dtype=float)
        else:
            sub2 = selected_views[selected_views["Method"] == m]
            if sub2.empty:
                out[m] = float("nan")
                continue
            vals = sub2.groupby("App")["cost"].size().to_numpy(dtype=float)

        if agg == "total":
            out[m] = float(np.nansum(vals))
        elif agg == "mean":
            out[m] = float(np.nanmean(vals))
        else:
            out[m] = float(np.nanmedian(vals))
    return out


def format_nsel(n: float, digits: int = 1) -> str:
    if np.isnan(n):
        return "—"
    return f"{n:.{digits}f}"


def draw_box_and_points(
    ax,
    df: pd.DataFrame,
    methods: List[str],
    nsel_map: Dict[str, float],
    *,
    ylabel: str,
    log_scale: bool,
    point_jitter: float,
    seed: int,
    nsel_digits: int,
    show_box: bool = True,
    rasterize_points: bool = True,
) -> None:
    xs = np.arange(1, len(methods) + 1, dtype=float)

    data = []
    for m in methods:
        vals = df.loc[df["Method"] == m, "cost"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        data.append(vals if len(vals) else np.array([np.nan], dtype=float))


    if show_box:
        ax.boxplot(
            data,
            positions=xs,
            widths=0.38,
            showfliers=False,
            medianprops={"linewidth": 2.1},
            boxprops={"linewidth": 1.7},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
        )


    rng = np.random.default_rng(seed)
    for i, m in enumerate(methods, start=1):
        sub = df[df["Method"] == m]
        if sub.empty:
            continue
        y = sub["cost"].to_numpy(dtype=float)


        x = i + rng.uniform(-point_jitter, point_jitter, size=len(y))

        ax.scatter(
            x,
            y,
            s=9,
            alpha=0.33,
            linewidths=0.0,
            rasterized=rasterize_points,
        )


    tick_labels = []
    for m in methods:
        n = nsel_map.get(m, float("nan"))

        tick_labels.append(f"{METHOD_LABEL.get(m, m)}\n(" + r"$\tilde{n}$" + f"={format_nsel(n, nsel_digits)})")

    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Method (tick labels show median #Selected@Budget across apps)")
    ax.set_ylabel(ylabel + (" (log scale)" if log_scale else ""))

    if log_scale:
        ax.set_yscale("log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=r"data\result_cost", type=str, help="Folder containing rq2_report_*.xlsx")
    ap.add_argument("--out_dir", default="out/fig3_granularity_box", type=str, help="Output folder")
    ap.add_argument("--cap_per_app", default=50, type=int, help="Main fig: cap views per (App,Method)")
    ap.add_argument("--seed", default=7, type=int, help="Random seed for sampling/jitter")
    ap.add_argument("--no_log", action="store_true", help="Disable log scale")
    ap.add_argument(
        "--nsel_agg",
        choices=["median", "mean", "total"],
        default="median",
        help="How to aggregate #Selected@Budget across apps for tick labels",
    )
    ap.add_argument("--nsel_digits", type=int, default=1, help="Digits for tick nSel display (default 1)")
    ap.add_argument(
        "--appendix_max_points",
        type=int,
        default=0,
        help="Appendix fig: optionally downsample total points per method for readability (0 = no downsample)",
    )
    ap.add_argument(
        "--appendix_show_box",
        action="store_true",
        help="If set, appendix also shows box summary (default: points only)",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    safe_mkdir(out_dir)

    files = sorted(in_dir.glob("rq2_report_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No rq2_report_*.xlsx found in: {in_dir}")

    print(f"[Scan] {len(files)} excels found in: {in_dir}")

    sel_parts: List[pd.DataFrame] = []
    nsel_parts: List[pd.DataFrame] = []

    for fp in files:
        if fp.name.startswith("~$"):
            continue
        app = app_name_from_filename(fp)

        sel = load_selected_views(fp)
        sel["App"] = app
        sel_parts.append(sel[["App", "Method", "cost"]])

        nsel = load_nsel_from_portfolio_or_red(fp)
        if not nsel.empty:
            nsel["App"] = app
            nsel_parts.append(nsel[["App", "Method", "nSel"]])

        print(f"  - loaded: {app} ({fp.name})")

    all_sel = pd.concat(sel_parts, ignore_index=True) if sel_parts else pd.DataFrame()
    if all_sel.empty:
        raise RuntimeError("No selected-view records loaded.")

    nsel_table = (
        pd.concat(nsel_parts, ignore_index=True)
        if nsel_parts
        else pd.DataFrame(columns=["App", "Method", "nSel"])
    )


    out_csv = out_dir / "Fig3_nSel_by_app_method.csv"
    if not nsel_table.empty:
        nsel_table.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved audit CSV: {out_csv}")
    else:
        print("[WARN] Could not load nSel from Portfolio/Redundancy. Tick ñ=... will fallback to SelectedViews counts.")


    nsel_map = compute_tick_nsel(
        nsel_table=nsel_table,
        selected_views=all_sel,
        methods=METHOD_ORDER,
        agg=args.nsel_agg,
    )


    balanced = app_balanced_sample(all_sel, cap_per_app=args.cap_per_app, seed=args.seed)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    style_axes(ax)

    draw_box_and_points(
        ax,
        balanced,
        METHOD_ORDER,
        nsel_map,
        ylabel=r"Per-view review cost, $cost(S)$",
        log_scale=(not args.no_log),
        point_jitter=0.18,
        seed=args.seed,
        nsel_digits=args.nsel_digits,
        show_box=True,
        rasterize_points=True,
    )

    out_png = out_dir / "Fig3_main_balanced_boxswarm_log.png"
    out_pdf = out_dir / "Fig3_main_balanced_boxswarm_log.pdf"
    save_fig(fig, out_png, out_pdf)
    print(f"[OK] Saved main figure: {out_pdf}")


    appendix_df = all_sel.copy()


    if args.appendix_max_points and args.appendix_max_points > 0:
        rng = np.random.default_rng(args.seed)
        parts = []
        for m in METHOD_ORDER:
            g = appendix_df[appendix_df["Method"] == m]
            if len(g) <= args.appendix_max_points:
                parts.append(g)
            else:
                idx = rng.choice(g.index.to_numpy(), size=args.appendix_max_points, replace=False)
                parts.append(g.loc[idx])
        appendix_df = pd.concat(parts, ignore_index=True) if parts else appendix_df

    fig2, ax2 = plt.subplots(figsize=(6.2, 3.6))
    style_axes(ax2)

    draw_box_and_points(
        ax2,
        appendix_df,
        METHOD_ORDER,
        nsel_map,
        ylabel=r"Per-view review cost, $cost(S)$",
        log_scale=(not args.no_log),
        point_jitter=0.18,
        seed=args.seed + 11,
        nsel_digits=args.nsel_digits,
        show_box=args.appendix_show_box,
        rasterize_points=True,
    )

    out_png2 = out_dir / "Fig3_appendix_full_pooled_points_log.png"
    out_pdf2 = out_dir / "Fig3_appendix_full_pooled_points_log.pdf"
    save_fig(fig2, out_png2, out_pdf2)
    print(f"[OK] Saved appendix figure: {out_pdf2}")

    print("\n[Caption note you can use]")
    print(
        f"Main uses app-balanced sampling (cap={args.cap_per_app} views per app-method) to avoid app dominance; "
        f"each dot is a selected view; y-axis is log-scaled due to heavy-tailed costs."
    )

    print("[Done]")


if __name__ == "__main__":
    main()

"""
Examples:

# Main + Appendix (recommended)
python rq2_fig3_granularity_balanced_pooled_clean.py --in_dir "...\data\result_cost" --out_dir "...\fig3" --cap_per_app 50

# Appendix too dense? render-only downsample per method
python rq2_fig3_granularity_balanced_pooled_clean.py --in_dir "...\data\result_cost" --out_dir "...\fig3" --appendix_max_points 2000

# If you want appendix to also show a box summary:
python rq2_fig3_granularity_balanced_pooled_clean.py --in_dir "...\data\result_cost" --out_dir "...\fig3" --appendix_show_box
"""
