from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


METHOD_ORDER = [
    "ECFA",
    "ECFA w/o cost",
    "ECFA (No Merge)",
    "Random-k",
]

METHOD_LABEL = {
    "ECFA": "ECFA",
    "ECFA w/o cost": "w/o Cost",
    "ECFA (No Merge)": "w/o Merge",
    "Random-k": "Random",
}

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def app_name_from_filename(fp: Path) -> str:

    m = re.match(r"rq2_report_(.+)\.xlsx$", fp.name)
    return m.group(1) if m else fp.stem

def style_axes(ax):
    ax.grid(True, alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def save_fig(fig, out_png: Path, out_pdf: Optional[Path] = None, dpi: int = 320):
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    out_svg = out_png.with_suffix(".svg")
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.1)
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def load_one_app(fp: Path) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(fp)
    need = [
        "Metrics",


    ]
    missing = [s for s in need if s not in xl.sheet_names]
    if missing:


        if "Metrics" not in xl.sheet_names:
            raise ValueError(f"{fp.name} missing sheet: Metrics")


    out = {"Metrics": xl.parse("Metrics")}
    return out


def fig2_budget_snapshots_agg(apps_data, out_dir: Path, *, metric: str = "report"):


    assert metric in ("report", "entity")
    cols = {
        0.25: f"Cov@0.25B ({'Report' if metric=='report' else 'Entity'})",
        0.50: f"Cov@0.50B ({'Report' if metric=='report' else 'Entity'})",
        0.75: f"Cov@0.75B ({'Report' if metric=='report' else 'Entity'})",
    }


    data = {p: {m: [] for m in METHOD_ORDER} for p in cols.keys()}

    for app, d in apps_data.items():
        met = d["Metrics"].copy()
        met["Method"] = met["Method"].astype(str)
        for p, c in cols.items():
            if c not in met.columns:
                continue
            for m in METHOD_ORDER:
                row = met[met["Method"] == m]
                if row.empty:
                    continue
                v = float(row.iloc[0][c])
                data[p][m].append(v)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
    })

    COLORS = {
        "ECFA": "red",
        "ECFA w/o cost": "orange",
        "ECFA (No Merge)": "purple",
        "Random-k": "0.4",
    }
    MARKERS = {
        "ECFA": "o",
        "ECFA w/o cost": "s",
        "ECFA (No Merge)": "D",
        "Random-k": "^",
    }

    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.3, 2.55),
        sharey=True
    )
    fig.subplots_adjust(wspace=0.15)

    for ax in axes:
        style_axes(ax)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.grid(True, axis="x", alpha=0.18)
        ax.grid(False, axis="y")

    ys = np.arange(len(METHOD_ORDER))[::-1]
    ylabels = [METHOD_LABEL.get(m, m) for m in METHOD_ORDER]

    for ax, p, panel in zip(axes, [0.25, 0.50, 0.75], ["(a)", "(b)", "(c)"]):
        meds, q1s, q3s = [], [], []
        for m in METHOD_ORDER:
            vals = np.array(data[p][m], dtype=float)
            if len(vals) == 0:
                meds.append(np.nan); q1s.append(np.nan); q3s.append(np.nan)
            else:
                meds.append(float(np.nanmedian(vals)))
                q1s.append(float(np.nanpercentile(vals, 25)))
                q3s.append(float(np.nanpercentile(vals, 75)))

        meds = np.array(meds, dtype=float)
        q1s  = np.array(q1s, dtype=float)
        q3s  = np.array(q3s, dtype=float)

        for i, m in enumerate(METHOD_ORDER):
            if np.isnan(meds[i]):
                continue
            y = ys[i]
            color = COLORS.get(m, "black")
            mk = MARKERS.get(m, "o")


            ax.errorbar(
                meds[i], y,
                xerr=[[meds[i] - q1s[i]], [q3s[i] - meds[i]]],
                fmt=mk,
                markersize=8.5,
                mfc=color, mec="black", mew=0.9,
                ecolor=color,
                elinewidth=2.6, capsize=4.0, capthick=2.2,
                zorder=3,
            )

        ax.set_title(f"{panel}  {int(p*100)}% budget", pad=6)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(ylabels)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    axes[0].set_ylabel("Method")


    fig.supxlabel("Coverage (median with IQR across apps)", y=-0.15)

    tag = "report" if metric == "report" else "entity"
    out_png = out_dir / f"Fig2_Snapshots_{tag}.png"
    out_pdf = out_dir / f"Fig2_Snapshots_{tag}.pdf"
    save_fig(fig, out_png, out_pdf, dpi=320)
    print(f"[Saved] {out_png}")


def main():


    default_in_dir = Path(__file__).parent / "data" / "result_cost"

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default=str(default_in_dir), help="Folder containing rq2_report_*.xlsx")
    ap.add_argument("--out_dir", type=str, default="out/fig2_box", help="Output folder (default: <in_dir>/../rq2_plots)")
    ap.add_argument("--metric", choices=["report", "entity"], default="report", help="Which coverage to plot as main")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()

    if not in_dir.exists():
        print(f"[Warn] Directory not found: {in_dir}")

        cwd_data_cost = Path.cwd() / "data" / "result_cost"
        if cwd_data_cost.exists():
            in_dir = cwd_data_cost
            print(f"[Info] Found data in: {in_dir}")
        else:
             raise FileNotFoundError(f"No data directory found at {in_dir} or {cwd_data_cost}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (in_dir.parent / "rq2_plots")
    safe_mkdir(out_dir)

    files = sorted(in_dir.glob("rq2_report_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No rq2_report_*.xlsx found in {in_dir}")

    print(f"[Scan] {len(files)} excels found in: {in_dir}")

    apps_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for fp in files:
        app = app_name_from_filename(fp)
        apps_data[app] = load_one_app(fp)
        print(f"  - loaded: {app} ({fp.name})")


    fig2_budget_snapshots_agg(apps_data, out_dir, metric=args.metric)

    print(f"\n[Done] All outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
