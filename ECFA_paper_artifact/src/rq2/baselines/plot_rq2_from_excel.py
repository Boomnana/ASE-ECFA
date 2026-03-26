#!/usr/bin/env python3


from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Iterable, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_out_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_num(x) -> float:

    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")

    s = s.replace("+/-", "±")
    if "±" in s:
        s = s.split("±", 1)[0].strip()

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return float("nan")
    try:
        return float(m.group(0))
    except Exception:
        return float("nan")


def _filter_methods(methods: List[str], include: Optional[str], exclude: Optional[str]) -> List[str]:
    mset = list(methods)
    if include:
        wanted = [x.strip() for x in include.split(",") if x.strip()]
        keep = []
        for m in mset:
            if m in wanted:
                keep.append(m)
        mset = keep
    if exclude:
        banned = {x.strip() for x in exclude.split(",") if x.strip()}
        mset = [m for m in mset if m not in banned]
    return mset


def _setup_matplotlib():

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def _read_sheet(xlsx: Path, sheet: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(xlsx, sheet_name=sheet)
    except Exception:
        return None


def plot_cov_vs_rho(df_cost_long: pd.DataFrame, out_dir: Path, *, cov_col: str, title: str, fname: str, methods: List[str]):
    if df_cost_long is None or df_cost_long.empty:
        return
    if cov_col not in df_cost_long.columns or "rho" not in df_cost_long.columns or "Method" not in df_cost_long.columns:
        return

    plt.figure(figsize=(8, 5), dpi=160)
    for m in methods:
        d = df_cost_long[df_cost_long["Method"] == m].copy()
        if d.empty:
            continue
        d = d.sort_values("rho")
        xs = d["rho"].astype(float).to_numpy()
        ys = d[cov_col].astype(float).to_numpy()


        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if xs.size == 0:
            continue


        uniq = {}
        for x, y in zip(xs, ys):
            uniq[x] = max(float(y), float(uniq.get(x, -1.0)))
        xs2 = np.array(sorted(uniq.keys()), dtype=float)
        ys2 = np.array([uniq[x] for x in xs2], dtype=float)

        plt.plot(xs2, ys2, marker="o", markersize=2.2, linewidth=1.6, label=m)

    plt.xlabel("Normalized cost (ρ)")
    plt.ylabel("Coverage")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = out_dir / fname
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_cov_vs_k(df_cost_long: pd.DataFrame, out_dir: Path, *, cov_col: str, title: str, fname: str, methods: List[str]):
    if df_cost_long is None or df_cost_long.empty:
        return
    if cov_col not in df_cost_long.columns or "step" not in df_cost_long.columns or "Method" not in df_cost_long.columns:
        return

    plt.figure(figsize=(8, 5), dpi=160)
    for m in methods:
        d = df_cost_long[df_cost_long["Method"] == m].copy()
        if d.empty:
            continue
        d = d.sort_values("step")
        xs = d["step"].astype(int).to_numpy()
        ys = d[cov_col].astype(float).to_numpy()
        plt.plot(xs, ys, marker="o", markersize=2.2, linewidth=1.6, label=m)

    plt.xlabel("Selected views (k)")
    plt.ylabel("Coverage")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = out_dir / fname
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_rank_scatter(df_sv: pd.DataFrame, out_dir: Path, *, y_col: str, title: str, fname: str, methods: List[str]):
    if df_sv is None or df_sv.empty:
        return
    if y_col not in df_sv.columns or "rank" not in df_sv.columns or "Method" not in df_sv.columns:
        return

    plt.figure(figsize=(8, 5), dpi=160)
    for m in methods:
        d = df_sv[df_sv["Method"] == m].copy()
        if d.empty:
            continue
        d = d.sort_values("rank")
        plt.plot(d["rank"].astype(int), d[y_col].astype(float), marker="o", markersize=3.0, linewidth=1.4, label=m)

    plt.xlabel("Rank in selection order")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = out_dir / fname
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_box(df_sv: pd.DataFrame, out_dir: Path, *, y_col: str, title: str, fname: str, methods: List[str], logy: bool = False):
    if df_sv is None or df_sv.empty:
        return
    if y_col not in df_sv.columns or "Method" not in df_sv.columns:
        return
    data = []
    labels = []
    for m in methods:
        d = df_sv[df_sv["Method"] == m][y_col].astype(float)
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        data.append(d.to_numpy())
        labels.append(m)
    if not data:
        return

    plt.figure(figsize=(max(8, 1.2 * len(labels)), 5), dpi=160)
    plt.boxplot(data, labels=labels, showfliers=True)
    if logy:
        plt.yscale("log")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = out_dir / fname
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_portfolio_bars(df_port: pd.DataFrame, out_dir: Path, *, cols: List[str], title: str, fname: str, methods: List[str]):
    if df_port is None or df_port.empty:
        return
    if "Method" not in df_port.columns:
        return
    d = df_port[df_port["Method"].isin(methods)].copy()
    if d.empty:
        return


    for c in cols:
        if c in d.columns:
            d[c] = d[c].apply(_parse_num)
    d = d.set_index("Method")

    x = np.arange(len(d.index))
    width = 0.8 / max(1, len(cols))

    plt.figure(figsize=(max(8, 1.2 * len(d.index)), 5), dpi=160)
    for i, c in enumerate(cols):
        if c not in d.columns:
            continue
        plt.bar(x + i * width, d[c].to_numpy(dtype=float), width=width, label=c)

    plt.xticks(x + width * (len(cols) - 1) / 2, d.index.tolist(), rotation=20, ha="right")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = out_dir / fname
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_redundancy(df_red: pd.DataFrame, out_dir: Path, *, cols: List[str], title: str, fname: str, methods: List[str]):
    if df_red is None or df_red.empty:
        return
    if "Method" not in df_red.columns:
        return
    d = df_red[df_red["Method"].isin(methods)].copy()
    if d.empty:
        return
    for c in cols:
        if c in d.columns:
            d[c] = d[c].apply(_parse_num)

    d = d.set_index("Method")
    x = np.arange(len(d.index))
    width = 0.8 / max(1, len(cols))

    plt.figure(figsize=(max(8, 1.2 * len(d.index)), 5), dpi=160)
    for i, c in enumerate(cols):
        if c not in d.columns:
            continue
        plt.bar(x + i * width, d[c].to_numpy(dtype=float), width=width, label=c)

    plt.xticks(x + width * (len(cols) - 1) / 2, d.index.tolist(), rotation=20, ha="right")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = out_dir / fname
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to the generated rq2_report_*.xlsx")
    ap.add_argument("--out_dir", default="", help="Where to write figures. Default: <xlsx_stem>_plots next to xlsx.")
    ap.add_argument("--include", default="", help="Comma-separated methods to include (exact match).")
    ap.add_argument("--exclude", default="", help="Comma-separated methods to exclude.")
    ap.add_argument("--show", action="store_true", help="Show figures interactively (also saves).")
    args = ap.parse_args()

    xlsx = Path(args.xlsx).expanduser().resolve()
    if not xlsx.exists():
        raise SystemExit(f"[Error] xlsx not found: {xlsx}")

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (xlsx.parent / f"{xlsx.stem}_plots")
    out_dir = _ensure_out_dir(out_dir)

    _setup_matplotlib()


    df_metrics = _read_sheet(xlsx, "Metrics")
    df_cost_long = _read_sheet(xlsx, "CostCurve_Long")
    df_port = _read_sheet(xlsx, "Portfolio")
    df_red_r = _read_sheet(xlsx, "Redundancy_Report")
    df_red_e = _read_sheet(xlsx, "Redundancy_Entity")
    df_sv = _read_sheet(xlsx, "SelectedViews_Stats_AllMethods")


    methods = []
    if df_metrics is not None and not df_metrics.empty and "Method" in df_metrics.columns:
        methods = [str(m) for m in df_metrics["Method"].tolist()]
    elif df_cost_long is not None and not df_cost_long.empty and "Method" in df_cost_long.columns:
        methods = sorted({str(m) for m in df_cost_long["Method"].tolist()})
    elif df_sv is not None and not df_sv.empty and "Method" in df_sv.columns:
        methods = sorted({str(m) for m in df_sv["Method"].tolist()})

    methods = _filter_methods(methods, args.include.strip() or None, args.exclude.strip() or None)

    if not methods:
        raise SystemExit("[Error] cannot infer methods from Excel (missing 'Method' column?)")


    plot_cov_vs_rho(df_cost_long, out_dir, cov_col="cov_report",
                    title="Report coverage vs normalized cost (ρ)",
                    fname="01_cov_report_vs_rho.png", methods=methods)
    plot_cov_vs_rho(df_cost_long, out_dir, cov_col="cov_entity",
                    title="Entity coverage vs normalized cost (ρ)",
                    fname="02_cov_entity_vs_rho.png", methods=methods)


    plot_cov_vs_k(df_cost_long, out_dir, cov_col="cov_report",
                  title="Report coverage vs selected views (k)",
                  fname="03_cov_report_vs_k.png", methods=methods)
    plot_cov_vs_k(df_cost_long, out_dir, cov_col="cov_entity",
                  title="Entity coverage vs selected views (k)",
                  fname="04_cov_entity_vs_k.png", methods=methods)


    plot_rank_scatter(df_sv, out_dir, y_col="cost",
                      title="Selected views: cost by selection rank",
                      fname="05_rank_vs_cost.png", methods=methods)
    plot_rank_scatter(df_sv, out_dir, y_col="|Gamma_eff|",
                      title="Selected views: |Gamma_eff| by selection rank",
                      fname="06_rank_vs_gamma_eff.png", methods=methods)

    plot_box(df_sv, out_dir, y_col="cost",
             title="Selected views: cost distribution (boxplot)",
             fname="07_cost_box.png", methods=methods, logy=False)
    plot_box(df_sv, out_dir, y_col="|Gamma_eff|",
             title="Selected views: |Gamma_eff| distribution (boxplot)",
             fname="08_gamma_eff_box.png", methods=methods, logy=False)


    plot_portfolio_bars(df_port, out_dir,
                        cols=["AvgCostPerView", "MedianCost", "P95Cost"],
                        title="Portfolio: typical view cost (avg/median/P95)",
                        fname="09_portfolio_cost_stats.png", methods=methods)
    plot_portfolio_bars(df_port, out_dir,
                        cols=["Top1CostShare", "HHI_Cost"],
                        title="Portfolio: cost concentration (Top1 share / HHI)",
                        fname="10_portfolio_concentration.png", methods=methods)


    plot_redundancy(df_red_r, out_dir,
                    cols=["AvgOverlap ↓", "MaxOverlap ↓", "Frac(J>0) ↓"],
                    title="Redundancy (Report sets): lower is better",
                    fname="11_redundancy_report.png", methods=methods)
    plot_redundancy(df_red_e, out_dir,
                    cols=["AvgOverlap ↓", "MaxOverlap ↓", "Frac(J>0) ↓"],
                    title="Redundancy (Entity sets): lower is better",
                    fname="12_redundancy_entity.png", methods=methods)


    summary_lines = []
    summary_lines.append(f"xlsx: {xlsx.as_posix()}")
    summary_lines.append(f"methods: {', '.join(methods)}")
    if df_metrics is not None and not df_metrics.empty:
        keep_cols = [c for c in df_metrics.columns if c in {
            "Method",
            "Coverage@K (Report)", "Coverage@K (Entity)",
            "nAUC@B (Report)", "nAUC@B (Entity)",
            "Cov@0.25B (Entity)", "Cov@0.50B (Entity)", "Cov@0.75B (Entity)",
            "Cov@0.25B (Report)", "Cov@0.50B (Report)", "Cov@0.75B (Report)",
        }]
        if keep_cols:
            d = df_metrics[df_metrics["Method"].isin(methods)][keep_cols].copy()
            summary_lines.append("\n[Metrics]")
            summary_lines.append(d.to_string(index=False))
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[OK] Wrote figures to: {out_dir.as_posix()}")
    print(f"[OK] Wrote summary to: {(out_dir / 'summary.txt').as_posix()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
