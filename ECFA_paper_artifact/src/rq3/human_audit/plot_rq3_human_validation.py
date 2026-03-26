
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent
    in_csv = base / "RQ3_point_range_table.csv"
    out_png = base / "rq3_human_validation_point_range.png"
    out_pdf = base / "rq3_human_validation_point_range.pdf"
    out_svg = base / "rq3_human_validation_point_range.svg"

    df = pd.read_csv(in_csv)


    method_order = ["ECFA", "Global", "MC-L5"]
    boundary_order = ["LLMCluster", "SBERT+HC", "TFIDF+HC"]

    df = df[df["method_display"].isin(method_order)].copy()
    df["method_display"] = pd.Categorical(df["method_display"], categories=method_order, ordered=True)
    df["boundary_display"] = pd.Categorical(df["boundary_display"], categories=boundary_order, ordered=True)
    df = df.sort_values(["boundary_display", "method_display"]).reset_index(drop=True)


    y_positions = []
    y = 0
    gap = 0.9
    for boundary in boundary_order[::-1]:
        sub = df[df["boundary_display"] == boundary]
        for _ in range(len(sub)):
            y_positions.append(y)
            y += 1
        y += gap
    df_plot = df.iloc[::-1].copy().reset_index(drop=True)
    df_plot["y"] = y_positions


    marker_map = {"ECFA": "o", "Global": "s", "MC-L5": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.8), sharey=True)

    specs = [
        ("success_at_20", "success_ci_low", "success_ci_high", "Success@20"),
        ("recall_at_20", "recall_ci_low", "recall_ci_high", "Recall@20"),
    ]

    for ax, (mean_col, low_col, high_col, xlabel) in zip(axes, specs):
        for method in method_order:
            sub = df_plot[df_plot["method_display"] == method]
            x = sub[mean_col].to_numpy()
            y = sub["y"].to_numpy()
            xerr = np.vstack([
                x - sub[low_col].to_numpy(),
                sub[high_col].to_numpy() - x
            ])
            ax.errorbar(
                x, y, xerr=xerr,
                fmt=marker_map[method],
                linestyle="none",
                capsize=3,
                label=method
            )

        ax.set_xlabel(xlabel)
        ax.set_xlim(0, max(df_plot[high_col].max() if high_col=="recall_ci_high" else df_plot["success_ci_high"].max(),
                           df_plot["recall_ci_high"].max()) + 0.05)
        ax.grid(True, axis="x", alpha=0.3)


    axes[0].set_yticks(df_plot["y"])
    axes[0].set_yticklabels(df_plot["boundary_display"].astype(str) + " | " + df_plot["method_display"].astype(str))
    axes[1].tick_params(axis="y", left=False, labelleft=False)


    ys = df_plot.groupby("boundary_display", observed=False)["y"].agg(["min", "max"]).sort_values("max", ascending=False)
    for ax in axes:
        for _, row in ys.iloc[:-1].iterrows():
            ax.axhline(row["min"] - 0.45, linewidth=0.8, alpha=0.4)

    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("RQ3 human-judged validation subset (mean with 95% bootstrap CI)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")

if __name__ == "__main__":
    main()
