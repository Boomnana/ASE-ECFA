import pandas as pd
import numpy as np
from scipy import stats

INPUT_PATH = r"glm4air结果\RQ1_entity_unification_results.xlsx"
K_MAX = 50

def compute_table_x(input_path: str, k_max: int = 50, use_wilcoxon: bool = False):

    df = pd.read_excel(input_path, sheet_name="coverage_curve")
    required_cols = {"app_id", "variant", "k", "report_coverage"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"coverage_curve 缺少列: {missing}")

    df = df.copy()
    df["k"] = df["k"].astype(int)


    cov5 = (
        df[df["k"] == 5][["app_id", "variant", "report_coverage"]]
        .rename(columns={"report_coverage": "Coverage@5"})
    )
    cov10 = (
        df[df["k"] == 10][["app_id", "variant", "report_coverage"]]
        .rename(columns={"report_coverage": "Coverage@10"})
    )


    auc_norm = (
        df[df["k"].between(1, k_max)]
        .groupby(["app_id", "variant"])["report_coverage"]
        .mean()
        .reset_index()
        .rename(columns={"report_coverage": f"AUC(1-{k_max})"})
    )

    per_app = (
        auc_norm.merge(cov5, on=["app_id", "variant"], how="left")
                .merge(cov10, on=["app_id", "variant"], how="left")
    )


    def summarize_metric(metric_col: str):
        pivot = per_app.pivot(index="app_id", columns="variant", values=metric_col)


        if "no_unify" not in pivot.columns or "unify" not in pivot.columns:
            raise ValueError(f"{metric_col} 缺少 no_unify / unify 配对数据")

        no = pivot["no_unify"].astype(float)
        uni = pivot["unify"].astype(float)
        diff = uni - no

        mean_no, std_no = no.mean(), no.std(ddof=1)
        mean_uni, std_uni = uni.mean(), uni.std(ddof=1)
        mean_diff = diff.mean()


        if use_wilcoxon:

            valid = (~no.isna()) & (~uni.isna())
            stat, p = stats.wilcoxon(uni[valid], no[valid])
            test_name = "Wilcoxon"
        else:
            stat, p = stats.ttest_rel(uni, no, nan_policy="omit")
            test_name = "paired t-test"

        return {
            "Metric": metric_col,
            "w/o unify (Mean ± Std)": f"{mean_no:.4f} ± {std_no:.4f}",
            "w/ unify (Mean ± Std)": f"{mean_uni:.4f} ± {std_uni:.4f}",
            "Δ Gain (mean paired)": f"{mean_diff:+.4f}",
            "p-value": float(p),
            "test": test_name
        }

    rows = []
    rows.append(summarize_metric("Coverage@5"))
    rows.append(summarize_metric("Coverage@10"))
    rows.append(summarize_metric(f"AUC(1-{k_max})"))

    table_x = pd.DataFrame(rows)


    out_path = "Table_X_across_app.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        per_app.sort_values(["app_id", "variant"]).to_excel(writer, index=False, sheet_name="per_app_metrics")
        table_x.to_excel(writer, index=False, sheet_name="Table_X")

    print("\n=== Table X (Across-App) ===")
    print(table_x.to_string(index=False))
    print(f"\n[Saved] {out_path}")
    return table_x, per_app

if __name__ == "__main__":

    compute_table_x(INPUT_PATH, k_max=K_MAX, use_wilcoxon=False)
