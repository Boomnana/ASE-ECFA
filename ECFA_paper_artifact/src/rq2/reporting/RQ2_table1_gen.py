from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_METHODS = [
    "ECFA",
    "ECFA w/o cost",
    "ECFA (No Merge)",
    "Random-k",
    "TopUniSize-k",
    "TopRawSize-k",
]

DISPLAY = {
    "ECFA": "ECFA (Full)",
    "ECFA w/o cost": "ECFA w/o Cost",
    "ECFA (No Merge)": "ECFA w/o Merge",
    "Random-k": "Random-k",
    "TopUniSize-k": "TopUniSize-k",
    "TopRawSize-k": "TopRawSize-k",
}

REF_METHOD = "ECFA"


def app_name_from_filename(fp: Path) -> str:
    m = re.match(r"rq2_report_(.+)\.xlsx$", fp.name)
    return m.group(1) if m else fp.stem


def normalize_method_name(s: str) -> str:
    s0 = str(s).strip()
    if s0.lower() in {"ecfa (full)", "ecfa full"}:
        return "ECFA"
    if s0.startswith("ECFA"):
        sl = s0.lower()
        if "w/o" in sl and "cost" in sl:
            return "ECFA w/o cost"
        if "no cost" in sl:
            return "ECFA w/o cost"
        if "no merge" in sl:
            return "ECFA (No Merge)"
        if "w/o merge" in sl:
            return "ECFA (No Merge)"
        if s0 == "ECFA":
            return "ECFA"
    if s0.startswith("Random-k") or s0.lower().startswith("random"):
        return "Random-k"
    if s0.startswith("TopUniSize-k"):
        return "TopUniSize-k"
    if s0.startswith("TopRawSize-k"):
        return "TopRawSize-k"
    return s0


def parse_float_pm(x) -> float:

    if x is None:
        return float("nan")
    if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
        return float(x)
    s = str(x).strip()
    if not s:
        return float("nan")
    if "±" in s:
        s = s.split("±", 1)[0].strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else float("nan")


def format_median_iqr(values: List[float], digits: int = 3) -> str:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return "—"
    med = float(np.nanmedian(arr))
    q1 = float(np.nanpercentile(arr, 25))
    q3 = float(np.nanpercentile(arr, 75))
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(med)} [{fmt.format(q1)}, {fmt.format(q3)}]"


def first_sheet(xl: pd.ExcelFile, names: List[str]) -> Optional[str]:
    for n in names:
        if n in xl.sheet_names:
            return n
    return None


def pick_first_col_by_prefix(df: pd.DataFrame, prefixes: List[str]) -> Optional[str]:

    cols = list(df.columns)

    for p in prefixes:
        if p in cols:
            return p

    for c in cols:
        sc = str(c).strip()
        for p in prefixes:
            if sc.startswith(str(p).strip()):
                return c
    return None


def load_one_file(fp: Path) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(fp)

    s_metrics = first_sheet(xl, ["Metrics"])
    if not s_metrics:
        raise ValueError(f"{fp.name} missing sheet: Metrics")

    s_red = first_sheet(xl, ["Redundancy_Report", "Redundancy"])
    s_port = first_sheet(xl, ["Portfolio"])
    s_sel = first_sheet(xl, ["SelectedViews_Stats_AllMethods", "SelectedViews_Stats"])
    s_meta = first_sheet(xl, ["meta", "Meta"])

    out: Dict[str, pd.DataFrame] = {"Metrics": xl.parse(s_metrics)}
    if s_red:
        out["Redundancy_Report"] = xl.parse(s_red)
    if s_port:
        out["Portfolio"] = xl.parse(s_port)
    if s_sel:
        out["SelectedViews_Stats"] = xl.parse(s_sel)
    if s_meta:
        out["Meta"] = xl.parse(s_meta)
    return out


def compute_one_app(app: str, sheets: Dict[str, pd.DataFrame], methods: List[str]) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    raw: Dict[str, pd.DataFrame] = {}

    met = sheets["Metrics"].copy()
    met["App"] = app
    met["Method_raw"] = met["Method"].astype(str)
    met["Method"] = met["Method_raw"].map(normalize_method_name)
    raw["Raw_Metrics"] = met

    red = sheets.get("Redundancy_Report")
    if red is not None:
        red = red.copy()
        red["App"] = app
        if "Method" in red.columns:
            red["Method_raw"] = red["Method"].astype(str)
            red["Method"] = red["Method_raw"].map(normalize_method_name)
        raw["Raw_Redundancy_Report"] = red

    port = sheets.get("Portfolio")
    if port is not None:
        port = port.copy()
        port["App"] = app
        if "Method" in port.columns:
            port["Method_raw"] = port["Method"].astype(str)
            port["Method"] = port["Method_raw"].map(normalize_method_name)
        raw["Raw_Portfolio"] = port

    sel = sheets.get("SelectedViews_Stats")
    if sel is not None:
        sel = sel.copy()
        sel["App"] = app
        if "Method" in sel.columns:
            sel["Method_raw"] = sel["Method"].astype(str)
            sel["Method"] = sel["Method_raw"].map(normalize_method_name)
        raw["Raw_SelectedViews_Stats"] = sel

    meta = sheets.get("Meta")
    if meta is not None:
        meta = meta.copy()
        meta["App"] = app
        raw["Raw_Meta"] = meta


    COL_COV_ENDPOINT = "Coverage@K (Report)"
    if COL_COV_ENDPOINT not in met.columns:
        raise ValueError(f"[{app}] Metrics missing required column: {COL_COV_ENDPOINT}")


    col_k = "K"
    col_maxoverlap = None
    col_frac_j0 = None
    if red is not None:
        col_maxoverlap = pick_first_col_by_prefix(red, ["MaxOverlap", "MaxOverlap ↓"])
        col_frac_j0 = pick_first_col_by_prefix(red, ["Frac(J>0)", "Frac(J>0) ↓", "Frac(J > 0)", "Frac(J > 0) ↓"])


    port_median_cost_col = None
    port_avg_cost_col = None
    port_total_cost_col = None
    port_nsel_col = None
    port_candidates_col = None
    if port is not None:
        port_median_cost_col = pick_first_col_by_prefix(port, ["MedianCost", "Std_MedianCost", "MedianCost ↓"])
        port_avg_cost_col = pick_first_col_by_prefix(port, ["AvgCostPerView", "AvgCostPerView ↓", "AvgCost"])
        port_total_cost_col = pick_first_col_by_prefix(port, ["TotalCost", "TotalCostSelected", "TotalCostSelected ↓"])
        port_nsel_col = pick_first_col_by_prefix(port, ["nSel", "n_selected", "Selected"])
        port_candidates_col = pick_first_col_by_prefix(port, ["#Candidates", "n_candidates", "nCandidates", "Candidates", "NumCandidates"])


    met_candidates_col = pick_first_col_by_prefix(met, ["#Candidates", "n_candidates", "nCandidates", "Candidates", "NumCandidates"])

    def median_over_rows(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return float("nan")
        vals = np.array([parse_float_pm(v) for v in df[col].tolist()], dtype=float)
        vals = vals[~np.isnan(vals)]
        return float(np.nanmedian(vals)) if len(vals) else float("nan")

    rows = []
    for m in methods:
        subm = met[met["Method"] == m]
        if subm.empty:
            continue

        cov_budget = median_over_rows(subm, COL_COV_ENDPOINT)


        nsel = float("nan")
        maxov = float("nan")
        frac_j0 = float("nan")
        if red is not None and "Method" in red.columns:
            subr = red[red["Method"] == m]
            if not subr.empty:
                if col_k in subr.columns:
                    nsel = parse_float_pm(subr.iloc[0][col_k])
                if col_maxoverlap is not None and col_maxoverlap in subr.columns:
                    maxov = parse_float_pm(subr.iloc[0][col_maxoverlap])
                if col_frac_j0 is not None and col_frac_j0 in subr.columns:
                    frac_j0 = parse_float_pm(subr.iloc[0][col_frac_j0])


        med_cost = float("nan")


        avg_cost = float("nan")

        if port is not None and "Method" in port.columns:
            subp = port[port["Method"] == m]
            if not subp.empty:

                if port_median_cost_col is not None and port_median_cost_col in subp.columns:
                    med_cost = parse_float_pm(subp.iloc[0][port_median_cost_col])


                if port_avg_cost_col is not None and port_avg_cost_col in subp.columns:
                    avg_cost = parse_float_pm(subp.iloc[0][port_avg_cost_col])


                nsel_eff = nsel
                if np.isnan(nsel_eff) and port_nsel_col is not None and port_nsel_col in subp.columns:
                    nsel_eff = parse_float_pm(subp.iloc[0][port_nsel_col])

                if port_total_cost_col is not None and port_total_cost_col in subp.columns and not np.isnan(nsel_eff) and nsel_eff > 0:
                    total_cost = parse_float_pm(subp.iloc[0][port_total_cost_col])
                    if not np.isnan(total_cost):
                        if np.isnan(med_cost):
                            med_cost = float(total_cost) / float(nsel_eff)
                        if np.isnan(avg_cost):
                            avg_cost = float(total_cost) / float(nsel_eff)


        n_cand = float("nan")
        if port is not None and "Method" in port.columns and port_candidates_col is not None:
            subp = port[port["Method"] == m]
            if not subp.empty and port_candidates_col in subp.columns:
                n_cand = parse_float_pm(subp.iloc[0][port_candidates_col])
        if np.isnan(n_cand) and met_candidates_col is not None and met_candidates_col in subm.columns:
            n_cand = median_over_rows(subm, met_candidates_col)

        rows.append({
            "App": app,
            "Method": m,
            "Coverage@Budget (Report)": cov_budget,
            "#Selected@Budget": nsel,
            "MedianCost/View": med_cost,
            "AvgCostPerView": avg_cost,
            "#Candidates": n_cand,
            "MaxOverlap": maxov,
            "Frac(J>0)": frac_j0,
        })

    per_app = pd.DataFrame(rows)
    return per_app, raw


def build_table1(per_app: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    out_rows = []
    for m in methods:
        sub = per_app[per_app["Method"] == m]
        if sub.empty:
            continue
        out_rows.append({
            "Method": DISPLAY.get(m, m),
            "Coverage@Budget (Report) ↑": format_median_iqr(sub["Coverage@Budget (Report)"].tolist(), digits=3),
            "#Selected@Budget ↑": format_median_iqr(sub["#Selected@Budget"].tolist(), digits=1),
            "MedianCost/View ↓": format_median_iqr(sub["MedianCost/View"].tolist(), digits=2),
            "#Candidates ↓": format_median_iqr(sub["#Candidates"].tolist(), digits=1),
            "MaxOverlap ↓": format_median_iqr(sub["MaxOverlap"].tolist(), digits=3),
        })
    return pd.DataFrame(out_rows)


def build_table1_avgcost(per_app: pd.DataFrame, methods: List[str]) -> pd.DataFrame:

    out_rows = []
    for m in methods:
        sub = per_app[per_app["Method"] == m]
        if sub.empty:
            continue
        out_rows.append({
            "Method": DISPLAY.get(m, m),
            "Coverage@Budget (Report) ↑": format_median_iqr(sub["Coverage@Budget (Report)"].tolist(), digits=3),
            "#Selected@Budget ↑": format_median_iqr(sub["#Selected@Budget"].tolist(), digits=1),
            "AvgCostPerView ↓": format_median_iqr(sub["AvgCostPerView"].tolist(), digits=2),
            "#Candidates ↓": format_median_iqr(sub["#Candidates"].tolist(), digits=1),
            "MaxOverlap ↓": format_median_iqr(sub["MaxOverlap"].tolist(), digits=3),
            "Frac(J>0) ↓": format_median_iqr(sub["Frac(J>0)"].tolist(), digits=3),
        })
    return pd.DataFrame(out_rows)


def build_table_delta(per_app: pd.DataFrame, methods: List[str], ref_method: str = REF_METHOD) -> pd.DataFrame:
    apps = sorted(per_app["App"].dropna().unique().tolist())

    def get_row(app: str, method: str) -> Optional[pd.Series]:
        sub = per_app[(per_app["App"] == app) & (per_app["Method"] == method)]
        if sub.empty:
            return None
        return sub.iloc[0]

    out_rows = []
    for m in methods:

        d_cov_pp = []
        d_sel = []
        d_avgcost = []
        d_cand = []
        d_maxov = []
        d_frac = []

        for app in apps:
            r_ref = get_row(app, ref_method)
            r_m = get_row(app, m)
            if r_ref is None or r_m is None:
                continue

            cov_ref = parse_float_pm(r_ref.get("Coverage@Budget (Report)"))
            cov_m = parse_float_pm(r_m.get("Coverage@Budget (Report)"))
            sel_ref = parse_float_pm(r_ref.get("#Selected@Budget"))
            sel_m = parse_float_pm(r_m.get("#Selected@Budget"))
            cost_ref = parse_float_pm(r_ref.get("AvgCostPerView"))
            cost_m = parse_float_pm(r_m.get("AvgCostPerView"))
            cand_ref = parse_float_pm(r_ref.get("#Candidates"))
            cand_m = parse_float_pm(r_m.get("#Candidates"))
            maxov_ref = parse_float_pm(r_ref.get("MaxOverlap"))
            maxov_m = parse_float_pm(r_m.get("MaxOverlap"))
            frac_ref = parse_float_pm(r_ref.get("Frac(J>0)"))
            frac_m = parse_float_pm(r_m.get("Frac(J>0)"))


            if not (np.isnan(cov_ref) or np.isnan(cov_m)):
                d_cov_pp.append((cov_m - cov_ref) * 100.0)
            if not (np.isnan(sel_ref) or np.isnan(sel_m)):
                d_sel.append(sel_m - sel_ref)


            if not (np.isnan(cost_ref) or np.isnan(cost_m)):
                d_avgcost.append(cost_ref - cost_m)
            if not (np.isnan(cand_ref) or np.isnan(cand_m)):
                d_cand.append(cand_ref - cand_m)
            if not (np.isnan(maxov_ref) or np.isnan(maxov_m)):
                d_maxov.append(maxov_ref - maxov_m)
            if not (np.isnan(frac_ref) or np.isnan(frac_m)):
                d_frac.append(frac_ref - frac_m)

        out_rows.append({
            "Method": DISPLAY.get(m, m),
            "ΔCoverage@Budget (pp) ↑": format_median_iqr(d_cov_pp, digits=1),
            "Δ#Selected@Budget ↑": format_median_iqr(d_sel, digits=1),
            "ΔAvgCostPerView ↑": format_median_iqr(d_avgcost, digits=2),
            "Δ#Candidates ↑": format_median_iqr(d_cand, digits=1),
            "ΔMaxOverlap ↑": format_median_iqr(d_maxov, digits=3),
            "ΔFrac(J>0) ↑": format_median_iqr(d_frac, digits=3),
        })

    return pd.DataFrame(out_rows)


def write_excel(
    out_path: Path,
    table1: pd.DataFrame,
    table1_avgcost: pd.DataFrame,
    table_delta: pd.DataFrame,
    appendix: pd.DataFrame,
    raw_sheets: Dict[str, pd.DataFrame],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        table1.to_excel(writer, sheet_name="Table1_AcrossApps", index=False)
        table1_avgcost.to_excel(writer, sheet_name="Table1_AcrossApps_AvgCost", index=False)
        table_delta.to_excel(writer, sheet_name="Table1_Delta_AcrossApps", index=False)
        appendix.to_excel(writer, sheet_name="Appendix_PerApp", index=False)
        appendix.to_excel(writer, sheet_name="Raw_Computed_PerApp", index=False)

        for name, df in raw_sheets.items():
            sname = name[:31]
            df.to_excel(writer, sheet_name=sname, index=False)

    print(f"[Saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        type=str,
        default=str(Path(__file__).parent / "data" / "result_cost"),
        help=r"Folder containing rq2_report_*.xlsx (default: ./data/result_cost)",
    )
    ap.add_argument(
        "--out_xlsx",
        type=str,
        default="out/RQ2_Endpoint_Tables.xlsx",
        help="Output excel path",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods to include",
    )
    ap.add_argument(
        "--ref_method",
        type=str,
        default=REF_METHOD,
        help="Reference method for delta table (default: ECFA)",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_xlsx = Path(args.out_xlsx).resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ref_method = normalize_method_name(args.ref_method)

    files = sorted(in_dir.glob("rq2_report_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No rq2_report_*.xlsx found in {in_dir}")

    per_app_list: List[pd.DataFrame] = []
    raw_collect: Dict[str, List[pd.DataFrame]] = {
        "Raw_Metrics": [],
        "Raw_Redundancy_Report": [],
        "Raw_Portfolio": [],
        "Raw_SelectedViews_Stats": [],
        "Raw_Meta": [],
    }

    for fp in files:
        if fp.name.startswith("~$"):
            continue
        app = app_name_from_filename(fp)
        sheets = load_one_file(fp)
        per_app, raw = compute_one_app(app, sheets, methods)
        per_app_list.append(per_app)

        for k in raw_collect.keys():
            if k in raw:
                raw_collect[k].append(raw[k])

    appendix = pd.concat(per_app_list, ignore_index=True) if per_app_list else pd.DataFrame()
    table1 = build_table1(appendix, methods)
    table1_avgcost = build_table1_avgcost(appendix, methods)
    table_delta = build_table_delta(appendix, methods, ref_method=ref_method)

    raw_sheets_out: Dict[str, pd.DataFrame] = {}
    for k, parts in raw_collect.items():
        raw_sheets_out[k] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    write_excel(out_xlsx, table1, table1_avgcost, table_delta, appendix, raw_sheets_out)

    print("\n[Done]")
    print("  Apps scanned:", len(files))
    print("  Appendix rows:", len(appendix))
    print("  Table1 rows:", len(table1))
    print("  Table1(avgcost) rows:", len(table1_avgcost))
    print("  Delta table rows:", len(table_delta))
    print("  Output:", out_xlsx)


if __name__ == "__main__":
    main()

"""
Run:
python rq2_export_endpoint_tables_final.py --in_dir "...\data\result_cost" --out_xlsx "...\out\RQ2_Endpoint_Tables.xlsx"

Optional method subset (recommended for main text):
python rq2_export_endpoint_tables_final.py --methods "ECFA,ECFA w/o cost,ECFA (No Merge),Random-k"

Delta reference method (default ECFA):
python rq2_export_endpoint_tables_final.py --ref_method "ECFA"
"""
