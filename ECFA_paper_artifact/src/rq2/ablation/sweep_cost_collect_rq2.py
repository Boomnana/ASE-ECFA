#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd


def parse_cost_schedule(spec: str) -> List[float]:


    spec = spec.strip()
    if not spec:
        raise ValueError("Empty cost schedule.")

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    costs: List[float] = []

    for p in parts:
        if ":" in p:
            toks = [t.strip() for t in p.split(":")]
            if len(toks) != 3:
                raise ValueError(f"Bad range spec '{p}', expected start:end:step")
            start, end, step = map(float, toks)
            if step <= 0:
                raise ValueError(f"Step must be > 0 in '{p}'")

            x = start

            while x <= end + 1e-9:
                costs.append(round(x, 6))
                x += step
        else:
            costs.append(float(p))


    costs = sorted(set(costs))
    return costs


def comb2(n: int) -> int:
    return n * (n - 1) // 2 if n >= 2 else 0


def run_one_cost(
    python_exec: str,
    base_script: Path,
    input_root: str,
    dataset_xlsx: str,
    apps: str,
    objective: str,
    cost_budget: float,
    K: int,
    merge_sim_thresh: float,
    red_eps: float,
    pool_pairs: int,
    rand_seed: int,
    extra_args: List[str],
    tmp_out_xlsx: Path,
) -> None:
    cmd = [
        python_exec,
        str(base_script),
        "--input_root", input_root,
        "--dataset_xlsx", dataset_xlsx,
        "--output_xlsx", str(tmp_out_xlsx),
        "--budget_mode", "cost",
        "--cost_budget", str(cost_budget),
        "--K", str(K),
        "--objective", objective,
        "--merge_sim_thresh", str(merge_sim_thresh),
        "--red_eps", str(red_eps),
        "--pool_pairs", str(pool_pairs),
        "--rand_seed", str(rand_seed),
    ]
    if apps.strip():
        cmd += ["--apps", apps.strip()]

    if extra_args:
        cmd += extra_args

    print(f"[Run] cost_budget={cost_budget} -> {tmp_out_xlsx.name}")
    subprocess.run(cmd, check=True)


def read_sheet(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name=sheet_name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_script", default="run_ecfa_experiment_patched.py",
                    help="Path to run_ecfa_experiment_patched.py")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use.")
    ap.add_argument("--input_root", required=True, help="Same as --input_root of base script.")
    ap.add_argument("--dataset_xlsx", required=True, help="Same as --dataset_xlsx of base script.")
    ap.add_argument("--apps", default="", help="Comma-separated apps; empty = all sheets in dataset_xlsx.")
    ap.add_argument("--objective", default="entity", choices=["report", "entity", "hybrid"])
    ap.add_argument("--K", type=int, default=50, help="K used inside base script; keep 50 unless you know why.")
    ap.add_argument("--merge_sim_thresh", type=float, default=0.6)
    ap.add_argument("--red_eps", type=float, default=0.01)
    ap.add_argument("--pool_pairs", type=int, default=10000)
    ap.add_argument("--rand_seed", type=int, default=42)


    ap.add_argument(
        "--cost_schedule",
        default="50:2000:50",
        help='Cost list spec. Examples: "50:2000:50" or "50:400:50,500:1000:100,1200:2000:200" or "50,100,150".'
    )

    ap.add_argument("--out_xlsx", required=True, help="Output Excel path for plotting.")
    ap.add_argument("--keep_tmp", action="store_true", help="Keep intermediate xlsx outputs (debug).")
    ap.add_argument("--tmp_dir", default="", help="Optional temp dir; default uses system temp.")
    ap.add_argument("--extra_args", default="", help="Extra args forwarded to base_script, e.g. '--use_defects'")

    args = ap.parse_args()

    base_script = Path(args.base_script).resolve()
    if not base_script.exists():
        raise FileNotFoundError(f"base_script not found: {base_script}")

    costs = parse_cost_schedule(args.cost_schedule)
    if not costs:
        raise ValueError("No costs parsed.")

    extra_args = args.extra_args.strip().split() if args.extra_args.strip() else []

    out_xlsx = Path(args.out_xlsx).resolve()
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(args.tmp_dir).resolve() if args.tmp_dir else Path(tempfile.mkdtemp(prefix="rq2_cost_sweep_"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    selected_long_rows: List[pd.DataFrame] = []
    curve_long_rows: List[pd.DataFrame] = []
    run_log: List[Dict[str, Any]] = []

    for cost_budget in costs:
        tmp_out = tmp_root / f"tmp_cost_{int(round(cost_budget))}.xlsx"

        try:
            run_one_cost(
                python_exec=args.python,
                base_script=base_script,
                input_root=args.input_root,
                dataset_xlsx=args.dataset_xlsx,
                apps=args.apps,
                objective=args.objective,
                cost_budget=cost_budget,
                K=args.K,
                merge_sim_thresh=args.merge_sim_thresh,
                red_eps=args.red_eps,
                pool_pairs=args.pool_pairs,
                rand_seed=args.rand_seed,
                extra_args=extra_args,
                tmp_out_xlsx=tmp_out,
            )

            df_sel = read_sheet(tmp_out, "RQ2_Appendix_Selected_PerApp")
            df_curve = read_sheet(tmp_out, "RQ2_CostCurve_Long")


            df_sel.insert(0, "CostBudget_Req", float(cost_budget))
            df_sel.insert(1, "CostBudget_Tag", f"cost{int(round(cost_budget))}")
            df_curve.insert(0, "CostBudget_Req", float(cost_budget))
            df_curve.insert(1, "CostBudget_Tag", f"cost{int(round(cost_budget))}")


            if "#Selected" in df_sel.columns:
                nsel = df_sel["#Selected"].fillna(0).astype(int)
                total_pairs = nsel.apply(comb2)
                df_sel["SelPairsTotal"] = total_pairs

                for frac_col, out_col in [
                    ("SelClueFrac(J>eps)", "SelCluePosPairs(J>eps)"),
                    ("SelReportFrac(J>eps)", "SelReportPosPairs(J>eps)"),
                ]:
                    if frac_col in df_sel.columns:
                        frac = df_sel[frac_col].fillna(0.0).astype(float)
                        df_sel[out_col] = (frac * total_pairs).round().astype(int)

            selected_long_rows.append(df_sel)
            curve_long_rows.append(df_curve)

            run_log.append({
                "CostBudget_Req": float(cost_budget),
                "tmp_out_xlsx": str(tmp_out),
                "status": "ok"
            })

            if not args.keep_tmp:

                try:
                    tmp_out.unlink(missing_ok=True)
                except Exception:
                    pass

        except subprocess.CalledProcessError as e:
            run_log.append({
                "CostBudget_Req": float(cost_budget),
                "tmp_out_xlsx": str(tmp_out),
                "status": f"failed: {e}"
            })
            print(f"[WARN] cost_budget={cost_budget} failed: {e}", file=sys.stderr)


    if not selected_long_rows:
        raise RuntimeError("No successful runs. Check paths and base_script arguments.")

    df_selected_all = pd.concat(selected_long_rows, ignore_index=True)
    df_curve_all = pd.concat(curve_long_rows, ignore_index=True) if curve_long_rows else pd.DataFrame()
    df_log = pd.DataFrame(run_log)


    with pd.ExcelWriter(out_xlsx) as xw:
        df_selected_all.to_excel(xw, sheet_name="RQ2_Selected_PerApp_Long", index=False)
        if not df_curve_all.empty:
            df_curve_all.to_excel(xw, sheet_name="RQ2_CostCurve_Long_All", index=False)
        df_log.to_excel(xw, sheet_name="Meta_RunLog", index=False)

    print(f"[Done] Aggregated sweep -> {out_xlsx}")
    print(f"[TempDir] {tmp_root} (kept={args.keep_tmp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
