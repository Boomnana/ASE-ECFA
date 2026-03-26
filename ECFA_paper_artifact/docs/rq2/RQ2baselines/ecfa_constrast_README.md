# RQ2 Baseline and ECFA Contrast Guide

## 1. Expected inputs

The scripts assume a layout like the following:

- `ROOT_glm4.7/<AppName>/unify/nodes.xlsx`
- `ROOT_glm4.7/<AppName>/unify/edges.xlsx`
- an RQ2 dataset workbook with one sheet per app (optional for defect lookup; not required for the main flow)

## 2. Common single-app command

```bash
python ecfa_contrast.py   --apps "IELTS Listening"   --objective entity   --budget 50   --budget_mode cost   --cost_budget 0   --k 2   --r_max 200   --merge_sim_thresh 0.5   --max_view_report_ratio 0.5   --cap_methods "ECFA,ECFA_NoCost"   --warmup_frac 0.30   --cost_power_start 1.80   --gain_power 1.00   --fixed_cost 30   --plot_curves auto   --run_tag "ielts_entity_cap05_anneal"
```

## 3. Budget protocols

### Protocol A: `--budget_mode count`

The budget is simply Top-K: at most K views are selected.

### Protocol B: `--budget_mode cost`

A shared cost budget `B_cost` is used for all methods.

- If `--cost_budget > 0`, that value is used directly.
- If `--cost_budget = 0`, the script first runs ECFA Top-K and uses the total ECFA cost as the shared budget.

Under cost mode, methods may stop before reaching K if the budget would be exceeded.
