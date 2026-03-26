# RQ1 Quick Start

## Minimal directory layout

```text
ROOT/
  App1/
    no_unify/nodes.xlsx
    no_unify/edges.xlsx
    unify/nodes.xlsx
    unify/edges.xlsx
    synonym_mapping.xlsx   # optional
  App2/
    ...
```

## One-command example

```powershell
python rq1_export.py `
  --root "C:/path/to/ROOT_glm4air" `
  --out "C:/path/to/rq1_entity_unification_results.xlsx" `
  --K 50 `
  --khop 2 `
  --universe ISSUE PHEN DIAG `
  --max_candidates 400 `
  --seed_test_n 20
```

## Expected outputs

- `rq1_entity_unification_results.xlsx`: connectivity, CER, coverage-curve, AUC, and delta tables
- `coverage_curve_<app>.png`: one figure per app
- `coverage_curve_across_apps.png`: across-app mean ± std figure
- `FIGURES` worksheet inside the Excel report with embedded plots
