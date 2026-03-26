# Recommended Execution Order

1. Run the issue-label audit.
2. Run the TFIDF dual-protocol experiment.
3. Optionally run the SBERT dual-protocol experiment.

## Issue-label audit

```powershell
python scripts/audit_issue_labels.py `
  --xlsx "outputs/rq3_filtered_ecfa/filtered_reports.xlsx" `
  --out_dir "outputs/issue_audit"
```

## TFIDF dual protocol

```powershell
python scripts/run_rq3_dual_protocol.py `
  --xlsx "outputs/rq3_filtered_ecfa/filtered_reports.xlsx" `
  --ecfa_per_query_csv "outputs/rq3_2a_fresh/20260308_143007_ecfa/ecfa/per_query_K20_ecfa.csv" `
  --clusters_csv "outputs/rq3_2a_fresh/20260308_143007_ecfa/boundaries/clusters_tfidf_K20.csv" `
  --out_dir "outputs/rq3_dual_protocol_tfidf"
```
