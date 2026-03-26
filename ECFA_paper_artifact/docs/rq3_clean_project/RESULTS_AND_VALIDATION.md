# Results and Validation Summary

## 1. What was fully rebuilt and re-run

### TFIDF+HC

The clean project re-ran the full **TFIDF+HC + ECFA** RQ3-2A pipeline using:

- `input/filtered_reports.csv`
- `input/boundaries/clusters_tfidf_hc_K20.csv`
- `input/ecfa/per_query_K20_ecfa.csv`

The new output is:

```text
output/run_main/tfidf_hc_K20/
```

The following comparison was performed against the previous validated reference file:

```text
input/reference/tfidf_cross_related_reference.csv
```

Validation result:

- `intra_only`: exact match
- `multi_L3`: exact match
- `multi_L5`: exact match
- `global`: exact match
- `ecfa` (= old `ecfa_hybrid_cross`): exact match

See:

```text
output/run_main/tfidf_validation.json
output/run_main/validation_summary.json
```

---

## 2. SBERT status in this delivery

### What was tested

A dedicated offline smoke test was executed on the provided model archive:

```text
script/smoke_test_sbert_model.py
```

Test result:

```text
output/sbert_smoke.json
```

The smoke test failed because the uploaded model archive does **not** contain the tokenizer assets required for local encoding.

### Consequence

Because the model archive is incomplete for offline inference, the clean project could not truly re-encode the corpus with SBERT inside this container.

Therefore, in this delivery:

- the **code path for SBERT is preserved**
- the **clean output structure for SBERT is preserved**
- the **SBERT result folder in `output/run_main/sbert_hc_K20/` is imported from the previously validated legacy run**
- shared baseline metrics were checked against the old reference and match exactly

Validated shared baselines:

- `intra_only`: exact match
- `multi_L3`: exact match
- `multi_L5`: exact match
- `global`: exact match

See:

```text
output/run_main/validation_summary.json
```

---

## 3. Why SBERT `ecfa` is not fully regenerated here

This clean project defines:

- `ecfa = ecfa_hybrid_cross`

However, the previous SBERT run available in the legacy project does **not** include a regenerated `ecfa_hybrid_cross` output under the current clean pipeline contract.

Since the offline model cannot encode locally in this container, this delivery does **not** fabricate a new SBERT `ecfa` row.

That is intentional.

The project remains honest and keeps:

- TFIDF canonical and exact
- SBERT shared baselines exact
- SBERT full re-run pending a complete offline model snapshot

---

## 4. Paper-facing files already generated

The clean project has already generated the paper-facing tables and plot-ready data:

```text
table/Table_X_boundary_diagnosis.csv
 table/Table_Y_crossrel_head_app_macro.csv
 table/Table_Z_crossrel_recall_app_macro.csv
 table/Table_W_multicluster_gain_over_intra.csv
 table/Table_V_robustness_success20.csv

 table/Figure_1_cross_query_ratio.csv
 table/Figure_2_multicluster_curve.csv
 table/Figure_3_method_comparison.csv
```

and corresponding quick-look `.tex` / `.png` outputs.

---

## 5. Bottom line

This delivery gives you:

1. a clean RQ3-2A codebase with `core/ + script/ + input/ + output/ + table/`
2. an exact, re-run, validated **TFIDF+HC** pipeline
3. a structurally compatible **SBERT** pipeline entry and output scaffold
4. explicit evidence that the provided SBERT model archive is incomplete for local offline re-encoding
5. paper-facing table/plot outputs already aligned to the markdown template
