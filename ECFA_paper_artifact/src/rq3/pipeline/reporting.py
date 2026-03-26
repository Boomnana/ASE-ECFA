from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ensure_dir, write_csv

MAIN_SUBSET = 'anchor_eligible'
SUBSET_ORDER = ['anchor_eligible', 'text_hard', 'dedup', 'all_cross']


def export_main_tables(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    eval_dir = run_dir / 'eval'
    out_dir = ensure_dir(run_dir / 'tables')
    outputs: list[Path] = []
    all_summary = []
    for bdir in sorted([p for p in eval_dir.iterdir() if p.is_dir()]):
        for subset in SUBSET_ORDER:
            sfile = bdir / subset / 'summary.csv'
            byfile = bdir / subset / 'by_app_summary.csv'
            if not sfile.exists():
                continue
            s = pd.read_csv(sfile)
            s['boundary'] = bdir.name
            s['subset'] = subset
            main = s[['boundary', 'subset', 'method', 'n_eval_queries', 'success_at_10', 'success_at_20', 'mrr_at_20', 'recall_at_20', 'recall_at_50', 'ndcg_at_20']].copy()
            outputs.append(write_csv(main, out_dir / f'Table_RQ3_2A_{subset}_{bdir.name}.csv'))
            all_summary.append(main)
            if byfile.exists():
                by_app = pd.read_csv(byfile)
                by_app['boundary'] = bdir.name
                by_app['subset'] = subset
                outputs.append(write_csv(by_app[['boundary', 'subset', 'app', 'method', 'n_eval_queries', 'success_at_20', 'mrr_at_20', 'recall_at_20', 'recall_at_50', 'ndcg_at_20']], out_dir / f'Table_RQ3_2A_by_app_{subset}_{bdir.name}.csv'))
        mainfile = bdir / MAIN_SUBSET / 'summary.csv'
        if mainfile.exists():
            main = pd.read_csv(mainfile)
            main['boundary'] = bdir.name
            outputs.append(write_csv(main[['boundary', 'method', 'n_eval_queries', 'success_at_10', 'success_at_20', 'mrr_at_20', 'recall_at_20', 'recall_at_50', 'ndcg_at_20']], out_dir / f'Table_RQ3_2A_MAIN_{bdir.name}.csv'))
    if all_summary:
        outputs.append(write_csv(pd.concat(all_summary, ignore_index=True), out_dir / 'Table_RQ3_2A_all_boundaries_all_subsets.csv'))
    return outputs


def export_plot_data(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    eval_dir = run_dir / 'eval'
    out_dir = ensure_dir(run_dir / 'plot_data')
    outputs: list[Path] = []
    success_rows = []
    recall_rows = []
    head_rows = []
    for bdir in sorted([p for p in eval_dir.iterdir() if p.is_dir()]):
        for subset in SUBSET_ORDER:
            sfile = bdir / subset / 'summary.csv'
            if not sfile.exists():
                continue
            s = pd.read_csv(sfile)
            boundary = bdir.name
            for _, r in s.iterrows():
                success_rows.append({'boundary': boundary, 'subset': subset, 'method': r['method'], 'k': 10, 'metric': 'Success', 'value': r['success_at_10']})
                success_rows.append({'boundary': boundary, 'subset': subset, 'method': r['method'], 'k': 20, 'metric': 'Success', 'value': r['success_at_20']})
                recall_rows.append({'boundary': boundary, 'subset': subset, 'method': r['method'], 'k': 20, 'metric': 'Recall', 'value': r['recall_at_20']})
                recall_rows.append({'boundary': boundary, 'subset': subset, 'method': r['method'], 'k': 50, 'metric': 'Recall', 'value': r['recall_at_50']})
                head_rows.append({'boundary': boundary, 'subset': subset, 'method': r['method'], 'metric': 'MRR@20', 'value': r['mrr_at_20']})
                head_rows.append({'boundary': boundary, 'subset': subset, 'method': r['method'], 'metric': 'nDCG@20', 'value': r['ndcg_at_20']})
    if success_rows:
        outputs.append(write_csv(pd.DataFrame(success_rows), out_dir / 'PlotData_RQ3_2A_success_curve.csv'))
    if recall_rows:
        outputs.append(write_csv(pd.DataFrame(recall_rows), out_dir / 'PlotData_RQ3_2A_recall_curve.csv'))
    if head_rows:
        outputs.append(write_csv(pd.DataFrame(head_rows), out_dir / 'PlotData_RQ3_2A_head_metrics.csv'))
    return outputs
