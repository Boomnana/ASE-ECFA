from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .aggregate_stats import (
    BootstrapConfig,
    app_query_balance,
    infer_metric_cols,
    pairwise_app_wins,
    summarize_app_macro,
    summarize_app_means,
    summarize_app_median,
    summarize_app_report_macro,
    summarize_query_macro,
    summarize_recall_report_micro,
    hierarchical_bootstrap_delta,
)
from .utils import ensure_dir, write_csv, write_json


DEFAULT_BOOTSTRAP_PAIRS = (
    ('ecfa', 'global'),
    ('ecfa', 'multi_L5'),
    ('multi_L5', 'intra_only'),
)


def export_subset_enhanced_stats(
    subset_dir: str | Path,
    *,
    bootstrap_cfg: BootstrapConfig,
    bootstrap_pairs: Sequence[tuple[str, str]] = DEFAULT_BOOTSTRAP_PAIRS,
) -> list[Path]:
    subset_dir = Path(subset_dir)
    perq_path = subset_dir / 'per_query_metrics.csv'
    outputs: list[Path] = []
    if not perq_path.exists():
        return outputs
    per_query = pd.read_csv(perq_path)
    stats_dir = ensure_dir(subset_dir / 'enhanced_stats')
    metric_cols = infer_metric_cols(per_query)
    recall_cols = [c for c in metric_cols if c.startswith('recall_at_')]

    if len(per_query) == 0:
        outputs.append(write_csv(pd.DataFrame(columns=['method', 'n_eval_queries', 'aggregation'] + metric_cols), stats_dir / 'query_macro_summary.csv'))
        outputs.append(write_csv(pd.DataFrame(columns=['app', 'method', 'n_eval_queries'] + metric_cols), stats_dir / 'by_app_summary_recomputed.csv'))
        outputs.append(write_csv(pd.DataFrame(columns=['method', 'n_apps', 'aggregation'] + metric_cols), stats_dir / 'app_macro_summary.csv'))
        outputs.append(write_csv(pd.DataFrame(columns=['method', 'n_apps', 'aggregation'] + metric_cols), stats_dir / 'app_median_summary.csv'))
        outputs.append(write_csv(pd.DataFrame(columns=['method', 'n_eval_queries', 'total_rel_items', 'aggregation'] + recall_cols), stats_dir / 'report_micro_recall_summary.csv'))
        outputs.append(write_csv(pd.DataFrame(columns=['app', 'method', 'n_eval_queries', 'total_rel_items'] + recall_cols), stats_dir / 'by_app_report_micro_recall.csv'))
        outputs.append(write_csv(pd.DataFrame(columns=['method', 'n_apps', 'aggregation'] + recall_cols), stats_dir / 'app_macro_report_micro_recall_summary.csv'))
        outputs.append(write_csv(pd.DataFrame(columns=['app', 'n_eval_queries', 'query_share', 'balance_vs_equal']), stats_dir / 'app_query_balance.csv'))
        outputs.append(
            write_csv(
                pd.DataFrame(
                    columns=[
                        'metric',
                        'method_a',
                        'method_b',
                        'n_apps',
                        'wins_a',
                        'ties',
                        'wins_b',
                        'mean_delta_app',
                        'median_delta_app',
                        'min_delta_app',
                        'max_delta_app',
                    ]
                ),
                stats_dir / 'pairwise_app_wins.csv',
            )
        )
        validation = {
            'has_per_query': True,
            'n_rows_per_query': int(per_query.shape[0]),
            'methods': [],
            'metrics': metric_cols,
            'bootstrap_n': int(bootstrap_cfg.n_boot),
            'bootstrap_enabled': bool(int(bootstrap_cfg.n_boot) > 0),
        }
        outputs.append(write_json(validation, stats_dir / 'validation_report.json'))
        return outputs

    query_macro = summarize_query_macro(per_query)
    outputs.append(write_csv(query_macro, stats_dir / 'query_macro_summary.csv'))

    by_app = summarize_app_means(per_query)
    outputs.append(write_csv(by_app, stats_dir / 'by_app_summary_recomputed.csv'))

    app_macro = summarize_app_macro(per_query)
    outputs.append(write_csv(app_macro, stats_dir / 'app_macro_summary.csv'))

    app_median = summarize_app_median(per_query)
    outputs.append(write_csv(app_median, stats_dir / 'app_median_summary.csv'))

    report_micro = summarize_recall_report_micro(per_query)
    outputs.append(write_csv(report_micro, stats_dir / 'report_micro_recall_summary.csv'))

    by_app_rm, app_macro_rm = summarize_app_report_macro(per_query)
    outputs.append(write_csv(by_app_rm, stats_dir / 'by_app_report_micro_recall.csv'))
    outputs.append(write_csv(app_macro_rm, stats_dir / 'app_macro_report_micro_recall_summary.csv'))

    balance = app_query_balance(per_query)
    outputs.append(write_csv(balance, stats_dir / 'app_query_balance.csv'))

    pairwise = pairwise_app_wins(by_app, metric_cols=metric_cols)
    outputs.append(write_csv(pairwise, stats_dir / 'pairwise_app_wins.csv'))

    if int(bootstrap_cfg.n_boot) > 0:
        boot_rows: list[dict] = []
        focus_metrics = [m for m in ['success_at_20', 'mrr_at_20', 'recall_at_20', 'recall_at_50', 'ndcg_at_20'] if m in metric_cols]
        for a, b in bootstrap_pairs:
            for metric in focus_metrics:
                for mode in ['query_macro', 'app_macro']:
                    boot_rows.append(
                        hierarchical_bootstrap_delta(
                            per_query,
                            metric=metric,
                            method_a=a,
                            method_b=b,
                            mode=mode,
                            cfg=bootstrap_cfg,
                        )
                    )
        outputs.append(write_csv(pd.DataFrame(boot_rows), stats_dir / 'bootstrap_pairwise_deltas.csv'))

    validation = {
        'has_per_query': True,
        'n_rows_per_query': int(per_query.shape[0]),
        'methods': sorted(per_query['method'].astype(str).unique().tolist()),
        'metrics': metric_cols,
        'bootstrap_n': int(bootstrap_cfg.n_boot),
        'bootstrap_enabled': bool(int(bootstrap_cfg.n_boot) > 0),
    }
    outputs.append(write_json(validation, stats_dir / 'validation_report.json'))
    return outputs
