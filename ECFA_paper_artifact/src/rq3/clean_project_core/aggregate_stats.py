from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .utils import ensure_dir, write_csv

KEY_METRICS = [
    'success_at_10',
    'success_at_20',
    'mrr_at_20',
    'recall_at_20',
    'recall_at_50',
    'ndcg_at_20',
]


@dataclass(frozen=True)
class BootstrapConfig:
    n_boot: int = 2000
    seed: int = 20260309
    ci: float = 0.95


def infer_metric_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(('success_at_', 'mrr_at_', 'recall_at_', 'ndcg_at_'))]
    order = [c for c in KEY_METRICS if c in cols]
    rest = [c for c in cols if c not in order]
    return order + sorted(rest)


def infer_rel_count_col(df: pd.DataFrame) -> str:
    for c in ['n_rel', 'n_cross_rel']:
        if c in df.columns:
            return c
    matches = [c for c in df.columns if c.startswith('n_') and c.endswith('_rel')]
    if matches:
        return matches[0]
    raise ValueError('No relation-count column found in per_query metrics.')


def summarize_query_macro(per_query: pd.DataFrame) -> pd.DataFrame:
    metric_cols = infer_metric_cols(per_query)
    rows: list[dict] = []
    for method, g in per_query.groupby('method', sort=True):
        row = {'method': method, 'n_eval_queries': int(g.shape[0]), 'aggregation': 'query_macro'}
        for c in metric_cols:
            row[c] = float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_app_means(per_query: pd.DataFrame) -> pd.DataFrame:
    metric_cols = infer_metric_cols(per_query)
    rows: list[dict] = []
    for (app, method), g in per_query.groupby(['app', 'method'], sort=True):
        row = {'app': app, 'method': method, 'n_eval_queries': int(g.shape[0])}
        for c in metric_cols:
            row[c] = float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_app_macro(per_query: pd.DataFrame) -> pd.DataFrame:
    by_app = summarize_app_means(per_query)
    metric_cols = infer_metric_cols(by_app)
    rows: list[dict] = []
    for method, g in by_app.groupby('method', sort=True):
        row = {'method': method, 'n_apps': int(g['app'].nunique()), 'aggregation': 'app_macro'}
        for c in metric_cols:
            row[c] = float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_app_median(per_query: pd.DataFrame) -> pd.DataFrame:
    by_app = summarize_app_means(per_query)
    metric_cols = infer_metric_cols(by_app)
    rows: list[dict] = []
    for method, g in by_app.groupby('method', sort=True):
        row = {'method': method, 'n_apps': int(g['app'].nunique()), 'aggregation': 'app_median'}
        for c in metric_cols:
            row[c] = float(g[c].median())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_recall_report_micro(per_query: pd.DataFrame) -> pd.DataFrame:
    rel_count_col = infer_rel_count_col(per_query)
    recall_cols = [c for c in infer_metric_cols(per_query) if c.startswith('recall_at_')]
    rows: list[dict] = []
    for method, g in per_query.groupby('method', sort=True):
        total_rel = float(g[rel_count_col].sum())
        row = {
            'method': method,
            'n_eval_queries': int(g.shape[0]),
            'total_rel_items': int(total_rel),
            'aggregation': 'report_micro',
        }
        for c in recall_cols:
            hits = (g[c] * g[rel_count_col]).sum()
            row[c] = float(hits / total_rel) if total_rel > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_app_report_macro(per_query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel_count_col = infer_rel_count_col(per_query)
    recall_cols = [c for c in infer_metric_cols(per_query) if c.startswith('recall_at_')]
    rows: list[dict] = []
    for (app, method), g in per_query.groupby(['app', 'method'], sort=True):
        total_rel = float(g[rel_count_col].sum())
        row = {'app': app, 'method': method, 'n_eval_queries': int(g.shape[0]), 'total_rel_items': int(total_rel)}
        for c in recall_cols:
            hits = (g[c] * g[rel_count_col]).sum()
            row[c] = float(hits / total_rel) if total_rel > 0 else 0.0
        rows.append(row)
    by_app = pd.DataFrame(rows)
    out_rows: list[dict] = []
    if len(by_app):
        for method, g in by_app.groupby('method', sort=True):
            row = {'method': method, 'n_apps': int(g['app'].nunique()), 'aggregation': 'app_macro_report_micro'}
            for c in recall_cols:
                row[c] = float(g[c].mean())
            out_rows.append(row)
    return by_app, pd.DataFrame(out_rows)


def app_query_balance(per_query: pd.DataFrame) -> pd.DataFrame:
    counts = (
        per_query.groupby('app', sort=True)['query_id']
        .nunique()
        .rename('n_eval_queries')
        .reset_index()
        .sort_values('n_eval_queries', ascending=False)
        .reset_index(drop=True)
    )
    total = float(counts['n_eval_queries'].sum()) or 1.0
    counts['query_share'] = counts['n_eval_queries'] / total
    counts['balance_vs_equal'] = counts['query_share'] / (1.0 / max(len(counts), 1))
    return counts


def pairwise_app_wins(by_app: pd.DataFrame, metric_cols: Sequence[str], ties_tol: float = 1e-12) -> pd.DataFrame:
    rows: list[dict] = []
    methods = sorted(by_app['method'].unique().tolist())
    for metric in metric_cols:
        piv = by_app.pivot(index='app', columns='method', values=metric)
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                if a not in piv.columns or b not in piv.columns:
                    continue
                delta = (piv[a] - piv[b]).dropna()
                if delta.empty:
                    continue
                wins = int((delta > ties_tol).sum())
                losses = int((delta < -ties_tol).sum())
                ties = int(delta.shape[0] - wins - losses)
                rows.append({
                    'metric': metric,
                    'method_a': a,
                    'method_b': b,
                    'n_apps': int(delta.shape[0]),
                    'wins_a': wins,
                    'ties': ties,
                    'wins_b': losses,
                    'mean_delta_app': float(delta.mean()),
                    'median_delta_app': float(delta.median()),
                    'min_delta_app': float(delta.min()),
                    'max_delta_app': float(delta.max()),
                })
    return pd.DataFrame(rows)


def _paired_delta_table(per_query: pd.DataFrame, metric: str, method_a: str, method_b: str) -> pd.DataFrame:
    piv = per_query.pivot_table(index=['app', 'query_id'], columns='method', values=metric, aggfunc='first')
    piv = piv[[c for c in [method_a, method_b] if c in piv.columns]].dropna()
    if method_a not in piv.columns or method_b not in piv.columns:
        return pd.DataFrame(columns=['app', 'query_id', 'delta'])
    out = piv.reset_index()[['app', 'query_id']].copy()
    out['delta'] = piv[method_a].to_numpy() - piv[method_b].to_numpy()
    return out


def hierarchical_bootstrap_delta(
    per_query: pd.DataFrame,
    *,
    metric: str,
    method_a: str,
    method_b: str,
    mode: str,
    cfg: BootstrapConfig,
) -> dict:
    if mode not in {'query_macro', 'app_macro'}:
        raise ValueError(f'Unsupported bootstrap mode: {mode}')
    delta_tbl = _paired_delta_table(per_query, metric, method_a, method_b)
    if delta_tbl.empty:
        return {
            'metric': metric,
            'method_a': method_a,
            'method_b': method_b,
            'mode': mode,
            'n_apps': 0,
            'n_pairs': 0,
            'observed_delta': np.nan,
            'ci_low': np.nan,
            'ci_high': np.nan,
            'p_nonpositive': np.nan,
            'bootstrap_mean': np.nan,
            'n_boot': int(cfg.n_boot),
        }

    apps = sorted(delta_tbl['app'].unique().tolist())
    app_to_vals = {app: grp['delta'].to_numpy(dtype=float) for app, grp in delta_tbl.groupby('app')}

    if mode == 'query_macro':
        obs = float(delta_tbl['delta'].mean())
    else:
        obs = float(delta_tbl.groupby('app')['delta'].mean().mean())

    rng = np.random.default_rng(cfg.seed)
    draws = np.empty(int(cfg.n_boot), dtype=float)
    for b in range(int(cfg.n_boot)):
        sampled_apps = rng.choice(apps, size=len(apps), replace=True)
        if mode == 'query_macro':
            sampled_vals = []
            for app in sampled_apps:
                vals = app_to_vals[app]
                idx = rng.integers(0, len(vals), size=len(vals))
                sampled_vals.append(vals[idx])
            draws[b] = float(np.concatenate(sampled_vals).mean())
        else:
            app_means = []
            for app in sampled_apps:
                vals = app_to_vals[app]
                idx = rng.integers(0, len(vals), size=len(vals))
                app_means.append(float(vals[idx].mean()))
            draws[b] = float(np.mean(app_means))

    alpha = 1.0 - float(cfg.ci)
    low = float(np.quantile(draws, alpha / 2.0))
    high = float(np.quantile(draws, 1.0 - alpha / 2.0))
    p_nonpositive = float(np.mean(draws <= 0.0))
    return {
        'metric': metric,
        'method_a': method_a,
        'method_b': method_b,
        'mode': mode,
        'n_apps': int(len(apps)),
        'n_pairs': int(delta_tbl.shape[0]),
        'observed_delta': obs,
        'ci_low': low,
        'ci_high': high,
        'p_nonpositive': p_nonpositive,
        'bootstrap_mean': float(draws.mean()),
        'n_boot': int(cfg.n_boot),
    }


def export_dual_protocol_stats(
    run_dir: str | Path,
    *,
    protocols: Iterable[str] = ('all_related', 'cross_related'),
    bootstrap_cfg: BootstrapConfig = BootstrapConfig(),
    bootstrap_pairs: Sequence[tuple[str, str]] = (
        ('ecfa_hybrid_allrel', 'global'),
        ('ecfa_hybrid_cross', 'global'),
        ('ecfa_head', 'global'),
        ('multi_L5', 'global'),
    ),
) -> list[Path]:
    run_dir = Path(run_dir)
    outputs: list[Path] = []
    stats_dir = ensure_dir(run_dir / 'enhanced_stats')

    summary_registry: dict[str, dict[str, pd.DataFrame]] = {}

    for protocol in protocols:
        perq_path = run_dir / protocol / 'per_query_metrics.csv'
        if not perq_path.exists():
            continue
        protocol_dir = ensure_dir(stats_dir / protocol)
        per_query = pd.read_csv(perq_path)
        metric_cols = infer_metric_cols(per_query)

        query_macro = summarize_query_macro(per_query)
        outputs.append(write_csv(query_macro, protocol_dir / 'query_macro_summary.csv'))

        by_app = summarize_app_means(per_query)
        outputs.append(write_csv(by_app, protocol_dir / 'by_app_summary_recomputed.csv'))

        app_macro = summarize_app_macro(per_query)
        outputs.append(write_csv(app_macro, protocol_dir / 'app_macro_summary.csv'))

        app_median = summarize_app_median(per_query)
        outputs.append(write_csv(app_median, protocol_dir / 'app_median_summary.csv'))

        recall_report_micro = summarize_recall_report_micro(per_query)
        outputs.append(write_csv(recall_report_micro, protocol_dir / 'report_micro_recall_summary.csv'))

        by_app_report_micro, app_macro_report_micro = summarize_app_report_macro(per_query)
        outputs.append(write_csv(by_app_report_micro, protocol_dir / 'by_app_report_micro_recall.csv'))
        outputs.append(write_csv(app_macro_report_micro, protocol_dir / 'app_macro_report_micro_recall_summary.csv'))

        summary_registry[protocol] = {
            'query_macro': query_macro,
            'app_macro': app_macro,
            'report_micro_recall': recall_report_micro,
            'app_macro_report_micro_recall': app_macro_report_micro,
        }

        balance = app_query_balance(per_query)
        outputs.append(write_csv(balance, protocol_dir / 'app_query_balance.csv'))

        pairwise = pairwise_app_wins(by_app, metric_cols=metric_cols)
        outputs.append(write_csv(pairwise, protocol_dir / 'pairwise_app_wins.csv'))

        boot_rows: list[dict] = []
        for a, b in bootstrap_pairs:
            for metric in ['success_at_20', 'mrr_at_20', 'recall_at_20', 'recall_at_50', 'ndcg_at_20']:
                if metric not in metric_cols:
                    continue
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
        outputs.append(write_csv(pd.DataFrame(boot_rows), protocol_dir / 'bootstrap_pairwise_deltas.csv'))

    if 'all_related' in summary_registry and 'cross_related' in summary_registry:
        for name in ['query_macro', 'app_macro', 'report_micro_recall', 'app_macro_report_micro_recall']:
            all_df = summary_registry['all_related'].get(name)
            cross_df = summary_registry['cross_related'].get(name)
            if all_df is None or cross_df is None or all_df.empty or cross_df.empty:
                continue
            key_cols = [c for c in ['method'] if c in all_df.columns and c in cross_df.columns]
            merged = all_df.merge(cross_df, on=key_cols, suffixes=('_all', '_cross'))
            rows: list[dict] = []
            for _, r in merged.iterrows():
                row = {'aggregation': name}
                for k in key_cols:
                    row[k] = r[k]
                for metric in ['success_at_20', 'mrr_at_20', 'recall_at_20', 'recall_at_50', 'ndcg_at_20']:
                    a_col = metric + '_all'
                    c_col = metric + '_cross'
                    if a_col not in merged.columns or c_col not in merged.columns:
                        continue
                    a = float(r[a_col])
                    c = float(r[c_col])
                    row[a_col] = a
                    row[c_col] = c
                    row[metric + '_retention'] = float(c / a) if abs(a) > 1e-12 else 0.0
                rows.append(row)
            if rows:
                outputs.append(write_csv(pd.DataFrame(rows), stats_dir / f'retention_{name}.csv'))
    return outputs
