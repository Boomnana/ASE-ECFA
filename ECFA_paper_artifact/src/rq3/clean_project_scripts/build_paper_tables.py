from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils import ensure_dir, write_csv

BOUNDARY_LABEL = {
    'tfidf_hc_K20': 'TFIDF+HC',
    'sbert_hc_K20': 'SBERT+HC',
}
METHOD_ORDER = ['intra_only', 'multi_L2', 'multi_L3', 'multi_L5', 'global', 'ecfa']
METHOD_LABEL = {
    'intra_only': 'Intra-only',
    'multi_L2': 'MultiCluster-L2',
    'multi_L3': 'MultiCluster-L3',
    'multi_L5': 'MultiCluster-L5',
    'global': 'Global',
    'ecfa': 'ECFA',
}


def _boundary_label(name: str) -> str:
    return BOUNDARY_LABEL.get(name, 'LLMCluster' if 'llm' in name.lower() else name)


def _boundary_dirs(run_dir: Path) -> list[Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name not in {'corpus', 'boundaries'}])


def _read_summary(subset_dir: Path, aggregation: str) -> pd.DataFrame:
    if aggregation == 'app_macro':
        p = subset_dir / 'enhanced_stats' / 'app_macro_summary.csv'
    elif aggregation == 'query_macro':
        p = subset_dir / 'enhanced_stats' / 'query_macro_summary.csv'
    else:
        raise ValueError(aggregation)
    return pd.read_csv(p if p.exists() else subset_dir / 'summary.csv')


def _to_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    spec = 'l' * len(cols)
    lines = [
        '\\begin{table*}[t]',
        '  \\centering',
        f'  \\caption{{{caption}}}',
        f'  \\label{{{label}}}',
        '  \\footnotesize',
        '  \\setlength{\\tabcolsep}{4pt}',
        '  \\renewcommand{\\arraystretch}{1.10}',
        f'  \\begin{{tabular}}{{{spec}}}',
        '    \\toprule',
        '    ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' \\\\',
        '    \\midrule',
    ]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f'{float(v):.4f}' if isinstance(v, float) else str(v))
        lines.append('    ' + ' & '.join(vals) + ' \\\\')
    lines += ['    \\bottomrule', '  \\end{tabular}', '\\end{table*}', '']
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description='Build paper-facing RQ3-2A tables.')
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--table_dir', default='table')
    ap.add_argument('--subset', default='all_cross')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    table_dir = ensure_dir(args.table_dir)

    diag_rows = []
    head_rows = []
    recall_rows = []
    gain_rows = []
    robust_rows = []

    for bdir in _boundary_dirs(run_dir):
        blabel = _boundary_label(bdir.name)
        diag = pd.read_csv(bdir / 'boundary_diagnosis.csv').iloc[0]
        diag_rows.append({
            'Boundary': blabel,
            '#Queries': int(diag['n_queries']),
            '#Cross Queries': int(diag['cross_query_count']),
            'Cross Query Ratio': float(diag['frac_cross_query']),
            'Avg. #Cross Relevant / Query': float(diag['avg_cross_rel']),
            'Median #Cross Relevant / Query': float(diag.get('median_cross_rel', 0.0)),
        })

        subset_dir = bdir / args.subset
        app_df = _read_summary(subset_dir, 'app_macro')
        query_df = _read_summary(subset_dir, 'query_macro')
        wins_path = subset_dir / 'enhanced_stats' / 'pairwise_app_wins.csv'
        wins_df = pd.read_csv(wins_path) if wins_path.exists() else pd.DataFrame()
        boot_path = subset_dir / 'enhanced_stats' / 'bootstrap_pairwise_deltas.csv'
        boot_df = pd.read_csv(boot_path) if boot_path.exists() else pd.DataFrame()

        use = app_df[app_df['method'].isin(METHOD_ORDER)].copy()
        use['method'] = pd.Categorical(use['method'], METHOD_ORDER, ordered=True)
        use = use.sort_values('method')
        for _, r in use.iterrows():
            head_rows.append({
                'Boundary': blabel,
                'Method': METHOD_LABEL[r['method']],
                'Success@20': float(r['success_at_20']),
                'MRR@20': float(r['mrr_at_20']),
                'nDCG@20': float(r['ndcg_at_20']),
            })
            recall_rows.append({
                'Boundary': blabel,
                'Method': METHOD_LABEL[r['method']],
                'Recall@20': float(r['recall_at_20']),
                'Recall@50': float(r['recall_at_50']),
            })

        intra = use[use['method'] == 'intra_only']
        if not intra.empty:
            intra = intra.iloc[0]
            for mk in ['multi_L2', 'multi_L3', 'multi_L5', 'global', 'ecfa']:
                rr = use[use['method'] == mk]
                if rr.empty:
                    continue
                rr = rr.iloc[0]
                gain_rows.append({
                    'Boundary': blabel,
                    'Method': METHOD_LABEL[mk],
                    'Delta Success@20': float(rr['success_at_20'] - intra['success_at_20']),
                    'Delta Recall@20': float(rr['recall_at_20'] - intra['recall_at_20']),
                    'Delta Recall@50': float(rr['recall_at_50'] - intra['recall_at_50']),
                })

        q_use = query_df[query_df['method'].isin(['intra_only', 'multi_L5', 'global', 'ecfa'])].copy()
        q_use['method'] = pd.Categorical(q_use['method'], ['intra_only', 'multi_L5', 'global', 'ecfa'], ordered=True)
        q_use = q_use.sort_values('method')
        for _, r in q_use.iterrows():
            wins = ''
            if len(wins_df):
                hit = wins_df[(wins_df['method_a'] == r['method']) & (wins_df['method_b'] == 'global') & (wins_df['metric'] == 'success_at_20')]
                if len(hit):
                    wins = int(hit.iloc[0]['wins_a'])
            delta_ci = ''
            if len(boot_df):
                hit = boot_df[(boot_df['method_a'] == r['method']) & (boot_df['method_b'] == 'global') & (boot_df['metric'] == 'success_at_20') & (boot_df['mode'] == 'app_macro')]
                if len(hit):
                    delta_ci = f"{float(hit.iloc[0]['delta']):.4f} [{float(hit.iloc[0]['ci_low']):.4f}, {float(hit.iloc[0]['ci_high']):.4f}]"
            robust_rows.append({
                'Boundary': blabel,
                'Method': METHOD_LABEL[r['method']],
                'Query-Macro Success@20': float(r['success_at_20']),
                'App-Macro Success@20': float(use[use['method'] == r['method']]['success_at_20'].iloc[0]),
                'Per-App Wins vs Global': wins,
                'Bootstrap Δ vs Global': delta_ci,
            })

    files = {
        'Table_X_boundary_diagnosis.csv': pd.DataFrame(diag_rows),
        'Table_Y_crossrel_head_app_macro.csv': pd.DataFrame(head_rows),
        'Table_Z_crossrel_recall_app_macro.csv': pd.DataFrame(recall_rows),
        'Table_W_multicluster_gain_over_intra.csv': pd.DataFrame(gain_rows),
        'Table_V_robustness_success20.csv': pd.DataFrame(robust_rows),
    }
    captions = {
        'Table_X_boundary_diagnosis.csv': ('RQ3-2A boundary diagnosis.', 'tab:rq3_2a_boundary_diagnosis'),
        'Table_Y_crossrel_head_app_macro.csv': ('RQ3-2A CrossRel head metrics (app-macro).', 'tab:rq3_2a_head_app_macro'),
        'Table_Z_crossrel_recall_app_macro.csv': ('RQ3-2A CrossRel recall metrics (app-macro).', 'tab:rq3_2a_recall_app_macro'),
        'Table_W_multicluster_gain_over_intra.csv': ('RQ3-2A gains over Intra-only.', 'tab:rq3_2a_gain_over_intra'),
        'Table_V_robustness_success20.csv': ('RQ3-2A robustness summary for Success@20.', 'tab:rq3_2a_robustness_success20'),
    }

    for name, df in files.items():
        write_csv(df, table_dir / name)
        cap, lab = captions[name]
        (table_dir / name.replace('.csv', '.tex')).write_text(_to_tex(df, cap, lab), encoding='utf-8')


if __name__ == '__main__':
    main()
