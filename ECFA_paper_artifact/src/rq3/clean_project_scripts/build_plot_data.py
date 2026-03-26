from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils import ensure_dir, write_csv

BOUNDARY_LABEL = {'tfidf_hc_K20': 'TFIDF+HC', 'sbert_hc_K20': 'SBERT+HC'}
METHOD_ORDER = ['intra_only', 'multi_L2', 'multi_L3', 'multi_L5', 'global', 'ecfa']
METHOD_LABEL = {'intra_only': 'Intra-only', 'multi_L2': 'L2', 'multi_L3': 'L3', 'multi_L5': 'L5', 'global': 'Global', 'ecfa': 'ECFA'}


def _boundary_label(name: str) -> str:
    return BOUNDARY_LABEL.get(name, 'LLMCluster' if 'llm' in name.lower() else name)


def _boundary_dirs(run_dir: Path) -> list[Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name not in {'corpus', 'boundaries'}])


def main() -> None:
    ap = argparse.ArgumentParser(description='Build plot-ready data and quick-look figures.')
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--table_dir', default='table')
    ap.add_argument('--subset', default='all_cross')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(args.table_dir)

    ratio_rows = []
    method_rows = []
    multi_rows = []

    for bdir in _boundary_dirs(run_dir):
        blabel = _boundary_label(bdir.name)
        diag = pd.read_csv(bdir / 'boundary_diagnosis.csv').iloc[0]
        ratio_rows.append({'Boundary': blabel, 'Cross Query Ratio': float(diag['frac_cross_query'])})
        subset_dir = bdir / args.subset
        app_df = pd.read_csv(subset_dir / 'enhanced_stats' / 'app_macro_summary.csv')
        app_df = app_df[app_df['method'].isin(METHOD_ORDER)].copy()
        for _, r in app_df.iterrows():
            method_rows.append({
                'Boundary': blabel,
                'Method': METHOD_LABEL[r['method']],
                'method_key': r['method'],
                'Success@20': float(r['success_at_20']),
                'Recall@50': float(r['recall_at_50']),
            })
        for mk in ['intra_only', 'multi_L2', 'multi_L3', 'multi_L5', 'global']:
            rr = app_df[app_df['method'] == mk]
            if rr.empty:
                continue
            multi_rows.append({'Boundary': blabel, 'Method': METHOD_LABEL[mk], 'method_key': mk, 'Success@20': float(rr.iloc[0]['success_at_20']), 'Recall@20': float(rr.iloc[0]['recall_at_20'])})

    ratio_df = pd.DataFrame(ratio_rows)
    method_df = pd.DataFrame(method_rows)
    multi_df = pd.DataFrame(multi_rows)
    write_csv(ratio_df, out_dir / 'Figure_1_cross_query_ratio.csv')
    write_csv(multi_df, out_dir / 'Figure_2_multicluster_curve.csv')
    write_csv(method_df, out_dir / 'Figure_3_method_comparison.csv')

    if not ratio_df.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.bar(ratio_df['Boundary'], ratio_df['Cross Query Ratio'])
        ax.set_ylabel('Cross Query Ratio')
        ax.set_title('Figure 1. Cross Query Ratio by Boundary')
        fig.tight_layout()
        fig.savefig(out_dir / 'Figure_1_cross_query_ratio.png', dpi=220)
        plt.close(fig)

    if not multi_df.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        order = ['intra_only', 'multi_L2', 'multi_L3', 'multi_L5', 'global']
        for boundary, g in multi_df.groupby('Boundary'):
            g = g.set_index('method_key').loc[[m for m in order if m in set(g['method_key'])]].reset_index()
            ax.plot(range(len(g)), g['Success@20'].to_numpy(), marker='o', label=boundary)
        ax.set_xticks(range(len(order)), [METHOD_LABEL[m] for m in order])
        ax.set_ylabel('Success@20')
        ax.set_title('Figure 2. MultiCluster Progressive Gain')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'Figure_2_multicluster_curve.png', dpi=220)
        plt.close(fig)

    if not method_df.empty:
        order = METHOD_ORDER
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for boundary, g in method_df.groupby('Boundary'):
            g = g.set_index('method_key').loc[[m for m in order if m in set(g['method_key'])]].reset_index()
            ax.plot(range(len(g)), g['Success@20'].to_numpy(), marker='o', label=boundary)
        ax.set_xticks(range(len(order)), [METHOD_LABEL[m] for m in order])
        ax.set_ylabel('Success@20')
        ax.set_title('Figure 3a. CrossRel Head Performance')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'Figure_3a_methods_success20.png', dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for boundary, g in method_df.groupby('Boundary'):
            g = g.set_index('method_key').loc[[m for m in order if m in set(g['method_key'])]].reset_index()
            ax.plot(range(len(g)), g['Recall@50'].to_numpy(), marker='o', label=boundary)
        ax.set_xticks(range(len(order)), [METHOD_LABEL[m] for m in order])
        ax.set_ylabel('Recall@50')
        ax.set_title('Figure 3b. CrossRel Recall Performance')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'Figure_3b_methods_recall50.png', dpi=220)
        plt.close(fig)


if __name__ == '__main__':
    main()
