from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils import write_json

METHOD_MAP = {
    'cluster_only': 'intra_only',
    'multi_L3': 'multi_L3',
    'multi_L5': 'multi_L5',
    'global': 'global',
    'ecfa_hybrid_cross': 'ecfa',
}
COMPARE_METHODS = ['intra_only', 'multi_L3', 'multi_L5', 'global', 'ecfa']
METRICS = ['success_at_20', 'mrr_at_20', 'recall_at_20', 'recall_at_50', 'ndcg_at_20']


def compare(new_summary: str, ref_summary: str, out_json: str, tol: float = 1e-9) -> None:
    new_df = pd.read_csv(new_summary)
    ref_df = pd.read_csv(ref_summary).copy()
    ref_df['method'] = ref_df['method'].map(lambda x: METHOD_MAP.get(x, x))
    new_df = new_df[new_df['method'].isin(COMPARE_METHODS)]
    ref_df = ref_df[ref_df['method'].isin(COMPARE_METHODS)]
    merged = new_df.merge(ref_df, on='method', suffixes=('_new', '_ref'))
    rows = []
    all_ok = True
    for _, r in merged.iterrows():
        rec = {'method': r['method']}
        ok = True
        for m in METRICS:
            dn = float(r[f'{m}_new'])
            dr = float(r[f'{m}_ref'])
            diff = abs(dn - dr)
            rec[m] = {'new': dn, 'ref': dr, 'abs_diff': diff, 'ok': diff <= tol}
            ok = ok and diff <= tol
        rec['ok'] = ok
        rows.append(rec)
        all_ok = all_ok and ok
    write_json({'all_ok': all_ok, 'comparisons': rows, 'tolerance': tol}, out_json)
    print(json.dumps({'all_ok': all_ok, 'n_methods': len(rows)}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description='Validate clean project against legacy reference summary.')
    ap.add_argument('--new_summary', required=True)
    ap.add_argument('--ref_summary', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--tol', type=float, default=1e-9)
    args = ap.parse_args()
    compare(args.new_summary, args.ref_summary, args.out_json, args.tol)


if __name__ == '__main__':
    main()
