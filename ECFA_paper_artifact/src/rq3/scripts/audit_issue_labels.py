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

from rq3fresh.data import load_reports
from rq3fresh.utils import ensure_dir, safe_str, write_csv, write_json


def normalize_text_key(s: str) -> str:
    s = safe_str(s).strip().lower()
    s = ''.join(ch for ch in s if ch.isalnum() or '\u4e00' <= ch <= '\u9fff')
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description='Audit issue_key width / noise for RQ3 labels')
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df, _ = load_reports(args.xlsx)
    df['text_key'] = df['text'].astype(str).map(normalize_text_key)
    rows = []
    for (app, issue_key), g in df.groupby(['app', 'issue_key']):
        n = int(g.shape[0])
        n_unique = int(g['text_key'].nunique())
        rows.append({
            'app': str(app),
            'issue_key': str(issue_key),
            'issue_size': n,
            'n_unique_text_key': n_unique,
            'unique_text_ratio': (n_unique / n) if n > 0 else 0.0,
            'median_text_len': float(g['text'].astype(str).str.len().median()),
        })
    audit = pd.DataFrame(rows).sort_values(['issue_size', 'unique_text_ratio'], ascending=[False, False]).reset_index(drop=True)
    write_csv(audit, out_dir / 'issue_audit.csv')
    summary = {
        'n_reports': int(df.shape[0]),
        'n_issue_groups': int(audit.shape[0]),
        'issue_size': {
            'median': float(audit['issue_size'].median()) if len(audit) else 0.0,
            'p90': float(audit['issue_size'].quantile(0.9)) if len(audit) else 0.0,
            'max': int(audit['issue_size'].max()) if len(audit) else 0,
        },
        'top_10_large_groups': audit.head(10).to_dict('records'),
    }
    write_json(summary, out_dir / 'issue_audit_summary.json')
    print(f'[OK] wrote {out_dir}')


if __name__ == '__main__':
    main()
