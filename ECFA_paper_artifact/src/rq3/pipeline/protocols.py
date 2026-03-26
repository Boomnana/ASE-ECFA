from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd

from rq3fresh.utils import normalize_report_id, safe_str


def build_allrel_map(df_reports: pd.DataFrame) -> tuple[dict[tuple[str, str], list[str]], pd.DataFrame]:


    df = df_reports.copy()
    df['app'] = df['app'].map(safe_str)
    df['id'] = df['id'].map(normalize_report_id)
    df['issue_key'] = df['issue_key'].map(safe_str)

    rel_map: Dict[Tuple[str, str], List[str]] = {}
    rows: list[dict] = []
    for (app, issue), g in df.groupby(['app', 'issue_key']):
        ids = [normalize_report_id(x) for x in g['id'].tolist()]
        if len(ids) <= 1:
            continue
        for qid in ids:
            rel = [x for x in ids if x != qid]
            rel_map[(str(app), str(qid))] = rel
            rows.append({
                'app': str(app),
                'query_id': str(qid),
                'issue_key': str(issue),
                'n_rel': int(len(rel)),
                'is_eval_query': 1,
            })
    perq = pd.DataFrame(rows).sort_values(['app', 'issue_key', 'query_id']).reset_index(drop=True)
    return rel_map, perq
