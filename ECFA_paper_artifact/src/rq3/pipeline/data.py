from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from .utils import APP_SHEETS_BLACKLIST, normalize_report_id, safe_str


@dataclass(frozen=True)
class LoadAudit:
    xlsx: str
    n_rows_raw: int
    n_rows_valid: int
    n_apps: int
    n_issues: int
    dropped_empty_id: int
    dropped_empty_text: int


@dataclass(frozen=True)
class CorpusAudit:
    n_rows_in: int
    n_rows_out: int
    n_apps: int
    n_issues: int
    n_singleton_issues_dropped: int


def pick_text_col(df: pd.DataFrame) -> str:
    if 'primary_clue' in df.columns:
        return 'primary_clue'
    if 'description' in df.columns:
        return 'description'
    for c in ['text', 'content', 'desc']:
        if c in df.columns:
            return c
    raise ValueError('No text column found (expected primary_clue/description/text).')


def load_reports(xlsx_path: str | Path, *, sheet_blacklist: set[str] = APP_SHEETS_BLACKLIST) -> tuple[pd.DataFrame, LoadAudit]:
    p = Path(xlsx_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    rows: list[pd.DataFrame] = []
    n_rows_raw = 0
    drop_empty_id = 0
    drop_empty_text = 0

    def _coerce_one(df: pd.DataFrame, default_app: str) -> None:
        nonlocal n_rows_raw, drop_empty_id, drop_empty_text
        if df is None or len(df) == 0:
            return
        n_rows_raw += int(df.shape[0])
        if 'id' not in df.columns or 'issue_key' not in df.columns:
            return
        text_col = pick_text_col(df)
        df = df.copy()
        if 'app' not in df.columns:
            df['app'] = str(default_app)
        else:
            df['app'] = df['app'].fillna('').astype(str).str.strip()
            df.loc[df['app'].astype(str).str.len() == 0, 'app'] = str(default_app)
        df['id'] = df['id'].map(normalize_report_id)
        df['issue_key'] = df['issue_key'].map(safe_str)
        df['text'] = df[text_col].map(safe_str)
        before = len(df)
        df = df[df['id'].astype(str).str.len() > 0]
        drop_empty_id += before - len(df)
        before = len(df)
        df = df[df['text'].astype(str).str.len() > 0]
        drop_empty_text += before - len(df)
        if len(df) == 0:
            return
        rows.append(df[['app', 'id', 'issue_key', 'text']])

    if p.suffix.lower() == '.csv':
        _coerce_one(pd.read_csv(str(p)), default_app=p.stem)
    else:
        xls = pd.ExcelFile(str(p))
        for sh in xls.sheet_names:
            if sh in sheet_blacklist:
                continue
            _coerce_one(pd.read_excel(str(p), sheet_name=sh), default_app=str(sh))

    if not rows:
        raise ValueError('No valid rows/sheets found in input.')

    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=['app', 'id']).reset_index(drop=True)
    audit = LoadAudit(
        xlsx=str(p),
        n_rows_raw=int(n_rows_raw),
        n_rows_valid=int(out.shape[0]),
        n_apps=int(out['app'].nunique()),
        n_issues=int(out.groupby(['app', 'issue_key']).ngroups),
        dropped_empty_id=int(drop_empty_id),
        dropped_empty_text=int(drop_empty_text),
    )
    return out, audit


def build_corpus(df_reports: pd.DataFrame, *, drop_singleton_issues: bool = True) -> tuple[pd.DataFrame, CorpusAudit]:
    if df_reports is None or len(df_reports) == 0:
        raise ValueError('df_reports is empty')
    df = df_reports.copy()
    df['issue_size'] = df.groupby(['app', 'issue_key'])['id'].transform('count')
    n_in = int(df.shape[0])
    singleton_issues = df.groupby(['app', 'issue_key'])['id'].count().eq(1)
    n_singleton_issues = int(singleton_issues.sum())
    if drop_singleton_issues:
        df = df[df['issue_size'] > 1].copy()
    audit = CorpusAudit(
        n_rows_in=n_in,
        n_rows_out=int(df.shape[0]),
        n_apps=int(df['app'].nunique()) if len(df) else 0,
        n_issues=int(df.groupby(['app', 'issue_key']).ngroups) if len(df) else 0,
        n_singleton_issues_dropped=n_singleton_issues if drop_singleton_issues else 0,
    )
    df = df.sort_values(['app', 'issue_key', 'id']).reset_index(drop=True)
    return df, audit
