from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

APP_SHEETS_BLACKLIST = {"QA_Summary", "Label_Guide"}
_SPLIT_RE = re.compile(r"[\|,;\s]+")


def safe_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def normalize_report_id(x: object) -> str:
    s = safe_str(x)
    if not s:
        return ""
    if re.fullmatch(r"\d+\.0+", s):
        return s.split('.', 1)[0]
    if re.fullmatch(r"\d+(\.\d+)?[eE]\+?\d+", s):
        try:
            return str(int(float(s)))
        except Exception:
            return s
    if re.fullmatch(r"\d+\.\d+", s):
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
    return s


def split_report_ids(s: object) -> list[str]:
    raw = safe_str(s)
    if not raw:
        return []
    parts = [p for p in _SPLIT_RE.split(raw) if p]
    return [rid for rid in (normalize_report_id(p) for p in parts) if rid]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)


    with p.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(list(df.columns))
        for row in df.itertuples(index=False, name=None):
            writer.writerow(list(row))
    return p


def write_json(obj: object, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


def read_clusters_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    req = {'app', 'id', 'cluster_id', 'issue_key'}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f'{p} missing columns: {sorted(miss)}')
    df = df.copy()
    df['app'] = df['app'].map(safe_str)
    df['id'] = df['id'].map(normalize_report_id)
    df['cluster_id'] = df['cluster_id'].map(safe_str)
    df['issue_key'] = df['issue_key'].map(safe_str)
    return df


def dedup_keep_order(ids: Iterable[str], *, drop: str | None = None, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    drop_norm = normalize_report_id(drop) if drop else None
    for x in ids:
        rid = normalize_report_id(x)
        if not rid:
            continue
        if drop_norm and rid == drop_norm:
            continue
        if rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
        if limit is not None and len(out) >= int(limit):
            break
    return out
