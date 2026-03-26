from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import re


REQUIRED_NODE_COLS = {"id", "name", "type", "source_row_index"}
REQUIRED_EDGE_COLS = {"id", "source_id", "target_id", "relation", "source_type", "target_type", "source_row_index"}


def split_report_ids(s: object) -> list[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    tokens = [t for t in re.split(r"[|,;\s]+", s) if t]
    out: list[str] = []
    seen = set()
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def load_nodes_xlsx(path: str | Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
    missing = REQUIRED_NODE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"nodes file missing columns: {sorted(missing)}; got {list(df.columns)}")
    df = df[list(REQUIRED_NODE_COLS)].copy()
    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].astype(str)
    df["type"] = df["type"].astype(str)
    df["source_row_index"] = df["source_row_index"].fillna("").astype(str)
    df["report_ids"] = df["source_row_index"].map(split_report_ids)
    return df


def load_edges_xlsx(path: str | Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
    missing = REQUIRED_EDGE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"edges file missing columns: {sorted(missing)}; got {list(df.columns)}")
    df = df[list(REQUIRED_EDGE_COLS)].copy()
    for c in ["id", "source_id", "target_id"]:
        df[c] = df[c].astype(str)
    for c in ["relation", "source_type", "target_type"]:
        df[c] = df[c].astype(str)
    df["source_row_index"] = df["source_row_index"].fillna("").astype(str)
    df["report_ids"] = df["source_row_index"].map(split_report_ids)
    return df


def load_defects_optional(path: Optional[str | Path], sheet_name=0) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, sheet_name=sheet_name, dtype=str)
    elif p.suffix.lower() in [".csv"]:
        df = pd.read_csv(p, dtype=str)
    elif p.suffix.lower() in [".tsv"]:
        df = pd.read_csv(p, sep="\t", dtype=str)
    elif p.suffix.lower() in [".jsonl"]:
        import json

        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"unsupported defect file suffix: {p.suffix}")
    if not {"id", "description"} <= set(df.columns):
        raise ValueError(f"defects file must have columns: id, description; got {list(df.columns)}")
    df = df[["id", "description"]].copy()
    df["id"] = df["id"].astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    return df
