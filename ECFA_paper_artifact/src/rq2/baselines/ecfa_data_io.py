from __future__ import annotations

"""
ECFA data acquisition / I/O interfaces (method-only).

This module intentionally contains NO algorithm logic. It provides:
- Robust parsing helpers for report ids stored in spreadsheets
- Loaders for evidence-graph artifacts (nodes/edges/defects)
- A provider interface that experiments can implement to plug in custom datasets

You can keep this file stable while iterating on ECFA algorithms in ecfa_method.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Protocol, Tuple, Any

import re
import pandas as pd


def normalize_report_id(x: str) -> str:


    x = (x or "").strip()
    if not x:
        return x


    if re.fullmatch(r"\d+\.0+", x):
        return x.split(".", 1)[0]


    if re.fullmatch(r"\d+(\.\d+)?[eE][\+\-]?\d+", x):
        try:

            return str(int(float(x)))
        except Exception:
            return x

    return x


def parse_report_ids_cell(cell: Any) -> List[str]:


    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple, set)):
        out = [normalize_report_id(str(x)) for x in cell if str(x).strip()]
        return [x for x in out if x]

    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []


    s2 = s.strip("[](){}")

    parts = re.split(r"[,\;\|\s]+", s2)
    out = [normalize_report_id(p) for p in parts if p and p.strip()]
    return [x for x in out if x]


REQUIRED_NODE_COLS = {"id", "name", "type", "source_row_index"}
REQUIRED_EDGE_COLS = {"id", "source_id", "target_id", "relation", "source_type", "target_type", "source_row_index"}


def _ensure_required_cols(df: pd.DataFrame, required: Set[str], kind: str) -> None:
    missing = sorted(list(set(required) - set(df.columns)))
    if missing:
        raise ValueError(f"{kind} missing required columns: {missing}")


def load_nodes_xlsx(path: str | Path, *, sheet_name: Optional[str] = None) -> pd.DataFrame:


    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0)
    _ensure_required_cols(df, REQUIRED_NODE_COLS, "nodes")

    if "report_ids" not in df.columns:


        if "source_row_index" in df.columns:
             df["report_ids"] = df["source_row_index"].apply(parse_report_ids_cell)
        else:
             df["report_ids"] = [[] for _ in range(len(df))]
    else:
        df["report_ids"] = df["report_ids"].apply(parse_report_ids_cell)

    df["id"] = df["id"].astype(str)
    df["type"] = df["type"].astype(str)
    df["name"] = df["name"].astype(str)
    return df


def load_edges_xlsx(path: str | Path, *, sheet_name: Optional[str] = None) -> pd.DataFrame:


    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0)
    _ensure_required_cols(df, REQUIRED_EDGE_COLS, "edges")

    if "report_ids" not in df.columns:

        if "source_row_index" in df.columns:
             df["report_ids"] = df["source_row_index"].apply(parse_report_ids_cell)
        else:
             df["report_ids"] = [[] for _ in range(len(df))]
    else:
        df["report_ids"] = df["report_ids"].apply(parse_report_ids_cell)


    df["id"] = df["id"].astype(str)
    df["source_id"] = df["source_id"].astype(str)
    df["target_id"] = df["target_id"].astype(str)
    df["relation"] = df["relation"].astype(str)
    df["source_type"] = df["source_type"].astype(str)
    df["target_type"] = df["target_type"].astype(str)
    return df


def load_defects_optional(path: str | Path | None, *, sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:


    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None

    df = pd.read_excel(p, sheet_name=sheet_name) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    if "description" in df.columns:
        df["description"] = df["description"].astype(str)
    return df


class ECFADataProvider(Protocol):


    def load_nodes(self) -> pd.DataFrame: ...
    def load_edges(self) -> pd.DataFrame: ...
    def load_defects(self) -> Optional[pd.DataFrame]: ...


@dataclass(frozen=True)
class XlsxEvidenceGraphProvider:


    nodes_path: Path
    edges_path: Path
    defects_path: Optional[Path] = None
    nodes_sheet: Optional[str] = None
    edges_sheet: Optional[str] = None
    defects_sheet: Optional[str] = None

    def load_nodes(self) -> pd.DataFrame:
        return load_nodes_xlsx(self.nodes_path, sheet_name=self.nodes_sheet)

    def load_edges(self) -> pd.DataFrame:
        return load_edges_xlsx(self.edges_path, sheet_name=self.edges_sheet)

    def load_defects(self) -> Optional[pd.DataFrame]:
        return load_defects_optional(self.defects_path, sheet_name=self.defects_sheet)


def load_evidence_graph(provider: ECFADataProvider) -> "EvidenceGraph":


    from ecfa_method import EvidenceGraph
    nodes = provider.load_nodes()
    edges = provider.load_edges()
    defects = provider.load_defects()
    return EvidenceGraph(nodes=nodes, edges=edges, defects=defects)
