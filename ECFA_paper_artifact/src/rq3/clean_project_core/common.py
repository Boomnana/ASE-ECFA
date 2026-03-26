from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from .representations import RepresentationStore


@dataclass
class BoundaryArtifacts:
    boundary_name: str
    clusters_df: pd.DataFrame
    rep_store: RepresentationStore
    rankings: Dict[str, Dict[Tuple[str, str], List[str]]]
    method_family: str
