from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from .hybrid_cross import HybridCrossConfig, build_hybrid_cross_rankings
from .utils import normalize_report_id


@dataclass(frozen=True)
class EcfaMethodConfig:
    per_query_csv: str
    max_rank_cap: int = 200
    prefer_inpool: bool = True
    global_topn: int = 120
    multi_topn: int = 80
    ecfa_topn: int = 120
    w_global: float = 0.55
    w_multi: float = 0.20
    w_ecfa: float = 0.15
    same_cluster_scale: float = 0.10
    out_cluster_bonus: float = 0.003
    ecfa_out_bonus: float = 0.002
    agreement_bonus: float = 0.002


def load_ecfa_rankings(
    per_query_csv: str | Path,
    *,
    max_rank_cap: int = 200,
    prefer_inpool: bool = True,
) -> Dict[Tuple[str, str], List[str]]:
    p = Path(per_query_csv)
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    cols = [c for c in df.columns if c.startswith('top_retrieved')]
    if not cols:
        raise ValueError(f'No top list columns found in {p}')
    maxk_fallback = int(df['maxk_dump'].max()) if 'maxk_dump' in df.columns and len(df) else int(max_rank_cap)

    def _col_key(c: str) -> tuple[int, int]:
        inpool = 1 if 'inpool' in c else 0
        if c.endswith('_maxk'):
            return (inpool, maxk_fallback)
        m = re.search(r'(\d+)$', c)
        k = int(m.group(1)) if m else 0
        return (inpool, k)

    best = sorted(cols, key=_col_key, reverse=True)[0]
    if prefer_inpool and 'inpool' not in best:
        inpool_cols = [c for c in cols if 'inpool' in c]
        if inpool_cols:
            best = sorted(inpool_cols, key=_col_key, reverse=True)[0]

    out: Dict[Tuple[str, str], List[str]] = {}
    for _, r in df.iterrows():
        app = str(r.get('app', '')).strip()
        qid = normalize_report_id(r.get('query_id'))
        raw = str(r.get(best, '') or '')
        parts = [pp for pp in raw.split('|') if pp and str(pp).strip()]
        seen: Set[str] = set()
        ranked: List[str] = []
        for x in parts:
            rid = normalize_report_id(x)
            if not rid or rid == qid or rid in seen:
                continue
            seen.add(rid)
            ranked.append(rid)
            if len(ranked) >= int(max_rank_cap):
                break
        out[(app, qid)] = ranked
    return out


def build_ecfa_rankings(
    *,
    q_keys: list[tuple[str, str]],
    clusters_df: pd.DataFrame,
    global_rankings: Dict[Tuple[str, str], List[str]],
    multi_rankings: Dict[Tuple[str, str], List[str]],
    cfg: EcfaMethodConfig,
) -> Dict[Tuple[str, str], List[str]]:
    ecfa_head = load_ecfa_rankings(
        cfg.per_query_csv,
        max_rank_cap=int(cfg.max_rank_cap),
        prefer_inpool=bool(cfg.prefer_inpool),
    )
    return build_hybrid_cross_rankings(
        q_keys=q_keys,
        clusters_df=clusters_df,
        global_rankings=global_rankings,
        multi_rankings=multi_rankings,
        ecfa_rankings=ecfa_head,
        cfg=HybridCrossConfig(
            global_topn=int(cfg.global_topn),
            multi_topn=int(cfg.multi_topn),
            ecfa_topn=int(cfg.ecfa_topn),
            w_global=float(cfg.w_global),
            w_multi=float(cfg.w_multi),
            w_ecfa=float(cfg.w_ecfa),
            same_cluster_scale=float(cfg.same_cluster_scale),
            out_cluster_bonus=float(cfg.out_cluster_bonus),
            ecfa_out_bonus=float(cfg.ecfa_out_bonus),
            agreement_bonus=float(cfg.agreement_bonus),
            max_rank_cap=int(cfg.max_rank_cap),
        ),
    )
