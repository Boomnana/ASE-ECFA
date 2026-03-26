from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd

from .methods import _cluster_lookup
from .utils import dedup_keep_order


@dataclass(frozen=True)
class HybridCrossConfig:
    global_topn: int = 120
    multi_topn: int = 80
    ecfa_topn: int = 120

    w_global: float = 0.55
    w_multi: float = 0.20
    w_ecfa: float = 0.15

    bias_global: int = 4
    bias_multi: int = 4
    bias_ecfa: int = 2

    same_cluster_scale: float = 0.10
    out_cluster_bonus: float = 0.003
    ecfa_out_bonus: float = 0.002
    agreement_bonus: float = 0.002

    max_rank_cap: int = 200


def _accumulate_rank_score(
    score: dict[str, float],
    sources: dict[str, int],
    ranked_ids: Iterable[str],
    *,
    src_bit: int,
    weight: float,
    bias: int,
) -> None:

    w = float(weight)
    b = int(bias)
    sc_get = score.get
    src_get = sources.get
    for i, rid in enumerate(ranked_ids, start=1):
        if not rid:
            continue
        score[rid] = sc_get(rid, 0.0) + w / float(i + b)
        sources[rid] = src_get(rid, 0) | src_bit


def build_hybrid_cross_rankings(
    *,
    q_keys: list[tuple[str, str]],
    clusters_df: pd.DataFrame,
    global_rankings: dict[tuple[str, str], list[str]],
    multi_rankings: dict[tuple[str, str], list[str]],
    ecfa_rankings: dict[tuple[str, str], list[str]],
    cfg: HybridCrossConfig,
) -> dict[tuple[str, str], list[str]]:
    id_to_cluster, _, _ = _cluster_lookup(clusters_df)
    out: dict[tuple[str, str], list[str]] = {}
    SRC_GLOBAL = 1
    SRC_MULTI = 2
    SRC_ECFA = 4

    gtop = int(cfg.global_topn)
    mtop = int(cfg.multi_topn)
    etop = int(cfg.ecfa_topn)
    same_scale = float(cfg.same_cluster_scale)
    out_bonus = float(cfg.out_cluster_bonus)
    ecfa_out_bonus = float(cfg.ecfa_out_bonus)
    agree_bonus = float(cfg.agreement_bonus)
    max_cap = int(cfg.max_rank_cap)

    for app, qid in q_keys:
        q_cluster = id_to_cluster.get((app, qid))
        score: dict[str, float] = {}
        srcs: dict[str, int] = {}

        _accumulate_rank_score(score, srcs, global_rankings.get((app, qid), [])[:gtop], src_bit=SRC_GLOBAL, weight=float(cfg.w_global), bias=int(cfg.bias_global))
        _accumulate_rank_score(score, srcs, multi_rankings.get((app, qid), [])[:mtop], src_bit=SRC_MULTI, weight=float(cfg.w_multi), bias=int(cfg.bias_multi))
        _accumulate_rank_score(score, srcs, ecfa_rankings.get((app, qid), [])[:etop], src_bit=SRC_ECFA, weight=float(cfg.w_ecfa), bias=int(cfg.bias_ecfa))

        ranked_rows: list[tuple[str, float, int, int]] = []
        for rid, base in score.items():
            c_cluster = id_to_cluster.get((app, rid))
            is_out_cluster = int(c_cluster is not None and c_cluster != q_cluster)
            adjusted = float(base)
            bits = srcs.get(rid, 0)
            if is_out_cluster:
                adjusted += out_bonus
                if bits & SRC_ECFA:
                    adjusted += ecfa_out_bonus
                if bits in (SRC_GLOBAL | SRC_MULTI, SRC_GLOBAL | SRC_ECFA, SRC_MULTI | SRC_ECFA, SRC_GLOBAL | SRC_MULTI | SRC_ECFA):
                    adjusted += agree_bonus
            else:
                adjusted *= same_scale
            ranked_rows.append((rid, adjusted, is_out_cluster, bits.bit_count()))

        ranked_rows.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
        out[(app, qid)] = dedup_keep_order([rid for rid, _, _, _ in ranked_rows], drop=qid, limit=max_cap)
    return out
