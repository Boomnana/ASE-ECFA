from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from rq3fresh.utils import dedup_keep_order, normalize_report_id


@dataclass(frozen=True)
class HybridAllRelConfig:
    global_topn: int = 120
    cluster_topn: int = 50
    ecfa_topn: int = 80
    w_global: float = 0.60
    w_cluster: float = 0.10
    w_ecfa: float = 0.30
    bias_global: int = 4
    bias_cluster: int = 4
    bias_ecfa: int = 2
    max_rank_cap: int = 200


def _accumulate_rank_score(score: dict[str, float], ranked_ids: Iterable[str], *, weight: float, bias: int) -> None:
    for i, rid in enumerate(ranked_ids, start=1):
        rid_n = normalize_report_id(rid)
        if not rid_n:
            continue
        score[rid_n] = score.get(rid_n, 0.0) + float(weight) / float(i + bias)


def build_hybrid_allrel_rankings(
    *,
    q_keys: list[tuple[str, str]],
    global_rankings: dict[tuple[str, str], list[str]],
    cluster_rankings: dict[tuple[str, str], list[str]],
    ecfa_rankings: dict[tuple[str, str], list[str]],
    cfg: HybridAllRelConfig,
) -> dict[tuple[str, str], list[str]]:


    out: dict[tuple[str, str], list[str]] = {}
    for app, qid in q_keys:
        score: dict[str, float] = {}
        _accumulate_rank_score(
            score,
            global_rankings.get((app, qid), [])[: int(cfg.global_topn)],
            weight=float(cfg.w_global),
            bias=int(cfg.bias_global),
        )
        _accumulate_rank_score(
            score,
            cluster_rankings.get((app, qid), [])[: int(cfg.cluster_topn)],
            weight=float(cfg.w_cluster),
            bias=int(cfg.bias_cluster),
        )
        _accumulate_rank_score(
            score,
            ecfa_rankings.get((app, qid), [])[: int(cfg.ecfa_topn)],
            weight=float(cfg.w_ecfa),
            bias=int(cfg.bias_ecfa),
        )
        ranked = [rid for rid, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))]
        out[(app, qid)] = dedup_keep_order(ranked, drop=qid, limit=int(cfg.max_rank_cap))
    return out
