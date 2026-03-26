from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .clustering import TfidfBoundaryConfig, cluster_tfidf_hc
from .common import BoundaryArtifacts
from .methods import rank_text_baselines
from .representations import RepresentationConfig, RepresentationStore
from .utils import read_clusters_csv


@dataclass(frozen=True)
class TfidfMethodConfig:
    k: int = 20
    ngram_min: int = 2
    ngram_max: int = 4
    multi_Ls: tuple[int, ...] = (2, 3, 5)
    max_rank_cap: int = 200
    clusters_csv: str = ''


def run_tfidf_method(
    corpus: pd.DataFrame,
    q_keys: list[tuple[str, str]],
    cfg: TfidfMethodConfig,
) -> BoundaryArtifacts:
    if str(cfg.clusters_csv or '').strip():
        clusters_df = read_clusters_csv(cfg.clusters_csv)
    else:
        clusters_df = cluster_tfidf_hc(
            corpus,
            TfidfBoundaryConfig(
                k=int(cfg.k),
                analyzer='char',
                ngram_min=int(cfg.ngram_min),
                ngram_max=int(cfg.ngram_max),
            ),
        )
    rep_store = RepresentationStore(
        corpus,
        RepresentationConfig(
            kind='tfidf_char',
            tfidf_analyzer='char',
            tfidf_ngram_min=int(cfg.ngram_min),
            tfidf_ngram_max=int(cfg.ngram_max),
        ),
    )
    rep_store.attach_clusters(clusters_df)
    rankings = rank_text_baselines(
        df_reports=corpus,
        clusters_df=clusters_df,
        rep_store=rep_store,
        q_keys=q_keys,
        multi_Ls=list(cfg.multi_Ls),
        max_rank_cap=int(cfg.max_rank_cap),
    )
    bname = f'tfidf_hc_K{int(cfg.k)}' if not str(cfg.clusters_csv or '').strip() else pd.io.common.stringify_path(cfg.clusters_csv).split('/')[-1].replace('clusters_', '').replace('.csv', '')
    return BoundaryArtifacts(
        boundary_name=bname,
        clusters_df=clusters_df,
        rep_store=rep_store,
        rankings=rankings,
        method_family='tfidf',
    )
