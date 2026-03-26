from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .clustering import SbertBoundaryConfig, cluster_sbert_hc
from .common import BoundaryArtifacts
from .methods import rank_text_baselines
from .representations import RepresentationConfig, RepresentationStore
from .utils import read_clusters_csv


@dataclass(frozen=True)
class SbertMethodConfig:
    k: int = 20
    model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    device: str = 'cpu'
    batch_size: int = 64
    multi_Ls: tuple[int, ...] = (2, 3, 5)
    max_rank_cap: int = 200
    clusters_csv: str = ''


def run_sbert_method(
    corpus: pd.DataFrame,
    q_keys: list[tuple[str, str]],
    cfg: SbertMethodConfig,
) -> BoundaryArtifacts:
    if str(cfg.clusters_csv or '').strip():
        clusters_df = read_clusters_csv(cfg.clusters_csv)
    else:
        clusters_df = cluster_sbert_hc(
            corpus,
            SbertBoundaryConfig(
                k=int(cfg.k),
                model=str(cfg.model),
                batch_size=int(cfg.batch_size),
                device=str(cfg.device),
            ),
        )
    rep_store = RepresentationStore(
        corpus,
        RepresentationConfig(
            kind='sbert',
            sbert_model=str(cfg.model),
            sbert_device=str(cfg.device),
            sbert_batch_size=int(cfg.batch_size),
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
    bname = f'sbert_hc_K{int(cfg.k)}' if not str(cfg.clusters_csv or '').strip() else pd.io.common.stringify_path(cfg.clusters_csv).split('/')[-1].replace('clusters_', '').replace('.csv', '')
    return BoundaryArtifacts(
        boundary_name=bname,
        clusters_df=clusters_df,
        rep_store=rep_store,
        rankings=rankings,
        method_family='sbert',
    )
