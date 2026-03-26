from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class TfidfBoundaryConfig:
    k: int = 20
    analyzer: str = 'char'
    ngram_min: int = 2
    ngram_max: int = 4


@dataclass(frozen=True)
class SbertBoundaryConfig:
    k: int = 20
    model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    batch_size: int = 64
    device: str = 'cpu'


def encode_sbert(texts: list[str], cfg: SbertBoundaryConfig) -> np.ndarray:
    from .sbert_backend import encode_texts
    return encode_texts(
        texts,
        model_name=str(cfg.model),
        device=str(cfg.device),
        batch_size=int(cfg.batch_size),
        max_length=128,
    )


def _cluster_from_distance(D: np.ndarray, k: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(n_clusters=min(k, D.shape[0]), metric='precomputed', linkage='average')
    except TypeError:
        model = AgglomerativeClustering(n_clusters=min(k, D.shape[0]), affinity='precomputed', linkage='average')
    return model.fit_predict(D).astype(int)


def cluster_tfidf_hc(df_reports: pd.DataFrame, cfg: TfidfBoundaryConfig) -> pd.DataFrame:
    rows = []
    for app, g in df_reports.groupby('app'):
        g = g.reset_index(drop=True)
        vec = TfidfVectorizer(analyzer=cfg.analyzer, ngram_range=(cfg.ngram_min, cfg.ngram_max), min_df=1)
        X = vec.fit_transform(g['text'].astype(str).tolist())
        D = 1.0 - cosine_similarity(X, X)
        np.fill_diagonal(D, 0.0)
        labels = _cluster_from_distance(D, cfg.k)
        for rid, issue_key, lab in zip(g['id'].tolist(), g['issue_key'].tolist(), labels.tolist()):
            rows.append({'app': str(app), 'id': str(rid), 'cluster_id': f'{app}__C{int(lab)}', 'issue_key': str(issue_key)})
    return pd.DataFrame(rows).sort_values(['app', 'cluster_id', 'id']).reset_index(drop=True)


def cluster_sbert_hc(df_reports: pd.DataFrame, cfg: SbertBoundaryConfig) -> pd.DataFrame:
    rows = []
    for app, g in df_reports.groupby('app'):
        g = g.reset_index(drop=True)
        emb = encode_sbert(g['text'].astype(str).tolist(), cfg)
        D = 1.0 - (emb @ emb.T)
        np.fill_diagonal(D, 0.0)
        labels = _cluster_from_distance(D, cfg.k)
        for rid, issue_key, lab in zip(g['id'].tolist(), g['issue_key'].tolist(), labels.tolist()):
            rows.append({'app': str(app), 'id': str(rid), 'cluster_id': f'{app}__C{int(lab)}', 'issue_key': str(issue_key)})
    return pd.DataFrame(rows).sort_values(['app', 'cluster_id', 'id']).reset_index(drop=True)
