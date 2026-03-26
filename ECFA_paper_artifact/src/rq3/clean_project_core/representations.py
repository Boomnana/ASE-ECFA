from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize as sk_normalize


@dataclass(frozen=True)
class RepresentationConfig:
    kind: str = "tfidf_char"
    tfidf_analyzer: str = "char"
    tfidf_ngram_min: int = 2
    tfidf_ngram_max: int = 4
    sbert_model: Optional[str] = None
    sbert_device: str = "cpu"
    sbert_batch_size: int = 64


class AppRepresentationIndex:
    def __init__(self, app: str, ids: list[str], texts: list[str], cfg: RepresentationConfig):
        self.app = str(app)
        self.ids = ids
        self.texts = texts
        self.id_to_row = {rid: i for i, rid in enumerate(ids)}
        self.cfg = cfg
        self.kind = cfg.kind
        self.vectorizer = None
        if cfg.kind == 'tfidf_char':
            vec = TfidfVectorizer(analyzer=cfg.tfidf_analyzer, ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max), min_df=1)
            X = vec.fit_transform(texts)
            X = sk_normalize(X, norm='l2', axis=1, copy=False)
            self.X = X
            self.vectorizer = vec
        elif cfg.kind == 'sbert':
            if not cfg.sbert_model:
                raise ValueError('sbert_model is required when kind=sbert')
            from .sbert_backend import encode_texts
            self.X = encode_texts(
                texts,
                model_name=str(cfg.sbert_model),
                device=str(cfg.sbert_device),
                batch_size=int(cfg.sbert_batch_size),
                max_length=128,
            )
        else:
            raise ValueError(f'Unknown representation kind: {cfg.kind}')
        self.cluster_rows: Dict[str, np.ndarray] = {}
        self.cluster_centroids: Dict[str, np.ndarray] = {}

    def has(self, rid: str) -> bool:
        return rid in self.id_to_row

    def rows(self, ids: Iterable[str]) -> np.ndarray:
        return np.asarray([self.id_to_row[rid] for rid in ids if rid in self.id_to_row], dtype=np.int32)

    def query_scores(self, query_ids: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
        qids = [qid for qid in query_ids if qid in self.id_to_row]
        q_rows = np.asarray([self.id_to_row[qid] for qid in qids], dtype=np.int32)
        if len(q_rows) == 0:
            return np.zeros((0, len(self.ids)), dtype=np.float32), [], q_rows
        if sparse.issparse(self.X):
            Q = self.X[q_rows]
            S = linear_kernel(Q, self.X)
            return np.asarray(S, dtype=np.float32), qids, q_rows
        Q = self.X[q_rows]
        S = Q @ self.X.T
        return np.asarray(S, dtype=np.float32), qids, q_rows

    def attach_clusters(self, clusters_df: pd.DataFrame) -> None:
        app_df = clusters_df[clusters_df['app'] == self.app].copy()
        app_df['id'] = app_df['id'].astype(str)
        for cid, g in app_df.groupby('cluster_id'):
            rows = self.rows(g['id'].astype(str).tolist())
            if len(rows) == 0:
                continue
            self.cluster_rows[str(cid)] = rows
            if sparse.issparse(self.X):
                centroid = np.asarray(self.X[rows].mean(axis=0)).reshape(-1).astype(np.float32)
            else:
                centroid = np.asarray(self.X[rows].mean(axis=0), dtype=np.float32)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid /= norm
            self.cluster_centroids[str(cid)] = centroid

    def centroid_scores(self, qid: str, cluster_ids: list[str]) -> np.ndarray:
        if qid not in self.id_to_row:
            return np.full(len(cluster_ids), -1.0, dtype=np.float32)
        qi = self.id_to_row[qid]
        if sparse.issparse(self.X):
            qv = np.asarray(self.X[qi].todense()).reshape(-1).astype(np.float32)
            n = float(np.linalg.norm(qv))
            if n > 0:
                qv /= n
        else:
            qv = np.asarray(self.X[qi], dtype=np.float32)
        sims = []
        for cid in cluster_ids:
            c = self.cluster_centroids.get(cid)
            sims.append(float(qv @ c) if c is not None else -1.0)
        return np.asarray(sims, dtype=np.float32)


class RepresentationStore:
    def __init__(self, df_reports: pd.DataFrame, cfg: RepresentationConfig):
        self.cfg = cfg
        self.by_app: Dict[str, AppRepresentationIndex] = {}
        for app, g in df_reports.groupby('app'):
            gg = g.sort_values('id').reset_index(drop=True)
            self.by_app[str(app)] = AppRepresentationIndex(str(app), gg['id'].astype(str).tolist(), gg['text'].astype(str).tolist(), cfg)

    def attach_clusters(self, clusters_df: pd.DataFrame) -> None:
        for app, idx in self.by_app.items():
            idx.attach_clusters(clusters_df)
