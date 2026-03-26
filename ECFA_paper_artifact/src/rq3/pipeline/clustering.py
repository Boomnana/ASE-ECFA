from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
_ST_MODEL_CACHE: Dict[Tuple[str, str], object] = {}


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


def _resolve_cached_snapshot(model_name: str) -> str:
    p = Path(str(model_name))
    if p.exists():
        return str(p)
    mid = str(model_name)
    if '/' not in mid:
        return mid
    org, name = mid.split('/', 1)
    hf_home = Path(os.environ.get('HF_HOME') or (Path.home() / '.cache' / 'huggingface'))
    snap_root = hf_home / 'hub' / f'models--{org}--{name}' / 'snapshots'
    if not snap_root.exists():
        return mid
    snaps = [d for d in snap_root.iterdir() if d.is_dir()]
    if not snaps:
        return mid
    snaps.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return str(snaps[0])


def _get_model(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError('sentence-transformers is not installed.') from e
    resolved = _resolve_cached_snapshot(str(model_name))
    key = (resolved, str(device))
    model = _ST_MODEL_CACHE.get(key)
    if model is None:
        model = SentenceTransformer(resolved, device=device)
        _ST_MODEL_CACHE[key] = model
    return model


def encode_sbert(texts: list[str], cfg: SbertBoundaryConfig) -> np.ndarray:
    model = _get_model(cfg.model, cfg.device)
    emb = model.encode(texts, batch_size=int(cfg.batch_size), show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


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
