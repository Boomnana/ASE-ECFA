from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

_MODEL_CACHE: Dict[Tuple[str, str], tuple[AutoTokenizer, AutoModel]] = {}


def resolve_cached_snapshot(model_name: str) -> str:
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


def get_model(model_name: str, device: str = 'cpu') -> tuple[AutoTokenizer, AutoModel]:
    resolved = resolve_cached_snapshot(model_name)
    key = (resolved, str(device))
    pair = _MODEL_CACHE.get(key)
    if pair is None:
        tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True, use_fast=False)
        model = AutoModel.from_pretrained(resolved, local_files_only=True)
        model.eval()
        model.to(device)
        pair = (tokenizer, model)
        _MODEL_CACHE[key] = pair
    return pair


@torch.inference_mode()
def encode_texts(
    texts: list[str],
    *,
    model_name: str,
    device: str = 'cpu',
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    tokenizer, model = get_model(model_name, device=device)
    outs: list[np.ndarray] = []
    for i in range(0, len(texts), int(batch_size)):
        batch = texts[i:i + int(batch_size)]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last_hidden = out.last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        outs.append(pooled.cpu().numpy().astype(np.float32))
    if not outs:
        return np.zeros((0, 384), dtype=np.float32)
    return np.vstack(outs)
