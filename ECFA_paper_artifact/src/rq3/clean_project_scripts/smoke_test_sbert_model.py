from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.sbert_backend import encode_texts, resolve_cached_snapshot


def main() -> None:
    ap = argparse.ArgumentParser(description='Smoke-test offline SBERT model loading and encoding.')
    ap.add_argument('--model', required=True)
    args = ap.parse_args()
    resolved = resolve_cached_snapshot(args.model)
    payload = {'resolved_model': resolved}
    try:
        emb = encode_texts(['hello world', 'GUI crash when tapping login button'], model_name=resolved, device='cpu', batch_size=2)
        payload.update({'ok': True, 'shape': list(emb.shape), 'first_norm': float((emb[0] ** 2).sum() ** 0.5)})
    except Exception as e:
        payload.update({'ok': False, 'error_type': type(e).__name__, 'error': str(e)})
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
