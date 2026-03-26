from __future__ import annotations

import sys
from pathlib import Path


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from rq3fresh.runner import PipelineConfig, run_pipeline


def parse_int_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(s).split(',') if x.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description='Fresh RQ3-2A pipeline (standard metrics only).')
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--out_root', default='outputs/rq3_2a_fresh')
    ap.add_argument('--run_tag', default=None)
    ap.add_argument('--input_root', default='input/ROOT_glm4.7')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--drop_singleton_issues', type=int, default=1)
    ap.add_argument('--run_tfidf', type=int, default=1)
    ap.add_argument('--run_sbert', type=int, default=1)
    ap.add_argument('--sbert_model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    ap.add_argument('--sbert_device', default='cpu')
    ap.add_argument('--sbert_batch', type=int, default=64)
    ap.add_argument('--success_ks', default='10,20')
    ap.add_argument('--recall_ks', default='20,50')
    ap.add_argument('--mrr_k', type=int, default=20)
    ap.add_argument('--ndcg_k', type=int, default=20)
    ap.add_argument('--max_rank_cap', type=int, default=200)
    ap.add_argument('--multi_Ls', default='3,5')
    ap.add_argument('--retr_repr', default='auto', choices=['auto', 'tfidf_char', 'sbert'])
    ap.add_argument('--run_ecfa', type=int, default=1)
    ap.add_argument('--ecfa_variant', default='head', choices=['head', 'legacy'])
    ap.add_argument('--ecfa_preset', default='aggressive', choices=['balanced', 'aggressive', 'ultra'])
    ap.add_argument('--ecfa_per_query_csv', default=None)
    ap.add_argument('--limit_queries', type=int, default=0)
    args = ap.parse_args()

    cfg = PipelineConfig(
        xlsx=args.xlsx,
        out_root=args.out_root,
        run_tag=args.run_tag,
        input_root=args.input_root,
        k=args.k,
        drop_singleton_issues=bool(int(args.drop_singleton_issues)),
        run_tfidf=bool(int(args.run_tfidf)),
        run_sbert=bool(int(args.run_sbert)),
        sbert_model=args.sbert_model,
        sbert_device=args.sbert_device,
        sbert_batch=args.sbert_batch,
        topk_success=parse_int_tuple(args.success_ks),
        topk_recall=parse_int_tuple(args.recall_ks),
        mrr_k=args.mrr_k,
        ndcg_k=args.ndcg_k,
        max_rank_cap=args.max_rank_cap,
        multi_Ls=parse_int_tuple(args.multi_Ls),
        retr_repr=args.retr_repr,
        run_ecfa=bool(int(args.run_ecfa)),
        ecfa_variant=args.ecfa_variant,
        ecfa_preset=args.ecfa_preset,
        ecfa_per_query_csv=args.ecfa_per_query_csv,
        limit_queries=args.limit_queries,
    )
    run_pipeline(cfg)


if __name__ == '__main__':
    main()
