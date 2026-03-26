from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .aggregate_stats import BootstrapConfig
from .data import build_corpus, load_reports
from .exporters import export_subset_enhanced_stats
from .method_ecfa import EcfaMethodConfig, build_ecfa_rankings
from .method_llmcluster import LlmClusterMethodConfig, run_llmcluster_method
from .method_sbert import SbertMethodConfig, run_sbert_method
from .method_tfidf import TfidfMethodConfig, run_tfidf_method
from .metrics import build_crossrel_map, evaluate_rankings_from_relmap
from .utils import ensure_dir, write_csv, write_json


@dataclass(frozen=True)
class MainConfig:
    xlsx: str
    ecfa_per_query_csv: str
    out_dir: str
    k: int = 20
    max_rank_cap: int = 200
    multi_Ls: tuple[int, ...] = (2, 3, 5)
    run_tfidf: bool = True
    run_sbert: bool = True
    run_llmcluster: bool = False
    tfidf_clusters_csv: str = ''
    sbert_clusters_csv: str = ''
    llm_clusters_csv: str = ''
    llm_repr_kind: str = 'tfidf_char'
    llm_model: str = 'glm-4.7'
    llm_api_url: str = ''
    llm_api_key: str = ''
    llm_api_key_env: str = 'ZHIPUAI_API_KEY'
    llm_timeout: int = 180
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_retries: int = 5
    llm_thinking_enabled: bool = False
    llm_verify_ssl: bool = True
    llm_use_env_proxy: bool = False
    llm_verbose: bool = True
    llm_text_column: str = 'auto'
    llm_max_per_iteration: int = 50
    llm_seed: int = 42
    llm_summary_sample_size: int = 8
    llm_context_soft_limit: int = 90000
    llm_mock: bool = False
    sbert_model: str = ''
    sbert_device: str = 'cpu'
    sbert_batch: int = 64
    success_ks: tuple[int, ...] = (10, 20)
    recall_ks: tuple[int, ...] = (20, 50)
    mrr_k: int = 20
    ndcg_k: int = 20
    bootstrap_n: int = 500
    bootstrap_seed: int = 20260309


def _parse_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(s).split(',') if x.strip())


def _rename_method_keys(rankings: Dict[str, Dict[Tuple[str, str], List[str]]]) -> Dict[str, Dict[Tuple[str, str], List[str]]]:
    out = dict(rankings)
    if 'cluster_only' in out:
        out['intra_only'] = out.pop('cluster_only')
    return out


def _rankings_to_frame(rankings: Dict[str, Dict[Tuple[str, str], List[str]]], topk: int = 50) -> pd.DataFrame:
    rows: list[dict] = []
    for method, rmap in rankings.items():
        for (app, qid), ranked in rmap.items():
            rows.append(
                {
                    'app': app,
                    'query_id': qid,
                    'method': method,
                    'ranked_topk': '|'.join(list(ranked)[: int(topk)]),
                    'ranked_len': int(len(ranked)),
                }
            )
    return pd.DataFrame(rows)


def _all_cross_inventory(perq_cross: pd.DataFrame) -> pd.DataFrame:
    if perq_cross is None or len(perq_cross) == 0:
        return pd.DataFrame(columns=['app', 'query_id', 'issue_key', 'cluster_id', 'n_rel', 'n_cross_rel', 'is_cross_query', 'subset'])
    out = perq_cross[perq_cross['is_cross_query'] == 1].copy()
    out['subset'] = 'all_cross'
    return out.reset_index(drop=True)


def _boundary_diag(perq_cross: pd.DataFrame, boundary_name: str, n_clusters: int) -> pd.DataFrame:
    cross_q_df = perq_cross[perq_cross['is_cross_query'] == 1].copy()
    return pd.DataFrame(
        [
            {
                'boundary': boundary_name,
                'n_queries': int(perq_cross.shape[0]),
                'n_clusters': int(n_clusters),
                'cross_query_count': int(cross_q_df.shape[0]),
                'frac_cross_query': float(cross_q_df.shape[0] / max(perq_cross.shape[0], 1)),
                'avg_rel': float(perq_cross['n_rel'].mean()) if len(perq_cross) else 0.0,
                'avg_cross_rel': float(cross_q_df['n_cross_rel'].mean()) if len(cross_q_df) else 0.0,
                'median_cross_rel': float(cross_q_df['n_cross_rel'].median()) if len(cross_q_df) else 0.0,
            }
        ]
    )


def run(cfg: MainConfig) -> Path:
    out_dir = ensure_dir(cfg.out_dir)
    corpus_dir = ensure_dir(Path(out_dir) / 'corpus')
    boundary_dir = ensure_dir(Path(out_dir) / 'boundaries')

    raw_reports, load_audit = load_reports(cfg.xlsx)
    corpus, corpus_audit = build_corpus(raw_reports, drop_singleton_issues=True)
    write_csv(corpus, corpus_dir / 'normalized_reports.csv')

    q_keys = list(corpus[['app', 'id']].itertuples(index=False, name=None))

    runs = []
    if cfg.run_tfidf:
        runs.append(
            run_tfidf_method(
                corpus,
                q_keys,
                TfidfMethodConfig(k=cfg.k, multi_Ls=cfg.multi_Ls, max_rank_cap=cfg.max_rank_cap, clusters_csv=cfg.tfidf_clusters_csv),
            )
        )
    if cfg.run_sbert:
        runs.append(
            run_sbert_method(
                corpus,
                q_keys,
                SbertMethodConfig(
                    k=cfg.k,
                    model=cfg.sbert_model,
                    device=cfg.sbert_device,
                    batch_size=cfg.sbert_batch,
                    multi_Ls=cfg.multi_Ls,
                    max_rank_cap=cfg.max_rank_cap,
                    clusters_csv=cfg.sbert_clusters_csv,
                ),
            )
        )
    if cfg.run_llmcluster:
        runs.append(
            run_llmcluster_method(
                corpus,
                q_keys,
                LlmClusterMethodConfig(
                    clusters_csv=cfg.llm_clusters_csv,
                    repr_kind=cfg.llm_repr_kind,
                    sbert_model=cfg.sbert_model,
                    sbert_device=cfg.sbert_device,
                    sbert_batch_size=cfg.sbert_batch,
                    llm_model=cfg.llm_model,
                    llm_api_url=cfg.llm_api_url,
                    llm_api_key=cfg.llm_api_key,
                    llm_api_key_env=cfg.llm_api_key_env,
                    llm_timeout=cfg.llm_timeout,
                    llm_temperature=cfg.llm_temperature,
                    llm_max_tokens=cfg.llm_max_tokens,
                    llm_retries=cfg.llm_retries,
                    llm_thinking_enabled=cfg.llm_thinking_enabled,
                    llm_verify_ssl=cfg.llm_verify_ssl,
                    llm_use_env_proxy=cfg.llm_use_env_proxy,
                    llm_verbose=cfg.llm_verbose,
                    llm_text_column=cfg.llm_text_column,
                    llm_max_per_iteration=cfg.llm_max_per_iteration,
                    llm_seed=cfg.llm_seed,
                    llm_summary_sample_size=cfg.llm_summary_sample_size,
                    llm_context_soft_limit=cfg.llm_context_soft_limit,
                    llm_mock=cfg.llm_mock,
                    multi_Ls=cfg.multi_Ls,
                    max_rank_cap=cfg.max_rank_cap,
                ),
            )
        )

    audits: list[dict] = []
    for boundary_run in runs:
        bname = boundary_run.boundary_name
        bdir = ensure_dir(Path(out_dir) / bname)
        write_csv(boundary_run.clusters_df, boundary_dir / f'clusters_{bname}.csv')

        rankings = _rename_method_keys(boundary_run.rankings)
        rankings['ecfa'] = build_ecfa_rankings(
            q_keys=q_keys,
            clusters_df=boundary_run.clusters_df,
            global_rankings=rankings['global'],
            multi_rankings=rankings['multi_L5'],
            cfg=EcfaMethodConfig(per_query_csv=cfg.ecfa_per_query_csv, max_rank_cap=cfg.max_rank_cap),
        )
        write_csv(_rankings_to_frame(rankings, topk=50), bdir / 'rankings_top50.csv')

        crossrel_map, perq_cross = build_crossrel_map(corpus, boundary_run.clusters_df)
        write_csv(perq_cross, bdir / 'crossrel_inventory.csv')
        write_csv(_boundary_diag(perq_cross, bname, int(boundary_run.clusters_df['cluster_id'].nunique())), bdir / 'boundary_diagnosis.csv')

        sdir = ensure_dir(bdir / 'all_cross')
        query_inventory = _all_cross_inventory(perq_cross)
        write_csv(query_inventory, sdir / 'query_inventory.csv')
        eval_keys = [(str(a), str(q)) for a, q in query_inventory[['app', 'query_id']].itertuples(index=False)]

        summary, per_query, by_app = evaluate_rankings_from_relmap(
            rankings_by_method=rankings,
            rel_map={k: crossrel_map.get(k, []) for k in eval_keys},
            boundary=bname,
            eval_keys=eval_keys,
            success_ks=cfg.success_ks,
            recall_ks=cfg.recall_ks,
            mrr_k=cfg.mrr_k,
            ndcg_k=cfg.ndcg_k,
            max_rank_cap=cfg.max_rank_cap,
        )
        write_csv(summary, sdir / 'summary.csv')
        write_csv(per_query, sdir / 'per_query_metrics.csv')
        write_csv(by_app, sdir / 'by_app_summary.csv')
        export_subset_enhanced_stats(sdir, bootstrap_cfg=BootstrapConfig(n_boot=cfg.bootstrap_n, seed=cfg.bootstrap_seed))

        audits.append(
            {
                'boundary': bname,
                'method_family': boundary_run.method_family,
                'n_ranked_queries': int(len(q_keys)),
                'n_cross_queries': int(query_inventory.shape[0]),
                'available_methods': sorted(rankings.keys()),
            }
        )

    write_json({'config': asdict(cfg), 'load_audit': asdict(load_audit), 'corpus_audit': asdict(corpus_audit), 'boundary_audits': audits}, Path(out_dir) / 'audit.json')
    return Path(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description='RQ3-2A clean main pipeline (paper-facing outputs only).')
    ap.add_argument('--xlsx', default='input/filtered_reports.csv')
    ap.add_argument('--ecfa_per_query_csv', default='input/ecfa/per_query_K20_ecfa.csv')
    ap.add_argument('--out_dir', default='output/run_main')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--multi_Ls', default='2,3,5')
    ap.add_argument('--max_rank_cap', type=int, default=200)
    ap.add_argument('--run_tfidf', type=int, default=1)
    ap.add_argument('--run_sbert', type=int, default=1)
    ap.add_argument('--run_llmcluster', type=int, default=0)
    ap.add_argument('--tfidf_clusters_csv', default='input/boundaries/clusters_tfidf_hc_K20.csv')
    ap.add_argument('--sbert_clusters_csv', default='input/boundaries/clusters_sbert_hc_K20.csv')
    ap.add_argument('--llm_clusters_csv', nargs='?', const='', default='')
    ap.add_argument('--llm_repr_kind', choices=['tfidf_char', 'sbert'], default='tfidf_char')
    ap.add_argument('--llm_model', default='glm-4.7')
    ap.add_argument('--llm_api_url', default='')
    ap.add_argument('--llm_api_key', default='')
    ap.add_argument('--llm_api_key_env', default='ZHIPUAI_API_KEY')
    ap.add_argument('--llm_timeout', type=int, default=180)
    ap.add_argument('--llm_temperature', type=float, default=0.0)
    ap.add_argument('--llm_max_tokens', type=int, default=4096)
    ap.add_argument('--llm_retries', type=int, default=5)
    ap.add_argument('--llm_thinking_enabled', type=int, default=0)
    ap.add_argument('--llm_verify_ssl', type=int, default=1)
    ap.add_argument('--llm_use_env_proxy', type=int, default=0)
    ap.add_argument('--llm_verbose', type=int, default=1)
    ap.add_argument('--llm_text_column', default='auto')
    ap.add_argument('--llm_max_per_iteration', type=int, default=50)
    ap.add_argument('--llm_seed', type=int, default=42)
    ap.add_argument('--llm_summary_sample_size', type=int, default=8)
    ap.add_argument('--llm_context_soft_limit', type=int, default=90000)
    ap.add_argument('--llm_mock', type=int, default=0)
    ap.add_argument('--sbert_model', default='')
    ap.add_argument('--sbert_device', default='cpu')
    ap.add_argument('--sbert_batch', type=int, default=64)
    ap.add_argument('--success_ks', default='10,20')
    ap.add_argument('--recall_ks', default='20,50')
    ap.add_argument('--mrr_k', type=int, default=20)
    ap.add_argument('--ndcg_k', type=int, default=20)
    ap.add_argument('--bootstrap_n', type=int, default=500)
    ap.add_argument('--bootstrap_seed', type=int, default=20260309)
    args = ap.parse_args()

    cfg = MainConfig(
        xlsx=args.xlsx,
        ecfa_per_query_csv=args.ecfa_per_query_csv,
        out_dir=args.out_dir,
        k=int(args.k),
        max_rank_cap=int(args.max_rank_cap),
        multi_Ls=_parse_ints(args.multi_Ls),
        run_tfidf=bool(int(args.run_tfidf)),
        run_sbert=bool(int(args.run_sbert)),
        run_llmcluster=bool(int(args.run_llmcluster)),
        tfidf_clusters_csv=str(args.tfidf_clusters_csv),
        sbert_clusters_csv=str(args.sbert_clusters_csv),
        llm_clusters_csv=str(args.llm_clusters_csv),
        llm_repr_kind=str(args.llm_repr_kind),
        llm_model=str(args.llm_model),
        llm_api_url=str(args.llm_api_url),
        llm_api_key=str(args.llm_api_key),
        llm_api_key_env=str(args.llm_api_key_env),
        llm_timeout=int(args.llm_timeout),
        llm_temperature=float(args.llm_temperature),
        llm_max_tokens=int(args.llm_max_tokens),
        llm_retries=int(args.llm_retries),
        llm_thinking_enabled=bool(int(args.llm_thinking_enabled)),
        llm_verify_ssl=bool(int(args.llm_verify_ssl)),
        llm_use_env_proxy=bool(int(args.llm_use_env_proxy)),
        llm_verbose=bool(int(args.llm_verbose)),
        llm_text_column=str(args.llm_text_column),
        llm_max_per_iteration=int(args.llm_max_per_iteration),
        llm_seed=int(args.llm_seed),
        llm_summary_sample_size=int(args.llm_summary_sample_size),
        llm_context_soft_limit=int(args.llm_context_soft_limit),
        llm_mock=bool(int(args.llm_mock)),
        sbert_model=str(args.sbert_model),
        sbert_device=str(args.sbert_device),
        sbert_batch=int(args.sbert_batch),
        success_ks=_parse_ints(args.success_ks),
        recall_ks=_parse_ints(args.recall_ks),
        mrr_k=int(args.mrr_k),
        ndcg_k=int(args.ndcg_k),
        bootstrap_n=int(args.bootstrap_n),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    run(cfg)


if __name__ == '__main__':
    main()
