from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .clustering import SbertBoundaryConfig, TfidfBoundaryConfig, cluster_sbert_hc, cluster_tfidf_hc
from .data import build_corpus, load_reports
from .metrics import build_crossrel_map, evaluate_rankings_from_relmap
from .methods import rank_text_baselines
from .representations import RepresentationConfig, RepresentationStore
from .reporting import export_main_tables, export_plot_data
from .subsets import build_report_anchor_inventory, build_subset_protocol
from .utils import ensure_dir, write_csv, write_json
from .ecfa_head import EcfaHeadRunConfig, run_ecfa_head_and_export_per_query, load_ecfa_rankings as load_ecfa_head_rankings
from .ecfa_legacy import EcfaRunConfig, run_ecfa_v3a_and_export_per_query, load_ecfa_rankings as load_ecfa_legacy_rankings


@dataclass(frozen=True)
class PipelineConfig:
    xlsx: str
    out_root: str
    run_tag: Optional[str] = None
    input_root: str = 'input/ROOT_glm4.7'
    k: int = 20
    drop_singleton_issues: bool = True
    run_tfidf: bool = True
    run_sbert: bool = False
    sbert_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    sbert_device: str = 'cpu'
    sbert_batch: int = 64
    topk_success: tuple[int, ...] = (10, 20)
    topk_recall: tuple[int, ...] = (20, 50)
    mrr_k: int = 20
    ndcg_k: int = 20
    max_rank_cap: int = 200
    multi_Ls: tuple[int, ...] = (3, 5)
    retr_repr: str = 'auto'
    run_ecfa: bool = True
    ecfa_variant: str = 'head'
    ecfa_preset: str = 'aggressive'
    ecfa_per_query_csv: Optional[str] = None
    limit_queries: int = 0


def _run_dir(cfg: PipelineConfig) -> Path:
    tag = cfg.run_tag or time.strftime('%Y%m%d_%H%M%S')
    return ensure_dir(Path(cfg.out_root) / tag)


def _ecfa_head_cfg(cfg: PipelineConfig, out_dir: Path) -> EcfaHeadRunConfig:
    preset = str(cfg.ecfa_preset).lower()
    if preset == 'balanced':
        params = dict(retr_top_views=0, pre_refine_cap=600, retr_shared_weight=1.0, wt_text=0.58, wt_support=0.18, wt_shared=0.18, wt_cost=0.06)
    elif preset == 'ultra':
        params = dict(retr_top_views=8, pre_refine_cap=100, retr_shared_weight=2.2, wt_text=0.38, wt_support=0.28, wt_shared=0.26, wt_cost=0.08)
    else:
        params = dict(retr_top_views=12, pre_refine_cap=160, retr_shared_weight=1.8, wt_text=0.44, wt_support=0.24, wt_shared=0.24, wt_cost=0.08)
    return EcfaHeadRunConfig(
        xlsx=Path(cfg.xlsx),
        input_root=Path(cfg.input_root),
        out_dir=out_dir,
        n_clusters=int(cfg.k),
        retr_view_pool='merged',
        ecfa_scope='pool',
        ecfa_budget=200,
        ecfa_r_max=0,
        topk_list=(20, 50, 100),
        max_rank_cap=int(cfg.max_rank_cap),
        candidate_ranker='hybrid',
        refine=True,
        **params,
    )


def _ecfa_legacy_cfg(cfg: PipelineConfig, out_dir: Path) -> EcfaRunConfig:
    return EcfaRunConfig(
        xlsx=Path(cfg.xlsx),
        input_root=Path(cfg.input_root),
        out_dir=out_dir,
        n_clusters=int(cfg.k),
        retr_view_pool='merged',
        ecfa_scope='pool',
        ecfa_budget=200,
        ecfa_r_max=0,
        topk_list=(20, 50, 100),
        max_rank_cap=int(cfg.max_rank_cap),
        retr_ranker='hybrid',
        retr_top_views=12,
        retr_shared_weight=1.8,
        refine=True,
        refine_max_candidates=160,
    )


def _repr_kind(boundary_name: str, explicit: str) -> str:
    if explicit != 'auto':
        return explicit
    if boundary_name.startswith('sbert'):
        return 'sbert'
    return 'tfidf_char'


def run_pipeline(cfg: PipelineConfig) -> Path:
    run_dir = _run_dir(cfg)
    raw_reports, load_audit = load_reports(cfg.xlsx)
    corpus, corpus_audit = build_corpus(raw_reports, drop_singleton_issues=bool(cfg.drop_singleton_issues))
    write_csv(corpus, run_dir / 'corpus' / 'normalized_reports.csv')
    write_json({'load_audit': asdict(load_audit), 'corpus_audit': asdict(corpus_audit), 'config': asdict(cfg)}, run_dir / 'audit.json')

    report_anchor_df = None
    if cfg.run_ecfa or cfg.ecfa_per_query_csv:
        print('[RQ3-2A] building report anchor inventory...')
        report_anchor_df = build_report_anchor_inventory(corpus, cfg.input_root)
        write_csv(report_anchor_df, run_dir / 'corpus' / 'report_anchor_inventory.csv')

    boundaries: list[tuple[str, object]] = []
    bdir = ensure_dir(run_dir / 'boundaries')
    if cfg.run_tfidf:
        print('[RQ3-2A] building TFIDF boundary...')
        tfcfg = TfidfBoundaryConfig(k=int(cfg.k), analyzer='char', ngram_min=2, ngram_max=4)
        tf = cluster_tfidf_hc(corpus, tfcfg)
        write_csv(tf, bdir / f'clusters_tfidf_K{int(cfg.k)}.csv')
        boundaries.append((f'tfidf_hc_K{int(cfg.k)}', tf))
    if cfg.run_sbert:
        print('[RQ3-2A] building SBERT boundary...')
        scfg = SbertBoundaryConfig(k=int(cfg.k), model=cfg.sbert_model, batch_size=int(cfg.sbert_batch), device=cfg.sbert_device)
        sb = cluster_sbert_hc(corpus, scfg)
        write_csv(sb, bdir / f'clusters_sbert_K{int(cfg.k)}.csv')
        boundaries.append((f'sbert_hc_K{int(cfg.k)}', sb))
    if not boundaries:
        raise ValueError('Enable at least one boundary.')

    ecfa_rankings = None
    ecfa_method_name = None
    if cfg.run_ecfa or cfg.ecfa_per_query_csv:
        print(f'[RQ3-2A] preparing ECFA ({cfg.ecfa_variant}, preset={cfg.ecfa_preset})...')
        ecfa_out_dir = ensure_dir(run_dir / 'ecfa')
        if cfg.ecfa_per_query_csv:
            ecfa_csv = Path(cfg.ecfa_per_query_csv)
        elif cfg.ecfa_variant == 'legacy':
            first_clusters = bdir / ('clusters_sbert_K%d.csv' % cfg.k if cfg.run_sbert else 'clusters_tfidf_K%d.csv' % cfg.k)
            ecfa_csv = run_ecfa_v3a_and_export_per_query(_ecfa_legacy_cfg(cfg, ecfa_out_dir), df_reports=corpus, clusters_csv=first_clusters)
            print(f'[RQ3-2A] ECFA legacy rankings: {ecfa_csv}')
        else:
            first_clusters = bdir / ('clusters_sbert_K%d.csv' % cfg.k if cfg.run_sbert else 'clusters_tfidf_K%d.csv' % cfg.k)
            ecfa_csv = run_ecfa_head_and_export_per_query(_ecfa_head_cfg(cfg, ecfa_out_dir), df_reports=corpus, clusters_csv=first_clusters)
            print(f'[RQ3-2A] ECFA head rankings: {ecfa_csv}')
        if cfg.ecfa_variant == 'legacy':
            ecfa_rankings = load_ecfa_legacy_rankings(ecfa_csv, max_rank_cap=int(cfg.max_rank_cap), prefer_inpool=True)
            ecfa_method_name = 'ecfa_legacy'
        else:
            ecfa_rankings = load_ecfa_head_rankings(ecfa_csv, max_rank_cap=int(cfg.max_rank_cap), prefer_inpool=True)
            ecfa_method_name = 'ecfa_head'

    for boundary_name, clusters_df in boundaries:
        eval_root = ensure_dir(run_dir / 'eval' / boundary_name)
        crossrel_map, perq_cross = build_crossrel_map(corpus, clusters_df)
        cross_q = perq_cross[perq_cross['is_cross_query'] == 1][['app', 'query_id']].drop_duplicates()
        if int(cfg.limit_queries) > 0:
            cross_q = cross_q.head(int(cfg.limit_queries))
        q_keys = list(zip(cross_q['app'].astype(str).tolist(), cross_q['query_id'].astype(str).tolist()))
        print(f'[RQ3-2A] boundary={boundary_name} cross_queries={len(q_keys)}')
        rep_cfg = RepresentationConfig(kind=_repr_kind(boundary_name, cfg.retr_repr), tfidf_analyzer='char', tfidf_ngram_min=2, tfidf_ngram_max=4, sbert_model=cfg.sbert_model, sbert_device=cfg.sbert_device, sbert_batch_size=int(cfg.sbert_batch))
        print(f'[RQ3-2A] boundary={boundary_name} building representation store ({rep_cfg.kind})...')
        rep_store = RepresentationStore(corpus, rep_cfg)
        rep_store.attach_clusters(clusters_df)
        print(f'[RQ3-2A] boundary={boundary_name} ranking baselines...')
        rankings = rank_text_baselines(df_reports=corpus, clusters_df=clusters_df, rep_store=rep_store, q_keys=q_keys, multi_Ls=list(cfg.multi_Ls), max_rank_cap=int(cfg.max_rank_cap))
        if ecfa_rankings is not None and ecfa_method_name is not None:
            rankings[ecfa_method_name] = {k: ecfa_rankings.get(k, []) for k in q_keys}

        subset_specs, subset_audit, dedup_rel_map, id_to_text_key = build_subset_protocol(
            df_reports=corpus,
            clusters_df=clusters_df,
            rep_store=rep_store,
            report_anchor_df=report_anchor_df,
        )
        write_csv(subset_audit, eval_root / 'subset_audit.csv')

        for subset_name, spec in subset_specs.items():
            subset_dir = ensure_dir(eval_root / subset_name)
            eval_keys = [k for k in spec.eval_keys if k in crossrel_map]
            if int(cfg.limit_queries) > 0:
                eval_keys = eval_keys[: int(cfg.limit_queries)]
            if spec.rel_mode == 'text_key':
                summary, per_query, by_app = evaluate_rankings_from_relmap(
                    rankings_by_method=rankings,
                    rel_map={k: dedup_rel_map.get(k, set()) for k in eval_keys},
                    boundary=boundary_name,
                    eval_keys=eval_keys,
                    success_ks=tuple(cfg.topk_success),
                    recall_ks=tuple(cfg.topk_recall),
                    mrr_k=int(cfg.mrr_k),
                    ndcg_k=int(cfg.ndcg_k),
                    max_rank_cap=int(cfg.max_rank_cap),
                    item_to_rel=lambda app, rid: id_to_text_key.get((app, rid), ''),
                    query_drop_rel={(a, q): id_to_text_key.get((a, q), '') for (a, q) in eval_keys},
                )
            else:
                summary, per_query, by_app = evaluate_rankings_from_relmap(
                    rankings_by_method=rankings,
                    rel_map={k: crossrel_map.get(k, []) for k in eval_keys},
                    boundary=boundary_name,
                    eval_keys=eval_keys,
                    success_ks=tuple(cfg.topk_success),
                    recall_ks=tuple(cfg.topk_recall),
                    mrr_k=int(cfg.mrr_k),
                    ndcg_k=int(cfg.ndcg_k),
                    max_rank_cap=int(cfg.max_rank_cap),
                )
            write_csv(summary, subset_dir / 'summary.csv')
            write_csv(per_query, subset_dir / 'per_query_metrics.csv')
            write_csv(by_app, subset_dir / 'by_app_summary.csv')

    export_main_tables(run_dir)
    export_plot_data(run_dir)
    print(f'[Done] run_dir={run_dir}')
    return run_dir
