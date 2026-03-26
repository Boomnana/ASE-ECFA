from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rq3fresh.clustering import (
    SbertBoundaryConfig,
    TfidfBoundaryConfig,
    cluster_sbert_hc,
    cluster_tfidf_hc,
)
from rq3fresh.data import build_corpus, load_reports
from rq3fresh.ecfa_head import load_ecfa_rankings
from rq3fresh.hybrid_allrel import HybridAllRelConfig, build_hybrid_allrel_rankings
from rq3fresh.metrics import build_crossrel_map, evaluate_rankings_from_relmap
from rq3fresh.methods import rank_text_baselines
from rq3fresh.protocols import build_allrel_map
from rq3fresh.representations import RepresentationConfig, RepresentationStore
from rq3fresh.utils import ensure_dir, normalize_report_id, write_csv


def _infer_mode(mode_arg: str, clusters_csv: str) -> str:
    mode = str(mode_arg or "auto").lower().strip()
    if mode in {"tfidf", "sbert"}:
        return mode
    s = str(clusters_csv or "").lower()
    if "sbert" in s:
        return "sbert"
    if "tfidf" in s:
        return "tfidf"
    return "tfidf"


def _boundary_name_from_path(p: str | Path, mode: str, k: int) -> str:
    if p:
        stem = Path(p).stem
        if stem.startswith("clusters_"):
            stem = stem[len("clusters_"):]

        if re.fullmatch(r"tfidf(_hc)?_K\d+", stem):
            stem = stem.replace("tfidf_", "tfidf_hc_") if stem.startswith("tfidf_K") else stem
        if re.fullmatch(r"sbert(_hc)?_K\d+", stem):
            stem = stem.replace("sbert_", "sbert_hc_") if stem.startswith("sbert_K") else stem
        return stem
    return f"{mode}_hc_K{int(k)}"


def _build_rep_cfg(args: argparse.Namespace, mode: str) -> RepresentationConfig:
    if mode == "sbert":
        if not str(args.sbert_model or "").strip():
            raise ValueError("--sbert_model is required when --mode sbert or when clusters_csv implies sbert")
        return RepresentationConfig(
            kind="sbert",
            sbert_model=str(args.sbert_model),
            sbert_device=str(args.sbert_device),
            sbert_batch_size=int(args.sbert_batch_size),
        )
    return RepresentationConfig(
        kind="tfidf_char",
        tfidf_analyzer="char",
        tfidf_ngram_min=int(args.tfidf_ngram_min),
        tfidf_ngram_max=int(args.tfidf_ngram_max),
    )


def _read_or_build_clusters(corpus: pd.DataFrame, args: argparse.Namespace, mode: str, out_dir: Path) -> tuple[pd.DataFrame, str]:
    if str(args.clusters_csv or "").strip():
        p = Path(str(args.clusters_csv))
        if not p.exists():
            raise FileNotFoundError(str(p))
        clusters = pd.read_csv(p)
        return clusters, _boundary_name_from_path(p, mode, int(args.k))

    if mode == "sbert":
        cfg = SbertBoundaryConfig(
            k=int(args.k),
            model=str(args.sbert_model),
            batch_size=int(args.sbert_batch_size),
            device=str(args.sbert_device),
        )
        clusters = cluster_sbert_hc(corpus, cfg)
    else:
        cfg = TfidfBoundaryConfig(
            k=int(args.k),
            analyzer="char",
            ngram_min=int(args.tfidf_ngram_min),
            ngram_max=int(args.tfidf_ngram_max),
        )
        clusters = cluster_tfidf_hc(corpus, cfg)
    boundary_name = f"{mode}_hc_K{int(args.k)}"
    write_csv(clusters, out_dir / f"clusters_{boundary_name}.csv")
    return clusters, boundary_name


def _coverage_report(corpus: pd.DataFrame, q_keys: list[tuple[str, str]], ecfa_rankings: dict[tuple[str, str], list[str]]) -> pd.DataFrame:
    corpus_pairs = {(str(a), normalize_report_id(i)) for a, i in zip(corpus["app"], corpus["id"])}
    rows = []
    missing = 0
    empty = 0
    for app, qid in q_keys:
        has_row = (app, qid) in corpus_pairs
        ranks = ecfa_rankings.get((app, qid))
        present = ranks is not None
        nonempty = bool(ranks)
        if not present:
            missing += 1
        elif not nonempty:
            empty += 1
        rows.append({
            "app": app,
            "query_id": qid,
            "in_corpus": int(has_row),
            "ecfa_row_present": int(present),
            "ecfa_rank_nonempty": int(nonempty),
            "ecfa_rank_len": int(len(ranks)) if ranks is not None else 0,
        })
    df = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {
            "n_queries": int(len(q_keys)),
            "n_ecfa_rows_present": int(df["ecfa_row_present"].sum()) if len(df) else 0,
            "n_ecfa_nonempty": int(df["ecfa_rank_nonempty"].sum()) if len(df) else 0,
            "n_missing": int(missing),
            "n_empty": int(empty),
        }
    ])
    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run dual protocol (All-related + CrossRel) evaluation")
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--clusters_csv", default="")
    ap.add_argument("--ecfa_per_query_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--mode", choices=["auto", "tfidf", "sbert"], default="auto")
    ap.add_argument("--sbert_model", default="")
    ap.add_argument("--sbert_device", default="cpu")
    ap.add_argument("--sbert_batch_size", type=int, default=64)
    ap.add_argument("--tfidf_ngram_min", type=int, default=2)
    ap.add_argument("--tfidf_ngram_max", type=int, default=4)
    ap.add_argument("--hybrid_global_topn", type=int, default=120)
    ap.add_argument("--hybrid_cluster_topn", type=int, default=50)
    ap.add_argument("--hybrid_ecfa_topn", type=int, default=80)
    ap.add_argument("--w_global", type=float, default=0.60)
    ap.add_argument("--w_cluster", type=float, default=0.10)
    ap.add_argument("--w_ecfa", type=float, default=0.30)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    mode = _infer_mode(args.mode, args.clusters_csv)

    raw, load_audit = load_reports(args.xlsx)
    corpus, corpus_audit = build_corpus(raw, drop_singleton_issues=True)
    write_csv(corpus, out_dir / "normalized_reports.csv")

    clusters, boundary_name = _read_or_build_clusters(corpus, args, mode, out_dir)
    write_csv(clusters, out_dir / f"clusters_{boundary_name}.csv")

    rep_cfg = _build_rep_cfg(args, mode)
    rep = RepresentationStore(corpus, rep_cfg)
    rep.attach_clusters(clusters)

    allrel_map, perq_all = build_allrel_map(corpus)
    q_keys = list(allrel_map)
    rankings = rank_text_baselines(
        df_reports=corpus,
        clusters_df=clusters,
        rep_store=rep,
        q_keys=q_keys,
        multi_Ls=[3, 5],
        max_rank_cap=200,
    )

    ecfa_rankings = load_ecfa_rankings(args.ecfa_per_query_csv, max_rank_cap=200, prefer_inpool=True)
    coverage_df, coverage_summary = _coverage_report(corpus, q_keys, ecfa_rankings)
    write_csv(coverage_df, out_dir / "ecfa_coverage_by_query.csv")
    write_csv(coverage_summary, out_dir / "ecfa_coverage_summary.csv")

    rankings["ecfa_head"] = {k: ecfa_rankings.get(k, []) for k in q_keys}
    rankings["ecfa_hybrid_allrel"] = build_hybrid_allrel_rankings(
        q_keys=q_keys,
        global_rankings=rankings["global"],
        cluster_rankings=rankings["cluster_only"],
        ecfa_rankings=rankings["ecfa_head"],
        cfg=HybridAllRelConfig(
            global_topn=int(args.hybrid_global_topn),
            cluster_topn=int(args.hybrid_cluster_topn),
            ecfa_topn=int(args.hybrid_ecfa_topn),
            w_global=float(args.w_global),
            w_cluster=float(args.w_cluster),
            w_ecfa=float(args.w_ecfa),
            max_rank_cap=200,
        ),
    )

    use_methods = ["cluster_only", "global", "multi_L3", "multi_L5", "ecfa_head", "ecfa_hybrid_allrel"]

    summary_all, perq_metrics_all, byapp_all = evaluate_rankings_from_relmap(
        rankings_by_method={k: rankings[k] for k in use_methods},
        rel_map=allrel_map,
        boundary=boundary_name,
        eval_keys=q_keys,
        success_ks=(10, 20),
        recall_ks=(20, 50),
        mrr_k=20,
        ndcg_k=20,
        max_rank_cap=200,
    )
    all_dir = ensure_dir(out_dir / "all_related")
    write_csv(summary_all, all_dir / "summary.csv")
    write_csv(perq_metrics_all, all_dir / "per_query_metrics.csv")
    write_csv(byapp_all, all_dir / "by_app_summary.csv")
    write_csv(perq_all, all_dir / "query_inventory.csv")

    crossrel_map, perq_cross = build_crossrel_map(corpus, clusters)
    cross_keys = [k for k, v in crossrel_map.items() if len(v) > 0]
    summary_cross, perq_metrics_cross, byapp_cross = evaluate_rankings_from_relmap(
        rankings_by_method={k: rankings[k] for k in use_methods},
        rel_map=crossrel_map,
        boundary=boundary_name,
        eval_keys=cross_keys,
        success_ks=(10, 20),
        recall_ks=(20, 50),
        mrr_k=20,
        ndcg_k=20,
        max_rank_cap=200,
    )
    cross_dir = ensure_dir(out_dir / "cross_related")
    write_csv(summary_cross, cross_dir / "summary.csv")
    write_csv(perq_metrics_cross, cross_dir / "per_query_metrics.csv")
    write_csv(byapp_cross, cross_dir / "by_app_summary.csv")
    write_csv(perq_cross, cross_dir / "query_inventory.csv")

    if len(summary_all) and len(summary_cross):
        key = ["boundary", "method"]
        merged = summary_all.merge(summary_cross, on=key, suffixes=("_all", "_cross"))
        out_rows = []
        for _, r in merged.iterrows():
            row = {
                "boundary": r["boundary"],
                "method": r["method"],
                "n_eval_queries_all": int(r["n_eval_queries_all"]),
                "n_eval_queries_cross": int(r["n_eval_queries_cross"]),
            }
            for metric in ["success_at_20", "mrr_at_20", "recall_at_20", "recall_at_50", "ndcg_at_20"]:
                a = float(r.get(metric + "_all", 0.0))
                c = float(r.get(metric + "_cross", 0.0))
                row[metric + "_all"] = a
                row[metric + "_cross"] = c
                row[metric + "_retention"] = (c / a) if a > 1e-12 else 0.0
            out_rows.append(row)
        write_csv(pd.DataFrame(out_rows), out_dir / "retention_summary.csv")

    audit = {
        "mode": mode,
        "boundary_name": boundary_name,
        "xlsx": str(args.xlsx),
        "clusters_csv": str(args.clusters_csv or ""),
        "ecfa_per_query_csv": str(args.ecfa_per_query_csv),
        "n_reports": int(corpus.shape[0]),
        "n_queries_all_related": int(len(q_keys)),
        "n_queries_cross_related": int(len(cross_keys)),
        "load_audit": asdict(load_audit),
        "corpus_audit": asdict(corpus_audit),
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] dual protocol outputs -> {out_dir}")


if __name__ == "__main__":
    main()
