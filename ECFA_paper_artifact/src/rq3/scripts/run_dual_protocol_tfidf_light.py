from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rq3fresh.data import load_reports, build_corpus
from rq3fresh.clustering import TfidfBoundaryConfig, cluster_tfidf_hc
from rq3fresh.representations import RepresentationConfig, RepresentationStore
from rq3fresh.methods import rank_text_baselines
from rq3fresh.ecfa_head import load_ecfa_rankings
from rq3fresh.metrics import build_crossrel_map, evaluate_rankings_from_relmap
from rq3fresh.protocols import build_allrel_map
from rq3fresh.hybrid_allrel import HybridAllRelConfig, build_hybrid_allrel_rankings
from rq3fresh.utils import ensure_dir, write_csv

METHODS = ['cluster_only', 'global', 'multi_L3', 'multi_L5', 'ecfa_head', 'ecfa_hybrid_allrel']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--ecfa_per_query_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    out_dir = ensure_dir(args.out_dir)
    raw,_ = load_reports(args.xlsx)
    corpus,_ = build_corpus(raw, drop_singleton_issues=True)
    clusters = cluster_tfidf_hc(corpus, TfidfBoundaryConfig(k=20, analyzer='char', ngram_min=2, ngram_max=4))
    rep = RepresentationStore(corpus, RepresentationConfig(kind='tfidf_char', tfidf_analyzer='char', tfidf_ngram_min=2, tfidf_ngram_max=4))
    rep.attach_clusters(clusters)
    allrel_map,_ = build_allrel_map(corpus)
    q_keys = list(allrel_map)
    rankings = rank_text_baselines(df_reports=corpus, clusters_df=clusters, rep_store=rep, q_keys=q_keys, multi_Ls=[3,5], max_rank_cap=200)
    ecfa_rankings = load_ecfa_rankings(args.ecfa_per_query_csv, max_rank_cap=200, prefer_inpool=True)
    rankings['ecfa_head'] = {k: ecfa_rankings.get(k, []) for k in q_keys}
    rankings['ecfa_hybrid_allrel'] = build_hybrid_allrel_rankings(
        q_keys=q_keys, global_rankings=rankings['global'], cluster_rankings=rankings['cluster_only'], ecfa_rankings=rankings['ecfa_head'], cfg=HybridAllRelConfig()
    )
    crossrel_map,_ = build_crossrel_map(corpus, clusters)
    cross_keys = [k for k,v in crossrel_map.items() if len(v)>0]
    all_summ=[]; cross_summ=[]
    for m in METHODS:
        s,_,_ = evaluate_rankings_from_relmap(rankings_by_method={m: rankings[m]}, rel_map=allrel_map, boundary='tfidf_hc_K20', eval_keys=q_keys, success_ks=(10,20), recall_ks=(20,50), mrr_k=20, ndcg_k=20, max_rank_cap=200)
        all_summ.append(s)
        s2,_,_ = evaluate_rankings_from_relmap(rankings_by_method={m: rankings[m]}, rel_map=crossrel_map, boundary='tfidf_hc_K20', eval_keys=cross_keys, success_ks=(10,20), recall_ks=(20,50), mrr_k=20, ndcg_k=20, max_rank_cap=200)
        cross_summ.append(s2)
    all_summary = pd.concat(all_summ, ignore_index=True)
    cross_summary = pd.concat(cross_summ, ignore_index=True)
    write_csv(all_summary, out_dir/'all_related_summary.csv')
    write_csv(cross_summary, out_dir/'cross_related_summary.csv')
    merged=all_summary.merge(cross_summary,on=['boundary','method'],suffixes=('_all','_cross'))
    rows=[]
    for _,r in merged.iterrows():
        row={'boundary':r['boundary'],'method':r['method']}
        for metric in ['success_at_20','mrr_at_20','recall_at_20','recall_at_50','ndcg_at_20']:
            a=float(r[f'{metric}_all']); c=float(r[f'{metric}_cross'])
            row[f'{metric}_all']=a; row[f'{metric}_cross']=c; row[f'{metric}_retention']=(c/a if a>1e-12 else 0.0)
        rows.append(row)
    write_csv(pd.DataFrame(rows), out_dir/'retention_summary.csv')
    print('[OK] wrote', out_dir)

if __name__=='__main__':
    main()
