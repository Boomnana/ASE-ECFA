

from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import pandas as pd
import networkx as nx


TYPE_MAP = {
    "功能模块": "MOD",
    "模块": "MOD",
    "用户操作": "OP",
    "操作": "OP",
    "界面元素": "ELEM",
    "UI元素": "ELEM",
    "元素": "ELEM",
    "用户感知现象": "PHEN",
    "现象": "PHEN",
    "诊断": "DIAG",
    "诊断信息": "DIAG",
    "问题陈述": "ISSUE",
    "问题": "ISSUE",
    "MOD": "MOD",
    "OP": "OP",
    "ELEM": "ELEM",
    "PHEN": "PHEN",
    "DIAG": "DIAG",
    "ISSUE": "ISSUE",
}
CANON_TYPES = ["MOD", "OP", "ELEM", "PHEN", "DIAG", "ISSUE", "UNK"]


def norm_type(x: Optional[str]) -> str:
    if x is None:
        return "UNK"
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return "UNK"
    t = TYPE_MAP.get(s, "UNK")
    return t if t in CANON_TYPES else "UNK"


def parse_prov(x) -> FrozenSet[str]:
    if x is None:
        return frozenset()
    if isinstance(x, float) and math.isnan(x):
        return frozenset()
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return frozenset()
    parts = [p.strip() for p in s.split("|")]
    parts = [p for p in parts if p]
    return frozenset(parts)


def safe_read_xlsx(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_excel(path, engine="openpyxl", dtype=str)


def norm_text(s: Optional[str]) -> str:

    if s is None:
        return ""
    x = str(s).strip()
    if not x or x.lower() == "nan":
        return ""
    x = x.lower()
    x = x.replace("\u3000", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def union_report_universe(node_prov: Dict[str, FrozenSet[str]]) -> Set[str]:
    D: Set[str] = set()
    for s in node_prov.values():
        D.update(s)
    return D


def type_counts(node_type: Dict[str, str], nodes: Iterable[str]) -> Dict[str, int]:
    c = Counter()
    for nid in nodes:
        c[node_type.get(nid, "UNK")] += 1
    return {t: int(c.get(t, 0)) for t in CANON_TYPES}


def make_sizes_equal_partition(n: int, m: int) -> List[int]:

    if m <= 0:
        return []
    m = min(m, n) if n > 0 else 0
    if m <= 0:
        return []
    base = n // m
    rem = n % m
    sizes = [base + (1 if i < rem else 0) for i in range(m)]
    sizes = [s for s in sizes if s > 0]
    return sizes


def adjust_sizes_to_target_count(sizes: List[int], target_m: int) -> List[int]:


    if target_m <= 0:
        target_m = 1
    total = sum(sizes)
    if total <= 0:
        return []

    target_m = min(target_m, total)
    sizes = [int(s) for s in sizes if int(s) > 0]
    if not sizes:
        return [total]


    while len(sizes) < target_m:
        i = max(range(len(sizes)), key=lambda k: sizes[k])
        if sizes[i] <= 1:
            break
        s = sizes[i]
        a = s // 2
        b = s - a
        if a <= 0 or b <= 0:
            break
        sizes[i] = a
        sizes.append(b)


    while len(sizes) > target_m:
        sizes.sort()
        s1 = sizes.pop(0)
        s2 = sizes.pop(0)
        sizes.append(s1 + s2)


    if sum(sizes) != total:

        diff = total - sum(sizes)
        sizes[0] += diff
    return sizes


@dataclass
class GraphData:
    app: str
    variant: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_type: Dict[str, str]
    node_prov: Dict[str, FrozenSet[str]]
    node_name: Dict[str, str]
    G_simple: nx.Graph
    type_conflicts: List[Tuple[str, Dict[str, int]]]


def build_graph(app: str, variant: str, nodes_path: Path, edges_path: Path) -> GraphData:
    nodes = safe_read_xlsx(nodes_path)
    edges = safe_read_xlsx(edges_path)


    for col in ["id", "source_row_index"]:
        if col not in nodes.columns:
            raise ValueError(f"[{app}/{variant}] nodes.xlsx missing column: {col}")
    for col in ["source_id", "target_id", "source_row_index"]:
        if col not in edges.columns:
            raise ValueError(f"[{app}/{variant}] edges.xlsx missing column: {col}")

    nodes["id"] = nodes["id"].astype(str)
    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)


    node_prov: Dict[str, FrozenSet[str]] = {}
    for _, r in nodes.iterrows():
        nid = str(r["id"])
        node_prov[nid] = parse_prov(r.get("source_row_index", ""))


    node_name: Dict[str, str] = {}
    if "name" in nodes.columns:
        for _, r in nodes.iterrows():
            nid = str(r["id"])
            node_name[nid] = str(r.get("name", "") if r.get("name", "") is not None else "")
    else:
        for nid in nodes["id"].tolist():
            node_name[nid] = ""


    node_type_votes: Dict[str, Counter] = defaultdict(Counter)
    if "type" in nodes.columns:
        for _, r in nodes.iterrows():
            nid = str(r["id"])
            t = norm_type(r.get("type", None))
            if t != "UNK":
                node_type_votes[nid][t] += 1
    if "source_type" in edges.columns:
        for _, r in edges.iterrows():
            nid = str(r["source_id"])
            t = norm_type(r.get("source_type", None))
            if t != "UNK":
                node_type_votes[nid][t] += 1
    if "target_type" in edges.columns:
        for _, r in edges.iterrows():
            nid = str(r["target_id"])
            t = norm_type(r.get("target_type", None))
            if t != "UNK":
                node_type_votes[nid][t] += 1

    node_type: Dict[str, str] = {}
    conflicts: List[Tuple[str, Dict[str, int]]] = []
    for nid in nodes["id"].tolist():
        votes = node_type_votes.get(nid, Counter())
        if not votes:
            node_type[nid] = "UNK"
            continue
        best, _ = votes.most_common(1)[0]
        node_type[nid] = best
        if len(votes) > 1:
            conflicts.append((nid, dict(votes)))


    edge_set: Set[Tuple[str, str]] = set()
    for _, r in edges.iterrows():
        u = str(r["source_id"])
        v = str(r["target_id"])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edge_set.add((a, b))

    G = nx.Graph()
    for nid in nodes["id"].tolist():
        G.add_node(nid)
    for (u, v) in edge_set:
        if u not in G:
            G.add_node(u)
            node_prov.setdefault(u, frozenset())
            node_type.setdefault(u, "UNK")
            node_name.setdefault(u, "")
        if v not in G:
            G.add_node(v)
            node_prov.setdefault(v, frozenset())
            node_type.setdefault(v, "UNK")
            node_name.setdefault(v, "")
        G.add_edge(u, v)

    return GraphData(app, variant, nodes, edges, node_type, node_prov, node_name, G, conflicts)


@dataclass
class Metrics:
    app: str
    variant: str
    n_nodes: int
    n_edges: int
    n_cc: int
    lcc_ratio: float
    lcc_n: int
    eligible_edges: int
    cer: float
    cer_j: float


    avg_node_gamma: float
    avg_edge_union_gamma: float
    avg_edge_inter_gamma: float


def compute_metrics(app: str, variant: str, G: nx.Graph, node_prov: Dict[str, FrozenSet[str]]) -> Metrics:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        return Metrics(app, variant, 0, 0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    comps = list(nx.connected_components(G))
    n_cc = len(comps)
    lcc_nodes = max(comps, key=len)
    lcc_n = len(lcc_nodes)
    lcc_ratio = lcc_n / n_nodes

    eligible = 0
    cer_cnt = 0
    cerj_sum = 0.0

    sum_node_gamma = 0
    sum_edge_union = 0
    sum_edge_inter = 0

    for v in G.nodes():
        sum_node_gamma += len(node_prov.get(v, frozenset()))
    avg_node_gamma = sum_node_gamma / n_nodes if n_nodes else 0.0

    for (u, v) in G.edges():
        pu = node_prov.get(u, frozenset())
        pv = node_prov.get(v, frozenset())
        if not pu or not pv:
            continue
        eligible += 1
        if pu != pv:
            cer_cnt += 1
        inter = len(pu.intersection(pv))
        union = len(pu.union(pv))
        cerj_sum += 1.0 - (inter / union)

        sum_edge_union += union
        sum_edge_inter += inter

    cer = (cer_cnt / eligible) if eligible else 0.0
    cer_j = (cerj_sum / eligible) if eligible else 0.0

    avg_edge_union_gamma = (sum_edge_union / eligible) if eligible else 0.0
    avg_edge_inter_gamma = (sum_edge_inter / eligible) if eligible else 0.0

    return Metrics(
        app,
        variant,
        n_nodes,
        n_edges,
        n_cc,
        lcc_ratio,
        lcc_n,
        eligible,
        cer,
        cer_j,
        avg_node_gamma,
        avg_edge_union_gamma,
        avg_edge_inter_gamma,
    )


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (m, math.sqrt(var))


def _wilcoxon_exact_pvalue(diffs: List[float]) -> float:


    vals = [d for d in diffs if abs(d) > 1e-12]
    n = len(vals)
    if n == 0:
        return 1.0

    abs_vals = [abs(d) for d in vals]
    order = sorted(range(n), key=lambda i: abs_vals[i])
    ranks = [0.0] * n
    i = 0
    r = 1
    while i < n:
        j = i
        while j + 1 < n and abs(abs_vals[order[j + 1]] - abs_vals[order[i]]) < 1e-12:
            j += 1
        avg = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        r += (j - i + 1)
        i = j + 1

    w_obs = sum(ranks[i] for i, d in enumerate(vals) if d > 0)

    all_ws = []
    for mask in range(1 << n):
        w = 0.0
        for i in range(n):
            if (mask >> i) & 1:
                w += ranks[i]
        all_ws.append(w)

    total = len(all_ws)
    le = sum(1 for w in all_ws if w <= w_obs + 1e-12)
    ge = sum(1 for w in all_ws if w >= w_obs - 1e-12)
    p = 2.0 * min(le / total, ge / total)
    return min(1.0, max(0.0, p))


def build_inflation_sheets(df_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:


    d = df_summary[df_summary["variant"].isin(["no_unify", "unify"])].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    metrics = ["avg_node_gamma", "avg_edge_union_gamma", "avg_edge_inter_gamma", "cer", "cer_j", "lcc_n", "lcc_ratio", "n_cc"]
    pivot = d.pivot_table(index="app", columns="variant", values=metrics, aggfunc="first")

    rows = []
    for app in pivot.index.tolist():
        row = {"App": app}
        for m in metrics:
            x = float(pivot.loc[app, (m, "no_unify")])
            y = float(pivot.loc[app, (m, "unify")])
            row[f"{m}_w/o"] = x
            row[f"{m}_w/"] = y
            row[f"{m}_delta"] = y - x
        rows.append(row)
    df_app = pd.DataFrame(rows)

    across_rows = []
    for m in metrics:
        xs = [float(pivot.loc[app, (m, "no_unify")]) for app in pivot.index.tolist()]
        ys = [float(pivot.loc[app, (m, "unify")]) for app in pivot.index.tolist()]
        mx, sx = _mean_std(xs)
        my, sy = _mean_std(ys)
        diffs = [yy - xx for xx, yy in zip(xs, ys)]
        p = _wilcoxon_exact_pvalue(diffs)

        if m == "lcc_ratio":
            mx2, sx2 = _mean_std([v * 100.0 for v in xs])
            my2, sy2 = _mean_std([v * 100.0 for v in ys])
            delta = (my2 - mx2)
            diffs2 = [(yy - xx) * 100.0 for xx, yy in zip(xs, ys)]
            p = _wilcoxon_exact_pvalue(diffs2)
            across_rows.append({
                "Metric": "LCC (%)",
                "w/o_mean": mx2, "w/o_std": sx2,
                "w/_mean": my2, "w/_std": sy2,
                "Delta": delta, "Delta_unit": "pp",
                "p_exact": p,
                "Formatted": f"{mx2:.1f}±{sx2:.1f} → {my2:.1f}±{sy2:.1f}",
            })
            continue

        if m in ("cer",):
            mx2, sx2 = _mean_std([v * 100.0 for v in xs])
            my2, sy2 = _mean_std([v * 100.0 for v in ys])
            delta = (my2 - mx2)
            diffs2 = [(yy - xx) * 100.0 for xx, yy in zip(xs, ys)]
            p = _wilcoxon_exact_pvalue(diffs2)
            across_rows.append({
                "Metric": "CER (%)",
                "w/o_mean": mx2, "w/o_std": sx2,
                "w/_mean": my2, "w/_std": sy2,
                "Delta": delta, "Delta_unit": "pp",
                "p_exact": p,
                "Formatted": f"{mx2:.1f}±{sx2:.1f} → {my2:.1f}±{sy2:.1f}",
            })
            continue

        across_rows.append({
            "Metric": m,
            "w/o_mean": mx, "w/o_std": sx,
            "w/_mean": my, "w/_std": sy,
            "Delta": my - mx, "Delta_unit": "abs",
            "p_exact": p,
            "Formatted": f"{mx:.3f}±{sx:.3f} → {my:.3f}±{sy:.3f}" if m in ("cer_j",) else f"{mx:.2f}±{sx:.2f} → {my:.2f}±{sy:.2f}",
        })

    df_across = pd.DataFrame(across_rows)

    def fmt_p(pv: float) -> str:
        if pv < 1e-4:
            return "<1e-4"
        return f"{pv:.3g}"

    def fmt_delta(val: float, unit: str) -> str:
        sign = "+" if val >= 0 else ""
        if unit == "pp":
            return f"{sign}{val:.1f}"
        return f"{sign}{val:.3f}" if abs(val) < 1 else f"{sign}{val:.1f}"

    df_across["p_fmt"] = df_across["p_exact"].apply(fmt_p)
    df_across["Delta_fmt"] = df_across.apply(lambda r: fmt_delta(float(r["Delta"]), str(r["Delta_unit"])), axis=1)
    return df_app, df_across


def load_aligned_clusters(aligned_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:


    if not aligned_path.exists():
        return None, None, None
    try:
        df = safe_read_xlsx(aligned_path)
    except Exception:
        return None, None, None
    if df.empty or len(df.columns) < 2:
        return df, None, None

    cols = list(df.columns)
    low = [str(c).strip().lower().replace("\n", " ") for c in cols]

    clu = None
    ent = None
    for i, c in enumerate(low):
        if clu is None and ("cluster" in c or "聚类" in c or "簇" in c or "规范" in c):
            clu = cols[i]
        if ent is None and ("entity" in c or "实体" in c or "变体" in c or "原词" in c):
            ent = cols[i]

    if clu is None or ent is None or clu == ent:
        clu, ent = cols[0], cols[1]
    return df, clu, ent


def build_type_aware_cluster_sizes(
    app: str,
    raw_gd: GraphData,
    unify_gd: GraphData,
    df_aligned: pd.DataFrame,
    cluster_col: str,
    entity_col: str,
) -> Tuple[Dict[str, List[int]], List[Dict[str, object]], Dict[str, object]]:


    ent2clu: Dict[str, str] = {}
    dup_conflict = 0
    for _, r in df_aligned.iterrows():
        ent = norm_text(r.get(entity_col, ""))
        clu = norm_text(r.get(cluster_col, ""))
        if not ent:
            continue
        if ent in ent2clu and ent2clu[ent] != clu:
            dup_conflict += 1
            continue
        ent2clu[ent] = clu if clu else ent

    raw_nodes = list(raw_gd.G_simple.nodes())
    raw_type = raw_gd.node_type
    raw_name = raw_gd.node_name
    unify_counts = type_counts(unify_gd.node_type, unify_gd.G_simple.nodes())


    clusters_by_type: Dict[str, Dict[str, List[str]]] = {t: defaultdict(list) for t in CANON_TYPES}
    matched_total = 0
    matched_by_type = Counter()
    raw_by_type = Counter()

    for nid in raw_nodes:
        t = raw_type.get(nid, "UNK")
        if t not in CANON_TYPES:
            t = "UNK"
        raw_by_type[t] += 1
        nm = norm_text(raw_name.get(nid, ""))
        if nm and nm in ent2clu:
            clu = ent2clu[nm]
            matched_total += 1
            matched_by_type[t] += 1
        else:
            clu = f"__UNMAPPED__{nid}"
        key = f"{t}::{clu}"
        clusters_by_type[t][key].append(nid)

    sizes_by_type: Dict[str, List[int]] = {}
    plan_rows: List[Dict[str, object]] = []
    mismatched_types = 0

    for t in CANON_TYPES:
        total_n = int(raw_by_type.get(t, 0))
        if total_n == 0:
            sizes_by_type[t] = []
            continue

        target_m = int(unify_counts.get(t, 0))
        if target_m <= 0:

            target_m = 1

        derived_sizes = [len(m) for m in clusters_by_type[t].values()]
        derived_m = len(derived_sizes)
        derived_sum = sum(derived_sizes)


        if derived_sum != total_n:
            mismatched_types += 1

        adjusted = adjust_sizes_to_target_count(derived_sizes, target_m)
        used = "aligned+adjusted" if derived_m != target_m else "aligned_exact"


        final_m = len(adjusted)
        final_sum = sum(adjusted)
        if final_m != min(target_m, total_n) or final_sum != total_n:

            adjusted = make_sizes_equal_partition(total_n, target_m)
            used = "fallback_equal"
            final_m = len(adjusted)
            final_sum = sum(adjusted)
            mismatched_types += 1

        sizes_by_type[t] = adjusted

        plan_rows.append({
            "App": app,
            "Type": t,
            "raw_n": total_n,
            "unify_target_groups": target_m,
            "aligned_clusters_derived": derived_m,
            "aligned_sum_sizes": derived_sum,
            "final_groups_used": final_m,
            "final_sum_sizes": final_sum,
            "matched_nodes_type": int(matched_by_type.get(t, 0)),
            "matched_rate_type": float(matched_by_type.get(t, 0)) / max(1, total_n),
            "plan_mode": used,
        })

    app_stats = {
        "app": app,
        "raw_nodes": len(raw_nodes),
        "matched_nodes": matched_total,
        "match_rate": matched_total / max(1, len(raw_nodes)),
        "ent2clu_size": len(ent2clu),
        "dup_entity_conflicts": dup_conflict,
        "mismatched_types": mismatched_types,
    }
    return sizes_by_type, plan_rows, app_stats


def rand_contract_once(
    app: str,
    seed: int,
    raw_gd: GraphData,
    unify_gd: GraphData,
    sizes_by_type: Dict[str, List[int]],
    mode_label: str,
) -> Tuple[nx.Graph, Dict[str, FrozenSet[str]], List[Dict[str, object]]]:
    rng = random.Random(seed)

    raw_nodes = list(raw_gd.G_simple.nodes())
    raw_type = raw_gd.node_type
    raw_prov = raw_gd.node_prov
    unify_counts = type_counts(unify_gd.node_type, unify_gd.G_simple.nodes())

    raw_by_type: Dict[str, List[str]] = {t: [] for t in CANON_TYPES}
    for nid in raw_nodes:
        t = raw_type.get(nid, "UNK")
        if t not in CANON_TYPES:
            t = "UNK"
        raw_by_type[t].append(nid)

    assign: Dict[str, str] = {}
    new_prov: Dict[str, FrozenSet[str]] = {}
    sanity_rows: List[Dict[str, object]] = []

    for t in CANON_TYPES:
        members = raw_by_type.get(t, [])
        if not members:
            continue
        rng.shuffle(members)


        sizes = sizes_by_type.get(t, [])
        if not sizes:

            m = max(1, int(unify_counts.get(t, 0)))
            sizes = make_sizes_equal_partition(len(members), m)


        if sum(sizes) != len(members):
            m = max(1, int(unify_counts.get(t, 0)))
            sizes = make_sizes_equal_partition(len(members), m)

        idx = 0
        created = 0
        for gi, sz in enumerate(sizes):
            group_nodes = members[idx: idx + sz]
            idx += sz
            if not group_nodes:
                continue
            new_id = f"{app}__RC__{t}__{mode_label}__{seed}__g{gi}"
            prov_union: Set[str] = set()
            for old in group_nodes:
                prov_union.update(raw_prov.get(old, frozenset()))
                assign[old] = new_id
            new_prov[new_id] = frozenset(prov_union)
            created += 1


        sanity_rows.append({
            "App": app,
            "Seed": seed,
            "Type": t,
            "raw_n": len(members),
            "groups_created": created,
            "sum_sizes": sum(sizes),
        })


    edge_set: Set[Tuple[str, str]] = set()
    for (u, v) in raw_gd.G_simple.edges():
        u2 = assign.get(u)
        v2 = assign.get(v)
        if u2 is None:
            u2 = f"{app}__RC__UNK__{mode_label}__{seed}__solo__{u}"
            new_prov.setdefault(u2, raw_prov.get(u, frozenset()))
        if v2 is None:
            v2 = f"{app}__RC__UNK__{mode_label}__{seed}__solo__{v}"
            new_prov.setdefault(v2, raw_prov.get(v, frozenset()))
        if u2 == v2:
            continue
        a, b = (u2, v2) if u2 < v2 else (v2, u2)
        edge_set.add((a, b))

    Gc = nx.Graph()
    for nid in new_prov.keys():
        Gc.add_node(nid)
    for (a, b) in edge_set:
        Gc.add_edge(a, b)

    return Gc, new_prov, sanity_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, default=r"ROOT_glm4.7", help=r"Path to ROOT_glm4.7")
    ap.add_argument("--apps", type=str, default="", help="Comma-separated apps (default: all).")
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--out_xlsx", type=str, default="rq1_metrics.xlsx")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.input_root)
    if not root.exists():
        raise FileNotFoundError(f"input_root not found: {root}")

    apps = [a.strip() for a in args.apps.split(",") if a.strip()] if args.apps.strip() else sorted(
        [p.name for p in root.iterdir() if p.is_dir()]
    )

    summary_rows: List[Dict[str, object]] = []
    rc_seed_rows: List[Dict[str, object]] = []
    baseline_plan_rows: List[Dict[str, object]] = []
    baseline_seed_plan_rows: List[Dict[str, object]] = []
    type_stats_rows: List[Dict[str, object]] = []
    conflict_rows: List[Dict[str, object]] = []
    sanity_app_rows: List[Dict[str, object]] = []
    aligned_stats_rows: List[Dict[str, object]] = []

    for app in apps:
        app_dir = root / app
        no_dir = app_dir / "no_unify"
        un_dir = app_dir / "unify"

        nodes_no = no_dir / "nodes.xlsx"
        edges_no = no_dir / "edges.xlsx"
        nodes_un = un_dir / "nodes.xlsx"
        edges_un = un_dir / "edges.xlsx"

        if not (nodes_no.exists() and edges_no.exists() and nodes_un.exists() and edges_un.exists()):
            print(f"[WARN] Skip {app}: missing no_unify/unify xlsx files.")
            continue

        raw_gd = build_graph(app, "no_unify", nodes_no, edges_no)
        uni_gd = build_graph(app, "unify", nodes_un, edges_un)


        for nid, votes in raw_gd.type_conflicts:
            conflict_rows.append({"App": app, "Variant": "no_unify", "NodeID": nid, "Votes": str(votes)})
        for nid, votes in uni_gd.type_conflicts:
            conflict_rows.append({"App": app, "Variant": "unify", "NodeID": nid, "Votes": str(votes)})


        raw_counts = type_counts(raw_gd.node_type, raw_gd.G_simple.nodes())
        uni_counts = type_counts(uni_gd.node_type, uni_gd.G_simple.nodes())
        for t in CANON_TYPES:
            type_stats_rows.append({
                "App": app, "Type": t,
                "n_nodes_no_unify": raw_counts.get(t, 0),
                "n_nodes_unify": uni_counts.get(t, 0),
                "delta": int(uni_counts.get(t, 0)) - int(raw_counts.get(t, 0))
            })


        D_raw = union_report_universe(raw_gd.node_prov)
        D_uni = union_report_universe(uni_gd.node_prov)
        universe_equal = (D_raw == D_uni)


        raw_empty_prov = sum(1 for n in raw_gd.G_simple.nodes() if len(raw_gd.node_prov.get(n, frozenset())) == 0)
        uni_empty_prov = sum(1 for n in uni_gd.G_simple.nodes() if len(uni_gd.node_prov.get(n, frozenset())) == 0)


        raw_typed = sum(1 for n in raw_gd.G_simple.nodes() if raw_gd.node_type.get(n, "UNK") != "UNK")
        uni_typed = sum(1 for n in uni_gd.G_simple.nodes() if uni_gd.node_type.get(n, "UNK") != "UNK")
        raw_typed_rate = raw_typed / max(1, raw_gd.G_simple.number_of_nodes())
        uni_typed_rate = uni_typed / max(1, uni_gd.G_simple.number_of_nodes())


        m_raw = compute_metrics(app, "no_unify", raw_gd.G_simple, raw_gd.node_prov)
        m_uni = compute_metrics(app, "unify", uni_gd.G_simple, uni_gd.node_prov)
        summary_rows.append(m_raw.__dict__)
        summary_rows.append(m_uni.__dict__)


        aligned_path = app_dir / "aligned_clusters.xlsx"
        df_aligned, clu_col, ent_col = load_aligned_clusters(aligned_path)

        mode_label = "node-count-matched"
        sizes_by_type: Dict[str, List[int]] = {}

        if df_aligned is not None and clu_col and ent_col:
            sizes_by_type, plan_rows, app_stats = build_type_aware_cluster_sizes(
                app=app, raw_gd=raw_gd, unify_gd=uni_gd,
                df_aligned=df_aligned, cluster_col=clu_col, entity_col=ent_col
            )
            baseline_plan_rows.extend(plan_rows)
            aligned_stats_rows.append({
                "App": app,
                "aligned_file": str(aligned_path),
                "cluster_col": str(clu_col),
                "entity_col": str(ent_col),
                **app_stats
            })

            if app_stats["match_rate"] >= 0.80:
                mode_label = "cluster-size-matched(type-aware)"
            else:
                mode_label = "node-count-matched(aligned_low_match)"
        else:

            for t in CANON_TYPES:
                total_n = raw_counts.get(t, 0)
                target_m = max(1, int(uni_counts.get(t, 0)))
                sizes_by_type[t] = make_sizes_equal_partition(total_n, target_m)


        print(f"\n[APP] {app}")
        print(
            f"  Raw:  |V|={m_raw.n_nodes} |E|={m_raw.n_edges} #CC={m_raw.n_cc} "
            f"LCC%={m_raw.lcc_ratio:.3f} LCC_N={m_raw.lcc_n} "
            f"CER={m_raw.cer:.3f} CER-J={m_raw.cer_j:.3f} eligibleE={m_raw.eligible_edges} "
            f"avg|G(v)|={m_raw.avg_node_gamma:.2f} avg|G(u)∪G(v)|={m_raw.avg_edge_union_gamma:.2f} avg|G(u)∩G(v)|={m_raw.avg_edge_inter_gamma:.2f}"
        )
        print(
            f"  Uni:  |V|={m_uni.n_nodes} |E|={m_uni.n_edges} #CC={m_uni.n_cc} "
            f"LCC%={m_uni.lcc_ratio:.3f} LCC_N={m_uni.lcc_n} "
            f"CER={m_uni.cer:.3f} CER-J={m_uni.cer_j:.3f} eligibleE={m_uni.eligible_edges} "
            f"avg|G(v)|={m_uni.avg_node_gamma:.2f} avg|G(u)∪G(v)|={m_uni.avg_edge_union_gamma:.2f} avg|G(u)∩G(v)|={m_uni.avg_edge_inter_gamma:.2f}"
        )
        print(f"  Universe |D| raw={len(D_raw)} uni={len(D_uni)} equal={universe_equal}")
        print(f"  ProvEmpty raw={raw_empty_prov}/{m_raw.n_nodes} uni={uni_empty_prov}/{m_uni.n_nodes} (empty prov nodes)")
        print(f"  TypeCoverage typed(raw)={raw_typed_rate:.2%} typed(uni)={uni_typed_rate:.2%} conflicts(raw)={len(raw_gd.type_conflicts)} conflicts(uni)={len(uni_gd.type_conflicts)}")
        if df_aligned is None or not clu_col or not ent_col:
            print("  aligned_clusters: NOT FOUND -> baseline node-count-matched")
        else:
            mr = aligned_stats_rows[-1]["match_rate"]
            print(f"  aligned_clusters: cols(cluster={clu_col}, entity={ent_col}), match_rate={mr:.2%} -> baseline {mode_label}")


        sanity_app_rows.append({
            "App": app,
            "raw_n_nodes": m_raw.n_nodes, "raw_n_edges": m_raw.n_edges,
            "unify_n_nodes": m_uni.n_nodes, "unify_n_edges": m_uni.n_edges,
            "raw_typed_rate": raw_typed_rate, "unify_typed_rate": uni_typed_rate,
            "raw_type_conflicts": len(raw_gd.type_conflicts), "unify_type_conflicts": len(uni_gd.type_conflicts),
            "raw_empty_prov_nodes": raw_empty_prov, "unify_empty_prov_nodes": uni_empty_prov,
            "D_raw": len(D_raw), "D_unify": len(D_uni), "D_equal": universe_equal,
            "baseline_mode": mode_label
        })


        for s in range(args.n_seeds):
            Gc, provc, seed_plan = rand_contract_once(
                app=app, seed=s, raw_gd=raw_gd, unify_gd=uni_gd,
                sizes_by_type=sizes_by_type, mode_label=mode_label
            )
            baseline_seed_plan_rows.extend(seed_plan)

            m_rc = compute_metrics(app, "rand_contract", Gc, provc)
            row = m_rc.__dict__.copy()
            row["seed"] = s
            row["mode"] = mode_label
            rc_seed_rows.append(row)

            if args.verbose and s == 0:

                if not (0.0 <= m_rc.cer <= 1.0 and 0.0 <= m_rc.cer_j <= 1.0 and 0.0 <= m_rc.lcc_ratio <= 1.0):
                    print(f"  [WARN] metric out of range on baseline seed {s}: {m_rc}")


    df_rc = pd.DataFrame(rc_seed_rows)
    agg_rows: List[Dict[str, object]] = []
    if not df_rc.empty:
        for app, g in df_rc.groupby("app"):
            row = {"app": app, "mode": g["mode"].iloc[0]}
            for col in ["n_nodes", "n_edges", "n_cc", "lcc_ratio", "lcc_n", "eligible_edges", "cer", "cer_j"]:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
            agg_rows.append(row)
    df_rc_agg = pd.DataFrame(agg_rows)


    out_path = Path(args.out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_summary = pd.DataFrame(summary_rows)
    df_types = pd.DataFrame(type_stats_rows)
    df_conf = pd.DataFrame(conflict_rows)
    df_sanity_app = pd.DataFrame(sanity_app_rows)
    df_aligned = pd.DataFrame(aligned_stats_rows)
    df_plan = pd.DataFrame(baseline_plan_rows)
    df_seed_plan = pd.DataFrame(baseline_seed_plan_rows)

    df_infl_app, df_infl_across = build_inflation_sheets(df_summary)

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_summary.to_excel(w, index=False, sheet_name="Summary")
        df_infl_app.to_excel(w, index=False, sheet_name="SanityInflation_App")
        df_infl_across.to_excel(w, index=False, sheet_name="SanityInflation_AcrossApp")
        df_rc.to_excel(w, index=False, sheet_name="RandContract_AllSeeds")
        df_rc_agg.to_excel(w, index=False, sheet_name="RandContract_Agg")
        df_types.to_excel(w, index=False, sheet_name="TypeStats")
        df_conf.to_excel(w, index=False, sheet_name="TypeConflicts")
        df_sanity_app.to_excel(w, index=False, sheet_name="SanityChecks_App")
        df_aligned.to_excel(w, index=False, sheet_name="AlignedClusters_Stats")
        df_plan.to_excel(w, index=False, sheet_name="BaselinePlan_Type")
        df_seed_plan.to_excel(w, index=False, sheet_name="BaselinePlan_SeedType")

    print(f"\n[DONE] Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
