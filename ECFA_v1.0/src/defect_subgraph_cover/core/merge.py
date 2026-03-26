from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from .subgraph import Subgraph, compute_cost


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def weighted_jaccard(parts_a: Dict[str, Set[str]], parts_b: Dict[str, Set[str]], weights: Dict[str, float]) -> float:
    score = 0.0
    wsum = 0.0
    for k, w in weights.items():
        if w <= 0:
            continue
        score += w * jaccard(parts_a.get(k, set()), parts_b.get(k, set()))
        wsum += w
    if wsum <= 0:
        raise ValueError("weights must have at least one positive entry")
    return score / wsum


def _signature_sets(
    node_ids: Iterable[str],
    node_type_by_id: Dict[str, str],
    type_alias: Dict[str, str],
) -> Dict[str, Set[str]]:
    sig: Dict[str, Set[str]] = {
        "ISSUE": set(),
        "PHEN": set(),
        "DIAG": set(),
        "OP": set(),
        "ELEM": set(),
        "MOD": set(),
    }
    for nid in node_ids:
        t_raw = node_type_by_id.get(str(nid))
        if not t_raw:
            continue
        t = type_alias.get(t_raw, t_raw)
        if t in sig:
            sig[t].add(str(nid))
    return sig


def merge_similar_subgraphs(
    subgraphs: List[Subgraph],
    thresh: float = 0.55,
    *,
    merge_mode: Optional[str] = None,
    node_type_by_id: Optional[Dict[str, str]] = None,
    weights: Optional[Dict[str, float]] = None,
    min_issue_jaccard: float = 0.2,
    min_phen_diag_jaccard_when_no_issue: float = 0.5,
    min_report_jaccard_when_no_issue: float = 0.0,
    max_nodes_after_merge: Optional[int] = None,
    type_alias: Optional[Dict[str, str]] = None,
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.01,
) -> List[Subgraph]:
    if merge_mode is None:
        mode = "report" if node_type_by_id is None else "signature"
    else:
        mode = str(merge_mode).strip().lower()
        if mode not in {"report", "signature"}:
            raise ValueError("merge_mode must be one of: report, signature")

    if mode == "report":
        def _report_sort_key(sg: Subgraph) -> tuple:
            return (
                len(sg.report_ids),
                len(sg.covered_universe),
                len(sg.node_ids),
                str(sg.subgraph_id),
                str(sg.seed_node),
            )

        remaining = sorted(subgraphs, key=_report_sort_key, reverse=True)
        merged: List[Subgraph] = []
        used = [False] * len(remaining)
        for i, sg in enumerate(remaining):
            if used[i]:
                continue
            rep_union = set(sg.report_ids)
            node_union = set(sg.node_ids)
            edge_union = set(sg.edge_ids)
            covered_union = set(sg.covered_universe)
            seeds = [sg.seed_node]

            changed = True
            while changed:
                changed = False
                for j in range(i + 1, len(remaining)):
                    if used[j]:
                        continue
                    sg2 = remaining[j]
                    if jaccard(rep_union, sg2.report_ids) >= thresh:
                        if max_nodes_after_merge is not None:
                            if len(node_union | set(sg2.node_ids)) > max_nodes_after_merge:
                                continue
                        used[j] = True
                        rep_union |= sg2.report_ids
                        node_union |= sg2.node_ids
                        edge_union |= sg2.edge_ids
                        covered_union |= sg2.covered_universe
                        seeds.append(sg2.seed_node)
                        changed = True

            used[i] = True
            new_id = sg.subgraph_id if len(seeds) == 1 else f"{sg.subgraph_id}_m{len(seeds)}"
            cost = compute_cost(rep_union, node_union, edge_union, alpha=alpha, beta=beta, gamma=gamma)
            merged.append(
                Subgraph(
                    subgraph_id=new_id,
                    seed_node=seeds[0],
                    report_ids=rep_union,
                    node_ids=node_union,
                    edge_ids=edge_union,
                    covered_universe=covered_union,
                    cost=cost,
                )
            )
        return merged

    if node_type_by_id is None:
        raise ValueError("node_type_by_id is required when merge_mode='signature'")

    if weights is None:
        weights = {"ISSUE": 0.45, "PHEN": 0.30, "DIAG": 0.20, "OP": 0.05}
    if type_alias is None:
        type_alias = {
            "问题陈述": "ISSUE",
            "用户感知现象": "PHEN",
            "系统诊断信息": "DIAG",
            "用户操作": "OP",
            "影响元素": "ELEM",
            "功能模块": "MOD",
        }

    remaining = subgraphs[:]
    sigs = [_signature_sets(sg.node_ids, node_type_by_id=node_type_by_id, type_alias=type_alias) for sg in remaining]

    def _signature_sort_key(sg_sig: tuple) -> tuple:
        sg, sig = sg_sig
        issue_n = len(sig.get("ISSUE", set()))
        phen_diag_n = len(sig.get("PHEN", set())) + len(sig.get("DIAG", set()))
        return (
            issue_n,
            phen_diag_n,
            len(sg.report_ids),
            len(sg.covered_universe),
            len(sg.node_ids),
            str(sg.subgraph_id),
            str(sg.seed_node),
        )

    paired = sorted(zip(remaining, sigs), key=_signature_sort_key, reverse=True)
    remaining = [p[0] for p in paired]
    sigs = [p[1] for p in paired]

    used = [False] * len(remaining)
    merged: List[Subgraph] = []

    for i, sg in enumerate(remaining):
        if used[i]:
            continue

        rep_union = set(sg.report_ids)
        node_union = set(sg.node_ids)
        edge_union = set(sg.edge_ids)
        covered_union = set(sg.covered_universe)
        seeds = [sg.seed_node]

        sig_union = {k: set(v) for k, v in sigs[i].items()}

        changed = True
        while changed:
            changed = False
            for j in range(i + 1, len(remaining)):
                if used[j]:
                    continue

                sg2 = remaining[j]
                sig2 = sigs[j]

                issue_j = jaccard(sig_union["ISSUE"], sig2["ISSUE"])
                phen_diag_union = sig_union["PHEN"] | sig_union["DIAG"]
                phen_diag_2 = sig2["PHEN"] | sig2["DIAG"]
                phen_diag_j = jaccard(phen_diag_union, phen_diag_2)

                if issue_j <= 1e-12 and phen_diag_j < min_phen_diag_jaccard_when_no_issue:
                    continue
                if issue_j <= 1e-12 and min_report_jaccard_when_no_issue > 0:
                    rep_j = jaccard(rep_union, sg2.report_ids)
                    if rep_j < min_report_jaccard_when_no_issue:
                        continue
                if issue_j > 1e-12 and issue_j < min_issue_jaccard and phen_diag_j < 0.35:
                    continue

                sim = weighted_jaccard(sig_union, sig2, weights)
                if sim < thresh:
                    continue

                if max_nodes_after_merge is not None:
                    if len(node_union | set(sg2.node_ids)) > max_nodes_after_merge:
                        continue

                used[j] = True
                rep_union |= set(sg2.report_ids)
                node_union |= set(sg2.node_ids)
                edge_union |= set(sg2.edge_ids)
                covered_union |= set(sg2.covered_universe)
                seeds.append(sg2.seed_node)

                for k in sig_union.keys():
                    sig_union[k] |= sig2.get(k, set())

                changed = True

        used[i] = True
        new_id = sg.subgraph_id if len(seeds) == 1 else f"{sg.subgraph_id}_m{len(seeds)}"
        cost = compute_cost(rep_union, node_union, edge_union, alpha=alpha, beta=beta, gamma=gamma)
        merged.append(
            Subgraph(
                subgraph_id=new_id,
                seed_node=seeds[0],
                report_ids=rep_union,
                node_ids=node_union,
                edge_ids=edge_union,
                covered_universe=covered_union,
                cost=cost,
            )
        )

    return merged
