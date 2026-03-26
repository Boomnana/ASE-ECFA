from __future__ import annotations

from collections import defaultdict, deque
from typing import List, Tuple

from ..core.index import GraphIndex
from ..core.subgraph import Subgraph


def stable_id_key(x: str) -> Tuple[int, int, str]:
    s = str(x)
    if s.isdigit():
        return (0, int(s), "")
    return (1, 0, s)


def _node_label(idx: GraphIndex, nid: str) -> str:
    r = idx.node_id_to_row.get(str(nid), {}) or {}
    name = (r.get("name") or "").strip()
    return name if name else str(nid)


def _node_type(idx: GraphIndex, nid: str) -> str:
    r = idx.node_id_to_row.get(str(nid), {}) or {}
    return str(r.get("type", "") or "")


def _edge_relation(idx: GraphIndex, eid: str) -> str:
    r = idx.edge_id_to_row.get(str(eid), {}) or {}
    return str(r.get("relation", "") or "")


def _edge_uv(idx: GraphIndex, eid: str) -> Tuple[str, str]:
    r = idx.edge_id_to_row.get(str(eid), {}) or {}
    return str(r.get("source_id", "") or ""), str(r.get("target_id", "") or "")


def _build_subgraph_adjacency(idx: GraphIndex, sg: Subgraph):
    node_set = set(map(str, sg.node_ids))
    adj = defaultdict(list)
    radj = defaultdict(list)

    for eid in sorted(map(str, sg.edge_ids), key=stable_id_key):
        u, v = _edge_uv(idx, eid)
        if not u or not v:
            continue
        if u not in node_set or v not in node_set:
            continue
        adj[u].append((v, eid))
        radj[v].append((u, eid))

    for u in list(adj.keys()):
        adj[u].sort(key=lambda t: (stable_id_key(t[0]), stable_id_key(t[1])))
    for v in list(radj.keys()):
        radj[v].sort(key=lambda t: (stable_id_key(t[0]), stable_id_key(t[1])))

    return adj, radj


def _pick_start_nodes(idx: GraphIndex, sg: Subgraph, *, only_core: bool = True) -> List[str]:
    core_types = {"问题陈述", "用户感知现象", "系统诊断信息", "用户操作"}
    nodes = sorted(list(map(str, sg.node_ids)), key=stable_id_key)
    if not only_core:
        return nodes
    core = [nid for nid in nodes if _node_type(idx, nid) in core_types]
    return core if core else nodes


def _path_score(idx: GraphIndex, node_path: List[str], eid_path: List[str]) -> Tuple[int, int, int]:
    preferred_types = {"问题陈述", "用户感知现象", "系统诊断信息", "用户操作"}
    pref_nodes = sum(1 for nid in node_path if _node_type(idx, nid) in preferred_types)
    return (len(eid_path), pref_nodes, len(node_path))


def _is_subsegment_nodes(small_nodes: List[str], big_nodes: List[str]) -> bool:
    n = len(small_nodes)
    m = len(big_nodes)
    if n > m:
        return False
    for i in range(m - n + 1):
        if big_nodes[i : i + n] == small_nodes:
            return True
    return False


def _canonical_path_key(node_path: List[str], eid_path: List[str]) -> Tuple:
    return (
        tuple(map(str, node_path)),
        tuple(map(str, eid_path)),
    )


def _extract_top_paths(
    idx: GraphIndex,
    sg: Subgraph,
    *,
    max_len: int = 4,
    top_k: int = 30,
    per_start_cap: int = 80,
    only_core_starts: bool = True,
    require_core_in_path: bool = True,
) -> List[List[Tuple[str, str, str]]]:
    adj, _ = _build_subgraph_adjacency(idx, sg)
    starts = _pick_start_nodes(idx, sg, only_core=only_core_starts)

    core_types = {"问题陈述", "用户感知现象", "系统诊断信息", "用户操作"}

    candidates: List[Tuple[List[str], List[str]]] = []

    for s in starts:
        q = deque()
        q.append((s, [s], []))
        expanded = 0

        while q and expanded < per_start_cap:
            u, node_path, eid_path = q.popleft()
            expanded += 1

            if eid_path:
                if require_core_in_path:
                    if any(_node_type(idx, nid) in core_types for nid in node_path):
                        candidates.append((node_path, eid_path))
                else:
                    candidates.append((node_path, eid_path))

            if len(eid_path) >= max_len:
                continue

            for v, eid in adj.get(u, []):
                if v in node_path:
                    continue
                q.append((v, node_path + [v], eid_path + [eid]))

    candidates.sort(
        key=lambda x: (
            _path_score(idx, x[0], x[1]),
            _canonical_path_key(x[0], x[1]),
        ),
        reverse=True,
    )

    kept: List[Tuple[List[str], List[str]]] = []
    for np, ep in candidates:
        redundant = False
        for knp, _ in kept:
            if _is_subsegment_nodes(np, knp):
                redundant = True
                break
        if not redundant:
            kept.append((np, ep))
            if len(kept) >= top_k:
                break

    out: List[List[Tuple[str, str, str]]] = []
    for np, ep in kept:
        triples: List[Tuple[str, str, str]] = []
        for i, eid in enumerate(ep):
            u = np[i]
            v = np[i + 1]
            rel = _edge_relation(idx, eid) or ""
            triples.append((u, rel, v))
        if triples:
            out.append(triples)

    return out


def _group_star_paths(idx: GraphIndex, paths_triples: List[List[Tuple[str, str, str]]]):
    len1, len_gt1 = [], []
    for t in paths_triples:
        if t:
            (len1 if len(t) == 1 else len_gt1).append(t)

    star_map = defaultdict(set)
    for t in len1:
        u, rel, v = t[0]
        star_map[(u, rel)].add(v)

    star_groups = []
    singles = []
    for (u, rel), vs in star_map.items():
        vs = sorted(list(vs), key=stable_id_key)
        if len(vs) > 1:
            star_groups.append((u, rel, vs))
        else:
            singles.append((u, rel, vs[0]))

    rev_map = defaultdict(set)
    for u, rel, v in singles:
        rev_map[(v, rel)].add(u)

    reverse_star_groups = []
    for (v, rel), us in rev_map.items():
        us = sorted(list(us), key=stable_id_key)
        if len(us) > 1:
            reverse_star_groups.append((us, rel, v))
        else:
            star_groups.append((us[0], rel, [v]))

    chain_map = defaultdict(set)
    for chain in len_gt1:
        prefix = tuple(chain[:-1])
        last_u, last_rel, last_v = chain[-1]
        key = (prefix, last_u, last_rel)
        chain_map[key].add(last_v)

    chain_star_groups = []
    simple_chains = []

    for (prefix, last_u, last_rel), vs in chain_map.items():
        vs = sorted(list(vs), key=stable_id_key)
        if len(vs) > 1:
            chain_star_groups.append((list(prefix), last_u, last_rel, vs))
        else:
            full = list(prefix) + [(last_u, last_rel, vs[0])]
            simple_chains.append(full)

    star_groups.sort(key=lambda x: (len(x[2]), stable_id_key(x[0])), reverse=True)
    reverse_star_groups.sort(key=lambda x: (len(x[0]), stable_id_key(x[2])), reverse=True)
    chain_star_groups.sort(key=lambda x: len(x[3]), reverse=True)
    simple_chains.sort(
        key=lambda ch: (
            _path_score(idx, [ch[0][0]] + [e[2] for e in ch], [e[1] for e in ch]),
            _canonical_path_key([ch[0][0]] + [e[2] for e in ch], [e[1] for e in ch]),
        ),
        reverse=True,
    )

    return star_groups, reverse_star_groups, chain_star_groups, simple_chains


def _format_star(idx: GraphIndex, u: str, rel: str, vs: List[str]) -> str:
    src = _node_label(idx, u)
    targets = ", ".join([_node_label(idx, v) for v in vs])
    return f"{src} --({rel})--> [{targets}]"


def _format_reverse_star(idx: GraphIndex, us: List[str], rel: str, v: str) -> str:
    srcs = ", ".join([_node_label(idx, u) for u in us])
    target = _node_label(idx, v)
    return f"[{srcs}] --({rel})--> {target}"


def _format_chain(idx: GraphIndex, triples: List[Tuple[str, str, str]]) -> str:
    if not triples:
        return ""
    u0, rel0, v0 = triples[0]
    s = f"{_node_label(idx, u0)} --({rel0})--> {_node_label(idx, v0)}"
    for (_, rel, v) in triples[1:]:
        s += f" --({rel})--> {_node_label(idx, v)}"
    return s


def _format_chain_star(
    idx: GraphIndex, prefix: List[Tuple[str, str, str]], last_u: str, last_rel: str, vs: List[str]
) -> str:
    base = _format_chain(idx, prefix) if prefix else _node_label(idx, last_u)
    targets = ", ".join([_node_label(idx, v) for v in vs])
    return f"{base} --({last_rel})--> [{targets}]"


def build_paths_descriptions(
    idx: GraphIndex,
    sg: Subgraph,
    *,
    max_len: int = 4,
    top_k: int = 30,
    max_output: int = 20,
    per_start_cap: int = 80,
    only_core_starts: bool = True,
) -> List[str]:
    raw = _extract_top_paths(
        idx,
        sg,
        max_len=max_len,
        top_k=top_k,
        per_start_cap=per_start_cap,
        only_core_starts=only_core_starts,
        require_core_in_path=True,
    )

    star_groups, reverse_star_groups, chain_star_groups, simple_chains = _group_star_paths(idx, raw)

    out: List[str] = []
    seen = set()

    for u, rel, vs in star_groups:
        if len(out) >= max_output:
            break
        desc = _format_star(idx, u, rel, vs)
        if desc not in seen:
            out.append(desc)
            seen.add(desc)

    for us, rel, v in reverse_star_groups:
        if len(out) >= max_output:
            break
        desc = _format_reverse_star(idx, us, rel, v)
        if desc not in seen:
            out.append(desc)
            seen.add(desc)

    for prefix, last_u, last_rel, vs in chain_star_groups:
        if len(out) >= max_output:
            break
        desc = _format_chain_star(idx, prefix, last_u, last_rel, vs)
        if desc not in seen:
            out.append(desc)
            seen.add(desc)

    for triples in simple_chains:
        if len(out) >= max_output:
            break
        desc = _format_chain(idx, triples)
        if desc not in seen:
            out.append(desc)
            seen.add(desc)

    if not out:
        src_rel_to_vs = defaultdict(list)
        for eid in sorted(map(str, sg.edge_ids), key=stable_id_key):
            u, v = _edge_uv(idx, eid)
            if not u or not v:
                continue
            rel = _edge_relation(idx, eid)
            src_rel_to_vs[(u, rel)].append(v)

        items = []
        for (u, rel), vs in src_rel_to_vs.items():
            dedup = []
            seen_v = set()
            for v in sorted(map(str, vs), key=stable_id_key):
                if v not in seen_v:
                    seen_v.add(v)
                    dedup.append(v)
            items.append((u, rel, dedup))

        items.sort(key=lambda x: (len(x[2]), stable_id_key(x[0]), x[1]), reverse=True)
        for u, rel, vs in items[:max_output]:
            out.append(_format_star(idx, u, rel, vs))

    return out
