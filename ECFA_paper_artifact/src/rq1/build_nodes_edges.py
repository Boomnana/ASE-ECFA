#!/usr/bin/env python3


from __future__ import annotations
import argparse, hashlib
from pathlib import Path
from typing import Dict, Tuple, List, Set
import pandas as pd

def stable_app_code_4digits(app_name: str) -> int:
    h = hashlib.md5(app_name.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return 1000 + (x % 9000)

def parse_row_index_to_set(x) -> Set[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return set()
    s = str(x).strip()
    if not s or s.lower() in {"nan","none","null"}:
        return set()
    if "|" in s:
        return {p.strip() for p in s.split("|") if p.strip()}
    return {s}

def merge_row_index(existing: str, new_ids: Set[str]) -> str:
    cur = set()
    if existing:
        cur |= parse_row_index_to_set(existing)
    cur |= set(new_ids)

    def keyfn(v: str):
        return (0, int(v)) if v.isdigit() else (1, v)
    return "|".join(sorted(cur, key=keyfn))

def load_triples(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    req = ["subject","subject_type","relation","object","object_type","source_row_index"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"[triples.xlsx] missing columns {miss}. Found: {list(df.columns)}")
    for c in req:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["subject"]!="") & (df["object"]!="") & (df["relation"]!="")]
    return df

def load_aligned_clusters(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_excel(path, dtype=str)
    if "cluster" not in df.columns or "entity" not in df.columns:
        raise ValueError(f"[aligned_clusters.xlsx] must have columns (cluster, entity). Found: {list(df.columns)}")
    df["cluster"] = df["cluster"].astype(str).str.strip()
    df["entity"]  = df["entity"].astype(str).str.strip()
    df = df[(df["cluster"]!="") & (df["entity"]!="") & (df["cluster"].str.lower()!="nan") & (df["entity"].str.lower()!="nan")]
    m = {}
    for _, r in df.iterrows():
        m[r["entity"]] = r["cluster"]
        m.setdefault(r["cluster"], r["cluster"])
    return m

def write_nodes_edges(out_dir: Path, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_df.to_excel(out_dir / "nodes.xlsx", index=False)
    edges_df.to_excel(out_dir / "edges.xlsx", index=False)

def build_consistent(nodes_raw_keys: List[Tuple[str,str]], app_code: int) -> Dict[Tuple[str,str], int]:


    keys = sorted(set(nodes_raw_keys), key=lambda x: (x[1], x[0]))
    mapping = {}
    for i, k in enumerate(keys):
        mapping[k] = app_code*1_000_000 + i
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    for app_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        triples_path = app_dir / "triples.xlsx"
        if not triples_path.exists():
            continue

        app_name = app_dir.name
        app_code = stable_app_code_4digits(app_name)
        triples = load_triples(triples_path)
        amap = load_aligned_clusters(app_dir / "aligned_clusters.xlsx")

        out_no = app_dir / "no_unify"
        out_u  = app_dir / "unify"
        if not args.overwrite:
            if (out_no/"nodes.xlsx").exists() and (out_no/"edges.xlsx").exists() and (out_u/"nodes.xlsx").exists() and (out_u/"edges.xlsx").exists():
                print(f"[SKIP] {app_name}: outputs exist. Use --overwrite.")
                continue


        raw_keys = []
        for r in triples.itertuples(index=False):
            raw_keys.append((r.subject, r.subject_type))
            raw_keys.append((r.object,  r.object_type))

        raw_id_map = build_consistent(raw_keys, app_code)


        canon_to_minid: Dict[Tuple[str,str], int] = {}
        def to_cluster(name: str) -> str:
            s = str(name).strip()
            return amap.get(s, s)

        for (raw_name, raw_type), rid in raw_id_map.items():
            ck = (to_cluster(raw_name), raw_type)
            canon_to_minid[ck] = min(canon_to_minid.get(ck, rid), rid)


        node_rows_no: Dict[int, Dict[str,str]] = {}
        edge_rows_no: List[Dict[str,str]] = []

        def add_node_no(name, typ, src_row):
            nid = raw_id_map[(name, typ)]
            if nid not in node_rows_no:
                node_rows_no[nid] = {"id": str(nid), "name": str(name), "type": str(typ), "source_row_index": ""}
            node_rows_no[nid]["source_row_index"] = merge_row_index(node_rows_no[nid]["source_row_index"], parse_row_index_to_set(src_row))
            return nid

        for eid, r in enumerate(triples.itertuples(index=False), start=0):
            sid = add_node_no(r.subject, r.subject_type, r.source_row_index)
            tid = add_node_no(r.object,  r.object_type,  r.source_row_index)
            edge_rows_no.append({
                "id": str(eid),
                "source_id": str(sid),
                "target_id": str(tid),
                "relation": str(r.relation),
                "source_type": str(r.subject_type),
                "target_type": str(r.object_type),
                "source_row_index": str(r.source_row_index),
            })

        nodes_no = pd.DataFrame(list(node_rows_no.values()), columns=["id","name","type","source_row_index"]).sort_values("id")
        edges_no = pd.DataFrame(edge_rows_no, columns=["id","source_id","target_id","relation","source_type","target_type","source_row_index"])

        write_nodes_edges(out_no, nodes_no.reset_index(drop=True), edges_no)


        node_rows_u: Dict[int, Dict[str,str]] = {}
        edge_rows_u: List[Dict[str,str]] = []

        def add_node_u(raw_name, typ, src_row):
            cname = to_cluster(raw_name)
            nid = canon_to_minid[(cname, typ)]
            if nid not in node_rows_u:
                node_rows_u[nid] = {"id": str(nid), "name": str(cname), "type": str(typ), "source_row_index": ""}
            node_rows_u[nid]["source_row_index"] = merge_row_index(node_rows_u[nid]["source_row_index"], parse_row_index_to_set(src_row))
            return nid

        for eid, r in enumerate(triples.itertuples(index=False), start=0):
            sid = add_node_u(r.subject, r.subject_type, r.source_row_index)
            tid = add_node_u(r.object,  r.object_type,  r.source_row_index)
            edge_rows_u.append({
                "id": str(eid),
                "source_id": str(sid),
                "target_id": str(tid),
                "relation": str(r.relation),
                "source_type": str(r.subject_type),
                "target_type": str(r.object_type),
                "source_row_index": str(r.source_row_index),
            })

        nodes_u = pd.DataFrame(list(node_rows_u.values()), columns=["id","name","type","source_row_index"]).sort_values("id")
        edges_u = pd.DataFrame(edge_rows_u, columns=["id","source_id","target_id","relation","source_type","target_type","source_row_index"])

        write_nodes_edges(out_u, nodes_u.reset_index(drop=True), edges_u)

        print(f"[OK] {app_name} app_code={app_code}: wrote consistent ids for no_unify & unify.")

    print("[DONE]")

if __name__ == "__main__":
    main()
