

from typing import Dict, List, Tuple
from entity_alignment.common import normalize_text

def closure_and_clusters(pairs: List[Tuple[str, str]], all_entities: List[str] = None) -> Dict[str, List[str]]:


    parent: Dict[str, str] = {}


    if all_entities:
        for e in all_entities:
            norm = normalize_text(e)
            parent[norm] = norm

    def find(x: str) -> str:

        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:

        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra


    for a, b in pairs:
        union(normalize_text(a), normalize_text(b))


    groups: Dict[str, List[str]] = {}
    for x in list(parent.keys()):
        rx = find(x)
        groups.setdefault(rx, []).append(x)
    for k in groups:

        groups[k] = sorted(list(set(groups[k])), key=lambda s: normalize_text(s).lower())
    return groups
