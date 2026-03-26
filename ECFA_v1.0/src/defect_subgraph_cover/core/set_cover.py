from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from .subgraph import Subgraph


@dataclass
class CoverResult:
    selected: List[Subgraph]
    covered: Set[str]
    uncovered: Set[str]


def greedy_set_cover(
    subgraphs: List[Subgraph],
    universe: Set[str],
    budget: int = 50,
    min_gain: int = 1,
) -> CoverResult:
    U = set(universe)
    covered: Set[str] = set()
    selected: List[Subgraph] = []
    remaining = subgraphs[:]

    while len(selected) < budget:
        best = None
        best_score = -1.0
        for sg in remaining:
            gain_set = sg.covered_universe - covered
            gain = len(gain_set)
            if gain < min_gain:
                continue
            score = gain / max(sg.cost, 1e-9)
            if score > best_score:
                best = sg
                best_score = score
        if best is None:
            break
        selected.append(best)
        covered |= best.covered_universe
        remaining = [sg for sg in remaining if sg.subgraph_id != best.subgraph_id]
        if covered == U:
            break

    uncovered = U - covered
    return CoverResult(selected=selected, covered=covered, uncovered=uncovered)
