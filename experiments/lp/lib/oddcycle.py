"""Exact separation of the odd-cycle inequalities of the cut polytope of K_n
(Barahona and Mahjoub), applied to one cut-vector column b_ij in [0, 1].

For a cycle C and an odd subset F of its edges the inequality is
    sum_{e in F} b_e - sum_{e in C \\ F} b_e <= |F| - 1,
equivalently sum_{e in F} (1 - b_e) + sum_{e in C \\ F} b_e >= 1. In the doubled graph
with vertices (v, 0), (v, 1), an edge ij of weight b_ij inside each layer and of weight
1 - b_ij across the layers, a (v,0)-(v,1) path is a closed walk through v with an odd
number of F-edges, of length equal to the left-hand side above. So a violated
inequality exists iff some shortest (v,0)-(v,1) path is shorter than 1 - tol.
A shortest path that visits both copies of some vertex u contains a shorter closed odd
walk through u; taking the repeated vertex of smallest span gives a simple cycle, which
is no longer, so the inequality returned is always on a simple cycle.
"""
import itertools

import networkx as nx


def _simple_cycle(path):
    """The closed sub-walk of `path` between the two copies of the vertex whose
    copies are closest together; it contains no other repeated vertex."""
    best = None
    first = {}
    for pos, (v, _layer) in enumerate(path):
        if v in first and (best is None or pos - first[v] < best[1] - best[0]):
            best = (first[v], pos)
        first[v] = pos
    a, b = best
    return path[a:b + 1]


def separate(n, b, tol):
    """Violated odd-cycle inequalities of the column b ((i, j) -> value, i < j).
    Returns a list of (coef, rhs, violation) with coef mapping (i, j) to +1 / -1."""
    H = nx.Graph()
    for (i, j) in itertools.combinations(range(n), 2):
        v = min(1.0, max(0.0, b[(i, j)]))
        H.add_edge((i, 0), (j, 0), weight=v)
        H.add_edge((i, 1), (j, 1), weight=v)
        H.add_edge((i, 0), (j, 1), weight=1.0 - v)
        H.add_edge((i, 1), (j, 0), weight=1.0 - v)
    out, seen = [], set()
    for v in range(n):
        length, path = nx.single_source_dijkstra(H, (v, 0), target=(v, 1), weight="weight")
        if length >= 1.0 - tol:
            continue
        cyc = _simple_cycle(path)
        coef, nF = {}, 0
        for (a, la), (c, lc) in zip(cyc[:-1], cyc[1:]):
            e = (min(a, c), max(a, c))
            cross = la != lc
            coef[e] = 1 if cross else -1
            nF += cross
        key = frozenset(coef.items())
        if key in seen:
            continue
        seen.add(key)
        lhs = sum(cf * b[e] for e, cf in coef.items())
        out.append((coef, nF - 1, lhs - (nF - 1)))
    return out
