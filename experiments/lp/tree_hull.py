"""Section 8.4: where the root gap of the lifted ladder model lies, on the synthetic
instances n = 4..7. Compatibility constraints sum_{S in K} y_S <= 1 over: none, the cover of
Proposition 4, and every maximal clique of the crossing graph; and, as the strongest possible
  compatibility description, the convex hull of the trees: (y, t) = sum_T lambda_T (y^T, t^T)
  over all (2n-5)!! trees, lambda >= 0, sum lambda = 1; with the rung, also the hull of y alone.
Each with and without the 3/4-rung indicator, always with the manifold imposed exactly.

    BME_THREADS=1 python tree_hull.py OUT.json

Progress to stderr.
"""
import itertools
import sys
from pathlib import Path

import networkx as nx
import numpy as np

PUB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PUB / "experiments/lp"))
import lifted_ladder as LL  # noqa: E402
from lib import membership  # noqa: E402
from lib.common import log, optimal_tree, optimum, parse_matrix, write_json  # noqa: E402
sys.path.insert(0, str(PUB))
from compact.solver.nni_incumbent import _gen_all_trees  # noqa: E402

import gurobipy as gp  # noqa: E402

def all_maximal(n, splits):
    def cross(a, b):
        full = (1 << n) - 1
        return all(x and x != 0 for x in (a & b, a & ~b & full, ~a & b & full, ~a & ~b & full))
    G = nx.Graph()
    G.add_nodes_from(range(len(splits)))
    G.add_edges_from((s, t) for s, t in itertools.combinations(range(len(splits)), 2)
                     if cross(splits[s], splits[t]))
    return [c for c in nx.find_cliques(G) if len(c) >= 2]


def check_edge_cover(n, splits, cover):
    full = (1 << n) - 1
    covered = {tuple(sorted(e)) for c in cover for e in itertools.combinations(c, 2)}
    for s, t in itertools.combinations(range(len(splits)), 2):
        a, b = splits[s], splits[t]
        if all((a & b, a & ~b & full, ~a & b & full, ~a & ~b & full)):
            assert (s, t) in covered, f"n={n}: crossing pair {s},{t} not covered"


def tree_vectors(n, splits, cutp):
    """(y^T, t^T) for every tree: y^T_S = [S is a split], t^T_S = [W_T[S] = 3/4]."""
    pairs = list(itertools.combinations(range(n), 2))
    idx = {m: s for s, m in enumerate(splits)}
    out = []
    for edges in _gen_all_trees(n):
        G = nx.Graph(edges)
        y = [0] * len(splits)
        for u, v in edges:
            if u < 0 and v < 0:
                H = G.copy()
                H.remove_edge(u, v)
                side = {x - 1 for x in nx.node_connected_component(H, u) if x > 0}
                if 0 in side:
                    side = set(range(n)) - side
                y[idx[sum(1 << i for i in side)]] = 1
        d = dict(nx.all_pairs_shortest_path_length(G))
        tw = {p: 2.0 ** -d[p[0] + 1][p[1] + 1] for p in pairs}
        t = [int(sum(tw[p] for p in cp) == 0.75) for cp in cutp]
        out.append((y, t))
    expect = int(np.prod(range(1, 2 * n - 4, 2)))
    assert len(out) == expect and len({tuple(y) for y, _ in out}) == expect, (n, len(out))
    return out


def solve(D, n, splits, cutp, tree, L, rung, cover, hull=None, joint=True):
    pairs = list(itertools.combinations(range(n), 2))
    tw, is_split, is_34 = LL.tree_values(tree, pairs, splits, cutp)
    m, w = LL.base_model(D, pairs)
    ns = len(splits)
    U = (n - 1) / 4.0
    y = m.addVars(ns, lb=0, ub=1)
    m.addConstr(gp.quicksum(y.values()) == n - 3)
    point = {w[p]: tw[p] for p in pairs}
    m.update()
    point.update({y[s]: float(is_split[s]) for s in range(ns)})
    if rung:
        t = m.addVars(ns, lb=0, ub=1)
        m.addConstr(gp.quicksum(t.values()) == 2 * (n - 3))
        for s in range(ns):
            Ws = gp.quicksum(w[p] for p in cutp[s])
            m.addConstr(y[s] + t[s] <= 1)
            m.addConstr(Ws >= 0.875 - 0.375 * y[s] - 0.125 * t[s])
            m.addConstr(Ws <= U - (U - 0.5) * y[s] - (U - 0.75) * t[s])
        m.update()
        point.update({t[s]: float(is_34[s]) for s in range(ns)})
    else:
        for s in range(ns):
            Ws = gp.quicksum(w[p] for p in cutp[s])
            m.addConstr(Ws >= 0.75 - 0.25 * y[s])
            m.addConstr(Ws <= U - (U - 0.5) * y[s])
    for clq in cover:
        m.addConstr(gp.quicksum(y[s] for s in clq) <= 1)
    if hull is not None:
        lam = m.addVars(len(hull), lb=0, ub=1)
        m.addConstr(lam.sum() == 1)
        for s in range(ns):
            m.addConstr(y[s] == gp.quicksum(lam[k] for k, (yv, _) in enumerate(hull) if yv[s]))
            if rung and joint:
                m.addConstr(t[s] == gp.quicksum(lam[k] for k, (_, tv) in enumerate(hull) if tv[s]))
        m.update()
        opt = [k for k, (yv, _) in enumerate(hull) if all(yv[s] == is_split[s] for s in range(ns))]
        assert len(opt) == 1
        point.update({lam[k]: float(k == opt[0]) for k in range(len(hull))})
    r = membership.solve_with_cuts(m, w, n, True, label="ladder", tree=tree, point=point)
    r["gap"] = (L - r["bound"]) / L
    return r


def main():
    out = Path(sys.argv[1])
    res = []
    for n in range(4, 8):
        inst = PUB / f"data/synthetic/test_n{n}.txt"
        D = parse_matrix(inst)
        L = optimum(D)
        tree = optimal_tree(D, instance=inst)
        splits = [m for m in range(1, 1 << n) if not m & 1 and 2 <= bin(m).count('1') <= n - 2]
        pairs = list(itertools.combinations(range(n), 2))
        cutp = [[p for p in pairs if ((m >> p[0]) & 1) != ((m >> p[1]) & 1)] for m in splits]
        covers = {"none": [], "prop4": LL.clique_cover(n, splits),
                  "all_maximal": all_maximal(n, splits)}
        hull = tree_vectors(n, splits, cutp)
        for k in ("prop4", "all_maximal"):
            check_edge_cover(n, splits, covers[k])
        for rung in (False, True):
            for k, cov in covers.items():
                r = solve(D, n, splits, cutp, tree, L, rung, cov)
                res.append(dict(n=n, rung=rung, cliques=k, n_cliques=len(cov), **r))
                log(f"n={n} rung={rung} cliques={k} ({len(cov)}): root gap {100 * r['gap']:.4f}%")
                write_json(out, res)
            r = solve(D, n, splits, cutp, tree, L, rung, [], hull=hull)
            res.append(dict(n=n, rung=rung, cliques="tree_hull", n_cliques=0, **r))
            log(f"n={n} rung={rung} cliques=tree_hull ({len(hull)} trees): root gap {100 * r['gap']:.4f}%")
            write_json(out, res)
            if rung:
                r = solve(D, n, splits, cutp, tree, L, rung, [], hull=hull, joint=False)
                res.append(dict(n=n, rung=rung, cliques="tree_hull_y_only", n_cliques=0, **r))
                log(f"n={n} rung={rung} cliques=tree_hull_y_only ({len(hull)} trees): root gap {100 * r['gap']:.4f}%")
                write_json(out, res)


if __name__ == "__main__":
    main()
