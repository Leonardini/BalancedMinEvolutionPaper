"""Section 8.4: root LP of the lifted cut-value-ladder model.

    BME_THREADS=1 python lifted_ladder.py INSTANCE SPLITS OUT.json \
        [--rung] [--cliques prop4|extended] [--manifold]

w_ij >= 2^-(n-1) with Kraft. For every nontrivial bipartition S an indicator
y_S in [0, 1] is tied to the cut value W[S] = sum_{i in S, j not in S} w_ij, with
sum_S y_S = n - 3 and U = (n - 1) / 4:
    two rungs (default)   W[S] >= 3/4 - y_S / 4,   W[S] <= U - (U - 1/2) y_S
    --rung                a second indicator t_S for the 3/4 rung, sum_S t_S = 2(n - 3),
                          y_S + t_S <= 1, W[S] >= 7/8 - 3/8 y_S - 1/8 t_S,
                          W[S] <= U - (U - 1/2) y_S - (U - 3/4) t_S
    --cliques prop4       sum_{S in Q} y_S <= 1 for the cliques Q^{j,k}_{x,y,z} of
                          Proposition 4 (j + k <= n)
    --cliques extended    the same families for every j, k, keeping those with
                          j + k > n only when they are pairwise crossing
    --manifold            manifold tangent cuts until no cut is violated or the bound
                          stalls (lib/membership.py, solve_with_cuts)
The record also has the min-cut relaxation (Kraft, w >= 2^-(n-1), all W[S] >= 1/2).
Both bounds also have their safe values and the optimum check at the SPLITS tree
(lib/membership.py, solve_with_cuts), at which y_S = 1 on the tree's splits and t_S = 1
on the bipartitions with W_T[S] = 3/4.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import membership  # noqa: E402
from lib.common import (gurobi_model, log, optimal_tree, optimum, parse_matrix,  # noqa: E402
                        peak_rss_mb, write_json)


def compatible(a, b):
    """Bipartitions given by their sides a, b not containing leaf 0."""
    x = a & b
    return x == a or x == b or x == 0


def clique_cover(n, splits, extended):
    """Index lists of the families Q^{j,k}_{x,y,z}, without duplicates."""
    full = (1 << n) - 1
    out, seen = [], set()
    for x in range(n):
        A = [m if (m >> x) & 1 else full & ~m for m in splits]
        size = [bin(a).count('1') for a in A]
        for y, z in itertools.combinations([v for v in range(n) if v != x], 2):
            h1, h2 = {}, {}
            for s, a in enumerate(A):
                yin, zin = (a >> y) & 1, (a >> z) & 1
                if yin and not zin:
                    h1.setdefault(size[s], []).append(s)
                elif zin and not yin:
                    h2.setdefault(size[s], []).append(s)
            for j, a1 in h1.items():
                for k, a2 in h2.items():
                    clq = a1 + a2
                    if len(clq) < 2:
                        continue
                    if j + k > n and not (extended and all(
                            not compatible(splits[p], splits[q])
                            for p, q in itertools.combinations(clq, 2))):
                        continue
                    key = tuple(sorted(clq))
                    if key not in seen:
                        seen.add(key)
                        out.append(clq)
    return out


def base_model(D, pairs):
    import gurobipy as gp
    from gurobipy import GRB
    n = len(D)
    m = gurobi_model()
    w = {p: m.addVar(lb=2.0 ** -(n - 1)) for p in pairs}
    m.setObjective(gp.quicksum(D[p] * w[p] for p in pairs), GRB.MINIMIZE)
    for i in range(n):
        m.addConstr(gp.quicksum(w[p] for p in pairs if i in p) == 0.5, name="kraft")
    m.update()
    # w has no upper bound of its own; Kraft (sum_j w_ij = 1/2) with w >= 0 gives
    # w_ij <= 1/2 at every feasible point, an implied bound for the safe bound.
    m._implied_hi = {v: 0.5 for v in w.values()}
    return m, w


def tree_values(tree, pairs, splits, cutp):
    """w_T, and per bipartition (mask) whether it is a split of T and whether
    W_T[S] = 3/4, exactly (dyadic sums)."""
    tw = {p: 2.0 ** -int(tree["tau"][p]) for p in pairs}
    masks = {sum(1 << i for i in S) for S in tree["splits"]}
    is_split = [m in masks for m in splits]
    W = [sum(tw[p] for p in cp) for cp in cutp]
    return tw, is_split, [x == 0.75 for x in W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("splits")
    ap.add_argument("out")
    ap.add_argument("--rung", action="store_true")
    ap.add_argument("--cliques", choices=["prop4", "extended"])
    ap.add_argument("--manifold", action="store_true")
    a = ap.parse_args()
    import gurobipy as gp

    D = parse_matrix(a.instance)
    n = D.shape[0]
    L = optimum(D, a.splits)
    tree = optimal_tree(D, splits_path=a.splits)
    pairs = list(itertools.combinations(range(n), 2))
    splits = [m for m in range(1, 1 << n) if not m & 1 and 2 <= bin(m).count('1') <= n - 2]
    cutp = [[p for p in pairs if ((m >> p[0]) & 1) != ((m >> p[1]) & 1)] for m in splits]
    rec = dict(instance=a.instance, n=n, optimum=L, rung=a.rung, cliques=a.cliques,
               manifold=a.manifold, bipartitions=len(splits))

    tw, is_split, is_34 = tree_values(tree, pairs, splits, cutp)
    m1, w1 = base_model(D, pairs)
    for cp in cutp:
        m1.addConstr(gp.quicksum(w1[p] for p in cp) >= 0.5, name="mincut")
    rec["mincut"] = membership.solve_with_cuts(m1, w1, n, False, label="min-cut", tree=tree,
                                               point={w1[p]: tw[p] for p in pairs})
    rec["mincut"]["gap"] = (L - rec["mincut"]["bound"]) / L
    write_json(a.out, rec)

    m, w = base_model(D, pairs)
    ns = len(splits)
    U = (n - 1) / 4.0
    y = m.addVars(ns, lb=0, ub=1)
    m.addConstr(gp.quicksum(y[s] for s in range(ns)) == n - 3, name="ysum")
    point = {w[p]: tw[p] for p in pairs}
    m.update()
    point.update({y[s]: float(is_split[s]) for s in range(ns)})
    if a.rung:
        t = m.addVars(ns, lb=0, ub=1)
        m.addConstr(gp.quicksum(t[s] for s in range(ns)) == 2 * (n - 3), name="tsum")
        for s in range(ns):
            Ws = gp.quicksum(w[p] for p in cutp[s])
            m.addConstr(y[s] + t[s] <= 1, name="y_plus_t")
            m.addConstr(Ws >= 0.875 - 0.375 * y[s] - 0.125 * t[s], name="rung_lo")
            m.addConstr(Ws <= U - (U - 0.5) * y[s] - (U - 0.75) * t[s], name="rung_hi")
        m.update()
        point.update({t[s]: float(is_34[s]) for s in range(ns)})
    else:
        for s in range(ns):
            Ws = gp.quicksum(w[p] for p in cutp[s])
            m.addConstr(Ws >= 0.75 - 0.25 * y[s], name="rung_lo")
            m.addConstr(Ws <= U - (U - 0.5) * y[s], name="rung_hi")
    if a.cliques:
        cover = clique_cover(n, splits, a.cliques == "extended")
        log(f"{len(cover)} cliques, {sum(map(len, cover))} nonzeros")
        rows = np.repeat(np.arange(len(cover)), [len(clq) for clq in cover])
        cols = np.concatenate([np.asarray(clq) for clq in cover])
        A = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(cover), ns))
        m.update()
        m.addMConstr(A, [y[s] for s in range(ns)], '<', np.ones(len(cover)), name="clique")
        rec["n_cliques"] = len(cover)
    rec["ladder"] = membership.solve_with_cuts(m, w, n, a.manifold, label="ladder", tree=tree,
                                               point=point)
    rec["ladder"]["gap"] = (L - rec["ladder"]["bound"]) / L
    rec["ladder"]["rows"] = m.NumConstrs
    rec["peak_rss_mb"] = peak_rss_mb()
    write_json(a.out, rec)
    log(f"min-cut gap {100 * rec['mincut']['gap']:.3f}%  ladder gap {100 * rec['ladder']['gap']:.3f}%")


if __name__ == "__main__":
    main()
