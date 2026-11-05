""""Spurious points with the splits held at 1/2" and appendix "The splitohedron": log-integral points of the splitohedron Sp(n), and of its
faces Q_T for one tree T of each shape.

    BME_THREADS=1 python lattice.py N OUTDIR [--shapes] [--floor] [--no-dc] [--rung]

A point is log-integral when every w_ij is a power of two, 2^-2 >= w_ij >= 2^-(n-1). In the
integer coordinates x_ij = 2^(n-1) w_ij of Forcey et al., Sp(n) is
    sum_{j != i} x_ij = 2^(n-2)                     (Kraft)
    1 <= x_ij <= 2^(n-3)                            (caterpillar, cherry clade-face)
    x_ka + x_kb - x_ab <= 2^(n-3)                   (intersecting cherry, k in the middle)
    sum_{i in S, j not in S} x_ij >= 2^(n-2)        (every cut, equivalent to the split rows)
and Q_T adds the equalities of the n-3 nontrivial splits of T.

Without --shapes: every log-integral point of Sp(n). With --shapes: for one tree of each
shape, every log-integral point of Q_T. Gurobi enumerates the points (binary level
indicators, solution pool, zero objective); each one is re-checked in integer arithmetic and
classified as a tree or not. --floor also holds every other cut at 3/4 or more (the next rung
of the cut-value ladder theorem), --no-dc drops the intersecting-cherry rows, and --rung
(with --shapes) holds the 2(n-3) cuts that are 3/4 at w_T at exactly 3/4. Writes
OUTDIR/points_n<N>[_shapes][_floor][_nodc][_rung].json; progress to stderr.
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


def say(msg):
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)


def all_trees(n):
    """Edge lists of every binary tree on leaves 0..n-1, by attaching leaf k to every edge."""
    trees = [[(0, n), (1, n), (2, n)]]
    for k in range(3, n):
        mid = n + k - 2
        nxt = []
        for E in trees:
            for e, (a, b) in enumerate(E):
                nxt.append(E[:e] + E[e + 1:] + [(a, mid), (mid, b), (k, mid)])
        trees = nxt
    return trees


def shape_representatives(n):
    """One tree per shape, built by attaching leaf k to every edge of each shape on k leaves
    and keeping the first tree of each new shape; avoids listing all (2n-5)!! trees."""
    reps = {"": [(0, n), (1, n), (2, n)]}
    for k in range(3, n):
        mid = n + k - 2
        nxt = {}
        for E in reps.values():
            for e, (a, b) in enumerate(E):
                F = E[:e] + E[e + 1:] + [(a, mid), (mid, b), (k, mid)]
                nxt.setdefault(shape_key(F, k + 1), F)
        reps = nxt
    return list(reps.values())


def leaf_distances(E, n):
    adj = {}
    for a, b in E:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    D = {}
    for s in range(n):
        dist, frontier = {s: 0}, [s]
        while frontier:
            nxt = []
            for v in frontier:
                for u in adj[v]:
                    if u not in dist:
                        dist[u] = dist[v] + 1
                        nxt.append(u)
            frontier = nxt
        for t in range(n):
            D[s, t] = dist[t]
    return D


def splits(E, n):
    """Nontrivial splits of a tree, each as the side without leaf 0."""
    out = set()
    for e in range(len(E)):
        adj = {}
        for f, (a, b) in enumerate(E):
            if f != e:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
        side, stack = {E[e][0]}, [E[e][0]]
        while stack:
            v = stack.pop()
            for u in adj.get(v, []):
                if u not in side:
                    side.add(u)
                    stack.append(u)
        leaves = frozenset(v for v in side if v < n)
        if 0 in leaves:
            leaves = frozenset(range(n)) - leaves
        if 2 <= len(leaves) <= n - 2:
            out.add(leaves)
    return out


def shape_key(E, n):
    """Canonical string of the unlabelled tree (AHU encoding from its centre)."""
    adj = {}
    for a, b in E:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    deg = {v: len(adj[v]) for v in adj}
    layer, removed = [v for v in adj if deg[v] == 1], set()
    remaining = len(adj)
    while remaining > 2:
        remaining -= len(layer)
        removed |= set(layer)
        nxt = []
        for v in layer:
            for u in adj[v]:
                if u not in removed:
                    deg[u] -= 1
                    if deg[u] == 1:
                        nxt.append(u)
        layer = nxt
    centres = [v for v in adj if v not in removed]

    def enc(v, parent):
        return "(" + "".join(sorted(enc(u, v) for u in adj[v] if u != parent)) + ")"

    if len(centres) == 1:
        return enc(centres[0], None)
    a, b = centres
    return "".join(sorted([enc(a, b), enc(b, a)]))


def enumerate_points(n, fixed_splits=(), floor=False, dc=True, rung=()):
    pairs = list(itertools.combinations(range(n), 2))
    levels = range(2, n)
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = int(os.environ["BME_THREADS"])
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions = 2_000_000_000
    m.Params.MIPGap = 0
    z = {(p, l): m.addVar(vtype=GRB.BINARY) for p in pairs for l in levels}
    X = {p: gp.quicksum(2 ** (n - 1 - l) * z[p, l] for l in levels) for p in pairs}
    xx = lambda i, j: X[(min(i, j), max(i, j))]
    for p in pairs:
        m.addConstr(gp.quicksum(z[p, l] for l in levels) == 1)
    for i in range(n):
        m.addConstr(gp.quicksum(xx(i, j) for j in range(n) if j != i) == 2 ** (n - 2))
    for a, b, k in itertools.permutations(range(n), 3):
        if a < b and dc:
            m.addConstr(xx(k, a) + xx(k, b) - xx(a, b) <= 2 ** (n - 3))
    cut = lambda S: gp.quicksum(xx(i, j) for i in S for j in range(n) if j not in S)
    fixed = set(fixed_splits)
    for size in range(2, n - 1):
        for S in itertools.combinations(range(1, n), size):
            low = 3 * 2 ** (n - 3) if floor and frozenset(S) not in fixed else 2 ** (n - 2)
            m.addConstr(cut(S) >= low)
    for S in fixed_splits:
        m.addConstr(cut(S) == 2 ** (n - 2))
    for S in rung:
        m.addConstr(cut(S) == 3 * 2 ** (n - 3))
    m.setObjective(0)
    m.optimize()
    pts = set()
    for s in range(m.SolCount):
        m.Params.SolutionNumber = s
        pts.add(tuple(int(round(sum(2 ** (n - 1 - l) * z[p, l].Xn for l in levels))) for p in pairs))
    return pairs, sorted(pts), m.Status


def check(n, pairs, x, fixed_splits=(), floor=False, dc=True):
    """Exact integer check of every row."""
    v = dict(zip(pairs, x))
    xx = lambda i, j: v[(min(i, j), max(i, j))]
    ok = all(1 <= t <= 2 ** (n - 3) and (t & (t - 1)) == 0 for t in x)
    ok &= all(sum(xx(i, j) for j in range(n) if j != i) == 2 ** (n - 2) for i in range(n))
    ok &= not dc or all(xx(k, a) + xx(k, b) - xx(a, b) <= 2 ** (n - 3) for a, b, k in itertools.permutations(range(n), 3))
    for size in range(2, n - 1):
        for S in itertools.combinations(range(1, n), size):
            c = sum(xx(i, j) for i in S for j in range(n) if j not in S)
            low = 3 * 2 ** (n - 3) if floor and frozenset(S) not in fixed_splits else 2 ** (n - 2)
            ok &= c >= low
    for S in fixed_splits:
        ok &= sum(xx(i, j) for i in S for j in range(n) if j not in S) == 2 ** (n - 2)
    return ok


def main():
    n, outdir = int(sys.argv[1]), Path(sys.argv[2])
    shapes = "--shapes" in sys.argv
    floor, dc = "--floor" in sys.argv, "--no-dc" not in sys.argv
    use_rung = "--rung" in sys.argv
    outdir.mkdir(parents=True, exist_ok=True)
    pairs = list(itertools.combinations(range(n), 2))
    xvec = lambda E: tuple(2 ** (n - 1 - leaf_distances(E, n)[p]) for p in pairs)
    jobs = [("Sp", None, ())]
    tree_x = {}
    if shapes:
        # In Q_T the only tree is T itself (a tree that displays all of T's splits is T).
        jobs = [("Q_T", E, tuple(splits(E, n))) for E in shape_representatives(n)]
        tree_x = {xvec(E): E for _, E, _ in jobs}
        say(f"n = {n}: {len(jobs)} tree shapes")
    else:
        trees = all_trees(n)
        tree_x = {xvec(E): E for E in trees}
        assert len(tree_x) == len(trees)
        say(f"n = {n}: {len(trees)} trees")
    out = []
    for label, E, fixed in jobs:
        t0 = time.time()
        rung = ()
        if use_rung:
            xT = [x for x, F in tree_x.items() if F is E][0]
            v = dict(zip(pairs, xT))
            rung = tuple(frozenset(S) for size in range(2, n - 1) for S in itertools.combinations(range(1, n), size)
                         if sum(v[(min(i, j), max(i, j))] for i in S for j in range(n) if j not in S) == 3 * 2 ** (n - 3))
            assert len(rung) == 2 * (n - 3), len(rung)
        _, pts, status = enumerate_points(n, fixed, floor, dc, rung)
        assert status == GRB.OPTIMAL, status
        fs = set(fixed)
        assert all(check(n, pairs, x, fs, floor, dc) for x in pts)
        is_tree = [x in tree_x and (E is None or tree_x[x] is E) for x in pts]
        rec = {"relaxation": label, "floor": floor, "dc": dc, "n": n, "points": len(pts), "trees": sum(is_tree),
               "non_trees": len(pts) - sum(is_tree), "seconds": round(time.time() - t0, 1),
               "pairs": [list(p) for p in pairs], "x": [list(x) for x in pts], "is_tree": is_tree}
        if E is not None:
            rec["tree_splits"] = sorted(sorted(S) for S in fixed)
            rec["tree_x"] = [list(x) for x, F in tree_x.items() if F is E][0]
            rec["edges"] = [list(e) for e in E]
            rec["shape"] = shape_key(E, n)
        say(f"{label}{' ' + rec.get('shape', '') if E is not None else ''}: {len(pts)} log-integral points, "
            f"{rec['trees']} trees, {rec['non_trees']} non-trees ({rec['seconds']} s)")
        out.append(rec)
    f = outdir / f"points_n{n}{'_shapes' if shapes else ''}{'_floor' if floor else ''}{'_nodc' if not dc else ''}{'_rung' if use_rung else ''}.json"
    f.write_text(json.dumps(out))
    say(f"wrote {f}")


if __name__ == "__main__":
    main()
