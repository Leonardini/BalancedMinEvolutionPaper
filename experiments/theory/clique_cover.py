"""Section 5.1 and Appendix H: edge clique covers of the crossing graph G on the nontrivial
bipartitions of [n].

    python clique_cover.py N OUT.json [--no-ilp]

Records, for G:
- the number of distinct cliques in the cover of Proposition 4 and in its subfamily of
  Appendix H, each checked to be a cover (crossing.covers);
- the fractional edge clique cover number, computed by symmetry: relabelling the leaves
  maps the covering LP to itself, so averaging an optimal solution over the relabellings
  gives an optimal one that is constant on each class of crossing pairs (pairs that a
  relabelling maps to each other). With N_o pairs in class o and c_o(C) of them inside the
  maximal clique C, the LP becomes
      primal  min sum_v t_v  s.t.  sum_v c_o(v) t_v >= N_o for every class o,  t >= 0
      dual    max sum_o N_o y_o  s.t.  sum_o c_o(v) y_o <= 1 for every vector v,  y >= 0,
  over the distinct count vectors v = (c_o(C))_o of the maximal cliques. Both optimal
  solutions are rounded to fractions and checked exactly; the dual gives a lower bound on
  every edge clique cover, the primal shows the bound is the LP value. The class of a pair
  (S, T) is the orbit of its quadrant sizes (|S & T|, |S - T|, |T - S|, |rest|) under
  swapping S and T and complementing either;
- unless --no-ilp, the minimum edge clique cover: every edge clique cover can be replaced
  by one made of maximal cliques of the same size, so the minimum is the optimum of
      min sum_C z_C  s.t.  sum_{C containing e} z_C >= 1 for every edge e of G,  z binary,
  over the maximal cliques C (networkx), solved to proven optimality (MIPGap 0).
Progress to stderr.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossing  # noqa: E402
import published as P  # noqa: E402

T0 = time.time()


def say(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"[{time.time() - T0:6.0f} s]", msg, file=sys.stderr, flush=True)


def pair_class(S, T, full):
    todo, orbit = [(len(S & T), len(S - T), len(T - S), len(full - S - T))], set()
    while todo:
        q = todo.pop()
        if q not in orbit:
            orbit.add(q)
            todo += [(q[0], q[2], q[1], q[3]), (q[2], q[3], q[0], q[1]), (q[1], q[0], q[3], q[2])]
    return min(orbit)


def exact(values, den=10**6):
    return [Fraction(v).limit_denominator(den) for v in values]


ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("out", type=Path)
ap.add_argument("--no-ilp", action="store_true")
a = ap.parse_args()
n, out = a.n, a.out
out.parent.mkdir(parents=True, exist_ok=True)
full = frozenset(range(n))
splits = crossing.bipartitions(n)
V = len(splits)
G = nx.Graph()
G.add_nodes_from(range(V))
cls = np.full((V, V), -1, dtype=np.int16)
classes = {}
for u, v in itertools.combinations(range(V), 2):
    if crossing.cross(splits[u], splits[v], n):
        G.add_edge(u, v)
        cls[u, v] = cls[v, u] = classes.setdefault(pair_class(splits[u], splits[v], full), len(classes))
K = len(classes)
iu = np.triu_indices(V, 1)
N = np.bincount(cls[iu][cls[iu] >= 0], minlength=K)
say(f"n={n}: {V} bipartitions, {G.number_of_edges()} crossing pairs in {K} classes")

prop4 = crossing.prop4_cliques(n, splits)
pruned = crossing.pruned_cliques(n, splits)
assert crossing.covers(n, splits, prop4) and crossing.covers(n, splits, pruned)
say(f"Proposition 4: {len(prop4)} cliques; Appendix H subfamily: {len(pruned)} cliques; both covers")

cliques, vecs, last, n_maximal = [], set(), time.time(), 0
for C in nx.find_cliques(G):
    if len(C) < 2:
        continue
    n_maximal += 1
    idx = np.array(C)
    sub = cls[np.ix_(idx, idx)]
    vecs.add(tuple((np.bincount(sub[sub >= 0], minlength=K) // 2).tolist()))
    if not a.no_ilp:
        cliques.append(frozenset(C))
    if time.time() - last >= 60:
        last = time.time()
        say(f"enumerating maximal cliques: {len(vecs)} distinct count vectors so far")
A = np.array(sorted(vecs), dtype=float)
dual = linprog(-N.astype(float), A_ub=A, b_ub=np.ones(len(A)), bounds=[(0, None)] * K, method="highs")
primal = linprog(np.ones(len(A)), A_ub=-A.T, b_ub=-N.astype(float), bounds=[(0, None)] * len(A), method="highs")
assert dual.status == 0 and primal.status == 0
y, t = exact(dual.x), exact(primal.x)
Ai = [[int(c) for c in row] for row in sorted(vecs)]
assert all(yo >= 0 for yo in y) and all(sum(c * yo for c, yo in zip(row, y)) <= 1 for row in Ai)
assert all(tv >= 0 for tv in t) and all(sum(Ai[v][o] * t[v] for v in range(len(Ai))) >= int(N[o]) for o in range(K))
lower, upper = sum(int(N[o]) * y[o] for o in range(K)), sum(t)
assert lower == upper, (lower, upper)
say(f"fractional edge clique cover {lower} = {float(lower):.4f} ({len(Ai)} count vectors)")

rec = dict(n=n, vertices=V, edges=G.number_of_edges(), pair_classes=K,
           proposition4_cliques=len(prop4), pruned_cliques=len(pruned),
           fractional_edge_clique_cover=str(lower), lower_bound=math.ceil(lower),
           maximal_cliques=n_maximal, count_vectors=len(Ai))
if not a.no_ilp:
    assert len(cliques) == n_maximal
    say(f"{len(cliques)} maximal cliques; set-cover integer program")
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = int(os.environ["BME_THREADS"])
    m.Params.MIPGap = 0
    z = m.addVars(len(cliques), vtype=GRB.BINARY)
    for u, v in G.edges():
        m.addConstr(gp.quicksum(z[c] for c, C in enumerate(cliques) if u in C and v in C) >= 1)
    m.setObjective(z.sum(), GRB.MINIMIZE)
    m.Params.TimeLimit = float(os.environ.get("BME_CAP", "3600"))
    lastip = [time.time()]

    def progress(model, where):
        if where == GRB.Callback.MIP and time.time() - lastip[0] >= 60:
            lastip[0] = time.time()
            say(f"incumbent {model.cbGet(GRB.Callback.MIP_OBJBST):.0f}, "
                f"bound {model.cbGet(GRB.Callback.MIP_OBJBND):.3f}, "
                f"{model.cbGet(GRB.Callback.MIP_NODCNT):.0f} nodes")

    m.optimize(progress)
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status {m.Status}")
    if abs(m.ObjBound - m.ObjVal) > 1e-6:
        raise RuntimeError(f"optimality not proven: bound {m.ObjBound}, value {m.ObjVal}")
    rec.update(min_edge_clique_cover=round(m.ObjVal))
rec["seconds"] = time.time() - T0
out.write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec))
if not a.no_ilp:
    assert rec["min_edge_clique_cover"] == P.MIN_EDGE_CLIQUE_COVER[n], rec
assert str(lower) == P.FRACTIONAL_EDGE_CLIQUE_COVER[n], rec
assert len(pruned) == P.PRUNED_CLIQUE_COVER[n], rec
