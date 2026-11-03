"""Which non-tree log-integral points lie outside P_n = conv(trees)?

    BME_THREADS=1 python outside_trees.py POINTS.json

For each record of POINTS.json (lattice.py), solves for each non-tree point the LP
lambda >= 0, sum lambda = 1, sum lambda_T x_T = x over the trees of that record's n. A point
with no solution lies outside P_n, so the hull of the log-integral points has a non-tree
vertex. Prints the counts and the index of one outside point (for an exact check).
"""
import itertools
import json
import os
import sys

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lattice import all_trees, leaf_distances  # noqa: E402

for r in json.load(open(sys.argv[1])):
    n = r["n"]
    pairs = list(itertools.combinations(range(n), 2))
    T = [[2 ** (n - 1 - leaf_distances(E, n)[p]) for p in pairs] for E in all_trees(n)]
    env = gp.Env(params={"OutputFlag": 0, "Threads": int(os.environ["BME_THREADS"])})
    m = gp.Model(env=env)
    lam = m.addVars(len(T), lb=0)
    m.addConstr(lam.sum() == 1)
    rows = [m.addConstr(gp.quicksum(T[t][k] * lam[t] for t in range(len(T))) == 0) for k in range(len(pairs))]
    outside = []
    for i, (x, tree) in enumerate(zip(r["x"], r["is_tree"])):
        if tree:
            continue
        for k, c in enumerate(rows):
            c.RHS = x[k]
        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            outside.append(i)
        else:
            assert m.Status == GRB.OPTIMAL, m.Status
    print(f"n = {n}: {r['non_trees']} non-tree log-integral points, {len(outside)} outside conv(trees); "
          f"first: {outside[:1]}")
