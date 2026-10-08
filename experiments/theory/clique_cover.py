"""Section 5.2: minimum edge clique cover of the crossing graph G on the nontrivial
bipartitions of [n].

    python clique_cover.py N OUT.json

Every edge clique cover can be replaced by one made of maximal cliques of the same
size, so the minimum is the optimum of the set-cover integer program
    min sum_C z_C  s.t.  sum_{C containing e} z_C >= 1 for every edge e of G,  z binary,
over the maximal cliques C (networkx), solved to proven optimality (MIPGap 0). Also
records the number of cliques in the cover of Proposition 4, for comparison.
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossing  # noqa: E402
import published as P  # noqa: E402


def say(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, file=sys.stderr, flush=True)


n, out = int(sys.argv[1]), Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
splits = crossing.bipartitions(n)
G = nx.Graph()
G.add_nodes_from(range(len(splits)))
G.add_edges_from((a, b) for a, b in itertools.combinations(range(len(splits)), 2)
                 if crossing.cross(splits[a], splits[b], n))
cliques = [frozenset(C) for C in nx.find_cliques(G) if len(C) >= 2]
say(f"n={n}: {G.number_of_nodes()} bipartitions, {G.number_of_edges()} crossing pairs, "
    f"{len(cliques)} maximal cliques")

m = gp.Model()
m.Params.OutputFlag = 0
m.Params.Threads = int(os.environ["BME_THREADS"])
m.Params.MIPGap = 0
z = m.addVars(len(cliques), vtype=GRB.BINARY)
for u, v in G.edges():
    m.addConstr(gp.quicksum(z[c] for c, C in enumerate(cliques) if u in C and v in C) >= 1)
m.setObjective(z.sum(), GRB.MINIMIZE)
m.Params.TimeLimit = float(os.environ.get("BME_CAP", "3600"))
t0 = time.time()
last = [t0]


def progress(model, where):
    if where == GRB.Callback.MIP and time.time() - last[0] >= 60:
        last[0] = time.time()
        say(f"incumbent {model.cbGet(GRB.Callback.MIP_OBJBST):.0f}, "
            f"bound {model.cbGet(GRB.Callback.MIP_OBJBND):.3f}, "
            f"{model.cbGet(GRB.Callback.MIP_NODCNT):.0f} nodes")


m.optimize(progress)
if m.Status != GRB.OPTIMAL:
    raise RuntimeError(f"Gurobi status {m.Status}")
cover = round(m.ObjVal)
if abs(m.ObjBound - m.ObjVal) > 1e-6:
    raise RuntimeError(f"optimality not proven: bound {m.ObjBound}, value {m.ObjVal}")
rec = dict(n=n, vertices=G.number_of_nodes(), edges=G.number_of_edges(),
           maximal_cliques=len(cliques), min_edge_clique_cover=cover,
           proposition4_cliques=len(crossing.prop4_cliques(n, splits)),
           seconds=time.time() - t0)
out.write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec))
assert cover == P.MIN_EDGE_CLIQUE_COVER[n], rec
