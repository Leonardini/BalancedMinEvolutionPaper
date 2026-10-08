"""Section 5.2, Conjecture 1: the integral points of the side-membership model and the
binary topologies on n leaves.

    python membership_model.py N LAMINARITY TIEBREAK OUT.json [--lex]
      LAMINARITY  nested-disjoint | quadrant
      TIEBREAK    on | off   (for even n: leaf 1 outside every split of size n/2)
      --lex       also order splits of equal size lexicographically (one point per tree)

Model (binary s_ik = 1 iff leaf i is on the reference side A_k of split k, k = 1..n-3):
  |A_k| <= floor(n/2)                       reference side is the minority side
  |A_k| <= |A_{k+1}|                        splits ordered by size
  |A_1| = |A_2| = 2, A_1 and A_2 disjoint   the first two splits are disjoint cherries
                                            (with the size order, every split is nontrivial)
  even n, TIEBREAK on: |A_k| + s_1k <= n/2  leaf 1 is outside every split of size n/2
  laminarity, for every pair k < l, either
    nested-disjoint: one binary z_kl,  s_ik - s_il <= 1 - z,  s_ik + s_il <= 1 + z,
                     |A_l| - |A_k| >= z    (z = 1: A_k strictly inside A_l; z = 0: disjoint)
    quadrant:        q_ikl = s_ik s_il (McCormick, exact at binary s); binaries e^ab_kl,
                     e^ab <= 1 - [leaf i in quadrant ab] for every i, sum_ab e^ab >= 1
                     (some quadrant is empty), and sum_i (s_ik + s_il - 2 q_ikl) >= 1
                     (the two sides differ)
  --lex: with v_k = sum_i 2^(n-1-i) s_ik (the membership vector read as a binary number,
         leaf 1 the most significant bit), for consecutive k:
           v_k - v_{k+1} >= 1 - 2^n (|A_{k+1}| - |A_k|)
         When |A_k| = |A_{k+1}| this is v_k > v_{k+1}; otherwise |A_{k+1}| - |A_k| >= 1 and,
         since |v_k - v_{k+1}| <= 2^n - 1, it is slack.
The B, P and w layers of the full model are determined by s and play no part here.

Without --lex, a tree with c_m splits of size m appears once per ordering of its
equal-size splits, prod_m c_m! points; the script checks this multiplicity for every split
system. The size order, the minority side and the tiebreak fix everything else, and the
first-two-cherries rule only reads the first two size-2 splits of that order. Under --lex
the order within each size class is fixed too, so each tree must appear exactly once.

Gurobi enumerates every feasible point (PoolSearchMode 2, zero objective). Each point is
read as a set of bipartitions; it is a tree iff it has n-3 distinct, nontrivial,
pairwise compatible bipartitions. Writes the number of points, of distinct split
systems, and of those that are trees, against (2n-5)!!.
"""
import collections
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent))
import published as P  # noqa: E402

POOL_CAP = 10_000_000          # an upper limit on stored points; reaching it is an error
PROGRESS_SECONDS = float(os.environ.get("BME_PROGRESS", "60"))


def say(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, file=sys.stderr, flush=True)


def double_factorial(k):
    p = 1
    while k > 1:
        p *= k
        k -= 2
    return p


def build(n, laminarity, tiebreak, lex):
    K = range(n - 3)
    half = n // 2
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = int(os.environ["BME_THREADS"])
    s = {(i, k): m.addVar(vtype=GRB.BINARY) for i in range(n) for k in K}
    size = {k: gp.quicksum(s[i, k] for i in range(n)) for k in K}
    for k in K:
        m.addConstr(size[k] <= half)
        if tiebreak and n % 2 == 0:
            m.addConstr(size[k] + s[0, k] <= half)
    for k in K[:-1]:
        m.addConstr(size[k] <= size[k + 1])
    m.addConstr(size[0] == 2)
    m.addConstr(size[1] == 2)
    for i in range(n):
        m.addConstr(s[i, 0] + s[i, 1] <= 1)
    if lex:
        value = {k: gp.quicksum(2 ** (n - 1 - i) * s[i, k] for i in range(n)) for k in K}
        for k in K[:-1]:
            m.addConstr(value[k] - value[k + 1] >= 1 - 2 ** n * (size[k + 1] - size[k]))
    for k, l in itertools.combinations(K, 2):
        if laminarity == "nested-disjoint":
            z = m.addVar(vtype=GRB.BINARY)
            for i in range(n):
                m.addConstr(s[i, k] - s[i, l] <= 1 - z)
                m.addConstr(s[i, k] + s[i, l] <= 1 + z)
            m.addConstr(size[l] - size[k] >= z)
        elif laminarity == "quadrant":
            q = {i: m.addVar(vtype=GRB.BINARY) for i in range(n)}
            e = {ab: m.addVar(vtype=GRB.BINARY) for ab in ((1, 1), (1, 0), (0, 1), (0, 0))}
            for i in range(n):
                m.addConstr(q[i] <= s[i, k])
                m.addConstr(q[i] <= s[i, l])
                m.addConstr(q[i] >= s[i, k] + s[i, l] - 1)
                member = {(1, 1): q[i], (1, 0): s[i, k] - q[i], (0, 1): s[i, l] - q[i],
                          (0, 0): 1 - s[i, k] - s[i, l] + q[i]}
                for ab in e:
                    m.addConstr(e[ab] <= 1 - member[ab])
            m.addConstr(gp.quicksum(e.values()) >= 1)
            m.addConstr(gp.quicksum(s[i, k] + s[i, l] - 2 * q[i] for i in range(n)) >= 1)
        else:
            raise SystemExit(f"unknown laminarity encoding {laminarity}")
    return m, s


def is_tree(system, n):
    if len(system) != n - 3:
        return False
    full = frozenset(range(n))
    if any(not 2 <= len(A) <= n - 2 for A in system):
        return False
    return all(not (A & B and A - B and B - A and full - A - B)
               for A, B in itertools.combinations(system, 2))


def main():
    if len(sys.argv) not in (5, 6) or (len(sys.argv) == 6 and sys.argv[5] != "--lex"):
        raise SystemExit(__doc__)
    n, laminarity, tb, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    lex = len(sys.argv) == 6
    if tb not in ("on", "off"):
        raise SystemExit("TIEBREAK must be on or off")
    tiebreak = tb == "on"
    out.parent.mkdir(parents=True, exist_ok=True)
    m, s = build(n, laminarity, tiebreak, lex)
    m.update()   # so that NumVars and NumConstrs are filled in for the log line
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions = POOL_CAP
    t0 = time.time()
    last = [t0]

    def progress(model, where):
        if where == GRB.Callback.MIP and time.time() - last[0] >= PROGRESS_SECONDS:
            last[0] = time.time()
            say(f"search: {int(model.cbGet(GRB.Callback.MIP_SOLCNT))} points so far, "
                f"{model.cbGet(GRB.Callback.MIP_NODCNT):.0f} nodes")

    say(f"n={n} laminarity={laminarity} tiebreak={tb} lex={lex}: {m.NumVars} variables, "
        f"{m.NumConstrs} constraints")
    m.optimize(progress)
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status {m.Status}")
    npts = m.SolCount
    if npts >= POOL_CAP:
        raise RuntimeError("the solution pool is full; the enumeration may be incomplete")
    say(f"search done: {npts} points in {time.time() - t0:.0f}s; reading them")
    svars = [s[i, k] for k in range(n - 3) for i in range(n)]
    full = frozenset(range(n))
    systems = collections.Counter()
    orderings = collections.defaultdict(set)
    for idx in range(npts):
        m.Params.SolutionNumber = idx
        x = m.getAttr("Xn", svars)
        order = tuple(frozenset(i for i in range(n) if x[k * n + i] > 0.5)
                      for k in range(n - 3))
        # one representative per bipartition: the side without leaf 1
        system = frozenset(A if 0 not in A else full - A for A in order)
        systems[system] += 1
        orderings[system].add(order)
        if time.time() - last[0] >= PROGRESS_SECONDS:
            last[0] = time.time()
            say(f"read {idx + 1}/{npts} points, {len(systems)} distinct systems")
    trees = sum(is_tree(S, n) for S in systems)
    multiplicity = collections.Counter(systems.values())
    rec = dict(n=n, laminarity=laminarity, tiebreak=tb, lex=lex, points=npts,
               distinct_split_systems=len(systems), tree_systems=trees,
               non_tree_systems=len(systems) - trees, topologies=double_factorial(2 * n - 5),
               points_per_system={str(c): multiplicity[c] for c in sorted(multiplicity)},
               seconds=time.time() - t0)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))
    if tiebreak or n % 2:
        assert len(systems) == trees == double_factorial(2 * n - 5), rec
        for system, count in systems.items():
            # every point of a system is a distinct ordering of its splits
            assert len(orderings[system]) == count, (sorted(map(sorted, system)), count)
            sizes = collections.Counter(min(len(A), n - len(A)) for A in system)
            expected = 1 if lex else math.prod(math.factorial(c) for c in sizes.values())
            assert count == expected, (sorted(map(sorted, system)), count, expected)
        if lex:
            assert npts == double_factorial(2 * n - 5), rec
        if n in P.MEMBERSHIP_SYSTEMS:
            assert len(systems) == P.MEMBERSHIP_SYSTEMS[n], rec
    elif n == 8:
        assert len(systems) == P.MEMBERSHIP_N8_NO_TIEBREAK, rec


if __name__ == "__main__":
    main()
