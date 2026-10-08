"""Section 8.5: can a quartet violated at the compact model's root point be repaired locally?

    BME_THREADS=1 python quartet_fixes.py INSTANCE SPLITS OUT.json

1. The compact solver's own root (root_exact, solver/bnb_balanced._solve_node): its cut
   loop in CPLEX, then the manifold constraint imposed exactly by MOSEK, separation at the
   conic optimum repeated until no cut is violated. w* is the conic optimum; every root
   row is checked exactly at T.
2. A quartet {a,b,c,d} whose resolution in the certified optimal tree T is ab|cd is
   VIOLATED at w* if w* fails one of T's strong four-point inequalities
       P0 >= 4 P1,  P0 >= 4 P2,   P0 = w_ab w_cd,  P1 = w_ac w_bd,  P2 = w_ad w_bc.
   Its SEPARATION at w* is log2 P0 - log2 max(P1, P2) (T asks for at least 2), and its
   WINNER is the pairing with the largest product (T's, or a wrong one).
3. A LOCAL FIX changes the quartet's six weights only. The changes that keep Kraft
   (sum_j w_ij = 1/2 for every leaf) are exactly
       Delta_ab = Delta_cd = s,  Delta_ac = Delta_bd = t,  Delta_ad = Delta_bc = -(s + t)
   (the Kraft rows of the quartet's four leaves have rank 4 on its six pairs). On this plane
   the manifold function g(w) = sum_{i<j} w_ij ln w_ij has gradient (ln(P0/P2), ln(P1/P2)),
   so it is smallest where P0 = P1 = P2 and grows as the three products separate.
   For each violated quartet the experiment computes the least MANIFOLD EXCESS of a local fix,
       min  g(w* + Delta(s, t)) - (3/2 - n) ln 2   (reported in log2 units)
       s.t. T's two four-point inequalities, 2^-(n-1) <= w <= 1/4,
   (a) with nothing else, (b) with every row of the root relaxation as well (Kraft, double
   cherry, the separated cuts), each solved globally by Gurobi on two variables. The value
   recorded is Gurobi's proven lower bound (a fix needs at least this much); "infeasible"
   means no local fix exists at all. w* itself has excess at most its conic tolerance, so a
   positive excess means that no local fix keeps the manifold.
"""
import math
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import safe_bound  # noqa: E402
from lib.common import (compact_point, log, optimal_tree, optimum, parse_matrix,  # noqa: E402
                        peak_rss_mb, tree_w, write_json)

from solver.bnb_balanced import _solve_node, new_lp_model  # noqa: E402
from solver.exact_node import new_counts as new_exact_counts  # noqa: E402

VIOL_TOL = 1e-9       # a quartet counts as violated above this (log2 units)
SOLVE_CAP = 60.0      # seconds per two-variable global solve


def pairings(q):
    a, b, c, d = q
    return [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]


def true_violation(W, q, rt):
    """log2(4 max_other / P_rt), P the three pairing products of q at W: positive iff W
    violates the strong four-point inequalities of resolution rt."""
    P = [W[x[0] - 1, x[1] - 1] * W[y[0] - 1, y[1] - 1] for x, y in pairings(q)]
    other = max(P[s] for s in range(3) if s != rt)
    return math.log2(4.0 * other / P[rt])


def root_exact(D, cap, tree):
    """The compact solver's root with the manifold exact (step 1). Returns (the CPLEX model
    with the root's rows, W*, record)."""
    n = D.shape[0]
    c, pti, pairs = new_lp_model(D)
    counts = defaultdict(int)
    exact_counts = new_exact_counts()
    t0 = time.time()
    bound, wv, decided = _solve_node(c, D, pti, pairs, len(pairs), [], [], counts, exact_counts,
                                     deadline=t0 + cap)
    if wv is None:
        raise RuntimeError("the root was not solved within the cap")
    seconds = time.time() - t0
    W = np.zeros((n, n))
    for (i, j), k in pti.items():
        W[i - 1, j - 1] = W[j - 1, i - 1] = wv[k]
    lp, _ = safe_bound.lp_of(c)
    pt = compact_point(tree, c.variables.get_names())
    chk = safe_bound.tree_check(lp, [pt[nm] for nm in c.variables.get_names()])
    rec = dict(bound=bound, seconds=seconds, cuts=dict(counts), exact_counts=exact_counts,
               capped=not decided or seconds >= cap,
               rows_cut_off_optimum=chk["rows_cut_off_optimum"] + chk["bounds_violated"],
               optimum_check=chk)
    return c, W, rec


def root_rows(c):
    """The root's linear rows over the w columns: (list of {pair: coef}, senses, rhs)."""
    names = c.variables.get_names()
    col = {k: tuple(int(x) for x in nm.split("_")[1:]) for k, nm in enumerate(names) if nm.startswith("w_")}
    rows = []
    for sp in c.linear_constraints.get_rows():
        rows.append({col[i]: v for i, v in zip(sp.ind, sp.val) if i in col})
    return rows, list(c.linear_constraints.get_senses()), list(c.linear_constraints.get_rhs())


def min_excess(q, rt, W, n, g_rest, rows_touching, row_tol):
    """The least manifold excess (ln units) of a local fix of q, T's resolution rt; with
    rows_touching (a list of (row, sense, rhs)) the root's rows are imposed too. Returns
    (status, proven lower bound, value, s, t)."""
    import gurobipy as gp
    from gurobipy import GRB, nlfunc
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.NonConvex = 2
    m.Params.FeasibilityTol = 1e-9
    m.Params.MIPGap = 1e-6
    m.Params.TimeLimit = SOLVE_CAP
    lo, hi = 2.0 ** -(n - 1), 0.25
    s = m.addVar(lb=-1, ub=1, name="s")
    t = m.addVar(lb=-1, ub=1, name="t")
    prs = [[(min(a, b), max(a, b)) for a, b in pr] for pr in pairings(q)]
    others = [k for k in range(3) if k != rt]
    w = {}
    for k, (cs, ct) in ((rt, (1, 0)), (others[0], (0, 1)), (others[1], (-1, -1))):
        for p in prs[k]:
            w[p] = m.addVar(lb=lo, ub=hi, name=f"w_{p[0]}_{p[1]}")
            m.addConstr(w[p] == W[p[0] - 1, p[1] - 1] + cs * s + ct * t)
    # T's four-point inequalities, scaled by 1 / P0(w*) so that they are of order 1 and the
    # solver's absolute feasibility tolerance is a relative one.
    (x, y) = prs[rt]
    scale = 1.0 / (W[x[0] - 1, x[1] - 1] * W[y[0] - 1, y[1] - 1])
    for k in others:
        (u, v) = prs[k]
        m.addQConstr(scale * w[x] * w[y] - 4.0 * scale * w[u] * w[v] >= 0.0)
    for row, sense, rhs in rows_touching:
        lhs = gp.quicksum(a * w[p] for p, a in row.items() if p in w)
        const = sum(a * W[p[0] - 1, p[1] - 1] for p, a in row.items() if p not in w)
        if sense in ("G", "E"):
            m.addConstr(lhs + const >= rhs - row_tol)
        if sense in ("L", "E"):
            m.addConstr(lhs + const <= rhs + row_tol)
    z = {p: m.addVar(lb=-GRB.INFINITY) for p in w}
    for p in w:
        m.addGenConstrNL(z[p], w[p] * nlfunc.log(w[p]))
    m.setObjective(gp.quicksum(z.values()) + g_rest - (1.5 - n) * math.log(2), GRB.MINIMIZE)
    m.optimize()
    # Every variable is bounded, so "infeasible or unbounded" means infeasible.
    if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return "infeasible", None, None, None, None
    if m.SolCount == 0:
        return f"no solution (status {m.Status})", m.ObjBound, None, None, None
    return ("optimal" if m.Status == GRB.OPTIMAL else f"status {m.Status}"), m.ObjBound, m.ObjVal, s.X, t.X


def main():
    inst, splits, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    D = parse_matrix(inst)
    n = D.shape[0]
    L = optimum(D, splits)
    Wt = tree_w(splits, n)
    tree = optimal_tree(D, splits_path=splits)
    c, W, root = root_exact(D, 3600, tree)
    log(f"root LB={root['bound']:.10f} gap={100 * (L - root['bound']) / L:.4f}%")
    rows, sen, rhs = root_rows(c)
    rows_by_pair = {}
    for r, row in enumerate(rows):
        for p in row:
            rows_by_pair.setdefault(p, []).append(r)

    def resid_signed(row, sense, b):
        """Slack of a row at w* (negative: violated)."""
        a = sum(v * W[p[0] - 1, p[1] - 1] for p, v in row.items())
        return a - b if sense == "G" else b - a if sense == "L" else -abs(a - b)

    def resid(row, sense, b):
        a = sum(v * W[p[0] - 1, p[1] - 1] for p, v in row.items())
        return max(b - a, 0.0) if sense == "G" else max(a - b, 0.0) if sense == "L" else abs(a - b)
    # w* satisfies the root rows only to the conic solver's tolerance; a fix gets the same slack.
    row_tol = max([1e-12] + [resid(r_, s_, b_) for r_, s_, b_ in zip(rows, sen, rhs)])
    iu = np.triu_indices(n, 1)
    g_full = float(np.sum(W[iu] * np.log(W[iu])))
    root.update(row_residual=row_tol, manifold_excess_ln=g_full - (1.5 - n) * math.log(2))
    log(f"w*: row residual {row_tol:.3g}, manifold excess {root['manifold_excess_ln']:.3g} (ln units)")

    res_of = {}
    for q in combinations(range(1, n + 1), 4):
        P = [Wt[x[0] - 1, x[1] - 1] * Wt[y[0] - 1, y[1] - 1] for x, y in pairings(q)]
        res_of[q] = int(np.argmax(P))
    violated = [q for q in res_of if true_violation(W, q, res_of[q]) > VIOL_TOL]
    log(f"{len(violated)} of {len(res_of)} quartets violated at w*")
    rec = dict(instance=inst, n=n, optimum=L, root=root, quartets=len(res_of),
               violated=len(violated), fixes=[])
    t0 = time.time()
    for k, q in enumerate(violated):
        rt = res_of[q]
        P = [W[x[0] - 1, x[1] - 1] * W[y[0] - 1, y[1] - 1] for x, y in pairings(q)]
        six = {(min(i, j), max(i, j)) for i, j in combinations(q, 2)}
        g_rest = g_full - sum(W[p[0] - 1, p[1] - 1] * math.log(W[p[0] - 1, p[1] - 1]) for p in six)
        touching = sorted({r for p in six for r in rows_by_pair.get(p, ())})
        # Rows of the quartet's pairs that are tight at w* and that a local move changes.
        tight = sum(1 for i in touching if sen[i] != "E" and resid_signed(rows[i], sen[i], rhs[i]) <= 1e-7
                    and any(p in six for p in rows[i]))
        r = dict(quartet=list(q), resolution=rt, tight_rows=tight,
                 separation=math.log2(P[rt] / max(P[j] for j in range(3) if j != rt)),
                 winner_is_T=bool(int(np.argmax(P)) == rt))
        for tag, rt_rows in (("box", []), ("rows", [(rows[i], sen[i], rhs[i]) for i in touching])):
            st, lb, val, sv, tv = min_excess(q, rt, W, n, g_rest, rt_rows, row_tol)
            r[tag] = dict(status=st, excess_lb=None if lb is None else lb / math.log(2),
                          excess=None if val is None else val / math.log(2), s=sv, t=tv)
        rec["fixes"].append(r)
        if (k + 1) % 200 == 0 or k + 1 == len(violated):
            log(f"{k + 1}/{len(violated)} quartets, {time.time() - t0:.0f}s")
            write_json(out, rec)

    def summary(tag):
        xs = [f[tag] for f in rec["fixes"]]
        feas = [x for x in xs if x["status"] != "infeasible"]
        best = np.array([x["excess"] for x in feas if x["excess"] is not None])
        unsettled = sum(x["status"] not in ("optimal", "infeasible") for x in xs)
        return dict(infeasible=len(xs) - len(feas), unsettled=unsettled,
                    keeps_manifold=int(np.sum(best <= 1e-7)),
                    excess_quantiles=[float(v) for v in np.quantile(best, [0, .1, .25, .5, .75, .9, 1])]
                    if len(best) else None)
    # Weights in which w* and the optimal tree's point differ: w_T satisfies every quartet, so
    # the fewest weights that must change for all of them to hold is at most this.
    differ = int(np.sum(np.abs(W[iu] - Wt[iu]) > 1e-9))
    rec["summary"] = dict(
        violated=len(violated), pairs=len(iu[0]), weights_differing_from_tree=differ,
        tight_rows_quantiles=[float(v) for v in np.quantile([f["tight_rows"] for f in rec["fixes"]], [0, .25, .5, .75, 1])],
        winner_is_T=sum(f["winner_is_T"] for f in rec["fixes"]),
        separation_quantiles=[float(v) for v in np.quantile([f["separation"] for f in rec["fixes"]],
                                                            [0, .1, .25, .5, .75, .9, 1])],
        box=summary("box"), rows=summary("rows"))
    rec["peak_rss_mb"] = peak_rss_mb()
    write_json(out, rec)
    log(f"summary: {rec['summary']}")


if __name__ == "__main__":
    main()
