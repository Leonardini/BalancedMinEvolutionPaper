"""The safe LP bound of lib/safe_bound.py on LPs whose optimum is known, under CPLEX and
gurobipy.

    BME_THREADS=1 python experiments/lp/tests/test_safe_bound.py

1. A five-variable LP with one row of each sense, an objective constant, and
   negative-cost variables held at their upper bounds by a nonzero reduced cost. Its
   optimum is OPT (worked out below by hand). The safe bound from the solver's duals
   must be within the check's tolerance of OPT; the bound from the exact duals must
   equal OPT exactly; with noise added to the duals the bound must stay <= OPT.
2. Sign conventions: the reported duals of each solver lie in the sign cones
   (>= 0 on G rows, <= 0 on L rows) and give the expected values.
3. An infinite bound: the bound is -inf when a column with an infinite upper bound
   gets a negative reduced cost, and finite again with an implied bound.
4. A ranged CPLEX row is rejected.
4b. The optimum check (tree_check) at the known optimal point: no row violated, the
   objective there is OPT exactly; at a perturbed point, the violation is exact.
4c. The tight-tolerance re-solve of certify (forced), under both solvers.
5. Random bounded LPs, both solvers: safe bound within the check's tolerance of the
   LP objective, and below it with noisy duals.
"""
import math
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import safe_bound as sb  # noqa: E402

os.environ.setdefault("BME_THREADS", "1")

#   min  -2 x1 + 2 x2 + x3 + 1/2 x4 - x5 + C0
#   G:   x1 + x2 + x3 >= 2
#   L:   x1 + x5      <= 3/2
#   E:   x2 + x4       = 3/4
#   x1 in [0,1], x2 in [0,1], x3 in [0,2], x4 in [0,1/2], x5 in [0,1]
# x4 is cheaper than x2 in E, so x4 = 1/2 (its bound), x2 = 1/4; x1 = 1 (its bound);
# x5 = 1/2 fills L; x3 = 3/4 fills G. Duals y = (1, -1, 1) from the basic columns
# x3, x5, x2; reduced costs r1 = -2 - 1 + 1 = -2 and r4 = 1/2 - 1 = -1/2, both
# negative at an upper bound. OPT = -2 + 1/2 + 3/4 + 1/4 - 1/2 + C0.
OBJ = [-2.0, 2.0, 1.0, 0.5, -1.0]
LB = [0.0] * 5
UB = [1.0, 1.0, 2.0, 0.5, 1.0]
ROWS = [([0, 1, 2], [1.0, 1.0, 1.0], "G", 2.0),
        ([0, 4], [1.0, 1.0], "L", 1.5),
        ([1, 3], [1.0, 1.0], "E", 0.75)]
C0 = 0.125
OPT = Fraction(-1) + Fraction(C0)
Y_EXACT = [1.0, -1.0, 1.0]
X_OPT = [1.0, 0.25, 0.75, 0.5, 0.5]
NAMES = [f"x{k + 1}" for k in range(5)]


def cplex_model(ub=UB):
    import cplex
    c = cplex.Cplex()
    for s in (c.set_log_stream, c.set_results_stream, c.set_warning_stream):
        s(None)
    c.parameters.threads.set(1)
    c.variables.add(obj=OBJ, lb=LB, ub=ub, names=NAMES)
    c.objective.set_offset(C0)
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(i, v) for i, v, _, _ in ROWS],
                             senses=[s for _, _, s, _ in ROWS], rhs=[r for *_, r in ROWS])
    c.solve()
    return c


def gurobi_model(ub=UB):
    import gurobipy as gp
    from gurobipy import GRB
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    x = m.addVars(5, lb=LB, ub=ub, obj=OBJ)
    m.ObjCon = C0
    gs = {"G": GRB.GREATER_EQUAL, "L": GRB.LESS_EQUAL, "E": GRB.EQUAL}
    for i, v, s, r in ROWS:
        m.addLConstr(gp.LinExpr(v, [x[k] for k in i]), gs[s], r)
    m.ModelSense = GRB.MINIMIZE
    m.optimize()
    return m, x


def with_duals(lp, y):
    return sb.LP(lp.A, lp.sense, lp.b, lp.c, lp.c0, lp.lo, lp.hi, y, lp.objective, lp.solver)


def check_known(name, model):
    rec = sb.certify(model, name)
    lp, _ = sb.lp_of(model)
    print(f"{name}: duals {list(lp.y)}, LP objective {lp.objective!r}, safe bound "
          f"{rec['safe_bound']!r}, safe - LP {rec['safe_minus_lp']:.3g}")
    assert abs(lp.objective - float(OPT)) <= 1e-12
    # Sign conventions: G >= 0, L <= 0, and the values worked out by hand.
    assert lp.y[0] >= 0 and lp.y[1] <= 0
    assert np.allclose(lp.y, Y_EXACT, atol=1e-9), lp.y
    assert Fraction(rec["safe_bound"]) <= OPT
    # The exact duals give OPT exactly.
    exact = sb.safe_bound(with_duals(lp, Y_EXACT))
    assert Fraction(exact["safe_bound"]) == OPT and exact["safe_minus_lp"] == 0.0, exact
    # Noisy duals: the bound stays valid, including after clipping.
    rng = np.random.default_rng(1)
    for scale in (1e-12, 1e-6, 1e-2, 1.0):
        for _ in range(200):
            r = sb.safe_bound(with_duals(lp, np.array(Y_EXACT) + scale * rng.standard_normal(3)))
            assert Fraction(r["safe_bound"]) <= OPT, (scale, r)
    # A dual of the wrong sign is clipped, and the bound is still valid.
    r = sb.safe_bound(with_duals(lp, [-0.5, 0.5, 1.0]))
    assert r["duals_clipped"] == 2 and Fraction(r["safe_bound"]) <= OPT, r
    # Negative control: the duals with the opposite sign convention must fail check().
    r = sb.safe_bound(with_duals(lp, -lp.y))
    rejected = False
    try:
        sb.check(r, f"{name} flipped duals")
    except AssertionError:
        rejected = True
    assert rejected, f"flipped duals passed the check: {r}"


def check_infinite(name, build):
    # x3 with no upper bound; duals with y_G > 1 give r3 = 1 - y_G < 0, so -inf.
    ub = list(UB)
    ub[2] = math.inf
    model = build(ub)
    lp, _ = sb.lp_of(model)
    assert math.isinf(lp.hi[2])
    r = sb.safe_bound(with_duals(lp, [1.5, -1.0, 1.0]))
    assert r["safe_infinite"] and r["safe_bound"] is None, r
    rejected = False
    try:
        sb.check(r, name)
    except AssertionError:
        rejected = True
    assert rejected, "check() must reject an infinite bound"
    # x3 <= 2 holds for every optimal point only, not every feasible one, so as an
    # implied bound here it is purely a test of the mechanics: the bound is finite.
    r = sb.safe_bound(with_duals(lp, [1.5, -1.0, 1.0]), implied_hi={2: 2.0})
    assert not r["safe_infinite"] and r["implied_bounds_used"] == 1, r
    print(f"{name}: infinite bound reported as -inf; finite with an implied bound")


def check_optimum(name, model, keys):
    rec = sb.certify(model, name, point=dict(zip(keys, X_OPT)))
    oc = rec["optimum_check"]
    assert rec["rows_cut_off_optimum"] == 0 and oc["max_violation"] == 0.0, oc
    assert Fraction(oc["tree_objective"]) == OPT and rec["safe_le_tree_objective"], rec
    lp, _ = sb.lp_of(model)
    # x3 = 1/2: the G row is short by exactly 1/4; x5 = 3/4: the L row over by 1/4;
    # x4 = 0.6 breaks its bound 1/2 and the E row by 0.1 (as floats, exactly).
    x = list(X_OPT)
    x[2], x[4], x[3] = 0.5, 0.75, 0.6
    tc = sb.tree_check(lp, x, tags=["G", "L", "E"])
    f = tc["families"]
    assert f["G"]["violated"] == 1 and f["G"]["max_violation"] == 0.25, f
    assert f["L"]["violated"] == 1 and f["L"]["max_violation"] == 0.25, f
    assert Fraction(f["E"]["max_violation"]) >= Fraction(0.25 + 0.6) - Fraction(0.75), f
    assert tc["bounds_violated"] == 1 and tc["rows_cut_off_optimum"] == 3, tc
    print(f"{name}: optimum check passes at the optimum and finds the perturbed rows")


def check_refine(name, model):
    """The tight-tolerance re-solve, forced by a negative threshold: same bound."""
    keep = sb.REFINE_REL
    sb.REFINE_REL = -1.0
    rec = sb.certify(model, f"{name} refined")
    sb.REFINE_REL = keep
    assert rec["refined"] and rec["resolved_copy"], rec
    assert Fraction(rec["safe_bound"]) == OPT, rec
    print(f"{name}: tight re-solve gives safe bound {rec['safe_bound']!r}")


def check_ranged():
    import cplex
    c = cplex.Cplex()
    for s in (c.set_log_stream, c.set_results_stream, c.set_warning_stream):
        s(None)
    c.variables.add(obj=[1.0], lb=[0.0], ub=[1.0])
    c.linear_constraints.add(lin_expr=[cplex.SparsePair([0], [1.0])], senses="R",
                             rhs=[0.25], range_values=[0.5])
    c.solve()
    try:
        sb.from_cplex(c)
    except ValueError as e:
        print(f"ranged row rejected: {e}")
        return
    raise AssertionError("a ranged row must be rejected")


def random_lps(trials=20):
    import cplex
    import gurobipy as gp
    from gurobipy import GRB
    rng = np.random.default_rng(7)
    worst = 0.0
    for t in range(trials):
        m, n = int(rng.integers(5, 40)), int(rng.integers(5, 30))
        A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.4)
        x0 = rng.random(n)
        sense = rng.choice(["G", "L", "E"], size=m)
        b = A @ x0 + np.where(sense == "G", -rng.random(m), np.where(sense == "L", rng.random(m), 0.0))
        obj = rng.standard_normal(n)
        lo, hi = np.zeros(n), np.ones(n) * 2
        c = cplex.Cplex()
        for s in (c.set_log_stream, c.set_results_stream, c.set_warning_stream):
            s(None)
        c.parameters.threads.set(1)
        c.variables.add(obj=obj.tolist(), lb=lo.tolist(), ub=hi.tolist())
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(np.flatnonzero(A[i]).tolist(), A[i][A[i] != 0].tolist())
                      for i in range(m)], senses=sense.tolist(), rhs=b.tolist())
        c.solve()
        g = gp.Model()
        g.Params.OutputFlag = 0
        g.Params.Threads = 1
        x = g.addMVar(n, lb=lo, ub=hi, obj=obj)
        gs = {"G": ">", "L": "<", "E": "="}
        for s in "GLE":
            k = sense == s
            if k.any():
                g.addMConstr(A[k], x, gs[s], b[k])
        g.ModelSense = GRB.MINIMIZE
        g.optimize()
        for name, model in (("cplex", c), ("gurobi", g)):
            rec = sb.certify(model, f"random {t} {name}")
            worst = max(worst, abs(rec["safe_minus_lp"]))
            lp, _ = sb.lp_of(model)
            for scale in (1e-6, 1e-1):
                y = lp.y + scale * rng.standard_normal(m)
                r = sb.safe_bound(with_duals(lp, y))
                if not r["safe_infinite"]:
                    assert r["safe_bound"] <= lp.objective + 1e-9 * max(1, abs(lp.objective)), r
    print(f"random LPs: {trials} x 2 solvers, largest |safe - LP| = {worst:.3g}")


def main():
    check_known("cplex", cplex_model())
    check_known("gurobi", gurobi_model()[0])
    check_infinite("cplex", cplex_model)
    check_infinite("gurobi", lambda ub: gurobi_model(ub)[0])
    check_optimum("cplex", cplex_model(), NAMES)
    g, gx = gurobi_model()
    check_optimum("gurobi", g, [gx[k] for k in range(5)])
    check_refine("cplex", cplex_model())
    check_refine("gurobi", gurobi_model()[0])
    check_ranged()
    random_lps()
    print("all safe-bound tests passed")


if __name__ == "__main__":
    main()
