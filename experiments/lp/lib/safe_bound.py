"""Rigorous ("safe") lower bound on the optimum of a solved LP, from its dual solution,
in exact rational arithmetic (Neumaier and Shcherbina, Math. Program. 99, 2004).

For   min c^T x + c0   s.t.   a_i^T x >= b_i (i in G), <= b_i (i in L), = b_i (i in E),
                              l <= x <= u,
any y with y_i >= 0 on G, y_i <= 0 on L and y_i free on E gives, with r = c - A^T y,

    LB(y) = c0 + b^T y + sum_j min(r_j l_j, r_j u_j)  <=  the LP optimum.

(For feasible x: c^T x = y^T A x + r^T x >= b^T y + sum_j min(r_j l_j, r_j u_j), since
y_i (a_i^T x - b_i) >= 0 row by row.) The solver's duals are clipped to these sign
cones, every number the solver stores (A, b, c, c0, l, u) and the clipped y is taken as
the exact dyadic rational it is, and LB(y) is evaluated exactly with Python integers.
The float reported is LB rounded down. A variable with r_j != 0 whose bound in the
direction of r_j is infinite makes LB = -inf; a caller may give a finite bound that
every feasible point satisfies (an implied bound, with its proof at the call site).

The bound is rigorous for the LP as stored in the solver. Whether that LP is a
relaxation of BME depends on every stored row holding at the optimal tree, which
rounding a real-valued coefficient (a manifold tangent, say) can break. tree_check
evaluates every stored row and bound at the point of a certified optimal tree, exactly,
and reports any row the point violates: such a row cuts off the optimum, and a bound
above L* is then possible. It also checks the safe bound against the tree's objective
value, exactly.

Both solvers use the convention of the formula for a minimisation: CPLEX
(solution.get_dual_values) and Gurobi (Constr.Pi) report y_i >= 0 on >= rows and
y_i <= 0 on <= rows, with reduced costs c - A^T y. tests/test_safe_bound.py checks
this on both, for all three senses.
"""
import math
import sys
import time
from fractions import Fraction

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

# The safe bound is at most the LP objective plus this relative slack (the LP optimum
# the solver reports is itself within its tolerances of the true one) ...
ABOVE_REL = 1e-9
# ... and at least the LP objective minus this relative amount. A sign-convention or
# extraction error shows up as a gap far larger than this.
BELOW_REL = 1e-4
# A safe bound further than this (relative) below the LP objective, or infinite, is
# recomputed from the duals of a copy re-solved at the solvers' tightest tolerances
# (TIGHT_TOL): at the default tolerance 1e-6, many [0, 1] columns with reduced costs
# of order 1e-6 can loosen the bound by far more than the LP's own accuracy.
REFINE_REL = 1e-9
TIGHT_TOL = 1e-9
# Bounds at or beyond these magnitudes are infinite in each solver.
CPLEX_INF = 1e20
GUROBI_INF = 1e100


class LP:
    """One solved LP, minimisation, as plain arrays: A (scipy CSR), sense ('G', 'L',
    'E' per row), b, c, c0, lo, hi (+-inf for infinite bounds), y (the duals as
    reported), objective (the solver's LP objective value)."""

    def __init__(self, A, sense, b, c, c0, lo, hi, y, objective, solver,
                 row_names=None, col_names=None):
        self.A = csr_matrix(A, dtype=np.float64)
        self.sense = np.array(list(sense))   # CPLEX may return the senses as one string
        self.b, self.c = np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64)
        self.c0 = float(c0)
        self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.objective = float(objective)
        self.solver = solver
        self.row_names, self.col_names = row_names, col_names
        m, n = self.A.shape
        for name, v, k in (("sense", self.sense, m), ("b", self.b, m), ("y", self.y, m),
                           ("c", self.c, n), ("lo", self.lo, n), ("hi", self.hi, n)):
            if len(v) != k:
                raise ValueError(f"{name} has length {len(v)}, expected {k}")
        bad = set(self.sense.tolist()) - {"G", "L", "E"}
        if bad:
            raise ValueError(f"unsupported row senses {bad} (ranged rows are not handled)")
        for name, v in (("A", self.A.data), ("b", self.b), ("c", self.c), ("y", self.y)):
            if not np.all(np.isfinite(v)):
                raise ValueError(f"non-finite entries in {name}")


# ---- exact dyadic arithmetic -----------------------------------------------------

def _dyadic(v):
    """Finite float64 array v -> (list of Python ints X, int E) with v_k = X_k 2^E
    exactly. Every float is m 2^e with an integer m of at most 53 bits."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return [], 0
    frac, e = np.frexp(v)
    mant = (frac * 2.0 ** 53).astype(np.int64)        # exact: |frac| < 1, 53 bits
    e = e.astype(np.int64) - 53
    nz = mant != 0
    if not nz.any():
        return [0] * len(v), 0
    # Strip trailing zero bits so that the common exponent is as large as possible.
    low = np.where(nz, mant & -mant, 1)
    tz = np.round(np.log2(low.astype(np.float64))).astype(np.int64)
    assert np.all(low == np.left_shift(np.int64(1), tz)), "trailing-zero count"
    mant = np.where(nz, mant >> tz, 0)
    e = np.where(nz, e + tz, 0)
    E = int(e[nz].min())
    shift = np.where(nz, e - E, 0)
    return [int(m) << int(s) for m, s in zip(mant.tolist(), shift.tolist())], E


def _frac(x, E):
    """The rational x 2^E."""
    return Fraction(x << E) if E >= 0 else Fraction(x, 1 << -E)


def _floor_float(q):
    """The largest float <= the rational q."""
    f = float(q)                      # correctly rounded
    if Fraction(f) > q:
        f = math.nextafter(f, -math.inf)
    assert Fraction(f) <= q
    return f


def _ceil_float(q):
    """The smallest float >= the rational q."""
    f = float(q)
    if Fraction(f) < q:
        f = math.nextafter(f, math.inf)
    assert Fraction(f) >= q
    return f


def _exact_products(M, v):
    """Compressed sparse M (CSR: per row; CSC: per column) with float data and a vector
    v of Python ints (the minor index): returns (sums, E) with
    sum_k M[major, k] v_k = sums[major] 2^E exactly, E the exponent of M's data."""
    Md, Em = _dyadic(M.data)
    vobj = np.empty(len(v), dtype=object)
    vobj[:] = v
    prod = np.empty(len(Md), dtype=object)
    prod[:] = Md
    prod = prod * vobj[M.indices]
    ptr = M.indptr
    out = [0] * (len(ptr) - 1)
    nonempty = np.flatnonzero(ptr[1:] > ptr[:-1])
    if len(nonempty):
        for j, t in zip(nonempty.tolist(), np.add.reduceat(prod, ptr[nonempty]).tolist()):
            out[j] = t
    return out, Em


def safe_bound(lp, implied_lo=None, implied_hi=None):
    """LB(y) for the duals of `lp` clipped to their sign cones, computed exactly.

    implied_lo / implied_hi: {column: value}, finite bounds that every feasible point
    satisfies, used only for columns whose own bound in that direction is infinite.

    Returns a record: safe_bound (float, rounded down; None if -inf), safe_infinite,
    lp_objective, safe_minus_lp (exact difference, as a float), the clipping applied to
    the duals and the size of the LP."""
    t0 = time.time()
    m, n = lp.A.shape
    lo, hi = lp.lo.copy(), lp.hi.copy()
    used_implied = 0
    for bounds, implied, inf in ((lo, implied_lo, -math.inf), (hi, implied_hi, math.inf)):
        for j, v in (implied or {}).items():
            if bounds[j] == inf:
                if not math.isfinite(v):
                    raise ValueError(f"implied bound of column {j} is not finite")
                bounds[j] = v
                used_implied += 1

    # Clip the duals to their sign cones.
    y = lp.y.copy()
    g, l_ = lp.sense == "G", lp.sense == "L"
    y[g] = np.maximum(y[g], 0.0)
    y[l_] = np.minimum(y[l_], 0.0)
    clip = np.abs(y - lp.y)

    Y, Ey = _dyadic(y)
    B, Eb = _dyadic(lp.b)
    C, Ec = _dyadic(lp.c)
    A = csc_matrix(lp.A)
    A.sum_duplicates()
    # A^T y column by column, exactly: S_j 2^(Ea + Ey).
    S, Ea = _exact_products(A, Y)

    # r_j = c_j - (A^T y)_j = R_j 2^Er.
    Eay = Ea + Ey
    Er = min(Ec, Eay)
    R = [(cj << (Ec - Er)) - (sj << (Eay - Er)) for cj, sj in zip(C, S)]

    # sum_j min(r_j lo_j, r_j hi_j): lo where r_j > 0, hi where r_j < 0.
    pos = [j for j in range(n) if R[j] > 0]
    neg = [j for j in range(n) if R[j] < 0]
    infinite = [j for j in pos if not math.isfinite(lo[j])] + \
               [j for j in neg if not math.isfinite(hi[j])]
    rec = dict(lp_objective=lp.objective, solver=lp.solver, rows=m, cols=n, nnz=int(A.nnz),
               duals_clipped=int((clip > 0).sum()), dual_clip_max=float(clip.max(initial=0.0)),
               dual_clip_l1=float(clip.sum()), implied_bounds_used=used_implied)
    if infinite:
        rec.update(safe_bound=None, safe_infinite=True, safe_minus_lp=None,
                   infinite_bound_columns=len(infinite), seconds=time.time() - t0)
        return rec

    Lv, El = _dyadic(np.where(np.isfinite(lo), lo, 0.0))
    Uv, Eu = _dyadic(np.where(np.isfinite(hi), hi, 0.0))
    box_lo = sum(R[j] * Lv[j] for j in pos)
    box_hi = sum(R[j] * Uv[j] for j in neg)
    bty = sum(bi * yi for bi, yi in zip(B, Y))
    LB = (_frac(bty, Eb + Ey) + _frac(box_lo, Er + El) + _frac(box_hi, Er + Eu)
          + Fraction(lp.c0))
    rec.update(safe_bound=_floor_float(LB), safe_infinite=False,
               safe_minus_lp=float(LB - Fraction(lp.objective)), seconds=time.time() - t0)
    return rec


def tree_check(lp, x, tags=None):
    """Every stored row and bound of `lp` evaluated exactly at the point x (finite
    floats, one per column), e.g. a certified optimal tree. The violation of a row is
    b - a.x (G), a.x - b (L) or |a.x - b| (E); a row with a positive violation cuts the
    point off. tags: one family name per row (default: the solver's row names without
    any [index] suffix). Returns per family the rows checked and violated, the largest
    violation (rounded up) and the worst row, plus the totals and the exact objective
    value at x."""
    m, n = lp.A.shape
    x = np.asarray(x, dtype=np.float64)
    if len(x) != n or not np.all(np.isfinite(x)):
        raise ValueError(f"point has {len(x)} entries (finite?), the LP {n} columns")
    if tags is None:
        names = lp.row_names if lp.row_names is not None else [""] * m
        tags = [nm.split("[", 1)[0] or "unnamed" for nm in names]
    if len(tags) != m:
        raise ValueError(f"{len(tags)} row tags for {m} rows")
    X, Ex = _dyadic(x)
    A = csr_matrix(lp.A)
    A.sum_duplicates()
    AX, Ea = _exact_products(A, X)
    B, Eb = _dyadic(lp.b)
    Ev = min(Eb, Ea + Ex)
    V = []
    for s_, ax, b in zip(lp.sense.tolist(), AX, B):
        d = (ax << (Ea + Ex - Ev)) - (b << (Eb - Ev))     # a.x - b
        V.append(-d if s_ == "G" else d if s_ == "L" else abs(d))
    fam = {}
    for i, (t, v) in enumerate(zip(tags, V)):
        f = fam.setdefault(t, dict(rows=0, violated=0, max_violation=0.0, worst_row=None,
                                   _v=None))
        f["rows"] += 1
        if v > 0:
            f["violated"] += 1
        if f["_v"] is None or v > f["_v"]:
            f["_v"], f["worst_row"] = v, i
    for f in fam.values():
        q = _frac(f.pop("_v"), Ev)
        f["max_violation"] = _ceil_float(q) if q > 0 else float(q)
    below = int(np.sum(x < lp.lo))
    above = int(np.sum(x > lp.hi))
    C, Ec = _dyadic(lp.c)
    obj = _frac(sum(c * xx for c, xx in zip(C, X)), Ec + Ex) + Fraction(lp.c0)
    worst = max(fam.items(), key=lambda kv: kv[1]["max_violation"], default=(None, None))
    return dict(rows_checked=m, rows_cut_off_optimum=sum(f["violated"] for f in fam.values()),
                bounds_violated=below + above,
                max_violation=worst[1]["max_violation"] if worst[1] else 0.0,
                worst_family=worst[0], families=fam, objective=obj)


def _log(msg):
    print(f"{time.strftime('%F %T')} {msg}", file=sys.stderr, flush=True)


def check(rec, label):
    """Fail loudly unless the safe bound is finite, at most the LP objective (plus
    ABOVE_REL) and within BELOW_REL of it, relative."""
    obj = rec["lp_objective"]
    scale = abs(obj) if obj != 0 else 1.0
    if rec["safe_infinite"]:
        raise AssertionError(f"{label}: safe bound is -inf ({rec['infinite_bound_columns']} "
                             f"columns with nonzero reduced cost and an infinite bound): {rec}")
    sb = rec["safe_bound"]
    if sb > obj + ABOVE_REL * scale:
        raise AssertionError(f"{label}: safe bound {sb!r} above the LP objective {obj!r}: {rec}")
    if obj - sb > BELOW_REL * scale:
        raise AssertionError(f"{label}: safe bound {sb!r} more than {BELOW_REL} (relative) "
                             f"below the LP objective {obj!r}; sign convention? {rec}")
    return rec


# ---- extraction from the solvers -------------------------------------------------

def from_gurobi(M):
    """The LP of a solved gurobipy model (or of the model a CPLEX-API shim wraps)."""
    from gurobipy import GRB
    if M.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi model not at an optimal LP solution (status {M.Status})")
    if M.IsMIP or M.IsQP or M.IsQCP:
        raise RuntimeError("not an LP (IsMIP/IsQP/IsQCP set)")
    if M.ModelSense != GRB.MINIMIZE:
        raise RuntimeError("not a minimisation")
    cons, vars_ = M.getConstrs(), M.getVars()
    sense = np.array([{"<": "L", ">": "G", "=": "E"}[s] for s in M.getAttr("Sense", cons)])
    lo = np.array(M.getAttr("LB", vars_), dtype=np.float64)
    hi = np.array(M.getAttr("UB", vars_), dtype=np.float64)
    lo[lo <= -GUROBI_INF] = -math.inf
    hi[hi >= GUROBI_INF] = math.inf
    return LP(M.getA(), sense, M.getAttr("RHS", cons), M.getAttr("Obj", vars_), M.ObjCon,
              lo, hi, M.getAttr("Pi", cons), M.ObjVal, "gurobi",
              M.getAttr("ConstrName", cons), M.getAttr("VarName", vars_))


def from_cplex(c):
    """The LP of a CPLEX object whose current solution is an optimal LP solution."""
    if c.get_problem_type() != c.problem_type.LP:
        raise RuntimeError("CPLEX problem type is not LP (no duals)")
    if c.solution.get_status() != c.solution.status.optimal:
        raise RuntimeError(f"CPLEX LP not optimal (status {c.solution.get_status()})")
    return cplex_data(c, with_solution=True)


def cplex_data(c, with_solution=False):
    """The rows, bounds and objective of CPLEX object c as an LP; with_solution also
    reads its duals and objective value (which must exist), otherwise they are left
    empty, e.g. for checking the rows of an LP that has no optimal solution."""
    if c.objective.get_sense() != c.objective.sense.minimize:
        raise RuntimeError("not a minimisation")
    m, n = c.linear_constraints.get_num(), c.variables.get_num()
    rows = c.linear_constraints.get_rows()
    indptr = np.zeros(m + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(r.ind) for r in rows])
    ind = np.fromiter((i for r in rows for i in r.ind), dtype=np.int64, count=indptr[-1])
    val = np.fromiter((v for r in rows for v in r.val), dtype=np.float64, count=indptr[-1])
    A = csr_matrix((val, ind, indptr), shape=(m, n))
    lo = np.array(c.variables.get_lower_bounds(), dtype=np.float64)
    hi = np.array(c.variables.get_upper_bounds(), dtype=np.float64)
    lo[lo <= -CPLEX_INF] = -math.inf
    hi[hi >= CPLEX_INF] = math.inf
    y, obj = ((c.solution.get_dual_values(), c.solution.get_objective_value())
              if with_solution else (np.zeros(m), math.nan))
    return LP(A, c.linear_constraints.get_senses(), c.linear_constraints.get_rhs(),
              c.objective.get_linear(), c.objective.get_offset(), lo, hi, y, obj, "cplex")


def _cplex_lp_copy(c, tight=False):
    """An LP copy of CPLEX object c (same rows, bounds, objective and parameters, no
    time limit), solved. For a model whose duals are unavailable: a MILP-typed model
    with no integer variables left, or one modified after its last solve. With
    `tight`, the simplex tolerances are TIGHT_TOL and the solve starts from c's basis
    when c has a current simplex solution. c itself is not touched."""
    import os
    import tempfile

    import cplex
    cp = cplex.Cplex(c)
    for s in (cp.set_log_stream, cp.set_results_stream, cp.set_warning_stream):
        s(None)
    with tempfile.TemporaryDirectory() as d:
        prm = os.path.join(d, "c.prm")
        c.parameters.write_file(prm)
        cp.parameters.read_file(prm)
    cp.parameters.timelimit.reset()
    if cp.get_problem_type() != cp.problem_type.LP:
        if cp.variables.get_num_integer() + cp.variables.get_num_binary() != 0:
            raise RuntimeError("model has integer variables; it is not an LP")
        cp.set_problem_type(cp.problem_type.LP)
    if tight:
        cp.parameters.simplex.tolerances.optimality.set(TIGHT_TOL)
        cp.parameters.simplex.tolerances.feasibility.set(TIGHT_TOL)
        cp.parameters.emphasis.numerical.set(1)
        cp.parameters.lpmethod.set(cp.parameters.lpmethod.values.dual)
        if c.get_problem_type() == c.problem_type.LP and \
                c.solution.get_status() == c.solution.status.optimal and \
                c.solution.get_method() in (c.solution.method.primal, c.solution.method.dual):
            cols, rows = c.solution.basis.get_basis()
            cp.start.set_start(cols, rows, [], [], [], [])
    cp.solve()
    return cp


def _is_shim(c):
    return hasattr(c, "_m") and hasattr(c, "_cons")


def lp_of(c, tight=False):
    """(LP, resolved) for a solved model: a gurobipy Model, the CPLEX-API gurobipy shim
    of compact/solver (read through the gurobipy model it wraps), or a cplex.Cplex. If the
    model's own solution has no duals (CPLEX MILP type, or rows added since the last
    solve), or with `tight` (re-solve at TIGHT_TOL), a copy is solved instead and
    resolved is True; the caller's model is never modified."""
    import gurobipy as gp
    from gurobipy import GRB
    if isinstance(c, gp.Model) or _is_shim(c):
        M = c if isinstance(c, gp.Model) else c._m
        # Rows added since the last solve are pending until update(), which then
        # discards the solution, so the status below tells whether Pi is current.
        M.update()
        if M.Status == GRB.OPTIMAL and not M.IsMIP and not tight:
            return from_gurobi(M), False
        cp = M.copy()
        cp.Params.Threads = M.Params.Threads
        cp.Params.TimeLimit = GRB.INFINITY
        if tight:
            cp.Params.OptimalityTol = TIGHT_TOL
            cp.Params.FeasibilityTol = TIGHT_TOL
            cp.Params.NumericFocus = 2
            cp.Params.Method = 1          # dual simplex: a basic solution, exact duals
        cp.optimize()
        return from_gurobi(cp), True
    if c.get_problem_type() == c.problem_type.LP and \
            c.solution.get_status() == c.solution.status.optimal and not tight:
        return from_cplex(c), False
    return from_cplex(_cplex_lp_copy(c, tight)), True


def certify(c, label, reported=None, implied_lo=None, implied_hi=None, tree=None,
            point=None, tags=None):
    """The safe bound of solved model c (see lp_of), checked (see check). `reported` is
    the bound the experiment records (default: the objective of c's own LP solution;
    required when c has none); the record has safe_minus_reported, the exact difference
    of the safe bound and it, and safe_minus_lp, against the objective of the LP whose
    duals were used. Implied bounds are given per variable object (gurobipy) or
    per column index (CPLEX).

    point: the certified optimal tree in the model's variables, as {variable: value}
    (gurobipy) or {name: value} (CPLEX, the shim), covering every column; with it the
    record has optimum_check (tree_check), rows_cut_off_optimum and
    safe_le_tree_objective (exact). A point that violates a row is reported (record and
    stderr), not raised. tree: the tree's dict from common.optimal_tree, for its source;
    None with no point means there is no certified tree.

    A bound from the solver's duals that is infinite or more than REFINE_REL below the
    LP objective is recomputed from a copy re-solved at TIGHT_TOL (refined: True; the
    first attempt under from_solver_duals). Both are valid; the second is tighter."""
    lp, resolved = lp_of(c)
    if reported is None:
        if resolved:
            raise ValueError(f"{label}: give the reported bound; the model's own LP has no duals")
        reported = lp.objective

    def cols(d):
        if not d:
            return d
        return {getattr(k, "index", k): v for k, v in d.items()}

    rec = safe_bound(lp, cols(implied_lo), cols(implied_hi))
    rec["resolved_copy"] = resolved
    rec["refined"] = False
    if rec["safe_infinite"] or \
            lp.objective - rec["safe_bound"] > REFINE_REL * (abs(lp.objective) or 1.0):
        first = {k: rec[k] for k in ("safe_bound", "safe_infinite", "lp_objective",
                                     "safe_minus_lp", "duals_clipped", "dual_clip_max",
                                     "resolved_copy", "seconds")}
        t0 = time.time()
        lp, _ = lp_of(c, tight=True)
        rec = safe_bound(lp, cols(implied_lo), cols(implied_hi))
        rec.update(resolved_copy=True, refined=True, from_solver_duals=first,
                   refine_seconds=time.time() - t0)
    rec["reported_bound"] = reported
    rec["safe_minus_reported"] = (None if rec["safe_bound"] is None else
                                  float(Fraction(rec["safe_bound"]) - Fraction(reported)))
    if point is None:
        rec["optimum_check"] = "no certified tree"
        rec["rows_cut_off_optimum"] = None
    else:
        t0 = time.time()
        x = np.full(lp.A.shape[1], np.nan)
        if all(isinstance(k, str) for k in point):
            # CPLEX raises if the model has no names, so they are read only here.
            names = lp.col_names if lp.col_names is not None else c.variables.get_names()
            where = {nm: j for j, nm in enumerate(names)}
            for k, v in point.items():
                x[where[k]] = v
        else:
            for k, v in point.items():
                x[k.index] = v
        if np.isnan(x).any():
            raise ValueError(f"{label}: the tree point leaves {int(np.isnan(x).sum())} "
                             f"columns unassigned")
        tc = tree_check(lp, x, tags)
        tobj = tc.pop("objective")
        tc.update(tree_source=tree["source"] if tree else None, tree_objective=float(tobj),
                  seconds=time.time() - t0)
        rec["optimum_check"] = tc
        rec["rows_cut_off_optimum"] = tc["rows_cut_off_optimum"] + tc["bounds_violated"]
        if rec["safe_bound"] is not None:
            rec["safe_le_tree_objective"] = Fraction(rec["safe_bound"]) <= tobj
        if rec["rows_cut_off_optimum"] or not rec.get("safe_le_tree_objective", True):
            bad = {k: v for k, v in tc["families"].items() if v["violated"]}
            _log(f"WARNING {label}: the optimal tree violates {tc['rows_cut_off_optimum']} "
                 f"rows and {tc['bounds_violated']} bounds of the LP (largest violation "
                 f"{tc['max_violation']:.3g}, family {tc['worst_family']}): {bad}; "
                 f"safe bound <= tree objective: {rec.get('safe_le_tree_objective')}")
    return check(rec, label)


def fields(rec, prefix=""):
    """The headline numbers, for placing next to a bound in a record: the safe bound,
    safe_minus_lp = safe bound - the reported (floating-point) bound, and the number of
    rows and bounds that cut off the optimal tree."""
    return {f"{prefix}safe_bound": rec["safe_bound"],
            f"{prefix}safe_minus_lp": rec["safe_minus_reported"],
            f"{prefix}rows_cut_off_optimum": rec["rows_cut_off_optimum"]}
