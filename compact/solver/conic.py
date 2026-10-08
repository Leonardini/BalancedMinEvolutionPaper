"""The node relaxation with the manifold imposed exactly, by a conic solver.

The convex manifold constraint sum_{i<j} w_ij log2 w_ij <= 3/2 - n is imposed as the single
exponential-cone constraint sum_{i<j} entr(w_ij) >= (n - 3/2) ln 2, entr(x) = -x ln x,
beside linear rows, through CVXPY and MOSEK's interior-point conic solver on one thread. Rows
are scaled by exact powers of two.

rigorous_bound turns the final solve's dual into a valid lower bound: for multipliers y of the
linear rows (sign-corrected) and mu >= 0 of the manifold row,
    b'y - mu*beta + sum_j min_{l <= w_j <= u} (r_j w_j + mu w_j ln w_j),
r = c - A'y, beta = (3/2 - n) ln 2, is at most the relaxation's optimum; each one-dimensional
minimum is at w_j = exp(-r_j/mu - 1) clipped to [l, u]. It is evaluated in double precision
with explicit bounds on every rounding error (safe_lagrangian_bound), so it holds whatever the
solver's accuracy, provided the arithmetic follows IEEE 754. With BME_BOUND_CHECK=1 it is
also evaluated in 60-digit arithmetic (lagrangian_bound) and the two are compared: the safe
bound must not exceed the 60-digit one.
"""
import math
import os
import sys

import cvxpy as cp
import mpmath
import numpy as np


def log(msg):
    print(msg, file=sys.stderr, flush=True)


mpmath.mp.dps = 60
CONIC_SOLVER = "MOSEK"


def _conic_solve(prob, tol):
    """Solve with MOSEK at feasibility and gap tolerance tol, one thread."""
    prob.solve(solver=cp.MOSEK, mosek_params={
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol, "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol, "MSK_IPAR_NUM_THREADS": 1})


TOL = 1e-10            # the conic solver's feasibility and gap tolerances
# If the conic solver fails at TOL, the solve is repeated at these looser tolerances (counted by
# the callers through the returned tolerance); the Lagrangian bound stays valid at any.
RETRY_TOLS = (1e-9, 1e-8)


def solve(Dv, n, rows, sen, rhs, lo, hi, allow_infeasible=False, manifold=True):
    """One conic solve. With allow_infeasible, an infeasible model returns
    ("infeasible", status) instead of raising."""
    m, N = len(rows), len(Dv)
    A = np.zeros((m, N))
    for r, row in enumerate(rows):
        for j, v in row.items():
            A[r, j] = v
    b = np.array(rhs, dtype=float)
    # Each row is divided by the power of two nearest its largest coefficient (exact in
    # floating point, so the feasible set and the bound are unchanged): some facet rows
    # carry coefficients up to 2^(n-4).
    # The returned A, b and multipliers refer to the scaled rows.
    big = np.abs(A).max(axis=1)
    scale = np.where(big > 0, 2.0 ** -np.round(np.log2(np.where(big > 0, big, 1.0))), 1.0)
    A *= scale[:, None]
    b *= scale
    w = cp.Variable(N)
    eq = [r for r in range(m) if sen[r] == "E"]
    ge = [r for r in range(m) if sen[r] == "G"]
    le = [r for r in range(m) if sen[r] == "L"]
    cons = [A[eq] @ w == b[eq], A[ge] @ w >= b[ge], A[le] @ w <= b[le],
            w >= lo, w <= hi]
    if manifold:
        cons.append(cp.sum(cp.entr(w)) >= (n - 1.5) * math.log(2))
    prob = cp.Problem(cp.Minimize(Dv @ w), cons)
    # A solve that fails, or stops at its iteration limit ("user_limit"), is repeated at the
    # looser tolerances; one that still has no usable status is reported as failed.
    usable = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE)
    for tol in (TOL,) + RETRY_TOLS:
        try:
            _conic_solve(prob, tol)
            if prob.status in usable:
                break
            why = f"status {prob.status}"
        except cp.SolverError:
            why = "solver error"
        if tol == RETRY_TOLS[-1]:
            if allow_infeasible:
                return "failed", None
            raise RuntimeError(f"{CONIC_SOLVER}: {why} at every tolerance")
        log(f"{CONIC_SOLVER}: {why} at tolerance {tol:g}; retrying looser")
    # An inaccurate solve is accepted: the Lagrangian bound computed from its duals is
    # valid for any multipliers, so inaccuracy can only weaken it. It is counted.
    if allow_infeasible and prob.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        return "infeasible", prob.status
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"{CONIC_SOLVER} status {prob.status}")
    # Duals as multipliers of rows written a.w (sense) b, sign-corrected to the cone of
    # each sense: >= 0 for G, <= 0 for L, free for E.
    y = np.zeros(m)
    y[eq] = -np.asarray(cons[0].dual_value)          # cvxpy: L = f + y'(Aw - b) for ==
    y[ge] = np.maximum(np.asarray(cons[1].dual_value), 0.0)
    y[le] = -np.maximum(np.asarray(cons[2].dual_value), 0.0)
    mu = max(float(cons[5].dual_value), 0.0) if manifold else 0.0
    return (np.asarray(w.value), float(prob.value), A, b, y, mu, prob.solver_stats.num_iters,
            prob.status == cp.OPTIMAL_INACCURATE)


def lagrangian_bound(Dv, A, b, y, mu, lo, hi, n):
    """min over the box of the Lagrangian, in 60-digit arithmetic."""
    mp = mpmath.mpf
    r = [mp(float(Dv[j])) - mpmath.fsum(mp(float(A[i, j])) * mp(float(y[i]))
                                        for i in np.nonzero(A[:, j])[0])
         for j in range(len(Dv))]
    beta = (mp(3) / 2 - n) * mpmath.log(2)
    total = mpmath.fsum(mp(float(bi)) * mp(float(yi)) for bi, yi in zip(b, y)) - mp(mu) * beta
    L, U = mp(lo), mp(hi)
    for rj in r:
        def f(x):
            return rj * x + (mp(mu) * x * mpmath.log(x) if mu > 0 else 0)
        cands = [L, U]
        if mu > 0:
            s = mpmath.exp(-rj / mp(mu) - 1)
            if L < s < U:
                cands.append(s)
        total += min(f(x) for x in cands)
    return total


U = 2.0 ** -53   # unit roundoff of double precision


def _gamma(k):
    """Higham's gamma_k = k u / (1 - k u): a sum or dot product of k terms computed in any
    order is within gamma_k times the sum of the terms' absolute values."""
    return k * U / (1.0 - k * U)


def safe_lagrangian_bound(Dv, A, b, y, mu, lo, hi, n):
    """The Lagrangian bound of lagrangian_bound, in double precision, minus a bound on its
    rounding error, so that it is at most the exact value.

    r = Dv - A'y is computed with error at most gamma_{m+2} (|Dv| + |A|'|y|) per entry; that
    bound is doubled to cover its own rounding, and r is lowered by it. Each one-dimensional
    minimum min_{lo <= x <= hi} (r x + mu x ln x) increases with r (its derivative in r is the
    minimiser, >= lo > 0), so evaluating it at the lowered r gives a lower bound. With mu > 0
    it is taken at an endpoint only when the stationary point exp(t), t = -r/mu - 1, lies
    outside [lo, hi] with t beyond ln lo or ln hi by more than its rounding error (plus 1e-6);
    otherwise as the unconstrained minimum -mu exp(t), which is never above the boxed one.
    Each value is lowered by 16 units of roundoff of the magnitudes it is computed from (exp
    and log in numpy are accurate to a few units); the sums by gamma of their length."""
    m, N = A.shape
    g = _gamma(m + 2)
    r = Dv - A.T @ y
    err = 2.0 * g * (np.abs(Dv) + np.abs(A).T @ np.abs(y))
    r_lo = np.nextafter(r - err - 2.0 * U * (np.abs(r) + err), -np.inf)
    if mu > 0:
        t = -r_lo / mu - 1.0
        if not np.all(np.isfinite(t)):
            raise AssertionError("safe bound: non-finite exponent")
        # The stationary point is exp(t). Rounding moves t by at most a few units of
        # roundoff of |t|, so with that much margin (plus 1e-6) the side of the box is decided
        # correctly; within the margin the unconstrained minimum is used.
        tm = 1e-6 + 8.0 * U * np.abs(t)
        below = t < math.log(lo) - tm
        above = t > math.log(hi) + tm
        mid = ~(below | above)
        st = np.exp(np.where(mid, t, 0.0))
        f_lo = r_lo * lo + mu * lo * math.log(lo)
        f_hi = r_lo * hi + mu * hi * math.log(hi)
        vals = np.where(below, f_lo, np.where(above, f_hi, -mu * st))
        margin = np.where(below, np.abs(r_lo * lo) + abs(mu * lo * math.log(lo)),
                          np.where(above, np.abs(r_lo * hi) + abs(mu * hi * math.log(hi)),
                                   (1.0 + np.abs(t)) * mu * st))
    else:
        vals = np.where(r_lo >= 0, r_lo * lo, r_lo * hi)
        margin = np.abs(vals)
    beta = (1.5 - n) * math.log(2.0)
    by = float(np.dot(b, y))
    sv = float(np.sum(vals))
    total = by - mu * beta + sv
    slack = (16.0 * U * float(np.sum(margin)) + 2.0 * _gamma(N) * float(np.sum(np.abs(vals)))
             + 2.0 * _gamma(m) * float(np.sum(np.abs(b * y))) + 16.0 * U * abs(mu * beta)
             + 8.0 * U * (abs(by) + abs(mu * beta) + abs(sv)))
    return float(np.nextafter(total - slack, -np.inf))


BOUND_CHECK = os.environ.get("BME_BOUND_CHECK") == "1"
# With BME_BOUND_CHECK=1: the largest relative amount by which the 60-digit bound exceeded the
# safe one, and the number of comparisons (instrumentation).
bound_check = {"compared": 0, "max_rel_shortfall": 0.0}


def _bound(Dv, A, b, y, mu, lo, hi, n):
    fast = safe_lagrangian_bound(Dv, A, b, y, mu, lo, hi, n)
    if BOUND_CHECK:
        exact = float(lagrangian_bound(Dv, A, b, y, mu, lo, hi, n))
        if fast > exact:
            raise AssertionError(f"safe bound {fast!r} exceeds the 60-digit bound {exact!r}")
        bound_check["compared"] += 1
        bound_check["max_rel_shortfall"] = max(bound_check["max_rel_shortfall"],
                                               (exact - fast) / max(1.0, abs(exact)))
    return fast


def rigorous_bound(Dv, A, b, y, mu, lo, hi, n, sen):
    """The Lagrangian bound of the last solve. Any multipliers of the equality rows give a
    valid bound; cvxpy's sign convention for them is not relied on: both signs are tried
    and the larger bound kept."""
    eq = np.array([s_ == "E" for s_ in sen[:len(y)]])
    y_flip = np.where(eq, -y, y)
    return max(_bound(Dv, A, b, y, mu, lo, hi, n), _bound(Dv, A, b, y_flip, mu, lo, hi, n))
