"""The compact root relaxation with the manifold imposed exactly, by a conic solver.

    BME_THREADS=1 python conic_manifold.py INSTANCE OUT.json

The compact solver imposes the convex manifold constraint
    sum_{i<j} w_ij log2 w_ij <= 3/2 - n
through Kelley tangent cuts, many nearly parallel planes. Here it is one constraint,
    sum_{i<j} entr(w_ij) >= (n - 3/2) ln 2,      entr(x) = -x ln x,
which is exponential-cone representable, solved with CVXPY and an interior-point conic
solver, MOSEK, on one thread. Everything else is the solver's root relaxation: Kraft, 2^-(n-1) <= w <= 1/4,
the double cherry, and the cuts its default loop separates (min-cut W[S] >= 1/2, perfect
matching, and the crossing cut W[A] + W[B] >= 5/4 for crossing 1/2-cuts), separated at
each conic optimum with the solver's own routines, one round at a time, until none is
violated.

Bounds recorded:
  primal      the conic solver's objective at the final solve;
  lagrangian  a rigorous lower bound from the final solve's dual: for multipliers y of
              the linear rows (sign-corrected) and mu >= 0 of the manifold row,
                  b'y - mu*beta + sum_j min_{l <= w_j <= u} (r_j w_j + mu w_j ln w_j),
              r = c - A'y, beta = (3/2 - n) ln 2, is at most the relaxation's optimum;
              each one-dimensional minimum is at w_j = exp(-r_j/mu - 1) clipped to [l, u].
              Evaluated in 60-digit arithmetic from the double-precision data.
Gaps are relative to the certified optimum (data/ground_truth), and every row, and the
manifold constraint, is evaluated at the optimal tree.
"""
import math
import sys
import time
from pathlib import Path

import cvxpy as cp
import mpmath
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "compact"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import log, optimal_tree, parse_matrix, peak_rss_mb, write_json  # noqa: E402
from solver.pm_separator import separate_pm  # noqa: E402
from solver.w_space import (EPS, build_W, cut_row_sparse, find_half_cuts,  # noqa: E402
                            first_crossing_pair, global_min_cut)

from solver.conic import (CONIC_SOLVER, RETRY_TOLS, TOL, _conic_solve,  # noqa: E402
                          rigorous_bound, solve)

MAX_ROUNDS = 5000


def base_rows(n, pti):
    """(rows, senses, rhs) of Kraft and the double cherry, rows as {index: coef}."""
    rows, sen, rhs = [], [], []
    for i in range(1, n + 1):
        rows.append({pti[(min(i, j), max(i, j))]: 1.0 for j in range(1, n + 1) if j != i})
        sen.append("E"), rhs.append(0.5)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                for apex, j, k in ((a, b, c), (b, a, c), (c, a, b)):
                    rows.append({pti[(min(apex, j), max(apex, j))]: 1.0,
                                 pti[(min(apex, k), max(apex, k))]: 1.0,
                                 pti[(min(j, k), max(j, k))]: -1.0})
                    sen.append("L"), rhs.append(0.25)
    return rows, sen, rhs


def conic_root(D, tree):
    """The converged conic root relaxation of D; tree (common.optimal_tree) or None."""
    n = D.shape[0]
    pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    pti = {p: k for k, p in enumerate(pairs)}
    Dv = np.array([D[i - 1, j - 1] for i, j in pairs], dtype=float)
    lo, hi = 2.0 ** -(n - 1), 0.25
    rows, sen, rhs = base_rows(n, pti)
    counts = dict(mincut=0, pm=0, crossing=0)
    n_inaccurate = 0
    t0 = time.time()
    rec = dict(n=n, solver=f"{CONIC_SOLVER} via CVXPY", tolerance=TOL,
               optimum=None if tree is None else float(tree["value"]), history=[])
    for rnd in range(1, MAX_ROUNDS + 1):
        w, val, A, b, y, mu, iters, inaccurate = solve(Dv, n, rows, sen, rhs, lo, hi)
        n_inaccurate += inaccurate
        W = build_W(list(w), n)
        added = None
        mc_val, mc_set = global_min_cut(W, n)
        if mc_val < 0.5 - EPS:
            idx, vals = cut_row_sparse(mc_set, n, pti)
            rows.append(dict(zip(idx, vals))), sen.append("G"), rhs.append(0.5)
            added = "mincut"
        else:
            _, pm_idx, pm_vals, pm_Cn = separate_pm(list(w), pairs, pti, n)
            if pm_idx is not None:
                rows.append({int(i): float(v) for i, v in zip(pm_idx, pm_vals)})
                sen.append("G"), rhs.append(float(pm_Cn))
                added = "pm"
            else:
                pair = first_crossing_pair(find_half_cuts(W, n))
                if pair is not None:
                    row = {}
                    for S in pair:
                        for i, v in zip(*cut_row_sparse(S, n, pti)):
                            row[i] = row.get(i, 0.0) + v
                    rows.append(row), sen.append("G"), rhs.append(1.25)
                    added = "crossing"
        if added:
            counts[added] += 1
        if rnd % 25 == 0 or not added:
            log(f"round {rnd}: bound {val:.12f}, cuts {counts}, {iters} IPM iterations, "
                f"{time.time() - t0:.0f}s")
            rec["history"].append(dict(round=rnd, bound=val, seconds=time.time() - t0))
        if not added:
            break
    else:
        raise RuntimeError(f"no convergence in {MAX_ROUNDS} rounds")
    lag = rigorous_bound(Dv, A, b, y, mu, lo, hi, n, sen)
    rec.update(rounds=rnd, inaccurate_solves=n_inaccurate, final_solve_inaccurate=bool(inaccurate),
               cuts=counts, rows=len(rows), primal_bound=val,
               lagrangian_bound=float(lag), lagrangian_minus_primal=float(lag - mpmath.mpf(val)),
               manifold_multiplier=mu, seconds=time.time() - t0)
    if tree is not None:
        L = float(tree["value"])
        rec["root_gap_primal"] = (L - val) / L
        rec["root_gap_lagrangian"] = (L - float(lag)) / L
        wt = np.array([2.0 ** -int(tree["tau"][i - 1, j - 1]) for i, j in pairs])
        viol = 0
        for row, s_, r_ in zip(rows, sen, rhs):
            ax = sum(v * wt[j] for j, v in row.items())
            viol += (s_ == "G" and ax < r_ - 1e-12) or (s_ == "L" and ax > r_ + 1e-12) or \
                    (s_ == "E" and abs(ax - r_) > 1e-12)
        rec["rows_cut_off_optimum"] = int(viol)
        rec["manifold_at_optimum"] = float(sum(x * math.log2(x) for x in wt)) - (1.5 - n)
    return rec


def solve_vars(cvec, n_w, n, rows, sen, rhs, lo, hi, allow_infeasible=False, manifold=True):
    """As solve(), for a model whose first n_w variables are w (the manifold constraint is
    on them) followed by other variables; lo and hi are per-variable arrays. Rows are
    scaled by powers of two in the same way."""
    m, N = len(rows), len(cvec)
    ri, ci, vv = [], [], []
    for r, row in enumerate(rows):
        for j, val in row.items():
            ri.append(r), ci.append(j), vv.append(val)
    A = sp.csr_matrix((vv, (ri, ci)), shape=(m, N))
    b = np.array(rhs, dtype=float)
    big = np.asarray(abs(A).max(axis=1).todense()).ravel()
    scale = np.where(big > 0, 2.0 ** -np.round(np.log2(np.where(big > 0, big, 1.0))), 1.0)
    A = sp.csr_matrix(sp.diags(scale) @ A)
    b *= scale
    v = cp.Variable(N)
    eq = [r for r in range(m) if sen[r] == "E"]
    ge = [r for r in range(m) if sen[r] == "G"]
    le = [r for r in range(m) if sen[r] == "L"]
    cons = [A[eq] @ v == b[eq], A[ge] @ v >= b[ge], A[le] @ v <= b[le], v >= lo, v <= hi]
    if manifold:
        cons.append(cp.sum(cp.entr(v[:n_w])) >= (n - 1.5) * math.log(2))
    prob = cp.Problem(cp.Minimize(cvec @ v), cons)
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
            raise cp.SolverError(f"{CONIC_SOLVER}: {why} at every tolerance")
        log(f"{CONIC_SOLVER}: {why} at tolerance {tol:g}; retrying looser")
    if allow_infeasible and prob.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        return "infeasible", prob.status
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"{CONIC_SOLVER} status {prob.status}")
    y = np.zeros(m)
    y[eq] = -np.asarray(cons[0].dual_value)
    y[ge] = np.maximum(np.asarray(cons[1].dual_value), 0.0)
    y[le] = -np.maximum(np.asarray(cons[2].dual_value), 0.0)
    mu = max(float(cons[5].dual_value), 0.0) if manifold else 0.0
    return (np.asarray(v.value), float(prob.value), A, b, y, mu, prob.solver_stats.num_iters,
            prob.status == cp.OPTIMAL_INACCURATE)


def rigorous_bound_vars(cvec, A, b, y, mu, lo, hi, n_w, sen):
    """The Lagrangian bound for solve_vars: the w coordinates carry mu * w ln w, the others
    are linear; both signs of the equality multipliers are tried."""
    mp = mpmath.mpf
    eq = np.array([s_ == "E" for s_ in sen[:len(y)]])
    best = None
    Ac = sp.csc_matrix(A)
    for yy in (y, np.where(eq, -y, y)):
        r = [mp(float(cvec[j])) - mpmath.fsum(
                 mp(float(a)) * mp(float(yy[i]))
                 for i, a in zip(Ac.indices[Ac.indptr[j]:Ac.indptr[j + 1]],
                                 Ac.data[Ac.indptr[j]:Ac.indptr[j + 1]]))
             for j in range(len(cvec))]
        beta = (mp(3) / 2 - n_w_to_n(n_w)) * mpmath.log(2)
        total = mpmath.fsum(mp(float(bi)) * mp(float(yi)) for bi, yi in zip(b, yy)) - mp(mu) * beta
        for j, rj in enumerate(r):
            Lj, Uj = mp(float(lo[j])), mp(float(hi[j]))
            if j < n_w and mu > 0:
                def f(x):
                    return rj * x + mp(mu) * x * mpmath.log(x)
                cands = [Lj, Uj]
                s_ = mpmath.exp(-rj / mp(mu) - 1)
                if Lj < s_ < Uj:
                    cands.append(s_)
                total += min(f(x) for x in cands)
            else:
                total += min(rj * Lj, rj * Uj)
        best = total if best is None else max(best, total)
    return best


def n_w_to_n(n_w):
    """The number of leaves n from the number of pairs n(n-1)/2."""
    n = int(round((1 + math.sqrt(1 + 8 * n_w)) / 2))
    assert n * (n - 1) // 2 == n_w
    return n


def main():
    inst, out = sys.argv[1], Path(sys.argv[2])
    D = parse_matrix(inst)
    rec = dict(instance=inst, **conic_root(D, optimal_tree(D, instance=inst)))
    rec["peak_rss_mb"] = peak_rss_mb()
    write_json(out, rec)
    log(f"done: primal {rec['primal_bound']:.12f}, Lagrangian {rec['lagrangian_bound']:.12f}, "
        f"gap {rec.get('root_gap_lagrangian', float('nan')):.4%}, {rec['rounds']} rounds")


if __name__ == "__main__":
    main()
