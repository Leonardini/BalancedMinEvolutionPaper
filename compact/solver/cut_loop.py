"""Cutting-plane separation for the BME LP relaxation.

``separate_cuts`` runs the separation loop on an existing LP model, adding
violated cuts of the enabled families until none is found.
"""

import math
import time
import warnings

import cplex
from .f45_separator import separate_f4, separate_f5
from .f6_separator import separate_f6
from .f7_separator import f7_rhs, separate_f7
from .paper_cuts import p50_rhs, separate_p50, separate_p53, separate_p54
from .pm_separator import separate_pm
from .w_space import (
    EPS,
    build_W,
    cut_row_sparse,
    first_crossing_pair,
    find_half_cuts,
    global_min_cut,
)

def _mincut_key(S, n):
    """A bipartition's key: the smaller side, or the lexicographically smaller sorted side
    if both have n/2 leaves."""
    S = sorted(S)
    T = sorted(set(range(1, n + 1)) - set(S))
    return tuple(min((len(S), S), (len(T), T))[1])


def _add_cut_mincut(c, mc_set, n, pair_to_idx):
    """Add the min-cut row W[S] >= 1/2 unless the model already holds it; returns whether it
    was added. A cut found violated again although it is in the model is violated within the
    LP's tolerance only: it is not added twice, and a warning says so."""
    keys = getattr(c, "_mincut_keys", None)
    if keys is None:
        keys = c._mincut_keys = set()
    key = _mincut_key(mc_set, n)
    if key in keys:
        warnings.warn(f"min-cut {key} found violated again although it is in the model; "
                      f"not added twice")
        return False
    keys.add(key)
    idx, vals = cut_row_sparse(mc_set, n, pair_to_idx)
    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=idx, val=vals)],
        senses=["G"],
        rhs=[0.5],
    )
    return True


def _add_cut_pm(c, pm_idx, pm_vals, pm_Cn):
    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[int(i) for i in pm_idx],
                                   val=[float(v) for v in pm_vals])],
        senses=["G"],
        rhs=[float(pm_Cn)],
    )


def _add_cut_f6(c, f6_cuts, n):
    coef = 2.0 ** (n - 4)
    for lo_idx, hi_idx in f6_cuts:
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[int(lo_idx), int(hi_idx)],
                                       val=[coef, -1.0])],
            senses=["G"],
            rhs=[0.0],
        )


def _add_cut_f7(c, f7_cuts, rhs):
    for idx_ab, idx_ac, idx_bc, idx_de in f7_cuts:
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[idx_ab, idx_ac, idx_bc, idx_de],
                val=[1.0, 1.0, 1.0, 1.0])],
            senses=["G"],
            rhs=[rhs],
        )


def _add_cut_f4(c, f4_cuts, coef):
    for idx_ab, idx_ac, idx_bc in f4_cuts:
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[idx_ab, idx_ac, idx_bc],
                val=[coef, coef, 1.0])],
            senses=["G"],
            rhs=[5.0 / 16.0],
        )


def _add_cut_f5(c, f5_cuts, coef):
    for idx_ab, idx_ac, idx_bc in f5_cuts:
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[idx_ab, idx_ac, idx_bc],
                val=[coef, coef, 1.0])],
            senses=["G"],
            rhs=[0.5],
        )


def _add_cut_p50(c, cuts, rhs):
    for idx_ij, idx_ik, idx_jk in cuts:
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[idx_ij, idx_ik, idx_jk],
                                       val=[1.0, 1.0, 1.0])],
            senses=["G"],
            rhs=[rhs],
        )


def _add_cut_p53(c, cuts, n):
    rhs = 2.0 ** (4 - n)
    for idx2a, idx2b, idx1c in cuts:
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[idx2a, idx2b, idx1c],
                                       val=[2.0, 2.0, 1.0])],
            senses=["G"],
            rhs=[rhs],
        )


def _add_cut_p54(c, cuts, n):
    coef = 2.0 ** (n - 4)
    for idx_ab, idx_cd, idx_ef in cuts:
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[idx_ab, idx_cd, idx_ef],
                                       val=[1.0, 1.0, -coef])],
            senses=["L"],
            rhs=[0.125],
        )


_LN2 = math.log(2.0)


def _manifold_cut(w_vals, n, np_pairs, tol=1e-6):
    """Tangent (gradient) cut for the phylogenetic-manifold constraint, eqn (9).

    Every tree satisfies sum_{i<j} w_ij log2(w_ij) = 3/2 - n, and since the LHS
    is convex, Conv(X) satisfies the convex inequality <= 3/2 - n.  We separate
    it by its tangent plane at the current LP optimum w* (Kelley's method):

        sum a_ij w_ij <= b,   a_ij = log2(w*_ij) + 1/ln2,
                              b = (3/2 - n) + (1/ln2) sum_{i<j} w*_ij.

    Returns (indices, coeffs, rhs) if w* violates the manifold, else None.
    """
    V = 0.0
    for wv in w_vals:
        if wv > 0.0:
            V += wv * math.log2(wv)
    target = 1.5 - n
    if V <= target + tol:
        return None
    # The tangent is taken at wbar = max(w*, 2^-(n-1)): a tangent of the convex
    # function at any positive point is a valid inequality, whereas a zero entry of w*
    # has no finite slope. Every tree has w >= 2^-(n-1), so when that bound is in the
    # model (it always is) wbar = w* and the cut is the usual tangent at w*.
    floor = 2.0 ** (-(n - 1))
    wbar = [max(wv, floor) for wv in w_vals]
    idx = list(range(np_pairs))
    a = [math.log2(wb) + 1.0 / _LN2 for wb in wbar]
    b = target + sum(wbar) / _LN2
    return idx, a, b


def _add_cut_manifold(c, idx, a, b):
    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=idx, val=a)], senses=["L"], rhs=[b])


def _add_cut_crossing(c, A, B, n, pair_to_idx):
    a_idx, a_vals = cut_row_sparse(A, n, pair_to_idx)
    b_idx, b_vals = cut_row_sparse(B, n, pair_to_idx)
    combined = {}
    for i, v in zip(a_idx, a_vals):
        combined[i] = combined.get(i, 0.0) + v
    for i, v in zip(b_idx, b_vals):
        combined[i] = combined.get(i, 0.0) + v
    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair(
            ind=list(combined.keys()), val=list(combined.values()))],
        senses=["G"],
        rhs=[1.25],
    )


def separate_cuts(c, D, pair_to_idx, pairs, np_pairs, cut_counts,
                  max_iter=5000, deadline=None, manifold_cap=0,
                  row_tags=None, disable=None, lean=True):
    """Run the cutting-plane separation loop on model *c* until convergence.

    Assumes z is ALREADY relaxed to continuous and does NOT restore it — the
    caller owns z's type.  All separated cuts (min-cut / PM / F6 / F7 / crossing)
    are added as permanent constraints and RETAINED in *c*; they are globally
    valid for every BME tree, so a persistent model accumulates them across
    many solves (used by the branch-and-bound driver in ``bnb.py``).

    ``cut_counts`` is a dict that is mutated in place (keys incremented), so a
    caller can accumulate totals across many node solves.

    ``deadline`` (absolute time.time() value) bounds the loop so a single node
    solve cannot overrun the caller's time budget; the latest valid LP bound is
    returned when it is hit.  The bound is valid regardless of early exit.

    Returns
    -------
    (lp_bound, w_vals, n_iter, decided)
      lp_bound : final LP objective (nan if no optimal solve was obtained)
      w_vals   : final LP w-vector (None if no optimal solve was obtained)
      n_iter   : number of cuts added in this call
      decided  : True iff the node's status is known — either a valid LP bound
                 was obtained or the node was PROVEN infeasible.  False means a
                 solve aborted (per-solve cap) before any optimal bound, so the
                 node's true status is unknown and the caller must not certify.
    """
    n = D.shape[0]
    n_iter = 0
    lp_bound = float('nan')
    w_vals = None
    _prev_lp = float('-inf')
    _f7_rhs = f7_rhs(n)
    _f4_coef = 2.0 ** (n - 5)
    _f5_coef = 2.0 ** (n - 4)
    _p50_rhs = p50_rhs(n)
    _manifold_added = 0

    # Analytics / configuration hooks.  ``disable`` is a set of OPTIONAL family
    # names to skip separating (for leave-one-family-out bound ablation; min-cut
    # is never skipped — it underpins the structural convergence check).
    # ``row_tags``, if a list, records the family of every constraint row added,
    # in order, so the caller can attribute the converged bound by family.
    # ``lean`` (default ON) selects the default cut set: min-cut (with its crossing
    # variant), PM and the manifold cut. lean=False also separates the facet
    # families F4-F7 and inequalities (50), (53), (54) of Catanzaro et al. (2020).
    disable = set(disable) if disable else set()
    if lean:
        disable |= {'f6', 'f7', 'f4', 'f5', 'p50', 'p53', 'p54'}

    def _t(fam, k):                       # count + (optionally) tag k new rows
        cut_counts[fam] += k
        if row_tags is not None:
            row_tags.extend([fam] * k)

    for _ in range(max_iter):
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                return lp_bound, w_vals, n_iter, not math.isnan(lp_bound)
            # An LP solve runs to optimality; only the run's own deadline stops it.
            c.parameters.timelimit.set(max(0.01, remaining))
        t_lp = time.perf_counter()
        c.solve()
        # Time inside CPLEX's LP solves (instrumentation; read by the node record).
        cut_counts['lp_seconds'] = cut_counts.get('lp_seconds', 0.0) + time.perf_counter() - t_lp
        if not c.solution.is_primal_feasible():
            # The only reason to treat a non-feasible solve as UNDECIDED is a
            # time-limit abort (status 11 wall-clock / 25 deterministic) before
            # any bound was obtained.  Every other non-feasible status means the
            # node is genuinely infeasible (3 = infeasible, 4 = infeasible-or-
            # unbounded, which here is infeasible since the objective is bounded)
            # — safe to prune.  Using status==3 alone was wrong: dual simplex +
            # presolve often reports an infeasible child LP as status 4.
            st = c.solution.get_status()
            aborted_no_bound = (st in (11, 25)) and math.isnan(lp_bound)
            return lp_bound, w_vals, n_iter, not aborted_no_bound
        # Only trust the objective of a solve that proved optimality.  With dual
        # simplex an aborted solve is primal-infeasible (handled above); a
        # non-optimal status here means the run's deadline was hit, so fall back to
        # the last optimal bound/w — a valid, if looser, lower bound (adding cuts only
        # raises the LP optimum).
        status = c.solution.get_status()
        if deadline is not None and status not in (1, 101):  # 1/101 = optimal
            return lp_bound, w_vals, n_iter, not math.isnan(lp_bound)

        lp_bound = c.solution.get_objective_value()
        # Relative convergence: stop when LP objective stops improving.
        # Use max(abs, rel) so the threshold scales with both the bound magnitude
        # (handles instances with tiny BME objectives) and a floor (handles
        # numerical noise when the bound is near zero).
        eps_convergence = max(EPS, 1e-6 * abs(lp_bound))
        if lp_bound <= _prev_lp + eps_convergence:
            # Bound has converged.  Before accepting, verify structural integrity:
            # F6/F7 cuts can shift the LP optimum to a new vertex that violates
            # either a bipartition (min-cut < 1/2) or the laminarity of the
            # tight cuts (a crossing pair of ≈1/2 bipartitions).  Both conditions
            # are required for the LP optimal w to be tree-realizable.
            # Reset _prev_lp to -inf so the convergence check doesn't immediately
            # re-fire if any structural repair cut is added.
            w_vals_chk = c.solution.get_values(list(range(np_pairs)))
            W_chk = build_W(w_vals_chk, n)

            mc_val_chk, mc_set_chk = global_min_cut(W_chk, n)
            if mc_val_chk < 0.5 - EPS and _add_cut_mincut(c, mc_set_chk, n, pair_to_idx):
                _t('mincut', 1)
                n_iter += 1
                _prev_lp = float('-inf')
                continue

            # Laminarity guard: the tight (≈1/2) bipartitions must form a
            # laminar family for the LP optimum to be tree-realizable.  They are the
            # minimum cuts, listed in polynomial time (find_half_cuts).
            half_chk = find_half_cuts(W_chk, n)
            cp_chk = first_crossing_pair(half_chk)
            if cp_chk is not None:
                _add_cut_crossing(c, cp_chk[0], cp_chk[1], n, pair_to_idx)
                _t('crossing', 1)
                n_iter += 1
                _prev_lp = float('-inf')
                continue

            # Phylogenetic-manifold tangent cut, eqn (9).  Added once the
            # polyhedral facets are exhausted (here, at the convergence point),
            # so it actually fires instead of being starved by min-cut.  Capped
            # per call to avoid the slow Kelley tail; _prev_lp is reset so the
            # polyhedral separators get a fresh shot at the shifted optimum.
            if _manifold_added < manifold_cap:
                mc = _manifold_cut(w_vals_chk, n, np_pairs)
                if mc is not None:
                    _add_cut_manifold(c, mc[0], mc[1], mc[2])
                    _t('manifold', 1)
                    _manifold_added += 1
                    n_iter += 1
                    _prev_lp = float('-inf')
                    continue

            # LP bound stable, all bipartitions satisfied, 1/2-cuts laminar.
            w_vals = w_vals_chk
            return lp_bound, w_vals, n_iter, True
        _prev_lp = lp_bound
        w_vals = c.solution.get_values(list(range(np_pairs)))
        W = build_W(w_vals, n)

        mc_val, mc_set = global_min_cut(W, n)
        if mc_val < 0.5 - EPS and _add_cut_mincut(c, mc_set, n, pair_to_idx):
            _t('mincut', 1)
            n_iter += 1
            continue

        if 'pm' not in disable:
            _, pm_idx, pm_vals, pm_Cn = separate_pm(w_vals, pairs, pair_to_idx, n)
            if pm_idx is not None:
                _add_cut_pm(c, pm_idx, pm_vals, pm_Cn)
                _t('pm', 1)
                n_iter += 1
                continue

        if 'f6' not in disable:
            f6_cuts = separate_f6(w_vals, pairs, n)
            if f6_cuts:
                _add_cut_f6(c, f6_cuts, n)
                _t('f6', len(f6_cuts))
                n_iter += 1
                continue

        if 'f7' not in disable:
            f7_cuts = separate_f7(w_vals, pairs, pair_to_idx, n)
            if f7_cuts:
                _add_cut_f7(c, f7_cuts, _f7_rhs)
                _t('f7', len(f7_cuts))
                n_iter += 1
                continue

        # F4 (48) and F5 (49): proven valid for all n>=6 (Catanzaro et al. 2020).
        if 'f4' not in disable:
            f4_cuts = separate_f4(w_vals, pairs, pair_to_idx, n)
            if f4_cuts:
                _add_cut_f4(c, f4_cuts, _f4_coef)
                _t('f4', len(f4_cuts))
                n_iter += 1
                continue

        if 'f5' not in disable:
            f5_cuts = separate_f5(w_vals, pairs, pair_to_idx, n)
            if f5_cuts:
                _add_cut_f5(c, f5_cuts, _f5_coef)
                _t('f5', len(f5_cuts))
                n_iter += 1
                continue

        # (50) triangle lower bound (valid, not facet).
        if 'p50' not in disable:
            p50_cuts = separate_p50(w_vals, pairs, pair_to_idx, n)
            if p50_cuts:
                _add_cut_p50(c, p50_cuts, _p50_rhs)
                _t('p50', len(p50_cuts))
                n_iter += 1
                continue

        # (53) and (54): two disjoint plus-pairs with a third pair.
        if 'p53' not in disable:
            p53_cuts = separate_p53(w_vals, pairs, pair_to_idx, n)
            if p53_cuts:
                _add_cut_p53(c, p53_cuts, n)
                _t('p53', len(p53_cuts))
                n_iter += 1
                continue

        if 'p54' not in disable:
            p54_cuts = separate_p54(w_vals, pairs, pair_to_idx, n)
            if p54_cuts:
                _add_cut_p54(c, p54_cuts, n)
                _t('p54', len(p54_cuts))
                n_iter += 1
                continue

        # Crossing cut: two half-cuts that cross (find_half_cuts lists them in polynomial time).
        if 'crossing' not in disable:
            half = find_half_cuts(W, n)
            cp = first_crossing_pair(half)
            if cp is not None:
                _add_cut_crossing(c, cp[0], cp[1], n, pair_to_idx)
                _t('crossing', 1)
                n_iter += 1
                continue

        # Manifold (eqn 9) — separate it HERE, at the cascade end, so the loop
        # cannot exit while the manifold is still violated.  The bound-convergence
        # block above kicks the first tangent off, but resetting _prev_lp routes
        # the next pass through the polyhedral cascade, which would otherwise
        # `return` below (manifold still violated) the moment those cuts are clean.
        # Re-checking here drives the manifold to its OWN convergence (g <= 3/2-n),
        # interleaved with a polyhedral re-check after each tangent.
        if _manifold_added < manifold_cap:
            mc = _manifold_cut(w_vals, n, np_pairs)
            if mc is not None:
                _add_cut_manifold(c, mc[0], mc[1], mc[2])
                _t('manifold', 1)
                _manifold_added += 1
                _prev_lp = float('-inf')
                n_iter += 1
                continue

        # No violated cut — converged.
        return lp_bound, w_vals, n_iter, True

    return lp_bound, w_vals, n_iter, True


