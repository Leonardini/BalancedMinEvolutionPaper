"""The node bound with the manifold constraint imposed exactly.

The node's LP (no manifold tangents) is first taken through the usual cut loop by CPLEX.
Its rows (base, cut pool and the node's own) are then solved once more by the conic solver
with the manifold as one exponential-cone constraint, by MOSEK, in one task kept for the
whole search, to which new cuts are appended and in which the node's rows are swapped
(conic_persistent.py). The
conic optimum is checked by the same separators (min-cut, perfect matching, crossing half-cuts). A violated
cut is added to the CPLEX model, as a cut found by the LP loop would be, and the LP loop and
the conic solve are repeated. The node's bound is the larger of the LP bound and the rigorous
Lagrangian bound of the last conic solve; both are valid. The point returned for branching
and tree recognition is the conic optimum.

The conic solver is never trusted to prune: if it fails, or reports the node infeasible
(a claim an interior-point method makes without an exact certificate), the node keeps its LP
bound and LP point, and the event is counted.
"""
import os
import time

import numpy as np

from . import conic
from .conic_persistent import PersistentConic
from .cut_loop import _add_cut_crossing, _add_cut_mincut, _add_cut_pm, separate_cuts
from .pm_separator import separate_pm
from .w_space import EPS, build_W, find_half_cuts, first_crossing_pair, global_min_cut

COUNT_KEYS = ("conic_solves", "conic_inaccurate", "conic_failed", "conic_infeasible_claim",
              "conic_cuts_mincut", "conic_cuts_pm", "conic_cuts_crossing", "conic_raised_bound", "conic_seconds")


def new_counts():
    return {k: 0 for k in COUNT_KEYS}


def _rows(c, np_pairs):
    rows = []
    for sp in c.linear_constraints.get_rows():
        row = {i: v for i, v in zip(sp.ind, sp.val) if i < np_pairs}
        if len(row) != len(sp.ind):
            raise AssertionError("a row of the LP involves the dummy variable")
        rows.append(row)
    return rows, list(c.linear_constraints.get_senses()), list(c.linear_constraints.get_rhs())


# With MOSEK, the conic model is kept in one task for the whole search (conic_persistent);
# BME_CONIC_CHECK=1 also solves it through CVXPY (conic.solve) at every solve and compares.
CONIC_CHECK = os.environ.get("BME_CONIC_CHECK") == "1"
conic_check = {"compared": 0, "max_rel_objective_diff": 0.0, "max_rel_bound_diff": 0.0}


def exact_refine(c, D, pair_to_idx, pairs, np_pairs, cut_counts, lp_bound, w_lp, deadline,
                 counts, node_block=(0, 0)):
    """(bound, w) of the node with the manifold exact; c holds the node's rows, at
    node_block = (start, count) of its linear constraints."""
    n = D.shape[0]
    Dv = np.array([D[i - 1, j - 1] for i, j in pairs], dtype=float)
    lo, hi = 2.0 ** -(n - 1), 0.25
    if getattr(c, "_conic_model", None) is None:
        c._conic_model = PersistentConic(Dv, n, lo, hi)
    while True:
        t_conic = time.perf_counter()
        out = c._conic_model.solve(c, node_block[0], node_block[1])
        if not isinstance(out[0], str):
            *out, sen = out
        counts["conic_seconds"] += time.perf_counter() - t_conic
        counts["conic_solves"] += 1
        if CONIC_CHECK:
            rows, sen_c, rhs = _rows(c, np_pairs)
            ref = conic.solve(Dv, n, rows, sen_c, rhs, lo, hi, allow_infeasible=True)
            if isinstance(ref[0], str) != isinstance(out[0], str):
                raise AssertionError(f"persistent conic model: {out[0] if isinstance(out[0], str) else 'solved'}"
                                     f", CVXPY: {ref[0] if isinstance(ref[0], str) else 'solved'}")
            if not isinstance(ref[0], str):
                scale = max(1.0, abs(ref[1]))
                conic_check["compared"] += 1
                conic_check["max_rel_objective_diff"] = max(conic_check["max_rel_objective_diff"],
                                                            abs(out[1] - ref[1]) / scale)
                b_p = float(conic.rigorous_bound(Dv, out[2], out[3], out[4], out[5], lo, hi, n, sen))
                b_r = float(conic.rigorous_bound(Dv, ref[2], ref[3], ref[4], ref[5], lo, hi, n, sen_c))
                conic_check["max_rel_bound_diff"] = max(conic_check["max_rel_bound_diff"],
                                                        abs(b_p - b_r) / scale)
        if isinstance(out[0], str):
            counts["conic_failed" if out[0] == "failed" else "conic_infeasible_claim"] += 1
            return lp_bound, w_lp
        w, _val, A, b, y, mu, _iters, inaccurate = out
        counts["conic_inaccurate"] += inaccurate
        bound = float(conic.rigorous_bound(Dv, A, b, y, mu, lo, hi, n, sen))
        W = build_W(list(w), n)
        added = False
        mc_val, mc_set = global_min_cut(W, n)
        if mc_val < 0.5 - EPS and _add_cut_mincut(c, mc_set, n, pair_to_idx):
            cut_counts["mincut"] += 1
            counts["conic_cuts_mincut"] += 1
            added = True
        else:
            _, pm_idx, pm_vals, pm_Cn = separate_pm(list(w), pairs, pair_to_idx, n)
            if pm_idx is not None:
                _add_cut_pm(c, pm_idx, pm_vals, pm_Cn)
                cut_counts["pm"] += 1
                counts["conic_cuts_pm"] += 1
                added = True
            else:
                pair = first_crossing_pair(find_half_cuts(W, n))
                if pair is not None:
                    _add_cut_crossing(c, pair[0], pair[1], n, pair_to_idx)
                    cut_counts["crossing"] += 1
                    counts["conic_cuts_crossing"] += 1
                    added = True
        if not added:
            if bound > lp_bound:
                counts["conic_raised_bound"] += 1
                return bound, list(w)
            return lp_bound, w_lp
        new_lp, new_w, _, decided = separate_cuts(
            c, D, pair_to_idx, pairs, np_pairs, cut_counts, deadline=deadline, manifold_cap=0,
            lean=True)
        if new_w is None:
            if decided:
                return new_lp, None          # the LP with the new valid cut is infeasible
            # The LP loop was cut short at the deadline; the last conic bound stays valid.
            return max(lp_bound, bound), list(w)
        lp_bound, w_lp = max(lp_bound, new_lp), new_w
