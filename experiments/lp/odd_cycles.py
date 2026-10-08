"""Section 8.4: are odd-cycle inequalities of the cut polytope violated at the root LP,
and do they raise the bound?

    BME_THREADS=1 python odd_cycles.py INSTANCE OUT.json MODEL [SPLITS]

The compact model has no split variables, so the inequalities have nothing to act on
there; they live on a model with one cut vector B_.k per internal split:
    MODEL = membership  the membership model of Section 5.2 at its full root
                        relaxation (Kraft, min-cut, manifold, split layer, coupling,
                        quadrant laminarity; lib/membership.py), where B = XOR(s)
    MODEL = xlevel      the level-variable LP with B and the per-split triangles,
                        with and without min-cut (lib/xlevel.py)
Each B_.k is separated exactly (lib/oddcycle.py) with violation tolerance TOL. The
record has the violations at the LP optimum before any odd-cycle cut (how many, the
largest, the cycle lengths) and the bound after separating them to convergence. The B
part of an LP optimum need not be unique, so the bound change is the solver-independent
quantity. SPLITS gives L*; without it L* is by brute force (small n). Both bounds of
each arm also have their safe values and the optimum check at the optimal tree (the
SPLITS tree, or a brute-force one), next to them as bound_before_safe_bound etc.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import membership, oddcycle, xlevel  # noqa: E402
from lib.common import log, optimal_tree, optimum, parse_matrix, peak_rss_mb, write_json  # noqa: E402

TOL = 1e-6


def columns(n, value, K):
    """value(i, j, k) of the solved model -> list of columns {(i, j): b}."""
    return [{(i, j): value(i, j, k) for i in range(n) for j in range(i + 1, n)} for k in K]


def census(n, cols):
    viol = [c for col in cols for c in oddcycle.separate(n, col, TOL)]
    return dict(violated=len(viol), max_violation=max((v for _, _, v in viol), default=0.0),
                cycle_lengths=dict(Counter(len(cf) for cf, _, _ in viol)))


def make_separator(n, var, K, stats):
    import gurobipy as gp

    def sep(M, value):
        added = 0
        for k, col in zip(K, columns(n, lambda i, j, k: value(var(i, j, k)), K)):
            for coef, rhs, _v in oddcycle.separate(n, col, TOL):
                M.addConstr(gp.quicksum(c * var(e[0], e[1], k) for e, c in coef.items()) <= rhs,
                            name="oddcycle")
                stats[len(coef)] += 1
                added += 1
        return added
    return sep


def run_arm(M, w, n, var, K, manifold, label, tree, point):
    """Converge without odd-cycle cuts, record the violations there, then separate
    them (with the manifold cuts, if any) to convergence."""
    before = membership.solve_with_cuts(M, w, n, manifold, label=f"{label} before",
                                        tree=tree, point=point)
    cen = census(n, columns(n, lambda i, j, k: membership.point_value(M, var(i, j, k)), K))
    stats = Counter()
    after = membership.solve_with_cuts(M, w, n, manifold, [make_separator(n, var, K, stats)],
                                       label=f"{label} after", tree=tree, point=point)
    out = dict(bound_before=before["bound"], loop_before=before, violations_before=cen,
               bound_after=after["bound"], loop_after=after,
               cuts_by_cycle_length=dict(stats))
    for k, r in (("bound_before", before), ("bound_after", after)):
        # An LP arm has its safe bound; an arm with the manifold has the rigorous conic
        # bound as its bound, and the exact row check at the tree.
        out.update({f"{k}_{f}": r[f] for f in ("safe_bound", "safe_minus_lp",
                                               "rows_cut_off_optimum") if f in r})
    return out


def main():
    inst, out, model = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    splits = sys.argv[4] if len(sys.argv) > 4 else None
    D = parse_matrix(inst)
    n = D.shape[0]
    L = optimum(D, splits)
    tree = optimal_tree(D, instance=inst, splits_path=splits)
    K = list(range(n - 3))
    rec = dict(instance=inst, n=n, optimum=L, model=model, tol=TOL, arms={})
    if model == "membership":
        M, w, layers = membership.build(D, mincut=True, split=True, laminarity="quadrant")
        B = layers["B"]
        arms = [("membership", M, w, lambda i, j, k: B[(i, j), k], True,
                 membership.tree_point(n, tree, w, layers))]
    elif model == "xlevel":
        arms = []
        for mc in (False, True):
            M, Bm, x = xlevel.build(D, strongtri=False, mincut=mc, cutpoly=True)
            arms.append((f"xlevel mincut={int(mc)}", M, {}, Bm, False,
                         xlevel.tree_point(n, tree, x, Bm)))
    else:
        raise ValueError("MODEL must be membership or xlevel")
    for name, M, w, var, manifold, point in arms:
        r = run_arm(M, w, n, var, K, manifold, name, tree, point)
        for k in ("bound_before", "bound_after"):
            r[k.replace("bound", "gap")] = (L - r[k]) / L
        rec["arms"][name] = r
        rec["peak_rss_mb"] = peak_rss_mb()
        write_json(out, rec)
        log(f"{name}: {r['violations_before']['violated']} violated before; "
            f"gap {100 * r['gap_before']:.4f}% -> {100 * r['gap_after']:.4f}%")


if __name__ == "__main__":
    main()
