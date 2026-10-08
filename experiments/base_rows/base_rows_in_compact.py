"""Table D10 with the manifold imposed exactly: which rows of the distance-indexed base
model close the compact model's root gap?

    BME_THREADS=1 python base_rows_in_compact.py INSTANCE SPLITS OUT.json

The arms, and the rows of the distance-indexed model they add, are those of
base_rows_lib.py (level variables x_ij^l with sum_l x_ij^l = 1 and w_ij = sum_l 2^-l x_ij^l;
the manifold as the linear equality sum_{i<j} sum_l 2 l 2^-l x_ij^l = 2n - 3; the
triangle inequalities tau_ik + tau_kj - tau_ij >= 2 on all triples, tau = sum_l l x; the
2-K-split inequalities, lifted for every K or unlifted with K <= 4, separated exactly).
The difference is the compact model's own manifold constraint: instead of tangent cuts
it is the one convex constraint sum_{i<j} entr(w_ij) >= (n - 3/2) ln 2 in every arm, and
each relaxation is solved by MOSEK (lp/conic_manifold.py, solve_vars). The compact
model's cuts (min-cut, PM, crossing) and the 2-K-splits are separated at each optimum
until none is violated. The bound of an arm is the rigorous Lagrangian bound of its last
solve, and every row is evaluated exactly at the optimal tree.

In arms with the linear manifold equality the convex constraint is left out: it is implied
(by convexity of t log t, w_ij log2 w_ij <= -sum_l l 2^-l x_ij^l for w_ij = sum_l 2^-l x_ij^l,
so g(w) <= -(n - 3/2) whenever the equality holds), and keeping both leaves the relaxation
with no interior. Those arms are then linear programs, and they are solved by CPLEX's dual
simplex method instead, with the Neumaier-Shcherbina safe bound (lp/lib/safe_bound.py) as
their rigorous bound.

The arms are nested (CONTAINS): an arm has every row of the arms listed for it, so their
rigorous bounds are valid for it too, and its reported bound is the largest of its own
and theirs. This matters where the conic solver ends "almost solved" (the triangle rows over the
level variables are highly degenerate), so that its own dual gives a looser bound than a
smaller arm's. The record keeps the arm's own bound and says which bound is reported.
"""
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cvxpy as cp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "compact"))
sys.path.insert(0, str(HERE.parent / "lp"))
sys.path.insert(0, str(HERE))
from conic_manifold import CONIC_SOLVER, base_rows, rigorous_bound_vars, solve_vars  # noqa: E402
from lib import safe_bound  # noqa: E402
from lib.common import log, optimal_tree, parse_matrix, peak_rss_mb  # noqa: E402
from base_rows_lib import ARMS, TOL, coefs_2k, tree_tau  # noqa: E402

import cplex  # noqa: E402
from solver.run_config import apply_solver_config  # noqa: E402
from solver.pm_separator import separate_pm  # noqa: E402
from solver.w_space import (EPS, build_W, cut_row_sparse, find_half_cuts,  # noqa: E402
                            first_crossing_pair, global_min_cut)


# Two more arms: the distance-indexed model's static four-point rows for every quartet
# containing leaf 1 (its --buneman O), transcribed from create_variables and
# create_buneman_cons of solver_gurobi.hpp: for {1, j, p, q} three resolution variables
# y in [0, 1] summing to 1, and six big-M rows with M = 2(n - 2). The solver's taxon
# order is the input order in our configuration (it logs the identity permutation).
ARMS_X = {k: v + (False,) for k, v in ARMS.items()}
ARMS_X["lift_buneman"] = (True, False, False, None, True)
ARMS_X["lift_base_buneman"] = (True, True, True, None, True)

class SimplexModel:
    """A CPLEX LP over the same variables, to which the rows are added as they appear."""

    def __init__(self, cvec, lo, hi):
        self.c = cplex.Cplex()
        apply_solver_config(self.c)
        for st in (self.c.set_log_stream, self.c.set_results_stream, self.c.set_warning_stream):
            st(None)
        self.c.objective.set_sense(self.c.objective.sense.minimize)
        self.c.variables.add(obj=[float(v) for v in cvec], lb=[float(v) for v in lo],
                             ub=[float(v) for v in hi], names=[f"v{k}" for k in range(len(cvec))])
        self.c.parameters.emphasis.numerical.set(1)
        self.c.parameters.lpmethod.set(self.c.parameters.lpmethod.values.dual)
        self.n_rows = 0

    def solve(self, rows, sen, rhs):
        new = range(self.n_rows, len(rows))
        if len(new):
            self.c.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=list(map(int, rows[r])), val=list(rows[r].values()))
                          for r in new],
                senses=[sen[r] for r in new], rhs=[float(rhs[r]) for r in new])
            self.n_rows = len(rows)
        self.c.solve()
        if self.c.solution.get_status() != 1:
            raise RuntimeError(f"CPLEX status {self.c.solution.get_status_string()}")
        return np.array(self.c.solution.get_values()), self.c.solution.get_objective_value()


CONTAINS = {"compact": [], "lift": ["compact"], "lift_manifold": ["compact", "lift"],
            "lift_triangle": ["compact", "lift"],
            "lift_base": ["compact", "lift", "lift_manifold", "lift_triangle"],
            "lift_base_2k": ["compact", "lift", "lift_manifold", "lift_triangle", "lift_base"],
            "lift_2k_K4": ["compact", "lift"],
            "lift_buneman": ["compact", "lift"],
            "lift_base_buneman": ["compact", "lift", "lift_manifold", "lift_triangle", "lift_base",
                                  "lift_buneman"]}


def root_bound(D, tree, arm, L_star):
    n = D.shape[0]
    use_lift, manifold_eq, triangles, split2k, buneman = ARMS_X[arm]
    pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    pti = {p: k for k, p in enumerate(pairs)}
    N = len(pairs)
    levels = list(range(2, n)) if use_lift else []
    xi = {}
    for p in pairs:
        for l in levels:
            xi[p, l] = N + len(xi)
    yi = {}
    if buneman:
        for j in range(2, n + 1):
            for p_ in range(j + 1, n + 1):
                for q in range(p_ + 1, n + 1):
                    for quad in ((1, j, p_, q), (1, p_, j, q), (1, q, j, p_)):
                        yi[quad] = N + len(xi) + len(yi)
    nv = N + len(xi) + len(yi)
    cvec = np.zeros(nv)
    for (i, j), k in pti.items():
        cvec[k] = D[i - 1, j - 1]
    lo = np.concatenate([np.full(N, 2.0 ** -(n - 1)), np.zeros(len(xi) + len(yi))])
    hi = np.concatenate([np.full(N, 0.25), np.ones(len(xi) + len(yi))])
    rows, sen, rhs = base_rows(n, pti)
    tags = ["kraft"] * n + ["double_cherry"] * (len(rows) - n)

    def add(row, s, b, tag):
        rows.append(row), sen.append(s), rhs.append(float(b)), tags.append(tag)

    def tau_terms(a, b):
        p = (min(a, b), max(a, b))
        return {xi[p, l]: float(l) for l in levels}

    counts = defaultdict(int)
    if use_lift:
        for p in pairs:
            add({xi[p, l]: 1.0 for l in levels}, "E", 1.0, "level")
            row = {xi[p, l]: 2.0 ** -l for l in levels}
            row[pti[p]] = -1.0
            add(row, "E", 0.0, "level")
        if manifold_eq:
            tv = sum(2.0 * tree[i, j] * 2.0 ** -tree[i, j] for i, j in pairs)
            assert abs(tv - (2 * n - 3)) < 1e-9, "manifold equality fails on the optimal tree"
            add({k: 2.0 * l * 2.0 ** -l for (p, l), k in xi.items()}, "E", 2 * n - 3, "manifold_eq")
        if triangles:
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    for k in range(1, n + 1):
                        if k in (i, j):
                            continue
                        assert tree[i, k] + tree[k, j] - tree[i, j] >= 2
                        row = defaultdict(float)
                        for a, b, sg in ((i, k, 1.0), (k, j, 1.0), (i, j, -1.0)):
                            for idx, v in tau_terms(a, b).items():
                                row[idx] += sg * v
                        add(dict(row), "G", 2.0, "triangle")
                        counts["triangle_rows"] += 1
    if buneman:
        M = 2 * (n - 2)

        def t(a, b):
            return tau_terms(a, b)

        def row_of(plus, minus, ycoef):
            row = defaultdict(float)
            for a, b in plus:
                for idx, v in t(a, b).items():
                    row[idx] += v
            for a, b in minus:
                for idx, v in t(a, b).items():
                    row[idx] -= v
            for idx, v in ycoef:
                row[idx] += v
            return dict(row)

        for (i, j, p_, q) in [k for k in yi if k[1] < k[2]]:
            YA, YB, YC = yi[(i, j, p_, q)], yi[(i, p_, j, q)], yi[(i, q, j, p_)]
            add({YA: 1.0, YB: 1.0, YC: 1.0}, "E", 1.0, "buneman")
            ij, ip, iq, jp, jq, pq = (i, j), (i, p_), (i, q), (j, p_), (j, q), (p_, q)
            # tip+tjq >= tij+tpq + 2(1-YC) - M YB, and the five others, as in the solver.
            for plus, minus, yc in (((ip, jq), (ij, pq), ((YC, 2.0), (YB, M))),
                                    ((iq, jp), (ij, pq), ((YB, 2.0), (YC, M))),
                                    ((ij, pq), (ip, jq), ((YC, 2.0), (YA, M))),
                                    ((iq, jp), (ip, jq), ((YA, 2.0), (YC, M))),
                                    ((ij, pq), (iq, jp), ((YB, 2.0), (YA, M))),
                                    ((ip, jq), (iq, jp), ((YA, 2.0), (YB, M)))):
                add(row_of(plus, minus, yc), "G", 2.0, "buneman")
                counts["buneman_rows"] += 1
    coef = coefs_2k(n) if split2k else None
    seen = set()
    t0 = time.time()
    rounds = 0
    stop = "no violated cut"
    last = None
    lp_model = SimplexModel(cvec, lo, hi) if manifold_eq else None
    while True:
        rounds += 1
        if lp_model is not None:
            v, val = lp_model.solve(rows, sen, rhs)
            A = b = y = None
            mu = 0.0
            inaccurate = False
            out = None
        try:
            if lp_model is None:
                out = solve_vars(cvec, N, n, rows, sen, rhs, lo, hi)
        except cp.SolverError:
            # The conic solver failed at every tolerance. With an earlier solve the arm stops there:
            # that solve's rows are a subset of the arm's, so its bound is valid (weaker).
            if last is None:
                raise
            stop = f"conic solver failed at round {rounds}; bound of round {rounds - 1}"
            log(f"  {arm}: {stop}")
            v, val, A, b, y, mu = last
            m_rows = len(y)
            break
        if out is not None:
            v, val, A, b, y, mu, iters, inaccurate = out
            last = (v, val, A, b, y, mu)
        counts["inaccurate_solves"] += inaccurate
        w = v[:N]
        W = build_W(list(w), n)
        new = 0
        mc_val, mc_set = global_min_cut(W, n)
        if mc_val < 0.5 - EPS:
            idx, vals = cut_row_sparse(mc_set, n, pti)
            add(dict(zip(idx, vals)), "G", 0.5, "mincut"); counts["mincut"] += 1; new += 1
        _, pm_idx, pm_vals, pm_Cn = separate_pm(list(w), pairs, pti, n)
        if pm_idx is not None:
            add({int(i): float(x) for i, x in zip(pm_idx, pm_vals)}, "G", pm_Cn, "pm")
            counts["pm"] += 1; new += 1
        pair = first_crossing_pair(find_half_cuts(W, n))
        if pair is not None:
            row = defaultdict(float)
            for S in pair:
                for i, x in zip(*cut_row_sparse(S, n, pti)):
                    row[i] += x
            add(dict(row), "G", 1.25, "crossing"); counts["crossing"] += 1; new += 1
        if split2k:
            lifted, k_max = split2k
            k_max = k_max or n - 2
            tau = np.zeros((n + 1, n + 1))
            for (p, l), k in xi.items():
                tau[p[0], p[1]] += l * v[k]
            tau = tau + tau.T
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    others = sorted((tau[i, k] + tau[j, k], k) for k in range(1, n + 1) if k not in (i, j))
                    prefix = 0.0
                    for K in range(1, k_max + 1):
                        prefix += others[K - 1][0]
                        S = [k for _, k in others[:K]]
                        if lifted:
                            rhs_val = sum(coef[l][K] * v[xi[(i, j), l]] for l in levels)
                            tree_rhs = coef[tree[i, j]][K]
                        else:
                            rhs_val = tree_rhs = min(coef[l][K] for l in levels)
                        if prefix >= rhs_val - TOL:
                            continue
                        assert sum(tree[i, k] + tree[j, k] for k in S) >= tree_rhs, \
                            "a 2-K-split cuts off the optimal tree"
                        row = defaultdict(float)
                        for k in S:
                            for a in (i, j):
                                for idx, x in tau_terms(a, k).items():
                                    row[idx] += x
                        if lifted:
                            for l in levels:
                                row[xi[(i, j), l]] -= float(coef[l][K])
                            r_, b_ = dict(row), 0.0
                        else:
                            r_, b_ = dict(row), float(rhs_val)
                        key = (tuple(sorted(r_.items())), b_)
                        if key in seen:
                            continue
                        seen.add(key)
                        add(r_, "G", b_, "2k"); counts["2k"] += 1; new += 1
        if rounds % 10 == 0 or new == 0:
            log(f"  {arm}: round {rounds}, bound {val:.10f}, rows {len(rows)}, {dict(counts)}, "
                f"{time.time() - t0:.0f}s")
        if new == 0:
            break
    x = np.zeros(nv)
    for (i, j), k in pti.items():
        x[k] = 2.0 ** -int(tree[i, j])
    for (p, l), k in xi.items():
        x[k] = 1.0 if tree[p[0], p[1]] == l else 0.0
    for (qa, qb, qc, qd), k in yi.items():
        # y of a resolution ab|cd is 1 at the tree iff it is the tree's resolution.
        x[k] = 1.0 if tree[qa, qb] + tree[qc, qd] < min(tree[qa, qc] + tree[qb, qd],
                                                       tree[qa, qd] + tree[qb, qc]) else 0.0
    if lp_model is not None:
        sb = safe_bound.certify(lp_model.c, arm, reported=val, tree={"source": "ground truth"},
                                point={f"v{k}": float(x[k]) for k in range(nv)}, tags=tags)
        return dict(bound=float(sb["safe_bound"]), primal=val, solver="CPLEX dual simplex",
                    stop=stop, finished=True, manifold_multiplier=None, rounds=rounds,
                    cut_counts=dict(counts), rows=len(rows), cols=nv,
                    rows_cut_off_optimum=sb["rows_cut_off_optimum"],
                    safe_minus_lp=sb["safe_minus_reported"], seconds=time.time() - t0,
                    root_gap_pct=100 * (L_star - float(sb["safe_bound"])) / L_star)
    m = len(y)
    if stop.startswith("conic"):
        assert m == m_rows
    bound = float(rigorous_bound_vars(cvec, A, b, y, mu, lo, hi, N, sen[:m]))
    lp = safe_bound.LP(A, sen[:m], b, cvec, 0.0, lo, hi, np.zeros(m), math.nan, CONIC_SOLVER)
    chk = safe_bound.tree_check(lp, x, tags[:m])
    return dict(bound=bound, primal=val, solver=CONIC_SOLVER, stop=stop, finished=stop == "no violated cut", manifold_multiplier=mu, rounds=rounds,
                cut_counts=dict(counts), rows=len(rows), cols=nv,
                rows_cut_off_optimum=chk["rows_cut_off_optimum"], seconds=time.time() - t0,
                root_gap_pct=100 * (L_star - bound) / L_star)


def main():
    inst, splits_file, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    D = parse_matrix(inst)
    n = D.shape[0]
    tree = tree_tau(splits_file, n)
    opt = optimal_tree(D, splits_path=splits_file)
    assert (opt["tau"] == tree[1:, 1:]).all(), "two readings of SPLITS disagree"
    L_star = float(opt["value"])
    rec = {"instance": inst, "n": n, "L_star": L_star, "manifold": f"exact ({CONIC_SOLVER})", "arms": {}}
    assert list(ARMS_X) == list(CONTAINS), "arms and their nesting must match"
    for arm in ARMS_X:
        r = root_bound(D, tree, arm, L_star)
        own = r["bound"]
        best = max([(own, arm)] + [(rec["arms"][a]["own_bound"], a) for a in CONTAINS[arm]])
        r.update(own_bound=own, own_root_gap_pct=r["root_gap_pct"], bound=best[0],
                 bound_from=best[1], root_gap_pct=100 * (L_star - best[0]) / L_star)
        rec["arms"][arm] = r
        comp = rec["arms"]["compact"]["root_gap_pct"]
        r["gap_closed_pct"] = 100 * (comp - r["root_gap_pct"]) / comp
        rec["peak_rss_mb"] = peak_rss_mb()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, default=float) + "\n")
        log(f"{arm}: root gap {r['root_gap_pct']:.4f}% (closes {r['gap_closed_pct']:.1f}% of the "
            f"compact gap), cut off {r['rows_cut_off_optimum']}, {r['seconds']:.0f}s")


if __name__ == "__main__":
    main()
