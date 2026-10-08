"""The level variables, rows and arms used by base_rows_in_compact.py (Table D10), which
imposes the manifold exactly; run alone, this file solves the same arms with the manifold
imposed by tangent cuts.

    BME_THREADS=1 python base_rows_lib.py INSTANCE SPLITS OUT.json

The 2-K-split inequalities of Catanzaro et al. are linear in the path lengths tau, not
in w, so they cannot be checked at the compact model's root point w* directly: many
fractional path-length vectors are consistent with one w*. We therefore add the
level variables of the distance-indexed model to the compact root LP,

    x_ij^l >= 0 (l = 2..n-1),   sum_l x_ij^l = 1,   w_ij = sum_l 2^-l x_ij^l,
    tau_ij = sum_l l x_ij^l,

and separate the 2-K-split inequalities over (tau, x), for a pair {i,j} and a set S of
K leaves other than i and j:

    unlifted:  sum_{k in S} (tau_ik + tau_jk) >= min_l c(l, K)
    lifted:    sum_{k in S} (tau_ik + tau_jk) >= sum_l c(l, K) x_ij^l

with c(l, K) as computed by the distance-indexed solver (precompute_2K_coefs). For
fixed {i,j} and K the most violated S is the K leaves with the smallest
tau_ik + tau_jk, so separation is exact.

Arms. Every arm runs the compact model's own cut loop (min-cut, PM, manifold tangents)
interleaved with any extra separation until neither finds a violated cut, so the arms
differ only in what is added:
    compact          nothing
    lift             the level variables alone (a check: the bound should not move)
    lift_manifold    + the manifold as the linear equality of the distance-indexed
                     model, sum_{i<j} sum_l 2 l 2^-l x_ij^l = 2n - 3
    lift_triangle    + the triangle inequalities tau_ik + tau_kj - tau_ij >= 2
    lift_base        + both (the base relaxation of the distance-indexed model)
    lift_base_2k     + both, and the lifted 2-K-splits for every K
    lift_2k_K4       + the unlifted 2-K-splits with K <= 4 only (the distance-indexed
                     solver's root separation), without manifold equality or triangles
SPLITS (the certified optimal tree) gives L*, and every added row is checked against it.
Each arm's bound also has its safe value (experiments/lp/lib/safe_bound.py, from the
final LP's duals in exact arithmetic) and the optimum check: every row of the final
LP evaluated exactly at the tree's point (w = 2^-tau, x one-hot at tau, z = 0), by
family, with rows_cut_off_optimum next to the bound.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "compact"))
import cplex  # noqa: E402
from solver.base_model import build_base_model  # noqa: E402
from solver.bnb_balanced import MANIFOLD_ROOT_CAP  # noqa: E402
from solver.cut_loop import separate_cuts  # noqa: E402
from solver.parse_matrix import parse_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lp"))
from lib import safe_bound  # noqa: E402
from lib.common import compact_point, optimal_tree  # noqa: E402

TOL = 1e-6


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", file=sys.stderr, flush=True)


def coefs_2k(n):
    """c[l][K] for l = 2..n-1, K = 1..n-2; a transcription of precompute_2K_coefs."""
    import heapq
    c = {}
    for l in range(2, n):
        c[l] = {}
        queues = [[1] for _ in range(l - 1)]
        sum_queues = l - 1
        max_queue = [1] * (l - 1)
        for K in range(1, l - 1):
            c[l][K] = K * (l + 2)
        cur = 0
        for K in range(l - 1, n - 2):
            m = heapq.heappop(queues[cur])
            heapq.heappush(queues[cur], m + 1)
            heapq.heappush(queues[cur], m + 1)
            max_queue[cur] = max(max_queue[cur], m + 1)
            sum_queues += m + 2
            c[l][K] = 2 * (sum_queues - max_queue[cur]) + K * l
            cur = (cur + 1) % (l - 1)
        c[l][n - 2] = 2 * sum_queues + (n - 2) * l
    return c


def tree_tau(splits_file, n):
    """Path-length matrix of the tree whose nontrivial splits are listed in SPLITS."""
    splits = []
    for line in Path(splits_file).read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            splits.append({int(v) for v in line.strip("{}").split(",")})
    if len(splits) != n - 3:
        raise ValueError(f"{splits_file}: {len(splits)} splits, expected {n - 3}")
    tau = np.full((n + 1, n + 1), 2, dtype=int)
    for S in splits:
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if (i in S) != (j in S):
                    tau[i, j] += 1
    np.fill_diagonal(tau, 0)
    return tau


class Lift:
    """Level variables x_ij^l appended to the compact model."""

    def __init__(self, c, n, pairs, pair_to_idx):
        self.c, self.n, self.pair_to_idx = c, n, pair_to_idx
        self.levels = list(range(2, n))
        first = c.variables.get_num()
        names = [f"x_{i}_{j}_{l}" for (i, j) in pairs for l in self.levels]
        c.variables.add(lb=[0.0] * len(names), ub=[1.0] * len(names), names=names)
        self.x = {}
        k = first
        for p in pairs:
            for l in self.levels:
                self.x[p, l] = k
                k += 1
        rows, senses, rhs = [], [], []
        for p in pairs:
            xs = [self.x[p, l] for l in self.levels]
            rows.append(cplex.SparsePair(ind=xs, val=[1.0] * len(xs)))
            senses.append("E"); rhs.append(1.0)
            rows.append(cplex.SparsePair(ind=xs + [pair_to_idx[p]],
                                         val=[2.0 ** -l for l in self.levels] + [-1.0]))
            senses.append("E"); rhs.append(0.0)
        c.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs)

    def add_manifold_equality(self, tree):
        n = self.n
        ind, val = [], []
        for (p, l), k in self.x.items():
            ind.append(k); val.append(2.0 * l * 2.0 ** -l)
        tree_val = sum(2.0 * tree[i, j] * 2.0 ** -tree[i, j]
                       for i in range(1, n + 1) for j in range(i + 1, n + 1))
        if abs(tree_val - (2 * n - 3)) > 1e-9:
            raise AssertionError(f"manifold equality fails on the optimal tree: {tree_val}")
        self.c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                                      senses=["E"], rhs=[float(2 * n - 3)])

    def add_triangles(self, tree):
        n = self.n
        rows = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                for k in range(1, n + 1):
                    if k in (i, j):
                        continue
                    if tree[i, k] + tree[k, j] - tree[i, j] < 2:
                        raise AssertionError(f"triangle ({i},{k},{j}) fails on the optimal tree")
                    ind, val = [], []
                    for a, b, sgn in ((i, k, 1.0), (k, j, 1.0), (i, j, -1.0)):
                        xi, xv = self.tau_terms(a, b)
                        ind += xi; val += [sgn * v for v in xv]
                    rows.append(cplex.SparsePair(ind=ind, val=val))
        self.c.linear_constraints.add(lin_expr=rows, senses=["G"] * len(rows),
                                      rhs=[2.0] * len(rows))
        return len(rows)

    def tau_terms(self, i, k):
        p = (min(i, k), max(i, k))
        return [self.x[p, l] for l in self.levels], [float(l) for l in self.levels]

    def tau_values(self):
        vals = self.c.solution.get_values()
        tau = np.zeros((self.n + 1, self.n + 1))
        for (p, l), k in self.x.items():
            tau[p[0], p[1]] += l * vals[k]
        return tau + tau.T, vals


def separate_2k(lift, coef, lifted, k_max, tree, cut_counts):
    """Add every most-violated 2-K-split (one per pair and K). Returns the number added."""
    n = lift.n
    tau, vals = lift.tau_values()
    added = 0
    rows, rhs_list = [], []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            others = sorted((tau[i, k] + tau[j, k], k) for k in range(1, n + 1) if k not in (i, j))
            prefix = 0.0
            for K in range(1, k_max + 1):
                prefix += others[K - 1][0]
                S = [k for _, k in others[:K]]
                if lifted:
                    rhs_val = sum(coef[l][K] * vals[lift.x[(i, j), l]] for l in lift.levels)
                    tree_rhs = coef[tree[i, j]][K]
                else:
                    rhs_val = tree_rhs = min(coef[l][K] for l in lift.levels)
                if prefix >= rhs_val - TOL:
                    continue
                # The optimal tree must satisfy the inequality (a check on the coefficients).
                tree_lhs = sum(tree[i, k] + tree[j, k] for k in S)
                if tree_lhs < tree_rhs:
                    raise AssertionError(f"2-K-split ({i},{j}),K={K},S={S} cuts off the optimal tree")
                ind, val = [], []
                for k in S:
                    for a in (i, j):
                        xi, xv = lift.tau_terms(a, k)
                        ind += xi; val += xv
                if lifted:
                    for l in lift.levels:
                        ind.append(lift.x[(i, j), l]); val.append(-float(coef[l][K]))
                    rhs_list.append(0.0)
                else:
                    rhs_list.append(float(rhs_val))
                rows.append(cplex.SparsePair(ind=ind, val=val))
                added += 1
    if rows:
        lift.c.linear_constraints.add(lin_expr=rows, senses=["G"] * len(rows), rhs=rhs_list)
    cut_counts["2k"] += added
    return added


ARMS = {
    # arm: (lift, manifold equality, triangles, 2-K-splits as (lifted, K_max) or None)
    "compact": (False, False, False, None),
    "lift": (True, False, False, None),
    "lift_manifold": (True, True, False, None),
    "lift_triangle": (True, False, True, None),
    "lift_base": (True, True, True, None),
    "lift_base_2k": (True, True, True, (True, None)),
    "lift_2k_K4": (True, False, False, (False, 4)),
}


def root_bound(D, tree, arm, opt_tree):
    n = D.shape[0]
    use_lift, manifold_eq, triangles, split2k = ARMS[arm]
    c, pair_to_idx, pairs = build_base_model(D, verbose=False)
    np_pairs = len(pairs)
    c.variables.set_types(np_pairs, c.variables.type.continuous)
    c.set_problem_type(c.problem_type.LP)
    c.parameters.emphasis.numerical.set(1)
    c.parameters.lpmethod.set(c.parameters.lpmethod.values.dual)
    cut_counts = defaultdict(int)
    n_base = c.linear_constraints.get_num()
    tags = ["kraft"] * n + ["double_cherry"] * (n_base - n)

    def tag_new(family):
        tags.extend([family] * (c.linear_constraints.get_num() - len(tags)))

    lift = None
    if use_lift:
        lift = Lift(c, n, pairs, pair_to_idx)
        tag_new("level")
        if manifold_eq:
            lift.add_manifold_equality(tree)
            tag_new("manifold_eq")
        if triangles:
            cut_counts["triangle_rows"] = lift.add_triangles(tree)
            tag_new("triangle")
    coef = coefs_2k(n) if split2k else None
    rounds = 0
    while True:
        rounds += 1
        added = 0
        if split2k and rounds > 1:
            c.solve()
            lifted, k_max = split2k
            added = separate_2k(lift, coef, lifted, k_max or n - 2, tree, cut_counts)
            tag_new("2k")
        bound, w_vals, n_new, decided = separate_cuts(
            c, D, pair_to_idx, pairs, np_pairs, cut_counts, manifold_cap=MANIFOLD_ROOT_CAP,
            lean=True, row_tags=tags)
        if not decided:
            raise RuntimeError(f"{arm}: LP did not solve")
        log(f"  {arm}: round {rounds} +{added} 2-K-splits, +{n_new} compact cuts, bound {bound:.10f}")
        if added == 0 and n_new == 0 and (rounds > 1 or not split2k):
            break
    assert len(tags) == c.linear_constraints.get_num(), "untagged rows"
    sb = safe_bound.certify(c, arm, reported=bound, tree=opt_tree, tags=tags,
                            point=compact_point(opt_tree, c.variables.get_names()))
    log(f"  {arm}: safe bound {sb['safe_bound']!r}, safe - LP {sb['safe_minus_reported']:.3g}, "
        f"rows cutting off the optimum {sb['rows_cut_off_optimum']}, {sb['seconds']:.1f}s")
    return {"bound": bound, **safe_bound.fields(sb), "rounds": rounds,
            "cut_counts": dict(cut_counts), "rows": c.linear_constraints.get_num(),
            "cols": c.variables.get_num(), "safe_bound_detail": sb}


def main():
    inst, splits_file, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    D = parse_matrix(inst)
    n = D.shape[0]
    tree = tree_tau(splits_file, n)
    opt_tree = optimal_tree(D, splits_path=splits_file)
    assert (opt_tree["tau"] == tree[1:, 1:]).all(), "two readings of SPLITS disagree"
    w_tree = np.where(tree > 0, 2.0 ** -tree.astype(float), 0.0)
    L_star = sum(D[i - 1, j - 1] * w_tree[i, j] for i in range(1, n + 1) for j in range(i + 1, n + 1))
    log(f"{inst}: n={n} L*={L_star:.10f}")
    rec = {"instance": inst, "n": n, "L_star": L_star, "arms": {}}
    for arm in ARMS:
        t0 = time.time()
        r = root_bound(D, tree, arm, opt_tree)
        r["seconds"] = time.time() - t0
        r["root_gap_pct"] = 100 * (L_star - r["bound"]) / L_star
        rec["arms"][arm] = r
        log(f"{arm}: bound {r['bound']:.10f} gap {r['root_gap_pct']:.4f}%")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1) + "\n")  # partial results survive
    base = rec["arms"]["compact"]["bound"]
    for arm in ARMS:
        rec["arms"][arm]["gap_closed_pct"] = 100 * (rec["arms"][arm]["bound"] - base) / (L_star - base)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
