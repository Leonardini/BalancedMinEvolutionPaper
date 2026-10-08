"""The side-membership model of Section 5.1 as an LP (all variables continuous), and
the cut loop shared by the w-space relaxations of Section 8.4.

Variables: w_ij in [2^-(n-1), 1/4]; for each internal split k = 0..n-4, side
memberships s_ik, separation indicators B_ijk (the XOR of s_ik and s_jk, by its four
facets) and P_ijk = B_ijk w_ij (McCormick). Layers, each switched on by an argument:
    Kraft         sum_j w_ij = 1/2                                   (always)
    mincut        W[S] >= 1/2 for every nontrivial bipartition S, written out in full
    split         s, B, P, the coupling sum_{i<j} P_ijk = 1/2, and the symmetry
                  breaking of Section 5.1: minority reference side (|A_k| <= n/2),
                  splits ordered by size, the first two of size 2 and disjoint, and
                  for even n leaf 1 outside every split of size n/2
    laminarity    pairwise compatibility of the splits, by one of two encodings:
                  'quadrant' - q_i = s_ik s_il (McCormick) and an indicator per
                     quadrant of the pair that is empty, at least one of them, plus
                     sum_i (s_ik + s_il - 2 q_i) >= 1 (the splits differ)
                  'nested'   - with the splits ordered by size, compatible means
                     A_k inside A_l or disjoint; one indicator z_kl:
                     s_ik - s_il <= 1 - z, s_ik + s_il <= 1 + z, |A_l| - |A_k| >= z
The manifold sum_{i<j} w_ij log2 w_ij <= 3/2 - n is imposed exactly, as the
exponential-cone constraint sum_{i<j} entr(w_ij) >= (n - 3/2) ln 2 solved by MOSEK
(solve_exact); a model without it is an LP solved by Gurobi.

Every row is named by its family, for the optimum check of lib/safe_bound.py, and
tree_point gives a certified optimal tree's point in these variables.
"""
import itertools
import math
import os
import time

from . import safe_bound
from .common import gurobi_model, log

# The bound-stall stop of solve_with_cuts: a rise of at most STALL_REL * |bound| over
# STALL_ROUNDS rounds.
STALL_ROUNDS = 500
STALL_REL = 1e-10


def trace():
    """Progress cadence of solve_with_cuts in rounds (BME_TRACE, default 50)."""
    return int(os.environ.get("BME_TRACE", "50"))


def bipartitions(n):
    """Every nontrivial bipartition once, as the side not containing leaf 0."""
    for k in range(2, n - 1):
        for S in itertools.combinations(range(1, n), k):
            yield set(S)


def build(D, mincut, split, laminarity=None):
    """Returns (model, w, layers) where w maps (i, j), i < j, to its variable and
    layers holds the s, B, P variable dicts, and the laminarity variables (quad: (k, l)
    -> (q, e); nest: (k, l) -> z) (empty without the split layer). The model's
    _implied_hi gives P <= 1/4 for the safe bound: P has no upper bound of its own,
    and every feasible point has P_ijk <= w_ij (a McCormick row) <= 1/4 (w's bound)."""
    import gurobipy as gp
    from gurobipy import GRB
    n = len(D)
    pairs = list(itertools.combinations(range(n), 2))
    wlo, whi = 2.0 ** -(n - 1), 0.25
    M = gurobi_model()
    M.Params.Method = 2
    w = {p: M.addVar(lb=wlo, ub=whi) for p in pairs}

    def we(i, j):
        return w[(i, j) if i < j else (j, i)]

    for i in range(n):
        M.addConstr(gp.quicksum(we(i, j) for j in range(n) if j != i) == 0.5, name="kraft")
    if mincut:
        for S in bipartitions(n):
            M.addConstr(gp.quicksum(w[p] for p in pairs if (p[0] in S) != (p[1] in S)) >= 0.5,
                        name="mincut")
    layers = {}
    if split:
        K = list(range(n - 3))
        m = n // 2
        s = {(i, k): M.addVar(lb=0, ub=1) for i in range(n) for k in K}
        B = {(p, k): M.addVar(lb=0, ub=1) for p in pairs for k in K}
        P = {(p, k): M.addVar(lb=0) for p in pairs for k in K}
        for (i, j) in pairs:
            p = (i, j)
            for k in K:
                M.addConstr(B[p, k] >= s[i, k] - s[j, k], name="xor")
                M.addConstr(B[p, k] >= s[j, k] - s[i, k], name="xor")
                M.addConstr(B[p, k] <= s[i, k] + s[j, k], name="xor")
                M.addConstr(B[p, k] <= 2 - s[i, k] - s[j, k], name="xor")
                M.addConstr(P[p, k] <= w[p], name="mccormick")
                M.addConstr(P[p, k] <= whi * B[p, k], name="mccormick")
                M.addConstr(P[p, k] >= w[p] - whi * (1 - B[p, k]), name="mccormick")
                M.addConstr(P[p, k] >= wlo * B[p, k], name="mccormick")
        for k in K:
            M.addConstr(gp.quicksum(P[p, k] for p in pairs) == 0.5, name="coupling")
        size = {k: gp.quicksum(s[i, k] for i in range(n)) for k in K}
        for k in K:
            M.addConstr(size[k] <= m, name="symbreak")
            if n % 2 == 0:
                M.addConstr(size[k] + s[0, k] <= m, name="symbreak")
        for k in K[:-1]:
            M.addConstr(size[k] <= size[k + 1], name="symbreak")
        M.addConstr(size[0] == 2, name="symbreak")
        M.addConstr(size[1] == 2, name="symbreak")
        for i in range(n):
            M.addConstr(s[i, 0] + s[i, 1] <= 1, name="symbreak")
        quad, nest = {}, {}
        if laminarity == "quadrant":
            for k, l in itertools.combinations(K, 2):
                q = {i: M.addVar(lb=0, ub=1) for i in range(n)}
                for i in range(n):
                    M.addConstr(q[i] <= s[i, k], name="quadrant")
                    M.addConstr(q[i] <= s[i, l], name="quadrant")
                    M.addConstr(q[i] >= s[i, k] + s[i, l] - 1, name="quadrant")
                e = {ab: M.addVar(lb=0, ub=1) for ab in [(0, 0), (0, 1), (1, 0), (1, 1)]}
                for i in range(n):
                    M.addConstr(e[1, 1] <= 1 - q[i], name="quadrant")
                    M.addConstr(e[1, 0] <= 1 - (s[i, k] - q[i]), name="quadrant")
                    M.addConstr(e[0, 1] <= 1 - (s[i, l] - q[i]), name="quadrant")
                    M.addConstr(e[0, 0] <= 1 - (1 - s[i, k] - s[i, l] + q[i]), name="quadrant")
                M.addConstr(gp.quicksum(e.values()) >= 1, name="quadrant")
                M.addConstr(gp.quicksum(s[i, k] + s[i, l] - 2 * q[i] for i in range(n)) >= 1,
                            name="quadrant")
                quad[k, l] = (q, e)
        elif laminarity == "nested":
            for k, l in itertools.combinations(K, 2):
                z = M.addVar(lb=0, ub=1)
                for i in range(n):
                    M.addConstr(s[i, k] - s[i, l] <= 1 - z, name="nested")
                    M.addConstr(s[i, k] + s[i, l] <= 1 + z, name="nested")
                M.addConstr(size[l] - size[k] >= z, name="nested")
                nest[k, l] = z
        elif laminarity is not None:
            raise ValueError(f"unknown laminarity encoding {laminarity!r}")
        layers = dict(s=s, B=B, P=P, K=K, quad=quad, nest=nest)
    M.setObjective(gp.quicksum(D[p] * w[p] for p in pairs), GRB.MINIMIZE)
    M.update()          # variables are hashable (dict keys) only once added
    if split:
        M._implied_hi = {v: whi for v in layers["P"].values()}
    return M, w, layers


def tree_point(n, tree, w, layers):
    """The point of tree (common.optimal_tree) in the variables of build: w = 2^-tau;
    with the split layer, split k is the k-th of the tree's splits in the order the
    symmetry breaking asks for (reference side the minority side, the side without
    leaf 0 when both have n/2 leaves; sorted by size, so the first two are cherries),
    s its membership, B = XOR(s), P = B w; the laminarity variables take the values
    that satisfy their rows for compatible splits (quadrant: q = s_k s_l, e = 1 on the
    empty quadrants; nested: z = 1 iff A_k is inside A_l). Every row, these included,
    is then checked exactly."""
    tau = tree["tau"]
    pt = {v: 2.0 ** -int(tau[p]) for p, v in w.items()}
    if not layers:
        return pt
    full = frozenset(range(n))
    sides = []
    for S in tree["splits"]:                       # the side without leaf 0
        sides.append(S if 2 * len(S) <= n else full - S)
    sides.sort(key=len)
    s, B, P = layers["s"], layers["B"], layers["P"]
    for k, A in enumerate(sides):
        for i in range(n):
            pt[s[i, k]] = float(i in A)
        for p, v in w.items():
            b = float((p[0] in A) != (p[1] in A))
            pt[B[p, k]] = b
            pt[P[p, k]] = b * pt[v]
    for (k, l), (q, e) in layers["quad"].items():
        Ak, Al = sides[k], sides[l]
        for i in range(n):
            pt[q[i]] = float(i in Ak and i in Al)
        quadrant = {(1, 1): Ak & Al, (1, 0): Ak - Al, (0, 1): Al - Ak, (0, 0): full - (Ak | Al)}
        for ab, v in e.items():
            pt[v] = float(not quadrant[ab])
    for (k, l), z in layers["nest"].items():
        pt[z] = float(sides[k] < sides[l])
    return pt


def solve_with_cuts(M, w, n, manifold, separators=(), label="", tree=None, point=None):
    """Re-optimise, adding the cuts of each separator (a function sep(M, value) of the
    model and of the value of each variable at the current optimum, returning the number
    of cuts it added), until one of two stops:
      converged      no separator finds a violated cut;
      bound stalled  the bound has risen by at most STALL_REL * |bound| over the last
                     STALL_ROUNDS rounds while cuts are still being added.
    Every cut raises the LP bound or leaves it unchanged, and the bound is at most the
    optimum, so it converges and the second stop is always reached. The record says which
    stop ended the loop and the bound's rise over the last window. With `manifold` the
    relaxation carries the manifold constraint exactly and is solved by solve_exact. Progress goes to stderr every `trace()`
    rounds. The record also has the safe value of the final bound (lib/safe_bound.py),
    from the duals of the LP that gave it (taken before the last round's cuts are
    added), with the optimum check at `point` (the tree's point; None: no certified
    tree). Returns the record."""
    from collections import deque

    from gurobipy import GRB
    if manifold:
        return solve_exact(M, w, n, separators, label, point)
    M._point = None
    every = trace()
    rec = dict(separator_cuts=[0] * len(separators), rounds=0,
               converged=False, bound_stalled=False,
               stop_rule=dict(stall_rounds=STALL_ROUNDS, stall_rel=STALL_REL))
    history = deque(maxlen=STALL_ROUNDS + 1)
    t0 = time.time()
    rnd = 0
    while True:
        M.optimize()
        if M.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{label}: Gurobi status {M.Status} in round {rnd}")
        bound = M.ObjVal
        history.append(bound)
        rise = history[-1] - history[0]
        stalled = len(history) == history.maxlen and rise <= STALL_REL * abs(bound)
        sb = None
        if stalled:
            # The loop stops after this round; take the duals before cuts are added.
            sb = safe_bound.certify(M, label, implied_hi=getattr(M, "_implied_hi", None),
                                    tree=tree, point=point)
        added = 0
        for k, sep in enumerate(separators):
            a = sep(M, lambda v: v.X)
            rec["separator_cuts"][k] += a
            added += a
        rnd += 1
        rec["rounds"] = rnd
        rec["bound"] = bound
        rec["bound_rise_last_window"] = rise
        if added == 0:
            rec["converged"] = True
            rec["stop"] = "no violated cut"
        elif stalled:
            rec["bound_stalled"] = True
            rec["stop"] = f"bound rose by {rise:.3g} over the last {STALL_ROUNDS} rounds"
        if "stop" in rec:
            if sb is None:
                sb = safe_bound.certify(M, label, implied_hi=getattr(M, "_implied_hi", None),
                                        tree=tree, point=point)
            rec.update(safe_bound.fields(sb), safe_bound_detail=sb)
        if rnd % every == 1 or "stop" in rec:
            log(f"{label} round {rnd}: LP={bound:.12f} "
                f"cuts={rec['separator_cuts']} rise over last {len(history) - 1} "
                f"rounds={rise:.3g} {time.time() - t0:.0f}s")
        if "stop" in rec:
            log(f"{label}: stop after {rnd} rounds: {rec['stop']}; safe bound "
                f"{rec['safe_bound']!r} (safe - LP {rec['safe_minus_lp']:.3g}), rows cutting "
                f"off the optimum: {rec['rows_cut_off_optimum']}")
            rec["seconds"] = time.time() - t0
            return rec


def point_value(M, v):
    """The value of variable v at the optimum of the last solve_with_cuts on M."""
    pt = getattr(M, "_point", None)
    return v.X if pt is None else pt[v]


def solve_exact(M, w, n, separators, label, point):
    """The relaxation of M (every row and bound of the Gurobi model, all continuous) with
    the manifold constraint sum_{i<j} entr(w_ij) >= (n - 3/2) ln 2 added, solved by MOSEK
    (conic_manifold.solve_vars); the separators run at each conic optimum until none finds
    a violated cut. The bound is the rigorous Lagrangian bound of the last solve
    (conic_manifold.rigorous_bound_vars, 60-digit arithmetic), and every row of the last
    relaxation is evaluated exactly at `point`, the tree's point {variable: value}
    (safe_bound.tree_check). A variable without an upper bound of its own takes the one in
    M._implied_hi, which every feasible point satisfies, so the relaxation is unchanged.
    Returns the record."""
    import numpy as np
    from conic_manifold import CONIC_SOLVER, rigorous_bound_vars, solve_vars
    every = trace()
    t0 = time.time()
    rec = dict(manifold="exact", solver=CONIC_SOLVER, separator_cuts=[0] * len(separators),
               rounds=0, inaccurate_solves=0)
    implied = getattr(M, "_implied_hi", None) or {}
    sense_of = {"<": "L", ">": "G", "=": "E"}
    while True:
        M.update()
        V = M.getVars()
        wcols = [w[p].index for p in sorted(w)]
        order = wcols + [v.index for v in V if v.index not in set(wcols)]
        pos = {c: k for k, c in enumerate(order)}
        A = M.getA().tocsr()
        cons = M.getConstrs()
        rows = []
        for r in range(A.shape[0]):
            lo_, hi_ = A.indptr[r], A.indptr[r + 1]
            rows.append({pos[int(c)]: float(a) for c, a in zip(A.indices[lo_:hi_], A.data[lo_:hi_])})
        sen = [sense_of[x] for x in M.getAttr("Sense", cons)]
        rhs = list(M.getAttr("RHS", cons))
        Vo = [V[c] for c in order]
        cvec = np.array([v.Obj for v in Vo])
        lo = np.array([v.LB for v in Vo])
        hi = np.array([min(v.UB, implied.get(v, v.UB)) for v in Vo])
        if not (np.all(np.isfinite(lo)) and np.all(hi < 1e20)):
            raise RuntimeError(f"{label}: a variable has an infinite bound")
        x, val, As, bs, y, mu, iters, inacc = solve_vars(cvec, len(w), n, rows, sen, rhs, lo, hi)
        rec["inaccurate_solves"] += int(inacc)
        value = {V[c]: float(x[k]) for k, c in enumerate(order)}
        added = 0
        for k, sep in enumerate(separators):
            a = sep(M, lambda v: value[v])
            rec["separator_cuts"][k] += a
            added += a
        rec["rounds"] += 1
        if rec["rounds"] % every == 1 or added == 0:
            log(f"{label} round {rec['rounds']}: conic={val + M.ObjCon:.12f} cuts={rec['separator_cuts']} "
                f"{time.time() - t0:.0f}s")
        if added == 0:
            break
    M._point = value     # the final conic optimum, read by point_value
    bound = float(rigorous_bound_vars(cvec, As, bs, y, mu, lo, hi, len(w), sen)) + M.ObjCon
    tags = [nm.split("[")[0] for nm in M.getAttr("ConstrName", cons)]
    lp = safe_bound.LP(As, sen, bs, cvec, M.ObjCon, lo, hi, np.zeros(len(sen)), math.nan, CONIC_SOLVER)
    rec.update(bound=bound, primal=val + M.ObjCon, manifold_multiplier=mu, converged=True,
               stop="no violated cut", rows=len(rows), seconds=time.time() - t0)
    if point is not None:
        chk = safe_bound.tree_check(lp, [point[v] for v in Vo], tags)
        rec.update(rows_cut_off_optimum=chk["rows_cut_off_optimum"] + chk["bounds_violated"],
                   optimum_check=chk)
    log(f"{label}: exact manifold, {rec['rounds']} rounds, bound {bound!r}, rows cutting off "
        f"the optimum: {rec.get('rows_cut_off_optimum')}")
    return rec
