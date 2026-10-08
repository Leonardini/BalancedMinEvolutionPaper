"""The root LP of the distance-indexed model's level variables, with optional split-cut
variables B, for Section 8.4.

Variables x_ijl in [0, 1] for l = 2..n-1, with sum_l x_ijl = 1, so that
w_ij = sum_l 2^-l x_ijl and tau_ij = sum_l l x_ijl. Always present: Kraft
(sum_j w_ij = 1/2) and the manifold equality sum_{i<j} 2 tau_ij 2^-tau_ij = 2n - 3 in
its linear form over x. Optional layers:
    strongtri  tau_ik + tau_kj - tau_ij >= 2 for every pair ij and third leaf k
    mincut     W[S] >= 1/2 for every nontrivial bipartition S
    cutpoly    B_ijk in [0, 1] for the n-3 internal splits k, linked by
               tau_ij - 2 = sum_k B_ijk, with the cut-polytope triangle inequalities
               B_ij <= B_ip + B_jp (three ways) and B_ij + B_ip + B_jp <= 2 per split
Every row is named by its family, for the optimum check of lib/safe_bound.py, and
tree_point gives a certified optimal tree's point in these variables.
"""
import itertools

from .common import gurobi_model
from .membership import bipartitions


def build(D, strongtri, mincut, cutpoly):
    """Returns (model, Bm, x) where Bm(i, j, k) is the B variable of pair ij in split k
    (None without cutpoly) and x maps (pair, level) to its variable."""
    import gurobipy as gp
    from gurobipy import GRB
    n = len(D)
    pairs = list(itertools.combinations(range(n), 2))
    L = list(range(2, n))
    M = gurobi_model()
    x = {(p, l): M.addVar(lb=0, ub=1) for p in pairs for l in L}
    for p in pairs:
        M.addConstr(gp.quicksum(x[p, l] for l in L) == 1, name="level_sum")
    w = {p: gp.quicksum((2.0 ** -l) * x[p, l] for l in L) for p in pairs}
    tau = {p: gp.quicksum(l * x[p, l] for l in L) for p in pairs}

    def key(i, j):
        return (min(i, j), max(i, j))

    for i in range(n):
        M.addConstr(gp.quicksum(w[key(i, j)] for j in range(n) if j != i) == 0.5, name="kraft")
    M.addConstr(gp.quicksum(2 * l * (2.0 ** -l) * x[p, l] for p in pairs for l in L) == 2 * n - 3,
                name="manifold_eq")
    if strongtri:
        for (i, j) in pairs:
            for k in range(n):
                if k != i and k != j:
                    M.addConstr(tau[key(i, k)] + tau[key(k, j)] - tau[(i, j)] >= 2,
                                name="strongtri")
    if mincut:
        for S in bipartitions(n):
            M.addConstr(gp.quicksum(w[p] for p in pairs if (p[0] in S) != (p[1] in S)) >= 0.5,
                        name="mincut")
    Bm = None
    if cutpoly:
        K = list(range(n - 3))
        B = {(i, j, k): M.addVar(lb=0, ub=1) for (i, j) in pairs for k in K}

        def Bm(i, j, k):
            return B[(min(i, j), max(i, j), k)]

        for (i, j) in pairs:
            M.addConstr(tau[(i, j)] == gp.quicksum(B[i, j, k] for k in K) + 2, name="tau_split")
        for k in K:
            for (i, j, p) in itertools.combinations(range(n), 3):
                M.addConstr(Bm(i, j, k) <= Bm(i, p, k) + Bm(j, p, k), name="cut_triangle")
                M.addConstr(Bm(i, p, k) <= Bm(i, j, k) + Bm(j, p, k), name="cut_triangle")
                M.addConstr(Bm(j, p, k) <= Bm(i, j, k) + Bm(i, p, k), name="cut_triangle")
                M.addConstr(Bm(i, j, k) + Bm(i, p, k) + Bm(j, p, k) <= 2, name="cut_triangle")
    M.setObjective(gp.quicksum(D[p] * w[p] for p in pairs), GRB.MINIMIZE)
    M.update()          # variables are hashable (dict keys) only once added
    return M, Bm, x


def tree_point(n, tree, x, Bm):
    """The point of tree (common.optimal_tree) in the variables of build: x one-hot at
    l = tau_ij; B_ijk = 1 iff the tree's k-th split separates i and j (any order of
    the splits: the model has no symmetry breaking)."""
    tau = tree["tau"]
    pt = {v: float(l == tau[p]) for (p, l), v in x.items()}
    if Bm is not None:
        for k, S in enumerate(tree["splits"]):
            for (i, j) in itertools.combinations(range(n), 2):
                pt[Bm(i, j, k)] = float((i in S) != (j in S))
    return pt
