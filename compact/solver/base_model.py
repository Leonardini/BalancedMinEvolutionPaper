"""Build the base CPLEX MIP model for BME.

Variables
---------
w_{i}_{j}  (i<j, 1-indexed): continuous, in [F1_lb, 1/4].
z          : binary, heavily penalised (Option B dummy variable).

Constraints
-----------
Kraft: sum_j w_ij = 1/2  for each leaf i.
Box upper: w_ij <= 1/4  (also enforced via variable upper bound).
F1 lower: w_ij >= 2^{-(n-1)}  (variable lower bound).
Double-cherry: w_ij + w_ik - w_jk <= 1/4  for every ordered triple (i; j, k).

z is penalised at DUMMY_PENALTY in the objective to force z=0 at optimality.
All branching is done via constraint-based branches (make_branch), never via
variable-bound changes on z.
"""

from itertools import combinations

import cplex
from .run_config import apply_solver_config

DUMMY_PENALTY = 1e6


def build_base_model(D, verbose=False):
    """Construct the CPLEX model.

    D: n×n numpy array (0-indexed distance matrix).

    Returns
    -------
    (c, pair_to_idx, pairs)
      c            : cplex.Cplex instance
      pair_to_idx  : dict (i,j) -> 0-based index in w-variable vector (i<j, 1-indexed)
      pairs        : list of (i,j) pairs in order (i<j, 1-indexed)
    """
    n = D.shape[0]
    pairs = list(combinations(range(1, n + 1), 2))
    np_pairs = len(pairs)
    pair_to_idx = {p: k for k, p in enumerate(pairs)}

    c = cplex.Cplex()
    apply_solver_config(c)
    if not verbose:
        c.set_log_stream(None)
        c.set_results_stream(None)
        c.set_warning_stream(None)

    c.objective.set_sense(c.objective.sense.minimize)

    # ---- w variables (continuous) -------------------------------------------
    f1_lb = 2.0 ** (-(n - 1))
    w_obj = [float(D[i - 1, j - 1]) for (i, j) in pairs]
    c.variables.add(
        obj=w_obj,
        lb=[f1_lb] * np_pairs,
        ub=[0.25] * np_pairs,
        names=[f"w_{i}_{j}" for (i, j) in pairs],
        types=['C'] * np_pairs,
    )

    # ---- dummy binary variable z --------------------------------------------
    c.variables.add(
        obj=[DUMMY_PENALTY],
        lb=[0.0],
        ub=[1.0],
        names=["z"],
        types=['B'],
    )

    # ---- Kraft equalities: sum_j w_ij = 1/2 ---------------------------------
    for i in range(1, n + 1):
        nbr_pairs = [(min(i, j), max(i, j)) for j in range(1, n + 1) if j != i]
        indices = [pair_to_idx[p] for p in nbr_pairs]
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=indices, val=[1.0] * len(indices))],
            senses=["E"],
            rhs=[0.5],
            names=[f"kraft_{i}"],
        )

    # ---- Double-cherry: w_ij + w_ik - w_jk <= 1/4 --------------------------
    for triple in combinations(range(1, n + 1), 3):
        a, b, cc = triple
        for apex, j, k in [(a, b, cc), (b, a, cc), (cc, a, b)]:
            ij = pair_to_idx[(min(apex, j), max(apex, j))]
            ik = pair_to_idx[(min(apex, k), max(apex, k))]
            jk = pair_to_idx[(min(j, k), max(j, k))]
            c.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[ij, ik, jk], val=[1.0, 1.0, -1.0])],
                senses=["L"],
                rhs=[0.25],
            )

    return c, pair_to_idx, pairs
