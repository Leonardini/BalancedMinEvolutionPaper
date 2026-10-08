"""F7 facet separation for the BME polytope.

F7 (Catanzaro [28], Table 1, row 7): for any 5 distinct leaves i1,i2,i3,i4,i5,
  w_{i1,i2} + w_{i1,i3} + w_{i2,i3} + w_{i4,i5} >= (4+rho) * 2^{k-n}
where k = floor(n/3), rho = n mod 3.

The LHS is a triangle on (i1,i2,i3) plus one additional pair (i4,i5).
Valid for all n >= 6 per reference [28].

Separation: for each triple (a,b,c), find the min-weight pair (d,e) disjoint
from {a,b,c}; add the cut when the total is < RHS - eps.
"""

import numpy as np

EPS = 1e-7


def f7_rhs(n):
    k = n // 3
    rho = n % 3
    return (4 + rho) * (2.0 ** (k - n))


def separate_f7(w_vals, pairs, pair_to_idx, n, eps=EPS, max_cuts=200):
    """Find violated F7 cuts.

    Returns list of (idx_ab, idx_ac, idx_bc, idx_de) — indices into w_vals,
    representing the constraint w[ab]+w[ac]+w[bc]+w[de] >= f7_rhs(n).
    """
    rhs = f7_rhs(n)
    w = np.asarray(w_vals, dtype=float)

    # Sort pair indices by w ascending; first disjoint entry = minimum.
    sorted_idx = np.argsort(w).tolist()

    found = []

    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = pair_to_idx[(a, b)]
            for c in range(b + 1, n + 1):
                ac = pair_to_idx[(a, c)]
                bc = pair_to_idx[(b, c)]
                tri_sum = w[ab] + w[ac] + w[bc]
                if tri_sum >= rhs - eps:
                    continue  # even paired with zero extra, can't violate

                tri_set = {a, b, c}
                for pidx in sorted_idx:
                    pi, pj = pairs[pidx]
                    if pi not in tri_set and pj not in tri_set:
                        if tri_sum + w[pidx] < rhs - eps:
                            found.append((int(ab), int(ac), int(bc), int(pidx)))
                        break  # sorted ascending: first disjoint is the minimum

                if len(found) >= max_cuts:
                    return found

    return found
