"""F4 and F5 facet separation for the BME polytope.

Both inequalities are asymmetric 3-leaf cuts with an apex leaf i1:

F4 (Table 1, row 4): 2^{n-5}*(w_{i1,i2}+w_{i1,i3}) + w_{i2,i3} >= 5/16
F5 (Table 1, row 5): 2^{n-4}*(w_{i1,i2}+w_{i1,i3}) + w_{i2,i3} >= 1/2

Both are claimed facet-defining for n=6 by Catanzaro et al. [28].
Validity for general n >= 6 is checked empirically; if invalid they cut off
tree-w vectors and the cut loop must skip them.

Separators return lists of (idx_apex_b, idx_apex_c, idx_bc):
  - first two indices correspond to the apex pairs (coefficient 2^{n-5} or 2^{n-4})
  - third index corresponds to the non-apex pair (coefficient 1)
"""

import numpy as np

EPS = 1e-7


def separate_f4(w_vals, pairs, pair_to_idx, n, eps=EPS, max_cuts=200):
    """Find violated F4 cuts.

    Returns list of (idx_ab, idx_ac, idx_bc) where the constraint is
    2^{n-5}*w[ab] + 2^{n-5}*w[ac] + w[bc] >= 5/16.
    """
    coef = 2.0 ** (n - 5)
    rhs = 5.0 / 16.0
    w = np.asarray(w_vals, dtype=float)

    found = []
    for apex in range(1, n + 1):
        for b in range(1, n + 1):
            if b == apex:
                continue
            for c in range(b + 1, n + 1):
                if c == apex:
                    continue
                ab = pair_to_idx[(min(apex, b), max(apex, b))]
                ac = pair_to_idx[(min(apex, c), max(apex, c))]
                bc = pair_to_idx[(b, c)]
                if coef * (w[ab] + w[ac]) + w[bc] < rhs - eps:
                    found.append((int(ab), int(ac), int(bc)))
                    if len(found) >= max_cuts:
                        return found
    return found


def separate_f5(w_vals, pairs, pair_to_idx, n, eps=EPS, max_cuts=200):
    """Find violated F5 cuts.

    Returns list of (idx_ab, idx_ac, idx_bc) where the constraint is
    2^{n-4}*w[ab] + 2^{n-4}*w[ac] + w[bc] >= 1/2.
    """
    coef = 2.0 ** (n - 4)
    rhs = 0.5
    w = np.asarray(w_vals, dtype=float)

    found = []
    for apex in range(1, n + 1):
        for b in range(1, n + 1):
            if b == apex:
                continue
            for c in range(b + 1, n + 1):
                if c == apex:
                    continue
                ab = pair_to_idx[(min(apex, b), max(apex, b))]
                ac = pair_to_idx[(min(apex, c), max(apex, c))]
                bc = pair_to_idx[(b, c)]
                if coef * (w[ab] + w[ac]) + w[bc] < rhs - eps:
                    found.append((int(ab), int(ac), int(bc)))
                    if len(found) >= max_cuts:
                        return found
    return found
