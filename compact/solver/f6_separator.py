"""Lazy F6 facet separation for the BME polytope.

F6 (Catanzaro Table 1): for any two disjoint pairs (i1,i2) and (i3,i4),
  2^{n-4} * w_{i1,i2}  >=  w_{i3,i4}.

Separation: scan (lo, hi) pairs in ascending w_lo / descending w_hi order;
early-exit when the best possible violation is exhausted.
"""

import numpy as np

EPS = 1e-7


def separate_f6(w_vals, pairs, n, eps=EPS, max_cuts=200):
    """Find violated F6 cuts.

    Returns list of (lo_idx, hi_idx) — indices into w_vals — representing
    cuts  2^{n-4} * w_vals[lo_idx] - w_vals[hi_idx] >= 0.
    Empty list when no violation is found.
    """
    n_pairs = len(pairs)
    coef = 2.0 ** (n - 4)
    w = np.asarray(w_vals, dtype=float)
    order = np.argsort(w)   # ascending
    w_hi_max = w[order[-1]]

    found = []
    for k_lo in range(n_pairs):
        i_lo = order[k_lo]
        w_lo = w[i_lo]
        if w_hi_max - coef * w_lo <= eps:
            break  # all remaining lo values are even larger; no more violations
        p_lo = pairs[i_lo]

        for k_hi in range(n_pairs - 1, -1, -1):
            i_hi = order[k_hi]
            if w[i_hi] - coef * w_lo <= eps:
                break
            if i_hi == i_lo:
                continue
            p_hi = pairs[i_hi]
            # Disjointness check.
            if p_hi[0] in p_lo or p_hi[1] in p_lo:
                continue
            found.append((i_lo, i_hi))
            if len(found) >= max_cuts:
                return found

    return found
