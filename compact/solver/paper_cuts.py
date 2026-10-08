"""Valid inequalities from Catanzaro, Pesenti & Wolsey (2020),
"On the Balanced Minimum Evolution polytope", Discrete Optimization 36.

All are stated there in x-coordinates (x_ij = 2^{n-1} w_ij) and proven valid
for Conv(X) for every n >= 6.  Converted to our w-coordinates (w = x/2^{n-1}):

  (50) Prop 7 — triangle lower bound (valid, not facet):
        w_ab + w_ac + w_bc >= (3 + rho) * 2^{k-n},   k=floor(n/3), rho=n-3k
  (53) Prop 9 — two doubled disjoint pairs + a third (facet n=6):
        2 w_ab + 2 w_cd + w_ef >= 2^{4-n}            (a..f distinct)
  (54) Prop 9 — two disjoint pairs minus a heavily-weighted third
        (facet n=6,7,8):
        w_ab + w_cd - 2^{n-4} w_ef <= 1/8            (a..f distinct)

F4 (48) and F5 (49) of the same paper are the triangle+apex inequalities
already implemented in f45_separator.py; the paper proves them valid for all
n >= 6 as well, so they are re-enabled in the cut loop.

Separation for the 6-taxa families (53), (54) is bounded/greedy (restricted to
the most extreme pairs) so it stays cheap as n grows toward the supplement
instances (n up to ~30).  Because every inequality here is VALID, a heuristic
separator that misses some violations is still correct — it only ever finds
genuine cuts, never excludes a tree.
"""

import numpy as np

EPS = 1e-7


def p50_rhs(n):
    k = n // 3
    rho = n - 3 * k
    return (3 + rho) * (2.0 ** (k - n))


def separate_p50(w_vals, pairs, pair_to_idx, n, eps=EPS, max_cuts=200):
    """Triangle lower bound (50).  Returns list of (idx_ij, idx_ik, idx_jk)."""
    rhs = p50_rhs(n)
    w = np.asarray(w_vals, dtype=float)
    found = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = pair_to_idx[(a, b)]
            wab = w[ab]
            for cc in range(b + 1, n + 1):
                ac = pair_to_idx[(a, cc)]
                bc = pair_to_idx[(b, cc)]
                if wab + w[ac] + w[bc] < rhs - eps:
                    found.append((int(ab), int(ac), int(bc)))
                    if len(found) >= max_cuts:
                        return found
    return found


def _smallest_pairs(w, pairs, n, k):
    """Indices of the k pairs with smallest w (ascending)."""
    order = np.argsort(w)
    return order[:k].tolist()


def _largest_pairs(w, pairs, n, k):
    """Indices of the k pairs with largest w (descending)."""
    order = np.argsort(w)[::-1]
    return order[:k].tolist()


def separate_p53(w_vals, pairs, pair_to_idx, n, eps=EPS, max_cuts=200,
                 cand=None):
    """Inequality (53): 2 w_ab + 2 w_cd + w_ef >= 2^{4-n}, a..f distinct.

    Violations need three small disjoint pairs, so we search among the
    ``cand`` smallest-weight pairs (default ~6n).  Returns a list of
    (idx2a, idx2b, idx1c) where the first two carry coefficient 2 and the
    third coefficient 1 (largest of the triple, to minimise the LHS).
    """
    rhs = 2.0 ** (4 - n)
    w = np.asarray(w_vals, dtype=float)
    if cand is None:
        # Keep C(cand,3) cheap as n grows (cand=40 -> ~10k triples).
        cand = min(len(pairs), 40)
    small = _smallest_pairs(w, pairs, n, cand)

    found = []
    m = len(small)
    for ii in range(m):
        p1 = small[ii]
        a, b = pairs[p1]
        for jj in range(ii + 1, m):
            p2 = small[jj]
            cc, dd = pairs[p2]
            if cc in (a, b) or dd in (a, b):
                continue
            for kk in range(jj + 1, m):
                p3 = small[kk]
                e, f = pairs[p3]
                if e in (a, b, cc, dd) or f in (a, b, cc, dd):
                    continue
                ws = [(w[p1], p1), (w[p2], p2), (w[p3], p3)]
                ws.sort()
                # Largest weight gets coefficient 1 (minimises LHS).
                c1 = ws[2][1]
                c2a, c2b = ws[0][1], ws[1][1]
                if 2 * w[c2a] + 2 * w[c2b] + w[c1] < rhs - eps:
                    found.append((int(c2a), int(c2b), int(c1)))
                    if len(found) >= max_cuts:
                        return found
    return found


def separate_p54(w_vals, pairs, pair_to_idx, n, eps=EPS, max_cuts=200,
                 minus_cand=None, plus_cand=None):
    """Inequality (54): w_ab + w_cd - 2^{n-4} w_ef <= 1/8, a..f distinct.

    A violation needs w_ef near its minimum (the -2^{n-4} coefficient is
    large) and two large disjoint plus-pairs.  We try the ``minus_cand``
    smallest pairs as (e,f) and, for each, greedily take the two largest
    disjoint pairs from the ``plus_cand`` largest.  Returns a list of
    (idx_ab, idx_cd, idx_ef) where ab, cd carry +1 and ef carries -2^{n-4}.
    """
    coef = 2.0 ** (n - 4)
    rhs = 0.125
    w = np.asarray(w_vals, dtype=float)
    if minus_cand is None:
        minus_cand = min(len(pairs), 40)
    if plus_cand is None:
        plus_cand = min(len(pairs), 60)
    small = _smallest_pairs(w, pairs, n, minus_cand)
    big = _largest_pairs(w, pairs, n, plus_cand)

    found = []
    for pm in small:
        e, f = pairs[pm]
        neg = coef * w[pm]
        # Greedily pick the two largest disjoint plus-pairs avoiding {e,f}.
        chosen = []
        used = {e, f}
        for pb in big:
            i, j = pairs[pb]
            if i in used or j in used:
                continue
            chosen.append(pb)
            used.update((i, j))
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            continue
        p_ab, p_cd = chosen
        if w[p_ab] + w[p_cd] - neg > rhs + eps:
            found.append((int(p_ab), int(p_cd), int(pm)))
            if len(found) >= max_cuts:
                return found
    return found
