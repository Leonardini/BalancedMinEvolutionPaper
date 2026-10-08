"""PM facet separation for the BME polytope.

Even n: min-weight perfect matching on {1..n}.
  c_n = (k+2) * 2^{-(k+2)},  k = n/2.

Odd n: pseudo-vertex near-matching.  Extend to {0, 1..n} with
  w[0, ell] = min_{j != ell} w[ell, j],
  then find a perfect matching on {0..n}.
  c_n = (k+6) * 2^{-(k+3)},  k = (n-1)/2.

Separation: if the minimum PM weight < c_n - eps, add the cut
  sum_{(i,j) in M} w_ij  >=  c_n.
"""

import networkx as nx
import numpy as np

EPS = 1e-7


def pm_constant(n):
    """Facet RHS constant c_n."""
    k = n // 2
    rho = n - 2 * k  # 0 for even, 1 for odd
    return (k + 2 + 4 * rho) * 2.0 ** (-(k + 2 + rho))


def separate_pm(w_vals, pairs, pair_to_idx, n, eps=EPS):
    """PM facet separation.

    Returns (val, cut_indices, cut_values, Cn).
    cut_indices and cut_values are None when no violation is found.
    """
    Cn = pm_constant(n)
    k = n // 2
    rho = n - 2 * k

    if rho == 0:
        return _separate_pm_even(w_vals, pairs, pair_to_idx, n, Cn, eps)
    else:
        return _separate_pm_odd(w_vals, pairs, pair_to_idx, n, Cn, eps)


def _separate_pm_even(w_vals, pairs, pair_to_idx, n, Cn, eps):
    G = nx.Graph()
    for ki, (i, j) in enumerate(pairs):
        G.add_edge(i, j, weight=float(w_vals[ki]))

    matching = nx.min_weight_matching(G)
    val = sum(G[u][v]['weight'] for u, v in matching)

    if val < Cn - eps:
        indices = [pair_to_idx[(min(u, v), max(u, v))] for u, v in matching]
        values = [1.0] * len(indices)
        return val, indices, values, Cn

    return val, None, None, Cn


def _separate_pm_odd(w_vals, pairs, pair_to_idx, n, Cn, eps):
    # Build W matrix for minimum-edge lookup.
    W = np.zeros((n + 1, n + 1))  # row/col 0 = pseudo-vertex; 1..n = real leaves
    for ki, (i, j) in enumerate(pairs):
        W[i, j] = W[j, i] = float(w_vals[ki])

    # Pseudo-vertex edges: w[0, ell] = min_{j != ell} W[ell, j]
    min_edge = {}
    for ell in range(1, n + 1):
        min_edge[ell] = float(min(W[ell, j] for j in range(1, n + 1) if j != ell))
        W[0, ell] = W[ell, 0] = min_edge[ell]

    G = nx.Graph()
    for ki, (i, j) in enumerate(pairs):
        G.add_edge(i, j, weight=float(w_vals[ki]))
    for ell in range(1, n + 1):
        G.add_edge(0, ell, weight=min_edge[ell])

    matching = nx.min_weight_matching(G)
    val = sum(G[u][v]['weight'] for u, v in matching)

    if val < Cn - eps:
        # Find ell_star (real leaf matched to pseudo-vertex 0).
        ell_star = None
        real_pairs_in_matching = []
        for u, v in matching:
            if u == 0:
                ell_star = v
            elif v == 0:
                ell_star = u
            else:
                real_pairs_in_matching.append((min(u, v), max(u, v)))

        # j_star = argmin w[ell_star, j] for j != ell_star.
        j_star = min(
            (j for j in range(1, n + 1) if j != ell_star),
            key=lambda j: W[ell_star, j]
        )

        # Cut row: near-matching pairs (coefficient 1) + pendant (coefficient 1).
        indices = [pair_to_idx[p] for p in real_pairs_in_matching]
        values = [1.0] * len(indices)
        pendant = (min(ell_star, j_star), max(ell_star, j_star))
        pendant_idx = pair_to_idx[pendant]
        if pendant_idx in indices:
            values[indices.index(pendant_idx)] += 1.0
        else:
            indices.append(pendant_idx)
            values.append(1.0)

        return val, indices, values, Cn

    return val, None, None, Cn
