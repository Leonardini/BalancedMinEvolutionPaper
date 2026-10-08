"""W-space helpers: pair indexing, cut rows, min-cut, half-cuts, laminarity,
tree extraction, and branching candidate selection.

Convention: leaves are 1-indexed integers 1..n throughout this module.
"""

from itertools import combinations

import networkx as nx
import numpy as np

from .cut_enum import half_cuts

EPS = 1e-7


# ---------------------------------------------------------------------------
# Pair indexing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cut-row helpers
# ---------------------------------------------------------------------------

def cut_row_sparse(S, n, pair_to_idx):
    """Sparse indicator for the cut W[S, S^c].

    Returns (indices, values) where indices are w-variable indices that have
    nonzero coefficient 1 in the cut sum.
    """
    S = frozenset(S)
    Sc = frozenset(range(1, n + 1)) - S
    indices = []
    for i in S:
        for j in Sc:
            p = (min(i, j), max(i, j))
            indices.append(pair_to_idx[p])
    return indices, [1.0] * len(indices)


# ---------------------------------------------------------------------------
# W matrix
# ---------------------------------------------------------------------------

def build_W(w_vals, n):
    """Build n×n W matrix from the flat pair vector (0-indexed internally)."""
    W = np.zeros((n, n))
    idx = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            W[i - 1, j - 1] = W[j - 1, i - 1] = w_vals[idx]
            idx += 1
    return W


# ---------------------------------------------------------------------------
# Global min-cut via Stoer-Wagner (networkx)
# ---------------------------------------------------------------------------

def global_min_cut(W, n=None):
    """Global minimum bipartition cut via Stoer-Wagner.

    W: n×n numpy array (0-indexed).
    Returns (cut_value, frozenset_S) where S is one side (1-indexed leaves).
    """
    if n is None:
        n = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if w > 1e-15:
                G.add_edge(i, j, weight=w)

    if not nx.is_connected(G):
        # Disconnected graph: min-cut is 0; return any component (1-indexed).
        comp = next(nx.connected_components(G))
        return 0.0, frozenset(x + 1 for x in comp)

    cut_val, (part0, _part1) = nx.stoer_wagner(G)
    return float(cut_val), frozenset(x + 1 for x in part0)


# ---------------------------------------------------------------------------
# Half-cuts: bipartitions with cut value ≈ 1/2
# ---------------------------------------------------------------------------

def make_masks(n):
    """Canonical non-singleton bipartitions of {1..n}.

    Each bipartition is represented once (as the smaller side; ties broken by
    the side containing leaf 1).  Returns list of frozensets.
    """
    masks = []
    for size in range(2, n - 1):  # non-trivial: size 2..n-2
        if 2 * size < n:
            for S in combinations(range(1, n + 1), size):
                masks.append(frozenset(S))
        elif 2 * size == n:
            # Equal-size bipartition: keep the side containing leaf 1.
            for S in combinations(range(1, n + 1), size):
                if 1 in S:
                    masks.append(frozenset(S))
        # 2*size > n: covered by the complement (smaller side already listed).
    return masks


_MASK_CACHE = {}


def _mask_table(n):
    """make_masks(n), its 0/1 indicator matrix, and the index of each set (cached)."""
    if n not in _MASK_CACHE:
        masks = make_masks(n)
        M = np.zeros((len(masks), n))
        for r, S in enumerate(masks):
            M[r, [i - 1 for i in S]] = 1.0
        _MASK_CACHE[n] = (masks, M, {S: r for r, S in enumerate(masks)})
    return _MASK_CACHE[n]


def cut_values(W, n):
    """Cut value W[S] of every set S of make_masks(n), in that order:
    W[S] = s.W.1 - s.W.s for the 0/1 indicator vector s of S."""
    _, M, _ = _mask_table(n)
    W = np.asarray(W, dtype=float)
    return M @ W.sum(axis=1) - np.einsum("ij,ij->i", M @ W, M)


def find_half_cuts(W, n, eps=EPS):
    """Return canonical bipartitions with cut value in [1/2 - eps, 1/2 + eps], in the order of
    make_masks(n), for a point that satisfies every min-cut row. Listed as minimum cuts in
    polynomial time (cut_enum.half_cuts), not by scanning all 2^(n-1) bipartitions."""
    return half_cuts(W, n, eps)


# ---------------------------------------------------------------------------
# Laminarity
# ---------------------------------------------------------------------------

def first_crossing_pair(sets):
    """Return first incompatible (crossing) pair, or None if laminar."""
    for a in range(len(sets) - 1):
        for b in range(a + 1, len(sets)):
            A, B = sets[a], sets[b]
            inter = A & B
            if inter and not A <= B and not B <= A:
                return A, B
    return None


# ---------------------------------------------------------------------------
# True weights (exact tree metric from a split family)
# ---------------------------------------------------------------------------

def true_weights(splits, n):
    """Patristic distances and w-values for the tree defined by a split family.

    d[i,j] = 2 + #{S in splits : exactly one of i,j is in S}
    w[i,j] = 2^{-d[i,j]}

    Returns (d, w) as n×n numpy arrays (0-indexed).
    """
    d = np.full((n, n), 2.0)
    np.fill_diagonal(d, 0.0)
    for S in splits:
        S_set = frozenset(S)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if (i in S_set) != (j in S_set):
                    d[i - 1, j - 1] += 1.0
    w = np.power(2.0, -d)
    np.fill_diagonal(w, 0.0)
    return d, w


# ---------------------------------------------------------------------------
# Tree extraction
# ---------------------------------------------------------------------------

def cherry_reduce(W, n, tol=1e-7):
    """Identify all 1/2-cuts of W by iterated cherry contraction — O(n^3).

    A *cherry* is a pair (a,b) with W[a,b] = 1/4 and W[a,c] = W[b,c] for every
    other node c.  Contracting it to a node p with W'[p,c] = 2 W[a,c] preserves
    every non-trivial 1/2-cut (proof: with Kraft, any split separating a and b
    has value 1/2 + sum of positive cross terms > 1/2, so no non-trivial 1/2-cut
    splits a cherry; cuts keeping a,b together are value-preserved exactly).

    Every binary tree has >= 2 cherries, so a genuine tree-w reduces all the way
    to a 3-leaf star, yielding its n-3 splits; if no cherry exists at some stage,
    W is not tree-realizable.  This replaces the O(2^n) bipartition enumeration.

    Returns a set of canonical splits (frozensets of 1-indexed leaves), or None
    if W is not a tree-w.
    """
    leaves = {i: frozenset([i]) for i in range(1, n + 1)}
    w = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            w[(i, j)] = W[i - 1, j - 1]

    def gw(i, j):
        return w[(i, j)] if i < j else w[(j, i)]

    splits = []
    cur = list(range(1, n + 1))
    nxt = n + 1
    while len(cur) > 3:
        found = None
        for ai in range(len(cur)):
            a = cur[ai]
            for bi in range(ai + 1, len(cur)):
                b = cur[bi]
                if abs(gw(a, b) - 0.25) > tol:
                    continue
                if all(abs(gw(a, c) - gw(b, c)) < tol
                       for c in cur if c != a and c != b):
                    found = (a, b)
                    break
            if found is not None:
                break
        if found is None:
            return None  # no cherry -> not tree-realizable
        a, b = found
        p = nxt
        nxt += 1
        leaves[p] = leaves[a] | leaves[b]
        splits.append(canonical_S(leaves[p], n))
        for c in cur:
            if c == a or c == b:
                continue
            w[(min(p, c), max(p, c))] = 2.0 * gw(a, c)
        cur = [c for c in cur if c != a and c != b] + [p]
    return set(splits)


def cherry_contract(W, n, tol=1e-7):
    """Partially contract W by collapsing every cherry, cascading.

    Returns (W_red, leaf_sets, cherry_splits):
      W_red        : m x m reduced weight matrix (m <= n)
      leaf_sets    : list of m frozensets — original leaves behind each reduced
                     node (reduced node k is labelled k+1 in 1..m)
      cherry_splits: canonical splits (over original leaves) of the contracted
                     cherries — these are exactly the non-trivial 1/2-cuts that
                     become trivial (singleton) in the reduced instance.

    A pair (a,b) is a cherry iff W[a,b]=1/4 and W[a,c]=W[b,c] for all other c.
    At an LP optimum the double-cherry constraints make W[a,b]=1/4 sufficient
    (it forces W[a,c]=W[b,c]); the second condition is still checked because it
    need not hold on the reduced matrix after the first contraction.

    By the cherry lemma, contraction preserves every non-trivial 1/2-cut (and
    every cut value of bipartitions that keep a cherry together), so the
    exponential searches (find_half_cuts / pick_branching_cut) can be run on the
    smaller W_red and lifted back.  m == n exactly when W has no cherry.
    """
    label_leaves = {i: frozenset([i]) for i in range(1, n + 1)}
    w = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            w[(i, j)] = W[i - 1, j - 1]

    def gw(i, j):
        return w[(i, j)] if i < j else w[(j, i)]

    cur = list(range(1, n + 1))
    nxt = n + 1
    cherry_splits = []
    changed = True
    while changed and len(cur) > 3:
        changed = False
        for ai in range(len(cur)):
            a = cur[ai]
            mate = None
            for bi in range(ai + 1, len(cur)):
                b = cur[bi]
                if abs(gw(a, b) - 0.25) > tol:
                    continue
                if all(abs(gw(a, c) - gw(b, c)) < tol
                       for c in cur if c != a and c != b):
                    mate = b
                    break
            if mate is not None:
                b = mate
                p = nxt
                nxt += 1
                label_leaves[p] = label_leaves[a] | label_leaves[b]
                cherry_splits.append(canonical_S(label_leaves[p], n))
                for c in cur:
                    if c != a and c != b:
                        w[(min(p, c), max(p, c))] = 2.0 * gw(a, c)
                cur = [c for c in cur if c != a and c != b] + [p]
                changed = True
                break

    m = len(cur)
    leaf_sets = [label_leaves[c] for c in cur]
    W_red = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            W_red[i, j] = W_red[j, i] = gw(cur[i], cur[j])
    return W_red, leaf_sets, cherry_splits


def _lift_subset(S_red, leaf_sets, n):
    """Lift a subset of reduced node labels (1..m) to a canonical original split."""
    orig = frozenset().union(*(leaf_sets[i - 1] for i in S_red))
    return canonical_S(orig, n)


def _project_split(S, leaf_sets, m):
    """Project an original-leaf split S onto reduced labels, or None if S
    straddles a contracted node (so it is not representable in the reduced
    instance)."""
    S = frozenset(S)
    red = set()
    for i in range(m):
        ls = leaf_sets[i]
        if ls <= S:
            red.add(i + 1)
        elif ls & S:
            return None
    return frozenset(red)


def pick_branching_cut_fast(W, n, forced, forbidden, tol=EPS):
    """O(n^2) branching: branch on a 'fractional cherry' pair when one exists.

    Because W[{a,b}] = 1 - 2 W[a,b], a pair with W[a,b] in (1/8, 1/4) has the
    2-element cut value W[{a,b}] in (1/2, 3/4) — a fractional split.  Branching
    on the split {a,b} (force W=1/2 == commit the cherry, vs forbid W>=3/4)
    excludes the current LP point and fits the existing split machinery, found
    in O(n^2) instead of the O(2^n) bipartition enumeration.

    Picks the most fractional such pair (W[a,b] closest to 3/16).  Falls back to
    settling the face (face.py) only when no fractional cherry exists.
    """
    excluded = {bipartition_key(s, n) for s in forced} | \
               {bipartition_key(s, n) for s in forbidden}
    best = None  # (dist_from_3/16, a, b)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            wab = W[a - 1, b - 1]
            if 0.125 + tol < wab < 0.25 - tol:
                S = canonical_S(frozenset((a, b)), n)
                if bipartition_key(S, n) in excluded:
                    continue
                dist = abs(wab - 0.1875)
                if best is None or dist < best[0]:
                    best = (dist, a, b)
    if best is not None:
        _, a, b = best
        S = canonical_S(frozenset((a, b)), n)
        wab = W[a - 1, b - 1]
        return {'mode': 'symmetric', 'S': S, 'value': 1.0 - 2.0 * wab}
    # No fractional cherry.  For small n the exhaustive bipartition search is
    # cheap (make_masks(n) <= 2^{n-1}) and EXACT — it finds a fractional cut
    # whenever the LP optimum is not a tree, so it avoids spurious mode='none'
    # (which would send the node to face.py).  Above the threshold fall back
    # to the cherry-contracted search.
    if n <= 20:
        return pick_branching_cut(W, n, forced, forbidden, tol=tol)
    return pick_branching_cut_cherry(W, n, forced, forbidden, tol=tol)


def pick_branching_cut_cherry(W, n, forced, forbidden, tol=EPS):
    """pick_branching_cut run on the cherry-contracted instance.

    Contracts cherries (O(n^3)), runs the exponential branching search on the
    smaller reduced matrix (O(2^m)), and lifts the chosen cut back to original
    leaves.  Falls back to the full search when no cherry exists (m == n).
    """
    W_red, leaf_sets, _ = cherry_contract(W, n)
    m = len(leaf_sets)
    # The fallback enumeration is O(2^m); only run it when m is small enough.
    # Above the threshold, decline to branch (mode none) — the caller then uses
    # its heuristic completion.  In practice this is rarely reached because the
    # O(n^2) fractional-cherry rule handles almost every node.
    if m > 20:
        return {'mode': 'none'}
    if m == n:
        return pick_branching_cut(W, n, forced, forbidden, tol=tol)

    forced_red = []
    for s in forced:
        r = _project_split(s, leaf_sets, m)
        if r is not None and 1 < len(r) < m - 1:
            forced_red.append(r)
    forbidden_red = []
    for s in forbidden:
        r = _project_split(s, leaf_sets, m)
        if r is not None and 1 < len(r) < m - 1:
            forbidden_red.append(r)

    b = pick_branching_cut(W_red, m, forced_red, forbidden_red, tol=tol)
    if b['mode'] == 'none':
        return {'mode': 'none'}
    return {'mode': b['mode'],
            'S': _lift_subset(b['S'], leaf_sets, n),
            'value': b.get('value')}


def extract_tree_cherry(W, n, D, tol=1e-6):
    """Tree recognition + extraction in O(n^3) via cherry contraction.

    Same return contract as ``extract_tree`` (dict {splits, w, d, objective} or
    None), but takes the W matrix directly and needs no exponential half-cut
    enumeration.  A final consistency check confirms the reconstructed tree-w
    equals W, rejecting the rare case where cherries exist at every step yet the
    internal (non-cherry) distances are not tree-consistent.
    """
    splits = cherry_reduce(W, n, tol=tol)
    if splits is None:
        return None
    d, w = true_weights(splits, n)
    # Reconstructed tree-w must match W (cherry steps only constrain cherry
    # pairs; verify the rest).
    if np.max(np.abs(w - W)) > tol:
        return None
    obj = float(np.sum(D * w)) / 2.0
    return {'splits': list(splits), 'w': w, 'd': d, 'objective': obj}


# ---------------------------------------------------------------------------
# Bipartition canonical form
# ---------------------------------------------------------------------------

def canonical_S(S, n):
    """Canonical representative: smaller side; ties broken by side with leaf 1."""
    S = frozenset(S)
    Sc = frozenset(range(1, n + 1)) - S
    if len(S) < len(Sc):
        return S
    if len(S) > len(Sc):
        return Sc
    return S if 1 in S else Sc


bipartition_key = canonical_S  # alias


# ---------------------------------------------------------------------------
# Branching candidate selection (Dakin + asymmetric fallback)
# ---------------------------------------------------------------------------

def pick_branching_cut(W, n, forced, forbidden, half_cuts=None, tol=EPS):
    """Select a branching bipartition.

    Returns dict with keys:
      mode  : 'symmetric' | 'asymmetric' | 'none'
      S     : frozenset (the chosen bipartition, canonical side)  [absent for 'none']
      value : float  [absent for 'none']

    'symmetric'  — W[S] ∈ (1/2, 3/4), closest to 5/8 (Dakin most-fractional)
    'asymmetric' — W[S] = 3/4; only valid when |halfCuts| == n-3 (binary tree LP)
    'none'       — no candidate found; caller handles by enumeration / NNI
    """
    if half_cuts is None:
        half_cuts = find_half_cuts(W, n)

    excluded = {bipartition_key(s, n) for s in forced} | \
               {bipartition_key(s, n) for s in forbidden}

    masks, M, index = _mask_table(n)
    vals = cut_values(W, n)
    allowed = np.ones(len(masks), dtype=bool)
    for key in excluded:
        allowed[index[key]] = False
    sym_best = None   # (dist_from_5/8, S, val)
    asym_best = None  # (len_S, S, val)

    # The first set (in make_masks order) closest to 5/8, as in a scan with strict <.
    sym = np.flatnonzero(allowed & (vals > 0.5 + tol) & (vals < 0.75 - tol))
    if sym.size:
        r = sym[np.argmin(np.abs(vals[sym] - 0.625))]
        sym_best = (abs(vals[r] - 0.625), masks[r], float(vals[r]))
    asym = np.flatnonzero(allowed & ~((vals > 0.5 + tol) & (vals < 0.75 - tol))
                          & (np.abs(vals - 0.75) < tol))
    if asym.size:
        r = asym[np.argmin(M[asym].sum(axis=1))]
        asym_best = (len(masks[r]), masks[r], float(vals[r]))

    if sym_best is not None:
        return {'mode': 'symmetric', 'S': sym_best[1], 'value': sym_best[2]}

    # Asymmetric oracle is only valid when the LP solution is a binary tree.
    if asym_best is not None and len(half_cuts) == n - 3:
        return {'mode': 'asymmetric', 'S': asym_best[1], 'value': asym_best[2]}

    return {'mode': 'none'}
