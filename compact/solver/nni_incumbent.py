"""Neighbor-Joining + NNI hill-climb for initial incumbents and mode=none fallback.

Tree representation: networkx.Graph with leaves 1..n (integers) and internal
nodes n+1..2n-2 (integers).  All edges are unweighted (unit length for BME).
"""

import networkx as nx
import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------------------
# Neighbor-Joining
# ---------------------------------------------------------------------------

def neighbor_joining(D):
    """Standard NJ algorithm.

    D: n×n numpy array (symmetric, zero diagonal, 0-indexed).
    Returns networkx.Graph with leaves 1..n, internal nodes n+1..2n-2.
    """
    n = D.shape[0]
    if n == 1:
        G = nx.Graph(); G.add_node(1); return G
    if n == 2:
        G = nx.Graph(); G.add_edge(1, 2); return G

    # Working distance dictionary (0-indexed node labels; leaves are 0..n-1).
    dist = {}
    for i in range(n):
        for j in range(n):
            dist[i, j] = float(D[i, j])

    active = list(range(n))
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    next_node = n  # next internal node (0-indexed)

    while len(active) > 3:
        k = len(active)
        row_sums = {i: sum(dist[i, j] for j in active if j != i) for i in active}

        best_q = float('inf')
        best_i = best_j = None
        for ai in range(len(active)):
            for aj in range(ai + 1, len(active)):
                iv, jv = active[ai], active[aj]
                q = (k - 2) * dist[iv, jv] - row_sums[iv] - row_sums[jv]
                if q < best_q:
                    best_q, best_i, best_j = q, iv, jv

        u = next_node; next_node += 1
        G.add_node(u + 1)           # 1-indexed in the graph
        G.add_edge(best_i + 1, u + 1)
        G.add_edge(best_j + 1, u + 1)

        for m in active:
            if m not in (best_i, best_j):
                new_d = (dist[best_i, m] + dist[best_j, m] - dist[best_i, best_j]) / 2.0
                dist[u, m] = dist[m, u] = new_d

        active = [x for x in active if x not in (best_i, best_j)]
        active.append(u)

    # Connect final 3 nodes to a center node.
    center = next_node
    G.add_node(center + 1)
    for node in active:
        G.add_edge(node + 1, center + 1)

    return G


# ---------------------------------------------------------------------------
# BME objective
# ---------------------------------------------------------------------------

def w_from_tree(tree, n):
    """n×n BME weight matrix W where W[i,j] = 2^{-path_length(i,j)}."""
    W = np.zeros((n, n))
    paths = dict(nx.all_pairs_shortest_path_length(tree))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                W[i - 1, j - 1] = 2.0 ** (-paths[i][j])
    return W


def bme_obj_from_tree(tree, D, n):
    """BME objective for a tree.  D is 0-indexed n×n array."""
    W = w_from_tree(tree, n)
    return float(np.sum(D * W)) / 2.0  # both symmetric


# ---------------------------------------------------------------------------
# NNI neighbours
# ---------------------------------------------------------------------------

def nni_neighbors(tree, n):
    """All 2*(n-3) NNI neighbours of an unrooted binary tree.

    Swaps subtrees around each internal edge (u,v) where both u and v
    are internal nodes (label > n).
    """
    neighbors = []
    internal_edges = [(u, v) for u, v in tree.edges() if u > n and v > n]
    for u, v in internal_edges:
        u_nbrs = [x for x in tree.neighbors(u) if x != v]
        v_nbrs = [x for x in tree.neighbors(v) if x != u]
        if len(u_nbrs) != 2 or len(v_nbrs) != 2:
            continue  # skip non-binary nodes (shouldn't happen)
        a1, a2 = u_nbrs
        b1, b2 = v_nbrs
        # NNI 1: swap subtree(a1) with subtree(b1)
        t1 = tree.copy()
        t1.remove_edge(u, a1); t1.remove_edge(v, b1)
        t1.add_edge(u, b1);    t1.add_edge(v, a1)
        neighbors.append(t1)
        # NNI 2: swap subtree(a1) with subtree(b2)
        t2 = tree.copy()
        t2.remove_edge(u, a1); t2.remove_edge(v, b2)
        t2.add_edge(u, b2);    t2.add_edge(v, a1)
        neighbors.append(t2)
    return neighbors


def _tree_key(tree, n):
    """Canonical frozenset-of-frozensets key for an unrooted tree topology."""
    splits = set()
    for u, v in tree.edges():
        # BFS from u avoiding edge (u,v)
        visited_u = {u}
        queue = [u]
        while queue:
            node = queue.pop()
            for nbr in tree.neighbors(node):
                if nbr == v and node == u:
                    continue  # don't cross the removed edge
                if nbr not in visited_u:
                    visited_u.add(nbr)
                    queue.append(nbr)
        leaves_u = frozenset(x for x in visited_u if x <= n)
        leaves_v = frozenset(x for x in tree.nodes() if x <= n) - leaves_u
        if len(leaves_u) >= 2 and len(leaves_v) >= 2:
            canonical = (leaves_u if (len(leaves_u) < len(leaves_v) or
                         (len(leaves_u) == len(leaves_v) and min(leaves_u) < min(leaves_v)))
                         else leaves_v)
            splits.add(canonical)
    return frozenset(splits)


# ---------------------------------------------------------------------------
# NNI hill-climb (greedy)
# ---------------------------------------------------------------------------

def nni_hill_climb(D, tree, n, tol=EPS, max_iter=1000):
    """Greedy NNI hill-climb: repeatedly apply the best single NNI move.

    Returns dict {objective, tree, w}.
    """
    base_obj = bme_obj_from_tree(tree, D, n)
    for _ in range(max_iter):
        nbrs = nni_neighbors(tree, n)
        if not nbrs:
            break
        best_nbr = None
        best_obj = base_obj
        for nbr in nbrs:
            obj = bme_obj_from_tree(nbr, D, n)
            if obj < best_obj - tol:
                best_obj = obj
                best_nbr = nbr
        if best_nbr is None:
            break
        tree = best_nbr
        base_obj = best_obj
    return {'objective': base_obj, 'tree': tree, 'w': w_from_tree(tree, n)}


# ---------------------------------------------------------------------------
# Exhaustive NNI improvement (DFS over all strictly-improving branches)
# ---------------------------------------------------------------------------

def nni_exhaustive_improvement(D, tree, n, tol=EPS):
    """Explore ALL strictly-improving NNI branches (DFS, deduplication).

    Returns the best tree found across all improvement paths.
    """
    obj0 = bme_obj_from_tree(tree, D, n)
    best = {'objective': obj0, 'tree': tree, 'w': w_from_tree(tree, n)}
    stack = [(tree, obj0)]
    visited = {_tree_key(tree, n)}

    while stack:
        cur_tree, cur_obj = stack.pop()
        for nbr in nni_neighbors(cur_tree, n):
            key = _tree_key(nbr, n)
            if key in visited:
                continue
            visited.add(key)
            obj = bme_obj_from_tree(nbr, D, n)
            if obj < cur_obj - tol:
                stack.append((nbr, obj))
                if obj < best['objective'] - tol:
                    best = {'objective': obj, 'tree': nbr, 'w': w_from_tree(nbr, n)}

    return best


# ---------------------------------------------------------------------------
# NJ + exhaustive NNI (initial incumbent)
# ---------------------------------------------------------------------------

_EXHAUSTIVE_NNI_MAX_N = 20


def nj_plus_nni(D, n):
    """Neighbor-Joining followed by NNI improvement.

    Uses exhaustive DFS for n <= _EXHAUSTIVE_NNI_MAX_N; greedy hill-climb
    for larger n (exhaustive DFS is intractable for n > 20).
    Returns dict {objective, tree, w}.
    """
    tree = neighbor_joining(D)
    if n <= _EXHAUSTIVE_NNI_MAX_N:
        return nni_exhaustive_improvement(D, tree, n)
    result = nni_hill_climb(D, tree, n)
    result['w'] = w_from_tree(result['tree'], n)
    return result


# ---------------------------------------------------------------------------
# Tree from a partial split family (for mode=none completion)
# ---------------------------------------------------------------------------

def complete_splits_to_tree(splits, n, D=None):
    """Build a binary tree containing all splits in `splits`.

    Uses NJ on the additive tree metric derived from true_weights(splits, n).
    If splits is empty, runs NJ on a uniform distance matrix.
    """
    from .w_space import true_weights

    if not splits:
        d_uniform = np.ones((n, n)) - np.eye(n)
        if D is not None:
            return neighbor_joining(D)
        return neighbor_joining(d_uniform)
    d, _ = true_weights(splits, n)
    return neighbor_joining(d)


# ---------------------------------------------------------------------------
# Full binary tree enumeration (for mode=none certification, small n)
# ---------------------------------------------------------------------------

def _double_factorial_minus5(n):
    """(2n-5)!! = number of unrooted binary trees on n leaves."""
    if n <= 2:
        return 0
    result = 1
    for k in range(1, 2 * n - 4, 2):  # 1, 3, ..., 2n-5
        result *= k
    return result


def _gen_all_trees(n_leaves):
    """Generate all unrooted binary trees on leaves 1..n_leaves as edge lists.

    Internal nodes are negative integers (-1, -2, ...) to avoid collisions
    with positive leaf labels.  Each tree is a list of (u, v) pairs.
    """
    if n_leaves == 3:
        yield [(1, -1), (2, -1), (3, -1)]
        return

    new_leaf = n_leaves
    for base in _gen_all_trees(n_leaves - 1):
        # Find the minimum (most negative) internal node in the base tree.
        internals = [v for u, v in base if v < 0] + [u for u, v in base if u < 0]
        base_min_int = min(internals) if internals else 0
        # Use a distinct new_int for each edge to avoid label collisions
        # (all new trees are independent objects, so reuse is safe across trees).
        for eu, ev in base:
            new_int = base_min_int - 1  # same label OK: each yield is a distinct list
            new_tree = [(a, b) for a, b in base if (a, b) != (eu, ev)]
            new_tree.extend([(eu, new_int), (new_int, ev), (new_leaf, new_int)])
            yield new_tree


