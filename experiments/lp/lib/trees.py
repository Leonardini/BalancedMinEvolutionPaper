"""Labelled unrooted binary trees: enumeration, path lengths, and the brute-force BME
optimum for small n."""
import numpy as np


def all_trees(n):
    """Every labelled unrooted binary tree on leaves 0..n-1 as an edge list,
    by stepwise insertion of leaf k onto every edge of each tree on k leaves."""
    def grow(edges, k, nxt):
        if k == n:
            yield edges
            return
        for idx, (u, v) in enumerate(edges):
            new = edges[:idx] + edges[idx + 1:] + [(u, nxt), (nxt, v), (nxt, k)]
            yield from grow(new, k + 1, nxt + 1)
    yield from grow([(0, n), (1, n), (2, n)], 3, n + 1)


def adjacency(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj


def leaf_distances(adj, n):
    tau = np.zeros((n, n), dtype=int)
    for s in range(n):
        dist = {s: 0}
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    stack.append(v)
        for t in range(n):
            tau[s, t] = dist[t]
    return tau


def brute_force_tree(D, progress=None):
    """(minimum, edges) of sum_{i<j} D_ij 2^-tau_ij over every tree, with the edge list
    of a minimising tree. `progress(k)` is called every 100,000 trees with the count
    so far."""
    n = len(D)
    best, arg = np.inf, None
    for k, edges in enumerate(all_trees(n), 1):
        tau = leaf_distances(adjacency(edges), n)
        val = np.sum(np.where(tau > 0, D * 2.0 ** (-tau.astype(float)), 0)) / 2
        if val < best:
            best, arg = val, edges
        if progress is not None and k % 100_000 == 0:
            progress(k)
    return best, arg


def brute_force_optimum(D, progress=None):
    """Minimum of sum_{i<j} D_ij 2^-tau_ij over every tree."""
    return brute_force_tree(D, progress)[0]


def splits_of_edges(edges, n):
    """The nontrivial splits of a tree on leaves 0..n-1 (internal nodes >= n), each as
    the frozenset of leaves on the side of an internal edge not containing leaf 0."""
    adj = adjacency(edges)
    out = []
    for u, v in edges:
        if u < n or v < n:
            continue
        side, stack = {v}, [v]
        while stack:
            a = stack.pop()
            for b in adj[a]:
                if b != u and b not in side:
                    side.add(b)
                    stack.append(b)
        leaves = frozenset(x for x in side if x < n)
        out.append(leaves if 0 not in leaves else frozenset(range(n)) - leaves)
    if len(out) != n - 3:
        raise ValueError(f"{len(out)} splits, expected n-3 = {n - 3}")
    return out
