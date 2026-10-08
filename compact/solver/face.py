"""Settle a node whose LP point has no split to branch on: enumerate its face exactly when
it is small, otherwise name a split to branch on.

The face of a node is the set of trees that contain every committed (forced) split and no
forbidden one. Committed cherries are contracted, repeatedly: a committed split that becomes
a cherry of two clusters after earlier contractions is contracted as well. Each cluster then
has a fixed inner tree. If leaf i lies e_i edges below the root of its cluster, a tree T in
the face has, for i in cluster A and j in cluster B != A, path length e_i + e_j + p'(A, B),
where p' is the path length between A and B in the reduced tree on the clusters. So

    L(T) = C + sum over cluster pairs A < B of 2^(-p'(A,B)) * d'(A, B),
    d'(A, B) = sum over i in A, j in B of 2^(-e_i - e_j) * D[i, j],

with C the part within clusters, the same for every tree in the face. When at most
ENUM_MAX_CLUSTERS clusters remain, face_enum.c enumerates the reduced trees exactly, and the
best is lifted and its objective recomputed from scratch and checked against C + its reduced
objective. Otherwise the node is branched on a split S that is undecided (neither committed
nor forbidden), crosses no committed split and has cut value W[S] = 1/2 at the current point:
its forbid child cuts off the point and its commit child contracts further. If there is no such
split, the node is branched on the cherries of a terminal vertex of the committed skeleton
(_cherry_children): every child commits one pair of clusters, so it cuts off the point and has
one cluster fewer.
"""
import ctypes
import math
import os
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

from .nni_incumbent import complete_splits_to_tree, nni_hill_climb
from .cut_enum import cuts_in_order
from .w_space import cherry_contract, true_weights

# Faces with at most this many clusters are settled by enumeration. BME_ENUM_MAX_CLUSTERS
# lowers it for tests only, so that face branching occurs on instances small enough to check
# against every tree.
ENUM_MAX_CLUSTERS = int(os.environ.get("BME_ENUM_MAX_CLUSTERS", "10"))
EPS = 1e-6
CHECK_RTOL = 1e-9

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "face_enum.c"
_LIB = _HERE / "_face_enum.so"
_lib = None


def _load():
    global _lib
    if _lib is None:
        if not _LIB.exists() or _LIB.stat().st_mtime < _SRC.stat().st_mtime:
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(_LIB), str(_SRC)], check=True)
        lib = ctypes.CDLL(str(_LIB))
        lib.enumerate_face.restype = ctypes.c_longlong
        lib.enumerate_face.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint), ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint)]
        _lib = lib
    return _lib


def _side(S, full):
    """The side of split S (either side given) that does not contain leaf 1."""
    S = frozenset(S)
    return full - S if 1 in S else S


def contract_committed(forced, n):
    """Contract committed cherries repeatedly.

    Returns (clusters, edges_below, within, consumed): the clusters as frozensets of leaves,
    e_i for every leaf i (dict), the within-cluster path lengths as {(i, j): p}, and the
    committed splits used up by the contraction (each a cluster of size >= 2)."""
    full = frozenset(range(1, n + 1))
    clusters = [frozenset([i]) for i in range(1, n + 1)]
    e = {i: 0 for i in range(1, n + 1)}
    within = {}
    sides = set()
    for S in forced:
        s = frozenset(S)
        sides.add(s)
        sides.add(full - s)
    consumed = []
    changed = True
    while changed and len(clusters) > 3:
        changed = False
        for a, b in combinations(range(len(clusters)), 2):
            X, Y = clusters[a], clusters[b]
            if X | Y in sides:
                for i in X | Y:
                    e[i] += 1
                for i in X:
                    for j in Y:
                        within[(min(i, j), max(i, j))] = e[i] + e[j]
                Z = X | Y
                clusters = [C for k, C in enumerate(clusters) if k not in (a, b)] + [Z]
                consumed.append(Z)
                changed = True
                break
    return clusters, e, within, consumed


def _crosses(A, B, full):
    return bool(A & B) and bool(A - B) and bool(B - A) and bool(full - (A | B))


def settle_face(W, n, D, forced, forbidden, branch=True):
    """Returns ('empty', None), ('tree', {'objective', 'w', 'trees'}), ('branch', S) or
    ('cherries', children); with branch=False, ('large', None) in place of the last two.

    A face too large to enumerate is branched on an undecided split at cut value 1/2 if there
    is one, otherwise on the cherries of a terminal vertex of the committed skeleton
    (_cherry_children): one child per pair of clusters there, each committing that pair."""
    full = frozenset(range(1, n + 1))
    for A, B in combinations(forced, 2):
        if _crosses(frozenset(A), frozenset(B), full):
            return "empty", None
    clusters, e, within, consumed = contract_committed(forced, n)
    m = len(clusters)
    consumed_keys = {_side(Z, full) for Z in consumed}
    forbidden_keys = {_side(S, full) for S in forbidden}
    if consumed_keys & forbidden_keys:
        return "empty", None
    # Clusters are numbered with the one holding leaf 1 first, so that a reduced split
    # written as the side without cluster 0 lifts to the side without leaf 1.
    clusters.sort(key=lambda C: (1 not in C, min(C)))
    which = {i: k for k, C in enumerate(clusters) for i in C}

    def reduced(S):
        """Bitmask over clusters of the side of S without leaf 1, or None if S is not a
        union of clusters."""
        side = _side(S, full)
        ks = {which[i] for i in side}
        if sum(len(clusters[k]) for k in ks) != len(side):
            return None
        return sum(1 << k for k in ks)

    red_forced = sorted({r for S in forced if _side(S, full) not in consumed_keys
                         for r in [reduced(S)]})
    if None in red_forced:
        # A committed split inside a cluster that is not one of its clades crosses one.
        return "empty", None
    red_forbidden = sorted({r for S in forbidden for r in [reduced(S)] if r is not None})

    if m <= ENUM_MAX_CLUSTERS:
        return _enumerate(n, D, clusters, e, within, consumed, red_forced, red_forbidden)
    if not branch:
        return "large", None
    # A half-cut to branch on if there is one (its commit child keeps the point and
    # contracts); otherwise the cherries of a terminal vertex.
    status, S = _branch_split(W, n, clusters, red_forced, red_forbidden)
    if status == "nohalf":
        return _cherry_children(W, clusters, red_forced, red_forbidden)
    return status, S


def _cut(W, S):
    """W[S]: the sum of w over the pairs that S separates."""
    inS = np.zeros(W.shape[0], dtype=bool)
    inS[[i - 1 for i in S]] = True
    return float(W[np.ix_(inS, ~inS)].sum())


def _cherry_children(W, clusters, red_forced, red_forbidden):
    """Children that each commit a pair of clusters as a cherry, at one terminal vertex.

    Clusters are the leaves of the skeleton whose edges are the committed splits that survive
    contraction; each such split has at least 3 clusters on either side (a side of 2 would
    have been contracted). A side containing no other committed side is the set of clusters
    at a terminal vertex; with no committed split the skeleton is a star and the set is all
    clusters. In every tree of the face the clusters at a terminal vertex hang as a rooted
    binary tree from its one skeleton edge (or form the whole tree), so some pair of them is
    a cherry. With the pairs of the smallest such set in a fixed order p_1, p_2, ..., child k
    commits p_k and forbids p_1, ..., p_{k-1}: the children partition the trees of the face.
    Pairs already forbidden are left out. Returns ('cherries', [(commit, [forbid, ...])]) with
    splits as leaf sets, or ('empty', None) if every pair is forbidden."""
    m = len(clusters)
    allc = (1 << m) - 1
    sides = {r for f in red_forced for r in (f, allc ^ f)}
    minimal = [T for T in sides if not any(U != T and U & T == U for U in sides)]
    T = min(minimal, key=lambda r: (bin(r).count("1"), r)) if minimal else allc
    members = [k for k in range(m) if T >> k & 1]
    if len(members) < 3:
        raise AssertionError(f"terminal vertex with {len(members)} clusters")
    forbidden = set(red_forbidden)

    def key(mask):  # the side without cluster 0, as stored in red_forced / red_forbidden
        return allc ^ mask if mask & 1 else mask

    def lift(mask):
        return frozenset().union(*(clusters[k] for k in range(m) if mask >> k & 1))

    pairs = [(a, b) for a, b in combinations(members, 2) if key((1 << a) | (1 << b)) not in forbidden]
    if not pairs:
        return "empty", None
    # The pairs the point joins most strongly first: the likeliest cherries.
    link = {(a, b): sum(W[i - 1, j - 1] for i in clusters[a] for j in clusters[b]) for a, b in pairs}
    pairs.sort(key=lambda p: (-link[p], p))
    splits = [lift(key((1 << a) | (1 << b))) for a, b in pairs]
    return "cherries", [(splits[k], splits[:k]) for k in range(len(splits))]


def incumbent_search(W, n, D, forced, forbidden):
    """A candidate incumbent for a node that is about to be branched on, never a bound.

    The cherries of the LP point (cherry_contract: w_ab = 1/4 and w_ac = w_bc for every other
    c, repeated on the contracted matrix) are treated as committed together with the node's
    own, and the face so restricted is enumerated exactly if at most ENUM_MAX_CLUSTERS
    clusters remain. It may miss trees of the node, so its best tree only updates the
    incumbent. If more clusters remain, the committed splits are completed to a tree and
    improved by nearest-neighbour interchanges. Returns ({'objective', 'w'} or None, how)."""
    _W_red, _sets, lp_cherries = cherry_contract(W, n)
    status, res = settle_face(W, n, D, list(forced) + list(lp_cherries), forbidden,
                              branch=False)
    if status == "tree":
        return res, "lp_cherries"
    if status == "empty":
        return None, "lp_cherries_empty"
    hc = nni_hill_climb(D, complete_splits_to_tree(forced, n, D), n)
    return hc, "nni"


def _enumerate(n, D, clusters, e, within, consumed, red_forced, red_forbidden):
    m = len(clusters)
    full = frozenset(range(1, n + 1))
    const = sum(2.0 ** -p * D[i - 1, j - 1] for (i, j), p in within.items())
    if m == 3:
        trees, red_obj, red_splits = 1, 0.0, []
        for a, b in combinations(range(3), 2):
            red_obj += 0.25 * sum(2.0 ** (-e[i] - e[j]) * D[i - 1, j - 1]
                                  for i in clusters[a] for j in clusters[b])
        if red_forced:
            raise AssertionError(f"committed splits {red_forced} left over three clusters")
    else:
        dt = np.zeros((m, m))
        for a, b in combinations(range(m), 2):
            dt[a, b] = sum(2.0 ** (-e[i] - e[j]) * D[i - 1, j - 1]
                           for i in clusters[a] for j in clusters[b])
        dt_c = np.ascontiguousarray(dt.ravel(), dtype=np.float64)
        fo = (ctypes.c_uint * max(1, len(red_forced)))(*red_forced)
        fb = (ctypes.c_uint * max(1, len(red_forbidden)))(*red_forbidden)
        best = ctypes.c_double()
        out = (ctypes.c_uint * m)()
        trees = _load().enumerate_face(
            m, dt_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), fo, len(red_forced),
            fb, len(red_forbidden), ctypes.byref(best), out)
        if trees < 0:
            raise RuntimeError(f"enumerate_face rejected m={m}")
        if trees == 0:
            return "empty", None
        red_obj = best.value
        red_splits = [out[k] for k in range(m - 3)]
    lifted = [frozenset().union(*(clusters[k] for k in range(m) if r >> k & 1))
              for r in red_splits]
    splits = [frozenset(Z) for Z in consumed if len(Z) < n - 1] + lifted
    if len({_side(S, full) for S in splits}) != n - 3:
        raise AssertionError(f"lifted tree has {len(splits)} splits, expected {n - 3}")
    _d, w = true_weights(splits, n)
    obj = float((D * w).sum()) / 2.0
    if abs(obj - (const + red_obj)) > CHECK_RTOL * max(1.0, abs(obj)):
        raise AssertionError(f"face objective mismatch: lifted {obj!r} vs reduced "
                             f"{const + red_obj!r}")
    return "tree", {"objective": obj, "w": w, "trees": int(trees)}


def _branch_split(W, n, clusters, red_forced, red_forbidden):
    """An undecided split of the face (a union of clusters, at least two on each side, not
    committed, not forbidden, crossing no committed split) with cut value 1/2 to branch on:
    the most balanced one (ties: least mask). Cuts of the graph aggregated over the clusters
    are listed in increasing order of value (cut_enum), in polynomial time per cut, so any
    number of clusters is handled. Returns ('branch', S), or ('nohalf', None) if there is no
    such split."""
    m = len(clusters)
    agg = np.zeros((m, m))
    for a, b in combinations(range(m), 2):
        agg[a, b] = agg[b, a] = sum(W[i - 1, j - 1] for i in clusters[a] for j in clusters[b])
    decided = set(red_forced) | set(red_forbidden)
    allc = (1 << m) - 1

    def acceptable(r):
        k = bin(r).count("1")
        if k < 2 or m - k < 2 or r in decided:
            return False
        return not any((r & f) and (r & ~f) and (f & ~r) and (allc & ~(r | f)) for f in red_forced)

    def balance(r):
        side = sum(len(clusters[k]) for k in range(m) if r >> k & 1)
        return min(side, n - side)

    best_half = None
    for value, side in cuts_in_order(agg, math.inf):
        if value > 0.5 + EPS:
            break
        r = sum(1 << k for k in side)          # side without cluster 0
        if not acceptable(r) or abs(value - 0.5) > EPS:
            continue
        key = (-balance(r), r)
        if best_half is None or key < best_half[0]:
            best_half = (key, r)
    if best_half is None:
        return "nohalf", None
    r = best_half[1]
    return "branch", frozenset().union(*(clusters[k] for k in range(m) if r >> k & 1))
