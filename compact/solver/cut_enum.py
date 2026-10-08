"""Cuts of a small weighted graph in order of value, in polynomial time per cut.

The splits of the solver's LP point with cut value 1/2 (its half-cuts) are global minimum
cuts, since the min-cut rows keep every cut value at least 1/2. A graph on m vertices has at
most C(m, 2) minimum cuts. They are listed here by the scheme of Vazirani and Yannakakis
(Lawler's partitioning): the space of bipartitions with vertex 0 on side A is split into
subproblems that fix some vertices to A and some to B; each subproblem's cheapest cut is
one maximum flow; the cheapest open subproblem is taken next, its cut reported, and its
remaining space split again by the free vertices, one subproblem per free vertex. Each
reported cut costs at most m maximum flows.

Maximum flows are computed in C (cut_enum.c), in floating point by shortest augmenting paths
on the dense residual matrix, which is exact up to rounding: a residual capacity below
FLOW_TOL counts as saturated.
"""
import ctypes
import heapq
import subprocess
from pathlib import Path

import numpy as np

FLOW_TOL = 1e-13


_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "cut_enum.c"
_LIB = _HERE / "_cut_enum.so"
_lib = None


def _load():
    global _lib
    if _lib is None:
        if not _LIB.exists() or _LIB.stat().st_mtime < _SRC.stat().st_mtime:
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(_LIB), str(_SRC)], check=True)
        lib = ctypes.CDLL(str(_LIB))
        lib.min_cut.restype = ctypes.c_double
        lib.min_cut.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_ulonglong,
                                ctypes.c_ulonglong, ctypes.c_double, ctypes.POINTER(ctypes.c_ulonglong)]
        _lib = lib
    return _lib


def _max_flow_cut(C, sources, sinks):
    """Minimum cut separating the vertex sets `sources` and `sinks` (bitmasks) in the
    undirected graph with symmetric capacity matrix C (contiguous float64). Returns
    (value, side) with side the bitmask of the sources' side (cut_enum.c)."""
    side = ctypes.c_ulonglong()
    value = _load().min_cut(C.shape[0], C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            sources, sinks, FLOW_TOL, ctypes.byref(side))
    return value, side.value


def cuts_in_order(C, limit):
    """Yield (value, side) for every bipartition of the vertices of the symmetric matrix C
    whose cut value is at most limit, in nondecreasing order of value, side being the part
    without vertex 0 (a frozenset of indices)."""
    C = np.ascontiguousarray(C, dtype=np.float64)
    m = C.shape[0]
    if m > 62:
        raise ValueError(f"cuts_in_order: {m} vertices (at most 62)")
    full = (1 << m) - 1
    heap = []
    counter = 0

    def push(fixA, fixB):
        nonlocal counter
        value, side = _max_flow_cut(C, fixA, fixB)
        if value <= limit:
            heapq.heappush(heap, (value, counter, fixA, fixB, side))
            counter += 1

    # Subproblem t: vertex t on side B, vertices 0..t-1 on side A.
    for t in range(1, m):
        push((1 << t) - 1, 1 << t)
    while heap:
        value, _, fixA, fixB, sideA = heapq.heappop(heap)
        sideB = full & ~sideA
        yield value, frozenset(v for v in range(m) if sideB >> v & 1)
        # Split the rest of this subproblem: free vertices v_1, v_2, ... in order; child i
        # keeps v_1..v_{i-1} where the reported cut put them and puts v_i on the other side.
        A, B = fixA, fixB
        for v in range(m):
            bit = 1 << v
            if (A | B) & bit:
                continue
            if sideA & bit:
                push(A, B | bit)
                A |= bit
            else:
                push(A | bit, B)
                B |= bit


def half_cuts(W, n, eps):
    """The half-cuts of the n x n weight matrix W (cut value within eps of 1/2) with both sides
    of at least two leaves, as canonical leaf sets (1-indexed: the smaller side, or the side
    with leaf 1 if both have n/2 leaves), sorted as make_masks(n) orders them. Every cut of
    value up to 1/2 + eps is listed, so cuts below 1/2 - eps (a point violating a min-cut
    row) are passed over, not missed."""
    full = frozenset(range(1, n + 1))
    out = set()
    for value, side in cuts_in_order(np.asarray(W, dtype=float), 0.5 + eps):
        if value < 0.5 - eps:
            continue
        S = frozenset(i + 1 for i in side)
        if 2 <= len(S) <= n - 2:
            T = full - S
            out.add(S if (len(S), 0 if 1 in S else 1) < (len(T), 0 if 1 in T else 1) else T)
    return sorted(out, key=lambda S: (len(S), sorted(S)))
