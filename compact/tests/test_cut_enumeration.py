"""The vectorised bipartition enumeration in w_space gives the same answers as a plain loop.

    python compact/tests/test_cut_enumeration.py

Checks find_half_cuts and pick_branching_cut against straightforward reference loops (for the branching set, up to
exact ties) on
tree points and on random mixtures of trees (fractional points), with random forced and
forbidden sets, for n = 6..13.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.nni_incumbent import neighbor_joining, w_from_tree  # noqa: E402
from solver.w_space import EPS, bipartition_key, find_half_cuts, make_masks, pick_branching_cut  # noqa: E402


def cut(W, S, n):
    Sc = frozenset(range(1, n + 1)) - S
    return sum(W[i - 1, j - 1] for i in S for j in Sc)


def ref_half_cuts(W, n, eps=EPS):
    return [S for S in make_masks(n) if abs(cut(W, S, n) - 0.5) <= eps]


def ref_pick(W, n, forced, forbidden, tol=EPS):
    excluded = {bipartition_key(s, n) for s in forced} | {bipartition_key(s, n) for s in forbidden}
    sym_best = asym_best = None
    for S in make_masks(n):
        if bipartition_key(S, n) in excluded:
            continue
        val = cut(W, S, n)
        if 0.5 + tol < val < 0.75 - tol:
            if sym_best is None or abs(val - 0.625) < sym_best[0]:
                sym_best = (abs(val - 0.625), S)
        elif abs(val - 0.75) < tol:
            if asym_best is None or len(S) < asym_best[0]:
                asym_best = (len(S), S)
    if sym_best is not None:
        return 'symmetric', sym_best[1]
    if asym_best is not None and len(ref_half_cuts(W, n)) == n - 3:
        return 'asymmetric', asym_best[1]
    return 'none', None


rng = random.Random(1)
checks = 0
for n in range(6, 14):
    for trial in range(20):
        trees = [w_from_tree(neighbor_joining(np.array([[0 if i == j else rng.random() for j in range(n)]
                                                         for i in range(n)]) + 0), n) for _ in range(3)]
        trees = [(T + T.T) / 2 for T in trees]
        lam = np.array([rng.random() for _ in trees]); lam /= lam.sum()
        for W in (trees[0], sum(l * T for l, T in zip(lam, trees))):
            assert find_half_cuts(W, n) == ref_half_cuts(W, n), (n, trial)
            masks = make_masks(n)
            forced = [masks[rng.randrange(len(masks))]]
            forbidden = [masks[rng.randrange(len(masks))] for _ in range(2)]
            got = pick_branching_cut(W, n, forced, forbidden)
            mode, S = ref_pick(W, n, forced, forbidden)
            assert got['mode'] == mode, (n, trial, got, mode)
            if mode == 'symmetric':
                # Exact ties (common at mixtures of trees) may be broken either way.
                assert abs(abs(cut(W, got['S'], n) - 0.625) - abs(cut(W, S, n) - 0.625)) <= 1e-12
            elif mode == 'asymmetric':
                assert len(got['S']) == len(S) and abs(cut(W, got['S'], n) - 0.75) < EPS
            checks += 2
print(f"ok: {checks} comparisons, n = 6..13")
