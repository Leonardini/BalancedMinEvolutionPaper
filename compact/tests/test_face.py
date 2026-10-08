"""settle_face against brute force: on random instances with random committed and forbidden
splits, the face's best tree, its objective and the face being empty must agree with a scan
of every tree; and on a face too large to enumerate, the half-cut or the cherries it branches
on must be undecided and cross no committed split.

    python compact/tests/test_face.py
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver import face  # noqa: E402
from solver.bnb_balanced import _splits_from_edges_fast  # noqa: E402
from solver.nni_incumbent import _gen_all_trees  # noqa: E402
from solver.w_space import canonical_S, true_weights  # noqa: E402


def brute(D, n, forced, forbidden):
    fk = {canonical_S(s, n) for s in forced}
    bk = {canonical_S(s, n) for s in forbidden}
    best, count = None, 0
    for edges in _gen_all_trees(n):
        keys = {canonical_S(s, n) for s in _splits_from_edges_fast(edges, n)}
        if not fk <= keys or keys & bk:
            continue
        count += 1
        _d, w = true_weights(list(keys), n)
        obj = float((D * w).sum()) / 2.0
        if best is None or obj < best:
            best = obj
    return best, count


def random_case(rng, n):
    D = np.random.default_rng(rng.randrange(1 << 30)).random((n, n)) * 10
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    trees = list(_gen_all_trees(n))
    tree_splits = _splits_from_edges_fast(rng.choice(trees), n)
    forced = rng.sample(tree_splits, rng.randint(0, len(tree_splits)))
    other = _splits_from_edges_fast(rng.choice(trees), n)
    forbidden = [s for s in rng.sample(other, rng.randint(0, min(3, len(other))))
                 if canonical_S(s, n) not in {canonical_S(f, n) for f in forced}]
    return D, forced, forbidden


def test_enumeration_matches_brute_force():
    rng = random.Random(1)
    checked = 0
    for n in (5, 6, 7, 8, 9):
        for _ in range(40):
            D, forced, forbidden = random_case(rng, n)
            status, res = face.settle_face(np.zeros((n, n)), n, D, forced, forbidden)
            best, count = brute(D, n, forced, forbidden)
            if status == "empty":
                assert count == 0, (n, forced, forbidden)
            else:
                assert status == "tree"
                assert res["trees"] == count
                assert abs(res["objective"] - best) <= 1e-9 * max(1, best)
            checked += 1
    assert checked == 200


def test_forced_crossing_is_empty():
    n = 6
    D = np.ones((n, n)) - np.eye(n)
    status, _ = face.settle_face(np.zeros((n, n)), n, D, [{1, 2, 3}, {3, 4}], [])
    assert status == "empty"


def _check_undecided(S, n, forced, full):
    assert canonical_S(S, n) not in {canonical_S(f, n) for f in forced}
    assert not any(face._crosses(S, frozenset(f), full) for f in forced)
    assert 2 <= len(S) <= n - 2


def test_large_face_branches_on_a_half_cut():
    # The point of a caterpillar tree with its first cherry committed: more than 10 clusters
    # remain, so settle_face must branch, on an undecided split at cut value 1/2.
    rng = random.Random(2)
    n = 14
    full = frozenset(range(1, n + 1))
    for _ in range(10):
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        _d, W = true_weights([frozenset(perm[:k]) for k in range(2, n - 1)], n)
        forced = [frozenset(perm[:2])]
        status, S = face.settle_face(W, n, np.ones((n, n)), forced, [])
        assert status == "branch"
        _check_undecided(S, n, forced, full)
        assert abs(face._cut(W, S) - 0.5) < 1e-9


def test_large_face_without_half_cut_branches_on_cherries():
    # A random point has no split at cut value 1/2: the children commit the pairs of clusters
    # at a terminal vertex, child k forbidding the pairs of children 1, ..., k-1.
    rng = random.Random(2)
    n = 14
    full = frozenset(range(1, n + 1))
    for _ in range(10):
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        forced = [frozenset(perm[:2])]
        W = np.random.default_rng(rng.randrange(1 << 30)).random((n, n)) * 0.25
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        status, children = face.settle_face(W, n, np.ones((n, n)), forced, [])
        assert status == "cherries"
        commits = [C for C, _ in children]
        for k, (C, F) in enumerate(children):
            _check_undecided(C, n, forced, full)
            assert list(F) == commits[:k]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for f in tests:
        f()
        print(f"ok: {f.__name__}")
