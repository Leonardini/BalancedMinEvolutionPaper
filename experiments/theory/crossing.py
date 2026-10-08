"""The crossing graph of Section 5.2 and the cliques of Proposition 4.

Leaves are 0..n-1 (leaf x here is leaf x+1 in the paper). A nontrivial bipartition is
stored as the frozenset of its block not containing leaf 0.
"""
import itertools


def bipartitions(n):
    """Nontrivial bipartitions of {0..n-1} (both blocks of size >= 2)."""
    rest = range(1, n)
    return [frozenset(S) for k in range(2, n - 1) for S in itertools.combinations(rest, k)]


def cross(A, B, n):
    full = frozenset(range(n))
    return bool(A & B and A - B and B - A and full - A - B)


def block_of(S, x, n):
    """The block of bipartition S that contains leaf x."""
    return frozenset(range(n)) - S if x not in S else S


def prop4_cliques(n, splits):
    """Distinct families Q^{j,k}_{x,y,z} (j + k <= n) with at least two members, as tuples
    of indices into splits."""
    out = set()
    for x in range(n):
        blocks = [block_of(S, x, n) for S in splits]
        for y, z in itertools.combinations([v for v in range(n) if v != x], 2):
            for j in range(2, n - 1):
                for k in range(2, n - j + 1):
                    Q = tuple(sorted(
                        [i for i, A in enumerate(blocks) if len(A) == j and y in A and z not in A]
                        + [i for i, A in enumerate(blocks) if len(A) == k and z in A and y not in A]))
                    if len(Q) >= 2:
                        out.add(Q)
    return sorted(out)
