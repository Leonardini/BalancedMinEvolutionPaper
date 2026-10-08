"""The crossing graph of Section 5.1, the cliques of Proposition 4 and the smaller cover of
Appendix H.

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
    """Distinct families Q^{j,k}_{x,y} (2 <= j <= k, j + k <= n) with at least two members, as
    tuples of indices into splits: the bipartitions whose x-block has size j and contains y,
    and those whose x-block has size k and does not contain y."""
    out = set()
    for x in range(n):
        blocks = [block_of(S, x, n) for S in splits]
        for y in range(n):
            if y == x:
                continue
            for j in range(2, n // 2 + 1):
                for k in range(j, n - j + 1):
                    Q = tuple(sorted(
                        [i for i, A in enumerate(blocks) if len(A) == j and y in A]
                        + [i for i, A in enumerate(blocks) if len(A) == k and y not in A]))
                    if len(Q) >= 2:
                        out.add(Q)
    return sorted(out)


def pruned_cliques(n, splits):
    """The subfamily of Appendix H, as distinct tuples of indices into splits: Q^{2,2}_{x,y}
    (which does not depend on y), and Q^{j,k}_{x,y} for 2 <= j < k <= n/2, taking one of
    Q^{j,k}_{x,y} = Q^{j,k}_{y,x} when k = n/2, and leaving out y = s(x) when j = 3, where
    s(x) = x + 1 mod n for odd n and s(x) = x XOR 1 (a perfect matching) for even n."""
    s = (lambda x: (x + 1) % n) if n % 2 else (lambda x: x ^ 1)
    out = set()
    for x in range(n):
        blocks = [block_of(S, x, n) for S in splits]
        for y in range(n):
            if y == x:
                continue
            for j, k in [(2, 2)] + [(j, k) for j in range(2, n // 2 + 1) for k in range(j + 1, n // 2 + 1)]:
                if j == 3 and y == s(x):
                    continue
                Q = tuple(sorted(
                    [i for i, A in enumerate(blocks) if len(A) == j and y in A]
                    + [i for i, A in enumerate(blocks) if len(A) == k and y not in A]))
                if len(Q) >= 2:
                    out.add(Q)
    return sorted(out)


def covers(n, splits, cliques):
    """Whether every clique is pairwise crossing and every crossing pair lies in a clique."""
    covered = set()
    for Q in cliques:
        for a, b in itertools.combinations(Q, 2):
            if not cross(splits[a], splits[b], n):
                return False
            covered.add((a, b))
    return all((a, b) in covered for a, b in itertools.combinations(range(len(splits)), 2)
               if cross(splits[a], splits[b], n))
