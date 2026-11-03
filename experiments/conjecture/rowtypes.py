"""Row types of conditions (7)-(8): the multisets of n - 1 path lengths in [2, n - 1] whose
Kraft sum is 1/2, i.e. the sorted rows that a matrix satisfying (4)-(9) can have. The
enumerators (backtrack_wide8.c, experiments/theory/backtrack.c) build the same list; the
coverage check compares its length with the count they print.
"""


def row_types(n):
    """Every row type for n leaves, as a sorted tuple of path lengths. Kraft scaled by
    2^(n-1): n - 1 powers of two 2^e, e <= n - 3, summing to 2^(n-2); a part 2^e is the
    path length n - 1 - e."""
    target, out = 2 ** (n - 2), []

    def rec(rest, left, maxp, cur):
        if left == 0:
            if rest == 0:
                out.append(tuple(sorted(n - 1 - (p.bit_length() - 1) for p in cur)))
            return
        if rest < left or rest > maxp * left:
            return
        p = maxp
        while p >= 1:
            if p <= rest:
                cur.append(p)
                rec(rest - p, left - 1, p, cur)
                cur.pop()
            p //= 2

    rec(target, n - 1, 2 ** (n - 3), [])
    return out


def nocherry_row_types(n):
    """Row types with no path length 2."""
    return [r for r in row_types(n) if 2 not in r]


def first_entries(n, nocherry=True):
    """The possible values of tau[0][1] when leaf 0's row is sorted: the smallest entry of
    some row type. Any other value leaves no way to complete row 0."""
    return sorted({r[0] for r in (nocherry_row_types(n) if nocherry else row_types(n))})
