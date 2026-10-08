"""Exact checks of conditions (4)-(10) on integer path-length matrices, and the
relabelling canonical form used to compare counterexamples (Section 6.3, Appendix C).

Conditions, for an n x n integer matrix tau (Catanzaro, Pesenti and Wolsey 2020):
  (4) zero diagonal; (5) symmetry; (6) triangle tau_ik + tau_kj - tau_ij >= slack
  (slack 2: strong; slack 0: weak); (7) 2 <= tau_ij <= n-1 off the diagonal;
  (8) Kraft, sum_j 2^-tau_ij = 1/2 for every i; (9) manifold,
  sum_{i<j} tau_ij 2^-tau_ij = n - 3/2; (10) strong four-point: on every quartet the two
  largest pair sums are equal and exceed the third by at least 2.
All arithmetic is exact (integers and Fractions).
"""
import itertools
from fractions import Fraction

# The 6-node matrix r of Section 6.3 (the reduced form of the counterexample).
R6 = [[0, 2, 3, 4, 5, 5],
      [2, 0, 4, 3, 5, 5],
      [3, 4, 0, 3, 3, 4],
      [4, 3, 3, 0, 4, 3],
      [5, 5, 3, 4, 0, 2],
      [5, 5, 4, 3, 2, 0]]


def read_dump(path):
    """Matrices written by backtrack --dump: n rows of integers, each matrix ended by '--'."""
    mats, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "--":
                if not cur:
                    raise ValueError(f"{path}: empty matrix block")
                mats.append(cur)
                cur = []
            elif line:
                cur.append([int(x) for x in line.split()])
    if cur:
        raise ValueError(f"{path}: last matrix is not terminated by '--' (truncated file?)")
    for M in mats:
        if any(len(row) != len(M) for row in M):
            raise ValueError(f"{path}: a matrix is not square")
    return mats


def read_matrix(path):
    """One whitespace-separated square integer matrix; lines starting with # are skipped."""
    with open(path) as f:
        rows = [[int(x) for x in line.split()] for line in f
                if line.strip() and not line.lstrip().startswith("#")]
    if any(len(r) != len(rows) for r in rows):
        raise ValueError(f"{path}: not a square matrix")
    return rows


def quartet_violations(tau):
    """Quartets (a<b<c<d) on which (10) fails, with their sorted pair sums."""
    out = []
    for a, b, c, d in itertools.combinations(range(len(tau)), 4):
        s = sorted([tau[a][b] + tau[c][d], tau[a][c] + tau[b][d], tau[a][d] + tau[b][c]])
        if s[1] != s[2] or s[2] - s[0] < 2:
            out.append(((a, b, c, d), s))
    return out


def conditions(tau, triangle_slack=2):
    """Dictionary condition -> bool for (4)-(10)."""
    n = len(tau)
    off = [(i, j) for i in range(n) for j in range(n) if i != j]
    half = Fraction(1, 2)
    return {
        "(4)": all(tau[i][i] == 0 for i in range(n)),
        "(5)": all(tau[i][j] == tau[j][i] for i, j in off),
        "(6)": all(tau[i][k] + tau[k][j] - tau[i][j] >= triangle_slack
                   for i, j, k in itertools.permutations(range(n), 3)),
        "(7)": all(2 <= tau[i][j] <= n - 1 for i, j in off),
        "(8)": all(sum(half ** tau[i][j] for j in range(n) if j != i) == half
                   for i in range(n)),
        "(9)": sum(tau[i][j] * half ** tau[i][j] for i, j in itertools.combinations(range(n), 2))
               == Fraction(2 * n - 3, 2),
        "(10)": not quartet_violations(tau),
    }


def is_counterexample(tau, triangle_slack=2):
    """True iff tau satisfies (4)-(9) and violates (10)."""
    c = conditions(tau, triangle_slack)
    return all(c[k] for k in ("(4)", "(5)", "(6)", "(7)", "(8)", "(9)")) and not c["(10)"]


def cherries(tau):
    """Pairs at path length 2."""
    return [(i, j) for i, j in itertools.combinations(range(len(tau)), 2) if tau[i][j] == 2]


def reduce_cherries(tau):
    """For a matrix whose leaves are partitioned into cherries, the reduced matrix
    r_ab = tau_pq - 2 (p in cherry a, q in cherry b), and the cherries in order.
    Raises if the cherries do not partition the leaves or tau is not constant between
    two cherries."""
    ch = cherries(tau)
    leaves = sorted(x for c in ch for x in c)
    if leaves != list(range(len(tau))):
        raise ValueError("the cherries do not partition the leaves")
    m = len(ch)
    r = [[0] * m for _ in range(m)]
    for a, b in itertools.permutations(range(m), 2):
        vals = {tau[p][q] for p in ch[a] for q in ch[b]}
        if len(vals) != 1:
            raise ValueError(f"tau is not constant between cherries {ch[a]} and {ch[b]}")
        r[a][b] = vals.pop() - 2
    return r, ch


def cherry_lift(r):
    """The matrix obtained by replacing node a of r by the cherry {2a, 2a+1}."""
    n = 2 * len(r)
    return [[0 if p == q else 2 if p // 2 == q // 2 else r[p // 2][q // 2] + 2
             for q in range(n)] for p in range(n)]


def canon(r):
    """Lexicographically smallest upper triangle over all relabellings, and the
    permutations attaining it."""
    m = len(r)
    best, arg = None, []
    for p in itertools.permutations(range(m)):
        flat = tuple(r[p[a]][p[b]] for a, b in itertools.combinations(range(m), 2))
        if best is None or flat < best:
            best, arg = flat, [p]
        elif flat == best:
            arg.append(p)
    return best, arg


def stabiliser_order(r):
    """Number of relabellings p with r[p[a]][p[b]] == r[a][b] for all a, b."""
    m = len(r)
    return sum(all(r[p[a]][p[b]] == r[a][b] for a in range(m) for b in range(m))
               for p in itertools.permutations(range(m)))
