"""Check the integer solutions of the distance-indexed model run without four-point cuts.

    python nontree_optima.py OUT.json RUNDIR...

Each RUNDIR is a finished run of Catanzaro et al.'s solver (results/distance_cut_families/<config>/<instance>/
seed<k>/) whose path-length output tau.txt is checked, in exact arithmetic, against
conditions (4)-(9) of the paper (zero diagonal, symmetry, strong triangle with even
excess, range 2..n-1, Kraft equality per leaf, manifold equality), the strong four-point
condition (10) on every quartet, and for cherries (pairs at path length 2).
"""
import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


def check(tau):
    n = len(tau)
    off = [(i, j) for i in range(n) for j in range(n) if i != j]
    rec = {
        "n": n,
        "zero_diagonal_and_symmetric": bool((np.diag(tau) == 0).all() and (tau == tau.T).all()),
        "range_2_to_n_minus_1": all(2 <= tau[i, j] <= n - 1 for i, j in off),
        "kraft": all(sum(Fraction(1, 2 ** (int(tau[i, j]) - 1)) for j in range(n) if j != i) == 1
                     for i in range(n)),
        "manifold": sum(Fraction(2 * int(tau[i, j]), 2 ** int(tau[i, j]))
                        for i in range(n) for j in range(i + 1, n)) == 2 * n - 3,
        "strong_triangle": all((tau[i, k] + tau[k, j] - tau[i, j]) >= 2
                               and (tau[i, k] + tau[k, j] - tau[i, j]) % 2 == 0
                               for i, j in off for k in range(n) if k not in (i, j)),
    }
    viol = 0
    for a, b, c, d in itertools.combinations(range(n), 4):
        s = sorted([tau[a, b] + tau[c, d], tau[a, c] + tau[b, d], tau[a, d] + tau[b, c]])
        if not (s[1] == s[2] and s[2] - s[0] >= 2):
            viol += 1
    rec["quartets"] = n * (n - 1) * (n - 2) * (n - 3) // 24
    rec["four_point_violations"] = viol
    rec["cherries"] = [[i + 1, j + 1] for i in range(n) for j in range(i + 1, n) if tau[i, j] == 2]
    rec["satisfies_4_to_9"] = all(rec[k] for k in ("zero_diagonal_and_symmetric", "range_2_to_n_minus_1",
                                                   "kraft", "manifold", "strong_triangle"))
    rec["is_tree_metric"] = rec["satisfies_4_to_9"] and viol == 0
    return rec


out = Path(sys.argv[1])
recs = {}
for d in sys.argv[2:]:
    tau = np.rint(np.loadtxt(Path(d) / "tau.txt", skiprows=1)).astype(int)
    recs[d] = check(tau)
    r = recs[d]
    print(f"{d}: (4)-(9) {r['satisfies_4_to_9']}, four-point violations "
          f"{r['four_point_violations']}/{r['quartets']}, cherries {len(r['cherries'])}", file=sys.stderr)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(recs, indent=1) + "\n")
