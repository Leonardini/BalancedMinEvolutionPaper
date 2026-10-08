"""Write the optimal tree of a certified run as a data/ground_truth/*.splits file.

    python experiments/tau_to_splits.py RUNDIR OUT.splits

RUNDIR is a run of the distance-indexed solver (its report must say certified); its
tau.txt is the path-length matrix of the optimal tree. The tree is rebuilt by repeatedly
joining a pair of current nodes at path length 2 (a cherry) into a new node one edge
closer to everything else; each joined group of 2..n-2 leaves is one side of a split.
The file lists the n-3 splits, smaller side first (1-indexed leaves), and the script
checks that the splits give back tau exactly.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
from parse_report import read_run  # noqa: E402


def splits_of_tau(tau):
    n = tau.shape[0]
    groups = [frozenset([i + 1]) for i in range(n)]
    dist = {(a, b): int(tau[a, b]) for a in range(n) for b in range(n) if a != b}
    alive = list(range(n))
    splits = []
    while len(alive) > 3:
        a, b = next((a, b) for i, a in enumerate(alive) for b in alive[i + 1:] if dist[a, b] == 2)
        c = len(groups)
        groups.append(groups[a] | groups[b])
        alive = [x for x in alive if x not in (a, b)]
        for x in alive:
            dist[c, x] = dist[x, c] = dist[a, x] - 1
        alive.append(c)
        splits.append(groups[c])
    full = frozenset(range(1, n + 1))
    return [S if len(S) < n - len(S) or (len(S) == n - len(S) and 1 in S) else full - S
            for S in splits]


def tau_of_splits(n, splits):
    T = np.full((n, n), 2, dtype=int)
    np.fill_diagonal(T, 0)
    for S in splits:
        for i in range(n):
            for j in range(n):
                if i != j and ((i + 1) in S) != ((j + 1) in S):
                    T[i, j] += 1
    return T


def main():
    rundir, out = Path(sys.argv[1]), Path(sys.argv[2])
    if read_run(rundir)["certified"] != "True":
        raise SystemExit(f"{rundir}: not certified")
    tau = np.rint(np.loadtxt(rundir / "tau.txt", skiprows=1)).astype(int)
    n = tau.shape[0]
    splits = sorted(splits_of_tau(tau), key=lambda S: (len(S), sorted(S)))
    if len(splits) != n - 3 or not (tau_of_splits(n, splits) == tau).all():
        raise AssertionError(f"{rundir}: the splits do not give back tau")
    out.write_text(f"n = {n}   internal splits = {len(splits)} (from {rundir})\n"
                   "splits (smaller side, 1-indexed leaves):\n"
                   + "".join("  {" + ", ".join(map(str, sorted(S))) + "}\n" for S in splits))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
