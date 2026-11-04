"""Are the end points of the lazy-mode runs trees? The evidence for the no-good fix.

    python experiments/nogood_endpoints.py [OUT.json]

For every lazy-mode run (configurations gki, c and ubtbc) of the released solver
(results/distance_cut_families, results/distance_cut_families_ubtbc_original) and of the
solver with the corrected no-good cut (results/distance_cut_families_nogoodfix), it reads the
end point tau.txt and reports:
  - the objective recomputed from tau.txt and the instance, against the reported value, so
    the matrix checked is the one the solver certified;
  - the number of quartets violating the four-point condition (the two largest of the three
    sums equal, and at least 2 above the smallest);
  - whether a tree can be rebuilt from tau by joining cherries, and gives tau back exactly;
  - the number of SOL REJECTED lines in the solver log (no-good cuts that fired).
"""
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_to_splits import splits_of_tau, tau_of_splits  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
SOURCES = {"released": ["distance_cut_families", "distance_cut_families_ubtbc_original"],
           "corrected": ["distance_cut_families_nogoodfix"]}
CONFIGS = ("gki", "c", "ubtbc")


def distances(path):
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    n, rest = int(rows[0][0]), rows[1:]
    if len(rest) == n * n:
        vals = [float(r[0]) for r in rest]
        return [vals[i * n:(i + 1) * n] for i in range(n)]
    return [[float(x) for x in r[1:]] for r in rest]


def read_tau(path):
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    n = int(rows[0][0])
    return np.array([[int(x) for x in r] for r in rows[1:1 + n]])


def check(run):
    tau = read_tau(run / "tau.txt")
    n = len(tau)
    D = distances(re.search(r"-i (\S+)", (run / "command.txt").read_text()).group(1))
    objective = sum(D[i][j] * 2.0 ** (n + 1 - tau[i, j]) for i in range(n) for j in range(i + 1, n))
    reported = float((run / "report.txt").read_text().strip().splitlines()[-1].split("|")[2])
    violated = 0
    for a, b, c, d in itertools.combinations(range(n), 4):
        s = sorted([tau[a, b] + tau[c, d], tau[a, c] + tau[b, d], tau[a, d] + tau[b, c]])
        violated += not (s[1] == s[2] and s[2] - s[0] >= 2)
    try:
        tree = bool((tau_of_splits(n, splits_of_tau(tau)) == tau).all())
    except StopIteration:  # no cherry left to join: not a tree
        tree = False
    assert tree == (violated == 0), f"{run}: four-point and rebuild disagree"
    return {"reported": reported, "recomputed": float(objective),
            "matches": bool(abs(objective - reported) <= 1e-9 * abs(reported)),
            "quartets": n * (n - 1) * (n - 2) * (n - 3) // 24, "four_point_violated": int(violated),
            "tree": tree, "rejected": (run / "log.txt").read_text().count("SOL REJECTED")}


def main():
    out = {}
    for build, dirs in SOURCES.items():
        for d in dirs:
            for run in sorted((RES / d).glob("*/*/seed*")):
                cfg = run.parent.parent.name
                if cfg in CONFIGS and (run / "tau.txt").exists() and (run / "report.txt").exists():
                    r = check(run)
                    out[f"{build}/{cfg}/{run.parent.name}/{run.name}"] = r
                    print(f"{build:9s} {cfg:5s} {run.parent.name:10s} {run.name}: value {r['reported']:.4f} "
                          f"(recomputed {'matches' if r['matches'] else 'DIFFERS'}), four-point violated on "
                          f"{r['four_point_violated']}/{r['quartets']} quartets, tree {r['tree']}, "
                          f"{r['rejected']} rejected")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
