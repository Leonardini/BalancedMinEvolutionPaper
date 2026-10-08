"""Check the certified optima of the distance-indexed solver exactly.

    python check_optima.py OUT.json

For every certified run in results/distance_solver, in the Table D9 configurations that separate
the four-point condition (d, full, cooff) and in results/robustness, read the solver's integer path-length
output tau.txt and check, per instance, that
  - every certified run returns the same tree (identical path-length matrix),
  - that matrix is a tree metric (strong four-point condition on every quartet), and
  - its BME value, sum_{i<j} D_ij 2^-tau_ij computed in exact rational arithmetic from the
    decimal strings of the input matrix, agrees with the value each run reports, and
  - the tree of data/ground_truth/<instance>*.splits, which the bound experiments read
    as the optimal tree, is a tree with exactly the same BME value (the same tree, or
    another optimal tree when the optimum is tied).
Runs of configurations without four-point separation (gki, c) are relaxations and are
excluded.
"""
import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "run"))
from parse_report import read_run  # noqa: E402

INST = REPO / "external/catanzaro/upstream/instances"
NAMED = {"Primates12": "01-Primates12.txt", "M17": "02-M17.txt", "M18": "03-M18.txt",
         "M43": "05-M43.txt", "20_B-HA": "supplement/20_B-HA-573-585-BMGE.txt"}


def instance_file(label):
    if label in NAMED:
        return INST / NAMED[label]
    hits = sorted((INST / "supplement").glob(label + "*.txt"))
    if len(hits) != 1:
        raise ValueError(f"{label}: {len(hits)} matching instance files")
    return hits[0]


def exact_matrix(path):
    """Distance matrix as exact Fractions, from either the flat or the labelled format."""
    lines = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    n = int(lines[0][0])
    rest = lines[1:]
    if len(rest) == n * n:
        vals = [Fraction(row[0]) for row in rest]
        return [vals[i * n:(i + 1) * n] for i in range(n)]
    if len(rest) == n:
        return [[Fraction(v) for v in row[1:]] for row in rest]
    raise ValueError(f"{path}: unrecognised format")


def is_tree_metric(T):
    n = len(T)
    for a, b, c, d in itertools.combinations(range(n), 4):
        s = sorted([T[a, b] + T[c, d], T[a, c] + T[b, d], T[a, d] + T[b, c]])
        if not (s[1] == s[2] and s[2] - s[0] >= 2):
            return False
    return True


def ground_truth_tau(label, n):
    """Path-length matrix of the tree in data/ground_truth (None if there is no file):
    tau_ij = 2 + the number of nontrivial splits that separate i and j."""
    hits = sorted((REPO / "data/ground_truth").glob(label + "*.splits"))
    if not hits:
        return None
    if len(hits) != 1:
        raise ValueError(f"{label}: {len(hits)} matching ground-truth files")
    lines = hits[0].read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.strip().startswith("splits"))
    splits = [frozenset(int(x) for x in ln.strip()[1:-1].replace(",", " ").split())
              for ln in lines[k + 1:] if ln.strip().startswith("{")]
    if len(splits) != n - 3:
        raise ValueError(f"{hits[0]}: {len(splits)} splits, expected {n - 3}")
    T = np.full((n, n), 2, dtype=int)
    np.fill_diagonal(T, 0)
    for S in splits:
        for i in range(n):
            for j in range(n):
                if i != j and ((i + 1) in S) != ((j + 1) in S):
                    T[i, j] += 1
    return T


runs = {}
dirs = sorted(REPO.glob("results/distance_solver/*/seed*")) + \
    [p for cfg in ("d", "full", "cooff") for p in sorted(REPO.glob(f"results/distance_cut_families/{cfg}/*/seed*"))] + \
    sorted(REPO.glob("results/robustness/*/*/seed*"))
for d in dirs:
    if not (d / "report.txt").exists():
        continue
    r = read_run(d)
    if r["certified"] != "True":
        continue
    T = np.rint(np.loadtxt(d / "tau.txt", skiprows=1)).astype(int)
    runs.setdefault(d.parent.name, []).append((str(d.relative_to(REPO)), T, float(r["best_sol"])))

out = {}
for label, lst in sorted(runs.items()):
    D = exact_matrix(instance_file(label))
    n = len(D)
    T0 = lst[0][1]
    value = sum(D[i][j] * Fraction(1, 2 ** int(T0[i, j])) for i in range(n) for j in range(i + 1, n))
    solver_scale = value * 2 ** (n + 1)   # the solver reports our value times 2^(n+1)
    rel = max(abs(Fraction(rep) - solver_scale) / solver_scale for _, _, rep in lst)
    out[label] = {
        "n": n, "certified_runs": len(lst), "runs": [d for d, _, _ in lst],
        "same_tree_in_all_runs": all((T == T0).all() for _, T, _ in lst),
        "is_tree_metric": is_tree_metric(T0),
        "exact_value": str(value), "exact_value_float": float(value),
        "max_relative_difference_from_reported": float(rel),
    }
    G = ground_truth_tau(label, n)
    if G is None:
        out[label]["ground_truth_file"] = None
    elif (G == T0).all():
        out[label]["ground_truth_file"] = "same tree"
    elif is_tree_metric(G) and sum(D[i][j] * Fraction(1, 2 ** int(G[i, j]))
                                   for i in range(n) for j in range(i + 1, n)) == value:
        out[label]["ground_truth_file"] = "another tree with exactly the same value"
    else:
        raise AssertionError(f"{label}: the tree in data/ground_truth is not optimal")
    print(f"{label:12s} runs={len(lst):2d} same tree={out[label]['same_tree_in_all_runs']} "
          f"tree={out[label]['is_tree_metric']} value={float(value):.12f} max rel diff={float(rel):.1e}",
          file=sys.stderr)
    if not (out[label]["same_tree_in_all_runs"] and out[label]["is_tree_metric"]):
        raise AssertionError(f"{label}: certified runs disagree or the tree is not a tree metric")
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
