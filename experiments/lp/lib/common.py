"""Shared helpers for the bound and count experiments: progress logging, durable
output, instances, certified optima, and Gurobi models on the configured thread count.

Every LP and QCP here runs on BME_THREADS threads (the launcher sets it); a missing
BME_THREADS is a hard error, so no model falls back to one thread per core.
"""
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "compact"))
sys.path.insert(0, str(REPO / "run"))
from solver.parse_matrix import parse_matrix  # noqa: E402,F401  (re-exported)
from solver.w_space import true_weights  # noqa: E402

from .trees import brute_force_optimum, brute_force_tree, splits_of_edges  # noqa: E402

# Objective values printed by the distance-indexed solver are ours times 2^(n+1)
# (Section 7.2). The one the manuscript quotes is cross-checked here.
PUBLISHED_SCALED_OPTIMUM = {"Primates12": 802.5893536}


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", file=sys.stderr, flush=True)


def threads():
    return int(os.environ["BME_THREADS"])


def gurobi_model(name="m"):
    import gurobipy as gp
    m = gp.Model(name)
    m.Params.OutputFlag = 0
    m.Params.Threads = threads()
    return m


def peak_rss_mb():
    # ru_maxrss is in bytes on macOS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def write_json(path, rec):
    """Write rec to path atomically, so a partial record on disk is always complete
    JSON (scripts rewrite their record after every arm)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    tmp.replace(path)


def read_splits(path):
    """The n and the n-3 nontrivial splits (frozensets of 1-indexed leaves) of a
    data/ground_truth/*.splits file."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split("=")[1].split()[0])
    splits, started = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("splits"):
            started = True
        elif started and s.startswith("{") and s.endswith("}"):
            splits.append(frozenset(int(x) for x in s[1:-1].replace(",", " ").split()))
    if len(splits) != n - 3:
        raise ValueError(f"{path}: {len(splits)} splits, expected n-3 = {n - 3}")
    return n, splits


def bme_value(D, W):
    return float((D * W).sum()) / 2.0


def tree_w(splits_path, n):
    """w = 2^-tau of the tree in a .splits file, as an n x n matrix (0-indexed)."""
    n_file, splits = read_splits(splits_path)
    if n_file != n:
        raise ValueError(f"{splits_path}: n = {n_file}, matrix has n = {n}")
    return true_weights([tuple(sorted(s)) for s in splits], n)[1]


def optimum(D, splits_path=None):
    """Certified optimum L*. With a .splits file: the value of that tree, cross-checked
    against the published value and, when it exists, against the certified run of the
    distance-indexed solver (results/distance_solver/<label>/seed1).
    Without one: brute force over every tree (small n only)."""
    n = D.shape[0]
    if splits_path is None:
        log(f"brute-force optimum over all trees, n={n}")
        return brute_force_optimum(D, progress=lambda k: log(f"  {k} trees"))
    L = bme_value(D, tree_w(splits_path, n))
    label = Path(splits_path).stem
    scaled = L * 2.0 ** (n + 1)
    if label in PUBLISHED_SCALED_OPTIMUM:
        pub = PUBLISHED_SCALED_OPTIMUM[label]
        assert abs(scaled - pub) <= 1e-9 * pub, f"{label}: {scaled} != published {pub}"
    rundir = REPO / "results" / "distance_solver" / label / "seed1"
    if (rundir / "report.txt").exists():
        from parse_report import read_run
        rec = read_run(rundir)
        if rec["certified"] == "True":
            best = float(rec["best_sol"])
            assert abs(scaled - best) <= 1e-7 * best, \
                f"{label}: tree value {scaled} != certified run {best} ({rundir})"
            log(f"{label}: L* matches the certified run in {rundir}")
    return L


def tau_of_splits(n, splits):
    """Path lengths (n x n, 0-indexed) of the tree with these nontrivial splits (sets of
    0-indexed leaves): tau_ij = 2 + the number of splits separating i and j."""
    tau = np.full((n, n), 2, dtype=int)
    for S in splits:
        side = np.array([i in S for i in range(n)])
        tau += side[:, None] != side[None, :]
    np.fill_diagonal(tau, 0)
    return tau


def optimal_tree(D, instance=None, splits_path=None):
    """A certified optimal tree of D for the optimum check of lib/safe_bound.py, as
    dict(n, splits, tau, value, source) with splits as frozensets of 0-indexed leaves (the side
    without leaf 0) and tau 0-indexed; None if there is none.
    The tree is: that of `splits_path`; else that of data/ground_truth/<label>.splits,
    where label is the instance file's stem without a leading "NN-"; else, for the
    synthetic instances test_n*, a brute-force optimum."""
    n = D.shape[0]
    if splits_path is None and instance is not None:
        stem = Path(instance).stem
        label = stem.split("-", 1)[1] if stem[:2].isdigit() and stem[2:3] == "-" else stem
        gt = REPO / "data" / "ground_truth" / f"{label}.splits"
        if gt.exists():
            splits_path = gt
        elif stem.startswith("test_n"):
            log(f"brute-force optimal tree, n={n}")
            val, edges = brute_force_tree(D, progress=lambda k: log(f"  {k} trees"))
            splits = splits_of_edges(edges, n)
            tau = tau_of_splits(n, splits)
            value = bme_value(D, np.where(tau > 0, 2.0 ** -tau.astype(float), 0.0))
            assert value == val, f"tree value {value} != brute-force minimum {val}"
            return dict(n=n, splits=splits, tau=tau, value=value, source="brute force")
        else:
            return None
    if splits_path is None:
        return None
    n_file, splits1 = read_splits(splits_path)
    if n_file != n:
        raise ValueError(f"{splits_path}: n = {n_file}, matrix has n = {n}")
    full = frozenset(range(n))
    splits = [frozenset(i - 1 for i in S) for S in splits1]
    splits = [S if 0 not in S else full - S for S in splits]
    tau = tau_of_splits(n, splits)
    value = bme_value(D, np.where(tau > 0, 2.0 ** -tau.astype(float), 0.0))
    return dict(n=n, splits=splits, tau=tau, value=value, source=str(splits_path))


def compact_point(tree, names):
    """The tree's point in the compact model's variables, by CPLEX name: w_i_j (1-indexed)
    = 2^-tau_ij, the dummy z = 0 (it appears in no row; its penalty keeps it at 0), and
    the level variables x_i_j_l of experiments/base_rows = [l == tau_ij]."""
    tau = tree["tau"]
    out = {}
    for nm in names:
        parts = nm.split("_")
        if nm == "z":
            out[nm] = 0.0
        elif parts[0] == "w" and len(parts) == 3:
            out[nm] = 2.0 ** -int(tau[int(parts[1]) - 1, int(parts[2]) - 1])
        elif parts[0] == "x" and len(parts) == 4:
            out[nm] = 1.0 if int(parts[3]) == tau[int(parts[1]) - 1, int(parts[2]) - 1] else 0.0
        else:
            raise ValueError(f"no tree value for the compact model's variable {nm!r}")
    return out
