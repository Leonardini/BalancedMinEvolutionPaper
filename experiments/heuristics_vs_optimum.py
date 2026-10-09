"""Compare the trees of NJ and of FastME's BME search with the certified optimal BME tree.

    python heuristics_vs_optimum.py OUT.json

For every real instance with a tree in data/ground_truth (the certified optimum), build
  - the neighbour-joining tree (FastME -m N, no topology search), and
  - FastME's BME tree (balanced greedy addition, then balanced NNI and SPR:
    -m B -n B -s, the search both solvers start from),
and record for each: its BME value relative to the optimum, and the number of nontrivial
splits of the optimal tree that it lacks (the Robinson-Foulds distance divided by two). A
heuristic tree counts as optimal when its BME value equals the optimum's to a relative
1e-12 (it may then differ from the stored tree only if the optimum is tied).
Progress to stderr.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from compact.solver.fastme_incumbent import _parse_newick  # noqa: E402
from compact.solver.parse_matrix import parse_matrix  # noqa: E402

INST = REPO / "external/catanzaro/upstream/instances"
NAMED = {"Primates12": "01-Primates12.txt", "M17": "02-M17.txt", "M18": "03-M18.txt"}
METHODS = {"nj": ["-m", "N"], "fastme": ["-m", "B", "-n", "B", "-s"]}
TIE = 1e-12
T0 = time.time()


def say(msg):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:5.0f} s]", msg, file=sys.stderr, flush=True)


def instance_file(label):
    if label in NAMED:
        return INST / NAMED[label]
    hits = sorted((INST / "supplement").glob(label + ".txt"))
    if len(hits) != 1:
        raise ValueError(f"{label}: {len(hits)} matching instance files")
    return hits[0]


def optimal_splits(path, n):
    lines = path.read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.strip().startswith("splits"))
    splits = [frozenset(int(x) for x in ln.strip()[1:-1].replace(",", " ").split())
              for ln in lines[k + 1:] if ln.strip().startswith("{")]
    if len(splits) != n - 3:
        raise ValueError(f"{path}: {len(splits)} splits, expected {n - 3}")
    return {canon(S, n) for S in splits}


def canon(S, n):
    """A bipartition as its side that does not contain leaf 1."""
    return frozenset(S) if 1 not in S else frozenset(range(1, n + 1)) - frozenset(S)


def tree_splits(G, n):
    out = set()
    for u, v in G.edges():
        if u > n and v > n:
            H = G.copy()
            H.remove_edge(u, v)
            out.add(canon({x for x in nx.node_connected_component(H, u) if x <= n}, n))
    return out


def tau_from_splits(splits, n):
    T = np.full((n, n), 2.0)
    np.fill_diagonal(T, 0)
    for S in splits:
        side = np.array([(i + 1) in S for i in range(n)])
        T[np.ix_(side, ~side)] += 1
        T[np.ix_(~side, side)] += 1
    return T


def bme(D, splits, n):
    return float(np.sum(np.triu(D * 2.0 ** -tau_from_splits(splits, n), 1)))


def run_fastme(D, flags):
    n = D.shape[0]
    exe = shutil.which("fastme")
    if exe is None:
        raise RuntimeError("fastme not found on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "d.phy", Path(tmp) / "t.nwk"
        rows = [f"{n}"] + [f"t{i + 1} " + " ".join(repr(float(x)) for x in D[i]) for i in range(n)]
        inp.write_text("\n".join(rows) + "\n")
        subprocess.run([exe, "-i", str(inp), "-o", str(out), *flags, "-T", "1"],
                       check=True, capture_output=True, cwd=tmp)
        return _parse_newick(out.read_text(), n)


def main():
    out = Path(sys.argv[1])
    version = subprocess.run(["fastme", "--version"], capture_output=True, text=True).stdout.strip()
    records = []
    for gt in sorted((REPO / "data/ground_truth").glob("*.splits")):
        label = gt.stem
        D = parse_matrix(instance_file(label))
        n = D.shape[0]
        opt = optimal_splits(gt, n)
        L = bme(D, opt, n)
        rec = {"instance": label, "n": n, "optimum": L}
        for name, flags in METHODS.items():
            S = tree_splits(run_fastme(D, flags), n)
            val = bme(D, S, n)
            gap = (val - L) / L
            if gap < -TIE:
                raise ValueError(f"{label}: {name} tree beats the certified optimum ({gap:.3e})")
            rec[name] = {"relative_gap": gap, "optimal": gap <= TIE,
                         "splits_missing": len(opt - S), "same_tree": S == opt,
                         "flags": " ".join(flags)}
        records.append(rec)
        say(f"{label} (n={n}): NJ gap {rec['nj']['relative_gap']:.2e}, {rec['nj']['splits_missing']} splits off; "
            f"FastME gap {rec['fastme']['relative_gap']:.2e}, {rec['fastme']['splits_missing']} splits off")
    summary = {m: {"instances": len(records), "optimal": sum(r[m]["optimal"] for r in records),
                   "same_tree": sum(r[m]["same_tree"] for r in records)} for m in METHODS}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fastme_version": version, "summary": summary, "instances": records}, indent=1) + "\n")
    say(f"summary {summary}")


if __name__ == "__main__":
    main()
