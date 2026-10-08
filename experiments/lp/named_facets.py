"""Table D3 (Section 8.2): root-LP gap closed by F1 and F6 on the synthetic instances.

    BME_THREADS=1 python named_facets.py OUT.json [L10]

Every inequality family is written out in full (no separation). Base: Kraft,
0 <= w <= 1/4, all cuts W[S] >= 1/2, the double cherry w_ij + w_jk - w_ik <= 1/4, and
all crossing cuts W[A] + W[B] >= 5/4 for crossing A, B. F1: w >= 2^-(n-1).
F6: w_kl <= 2^(n-4) w_ij for distinct i, j, k, l. L* is by brute force over every tree.
L10, if given, is the certified n=10 optimum quoted in the manuscript; the brute-force
value must agree with it to its printed precision.
Each bound also has its safe value (lib/safe_bound.py): <arm>_safe_bound and
<arm>_safe_minus_lp in the row, and <arm>_rows_cut_off_optimum, the number of rows the
brute-force optimal tree violates (exactly); the details under safe_bound_detail.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import safe_bound  # noqa: E402
from lib.common import REPO, gurobi_model, log, optimal_tree, parse_matrix, peak_rss_mb, write_json  # noqa: E402

SIZES = range(6, 11)


def lp_bound(D, f1, f6, tree):
    from gurobipy import GRB
    n = len(D)
    full = (1 << n) - 1
    P = list(itertools.combinations(range(n), 2))
    idx = {p: k for k, p in enumerate(P)}
    pid = lambda a, b: idx[(min(a, b), max(a, b))]  # noqa: E731
    cut = lambda m: np.array([1.0 if ((m >> i) & 1) != ((m >> j) & 1) else 0.0 for i, j in P])  # noqa: E731
    masks = [m for m in range(1, full) if not m & 1 and 2 <= bin(m).count('1') <= n - 2]
    C = {m: cut(m) for m in masks}
    rows = [-C[m] for m in masks]
    rhs = [-0.5] * len(masks)
    tags = ["mincut"] * len(masks)
    for i, j, k in itertools.permutations(range(n), 3):
        r = np.zeros(len(P))
        r[pid(i, j)] += 1
        r[pid(j, k)] += 1
        r[pid(i, k)] -= 1
        rows.append(r)
        rhs.append(0.25)
        tags.append("double_cherry")
    crosses = lambda a, b: a & b and a & ~b & full and ~a & b & full and ~a & ~b & full  # noqa: E731
    for a, b in itertools.combinations(masks, 2):
        if crosses(a, b):
            rows.append(-(C[a] + C[b]))
            rhs.append(-1.25)
            tags.append("crossing")
    if f6:
        for i, j in P:
            for k, l in P:
                if len({i, j, k, l}) == 4:
                    r = np.zeros(len(P))
                    r[idx[(k, l)]] = 1
                    r[idx[(i, j)]] = -2.0 ** (n - 4)
                    rows.append(r)
                    rhs.append(0.0)
                    tags.append("f6")
    Aeq = np.array([[1.0 if i in p else 0.0 for p in P] for i in range(n)])
    lo = 2.0 ** -(n - 1) if f1 else 0.0
    m = gurobi_model()
    w = m.addMVar(len(P), lb=lo, ub=0.25)
    m.addMConstr(csr_matrix(np.array(rows)), w, '<', np.array(rhs))
    m.addMConstr(csr_matrix(Aeq), w, '=', np.full(n, 0.5))
    m.setObjective(np.array([D[i, j] for i, j in P]) @ w, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"n={n} f1={f1} f6={f6}: Gurobi status {m.Status}")
    tw = [2.0 ** -int(tree["tau"][p]) for p in P]
    sb = safe_bound.certify(m, f"n={n} f1={f1} f6={f6}", tree=tree,
                            point=dict(zip(w.tolist(), tw)), tags=tags + ["kraft"] * n)
    return m.ObjVal, len(rows), sb


def main():
    out = Path(sys.argv[1])
    l10 = float(sys.argv[2]) if len(sys.argv) > 2 else None
    rec = {"rows": []}
    for n in SIZES:
        D = parse_matrix(REPO / "data" / "synthetic" / f"test_n{n}.txt")
        assert D.shape[0] == n
        tree = optimal_tree(D, instance=REPO / "data" / "synthetic" / f"test_n{n}.txt")
        opt = tree["value"]
        if n == 10 and l10 is not None:
            assert abs(opt - l10) <= 5e-7, f"brute-force L*={opt} != quoted {l10}"
        (b0, r0, s0), (b1, _, s1), (b2, r2, s2) = (
            lp_bound(D, False, False, tree), lp_bound(D, True, False, tree),
            lp_bound(D, True, True, tree))
        row = dict(n=n, opt=opt, base=b0, f1=b1, f1_f6=b2, closed=(b2 - b0) / (opt - b0),
                   rows_base=r0, rows_f6=r2)
        detail = dict(base=s0, f1=s1, f1_f6=s2)
        for arm, sr in detail.items():
            row.update(safe_bound.fields(sr, f"{arm}_"))
        row["safe_bound_detail"] = detail
        rec["rows"].append(row)
        rec["peak_rss_mb"] = peak_rss_mb()
        write_json(out, rec)
        log(f"n={n:2d}  opt={opt:.6f}  base={b0:.6f}  +F1={b1:.6f}  +F1+F6={b2:.6f}  "
            f"closed={row['closed']:.1%}  safe-LP: "
            + " ".join(f"{a}={sr['safe_minus_reported']:.2g} ({sr['seconds']:.1f}s)" for a, sr in detail.items()))


if __name__ == "__main__":
    main()
