"""How well conditioned are the two root LPs, as the number of leaves grows?

    BME_THREADS=1 python conditioning.py INSTANCE N_MIN N_MAX OUT.json

For each n in N_MIN..N_MAX the instance is cropped to its first n leaves, and two root LPs
are built:
    compact   the root LP of the compact solver of this paper, as its branch-and-bound
              builds it (one pass of the cut loop: min-cut, PM, manifold tangent cuts);
    compact_without_manifold   the same with no manifold tangent cuts (Kelley's tangents
              approach one smooth constraint by many nearly parallel planes);
    distance  the static model of the distance-indexed solver of Catanzaro et al. (its
              Section 7.1 configuration), exported before any callback cut and solved as
              an LP (integrality dropped).
For each LP we record, after solving it with Gurobi:
    kappa_exact      Gurobi's KappaExact for the optimal basis (Gurobi does not document
                     whether this is for its internally scaled model or the original);
    basis_cond       the 2-norm condition number of the optimal basis matrix of the model
                     as written (largest over smallest singular value), computed here: the
                     basis is the columns of [A, I] (structural and slack) that Gurobi marks
                     basic, which the simplex method inverts;
    basis_cond_ruiz  the same after Ruiz's equilibration (2001) of [A, I]: every row and
                     every column is divided by the square root of its max-norm, repeatedly,
                     until all row and column max-norms are within RUIZ_TOL of 1. Condition
                     numbers are invariant under multiplying the whole matrix by a scalar but
                     not under separate row and column scaling, so both the raw and the
                     equilibrated values are reported;
    coef_range       max |a_ij| / min |a_ij| over the nonzeros of the constraint matrix;
    obj_range        the same ratio over the nonzero objective coefficients;
    A_cond_2norm     largest over smallest nonzero singular value of the whole constraint
                     matrix (computed only while the matrix has at most DENSE_MAX entries).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gurobipy as gp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "compact"))
from solver.base_model import build_base_model  # noqa: E402
from solver.bnb_balanced import MANIFOLD_ROOT_CAP  # noqa: E402
from solver.cut_loop import separate_cuts  # noqa: E402
from solver.parse_matrix import parse_matrix  # noqa: E402

DENSE_MAX = 4e7
RUIZ_TOL = 1e-6
RUIZ_MAX_ITERS = 1000
CAT_FULL = ("-O B --heuk 4 --heuls --str O --buneman O --co G --coR 0.05 --coN 0.005 "
            "--gkiR 0.05 --gkiN 0.01 --wbR 0.125 --2KR 0.001").split()


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", file=sys.stderr, flush=True)


def measure(lp_file):
    m = gp.read(str(lp_file))
    m.Params.OutputFlag = 0
    m.Params.Threads = int(os.environ["BME_THREADS"])
    m = m.relax()
    m.Params.OutputFlag = 0
    m.Params.Threads = int(os.environ["BME_THREADS"])
    m.optimize()
    if m.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"{lp_file}: Gurobi status {m.Status}")
    A = m.getA()
    a = np.abs(A.data[A.data != 0])
    c = np.abs(np.array([v.Obj for v in m.getVars()]))
    c = c[c != 0]
    rec = dict(rows=m.NumConstrs, cols=m.NumVars, nonzeros=int(A.nnz), objective=m.ObjVal,
               kappa_exact=m.KappaExact, coef_range=float(a.max() / a.min()),
               **basis_conditioning(m, A),
               obj_range=float(c.max() / c.min()) if c.size else None)
    if A.shape[0] * A.shape[1] <= DENSE_MAX:
        s = np.linalg.svd(A.toarray(), compute_uv=False)
        s = s[s > s.max() * max(A.shape) * np.finfo(float).eps]
        rec["A_cond_2norm"] = float(s.max() / s.min())
        rec["A_rank"] = int(s.size)
    else:
        rec["A_cond_2norm"] = None
    return rec


def basis_conditioning(m, A):
    """2-norm condition number of the optimal basis of [A, I], before and after row/column
    equilibration (None if the basis is too large for a dense SVD)."""
    rows = A.shape[0]
    if rows * rows > DENSE_MAX:
        return dict(basis_cond=None, basis_cond_ruiz=None, ruiz_iterations=None, basis_size=rows)
    vb = [v.VBasis for v in m.getVars()]
    cb = [c.CBasis for c in m.getConstrs()]
    cols = [j for j, b in enumerate(vb) if b == 0]
    slacks = [i for i, b in enumerate(cb) if b == 0]
    if len(cols) + len(slacks) != rows:
        raise AssertionError(f"basis has {len(cols) + len(slacks)} columns for {rows} rows")
    Ad = A.toarray()
    B = np.hstack([Ad[:, cols], np.eye(rows)[:, slacks]])
    s = np.linalg.svd(B, compute_uv=False)
    G = np.hstack([Ad, np.eye(rows)])
    for it in range(1, RUIZ_MAX_ITERS + 1):
        rn = np.abs(G).max(axis=1)
        cn = np.abs(G).max(axis=0)
        if max(np.abs(rn - 1).max(), np.abs(cn[cn > 0] - 1).max()) <= RUIZ_TOL:
            break
        cn[cn == 0] = 1.0
        G = G / np.sqrt(rn)[:, None] / np.sqrt(cn)[None, :]
    else:
        raise RuntimeError(f"Ruiz equilibration did not reach {RUIZ_TOL} in {RUIZ_MAX_ITERS} iterations")
    sr = np.linalg.svd(G[:, cols + [Ad.shape[1] + i for i in slacks]], compute_uv=False)
    return dict(basis_cond=float(s.max() / s.min()), basis_cond_ruiz=float(sr.max() / sr.min()),
                ruiz_iterations=it, basis_size=rows)


def compact_lp(D, path, manifold=True):
    c, pti, pairs = build_base_model(D, verbose=False)
    npp = len(pairs)
    c.variables.set_types(npp, c.variables.type.continuous)
    c.set_problem_type(c.problem_type.LP)
    c.parameters.emphasis.numerical.set(1)
    c.parameters.lpmethod.set(c.parameters.lpmethod.values.dual)
    counts = {k: 0 for k in ("mincut", "pm", "f6", "f7", "f4", "f5", "p50", "p53", "p54",
                             "manifold", "crossing")}
    # The root LP exactly as the branch-and-bound solver builds it: one pass of its cut
    # loop, with the manifold tangent cuts capped as at the solver's root.
    _, _, _, decided = separate_cuts(c, D, pti, pairs, npp, counts,
                                     manifold_cap=MANIFOLD_ROOT_CAP if manifold else 0, lean=True)
    if not decided:
        raise RuntimeError("compact root LP did not solve")
    c.write(str(path))
    return counts


def distance_lp(inst, n, path):
    exe = REPO / "external/catanzaro/build/bin/solver_bmep"
    labeled = ["--labeled"] if "supplement" in str(inst) else []
    # The solver runs in a temporary directory, so the instance path must be absolute.
    subprocess.run([str(exe), "-i", str(Path(inst).resolve()), "-n", str(n), *labeled, *CAT_FULL,
                    "--export", str(path)], check=True, capture_output=True,
                   cwd=path.parent, env=dict(os.environ, OMP_NUM_THREADS="1"))


def main():
    inst, n_min, n_max, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    D_full = parse_matrix(inst)
    rec = {"instance": inst, "sizes": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for n in range(n_min, n_max + 1):
            D = D_full[:n, :n]
            t0 = time.time()
            counts = compact_lp(D, Path(tmp) / "compact.lp")
            comp = measure(Path(tmp) / "compact.lp")
            comp["cuts"] = counts
            compact_lp(D, Path(tmp) / "compact_nomanifold.lp", manifold=False)
            nom = measure(Path(tmp) / "compact_nomanifold.lp")
            distance_lp(inst, n, Path(tmp) / "distance.mps")
            dist = measure(Path(tmp) / "distance.mps")
            rec["sizes"][n] = {"compact": comp, "compact_without_manifold": nom, "distance": dist,
                               "seconds": time.time() - t0}
            log(f"n={n}: compact kappa {comp['kappa_exact']:.3e} basis {comp['basis_cond']} Ruiz "
                f"{comp['basis_cond_ruiz']} coef range {comp['coef_range']:.2e}; distance kappa "
                f"{dist['kappa_exact']:.3e} basis {dist['basis_cond']} Ruiz {dist['basis_cond_ruiz']} "
                f"coef range {dist['coef_range']:.2e} "
                f"obj range {dist['obj_range']:.2e}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
