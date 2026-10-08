"""Check both solvers' certified optima against exhaustive enumeration.

    BME_THREADS=1 python certificates_vs_enumeration.py LABEL INSTANCE N_MIN N_MAX OUT.json [REPS]

For each n in N_MIN..N_MAX the instance is restricted to n of its leaves (the decimal
strings of the input are kept exactly): its first n leaves if REPS is absent, otherwise
REPS random subsets of n leaves per size, each drawn by random.Random("LABEL-n-rep") and
kept in their original order (input files are often sorted by an identifier, so the first
n leaves can be closely related; random subsets are the other regime). Then:
  1. every one of the (2n-5)!! trees is enumerated (enumerate_trees.c); the TOP trees of
     smallest value in double precision are re-ranked in exact rational arithmetic, which
     gives the exact optimum, the number of trees attaining it, and the exact relative gap
     to the best non-optimal tree. The double-precision values of the best and the TOP-th
     kept tree must differ by more than MARGIN (relative), so that rounding cannot have
     pushed the true optimum out of the kept set;
  2. the distance-indexed solver of Catanzaro et al. (Section 7.1 configuration, one
     thread, 3600 s cap) and the compact solver of this paper (experiments/compact_solver.py,
     sequential search, one thread, 3600 s cap) are run on the crop;
  3. each solver's answer is compared with the enumeration: whether it claims a certificate,
     whether its tree (distance-indexed solver: the path-length output) is one of the
     optimal trees, and the exact relative difference between its value and the optimum.
A certificate whose tree is not optimal, or whose value differs from the optimum by more
than CERT_REL, is reported as a false certificate.
"""
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "run"))
sys.path.insert(0, str(REPO / "experiments"))
from parse_report import read_run  # noqa: E402

ENUM = REPO / "experiments/enumeration/build/enumerate_trees"
TOP = 200
MARGIN = 1e-9
CERT_REL = 1e-9
CAP = 3600


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", file=sys.stderr, flush=True)


def decimal_matrix(path):
    """The matrix as decimal strings, from the flat or the labelled format."""
    lines = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    n = int(lines[0][0])
    rest = lines[1:]
    if len(rest) == n * n:
        vals = [row[0] for row in rest]
        return [vals[i * n:(i + 1) * n] for i in range(n)]
    if len(rest) == n:
        return [row[1:] for row in rest]
    raise ValueError(f"{path}: unrecognised format")


def exact_value(Dq, tau):
    n = len(Dq)
    return sum(Dq[i][j] * Fraction(1, 2 ** int(tau[i][j])) for i in range(n) for j in range(i + 1, n))


def enumerate_crop(crop_file, Dq):
    out = subprocess.run([str(ENUM), str(crop_file), str(TOP)], check=True, capture_output=True,
                         text=True).stdout.split("\n")
    count = int(out[0].split()[1])
    n = len(Dq)
    trees, k = [], 1
    while k < len(out) and out[k].startswith("value"):
        v = float(out[k].split()[1])
        tau = [list(map(int, out[k + 1 + i].split())) for i in range(n)]
        trees.append((v, tau))
        k += 1 + n
    if len(trees) == TOP and (trees[-1][0] - trees[0][0]) <= MARGIN * abs(trees[0][0]):
        raise AssertionError(f"n={n}: the {TOP} kept trees span less than {MARGIN} relative")
    exact = sorted(((exact_value(Dq, t), t) for _, t in trees), key=lambda x: x[0])
    best = exact[0][0]
    optimal = [t for v, t in exact if v == best]
    runner_up = next((v for v, _ in exact if v != best), None)
    return dict(trees=count, optimum=best, optimal_trees=optimal,
                multiplicity=len(optimal),
                gap_to_runner_up=None if runner_up is None else (runner_up - best) / best)


def run_distance(crop_file, workdir):
    out = workdir / "distance"
    # The wrapper runs the solver inside its output directory, so the path must be absolute.
    subprocess.run([str(REPO / "run/catanzaro_one.sh"), str(out), "1", "1", str(CAP),
                    str(Path(crop_file).resolve()),
                    *("-O B --heuk 4 --heuls --str O --buneman O --co G --coR 0.05 --coN 0.005 "
                      "--gkiR 0.05 --gkiN 0.01 --wbR 0.125 --2KR 0.001").split()],
                   check=True, capture_output=True, text=True)
    if not (out / "report.txt").exists():
        # Keep the evidence: the run directory is temporary.
        tail = lambda f: (out / f).read_text()[-2000:] if (out / f).exists() else None
        return dict(finished=False, log_tail=tail("log.txt"), time_tail=tail("time.txt"))
    r = read_run(out)
    tau = np.rint(np.loadtxt(out / "tau.txt", skiprows=1, ndmin=2)).astype(int)
    n = int(Path(crop_file).read_text().split()[0])
    if tau.shape != (n, n):
        raise RuntimeError(f"{crop_file}: the solver wrote a {tau.shape} path-length matrix, expected {n}x{n}")
    tau = tau.tolist()
    return dict(finished=True, certified=r["certified"] == "True", time=float(r["time"]),
                nodes=int(float(r["nodes"])), tau=tau)


def run_compact(crop_file, workdir):
    out = workdir / "compact.json"
    subprocess.run([sys.executable, str(REPO / "experiments/compact_solver.py"),
                    str(crop_file), str(CAP), str(out)], check=True, capture_output=True, text=True)
    r = json.loads(out.read_text())
    return dict(finished=True, certified=bool(r["certified"]), objective=r["objective"],
                lb=r["lb"], nodes=r["n_nodes"], time=r["seconds"], rule=r["rule"])


def main():
    label, inst, n_min, n_max, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else None
    full = decimal_matrix(inst)
    N = len(full)
    if reps is None:
        draws = [(n, None, list(range(n))) for n in range(n_min, n_max + 1)]
    else:
        draws = [(n, r, sorted(random.Random(f"{label}-{n}-{r}").sample(range(N), n)))
                 for n in range(n_min, n_max + 1) for r in range(1, reps + 1)]
    crops = out.parent / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    rec = {"label": label, "instance": inst, "top_kept": TOP, "cert_rel": CERT_REL,
           "leaves": "first n" if reps is None else f"{reps} random subsets per n", "sizes": {}}
    for n, r, leaves in draws:
        key = str(n) if r is None else f"{n}_r{r}"
        Ds = [[full[i][j] for j in leaves] for i in leaves]
        Dq = [[Fraction(x) for x in row] for row in Ds]
        crop_file = crops / f"{label}_n{key}.txt"
        crop_file.write_text(f"{n}\n" + "".join(f"{x}\n" for row in Ds for x in row))
        t0 = time.time()
        en = enumerate_crop(crop_file, Dq)
        t_enum = time.time() - t0
        with tempfile.TemporaryDirectory() as tmp:
            dist = run_distance(crop_file, Path(tmp))
            comp = run_compact(crop_file, Path(tmp))
        opt = en["optimum"]
        if dist["finished"]:
            val = exact_value(Dq, dist["tau"])
            dist["tree_is_optimal"] = dist["tau"] in en["optimal_trees"]
            dist["rel_diff_from_optimum"] = float((val - opt) / opt)
            dist["false_certificate"] = dist["certified"] and not dist["tree_is_optimal"]
            del dist["tau"]
        comp["rel_diff_from_optimum"] = float((Fraction(comp["objective"]) - opt) / opt)
        comp["false_certificate"] = comp["certified"] and abs(comp["rel_diff_from_optimum"]) > CERT_REL
        rec["sizes"][key] = dict(
            leaves=[i + 1 for i in leaves], trees=en["trees"], enumeration_seconds=t_enum, optimum=str(opt), optimum_float=float(opt),
            optimal_trees=en["multiplicity"],
            gap_to_runner_up=None if en["gap_to_runner_up"] is None else float(en["gap_to_runner_up"]),
            distance=dist, compact=comp)
        log(f"{label} n={key}: {en['trees']} trees, optimum {float(opt):.12g} x{en['multiplicity']}, "
            f"runner-up gap {rec['sizes'][key]['gap_to_runner_up']}; distance cert={dist.get('certified')} "
            f"optimal tree={dist.get('tree_is_optimal')}; compact cert={comp['certified']} "
            f"rel diff {comp['rel_diff_from_optimum']:.1e}")
        out.write_text(json.dumps(rec, indent=1) + "\n")
    false = [(n, s) for n, r in rec["sizes"].items() for s in ("distance", "compact")
             if r[s].get("false_certificate")]
    if false:
        log(f"FALSE CERTIFICATES: {false}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
