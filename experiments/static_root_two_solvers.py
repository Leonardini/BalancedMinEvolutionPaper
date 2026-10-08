"""The distance-indexed model's static root, given to Gurobi and to CPLEX.

    python static_root_two_solvers.py LABEL INSTANCE OUT.json [--labeled] [--only gurobi|cplex]
                                      [--ref VALUE]

The model is the one the distance-indexed solver builds for the generalized-Kraft
configuration of Table D9 (--str A: level indicators, convexity, Kraft, the linear
manifold equality, and the triangle inequalities with integer parity variables),
exported before any callback cut (our --export option), so generalized Kraft itself is
not in it. Each solver, on one thread, then solves:
  lp        the LP relaxation (integrality dropped);
  root_on   the MIP stopped after the root node, with its general-purpose cutting
            planes at their defaults (Gurobi NodeLimit 1; CPLEX node limit 0);
  root_off  the same with every class of general-purpose cut switched off (Gurobi Cuts 0;
            CPLEX every mip.cuts parameter at -1);
  root_max  the same with every class at its most aggressive setting (Gurobi Cuts 3;
            CPLEX every mip.cuts parameter at its upper limit).
Each records the bound, the time and, for CPLEX, the cuts added by class. Gaps are
relative to the certified optimum (results/check_optima.json), on the solver's scale; for an
instance with no certified optimum (M43), --ref gives the best known value instead, and the
gaps are then upper bounds on the true gaps. --only runs one solver (the record then has
only that solver), so the two can run side by side.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cplex
import gurobipy as gp

REPO = Path(__file__).resolve().parents[1]
EXE = REPO / "external/catanzaro/build/bin/solver_bmep"
CAP = 3600.0
FLAGS = ["-O", "B", "--str", "A"]


def export(inst, path, labeled):
    subprocess.run([str(EXE), "-i", str(Path(inst).resolve()), *(["--labeled"] if labeled else []),
                    *FLAGS, "--export", str(path)], check=True, capture_output=True,
                   cwd=path.parent, env=dict(os.environ, OMP_NUM_THREADS="1"))


def gurobi(path):
    out = {}
    m = gp.read(str(path))
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    r = m.relax()
    r.Params.OutputFlag = 0
    r.Params.Threads = 1
    t0 = time.time()
    r.optimize()
    out["lp"] = dict(bound=r.ObjVal, seconds=time.time() - t0)
    for name, cuts in (("root_on", -1), ("root_off", 0), ("root_max", 3)):
        m.reset()
        m.Params.NodeLimit = 1
        m.Params.Cuts = cuts
        m.Params.TimeLimit = CAP
        t0 = time.time()
        m.optimize()
        out[name] = dict(bound=m.ObjBound, seconds=time.time() - t0, status=int(m.Status))
    return out


def cplex_solve(path, cuts_off, lp=False, cuts_max=False):
    c = cplex.Cplex()
    for s in (c.set_log_stream, c.set_results_stream, c.set_warning_stream):
        s(None)
    c.read(str(path))
    c.parameters.threads.set(1)
    c.parameters.timelimit.set(CAP)
    if lp:
        c.set_problem_type(c.problem_type.LP)
    else:
        c.parameters.mip.limits.nodes.set(0)
        if cuts_off:
            for p, _ in c.parameters.mip.cuts.get_all():
                p.set(-1)
        if cuts_max:
            for p, _ in c.parameters.mip.cuts.get_all():
                p.set(p.max())
    t0 = time.time()
    c.solve()
    rec = dict(seconds=time.time() - t0, status=c.solution.get_status_string())
    if lp:
        rec["bound"] = c.solution.get_objective_value()
    else:
        rec["bound"] = c.solution.MIP.get_best_objective()
        ct = c.solution.MIP.cut_type
        rec["cuts"] = {k: c.solution.MIP.get_num_cuts(getattr(ct, k)) for k in dir(ct)
                       if not k.startswith("_") and isinstance(getattr(ct, k), int)
                       and c.solution.MIP.get_num_cuts(getattr(ct, k))}
    return rec


def main():
    label, inst, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    args = sys.argv[4:]
    labeled = "--labeled" in args
    only = args[args.index("--only") + 1] if "--only" in args else None
    if "--ref" in args:
        L, ref = float(args[args.index("--ref") + 1]), "best known value (not certified)"
    else:
        opt = json.loads((REPO / "results/check_optima.json").read_text())[label]
        L, ref = float(opt["exact_value_float"]) * 2 ** (opt["n"] + 1), "certified optimum"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.mps"
        export(inst, path, labeled)
        rec = dict(label=label, instance=inst, flags=" ".join(FLAGS), optimum_solver_scale=L,
                   reference=ref)
        if only in (None, "gurobi"):
            rec["gurobi"] = gurobi(path)
        if only in (None, "cplex"):
            rec["cplex"] = dict(lp=cplex_solve(path, False, lp=True),
                                root_on=cplex_solve(path, False),
                                root_off=cplex_solve(path, True),
                                root_max=cplex_solve(path, False, cuts_max=True))
    solvers = [s for s in ("gurobi", "cplex") if s in rec]
    for s in solvers:
        for k, r in rec[s].items():
            r["root_gap_pct"] = 100 * (L - r["bound"]) / L
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(label, {s: {k: round(r["root_gap_pct"], 4) for k, r in rec[s].items()}
                  for s in solvers}, file=sys.stderr)


if __name__ == "__main__":
    main()
