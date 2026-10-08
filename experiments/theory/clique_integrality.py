"""Section 5.1: the clique relaxation over the cover of Proposition 4 is not integral.

    python clique_integrality.py OUT.json [TRIALS] [SEED] [N_MAX]

(a) The induced 5-cycle of the text at n = 6: consecutive bipartitions cross,
    non-consecutive ones are compatible.
(b) For n = 5..N_MAX (default 8), maximise a random objective c ~ U[0,1)^splits over
        { y in [0,1]^splits : sum_{S in Q} y_S <= 1 for every clique Q of Proposition 4 }
    by dual simplex (so the optimum returned is a vertex), TRIALS times, and record which
    optima are fractional and the distinct values of their fractional coordinates.

A coordinate counts as integral when it is within the solver's feasibility tolerance
(Gurobi's FeasibilityTol, read from the model) of 0 or 1; a smaller threshold would
count the solver's rounding noise as fractional values. Fractional values are recorded
as the fraction of denominator at most MAX_DEN within that tolerance, or, when there is
none, as the float itself. The random sequence does not depend on N_MAX, so a run with
a smaller N_MAX reproduces the first rows of a full run.
"""
import json
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossing  # noqa: E402

MAX_DEN = 1000


def say(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, file=sys.stderr, flush=True)


def five_cycle():
    n = 6
    # {2,3,6}, {5,6}, {2,6}, {2,3}, {3,4} in the paper's labels; stored 0-based.
    cyc = [frozenset(x - 1 for x in S) for S in ({2, 3, 6}, {5, 6}, {2, 6}, {2, 3}, {3, 4})]
    L = len(cyc)
    consecutive = all(crossing.cross(cyc[i], cyc[(i + 1) % L], n) for i in range(L))
    others = all(not crossing.cross(cyc[i], cyc[j], n)
                 for i in range(L) for j in range(i + 2, L) if not (i == 0 and j == L - 1))
    return dict(n=n, consecutive_cross=consecutive, non_consecutive_compatible=others)


def random_objectives(n, trials, rng):
    splits = crossing.bipartitions(n)
    cliques = crossing.prop4_cliques(n, splits)
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = int(os.environ["BME_THREADS"])
    m.Params.Method = 1
    y = m.addVars(len(splits), lb=0, ub=1)
    for Q in cliques:
        m.addConstr(gp.quicksum(y[i] for i in Q) <= 1)
    tol = m.Params.FeasibilityTol
    fractional, values, all_half_vertex, frac_half_only = 0, {}, 0, 0
    for _ in range(trials):
        c = [rng.random() for _ in splits]
        m.setObjective(gp.quicksum(c[i] * y[i] for i in range(len(splits))), GRB.MAXIMIZE)
        m.reset()
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"n={n}: Gurobi status {m.Status}")
        x = [y[i].X for i in range(len(splits))]
        frac = [v for v in x if abs(v - round(v)) > tol]
        if frac:
            fractional += 1
            for v in frac:
                key = as_fraction(v, tol)
                values[key] = values.get(key, 0) + 1
            if all(abs(v - 0.5) <= tol for v in frac):
                frac_half_only += 1
            if all(abs(v - 0.5) <= tol for v in x):
                all_half_vertex += 1
    return dict(n=n, splits=len(splits), cliques=len(cliques), trials=trials,
                integrality_tolerance=tol,
                fractional_optima=fractional,
                fractional_values=sorted(values, key=sort_key),
                fractional_value_counts={k: values[k] for k in sorted(values, key=sort_key)},
                every_fractional_coordinate_is_half=(set(values) <= {"1/2"}),
                optima_whose_fractional_coordinates_are_all_half=frac_half_only,
                optima_with_every_coordinate_half=all_half_vertex)


def as_fraction(v, tol):
    """The fraction of denominator <= MAX_DEN within tol of v, as a string, or else
    repr(v)."""
    f = Fraction(v).limit_denominator(MAX_DEN)
    return str(f) if abs(float(f) - v) <= tol else repr(v)


def sort_key(s):
    return float(Fraction(s))


def main():
    out = Path(sys.argv[1])
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 12345
    n_max = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rec = dict(seed=seed, five_cycle=five_cycle(), lp=[])
    for n in range(5, n_max + 1):
        rec["lp"].append(random_objectives(n, trials, rng))
        say(json.dumps(rec["lp"][-1]))
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    assert rec["five_cycle"]["consecutive_cross"] and rec["five_cycle"]["non_consecutive_compatible"]
    assert all(r["fractional_optima"] > 0 for r in rec["lp"]), "some n gave only integral optima"


if __name__ == "__main__":
    main()
