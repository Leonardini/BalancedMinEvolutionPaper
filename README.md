# Reproducing "A Compact Weight-Space Integer Programming Formulation for Balanced Minimum Evolution"

This repository contains everything needed to regenerate every computed result in the
paper: the code, the input data, the commands, and the raw output of our own runs. The
paper itself is not here.

The paper compares two exact solvers for the Balanced Minimum Evolution (BME) problem:

- **the compact solver**: ours, a branch-and-bound method in the n(n−1)/2 weights
  w_ij = 2^(−τ_ij), where τ_ij is the number of edges between leaves i and j (Python,
  with CPLEX solving the linear programs and MOSEK the convex manifold constraint);
- **the distance-indexed solver**: the branch-and-cut solver of Catanzaro, Pesenti,
  Sapucaia and Wolsey (Math. Program. 2026), which works with the path lengths τ and
  with indicator variables for their values (C++, with Gurobi).

## What you need

| software | version we used | used for |
|---|---|---|
| macOS on Apple silicon | 27.0 | everything (other Unix systems should work with small changes; the memory watchdog reads swap with macOS's `sysctl`) |
| Python | 3.12 | the compact solver and every experiment script |
| CPLEX Python API | 22.1.2 | the linear programs of the compact solver |
| MOSEK Python API | 11.2.5 | the manifold constraint at every node of the compact solver |
| Gurobi, with gurobipy | 12.0.1 | the distance-indexed solver, and most bound experiments |
| CVXPY | 1.9.3 | the bound experiments that impose the manifold constraint exactly (with MOSEK) |
| FastME | 2.1.6.3 | the starting tree of both solvers |
| boost | Homebrew | building the distance-indexed solver |
| R, with packages `rcdd` and `jsonlite` | 4.5.3 | the facets of the BME polytope for n = 6 (Section 4) |
| a C compiler | Apple clang 16 | the enumerators and two small extensions of the compact solver |

CPLEX, MOSEK and Gurobi need licences; all three offer free academic licences.

## Setting up

```bash
python3.12 -m venv ~/venvs/bmepaper
~/venvs/bmepaper/bin/pip install numpy scipy networkx mpmath cplex==22.1.2.1 mosek==11.2.5 \
    gurobipy==12.0.1 cvxpy==1.9.3
brew install fastme boost
Rscript -e 'install.packages(c("rcdd", "jsonlite"))'

# The distance-indexed solver and the benchmark matrices are not redistributed here.
# fetch.sh downloads them from the authors' site at a fixed version and checks the
# checksum; build.sh applies our patch and builds the solver.
external/catanzaro/fetch.sh
external/catanzaro/build.sh     # set GUROBI_HOME if Gurobi is not in /Library/gurobi1201
```

The scripts expect the Python interpreter at `~/venvs/bmepaper/bin/python`; set
`BME_PYTHON` to use another one.

Our patch (`external/catanzaro/catanzaro_repro.patch`) leaves the algorithm unchanged. It
links Gurobi 12 instead of Gurobi 10, finds boost under Homebrew, and adds an
`--export FILE` option that writes the model before any cut is added (used for Table D12).

To check the installation, run the tests (a few minutes on one core):

```bash
~/venvs/bmepaper/bin/python compact/tests/test_face.py
~/venvs/bmepaper/bin/python compact/tests/test_cut_enumeration.py
BME_THREADS=1 ~/venvs/bmepaper/bin/python experiments/lp/tests/test_safe_bound.py
```

## Running everything

All settings shared by the experiments (thread counts, the time cap, the seeds, the
solver options) are in `run/config.sh`. There are two kinds of run, which must not run at
the same time, because the first kind measures time:

```bash
run/timed_runs.sh       # every solve that reports a time: 10 threads, 3600 s cap, one at a time
run/counting_runs.sh    # everything else: bounds and counts, one thread each, 10 at a time
```

- **Timed runs** are the runs of either solver that search for a certified optimum. Each
  uses 10 threads (the compact solver: 10 worker processes of one thread each) and stops
  after 3600 s. Each run uses random seed 1, and also seeds 2 and 3 if seed 1 certified the
  optimum, so that times can be reported as a median and a range; the runs of Table E4
  use seed 1 only. On our laptop (14 cores, 24 GB of memory) they take about a day.
- **Counting runs** compute a bound, a count, or the result of an exhaustive search, and
  report no time. Each uses one thread, and ten run side by side. They are listed in
  `run/counting_runs.d/*.tasks`, one shell command per line, with comments saying which
  result each line produces. Run them after the timed runs: some read the certified
  optima. They take about a day.

Both scripts skip work that has already finished, so they can be stopped and restarted.
Since `results/` holds our own output, they do nothing on a fresh clone: to regenerate
everything, move `results/` aside first (and compare afterwards); to regenerate one
result, delete its output. Both log one line per step (`results/timed_runs.log`, `results/counting_runs.log`).
`run/watchdog.sh` runs alongside them and stops the largest experiment if their combined
memory passes 12 GB.

To regenerate a single result, run its line from the table below, or its line of a
`.tasks` file from the repository root with `BME_THREADS=1` and `PY` and `RESULTS` set as
in `run/config.sh`.

Then rebuild the summaries:

```bash
~/venvs/bmepaper/bin/python experiments/check_optima.py results/check_optima.json
~/venvs/bmepaper/bin/python experiments/summarize.py          # writes results/SUMMARY.md
~/venvs/bmepaper/bin/python experiments/appendix_tables.py    # writes results/APPENDIX_TABLES.md
```

`results/APPENDIX_TABLES.md` holds the tables of Appendices D and E of the paper, as
printed there. `results/SUMMARY.md` holds the other computed results: the certified
optima, and the results of Sections 4, 5.1, 5.2 and 6.3 and Appendix C.

## Where each result comes from

| in the paper | script | output under `results/` |
|---|---|---|
| Section 4: facets of the BME polytope for n = 6 | `experiments/theory/p6_facets.R` | `p6_facets/` |
| Section 5.1, Conjecture 1: integral points of the membership model | `experiments/theory/membership_model.py` | `conjecture1/` |
| Section 5.2: minimum edge clique covers; fractional optima | `experiments/theory/clique_cover.py`, `clique_integrality.py` | `clique_cover/` |
| Section 6.3, Theorem 5, Appendix C, Table C1 | `experiments/theory/theorem5.sh` (enumerator `backtrack.c`) | `theorem5/` |
| Section 6.3: the cherry lift; the matrix of Catanzaro et al. (2026, p. 492) | `experiments/theory/cherry_lift.py`, `catanzaro_p492.py` | `theorem5/` |
| Section 8.1, Tables D1 and D2: the compact solver | `experiments/compact_solver.sh` (one run: `compact_solver.py`) | `compact_solver/` |
| Section 8.2, Table D3: facets F1 and F6 | `experiments/lp/named_facets.py` | `lp/named_facets.json` |
| Section 8.3, Table D4: shares of the cut families | `experiments/lp/cut_family_shares.py` | `lp/cut_family_shares/` |
| Section 8.4, Table D5: membership model; lifted ladder model | `experiments/lp/membership_layers.py`, `lifted_ladder.py` | `lp/membership_layers/`, `lp/lifted_ladder/` |
| Section 8.4, Table D6: per-split triangles; odd cycles | `experiments/lp/per_split_triangles.py`, `odd_cycles.py` | `lp/per_split_triangles/`, `lp/odd_cycles/` |
| Section 8.5, Table D7: local fixes of the quartets violated at the root | `experiments/lp/quartet_fixes.py` | `lp/quartet_fixes/` |
| Section 8.6, Tables D8 and D2: the distance-indexed solver | `experiments/distance_solver.sh` | `distance_solver/` |
| Section 8.7, Table D9: the distance-indexed solver by cut family | `experiments/distance_cut_families.sh` | `distance_cut_families/` |
| Section 8.7: the non-tree optima | `experiments/theory/nontree_optima.py` | `theorem5/nontree_optima.json` |
| Section 8.7, Table D10: rows of the distance-indexed model added to the compact one | `experiments/base_rows/base_rows_in_compact.py` | `base_rows/` |
| Section 8.7, Table D11: the distance-indexed root with Gurobi's cuts on and off | `experiments/distance_root_cuts.sh` | `distance_root_cuts/` |
| Section 8.7, Table D12: the static root in Gurobi and CPLEX | `experiments/static_root_two_solvers.py` | `static_root_two_solvers/` |
| Appendix E: the certified optima, checked exactly | `experiments/check_optima.py` | `check_optima.json` |
| Appendix E, Table E1: safe bounds | inside every bound script (`experiments/lp/lib/safe_bound.py`) | in each JSON record |
| Appendix E, Table E2: certificates against enumeration of all trees | `experiments/enumeration/certificates_vs_enumeration.py` | `enumeration/` |
| Appendix E, Table E3: condition numbers | `experiments/conditioning.py` | `conditioning/` |
| Appendix E, Table E4: one cut family off; strict tolerances | `experiments/distance_robustness.sh` | `robustness/` |

## Reading the output

- **A run of the distance-indexed solver** is a directory `.../<instance>/seed<k>/`. It
  holds the solver's log (`log.txt`), its one-line report (`report.txt`, read by
  `run/parse_report.py`), the command and the Gurobi settings it ran with (`command.txt`,
  `gurobi.env`), the path-length matrix of its best tree (`tau.txt`) and its time and peak
  memory (`time.txt`). The solver reports objective values multiplied by 2^(n+1)
  relative to the paper.
- **A run of the compact solver** is a file `seed<k>.json`: the best value found, the lower
  bound, whether it certified, the node count, the time and the root bound, with a log of
  every node and the lower bound over time. Its progress lines are in `seed<k>.err`.
- **A bound experiment** writes one JSON file. Next to each bound it records
  `safe_bound`, a lower bound computed in exact arithmetic from the solver's dual
  solution, and `rows_cut_off_optimum`, the number of rows of the final linear program
  that the optimal tree violates (0 in every run).
- Every script logs its progress with timestamps to standard error; the run scripts keep
  these logs next to the results.

## Layout

- `compact/solver/`: the compact solver. `bnb_balanced.py` holds the branch and bound
  (the sequential search, and the processing of one node), `bnb_parallel.py` and
  `parallel_worker.py` the parallel search, `exact_node.py` and `conic_persistent.py`
  the manifold constraint at each node, `cut_loop.py` the separation of cuts, `face.py`
  the nodes whose point has no split to branch on. `compact/tests/` holds its tests.
- `external/catanzaro/`: the download, patch and build scripts for the distance-indexed
  solver.
- `data/synthetic/`: the synthetic instances, n = 4, …, 10, each the leading part of the
  next.
- `data/ground_truth/`: the certified optimal tree of each real instance, as its list of
  splits, written from a certified run of the distance-indexed solver by
  `experiments/tau_to_splits.py`. The bound experiments read these trees;
  `experiments/check_optima.py` checks that each is optimal.
- `data/theory/`: the 12×12 matrix printed by Catanzaro et al. (2026, p. 492).
- `experiments/`: one script per result (see the table above).
- `run/`: the shared settings, the two run scripts and their task lists, the wrapper that
  runs the distance-indexed solver once (`catanzaro_one.sh`) and the memory watchdog.
- `results/`: the output of our runs, in the layout above.
