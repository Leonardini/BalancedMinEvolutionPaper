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
| FastME | 2.1.6.3 | the starting tree of both solvers, and the NJ and FastME trees of table "NJ and FastME against the certified optimum" |
| boost | Homebrew | building the distance-indexed solver |
| R, with packages `rcdd` and `jsonlite` | 4.5.3 | the facets of the BME polytope for n = 6 (Results, "The minimum-cut family accounts for most of the root bound") |
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
`--export FILE` option that writes the model before any cut is added (used for table "The distance-indexed model's static root in two MIP solvers").

To check the installation, run the tests (a few minutes on one core):

```bash
~/venvs/bmepaper/bin/python compact/tests/test_face.py
~/venvs/bmepaper/bin/python compact/tests/test_cut_enumeration.py
BME_THREADS=1 ~/venvs/bmepaper/bin/python experiments/lp/tests/test_safe_bound.py
~/venvs/bmepaper/bin/python experiments/conjecture/tests/test_partition.py
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
  optimum, so that times can be reported as a median and a range; the runs of table "The distance-indexed solver with one cut family switched off, or with strict tolerances"
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

`results/APPENDIX_TABLES.md` holds the tables of the appendices "Tables of results" and
"Numerical reliability" of the paper, as printed there. `results/SUMMARY.md` holds the other computed results: the certified
optima, the clique covers, the spurious points of the faces Q_T, the facets of P_6, the
enumeration behind the sufficiency theorem and the evidence for the cherry conjecture (Discussion and open problems). It ends with every number of `experiments/theory/published.json` (the numbers stated in
the paper) that a result file restates, recomputed from that file; `summarize.py` stops with
an error if one of them disagrees.

## Where each result comes from

| in the paper | script | output under `results/` |
|---|---|---|
| Results, "The minimum-cut family accounts for most of the root bound": facets of the BME polytope for n = 6 | `experiments/theory/p6_facets.R` | `p6_facets/` |
| Appendix E: vertices of the splitohedron for n = 5, 6 | `experiments/theory/splitohedron.R` | `splitohedron/` |
| "Spurious points with the splits held at 1/2" and appendix "The splitohedron": spurious points of the faces Q_T of the splitohedron (log-integral points without and with the ladder floor, and without the double cherry; vertices of Q_T; hull of the log-integral points) | `experiments/theory/lattice.py`, `qt_vertices.R`, `hull_vertices.R` (and `outside_trees.py`, which tests whether a log-integral point lies outside the BME polytope) | `splitohedron/faces/` |
| "Incompatibility and a polynomial clique cover" and appendix "Clique covers of the crossing graph": edge clique covers (the polynomial-laminarity proposition and its subfamily, the minimum, the fractional minimum); fractional optima of the clique relaxation | `experiments/theory/clique_cover.py`, `clique_integrality.py` | `clique_cover/` |
| "Sufficiency of the path-length conditions": the sufficiency theorem, its computer-assisted proof (appendix "Proofs") and table "Canonical survivors of the enumeration" | `experiments/theory/theorem5.sh` (enumerator `backtrack.c`) | `theorem5/` |
| "Sufficiency of the path-length conditions": the cherry lift; the matrix of Catanzaro et al. (2026, p. 492) | `experiments/theory/cherry_lift.py`, `catanzaro_p492.py` | `theorem5/` |
| Discussion and open problems, the cherry conjecture: exhaustive search for matrices satisfying (4)–(9) without a cherry, n ≤ 15 | `experiments/conjecture/nocherry_search.py` (enumerator `backtrack_wide8.c`; `coverage.py` checks that the pieces cover the search) | `conjecture/nocherry/` |
| Discussion and open problems, the cherry conjecture: the manifold alone forces a cherry for small n | `experiments/conjecture/cherry_bound.py` | `conjecture/cherry_bound.json` |
| Results, "The compact solver certifies every instance up to n = 20", table "The compact solver" and table "The two solvers": the compact solver | `experiments/compact_solver.sh` (one run: `compact_solver.py`) | `compact_solver/` |
| Results, "The minimum-cut family accounts for most of the root bound", table "Root bound with F1 and F6" | `experiments/lp/named_facets.py` | `lp/named_facets.json` |
| Results, "The minimum-cut family accounts for most of the root bound", table "Shares of the root cut-closeable gap" | `experiments/lp/cut_family_shares.py` | `lp/cut_family_shares/` |
| Results, "The minimum-cut family accounts for most of the root bound": the lifted ladder model (table "Root gap on Primates12 of the lifted ladder model"), and its gap with the exact hull of the trees for n ≤ 7 | `experiments/lp/lifted_ladder.py`, `tree_hull.py` | `lp/lifted_ladder/` |
| Results, "The minimum-cut family accounts for most of the root bound", table "Local fixes of the quartets violated at the compact model's root" | `experiments/lp/quartet_fixes.py` | `lp/quartet_fixes/` |
| Results, "The distance-indexed solver is faster on four of the six instances both certify", table "The distance-indexed solver" and table "The two solvers": the distance-indexed solver | `experiments/distance_solver.sh` | `distance_solver/` |
| Results, "The root bound needs separated families and solver cuts together", table "The distance-indexed model by cut configuration" | `experiments/distance_cut_families.sh` | `distance_cut_families/` |
| Results, "Four-point separation speeds certification", table "The distance-indexed model by cut configuration" and table "The lazy-mode configurations with the released and the corrected no-good cut": the configurations without four-point separation with the corrected no-good cut (`external/catanzaro/nogood_index_fix.patch`, built by `external/catanzaro/build.sh --fix-nogood`), and the published UBT-BC configuration with both builds | `experiments/rerun_nogoodfix.sh` | `distance_cut_families_nogoodfix/`, `distance_cut_families_ubtbc_original/` |
| Results, "Four-point separation speeds certification": whether each of those runs ends at a tree (four-point condition on every quartet, tree rebuilt, objective recomputed) | `experiments/nogood_endpoints.py` | `nogood_endpoints.json` |
| Results, "Four-point separation speeds certification": the non-tree optima | `experiments/theory/nontree_optima.py` | `theorem5/nontree_optima.json` |
| Results, "The root bound needs separated families and solver cuts together", table "Rows of the distance-indexed base model added to the compact root relaxation" | `experiments/base_rows/base_rows_in_compact.py` | `base_rows/` |
| Results, "The root bound needs separated families and solver cuts together", table "Where the distance-indexed model's root bound comes from": Gurobi's cuts on and off | `experiments/distance_root_cuts.sh` | `distance_root_cuts/` |
| Results, "The root bound needs separated families and solver cuts together", table "The distance-indexed model's static root in two MIP solvers" | `experiments/static_root_two_solvers.py` | `static_root_two_solvers/` |
| Results, "NJ and FastME miss the optimal tree on 9 and 4 of 16 real instances", table "NJ and FastME against the certified optimum" | `experiments/heuristics_vs_optimum.py` | `heuristics_vs_optimum.json` |
| Appendix J: the certified optima, checked exactly | `experiments/check_optima.py` | `check_optima.json` |
| Numerical reliability, table "Safe bounds and the exact check at the optimal tree" | inside every bound script (`experiments/lp/lib/safe_bound.py`) | in each JSON record |
| Numerical reliability, table "Certificates against exhaustive enumeration" | `experiments/enumeration/certificates_vs_enumeration.py` | `enumeration/` |
| Numerical reliability, table "Condition number of the optimal root basis" | `experiments/conditioning.py` | `conditioning/` |
| Numerical reliability, table "The distance-indexed solver with one cut family switched off, or with strict tolerances" | `experiments/distance_robustness.sh` | `robustness/` |

## The search without cherries (the cherry conjecture)

`backtrack_wide8.c` enumerates the matrices satisfying (4)–(9) with no entry 2, with leaf 0's
row in non-decreasing order. Fixing the first values of that row, tau[0][1..k] = v1 ≤ … ≤ vk,
cuts the search into independent pieces; `nocherry_search.py top N BIN DIR --depth K` runs
every piece of depth K (v1 ranges over the smallest entries of the row types), and
`nocherry_search.py split N BIN DIR PREFIX…` replaces a piece that is too slow by its pieces
one level deeper. `coverage.py DIR N` checks that the finished pieces in DIR cover the whole
search, lists the prefixes still missing and the parts covered twice, and exits 1 unless the
search is covered with no survivor. For n ≤ 14 each search runs as one piece
(`run/counting_runs.d/conjecture.tasks`).

The n = 15 search (its core time is in `results/SUMMARY.md`) ran in pieces, ten at a time: every piece of depth
5; then the slowest of those (3_3_3_4_5, 3_3_3_4_6, 3_3_3_4_7, 3_3_3_5_5) split to depth 7 (in
one step; `split` goes one level at a time, so it takes two); the
five pieces of depth 7 that did not finish (3_3_3_4_5_6_9, 3_3_3_4_5_7_7, 3_3_3_4_5_7_8,
3_3_3_4_5_8_8, 3_3_3_5_5_5_6) split to depth 8; and the slowest of these split to depth 9:

```bash
make -s -C experiments/conjecture
B=experiments/conjecture/build/backtrack_wide8; D=results/conjecture/nocherry/n15
$PY experiments/conjecture/nocherry_search.py top 15 $B $D --depth 5 --workers 10
$PY experiments/conjecture/nocherry_search.py split 15 $B $D 3_3_3_4_5 3_3_3_4_6 3_3_3_4_7 3_3_3_5_5 --workers 10
# ... and so on, splitting each piece that is too slow; coverage.py lists what is missing:
$PY experiments/conjecture/coverage.py $D 15
```

`results/conjecture/nocherry/provenance.txt` lists which run wrote which file. The original
driver missed one slice, 3_3_3_5_5_5_6 with t08 = 8 (it split t08 = 6 a second time instead);
its seven pieces t09 = 8..14 were run in November 2026, and with them `coverage.py` reports
n = 15 as covered with no survivor. The equivalent command is

```bash
$PY experiments/conjecture/nocherry_search.py split 15 $B $D 3_3_3_5_5_5_6_8 --workers 7
```

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
- `data/synthetic/`: the synthetic instances test_n4 to test_n10 (n = 4, …, 10); test_n5 is a
  separate tree metric, and each of the others is a leading square submatrix of test_n10.
- `data/ground_truth/`: the certified optimal tree of each real instance, as its list of
  splits, written from a certified run of the distance-indexed solver by
  `experiments/tau_to_splits.py`. The bound experiments read these trees;
  `experiments/check_optima.py` checks that each is optimal.
- `data/theory/`: the 12×12 matrix printed by Catanzaro et al. (2026, p. 492).
- `experiments/`: one script per result (see the table above); `experiments/conjecture/`
  holds the scripts of the cherry conjecture.
- `run/`: the shared settings, the two run scripts and their task lists, the wrapper that
  runs the distance-indexed solver once (`catanzaro_one.sh`) and the memory watchdog.
- `results/`: the output of our runs, in the layout above.
