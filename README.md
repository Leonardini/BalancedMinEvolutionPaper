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
| FastME | 2.1.6.3 | the starting tree of both solvers, and the NJ and FastME trees of Table I13 |
| boost | Homebrew | building the distance-indexed solver |
| R, with packages `rcdd` and `jsonlite` | 4.5.3 | the facets of the BME polytope for n = 6 (Section 7.1) |
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
`--export FILE` option that writes the model before any cut is added (used for Table I12).

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
  optimum, so that times can be reported as a median and a range; the runs of Table J4
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

`results/APPENDIX_TABLES.md` holds the tables of Appendices I and J of the paper, as
printed there. `results/SUMMARY.md` holds the other computed results: the certified
optima, and the results of Sections 4.2, 5.1, 7.1, 7.2, 8.3 and 11 and Appendices D, E and
H. It ends with every number of `experiments/theory/published.json` (the numbers stated in
the paper) that a result file restates, recomputed from that file; `summarize.py` stops with
an error if one of them disagrees.

## Where each result comes from

| in the paper | script | output under `results/` |
|---|---|---|
| Section 7.1: facets of the BME polytope for n = 6 | `experiments/theory/p6_facets.R` | `p6_facets/` |
| Appendix E: vertices of the splitohedron for n = 5, 6 | `experiments/theory/splitohedron.R` | `splitohedron/` |
| Section 5.1 and Appendix E: spurious points of the faces Q_T of the splitohedron (log-integral points without and with the ladder floor, and without the double cherry; vertices of Q_T; hull of the log-integral points) | `experiments/theory/lattice.py`, `qt_vertices.R`, `hull_vertices.R` (and `outside_trees.py`, which tests whether a log-integral point lies outside the BME polytope) | `splitohedron/faces/` |
| Section 7.2, Conjecture 6: integral points of the membership model | `experiments/theory/membership_model.py` | `conjecture1/` |
| Section 4.2 and Appendix D: edge clique covers (Proposition 4 and its subfamily, the minimum, the fractional minimum); fractional optima of the clique relaxation | `experiments/theory/clique_cover.py`, `clique_integrality.py` | `clique_cover/` |
| Section 8.3, Theorem 7, Appendix H, Table H1 | `experiments/theory/theorem5.sh` (enumerator `backtrack.c`) | `theorem5/` |
| Section 8.3: the cherry lift; the matrix of Catanzaro et al. (2026, p. 492) | `experiments/theory/cherry_lift.py`, `catanzaro_p492.py` | `theorem5/` |
| Section 11, Conjecture 8: exhaustive search for matrices satisfying (4)–(9) without a cherry, n ≤ 14 (and n = 15, one slice still to run; see below) | `experiments/conjecture/nocherry_search.py` (enumerator `backtrack_wide8.c`; `coverage.py` checks that the pieces cover the search) | `conjecture/nocherry/` |
| Section 11, Conjecture 8: the manifold alone forces a cherry for small n | `experiments/conjecture/cherry_bound.py` | `conjecture/cherry_bound.json` |
| Section 10.1, Tables I1 and I2: the compact solver | `experiments/compact_solver.sh` (one run: `compact_solver.py`) | `compact_solver/` |
| Section 10.2, Table I3: facets F1 and F6 | `experiments/lp/named_facets.py` | `lp/named_facets.json` |
| Section 10.3, Table I4: shares of the cut families | `experiments/lp/cut_family_shares.py` | `lp/cut_family_shares/` |
| Section 10.4, Table I5: membership model; lifted ladder model, and its gap with the exact hull of the trees for n ≤ 7 | `experiments/lp/membership_layers.py`, `lifted_ladder.py`, `tree_hull.py` | `lp/membership_layers/`, `lp/lifted_ladder/` |
| Section 10.4, Table I6: per-split triangles; odd cycles | `experiments/lp/per_split_triangles.py`, `odd_cycles.py` | `lp/per_split_triangles/`, `lp/odd_cycles/` |
| Section 10.5, Table I7: local fixes of the quartets violated at the root | `experiments/lp/quartet_fixes.py` | `lp/quartet_fixes/` |
| Section 10.6, Tables I8 and I2: the distance-indexed solver | `experiments/distance_solver.sh` | `distance_solver/` |
| Section 10.7, Table I9: the distance-indexed solver by cut family | `experiments/distance_cut_families.sh` | `distance_cut_families/` |
| Section 10.7, Tables I9 and I10: the configurations without four-point separation with the corrected no-good cut (`external/catanzaro/nogood_index_fix.patch`, built by `external/catanzaro/build.sh --fix-nogood`), and the published UBT-BC configuration with both builds | `experiments/rerun_nogoodfix.sh` | `distance_cut_families_nogoodfix/`, `distance_cut_families_ubtbc_original/` |
| Section 10.7: whether each of those runs ends at a tree (four-point condition on every quartet, tree rebuilt, objective recomputed) | `experiments/nogood_endpoints.py` | `nogood_endpoints.json` |
| Section 10.7: the non-tree optima | `experiments/theory/nontree_optima.py` | `theorem5/nontree_optima.json` |
| Section 10.7, Table I10: rows of the distance-indexed model added to the compact one | `experiments/base_rows/base_rows_in_compact.py` | `base_rows/` |
| Section 10.7, Table I11: the distance-indexed root with Gurobi's cuts on and off | `experiments/distance_root_cuts.sh` | `distance_root_cuts/` |
| Section 10.7, Table I12: the static root in Gurobi and CPLEX | `experiments/static_root_two_solvers.py` | `static_root_two_solvers/` |
| Section 10.8, Table I13: NJ and FastME against the certified optimal trees | `experiments/heuristics_vs_optimum.py` | `heuristics_vs_optimum.json` |
| Appendix J: the certified optima, checked exactly | `experiments/check_optima.py` | `check_optima.json` |
| Appendix J, Table J1: safe bounds | inside every bound script (`experiments/lp/lib/safe_bound.py`) | in each JSON record |
| Appendix J, Table J2: certificates against enumeration of all trees | `experiments/enumeration/certificates_vs_enumeration.py` | `enumeration/` |
| Appendix J, Table J3: condition numbers | `experiments/conditioning.py` | `conditioning/` |
| Appendix J, Table J4: one cut family off; strict tolerances | `experiments/distance_robustness.sh` | `robustness/` |

## The search without cherries (Conjecture 8)

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

`results/conjecture/nocherry/provenance.txt` lists which original run wrote which file. One
slice was never run: 3_3_3_5_5_5_6 with t08 = 8 (the original driver split t08 = 6 a second
time instead). Until its seven pieces are in `results/conjecture/nocherry/n15/`,
`coverage.py` reports n = 15 as not covered, and `published.json` (`cherry_conjecture`)
records the paper's claim as n ≤ 14 with that slice pending. To finish it:

```bash
$PY experiments/conjecture/nocherry_search.py split 15 $B $D 3_3_3_5_5_5_6_8 --workers 7
$PY experiments/conjecture/coverage.py $D 15        # exits 0 once n = 15 is covered
```

and then set `exhaustive_max_n` to 15 and remove the pending entry in `published.json`, once
the paper states n ≤ 15; `summarize.py` stops until the two agree.

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
  holds the scripts of Conjecture 8.
- `run/`: the shared settings, the two run scripts and their task lists, the wrapper that
  runs the distance-indexed solver once (`catanzaro_one.sh`) and the memory watchdog.
- `results/`: the output of our runs, in the layout above.
