# Settings shared by every experiment. Source this file; do not copy values out of it.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${BME_PYTHON:-$HOME/venvs/bmepaper/bin/python}

# Global solves (Catanzaro's solver and ours) all use one thread count and one cap, so
# their times and node counts are comparable. Counting runs use one thread.
TIMED_THREADS=10
TIMED_CAP=3600
SEEDS=${SEEDS:-"1 2 3"}
COUNT_THREADS=1
COUNT_PARALLEL=10

CATANZARO_SRC=$REPO/external/catanzaro/upstream
CATANZARO_BIN=$REPO/external/catanzaro/build/bin/solver_bmep
INST=$CATANZARO_SRC/instances
SUP=$INST/supplement
RESULTS=$REPO/results

# The configuration of Section 7.1: every separation family at the implementation's
# default thresholds, LP-NJ incumbent with FastME local search.
CAT_FULL="-O B --heuk 4 --heuls --str O --buneman O --co G --coR 0.05 --coN 0.005 --gkiR 0.05 --gkiN 0.01 --wbR 0.125 --2KR 0.001"

# True if a finished run of Catanzaro's solver in directory $1 certified optimality.
# A run with no report (crashed or killed) counts as not certified.
certified() {
  [ -f "$1/report.txt" ] && [ "$("$PY" "$REPO/run/parse_report.py" "$1" certified)" = True ]
}
