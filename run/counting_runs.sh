#!/bin/bash
# Every run that reports a bound, a count or the result of a search, and no time. Each task
# runs on one thread; COUNT_PARALLEL tasks (run/config.sh) run at once. Run it after
# run/timed_runs.sh, never alongside it: some tasks read the certified optima of the timed
# runs.
#
# Tasks are listed in run/counting_runs.d/*.tasks, one shell command per line (blank lines and
# lines starting with # are skipped), and run by run_tasks.py from the repository root with
# BME_THREADS=1 and OMP_NUM_THREADS=1 set. Each task writes its own output under results/;
# its stdout and stderr go to results/counting_logs/<file>_<line>.log. Finished tasks are
# skipped, so the script can be stopped and restarted.
set -euo pipefail
source "$(dirname "$0")/config.sh"
mkdir -p "$RESULTS/counting_logs"
exec >> "$RESULTS/counting_runs.log" 2>&1
echo "$(date '+%F %T') counting runs start (pid $$, $COUNT_PARALLEL at a time, 1 thread each)"
"$REPO/run/watchdog.sh" $$ &
export REPO PY RESULTS BME_THREADS=$COUNT_THREADS OMP_NUM_THREADS=1
"$PY" "$REPO/run/run_tasks.py" "$COUNT_PARALLEL" "$REPO"/run/counting_runs.d/*.tasks
echo "$(date '+%F %T') counting runs done"
