#!/bin/bash
# Every solve that reports a time: TIMED_THREADS threads (run/config.sh), TIMED_CAP seconds,
# one at a time. Nothing else from this repository may run alongside it. Seed 1 of every
# experiment runs before any seed 2, so an interrupted run still has one complete pass.
# Finished runs are skipped, so the script can be stopped and restarted.
# Progress: results/timed_runs.log (one line per run start and end) and results/watchdog.log.
set -euo pipefail
source "$(dirname "$0")/config.sh"
mkdir -p "$RESULTS"
exec >> "$RESULTS/timed_runs.log" 2>&1
echo "$(date '+%F %T') timed runs start (pid $$, threads $TIMED_THREADS, cap $TIMED_CAP)"
"$REPO/run/watchdog.sh" $$ &
for s in $SEEDS; do
  # A failed experiment is logged and the run moves on to the next one.
  SEEDS=$s "$REPO/experiments/distance_solver.sh" || echo "FAILED distance_solver seed $s"
  if [ "$s" = 1 ]; then "$REPO/experiments/distance_robustness.sh" || echo "FAILED distance_robustness"; fi
  SEEDS=$s "$REPO/experiments/distance_cut_families.sh" timed || echo "FAILED distance_cut_families seed $s"
  SEEDS=$s "$REPO/experiments/compact_solver.sh" || echo "FAILED compact_solver seed $s"
done
echo "$(date '+%F %T') timed runs done"
