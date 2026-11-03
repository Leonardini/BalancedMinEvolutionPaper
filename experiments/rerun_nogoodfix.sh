#!/bin/bash
# Rerun the lazy-mode configurations (gki, c) of distance_cut_families.sh with the fixed
# no-good cut (external/catanzaro/build.sh --fix-nogood), into a separate results folder,
# so the published results stay as they are.
#   rerun_nogoodfix.sh [--wait-for PATTERN]
# --wait-for: start only once no process matches PATTERN (pgrep -f) and the 1-minute load
# is below 3, so the timed runs do not share the machine. 10 threads, ~1.5 GB per run.
# A watchdog stops the run if any solver exceeds 8 GB RSS or swap use exceeds 4 GB.
# Progress: $OUT/run.log (one start/end line per run). Stop: pkill -f rerun_nogoodfix.sh; pkill -f solver_bmep
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../run/config.sh"
export CATANZARO_BIN=$REPO/external/catanzaro/build_nogoodfix/bin/solver_bmep
export OUT=$RESULTS/distance_cut_families_nogoodfix
export ONLY="gki c"
[ -x "$CATANZARO_BIN" ] || { echo "build first: external/catanzaro/build.sh --fix-nogood" >&2; exit 1; }
mkdir -p "$OUT"
log=$OUT/run.log
if [ "${1:-}" = --wait-for ]; then
  pattern=$2
  echo "$(date '+%F %T') waiting for: $pattern" >> "$log"
  while pgrep -f "$pattern" > /dev/null; do sleep 60; done
  while [ "$(sysctl -n vm.loadavg | awk '{print int($2)}')" -ge 3 ]; do sleep 60; done
  echo "$(date '+%F %T') machine free; starting" >> "$log"
fi
watchdog() {
  while sleep 30; do
    swap_mb=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}')
    rss_kb=$(ps -Ao rss=,command= | awk '/solver_bmep/ && !/awk/ {if ($1>m) m=$1} END {print m+0}')
    echo "$(date '+%F %T') watchdog swap=${swap_mb}MB max_solver_rss=$((rss_kb/1024))MB" >> "$OUT/watchdog.log"
    if [ "$rss_kb" -gt 8388608 ] || [ "$swap_mb" -gt 4096 ]; then
      echo "$(date '+%F %T') WATCHDOG KILL swap=${swap_mb}MB rss=${rss_kb}KB" >> "$log"
      pkill -f "$CATANZARO_BIN"; pkill -f distance_cut_families.sh
    fi
  done
}
watchdog & wd=$!
trap 'kill $wd 2>/dev/null' EXIT
echo "$(date '+%F %T') binary $CATANZARO_BIN" >> "$log"
"$here/distance_cut_families.sh" timed >> "$log" 2>&1
echo "$(date '+%F %T') all done" >> "$log"
