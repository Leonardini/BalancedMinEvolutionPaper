#!/bin/bash
# Memory watchdog for run/timed_runs.sh and run/counting_runs.sh: samples every 30 s and
# writes one line per sample to $RESULTS/watchdog.log. It kills the largest of the launcher's
# descendant processes if
#   - their combined RSS exceeds RSS_CAP_MB (default 12 GB), or
#   - swap use is above SWAP_FLOOR_MB and grew by more than SWAP_STEP_MB in one sample.
# It exits when the launcher (PID given as $1) exits. Swap is read with macOS's sysctl.
#   watchdog.sh LAUNCHER_PID
source "$(dirname "$0")/config.sh"
launcher=$1
RSS_CAP_MB=${RSS_CAP_MB:-12288}
SWAP_FLOOR_MB=${SWAP_FLOOR_MB:-4096}
SWAP_STEP_MB=${SWAP_STEP_MB:-1024}
LOG=$RESULTS/watchdog.log
descendants() {  # every descendant PID of $1, one per line
  local c
  for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done
}
prev_swap=$(sysctl -n vm.swapusage | awk '{print $6}' | sed 's/M//' | cut -d. -f1)  # compare growth from the start, not from zero
mkdir -p "$RESULTS"
while kill -0 "$launcher" 2> /dev/null; do
  swap=$(sysctl -n vm.swapusage | awk '{print $6}' | sed 's/M//' | cut -d. -f1)
  snap=$(ps -axo pid=,rss=,command= | awk -v pids=" $(descendants "$launcher" | tr '\n' ' ') " 'index(pids, " " $1 " ")')
  total_mb=$(printf '%s\n' "$snap" | awk '{s+=$2} END {print int(s/1024)}')
  big=$(printf '%s\n' "$snap" | sort -k2 -n -r | head -1)
  big_pid=$(printf '%s' "$big" | awk '{print $1}')
  big_mb=$(printf '%s' "$big" | awk '{print int($2/1024)}')
  echo "$(date '+%F %T') swap=${swap}M rss_total=${total_mb}M largest=${big_pid:-none}:${big_mb:-0}M load=$(sysctl -n vm.loadavg | awk '{print $2}')" >> "$LOG"
  if [ -n "$big_pid" ] && [ "$total_mb" -gt "$RSS_CAP_MB" ]; then
    echo "$(date '+%F %T') KILL $big_pid: combined RSS ${total_mb}M > ${RSS_CAP_MB}M" >> "$LOG"
    kill "$big_pid"
  elif [ -n "$big_pid" ] && [ "$swap" -gt "$SWAP_FLOOR_MB" ] && [ $((swap - prev_swap)) -gt "$SWAP_STEP_MB" ]; then
    echo "$(date '+%F %T') KILL $big_pid: swap ${swap}M grew by $((swap - prev_swap))M" >> "$LOG"
    kill "$big_pid"
  fi
  prev_swap=$swap
  sleep 30
done
echo "$(date '+%F %T') launcher $launcher exited; watchdog done" >> "$LOG"
