#!/bin/bash
# Section 8.1, Tables D1 and D2: the compact solver, timed. Ten workers of one thread each,
# TIMED_CAP seconds, one run at a time. Seed 1 runs everywhere; seeds 2 and 3 run only where
# seed 1 certified, because only those runs report a time and node count that a median and
# range describe.
set -euo pipefail
source "$(dirname "$0")/../run/config.sh"
OUT=$RESULTS/compact_solver
LIST=""
for n in 5 6 7 8 9 10; do
  LIST="$LIST
test_n$n|$REPO/data/synthetic/test_n$n.txt"
done
LIST="$LIST
Primates12|$INST/01-Primates12.txt
M17|$INST/02-M17.txt
M18|$INST/03-M18.txt
20_euros2|$SUP/20_euros2.txt
20_rosids|$SUP/20_rosids.txt
20_B-HA|$SUP/20_B-HA-573-585-BMGE.txt
21_nucleic|$SUP/21_nucleic_M2839_470x829_2006.txt
25_proteic|$SUP/25_proteic_M2624_139x348_2006-BMGE.txt"
for seed in $SEEDS; do
  echo "$LIST" | grep . | while IFS='|' read -r label file; do
    f=$OUT/$label/seed$seed.json
    [ -f "$f" ] && { echo "skip (done): $f"; continue; }
    if [ "$seed" != 1 ] && ! grep -qs '"certified": true' "$OUT/$label/seed1.json"; then
      echo "skip $label seed $seed: seed 1 did not certify"; continue
    fi
    echo "$(date '+%F %T') start $label seed $seed"
    mkdir -p "$(dirname "$f")"
    BME_THREADS=1 BME_WORKERS=$TIMED_THREADS BME_SEED=$seed "$PY" "$REPO/experiments/compact_solver.py" \
        "$file" "$TIMED_CAP" "$f" < /dev/null > /dev/null 2> "${f%.json}.err" \
        || echo "run failed: $label seed $seed (see ${f%.json}.err)"
  done
done
