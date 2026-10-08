#!/bin/bash
# Section 8.7, Table D9: root gap of the distance-indexed model as cut families are switched
# on, plus the circular-order leave-one-out.
#   distance_cut_families.sh timed   configurations with separated cuts: full MIP runs, whose
#                            converged root bound ("Last Root BD") and node count we
#                            report. 10 threads, 3600 s cap; seeds 2-3 only where seed 1
#                            certified (as for Table D8).
#   distance_cut_families.sh counts  configurations without separated cuts: the root LP alone
#                            (--relax), one thread. A bound, not a timing.
# Columns: (a) base; (b) + strong triangles on all triples; (gki) + generalized Kraft;
# (c) + 2-K-split and circular order; (d) + four-point. "full" is the Section 7.1
# configuration (Table D8) and "cooff" the same with circular-order cuts switched off.
set -euo pipefail
source "$(dirname "$0")/../run/config.sh"
mode=$1
OUT=$RESULTS/distance_cut_families
HEU="-O B --heuk 4 --heuls"
GKI="--gkiR 0.05 --gkiN 0.01"
SPLIT2K="--2KR 0.001"
CO="--co G --coR 0.05 --coN 0.005"
FOURPT="--buneman O --wbR 0.125"
INSTANCES="M17|$INST/02-M17.txt|
M18|$INST/03-M18.txt|
20_euros2|$SUP/20_euros2.txt|--labeled
20_rosids|$SUP/20_rosids.txt|--labeled"
if [ "$mode" = counts ]; then
  CONFIGS="a|--str N --relax
b|--str A --relax"
  threads=$COUNT_THREADS; seeds=1
elif [ "$mode" = timed ]; then
  CONFIGS="gki|--str A $GKI
c|--str A $GKI $SPLIT2K $CO
d|--str A $GKI $SPLIT2K $CO $FOURPT
full|--str O --buneman O $GKI --wbR 0.125 $SPLIT2K $CO
cooff|--str O --buneman O $GKI --wbR 0.125 $SPLIT2K --co N --coR -1 --coN -1"
  threads=$TIMED_THREADS; seeds=$SEEDS
else
  echo "usage: $0 timed|counts" >&2; exit 1
fi
for seed in $seeds; do
  echo "$CONFIGS" | while IFS='|' read -r cfg flags; do
    echo "$INSTANCES" | while IFS='|' read -r label file lab; do
      dir=$OUT/$cfg/$label
      if [ "$seed" != 1 ] && ! certified "$dir/seed1"; then
        echo "skip $cfg/$label seed $seed: seed 1 did not certify"; continue
      fi
      # shellcheck disable=SC2086
      "$REPO/run/catanzaro_one.sh" "$dir/seed$seed" "$threads" "$seed" "$TIMED_CAP" "$file" \
          $HEU $flags $lab
    done
  done
done
