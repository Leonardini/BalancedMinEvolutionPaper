#!/bin/bash
# Section 8.6, Table D8 and the distance-indexed column of Table D2: the distance-indexed
# solver of Catanzaro et al., configuration of Section 7.1, on every instance.
# 20_rosids is not run here: Table D8 reads its runs from distance_cut_families.sh,
# whose configuration full is this configuration.
# TIMED: 10 threads, 3600 s cap, one instance at a time. Seed 1 runs everywhere; seeds 2
# and 3 run only where seed 1 certified, because only those runs report a time and node
# count that a median and range describe (an uncapped run reports a gap at the cap).
set -euo pipefail
source "$(dirname "$0")/../run/config.sh"
OUT=$RESULTS/distance_solver
# label | instance file | extra reader flag
LIST="Primates12|$INST/01-Primates12.txt|
M17|$INST/02-M17.txt|
M18|$INST/03-M18.txt|
20_euros2|$SUP/20_euros2.txt|--labeled
20_B-HA|$SUP/20_B-HA-573-585-BMGE.txt|--labeled
21_nucleic|$SUP/21_nucleic_M2839_470x829_2006.txt|--labeled
22_euros2|$SUP/22_euros2.txt|--labeled
23_euros2|$SUP/23_euros2.txt|--labeled
24_proteic|$SUP/24_proteic_M2577_40x12260_2005.txt|--labeled
25_proteic|$SUP/25_proteic_M2624_139x348_2006-BMGE.txt|--labeled
26_proteic|$SUP/26_proteic_M2624_139x348_2006-BMGE.txt|--labeled
27_nucleic|$SUP/27_nucleic_M2573_346x897_2006.txt|--labeled
28_nucleic|$SUP/28_nucleic_M2573_346x897_2006.txt|--labeled
29_B-NS1|$SUP/29_B-NS1-284-344-BMGE.txt|--labeled
30_B-NS1|$SUP/30_B-NS1-284-344-BMGE.txt|--labeled
M43|$INST/05-M43.txt|"
for seed in $SEEDS; do
  echo "$LIST" | while IFS='|' read -r label file lab; do
    if [ "$seed" != 1 ] && ! certified "$OUT/$label/seed1"; then
      echo "skip $label seed $seed: seed 1 did not certify"; continue
    fi
    # shellcheck disable=SC2086
    "$REPO/run/catanzaro_one.sh" "$OUT/$label/seed$seed" "$TIMED_THREADS" "$seed" "$TIMED_CAP" \
        "$file" $CAT_FULL $lab
  done
done
