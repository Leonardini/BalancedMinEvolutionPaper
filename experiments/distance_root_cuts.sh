#!/bin/bash
# Section 8.7, Table D11. How much of the distance-indexed model's root bound comes from Gurobi's general-purpose
# MIP cuts, which use the integrality of the level indicators? The generalized-Kraft
# configuration of Table D9 (--str A, generalized Kraft), stopped at the root node
# (NodeLimit 1), with Gurobi's cuts on (default) and off (Cuts 0). One thread: the result
# is a bound, not a time.
set -euo pipefail
source "$(dirname "$0")/../run/config.sh"
OUT=$RESULTS/distance_root_cuts
for pair in "M17|$INST/02-M17.txt|" "M18|$INST/03-M18.txt|" "20_euros2|$SUP/20_euros2.txt|--labeled" \
            "20_rosids|$SUP/20_rosids.txt|--labeled"; do
  IFS='|' read -r label file lab <<< "$pair"
  for v in cuts_off cuts_on; do
    extra="NodeLimit 1"
    [ "$v" = cuts_off ] && extra="NodeLimit 1
Cuts 0"
    # shellcheck disable=SC2086
    EXTRA_GUROBI_ENV="$extra" "$REPO/run/catanzaro_one.sh" "$OUT/$label/$v" "$COUNT_THREADS" 1 "$TIMED_CAP" \
        "$file" -O B --heuk 4 --heuls --str A --gkiR 0.05 --gkiN 0.01 $lab > /dev/null
    echo "$label $v: last root bound $("$PY" "$REPO/run/parse_report.py" "$OUT/$label/$v" last_root_bound)"
  done
done
