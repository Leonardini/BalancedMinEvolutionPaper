#!/bin/bash
# Appendix E, Table E4. Is the distance-indexed solver's certificate robust at n = 29 and 30, where its objective
# coefficients span about 2^29? On 29_B-NS1 and 30_B-NS1 (seed 1, timed settings):
#   no_gki, no_2k, no_co, no_wb   the Section 7.1 configuration with one separated cut family
#                                 switched off (the strong triangle and four-point rows stay,
#                                 since without them the model admits non-trees);
#   strict                        the Section 7.1 configuration with Gurobi's numerical
#                                 emphasis at its maximum (NumericFocus 3) and its
#                                 feasibility, optimality and integrality tolerances at their
#                                 minimum, 1e-9.
# experiments/check_optima.py then checks that every certified run returns the same tree.
set -euo pipefail
source "$(dirname "$0")/../run/config.sh"
OUT=$RESULTS/robustness
BASE="-O B --heuk 4 --heuls --str O --buneman O"
GKI="--gkiR 0.05 --gkiN 0.01"; K2="--2KR 0.001"; CO="--co G --coR 0.05 --coN 0.005"; WB="--wbR 0.125"
CONFIGS="no_gki|$BASE $WB $K2 $CO
no_2k|$BASE $GKI $WB $CO
no_co|$BASE $GKI $WB $K2 --co N --coR -1 --coN -1
no_wb|$BASE $GKI $K2 $CO
strict|$CAT_FULL"
STRICT_ENV="NumericFocus 3
FeasibilityTol 1e-9
OptimalityTol 1e-9
IntFeasTol 1e-9"
for inst in 29_B-NS1-284-344-BMGE 30_B-NS1-284-344-BMGE; do
  label=${inst%%-284*}
  echo "$CONFIGS" | while IFS='|' read -r cfg flags; do
    extra=""; [ "$cfg" = strict ] && extra=$STRICT_ENV
    # shellcheck disable=SC2086
    EXTRA_GUROBI_ENV=$extra "$REPO/run/catanzaro_one.sh" "$OUT/$cfg/$label/seed1" "$TIMED_THREADS" 1 \
        "$TIMED_CAP" "$SUP/$inst.txt" $flags --labeled
  done
done
