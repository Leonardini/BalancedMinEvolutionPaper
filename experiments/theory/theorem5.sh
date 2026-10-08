#!/bin/bash
# One run of the enumerator behind Theorem 5 (Appendix C), followed by its check.
#   theorem5.sh NAME N [backtrack options]
# Builds experiments/theory/build/backtrack if it is missing or older than its source,
# runs it, and writes to $RESULTS/theorem5/:
#   NAME.out   the enumerator's stdout (summary line), moved into place on completion
#   NAME.log   its timestamped progress (also on stderr)
#   NAME.dump  every survivor that violates (10)
#   NAME.json  the counts and the comparison with the paper (theorem5_check.py)
set -euo pipefail
source "$(dirname "$0")/../../run/config.sh"
name=$1; n=$2; shift 2
SRC=$REPO/experiments/theory/backtrack.c
BIN=$REPO/experiments/theory/build/backtrack
OUT=$RESULTS/theorem5
mkdir -p "$OUT" "$(dirname "$BIN")"
if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
  # Build under a private name, then rename: concurrent tasks never see a partial binary.
  cc -O3 -o "$BIN.$$" "$SRC"
  mv -f "$BIN.$$" "$BIN"
fi
"$BIN" "$n" "$@" --dump "$OUT/$name.dump" --progress "${BT_PROGRESS:-60}" \
  > "$OUT/$name.out.part" 2> >(tee "$OUT/$name.log" >&2)
mv -f "$OUT/$name.out.part" "$OUT/$name.out"
"$PY" "$REPO/experiments/theory/theorem5_check.py" "$OUT" "$name" "$n" "$@"
