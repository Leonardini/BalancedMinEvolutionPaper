#!/bin/bash
# One run of Catanzaro's solver in its own directory.
#   catanzaro_one.sh OUTDIR THREADS SEED CAP INSTANCE [solver flags...]
# OUTDIR gets gurobi.env (Gurobi reads it from the working directory), the solver's
# report (report.txt), the path-length matrix (tau.txt), stdout+stderr (log.txt) and
# /usr/bin/time's resource summary, including peak RSS (time.txt).
set -euo pipefail
out=$1; threads=$2; seed=$3; cap=$4; inst=$5; shift 5
# The solver runs inside $out, so a relative instance path would silently point nowhere.
[ -f "$inst" ] || { echo "no instance file: $inst" >&2; exit 1; }
inst=$(cd "$(dirname "$inst")" && pwd)/$(basename "$inst")
source "$(dirname "$0")/config.sh"
[ -x "$CATANZARO_BIN" ] || { echo "build the solver first: external/catanzaro/build.sh" >&2; exit 1; }
command -v fastme > /dev/null || { echo "fastme is not on PATH; --heuls would silently do nothing" >&2; exit 1; }
if [ -f "$out/report.txt" ]; then echo "skip (done): $out"; exit 0; fi
mkdir -p "$out"
printf "Threads %s\nSeed %s\n" "$threads" "$seed" > "$out/gurobi.env"
# Extra Gurobi parameters, one "Name value" per line, e.g. for the strict-tolerance runs.
if [ -n "${EXTRA_GUROBI_ENV:-}" ]; then printf "%s\n" "$EXTRA_GUROBI_ENV" >> "$out/gurobi.env"; fi
printf "%s\n" "$CATANZARO_BIN -i $inst $* -t $cap" > "$out/command.txt"
echo "$(date '+%F %T') start $out"
# FastME (called by --heuls) otherwise starts one thread per core.
export OMP_NUM_THREADS=1
( cd "$out" && /usr/bin/time -l "$CATANZARO_BIN" -i "$inst" "$@" -t "$cap" \
      -R report.part -o tau.txt < /dev/null > log.txt 2> time.txt ) || echo "solver exit $? for $out" >&2
# The report is written only at the end, so a report.txt marks a finished run.
[ -f "$out/report.part" ] && mv "$out/report.part" "$out/report.txt"
echo "$(date '+%F %T') end   $out"
