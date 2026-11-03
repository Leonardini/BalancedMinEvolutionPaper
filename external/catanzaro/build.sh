#!/bin/bash
# Build Catanzaro et al.'s solver with our patch (catanzaro_repro.patch):
#   - links Gurobi 12 (-lgurobi120) instead of the makefile's Gurobi 10, and finds boost
#     under Homebrew (override BOOST_FLAGS otherwise);
#   - adds --export FILE, which writes the model before any callback cut and exits
#     (used for the M43 CPLEX run).
#   build.sh --fix-nogood also applies nogood_index_fix.patch and builds into build_nogoodfix/:
#     the lazy-mode no-good cut (add_projected_buneman_cut) indexes tau by the violating
#     quartet's leaves instead of by the local positions 0..3, and rounds tau to the nearest
#     level. Used to rerun the lazy-mode configurations gki and c of distance_cut_families.sh.
# Expects the original source in external/catanzaro/upstream (see README.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
fix=; dest=build
if [ "${1:-}" = --fix-nogood ]; then fix=1; dest=build_nogoodfix; fi
: "${GUROBI_HOME:=/Library/gurobi1201/macos_universal2}"
[ -d "$here/upstream/src" ] || { echo "put the solver's source in $here/upstream" >&2; exit 1; }
rm -rf "$here/$dest"
cp -R "$here/upstream" "$here/$dest"
( cd "$here/$dest" && patch -p1 < "$here/catanzaro_repro.patch" \
    && { [ -z "$fix" ] || patch -p1 < "$here/nogood_index_fix.patch"; } && mkdir -p bin \
    && make GUROBI_HOME="$GUROBI_HOME" solver_bmep )
otool -L "$here/$dest/bin/solver_bmep" | grep -i gurobi
