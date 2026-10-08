#!/bin/bash
# Build Catanzaro et al.'s solver with our patch (catanzaro_repro.patch):
#   - links Gurobi 12 (-lgurobi120) instead of the makefile's Gurobi 10, and finds boost
#     under Homebrew (override BOOST_FLAGS otherwise);
#   - adds --export FILE, which writes the model before any callback cut and exits
#     (used for the M43 CPLEX run).
# Expects the original source in external/catanzaro/upstream (see README.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${GUROBI_HOME:=/Library/gurobi1201/macos_universal2}"
[ -d "$here/upstream/src" ] || { echo "put the solver's source in $here/upstream" >&2; exit 1; }
rm -rf "$here/build"
cp -R "$here/upstream" "$here/build"
( cd "$here/build" && patch -p1 < "$here/catanzaro_repro.patch" && mkdir -p bin \
    && make GUROBI_HOME="$GUROBI_HOME" solver_bmep )
otool -L "$here/build/bin/solver_bmep" | grep -i gurobi
