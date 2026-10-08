#!/bin/bash
# Download the solver of Catanzaro, Pesenti, Sapucaia and Wolsey (Math. Program. 2026)
# from the authors' site, at a fixed commit, and unpack it into external/catanzaro/upstream.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
URL=https://github.com/DanieleCatanzaro/danielecatanzaro.github.io/raw/72311e627a830c5faafe30bcf158bf2866a12ff5/SitoWebMedia/bmep_ilp_model-main.zip
SHA256=d90be5236bf80f5dcc69d34f3cd2d51f4669d6e5e28a3499c21f8dd3298ed8b7
tmp=$(mktemp -d)
curl -sSL -o "$tmp/src.zip" "$URL"
echo "$SHA256  $tmp/src.zip" | shasum -a 256 -c -
unzip -q "$tmp/src.zip" -d "$tmp"
rm -rf "$here/upstream"
mv "$tmp/bmep_ilp_model-main" "$here/upstream"
rm -rf "$tmp"
