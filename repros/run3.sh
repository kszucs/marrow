#!/bin/sh
# usage: run3.sh "<nrows> <page_size> <morsel> <mode>" ...
cd "$(dirname "$0")/.." || exit 1
for spec in "$@"; do
  # shellcheck disable=SC2086
  out=$(pixi run -e dev python repros/synth.py $spec 2>&1)
  rc=$?
  echo "=== [$spec] rc=$rc"
  echo "$out" | tail -4
done
