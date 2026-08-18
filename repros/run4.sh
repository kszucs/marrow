#!/bin/sh
# usage: run4.sh "<nrows> <cluster> <morsel> <source>" ...
cd "$(dirname "$0")/.." || exit 1
for spec in "$@"; do
  # shellcheck disable=SC2086
  out=$(pixi run -e dev python repros/synth2.py $spec 2>&1)
  rc=$?
  echo "=== [$spec] rc=$rc"
  echo "$out" | tail -3
done
