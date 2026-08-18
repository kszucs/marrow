#!/bin/sh
# usage: run2.sh "<cols> <pred> <morsel>" ...
cd "$(dirname "$0")/.." || exit 1
for spec in "$@"; do
  # shellcheck disable=SC2086
  out=$(pixi run -e dev python repros/bisect2.py $spec 2>&1)
  rc=$?
  echo "=== [$spec] rc=$rc"
  echo "$out" | tail -3
done
