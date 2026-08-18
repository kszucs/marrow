#!/bin/sh
# Run each bisect case in its own process and report the exit code.
cd "$(dirname "$0")/.." || exit 1
for c in "$@"; do
  out=$(pixi run -e dev python repros/bisect.py "$c" 2>&1)
  rc=$?
  echo "=== $c rc=$rc"
  echo "$out" | tail -4
done
