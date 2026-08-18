#!/bin/sh
# usage: run_cb.sh <so-name-under-repros/build|CURRENT> <query> ...
cd "$(dirname "$0")/.." || exit 1
if [ "$1" != "CURRENT" ]; then
  cp repros/build/"$1" python/marrow/libmarrow.so || exit 1
fi
shift
for q in "$@"; do
  out=$(pixi run -e dev python python/marrow/tests/clickbench_alpha.py --only "$q" 2>&1)
  rc=$?
  echo "=== $q rc=$rc"
  echo "$out" | tail -2
done
