#!/bin/sh
# usage: run_with.sh <so-name-under-repros/build> "<bisect2 args>"
cd "$(dirname "$0")/.." || exit 1
cp repros/build/"$1" python/marrow/libmarrow.so || exit 1
shift
for spec in "$@"; do
  # shellcheck disable=SC2086
  out=$(pixi run -e dev python repros/bisect2.py $spec 2>&1)
  rc=$?
  echo "=== [$spec] rc=$rc"
  echo "$out" | tail -3
done
cp repros/build/libmarrow_release.so.bak python/marrow/libmarrow.so
