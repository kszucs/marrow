#!/bin/sh
# Swap in the ASSERT=all libmarrow, run a bisect2 case, swap back.
cd "$(dirname "$0")/.." || exit 1
cp python/marrow/libmarrow.so repros/build/libmarrow_release.so.bak 2>/dev/null
cp repros/build/libmarrow_assert.so python/marrow/libmarrow.so
# shellcheck disable=SC2086
pixi run -e dev python repros/bisect2.py $1
rc=$?
cp repros/build/libmarrow_release.so.bak python/marrow/libmarrow.so
echo "rc=$rc"
