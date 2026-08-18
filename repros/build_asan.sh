#!/bin/sh
# usage: build_asan.sh <src.mojo> <out>
cd "$(dirname "$0")/.." || exit 1
LIB="$PWD/.pixi/envs/asan/lib/libclang_rt.asan_osx_dynamic.dylib"
if [ ! -f "$LIB" ]; then
  echo "no asan runtime at $LIB" >&2
  exit 1
fi
exec pixi run -e asan mojo build --sanitize address --shared-libasan \
  -Xlinker -rpath -Xlinker "$PWD/.pixi/envs/asan/lib" \
  -Xlinker "$LIB" \
  -I . "$1" -o "$2"
