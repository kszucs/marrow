#!/usr/bin/env bash
# The native libraries a wheel bundles beside `libmarrow` -- the Parquet page
# codecs and `libopendal_c` -- into PREFIX/lib, for cibuildwheel's `before-all`.
# The wheel build finds them through `MARROW_CODEC_LIB_DIR` (see
# `codec_lib_dir()` in python/marrow/compile.py).
#
# Locally a pixi environment provides the codecs. cibuildwheel builds from a
# bare Python -- inside the manylinux container on Linux -- so this installs the
# same conda-forge packages with micromamba, at the versions whose licence texts
# are committed under `licenses/` (`devkit/tests/test_licenses.py` compares
# them), and builds OpenDAL with a conda-forge Rust, as the `opendal` pixi
# environment does. Nothing from the Rust environment ships.
#
#     bash scripts/wheel_deps.sh /tmp/marrow-deps
set -euo pipefail

PREFIX="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/tmp/micromamba}"

case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) PLATFORM=osx-arm64 ;;
    Linux-x86_64) PLATFORM=linux-64 ;;
    Linux-aarch64) PLATFORM=linux-aarch64 ;;
    *)
        echo "wheel_deps: no wheel is built for $(uname -sm)" >&2
        exit 1
        ;;
esac

MICROMAMBA="$MAMBA_ROOT_PREFIX/bin/micromamba"
if [ ! -x "$MICROMAMBA" ]; then
    # The manylinux image may lack bzip2, which GNU tar needs for `-j`.
    if [ "$(uname -s)" = Linux ] && ! command -v bzip2 >/dev/null; then
        dnf install -y bzip2 >&2
    fi
    mkdir -p "$MAMBA_ROOT_PREFIX"
    curl -fsSL "https://micro.mamba.pm/api/micromamba/$PLATFORM/latest" |
        tar -xj -C "$MAMBA_ROOT_PREFIX" bin/micromamba
fi

"$MICROMAMBA" create --yes --quiet --prefix "$PREFIX" --channel conda-forge \
    zstd=1.5.7 snappy=1.2.2 lz4-c=1.10.0 brotli=1.2.0 zlib=1.3.2

# The pip-installed `mojo` needs GLIBCXX_3.4.30, which manylinux_2_34's own
# libstdc++ (GCC 11) does not have, although Modular tags the wheel
# manylinux_2_34. conda-forge's, which snappy pulls in, does. PREFIX/cxx holds
# only that runtime, so `LD_LIBRARY_PATH` can hand it to the build without also
# swapping the codecs in under every other process. Nothing of it ships, and the
# wheel is tested outside this container (see python/pyproject.toml).
if [ "$(uname -s)" = Linux ]; then
    mkdir -p "$PREFIX/cxx"
    ln -sf "$PREFIX/lib/libstdc++.so.6" "$PREFIX/lib/libgcc_s.so.1" "$PREFIX/cxx/"
fi

RUST="$PREFIX-rust"
"$MICROMAMBA" create --yes --quiet --prefix "$RUST" --channel conda-forge \
    "rust>=1.91" cmake

LIB="$(PATH="$RUST/bin:$PATH" bash "$ROOT/scripts/build_opendal_c.sh")"
cp "$LIB" "$PREFIX/lib/"
echo "wheel_deps: codecs and $(basename "$LIB") are in $PREFIX/lib" >&2
