#!/usr/bin/env bash
# Clone and build Apache OpenDAL's C binding -- the `libopendal_c` that
# `marrow/io/opendal.mojo` dlopens.
#
# One script rather than an inline command because two places need it: the
# `opendal` pixi environment, for running the storage tests, and the wheel
# build, which embeds the result. The tag and the feature list must have a
# single definition -- the C ABI is not stable in the way a vendored mirror
# needs, so a second copy that drifts is a silently corrupt mirror rather than
# a build failure. The `comptime assert size_of[...]` guards in `opendal.mojo`
# are the second line of defence, not the first.
#
# Prints the built library's path on stdout, so a caller can do:
#     export MARROW_OPENDAL_LIBRARY="$(scripts/build_opendal_c.sh)"
#
# `build_opendal_c.sh clone` stops after the checkout. CI needs that seam: the
# cargo cache key is the cloned SHA, so it has to be computable between the
# clone and the build.
set -euo pipefail

# v0.59.0 is the latest *released* tag. The working tree calls itself 0.59.1,
# but that is unreleased and every struct this binding mirrors is byte-identical
# between the two.
TAG="v0.59.0"
# Services are cargo features: a stock build has only `memory`, not even `fs`.
FEATURES="opendal/services-fs,opendal/services-s3,opendal/services-http,opendal/services-hf"

ROOT="${MARROW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="$ROOT/vendor/opendal"

if [ ! -d "$SRC" ]; then
    git clone --depth 1 --branch "$TAG" \
        https://github.com/apache/opendal.git "$SRC" >&2
fi

if [ "${1:-}" = "clone" ]; then
    exit 0
fi

# If aws-lc-rs fails to build (it wants cmake, and sometimes nasm), swap the
# transport:  --no-default-features --features opendal/http-transport-reqwest-native-tls,...
cargo build --release \
    --manifest-path "$SRC/bindings/c/Cargo.toml" \
    --features "$FEATURES" >&2

OUT="$SRC/bindings/c/target/release/libopendal_c"
if [ -f "$OUT.dylib" ]; then
    echo "$OUT.dylib"
elif [ -f "$OUT.so" ]; then
    echo "$OUT.so"
else
    echo "build_opendal_c: cargo succeeded but no libopendal_c found at $OUT.*" >&2
    exit 1
fi
