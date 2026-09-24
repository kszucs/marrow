#!/usr/bin/env bash
# The Linux leg of .github/workflows/wheels.yml, run locally through
# compose.yaml's `wheel` service: cibuildwheel with python/pyproject.toml's
# configuration, then `devkit wheel check --require opendal`, for the Docker
# host's own architecture. That is aarch64 on Apple Silicon, where linux-64
# Mojo cannot run under emulation, and x86_64 -- the architecture CI builds --
# on an x86_64 host.
#
# cibuildwheel copies its working directory into the build container whole, so
# it is handed an export of what CI would check out rather than the working
# tree: `git archive` of the tracked content, uncommitted edits included.
# `git stash create` records that content as a commit object without touching
# the tree, the index or the stash list; a new file must be `git add`ed to come
# along. Nothing ignored comes along either -- above all not a host-built
# python/marrow/libmarrow.so, which python/build.py would package instead of
# building one for Linux.
#
#     pixi run -e docker wheel_linux     # wheels land in .wheel-linux/wheelhouse/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/.wheel-linux"

rm -rf "$OUT"
mkdir -p "$OUT/src" "$OUT/wheelhouse"
REF="$(git -C "$ROOT" stash create)"
git -C "$ROOT" archive "${REF:-HEAD}" | tar -x -C "$OUT/src"

case "$(docker info --format '{{.Architecture}}')" in
    aarch64 | arm64) ARCH=aarch64 ;;
    x86_64 | amd64) ARCH=x86_64 ;;
    *)
        echo "wheel_linux: no wheel is built for $(docker info --format '{{.Architecture}}')" >&2
        exit 1
        ;;
esac

docker compose -f "$ROOT/compose.yaml" run --rm -e CIBW_ARCHS_LINUX="$ARCH" wheel
