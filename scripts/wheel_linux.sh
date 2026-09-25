#!/usr/bin/env bash
# The Linux wheel leg on a laptop: compose.yaml's `wheel` service runs
# scripts/wheel_ci.sh -- what .github/workflows/wheels.yml runs -- for the Docker
# host's own architecture. That is aarch64 on Apple Silicon, where x86_64 Mojo
# cannot run under emulation; CI builds x86_64.
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

docker compose -f "$ROOT/compose.yaml" run --rm wheel
