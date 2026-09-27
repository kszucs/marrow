#!/usr/bin/env bash
# The wheel leg, one definition: cibuildwheel with python/pyproject.toml's
# configuration, `devkit wheel check`, and the installed-wheel tests.
# .github/workflows/wheels.yml runs it on each runner; compose.yaml's `wheel`
# service runs it on a laptop (`pixi run -e docker wheel_linux`).
#
# The tests run here, on the machine that ran cibuildwheel, not in cibuildwheel:
# a Linux wheel is manylinux_2_35, which pip in the manylinux_2_34 build
# container rightly refuses (see [tool.cibuildwheel.linux] in
# python/pyproject.toml). Every wheel bundles OpenDAL, so its test must not
# skip for want of it.
#
#     bash scripts/wheel_ci.sh wheelhouse
set -euo pipefail

OUT="$1"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# devkit's CLI imports click, and rich and psutil through its progress display.
python -m pip install -q cibuildwheel==3.2.1 click rich psutil
python -m cibuildwheel python --output-dir "$OUT"
python -m devkit wheel check --require opendal "$OUT"/*.whl
MARROW_REQUIRE_OPENDAL=1 bash scripts/wheel_test.sh "$OUT"/*.whl
