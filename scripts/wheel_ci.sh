#!/usr/bin/env bash
# The wheel leg, one definition: cibuildwheel with python/pyproject.toml's
# configuration, `devkit wheel check`, and for Linux the installed-wheel tests.
# .github/workflows/wheels.yml runs it on each runner; compose.yaml's `wheel`
# service runs it on a laptop (`pixi run -e docker wheel_linux`).
#
# cibuildwheel tests a macOS wheel itself. A Linux wheel is manylinux_2_35,
# which pip in the manylinux_2_34 build container rightly refuses, so it is
# tested here, on the machine that ran cibuildwheel -- see
# [tool.cibuildwheel.linux] in python/pyproject.toml.
#
#     bash scripts/wheel_ci.sh wheelhouse
set -euo pipefail

OUT="$1"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# devkit's CLI imports click, and rich and psutil through its progress display.
python -m pip install -q cibuildwheel==3.2.1 click rich psutil
python -m cibuildwheel python --output-dir "$OUT"
python -m devkit wheel check --require opendal "$OUT"/*.whl
if [ "$(uname -s)" = Linux ]; then
    bash scripts/wheel_test.sh "$OUT"/*.whl
fi
