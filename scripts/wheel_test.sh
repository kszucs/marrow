#!/usr/bin/env bash
# Test a marrow wheel the way a user gets it: installed into a fresh virtual
# environment, with the Python suite run against that install rather than the
# checkout. The one definition of that test, for every platform:
#
#   - `pixi run -e wheel wheel` runs it on the macOS wheel it builds;
#   - cibuildwheel runs it as the macOS `test-command`;
#   - Linux wheels are tested after the build, on the machine that ran
#     cibuildwheel -- wheels.yml's runner, or the compose `wheel` service --
#     because pip in the manylinux_2_34 build container rightly refuses a
#     manylinux_2_35 wheel (see [tool.cibuildwheel.linux] in pyproject.toml).
#
#     bash scripts/wheel_test.sh dist/repaired/marrow-*.whl
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: wheel_test.sh WHEEL (got $# arguments)" >&2
    exit 2
fi
WHEEL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$(mktemp -d)/venv"

"${PYTHON:-python}" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q "$WHEEL" pytest "pyarrow>=23.0.1,<24" "numpy>=2.4.4,<3"

# A wheel that ships OpenDAL must read through it: the `fs://` test skips when
# the library is absent, which is right for a checkout and wrong here.
if "$VENV/bin/python" -c "import sys, zipfile; sys.exit(not any('libopendal_c' in n for n in zipfile.ZipFile(sys.argv[1]).namelist()))" "$WHEEL"; then
    export MARROW_REQUIRE_OPENDAL=1
fi

# Everything below runs from inside the venv, away from the checkout, and the
# import must resolve to the install -- a source tree on the path would make
# every test below pass without touching the wheel.
cd "$VENV"
"$VENV/bin/python" -c "import marrow, sys; assert marrow.__file__.startswith(sys.prefix), marrow.__file__"

# `pythonpath=$ROOT` keeps `devkit` importable (the golden corpus) and drops
# pytest.ini's `python`; `python_files` leaves the benchmarks out;
# `test_build.py` tests the build hook itself, which needs hatchling and a
# checkout, not a wheel.
"$VENV/bin/python" -m pytest "$ROOT/python/marrow/tests" \
    -o pythonpath="$ROOT" -o "python_files=test_*.py" \
    --noconftest -p no:cacheprovider -m "not gpu" \
    --ignore="$ROOT/python/marrow/tests/test_build.py"
