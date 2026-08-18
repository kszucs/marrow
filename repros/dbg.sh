#!/bin/sh
# usage: dbg.sh <cols> <pred> <morsel> [source]
cd "$(dirname "$0")/.." || exit 1
PY=$(pixi run -e dev python -c "import sys; print(sys.executable)")
exec pixi run -e dev lldb \
  -o run \
  -o "thread backtrace all" \
  -o quit \
  --batch -- "$PY" repros/bisect2.py "$@"
