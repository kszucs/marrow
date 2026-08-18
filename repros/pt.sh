#!/bin/sh
# usage: pt.sh <logfile> <pytest args...>
cd "$(dirname "$0")/.." || exit 1
LOG="$1"
shift
pixi run -e dev pytest -v "$@" > "$LOG" 2>&1
echo "rc=$?"
tail -30 "$LOG"
