#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
N=${1:-1}
FILES=$(/bin/ls -t "$HOME"/Library/Logs/DiagnosticReports/python3.14-*.ips | head -"$N")
# shellcheck disable=SC2086
exec python3 repros/showips.py $FILES
