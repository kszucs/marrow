#!/usr/bin/env bash
# Benchmark the same kernels across a series of commits, to see the performance
# curve rather than a single before/after.
#
#     benchmarks/history.sh <ref> [<ref> ...]
#
# For each ref it checks out the whole tree (so each commit is benchmarked with
# *its own* benchmark sources), runs the selection, and appends one row per
# benchmark to a TSV on stdout. The working branch is restored at the end even
# if a ref fails to build.
#
# Why this exists: on this machine the same binary varies ~10-18% run to run, so
# a single before/after pair cannot distinguish a real regression from noise.
# Passing REPEATS=3 and reading the *minimum* across repeats is far more robust —
# interference only ever makes a run slower, never faster.
#
#     REPEATS=3 SELECT='filter or groupby' benchmarks/history.sh HEAD~10 HEAD
#
set -uo pipefail

SELECT="${SELECT:-take or filter or groupby or sort}"
FILES="${FILES:-marrow/kernels/tests}"
REPEATS="${REPEATS:-2}"

start_ref=$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)
restore() { git checkout -q "$start_ref" 2>/dev/null; }
trap restore EXIT

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "history.sh: working tree is dirty — commit or stash first" >&2
    exit 1
fi

printf 'commit\tdate\tsubject\tbenchmark\trun\tmedian\tmin\n'

for ref in "$@"; do
    sha=$(git rev-parse --short "$ref") || continue
    date=$(git show -s --format=%ad --date=format:'%m-%d_%H:%M' "$sha")
    subject=$(git show -s --format=%s "$sha" | cut -c1-48 | tr '\t' ' ')

    git checkout -q "$sha" 2>/dev/null || { echo "SKIP $sha (checkout)" >&2; continue; }

    for run in $(seq 1 "$REPEATS"); do
        # pytest-benchmark's table columns are: Name Min Max Mean StdDev Median ...
        # Units vary per row (ns/us/ms), so carry the unit from the group header.
        pixi run -e dev pytest --benchmark "$FILES" -k "$SELECT" 2>/dev/null \
        | awk -v c="$sha" -v d="$date" -v s="$subject" -v r="$run" '
            /^Name \(time in/ { unit = $4; sub(/\)$/, "", unit); next }
            /^bench_/ {
                mult = (unit == "ns") ? 1e-3 : (unit == "ms") ? 1e3 : 1
                printf "%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\n", c, d, s, $1, r, $6*mult, $2*mult
            }'
    done
done
