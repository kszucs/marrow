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
#
#     REPEATS=3 SELECT='filter or groupby' benchmarks/history.sh HEAD~10 HEAD
#
# **Repeats are interleaved across refs, not nested per ref, and that is load
# bearing.** The first version of this script ran all repeats for ref A, then
# all for ref B, and reported a confident 20% regression on `groupby_sum_1m` at
# the last ref — both of its runs above every run of every earlier ref. It was
# not real: re-measuring that same commit in a shorter sweep put it back in line
# with the rest. The machine simply got slower over half an hour of continuous
# compilation, so whichever ref was measured *last* looked worst. Taking the
# minimum across repeats does not save you, because all of a ref's repeats sit
# in the same contaminated window.
#
# Interleaving spreads that drift evenly over every ref instead of dumping it on
# the final one. It costs nothing after the first pass: the Mojo artifact cache
# is content addressed, so revisiting a commit recompiles nothing.
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

for run in $(seq 1 "$REPEATS"); do
    for ref in "$@"; do
        sha=$(git rev-parse --short "$ref") || continue
        date=$(git show -s --format=%ad --date=format:'%m-%d_%H:%M' "$sha")
        subject=$(git show -s --format=%s "$sha" | cut -c1-48 | tr '\t' ' ')

        git checkout -q "$sha" 2>/dev/null || { echo "SKIP $sha (checkout)" >&2; continue; }

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
