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
scratch=$(mktemp -d)
cleanup() {
    git checkout -q "$start_ref" 2>/dev/null
    rm -rf "$scratch"
}
trap cleanup EXIT

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "history.sh: working tree is dirty — commit or stash first" >&2
    exit 1
fi

printf 'commit\tdate\tsubject\tbenchmark\trun\tmedian_ns\tmin_ns\n'

for run in $(seq 1 "$REPEATS"); do
    for ref in "$@"; do
        sha=$(git rev-parse --short "$ref") || continue
        date=$(git show -s --format=%ad --date=format:'%m-%d_%H:%M' "$sha")
        subject=$(git show -s --format=%s "$sha" | cut -c1-48 | tr '\t' ' ')

        git checkout -q "$sha" 2>/dev/null || { echo "SKIP $sha (checkout)" >&2; continue; }

        # The numbers come from `--save-benchmarks`, which writes every
        # statistic as JSON, and not from the terminal table. That table is
        # scraped by column position, and its columns are configurable
        # (`--benchmark-columns`) and its unit is per-row (ns/us/ms) — so a
        # scraper reports the wrong statistic, in the wrong unit, without ever
        # looking wrong. `--benchmark-history` points the rolling series at the
        # scratch directory: without it every ref in the sweep would be appended
        # to `benchmarks/data.json`, which is the dashboard's series and is
        # meant to hold one entry per commit of the branch, not a re-measurement
        # of ten old ones.
        #
        # $FILES is intentionally unquoted: it may name several bench files, and
        # each must reach pytest as its own argument. Quoting it passes one
        # argument containing spaces, which matches no path — pytest then
        # collects nothing and this ref contributes no rows. Point it at
        # `bench_*.mojo` files rather than a directory, too: a directory drags in
        # test files, and one that fails to build takes the whole run down with
        # the same symptom.
        rm -f "$scratch/latest.json"
        # shellcheck disable=SC2086
        pixi run -e dev pytest --benchmark $FILES -k "$SELECT" \
            --save-benchmarks "$scratch" \
            --benchmark-history "$scratch/data.json" >"$scratch/run.log" 2>&1

        if [ ! -f "$scratch/latest.json" ]; then
            # Nothing was measured: a build failure, a selection that matched
            # nothing, or a ref predating `--save-benchmarks`. Say so — the
            # previous version of this script emitted zero rows in silence, and
            # a missing ref reads as a gap in the curve rather than as an error.
            echo "SKIP $sha (no benchmarks ran; see below)" >&2
            tail -5 "$scratch/run.log" >&2
            continue
        fi

        # `python3`, not the checked-out tree's tooling: this reads one JSON
        # file and must behave the same for every ref in the sweep, including
        # ones whose `devkit/` differs from the working branch's.
        python3 -c '
import json, sys

path, commit, date, subject, run = sys.argv[1:]
for result in json.load(open(path))["results"]:
    if "median_ns" not in result:
        continue  # no stats: a benchmark that never completed a round
    print(
        "\t".join(
            [
                commit, date, subject, result["name"], run,
                "%.3f" % result["median_ns"],
                "%.3f" % result["min_ns"],
            ]
        )
    )
' "$scratch/latest.json" "$sha" "$date" "$subject" "$run"
    done
done
