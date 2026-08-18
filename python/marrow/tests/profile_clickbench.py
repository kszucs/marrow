"""Profile driver: run one ClickBench query in a loop under a profiler.

Not a test — a standalone workload for `pixi run profile clickbench-q1`, which
builds `libmarrow.so` with debug info and records an Instruments trace. The
query set is the same registry `test_clickbench.py` and `bench_clickbench.py`
drive, so what you profile is exactly what is tested and timed.

    pixi run profile clickbench-q1              # trace, opens Instruments
    pixi run profile --sample clickbench-q34    # text sample, no GUI

Selected by `MARROW_PROFILE_QUERY` (a registry name such as `q01`, or the
friendlier `q1`); `MARROW_PROFILE_REPEATS` sets the iteration count. The loop
exists so the profiler collects enough samples inside the query rather than
inside interpreter start-up -- a single 10 ms query is invisible next to the
~200 ms of importing marrow and opening the file.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clickbench as cb  # noqa: E402


def _resolve(name):
    """`q1`, `Q1` and `q01` all name the same query."""
    key = name.lower().removeprefix("clickbench-")
    if key in cb.QUERIES:
        return cb.QUERIES[key]
    if key.startswith("q") and key[1:].isdigit():
        padded = f"q{int(key[1:]):02d}"
        if padded in cb.QUERIES:
            return cb.QUERIES[padded]
    raise SystemExit(
        f"unknown query {name!r}; known: {', '.join(sorted(cb.QUERIES))}"
    )


def main():
    name = os.environ.get("MARROW_PROFILE_QUERY", "q01")
    repeats = int(os.environ.get("MARROW_PROFILE_REPEATS", "20"))

    if not cb.HAVE_DATA:
        raise SystemExit(
            f"dataset not found at {cb.HITS}; set MARROW_CLICKBENCH_HITS"
        )

    q = _resolve(name)
    if q.unsupported:
        raise SystemExit(f"{q.name} is UNSUPPORTED: {q.unsupported}")

    import marrow as ma

    print(f"{q.name}: {q.sql}", flush=True)

    # Opening the plan is not what we are profiling, but `read_parquet` reads
    # the footer, so hoist it out of the loop and time the query alone.
    table = ma.read_parquet(cb.HITS)
    q.marrow(table).collect()  # warm: first call pays one-time set-up

    t0 = time.perf_counter()
    for _ in range(repeats):
        q.marrow(table).collect()
    total = time.perf_counter() - t0
    print(
        f"{repeats} runs in {total * 1000:.1f} ms "
        f"({total / repeats * 1000:.2f} ms/run)",
        flush=True,
    )


if __name__ == "__main__":
    main()
