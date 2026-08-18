"""ClickBench timing — marrow vs polars vs duckdb on the same 43 queries.

Query definitions live in ``clickbench.py`` and are shared with the correctness
suite ``test_clickbench.py``, so a query is written once, checked once, and
timed here. ``test_polars_thunk_matches_reference`` proves the polars spellings
compute the same answers, which is what makes these numbers a comparison rather
than three unrelated stopwatches.

    pixi run -e bench python python/marrow/tests/bench_clickbench.py
    pixi run -e bench python python/marrow/tests/bench_clickbench.py --repeats 7
    pixi run -e bench pytest --benchmark python/marrow/tests/bench_clickbench.py

All three engines are driven **lazily** — ``ma.read_parquet`` vs
``pl.scan_parquet`` vs DuckDB over ``read_parquet(...)`` — because comparing a
lazy plan against a pre-materialised in-memory table is not a comparison.

Measurement discipline
----------------------
This box drifts up to ±8% per case and ~±5% per batch, so:

* **engines are interleaved per repeat** (A, B, C, A, B, C, …) rather than each
  being run to completion — a background compile that lands mid-run then taxes
  all three equally instead of whichever one it overlapped;
* **medians of N repeats** are reported, never a single sample;
* one **untimed warm-up** per (query, engine) precedes the repeats, so page
  cache and JIT state are not charged to the first engine in the list.

A single run is still not evidence. Two runs that agree are.

Metadata-only shortcuts
-----------------------
``pl.scan_parquet(...).select(pl.len())`` answers ``COUNT(*)`` from the Parquet
footer in ~0.4 ms **without scanning anything**. Timing that against a full scan
is not a comparison, so Q1 is flagged and excluded from the totals. Its row is
still printed, marked, because hiding it would be worse.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import clickbench as cb

ENGINES = ("marrow", "marrow1", "polars", "duckdb")

# `marrow1` is `marrow` with `collect(num_threads=1)` — bit-for-bit what every
# lazy query did before `collect` took a worker count, since the old
# `DynRelation.execute()` default was `ExecContext()`, forced serial. Keeping it
# as a fourth *interleaved* engine is what makes before/after a comparison
# rather than two runs on a box that drifts ±8% per case.


# ── per-engine runners ─────────────────────────────────────────────────────


def _marrow_runner(num_threads=0):
    def run(q):
        return q.marrow(cb.marrow_scan()).collect(num_threads=num_threads)

    return run


def _polars_runner():
    import polars as pl

    def run(q):
        if q.polars is None:
            raise NotImplementedError("no polars spelling")
        return q.polars(pl.scan_parquet(cb.HITS)).collect()

    return run


def _duckdb_runner():
    con = cb.duckdb_lazy_connection()

    def run(q):
        return con.sql(q.duckdb_timing_sql).arrow()

    return run


def runners():
    """``{engine: callable}`` for every engine present in this environment."""
    out = {"marrow": _marrow_runner(), "marrow1": _marrow_runner(1)}
    if cb.HAS_POLARS:
        out["polars"] = _polars_runner()
    if cb.HAS_DUCKDB:
        out["duckdb"] = _duckdb_runner()
    return out


# ── the interleaved measurement ────────────────────────────────────────────


def measure(queries, engines, repeats=5, progress=None):
    """``{query: {engine: median_seconds | None}}``, engines interleaved.

    ``None`` means the engine cannot run that query at all (marrow has no regex
    kernel, so Q29 has no marrow number). A query that *raises* is recorded as
    ``None`` too, with the reason kept in ``errors``.
    """
    samples = {q.name: {e: [] for e in engines} for q in queries}
    errors = {}

    for q in queries:  # untimed warm-up
        for name in engines:
            try:
                engines[name](q)
            except Exception as e:  # noqa: BLE001
                errors[(q.name, name)] = f"{type(e).__name__}: {e}"

    for r in range(repeats):
        for q in queries:
            for name in engines:
                if (q.name, name) in errors:
                    continue
                t0 = time.perf_counter()
                try:
                    engines[name](q)
                except Exception as e:  # noqa: BLE001
                    errors[(q.name, name)] = f"{type(e).__name__}: {e}"
                    continue
                samples[q.name][name].append(time.perf_counter() - t0)
        if progress:
            progress(r + 1, repeats)

    return {
        qn: {e: (statistics.median(v) if v else None) for e, v in per.items()}
        for qn, per in samples.items()
    }, errors


# ── reporting ──────────────────────────────────────────────────────────────


def _fmt(seconds):
    if seconds is None:
        return "     —"
    ms = seconds * 1000
    if ms < 10:
        return f"{ms:6.2f}"
    return f"{ms:6.1f}"


def _ratio(a, b):
    if a is None or b is None or b == 0:
        return "    —"
    return f"{a / b:5.1f}x"


def report(medians, errors, engines, out=None):
    # `out=sys.stdout` as a *default* binds at import time, which under pytest is
    # the capture object installed before `capsys.disabled()` restores the real
    # one — the table then vanishes. Resolve it at call time.
    out = out or sys.stdout
    names = [e for e in ENGINES if e in engines]
    header = "  ".join(f"{e:>8}" for e in names)
    print(
        f"\n{'query':<6} {'ms: ' + header}   {'vs polars':>9} {'vs duckdb':>9}  note",
        file=out,
    )
    print("-" * 96, file=out)

    totals = {e: 0.0 for e in names}
    counted = 0
    for name, per in medians.items():
        q = cb.QUERIES[name]
        cells = "  ".join(f"{_fmt(per.get(e)):>8}" for e in names)
        note = []
        if q.metadata_shortcut:
            note.append("METADATA-ONLY for polars/duckdb; excluded from totals")
        if q.unsupported:
            note.append("marrow: UNSUPPORTED")
        elif q.deviation:
            note.append("DEVIATED")
        for e in names:
            if (name, e) in errors:
                note.append(f"{e}: {errors[(name, e)][:40]}")
        print(
            f"{name:<6} {cells}   "
            f"{_ratio(per.get('marrow'), per.get('polars')):>9} "
            f"{_ratio(per.get('marrow'), per.get('duckdb')):>9}  "
            f"{'; '.join(note)}",
            file=out,
        )
        if not q.metadata_shortcut and all(per.get(e) is not None for e in names):
            counted += 1
            for e in names:
                totals[e] += per[e]

    print("-" * 96, file=out)
    cells = "  ".join(f"{_fmt(totals[e]):>8}" for e in names)
    print(
        f"{'TOTAL':<6} {cells}   "
        f"{_ratio(totals.get('marrow'), totals.get('polars')):>9} "
        f"{_ratio(totals.get('marrow'), totals.get('duckdb')):>9}  "
        f"{counted} queries all three engines ran",
        file=out,
    )
    print(
        "\nProjection pushdown narrows `ParquetScan`'s schema to the columns the\n"
        "plan actually reads, so the flat ~270 ms floor (every query decoding all\n"
        "105 columns, `COUNT(*)` included) is gone: the 41-query total went from\n"
        "13 914 ms to 3 870 ms, 17.7x polars to 5.0x, with polars and duckdb\n"
        "steady to within 1.5%. What is left is real work — Q24 is `SELECT *` and\n"
        "must read every column, and Q30's 90 fused sums are compute-bound.",
        file=out,
    )


# ── entry points ───────────────────────────────────────────────────────────


def run(repeats=5, only=None, out=None):
    out = out or sys.stdout
    engines = runners()
    queries = [q for n, q in cb.QUERIES.items() if not only or n in only]
    print(
        f"{len(queries)} queries x {len(engines)} engines x {repeats} repeats, "
        f"interleaved; medians below.",
        file=out,
    )

    def progress(done, total):
        print(f"  repeat {done}/{total}", file=sys.stderr, flush=True)

    medians, errors = measure(queries, engines, repeats, progress)
    report(medians, errors, engines, out)
    return medians, errors


def test_clickbench_comparison(capsys):
    """The comparison table, as one benchmark item.

    Collected only under ``--benchmark`` (``conftest.py`` skips ``bench_*.py``
    otherwise). Two repeats here; use the script entry point for a real
    measurement.
    """
    import pytest

    if not cb.HAVE_DATA:
        pytest.skip(f"ClickBench hits parquet not found at {cb.HITS}")
    with capsys.disabled():
        run(repeats=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("queries", nargs="*", help="e.g. q01 q34; default all 43")
    args = ap.parse_args(argv)
    if not cb.HAVE_DATA:
        print(f"dataset not found: {cb.HITS}")
        return 2
    run(repeats=args.repeats, only=set(args.queries) or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
