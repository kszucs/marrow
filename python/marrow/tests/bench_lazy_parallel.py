"""Thread scaling of the **lazy** query path — `LazyTable.collect(num_threads=)`.

The eager surface has taken ``num_threads=`` since it was written
(``RecordBatch.group_by(..., num_threads=0)``); the lazy path had no spelling at
all, so `DynRelation.execute()` fell through to `ExecContext()` — **serial** —
on every query. These cases exist to show what the thread count actually buys,
per operator, rather than as one end-to-end total that hides which operator did
or did not scale.

    pixi run -e bench pytest --benchmark python/marrow/tests/bench_lazy_parallel.py

Reading the numbers
-------------------
* ``drift_control`` touches **none** of the changed code (an eager numpy sum on
  the same data). Its spread across a batch is this box's noise floor — up to
  ±8% per case. Subtract it before attributing a delta to a thread count.
* pytest-benchmark scales each row's unit independently. Read the unit column.
* Before the ``num_threads=`` parameter existed the four thread counts were the
  *same* measurement four times; that is the baseline this file was written to
  record.
"""

import inspect

import numpy as np
import pyarrow as pa
import pytest

import marrow as ma

# 1M rows: comfortably past `_PARALLEL_MIN_ROWS` (60k) and
# `_PARALLEL_ALWAYS_ROWS` (200k), so the group-by strategy choice is live.
N = 1_000_000

THREADS = [1, 2, 4, 8]

# `collect(num_threads=)` is what this file measures; probe once rather than
# swallowing a TypeError per call, which would also hide a real signature bug.
_HAS_THREADS = "num_threads" in inspect.signature(ma.LazyTable.collect).parameters


def _collect(lazy, num_threads):
    if _HAS_THREADS:
        return lazy.collect(num_threads=num_threads)
    return lazy.collect()


# ---------------------------------------------------------------------------
# Data — built once per session, shared by every thread count
# ---------------------------------------------------------------------------


def _batch(num_groups, seed=0):
    """Ingested through the Arrow C protocol (zero-copy), as `bench_groupby` does."""
    rng = np.random.default_rng(seed)
    return ma.record_batch(
        pa.record_batch(
            {
                "k": rng.integers(0, num_groups, N).astype(np.int32),
                "v": np.arange(N, dtype=np.float64),
            }
        )
    )


@pytest.fixture(scope="session")
def low_card():
    """100 groups — the `_thread_local` partial-aggregation strategy's shape."""
    return _batch(100)


@pytest.fixture(scope="session")
def high_card():
    """500k groups — the radix-partition strategy's shape."""
    return _batch(500_000)


@pytest.fixture(scope="session")
def join_sides():
    rng = np.random.default_rng(1)
    left = ma.record_batch(
        pa.record_batch(
            {
                "k": np.arange(N, dtype=np.int64),
                "v": np.arange(N, dtype=np.float64),
            }
        )
    )
    right = ma.record_batch(
        pa.record_batch(
            {
                "k": rng.integers(0, N, N).astype(np.int64),
                "w": np.arange(N, dtype=np.float64),
            }
        )
    )
    return left, right


@pytest.fixture(params=THREADS, ids=[f"t{t}" for t in THREADS])
def threads(request):
    return request.param


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="lazy_groupby_low_card")
def test_groupby_low_cardinality(benchmark, low_card, threads):
    """1M rows into 100 groups — few, large groups."""
    benchmark.extra_info.update(threads=threads, rows=N, groups=100)
    t = ma.memtable(low_card).aggregate(by=["k"], total=("sum", "v"))
    benchmark(_collect, t, threads)


@pytest.mark.benchmark(group="lazy_groupby_high_card")
def test_groupby_high_cardinality(benchmark, high_card, threads):
    """1M rows into ~500k groups — many, tiny groups."""
    benchmark.extra_info.update(threads=threads, rows=N, groups=500_000)
    t = ma.memtable(high_card).aggregate(by=["k"], total=("sum", "v"))
    benchmark(_collect, t, threads)


@pytest.mark.benchmark(group="lazy_filter")
def test_filter(benchmark, high_card, threads):
    """A selective predicate over 1M rows — `FilterProcessor` calls `filter`
    once per column per morsel, so this is the per-morsel scaling case."""
    benchmark.extra_info.update(threads=threads, rows=N)
    t = ma.memtable(high_card).filter(ma.col("v") < ma.lit(500_000.0))
    benchmark(_collect, t, threads)


@pytest.mark.benchmark(group="lazy_join")
def test_join(benchmark, join_sides, threads):
    """1M x 1M inner hash join — `HashJoin` has parallel build and probe."""
    left, right = join_sides
    benchmark.extra_info.update(threads=threads, rows=N)
    t = ma.memtable(left).join(ma.memtable(right), on="k", how="inner")
    benchmark(_collect, t, threads)


@pytest.mark.benchmark(group="lazy_sort")
def test_sort(benchmark, high_card, threads):
    """`SortProcessor` already held an `ExecContext`; before this change the one
    it held was always the serial default."""
    benchmark.extra_info.update(threads=threads, rows=N)
    t = ma.memtable(high_card).order_by("k")
    benchmark(_collect, t, threads)


@pytest.mark.benchmark(group="drift_control")
def test_drift_control(benchmark):
    """Touches no marrow code at all — the box's noise floor for this batch."""
    data = np.arange(N, dtype=np.float64)
    benchmark(np.sum, data)
