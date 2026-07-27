"""Benchmarks for sort: marrow vs PyArrow vs Polars vs DuckDB vs NumPy.

Single-threaded throughout for a fair comparison with marrow's serial Phase 1.

Run with:
    pixi run -e bench pytest python/marrow/tests/bench_sort.py --benchmark
    pixi run -e bench pytest python/marrow/tests/bench_sort.py --benchmark --competition
"""

import os

# Must be set before importing polars/numpy so their thread pools initialise at 1.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
import pytest
import marrow as ma

try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

RNG = np.random.default_rng(42)

SIZES = [10_000, 100_000, 1_000_000, 10_000_000]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=SIZES, ids=[f"n={n}" for n in SIZES], scope="session")
def n(request):
    return request.param


@pytest.fixture(scope="session")
def int64_arrays(n):
    vals = RNG.integers(-(2**62), 2**62, size=n, dtype=np.int64)
    return {
        "pa": pa.chunked_array([pa.array(vals, type=pa.int64())]),
        "pl": pl.Series(vals),
        "np": vals,
        "ma": ma.array(vals.tolist(), type=ma.int64()),
    }


@pytest.fixture(scope="session")
def float64_arrays(n):
    vals = RNG.standard_normal(n)
    return {
        "pa": pa.chunked_array([pa.array(vals, type=pa.float64())]),
        "pl": pl.Series(vals),
        "np": vals,
        "ma": ma.array(vals.tolist(), type=ma.float64()),
    }


@pytest.fixture(scope="session")
def duck_con(int64_arrays, float64_arrays):
    if not _HAS_DUCKDB:
        return None
    con = duckdb.connect(config={"threads": 1})
    con.register("ti", pa.table({"v": int64_arrays["pa"]}))
    con.register("tf", pa.table({"v": float64_arrays["pa"]}))
    return con


# ---------------------------------------------------------------------------
# int64 sort
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="sort_int64")
def test_marrow_sort_indices_int64(benchmark, int64_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n)
    arr = int64_arrays["ma"]
    benchmark(ma.compute.sort_indices, arr)


@pytest.mark.benchmark(group="sort_int64")
def test_marrow_sort_indices_int64_serial(benchmark, int64_arrays, n):
    benchmark.extra_info.update(lib="marrow_serial", n=n)
    arr = int64_arrays["ma"]
    benchmark(ma.compute.sort_indices, arr, ctx=ma.ExecutionContext.serial())


@pytest.mark.benchmark(group="sort_int64")
def test_marrow_sort_int64(benchmark, int64_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n)
    arr = int64_arrays["ma"]
    benchmark(ma.compute.sort, arr)


@pytest.mark.benchmark(group="sort_int64")
def test_pyarrow_sort_indices_int64(benchmark, int64_arrays, n):
    benchmark.extra_info.update(lib="pyarrow", n=n)
    benchmark(pc.sort_indices, int64_arrays["pa"])


@pytest.mark.benchmark(group="sort_int64")
def test_polars_argsort_int64(benchmark, int64_arrays, n):
    benchmark.extra_info.update(lib="polars", n=n)
    benchmark(int64_arrays["pl"].arg_sort)


@pytest.mark.benchmark(group="sort_int64")
def test_numpy_argsort_int64(benchmark, int64_arrays, n):
    benchmark.extra_info.update(lib="numpy", n=n)
    benchmark(np.argsort, int64_arrays["np"])


@pytest.mark.benchmark(group="sort_int64")
@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb not installed")
def test_duckdb_sort_int64(benchmark, duck_con, n):
    benchmark.extra_info.update(lib="duckdb", n=n)

    def _run():
        duck_con.execute("SELECT row_number() OVER (ORDER BY v) FROM ti").fetchall()

    benchmark(_run)


# ---------------------------------------------------------------------------
# float64 sort
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="sort_float64")
def test_marrow_sort_indices_float64(benchmark, float64_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n)
    arr = float64_arrays["ma"]
    benchmark(ma.compute.sort_indices, arr)


@pytest.mark.benchmark(group="sort_float64")
def test_marrow_sort_indices_float64_serial(benchmark, float64_arrays, n):
    benchmark.extra_info.update(lib="marrow_serial", n=n)
    arr = float64_arrays["ma"]
    benchmark(ma.compute.sort_indices, arr, ctx=ma.ExecutionContext.serial())


@pytest.mark.benchmark(group="sort_float64")
def test_marrow_sort_float64(benchmark, float64_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n)
    arr = float64_arrays["ma"]
    benchmark(ma.compute.sort, arr)


@pytest.mark.benchmark(group="sort_float64")
def test_pyarrow_sort_indices_float64(benchmark, float64_arrays, n):
    benchmark.extra_info.update(lib="pyarrow", n=n)
    benchmark(pc.sort_indices, float64_arrays["pa"])


@pytest.mark.benchmark(group="sort_float64")
def test_polars_argsort_float64(benchmark, float64_arrays, n):
    benchmark.extra_info.update(lib="polars", n=n)
    benchmark(float64_arrays["pl"].arg_sort)


@pytest.mark.benchmark(group="sort_float64")
def test_numpy_argsort_float64(benchmark, float64_arrays, n):
    benchmark.extra_info.update(lib="numpy", n=n)
    benchmark(np.argsort, float64_arrays["np"])


@pytest.mark.benchmark(group="sort_float64")
@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb not installed")
def test_duckdb_sort_float64(benchmark, duck_con, n):
    benchmark.extra_info.update(lib="duckdb", n=n)

    def _run():
        duck_con.execute("SELECT row_number() OVER (ORDER BY v) FROM tf").fetchall()

    benchmark(_run)
