"""Sort comparison: marrow vs pyarrow vs polars, single- AND multi-threaded.

Threading mode is chosen by env so the same file covers both — run it twice:

    # multi-threaded (all cores)
    pixi run -e bench pytest python/marrow/tests/bench_sort_compare.py \\
        --benchmark --competition

    # single-threaded (fair 1-thread comparison)
    MARROW_SORT_SERIAL=1 pixi run -e bench pytest \\
        python/marrow/tests/bench_sort_compare.py --benchmark --competition

Covers single-column ``sort_indices`` and multi-column ``sort_by`` (the
column-oriented LSD path) over random int64/int32 at 1M and 10M rows.
"""

import os

SERIAL = bool(os.environ.get("MARROW_SORT_SERIAL"))
# Polars/pyarrow read their thread-pool size at import, so pin before importing.
if SERIAL:
    os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
import marrow as ma

if SERIAL:
    pa.set_cpu_count(1)

_NT = 1 if SERIAL else 0  # marrow num_threads: 1 = serial, 0 = all cores
_CTX = ma.ExecutionContext.serial() if SERIAL else ma.ExecutionContext.parallel(0)
_MODE = "1t" if SERIAL else "mt"

SIZES = [1_000_000, 10_000_000]
_rng = np.random.default_rng(0)


@pytest.fixture(params=SIZES, ids=[f"n={n}" for n in SIZES], scope="session")
def n(request):
    return request.param


@pytest.fixture(scope="session")
def data(n):
    key = _rng.integers(0, 1000, n).astype(np.int32)
    val = _rng.integers(0, 2**60, n).astype(np.int64)
    tbl = pa.table({"k": key, "v": val})
    return dict(
        n=n,
        pa_val=pa.array(val),
        ma_val=ma.array(pa.array(val)),
        pl_val=pl.Series(val),
        pa_tbl=tbl,
        ma_rb=ma.record_batch(tbl.to_batches()[0]),
        pl_df=pl.from_arrow(tbl),
    )


def _info(benchmark, lib, n):
    benchmark.extra_info.update(lib=lib, n=n, mode=_MODE)


# ── single-column sort_indices (random int64) ────────────────────────────────


@pytest.mark.benchmark(group=f"sort1col_{_MODE}")
def test_marrow_1col(benchmark, data, n):
    _info(benchmark, "marrow", n)
    benchmark(lambda: ma.compute.sort_indices(data["ma_val"], ctx=_CTX))


@pytest.mark.benchmark(group=f"sort1col_{_MODE}")
def test_pyarrow_1col(benchmark, data, n):
    _info(benchmark, "pyarrow", n)
    benchmark(lambda: pc.sort_indices(data["pa_val"]))


@pytest.mark.benchmark(group=f"sort1col_{_MODE}")
def test_polars_1col(benchmark, data, n):
    _info(benchmark, "polars", n)
    benchmark(lambda: data["pl_val"].arg_sort())


# ── multi-column sort (int32 key + int64) ────────────────────────────────────

_ORDER = [("k", "ascending"), ("v", "ascending")]


@pytest.mark.benchmark(group=f"sort2col_{_MODE}")
def test_marrow_2col(benchmark, data, n):
    _info(benchmark, "marrow", n)
    benchmark(lambda: data["ma_rb"].sort_by(_ORDER, num_threads=_NT))


@pytest.mark.benchmark(group=f"sort2col_{_MODE}")
def test_pyarrow_2col(benchmark, data, n):
    _info(benchmark, "pyarrow", n)
    benchmark(lambda: data["pa_tbl"].sort_by(_ORDER))


@pytest.mark.benchmark(group=f"sort2col_{_MODE}")
def test_polars_2col(benchmark, data, n):
    _info(benchmark, "polars", n)
    benchmark(lambda: data["pl_df"].sort(["k", "v"]))
