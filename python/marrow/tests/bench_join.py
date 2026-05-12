"""Benchmarks for hash join: marrow vs PyArrow vs Polars.

All benchmarks run single-threaded for fair comparison with marrow.
"""

import os

import pytest

try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False
import pyarrow as pa
import polars as pl
import marrow as ma

# Force single-threaded execution for fair comparison.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

SIZES = [10_000, 100_000, 1_000_000, 10_000_000]


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_tables(n, selectivity=1.0):
    """Build left/right tables with int64 key + int64 payload.

    Args:
        n: Number of rows per side.
        selectivity: Fraction of right keys that match a left key.
            1.0 = all match (dense join), 0.5 = half match, etc.
    """
    left_keys = list(range(n))
    match_bound = int(n * selectivity)
    right_keys = [i % match_bound if match_bound > 0 else n + i for i in range(n)]
    left_vals = list(range(0, n * 10, 10))
    right_vals = list(range(0, n * 100, 100))

    return {
        "pa_left": pa.table({"k": left_keys, "v": left_vals}),
        "pa_right": pa.table({"k": right_keys, "w": right_vals}),
        "pl_left": pl.DataFrame({"k": left_keys, "v": left_vals}),
        "pl_right": pl.DataFrame({"k": right_keys, "w": right_vals}),
        "ma_left": ma.record_batch(
            {
                "k": ma.array(left_keys, type=ma.int64()),
                "v": ma.array(left_vals, type=ma.int64()),
            }
        ),
        "ma_right": ma.record_batch(
            {
                "k": ma.array(right_keys, type=ma.int64()),
                "w": ma.array(right_vals, type=ma.int64()),
            }
        ),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=SIZES, ids=[f"n={n}" for n in SIZES], scope="session")
def n(request):
    return request.param


@pytest.fixture(scope="session")
def tables(n):
    return _make_tables(n)


@pytest.fixture(scope="session")
def tables_half(n):
    return _make_tables(n, selectivity=0.5)


@pytest.fixture(scope="session")
def duck_con(tables):
    if not _HAS_DUCKDB:
        return None
    con = duckdb.connect(config={"threads": 1})
    con.register("l", tables["pa_left"])
    con.register("r", tables["pa_right"])
    return con


@pytest.fixture(scope="session")
def duck_con_half(tables_half):
    if not _HAS_DUCKDB:
        return None
    con = duckdb.connect(config={"threads": 1})
    con.register("l", tables_half["pa_left"])
    con.register("r", tables_half["pa_right"])
    return con


# ---------------------------------------------------------------------------
# Shared parametrize mark
# ---------------------------------------------------------------------------

_JOIN_TYPES = [
    ("inner", "inner", "inner"),
    ("left", "left outer", "left"),
    ("semi", "left semi", "semi"),
]
_join_ids = [j[0] for j in _JOIN_TYPES]
_join_params_pa = pytest.mark.parametrize(
    "pa_type", [j[1] for j in _JOIN_TYPES], ids=_join_ids
)
_join_params_pl = pytest.mark.parametrize(
    "pl_type", [j[2] for j in _JOIN_TYPES], ids=_join_ids
)
_join_params_ma = pytest.mark.parametrize(
    "join_type", [j[0] for j in _JOIN_TYPES], ids=_join_ids
)

_DUCK_SQL = {
    "inner": "SELECT * FROM l JOIN r ON l.k = r.k",
    "left": "SELECT * FROM l LEFT JOIN r ON l.k = r.k",
    "semi": "SELECT l.* FROM l SEMI JOIN r ON l.k = r.k",
}
_skip_no_duckdb = pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb not installed")


# ---------------------------------------------------------------------------
# Full selectivity (all keys match)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="join")
@_join_params_pa
def test_pyarrow_join(benchmark, tables, n, pa_type):
    benchmark.extra_info.update(lib="pyarrow", n=n)
    left, right = tables["pa_left"], tables["pa_right"]
    benchmark(left.join, right, keys="k", join_type=pa_type)


@pytest.mark.benchmark(group="join")
@_join_params_pl
def test_polars_join(benchmark, tables, n, pl_type):
    benchmark.extra_info.update(lib="polars", n=n)
    left, right = tables["pl_left"], tables["pl_right"]
    benchmark(left.join, right, on="k", how=pl_type)


@pytest.mark.benchmark(group="join")
@_join_params_ma
def test_marrow_join(benchmark, tables, n, join_type):
    benchmark.extra_info.update(lib="marrow", n=n)
    left, right = tables["ma_left"], tables["ma_right"]
    # num_threads=1 — single-threaded comparison.
    benchmark(left.join, right, ["k"], None, join_type, 1)


# ---------------------------------------------------------------------------
# 50% selectivity (half of right keys match)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="join")
@_join_params_pa
def test_pyarrow_join_half(benchmark, tables_half, n, pa_type):
    benchmark.extra_info.update(lib="pyarrow", n=n)
    left, right = tables_half["pa_left"], tables_half["pa_right"]
    benchmark(left.join, right, keys="k", join_type=pa_type)


@pytest.mark.benchmark(group="join")
@_join_params_pl
def test_polars_join_half(benchmark, tables_half, n, pl_type):
    benchmark.extra_info.update(lib="polars", n=n)
    left, right = tables_half["pl_left"], tables_half["pl_right"]
    benchmark(left.join, right, on="k", how=pl_type)


@pytest.mark.benchmark(group="join")
@_join_params_ma
def test_marrow_join_half(benchmark, tables_half, n, join_type):
    benchmark.extra_info.update(lib="marrow", n=n)
    left, right = tables_half["ma_left"], tables_half["ma_right"]
    # num_threads=1 — single-threaded comparison.
    benchmark(left.join, right, ["k"], None, join_type, 1)


# ---------------------------------------------------------------------------
# DuckDB (skipped when duckdb is not installed)
# ---------------------------------------------------------------------------


@_skip_no_duckdb
@pytest.mark.benchmark(group="join")
@_join_params_ma
def test_duckdb_join(benchmark, duck_con, n, join_type):
    benchmark.extra_info.update(lib="duckdb", n=n)
    q = _DUCK_SQL[join_type]
    benchmark(lambda: duck_con.execute(q).arrow())


@_skip_no_duckdb
@pytest.mark.benchmark(group="join")
@_join_params_ma
def test_duckdb_join_half(benchmark, duck_con_half, n, join_type):
    benchmark.extra_info.update(lib="duckdb", n=n)
    q = _DUCK_SQL[join_type]
    benchmark(lambda: duck_con_half.execute(q).arrow())
