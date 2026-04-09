"""Benchmarks for marrow compute kernels vs PyArrow and Polars."""

import random

import pytest
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
import marrow as ma


SIZES = [10_000, 100_000, 1_000_000]


def _make_int64(n):
    return list(range(n))


def _make_int64_nulls(n):
    return [i if i % 10 != 0 else None for i in range(n)]


def _make_float64(n):
    return [float(i) for i in range(n)]


def _make_float64_nulls(n):
    return [float(i) if i % 10 != 0 else None for i in range(n)]


def _make_bool(n):
    return [i % 3 == 0 for i in range(n)]


def _make_string(n):
    return [f"item-{i}" for i in range(n)]


def _make_mask_random(n, pct):
    """Random selection — unpredictable for branch predictor."""
    rng = random.Random(42)
    return [rng.random() < pct / 100 for _ in range(n)]


def _make_mask_clustered(n, pct, run_len=1024):
    """Clustered runs: blocks of True then blocks of False.

    Good for testing run-skipping optimisations (zero-word fast path).
    """
    true_runs = max(1, int(n * pct / 100 / run_len))
    result = [False] * n
    rng = random.Random(42)
    starts = sorted(rng.sample(range(max(1, n - run_len)), true_runs))
    for s in starts:
        for i in range(s, min(s + run_len, n)):
            result[i] = True
    return result


def _make_mask_head(n, pct):
    """First pct% elements selected, rest False — tests early-exit / tail."""
    cutoff = n * pct // 100
    return [i < cutoff for i in range(n)]


def _build_arrays(n, lib, types):
    """Build a dict of named arrays for a given size and library."""
    array_fn = lib.array
    return {
        "int64_a": array_fn(_make_int64(n), type=types["int64"]),
        "int64_b": array_fn(_make_int64(n), type=types["int64"]),
        "int64_nulls": array_fn(_make_int64_nulls(n), type=types["int64"]),
        "float64_a": array_fn(_make_float64(n), type=types["float64"]),
        "float64_b": array_fn(_make_float64(n), type=types["float64"]),
        "float64_nulls": array_fn(_make_float64_nulls(n), type=types["float64"]),
        "bool": array_fn(_make_bool(n), type=types["bool_"]),
        "string": array_fn(_make_string(n), type=types["string"]),
        "mask_90": array_fn([i % 10 != 0 for i in range(n)], type=types["bool_"]),
        "mask_50": array_fn([i % 2 == 0 for i in range(n)], type=types["bool_"]),
        "mask_10": array_fn([i % 10 == 0 for i in range(n)], type=types["bool_"]),
        "mask_1": array_fn([i % 100 == 0 for i in range(n)], type=types["bool_"]),
        "mask_0": array_fn([False for _ in range(n)], type=types["bool_"]),
        # distribution variants (~50% selectivity)
        "mask_rand50": array_fn(_make_mask_random(n, 50), type=types["bool_"]),
        "mask_clustered50": array_fn(_make_mask_clustered(n, 50), type=types["bool_"]),
        "mask_head50": array_fn(_make_mask_head(n, 50), type=types["bool_"]),
        # random at different selectivities
        "mask_rand10": array_fn(_make_mask_random(n, 10), type=types["bool_"]),
        "mask_rand90": array_fn(_make_mask_random(n, 90), type=types["bool_"]),
    }


MA_TYPES = {
    "int64": ma.int64(),
    "float64": ma.float64(),
    "bool_": ma.bool_(),
    "string": ma.string(),
}
PA_TYPES = {
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool_": pa.bool_(),
    "string": pa.string(),
}


@pytest.fixture(params=SIZES, ids=[f"n={n}" for n in SIZES], scope="session")
def n(request):
    return request.param


@pytest.fixture(scope="session")
def ma_arrays(n):
    return _build_arrays(n, ma, MA_TYPES)


@pytest.fixture(scope="session")
def pa_arrays(n):
    return _build_arrays(n, pa, PA_TYPES)


@pytest.fixture(scope="session")
def pl_arrays(n):
    return {
        "int64_a": pl.Series(_make_int64(n), dtype=pl.Int64),
        "int64_b": pl.Series(_make_int64(n), dtype=pl.Int64),
        "int64_nulls": pl.Series(_make_int64_nulls(n), dtype=pl.Int64),
        "float64_a": pl.Series(_make_float64(n), dtype=pl.Float64),
        "float64_b": pl.Series(_make_float64(n), dtype=pl.Float64),
        "float64_nulls": pl.Series(_make_float64_nulls(n), dtype=pl.Float64),
        "bool": pl.Series(_make_bool(n), dtype=pl.Boolean),
        "string": pl.Series(_make_string(n), dtype=pl.String),
        "mask_90": pl.Series([i % 10 != 0 for i in range(n)], dtype=pl.Boolean),
        "mask_50": pl.Series([i % 2 == 0 for i in range(n)], dtype=pl.Boolean),
        "mask_10": pl.Series([i % 10 == 0 for i in range(n)], dtype=pl.Boolean),
        "mask_1": pl.Series([i % 100 == 0 for i in range(n)], dtype=pl.Boolean),
        "mask_0": pl.Series([False for _ in range(n)], dtype=pl.Boolean),
        "mask_rand50": pl.Series(_make_mask_random(n, 50), dtype=pl.Boolean),
        "mask_clustered50": pl.Series(_make_mask_clustered(n, 50), dtype=pl.Boolean),
        "mask_head50": pl.Series(_make_mask_head(n, 50), dtype=pl.Boolean),
        "mask_rand10": pl.Series(_make_mask_random(n, 10), dtype=pl.Boolean),
        "mask_rand90": pl.Series(_make_mask_random(n, 90), dtype=pl.Boolean),
    }


# ── Shared parametrize marks ─────────────────────────────────────────────────

# Binary: int64 and float64 (a + b arrays)
_BINARY_NUM_CASES = [
    ("int64",   "int64_a",   "int64_b"),
    ("float64", "float64_a", "float64_b"),
]
_binary_num_params = pytest.mark.parametrize(
    "dtype,a,b", _BINARY_NUM_CASES, ids=[c[0] for c in _BINARY_NUM_CASES]
)

# Binary arithmetic: int64, float64, and int64 with nulls
_ARITH_CASES = [
    ("int64",       "int64_a",    "int64_b"),
    ("float64",     "float64_a",  "float64_b"),
    ("int64_nulls", "int64_nulls", "int64_a"),
]
_arith_params = pytest.mark.parametrize(
    "dtype,a,b", _ARITH_CASES, ids=[c[0] for c in _ARITH_CASES]
)

# Unary sum/aggregate: int64, float64, and int64 with nulls
_SUM_CASES = [
    ("int64",       "int64_a"),
    ("float64",     "float64_a"),
    ("int64_nulls", "int64_nulls"),
]
_sum_params = pytest.mark.parametrize(
    "dtype,key", _SUM_CASES, ids=[c[0] for c in _SUM_CASES]
)

# Unary numeric: int64 and float64
_UNARY_NUM_CASES = [
    ("int64",   "int64_a"),
    ("float64", "float64_a"),
]
_unary_num_params = pytest.mark.parametrize(
    "dtype,key", _UNARY_NUM_CASES, ids=[c[0] for c in _UNARY_NUM_CASES]
)

# Nullable: int64_nulls and float64_nulls
_NULL_CASES = [
    ("int64",   "int64_nulls"),
    ("float64", "float64_nulls"),
]
_null_params = pytest.mark.parametrize(
    "dtype,key", _NULL_CASES, ids=[c[0] for c in _NULL_CASES]
)

# Filter cases: (id, dtype_key, mask_key, selectivity, distribution)
_FILTER_CASES = [
    ("int64_50pct",       "int64_a",   "mask_50",          50, "uniform"),
    ("int64_90pct",       "int64_a",   "mask_90",          90, "uniform"),
    ("int64_10pct",       "int64_a",   "mask_10",          10, "uniform"),
    ("int64_1pct",        "int64_a",   "mask_1",            1, "uniform"),
    ("int64_0pct",        "int64_a",   "mask_0",            0, "uniform"),
    ("float64_50pct",     "float64_a", "mask_50",          50, "uniform"),
    ("string_50pct",      "string",    "mask_50",          50, "uniform"),
    ("string_10pct",      "string",    "mask_10",          10, "uniform"),
    ("int64_rand50",      "int64_a",   "mask_rand50",      50, "random"),
    ("int64_rand10",      "int64_a",   "mask_rand10",      10, "random"),
    ("int64_rand90",      "int64_a",   "mask_rand90",      90, "random"),
    ("int64_clustered50", "int64_a",   "mask_clustered50", 50, "clustered"),
    ("int64_head50",      "int64_a",   "mask_head50",      50, "head"),
]
_filter_params = pytest.mark.parametrize(
    "dtype_key,mask_key,selectivity,distribution",
    [c[1:] for c in _FILTER_CASES],
    ids=[c[0] for c in _FILTER_CASES],
)


# ── Arithmetic ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="arithmetic")
@_arith_params
def test_marrow_add(benchmark, ma_arrays, n, dtype, a, b):
    benchmark.extra_info.update(lib="marrow", n=n, dtype=dtype)
    benchmark(ma.add, ma_arrays[a], ma_arrays[b])


@pytest.mark.benchmark(group="arithmetic")
@_arith_params
def test_pyarrow_add(benchmark, pa_arrays, n, dtype, a, b):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype=dtype)
    benchmark(pc.add, pa_arrays[a], pa_arrays[b])


@pytest.mark.benchmark(group="arithmetic")
def test_marrow_sub(benchmark, ma_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n, dtype="int64")
    benchmark(ma.sub, ma_arrays["int64_a"], ma_arrays["int64_b"])


@pytest.mark.benchmark(group="arithmetic")
def test_pyarrow_sub(benchmark, pa_arrays, n):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype="int64")
    benchmark(pc.subtract, pa_arrays["int64_a"], pa_arrays["int64_b"])


@pytest.mark.benchmark(group="arithmetic")
def test_marrow_mul(benchmark, ma_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n, dtype="int64")
    benchmark(ma.mul, ma_arrays["int64_a"], ma_arrays["int64_b"])


@pytest.mark.benchmark(group="arithmetic")
def test_pyarrow_mul(benchmark, pa_arrays, n):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype="int64")
    benchmark(pc.multiply, pa_arrays["int64_a"], pa_arrays["int64_b"])


@pytest.mark.benchmark(group="arithmetic")
def test_marrow_div(benchmark, ma_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n, dtype="float64")
    benchmark(ma.div, ma_arrays["float64_a"], ma_arrays["float64_b"])


@pytest.mark.benchmark(group="arithmetic")
def test_pyarrow_div(benchmark, pa_arrays, n):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype="float64")
    benchmark(pc.divide, pa_arrays["float64_a"], pa_arrays["float64_b"])


# ── Aggregate ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="aggregate")
@_sum_params
def test_marrow_sum(benchmark, ma_arrays, n, dtype, key):
    benchmark.extra_info.update(lib="marrow", n=n, dtype=dtype)
    benchmark(ma.sum_, ma_arrays[key])


@pytest.mark.benchmark(group="aggregate")
@_sum_params
def test_pyarrow_sum(benchmark, pa_arrays, n, dtype, key):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype=dtype)
    benchmark(pc.sum, pa_arrays[key])


@pytest.mark.benchmark(group="aggregate")
def test_marrow_product(benchmark, ma_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n, dtype="int64")
    benchmark(ma.product, ma_arrays["int64_a"])


@pytest.mark.benchmark(group="aggregate")
def test_pyarrow_product(benchmark, pa_arrays, n):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype="int64")
    benchmark(pc.product, pa_arrays["int64_a"])


@pytest.mark.benchmark(group="aggregate")
@_unary_num_params
def test_marrow_min(benchmark, ma_arrays, n, dtype, key):
    benchmark.extra_info.update(lib="marrow", n=n, dtype=dtype)
    benchmark(ma.min_, ma_arrays[key])


@pytest.mark.benchmark(group="aggregate")
@_unary_num_params
def test_pyarrow_min(benchmark, pa_arrays, n, dtype, key):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype=dtype)
    benchmark(pc.min, pa_arrays[key])


@pytest.mark.benchmark(group="aggregate")
def test_marrow_max(benchmark, ma_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n, dtype="int64")
    benchmark(ma.max_, ma_arrays["int64_a"])


@pytest.mark.benchmark(group="aggregate")
def test_pyarrow_max(benchmark, pa_arrays, n):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype="int64")
    benchmark(pc.max, pa_arrays["int64_a"])


@pytest.mark.benchmark(group="aggregate")
def test_marrow_any(benchmark, ma_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n)
    benchmark(ma.any_, ma_arrays["bool"])


@pytest.mark.benchmark(group="aggregate")
def test_pyarrow_any(benchmark, pa_arrays, n):
    benchmark.extra_info.update(lib="pyarrow", n=n)
    benchmark(pc.any, pa_arrays["bool"])


@pytest.mark.benchmark(group="aggregate")
def test_marrow_all(benchmark, ma_arrays, n):
    benchmark.extra_info.update(lib="marrow", n=n)
    benchmark(ma.all_, ma_arrays["bool"])


@pytest.mark.benchmark(group="aggregate")
def test_pyarrow_all(benchmark, pa_arrays, n):
    benchmark.extra_info.update(lib="pyarrow", n=n)
    benchmark(pc.all, pa_arrays["bool"])


# ── Filter ────────────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="filter")
@_filter_params
def test_marrow_filter(benchmark, ma_arrays, n, dtype_key, mask_key, selectivity, distribution):
    dtype = dtype_key.split("_")[0]
    benchmark.extra_info.update(lib="marrow", n=n, dtype=dtype, selectivity=selectivity, distribution=distribution)
    benchmark(ma.filter_, ma_arrays[dtype_key], ma_arrays[mask_key])


@pytest.mark.benchmark(group="filter")
@_filter_params
def test_pyarrow_filter(benchmark, pa_arrays, n, dtype_key, mask_key, selectivity, distribution):
    dtype = dtype_key.split("_")[0]
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype=dtype, selectivity=selectivity, distribution=distribution)
    benchmark(pc.filter, pa_arrays[dtype_key], pa_arrays[mask_key])


@pytest.mark.benchmark(group="filter")
@_filter_params
def test_polars_filter(benchmark, pl_arrays, n, dtype_key, mask_key, selectivity, distribution):
    dtype = dtype_key.split("_")[0]
    benchmark.extra_info.update(lib="polars", n=n, dtype=dtype, selectivity=selectivity, distribution=distribution)
    benchmark(pl_arrays[dtype_key].filter, pl_arrays[mask_key])


# ── Drop nulls ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="drop_nulls")
@_null_params
def test_marrow_drop_nulls(benchmark, ma_arrays, n, dtype, key):
    benchmark.extra_info.update(lib="marrow", n=n, dtype=dtype)
    benchmark(ma.drop_nulls, ma_arrays[key])


@pytest.mark.benchmark(group="drop_nulls")
@_null_params
def test_pyarrow_drop_nulls(benchmark, pa_arrays, n, dtype, key):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype=dtype)
    benchmark(pc.drop_null, pa_arrays[key])


@pytest.mark.benchmark(group="drop_nulls")
@_null_params
def test_polars_drop_nulls(benchmark, pl_arrays, n, dtype, key):
    benchmark.extra_info.update(lib="polars", n=n, dtype=dtype)
    benchmark(pl_arrays[key].drop_nulls)


# ── Comparison ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="comparison")
@_binary_num_params
def test_marrow_equal(benchmark, ma_arrays, n, dtype, a, b):
    benchmark.extra_info.update(lib="marrow", n=n, dtype=dtype)
    benchmark(ma.equal, ma_arrays[a], ma_arrays[b])


@pytest.mark.benchmark(group="comparison")
@_binary_num_params
def test_pyarrow_equal(benchmark, pa_arrays, n, dtype, a, b):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype=dtype)
    benchmark(pc.equal, pa_arrays[a], pa_arrays[b])


@pytest.mark.benchmark(group="comparison")
@_binary_num_params
def test_marrow_less(benchmark, ma_arrays, n, dtype, a, b):
    benchmark.extra_info.update(lib="marrow", n=n, dtype=dtype)
    benchmark(ma.less, ma_arrays[a], ma_arrays[b])


@pytest.mark.benchmark(group="comparison")
@_binary_num_params
def test_pyarrow_less(benchmark, pa_arrays, n, dtype, a, b):
    benchmark.extra_info.update(lib="pyarrow", n=n, dtype=dtype)
    benchmark(pc.less, pa_arrays[a], pa_arrays[b])
