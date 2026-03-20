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
    """Random 50% selection — unpredictable for branch predictor."""
    rng = random.Random(42)
    return [rng.random() < pct / 100 for _ in range(n)]


def _make_mask_clustered(n, pct, run_len=1024):
    """Clustered runs: blocks of run_len True then blocks of False.

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
        # distribution variants (all ~50% selectivity)
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
        # distribution variants (all ~50% selectivity)
        "mask_rand50": pl.Series(_make_mask_random(n, 50), dtype=pl.Boolean),
        "mask_clustered50": pl.Series(_make_mask_clustered(n, 50), dtype=pl.Boolean),
        "mask_head50": pl.Series(_make_mask_head(n, 50), dtype=pl.Boolean),
        # random at different selectivities
        "mask_rand10": pl.Series(_make_mask_random(n, 10), dtype=pl.Boolean),
        "mask_rand90": pl.Series(_make_mask_random(n, 90), dtype=pl.Boolean),
    }


# ── add ───────────────────────────────────────────────────────────────────────


def test_marrow_add_int64(benchmark, ma_arrays):
    a, b = ma_arrays["int64_a"], ma_arrays["int64_b"]
    benchmark(ma.add, a, b)


def test_pyarrow_add_int64(benchmark, pa_arrays):
    a, b = pa_arrays["int64_a"], pa_arrays["int64_b"]
    benchmark(pc.add, a, b)


def test_marrow_add_float64(benchmark, ma_arrays):
    a, b = ma_arrays["float64_a"], ma_arrays["float64_b"]
    benchmark(ma.add, a, b)


def test_pyarrow_add_float64(benchmark, pa_arrays):
    a, b = pa_arrays["float64_a"], pa_arrays["float64_b"]
    benchmark(pc.add, a, b)


# ── sub ───────────────────────────────────────────────────────────────────────


def test_marrow_sub_int64(benchmark, ma_arrays):
    a, b = ma_arrays["int64_a"], ma_arrays["int64_b"]
    benchmark(ma.sub, a, b)


def test_pyarrow_sub_int64(benchmark, pa_arrays):
    a, b = pa_arrays["int64_a"], pa_arrays["int64_b"]
    benchmark(pc.subtract, a, b)


# ── mul ───────────────────────────────────────────────────────────────────────


def test_marrow_mul_int64(benchmark, ma_arrays):
    a, b = ma_arrays["int64_a"], ma_arrays["int64_b"]
    benchmark(ma.mul, a, b)


def test_pyarrow_mul_int64(benchmark, pa_arrays):
    a, b = pa_arrays["int64_a"], pa_arrays["int64_b"]
    benchmark(pc.multiply, a, b)


# ── div ───────────────────────────────────────────────────────────────────────


def test_marrow_div_float64(benchmark, ma_arrays):
    a, b = ma_arrays["float64_a"], ma_arrays["float64_b"]
    benchmark(ma.div, a, b)


def test_pyarrow_div_float64(benchmark, pa_arrays):
    a, b = pa_arrays["float64_a"], pa_arrays["float64_b"]
    benchmark(pc.divide, a, b)


# ── arithmetic with nulls ────────────────────────────────────────────────────


def test_marrow_add_int64_nulls(benchmark, ma_arrays):
    a, b = ma_arrays["int64_nulls"], ma_arrays["int64_a"]
    benchmark(ma.add, a, b)


def test_pyarrow_add_int64_nulls(benchmark, pa_arrays):
    a, b = pa_arrays["int64_nulls"], pa_arrays["int64_a"]
    benchmark(pc.add, a, b)


# ── sum ───────────────────────────────────────────────────────────────────────


def test_marrow_sum_int64(benchmark, ma_arrays):
    benchmark(ma.sum_, ma_arrays["int64_a"])


def test_pyarrow_sum_int64(benchmark, pa_arrays):
    benchmark(pc.sum, pa_arrays["int64_a"])


def test_marrow_sum_float64(benchmark, ma_arrays):
    benchmark(ma.sum_, ma_arrays["float64_a"])


def test_pyarrow_sum_float64(benchmark, pa_arrays):
    benchmark(pc.sum, pa_arrays["float64_a"])


def test_marrow_sum_int64_nulls(benchmark, ma_arrays):
    benchmark(ma.sum_, ma_arrays["int64_nulls"])


def test_pyarrow_sum_int64_nulls(benchmark, pa_arrays):
    benchmark(pc.sum, pa_arrays["int64_nulls"])


# ── product ───────────────────────────────────────────────────────────────────


def test_marrow_product_int64(benchmark, ma_arrays):
    benchmark(ma.product, ma_arrays["int64_a"])


def test_pyarrow_product_int64(benchmark, pa_arrays):
    benchmark(pc.product, pa_arrays["int64_a"])


# ── min / max ─────────────────────────────────────────────────────────────────


def test_marrow_min_int64(benchmark, ma_arrays):
    benchmark(ma.min_, ma_arrays["int64_a"])


def test_pyarrow_min_int64(benchmark, pa_arrays):
    benchmark(pc.min, pa_arrays["int64_a"])


def test_marrow_max_int64(benchmark, ma_arrays):
    benchmark(ma.max_, ma_arrays["int64_a"])


def test_pyarrow_max_int64(benchmark, pa_arrays):
    benchmark(pc.max, pa_arrays["int64_a"])


def test_marrow_min_float64(benchmark, ma_arrays):
    benchmark(ma.min_, ma_arrays["float64_a"])


def test_pyarrow_min_float64(benchmark, pa_arrays):
    benchmark(pc.min, pa_arrays["float64_a"])


# ── any / all ─────────────────────────────────────────────────────────────────


def test_marrow_any(benchmark, ma_arrays):
    benchmark(ma.any_, ma_arrays["bool"])


def test_pyarrow_any(benchmark, pa_arrays):
    benchmark(pc.any, pa_arrays["bool"])


def test_marrow_all(benchmark, ma_arrays):
    benchmark(ma.all_, ma_arrays["bool"])


def test_pyarrow_all(benchmark, pa_arrays):
    benchmark(pc.all, pa_arrays["bool"])


# ── filter (50% selectivity) ─────────────────────────────────────────────────


def test_marrow_filter_int64_50pct(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["int64_a"], ma_arrays["mask_50"])


def test_pyarrow_filter_int64_50pct(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["int64_a"], pa_arrays["mask_50"])


def test_polars_filter_int64_50pct(benchmark, pl_arrays):
    a, m = pl_arrays["int64_a"], pl_arrays["mask_50"]
    benchmark(a.filter, m)


def test_marrow_filter_float64_50pct(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["float64_a"], ma_arrays["mask_50"])


def test_pyarrow_filter_float64_50pct(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["float64_a"], pa_arrays["mask_50"])


def test_polars_filter_float64_50pct(benchmark, pl_arrays):
    a, m = pl_arrays["float64_a"], pl_arrays["mask_50"]
    benchmark(a.filter, m)


# ── filter (90% selectivity) ─────────────────────────────────────────────────


def test_marrow_filter_int64_90pct(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["int64_a"], ma_arrays["mask_90"])


def test_pyarrow_filter_int64_90pct(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["int64_a"], pa_arrays["mask_90"])


def test_polars_filter_int64_90pct(benchmark, pl_arrays):
    a, m = pl_arrays["int64_a"], pl_arrays["mask_90"]
    benchmark(a.filter, m)


# ── filter (10% selectivity) ─────────────────────────────────────────────────


def test_marrow_filter_int64_10pct(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["int64_a"], ma_arrays["mask_10"])


def test_pyarrow_filter_int64_10pct(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["int64_a"], pa_arrays["mask_10"])


def test_polars_filter_int64_10pct(benchmark, pl_arrays):
    a, m = pl_arrays["int64_a"], pl_arrays["mask_10"]
    benchmark(a.filter, m)


# ── filter (1% selectivity) ──────────────────────────────────────────────────


def test_marrow_filter_int64_1pct(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["int64_a"], ma_arrays["mask_1"])


def test_pyarrow_filter_int64_1pct(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["int64_a"], pa_arrays["mask_1"])


def test_polars_filter_int64_1pct(benchmark, pl_arrays):
    a, m = pl_arrays["int64_a"], pl_arrays["mask_1"]
    benchmark(a.filter, m)


# ── filter (0% selectivity — none selected) ──────────────────────────────────


def test_marrow_filter_int64_0pct(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["int64_a"], ma_arrays["mask_0"])


def test_pyarrow_filter_int64_0pct(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["int64_a"], pa_arrays["mask_0"])


def test_polars_filter_int64_0pct(benchmark, pl_arrays):
    a, m = pl_arrays["int64_a"], pl_arrays["mask_0"]
    benchmark(a.filter, m)


# ── filter string ─────────────────────────────────────────────────────────────


def test_marrow_filter_string_50pct(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["string"], ma_arrays["mask_50"])


def test_pyarrow_filter_string_50pct(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["string"], pa_arrays["mask_50"])


def test_polars_filter_string_50pct(benchmark, pl_arrays):
    a, m = pl_arrays["string"], pl_arrays["mask_50"]
    benchmark(a.filter, m)


def test_marrow_filter_string_10pct(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["string"], ma_arrays["mask_10"])


def test_pyarrow_filter_string_10pct(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["string"], pa_arrays["mask_10"])


def test_polars_filter_string_10pct(benchmark, pl_arrays):
    a, m = pl_arrays["string"], pl_arrays["mask_10"]
    benchmark(a.filter, m)


# ── filter distributions: random 50% (unpredictable mask) ────────────────────


def test_marrow_filter_int64_rand50(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["int64_a"], ma_arrays["mask_rand50"])


def test_pyarrow_filter_int64_rand50(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["int64_a"], pa_arrays["mask_rand50"])


def test_polars_filter_int64_rand50(benchmark, pl_arrays):
    a, m = pl_arrays["int64_a"], pl_arrays["mask_rand50"]
    benchmark(a.filter, m)


# ── filter distributions: random 10% ────────────────────────────────────────


def test_marrow_filter_int64_rand10(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["int64_a"], ma_arrays["mask_rand10"])


def test_pyarrow_filter_int64_rand10(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["int64_a"], pa_arrays["mask_rand10"])


def test_polars_filter_int64_rand10(benchmark, pl_arrays):
    a, m = pl_arrays["int64_a"], pl_arrays["mask_rand10"]
    benchmark(a.filter, m)


# ── filter distributions: random 90% ────────────────────────────────────────


def test_marrow_filter_int64_rand90(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["int64_a"], ma_arrays["mask_rand90"])


def test_pyarrow_filter_int64_rand90(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["int64_a"], pa_arrays["mask_rand90"])


def test_polars_filter_int64_rand90(benchmark, pl_arrays):
    a, m = pl_arrays["int64_a"], pl_arrays["mask_rand90"]
    benchmark(a.filter, m)


# ── filter distributions: clustered runs (~50%) ─────────────────────────────


def test_marrow_filter_int64_clustered50(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["int64_a"], ma_arrays["mask_clustered50"])


def test_pyarrow_filter_int64_clustered50(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["int64_a"], pa_arrays["mask_clustered50"])


def test_polars_filter_int64_clustered50(benchmark, pl_arrays):
    a, m = pl_arrays["int64_a"], pl_arrays["mask_clustered50"]
    benchmark(a.filter, m)


# ── filter distributions: head-heavy (~50%) ─────────────────────────────────


def test_marrow_filter_int64_head50(benchmark, ma_arrays):
    benchmark(ma.filter_, ma_arrays["int64_a"], ma_arrays["mask_head50"])


def test_pyarrow_filter_int64_head50(benchmark, pa_arrays):
    benchmark(pc.filter, pa_arrays["int64_a"], pa_arrays["mask_head50"])


def test_polars_filter_int64_head50(benchmark, pl_arrays):
    a, m = pl_arrays["int64_a"], pl_arrays["mask_head50"]
    benchmark(a.filter, m)


# ── drop_nulls ────────────────────────────────────────────────────────────────


def test_marrow_drop_nulls_int64(benchmark, ma_arrays):
    benchmark(ma.drop_nulls, ma_arrays["int64_nulls"])


def test_pyarrow_drop_nulls_int64(benchmark, pa_arrays):
    benchmark(pc.drop_null, pa_arrays["int64_nulls"])


def test_polars_drop_nulls_int64(benchmark, pl_arrays):
    benchmark(pl_arrays["int64_nulls"].drop_nulls)


def test_marrow_drop_nulls_float64(benchmark, ma_arrays):
    benchmark(ma.drop_nulls, ma_arrays["float64_nulls"])


def test_pyarrow_drop_nulls_float64(benchmark, pa_arrays):
    benchmark(pc.drop_null, pa_arrays["float64_nulls"])


def test_polars_drop_nulls_float64(benchmark, pl_arrays):
    benchmark(pl_arrays["float64_nulls"].drop_nulls)


# ── equal ────────────────────────────────────────────────────────────────────


def test_marrow_equal_int64(benchmark, ma_arrays):
    a, b = ma_arrays["int64_a"], ma_arrays["int64_b"]
    benchmark(ma.equal, a, b)


def test_pyarrow_equal_int64(benchmark, pa_arrays):
    a, b = pa_arrays["int64_a"], pa_arrays["int64_b"]
    benchmark(pc.equal, a, b)


def test_marrow_equal_float64(benchmark, ma_arrays):
    a, b = ma_arrays["float64_a"], ma_arrays["float64_b"]
    benchmark(ma.equal, a, b)


def test_pyarrow_equal_float64(benchmark, pa_arrays):
    a, b = pa_arrays["float64_a"], pa_arrays["float64_b"]
    benchmark(pc.equal, a, b)


# ── less ─────────────────────────────────────────────────────────────────────


def test_marrow_less_int64(benchmark, ma_arrays):
    a, b = ma_arrays["int64_a"], ma_arrays["int64_b"]
    benchmark(ma.less, a, b)


def test_pyarrow_less_int64(benchmark, pa_arrays):
    a, b = pa_arrays["int64_a"], pa_arrays["int64_b"]
    benchmark(pc.less, a, b)


def test_marrow_less_float64(benchmark, ma_arrays):
    a, b = ma_arrays["float64_a"], ma_arrays["float64_b"]
    benchmark(ma.less, a, b)


def test_pyarrow_less_float64(benchmark, pa_arrays):
    a, b = pa_arrays["float64_a"], pa_arrays["float64_b"]
    benchmark(pc.less, a, b)
