"""Benchmarks for hash join: marrow vs PyArrow vs Polars."""

import pytest
import pyarrow as pa
import polars as pl
import marrow as ma


SIZES = [10_000, 100_000, 1_000_000]


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
            {"k": ma.array(left_keys, type=ma.int64()), "v": ma.array(left_vals, type=ma.int64())}
        ),
        "ma_right": ma.record_batch(
            {"k": ma.array(right_keys, type=ma.int64()), "w": ma.array(right_vals, type=ma.int64())}
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


# ---------------------------------------------------------------------------
# Inner join — all keys match (1:1 dense join)
# ---------------------------------------------------------------------------


def test_pyarrow_inner_join(benchmark, tables):
    left, right = tables["pa_left"], tables["pa_right"]
    benchmark(left.join, right, keys="k", join_type="inner")


def test_polars_inner_join(benchmark, tables):
    left, right = tables["pl_left"], tables["pl_right"]
    benchmark(left.join, right, on="k", how="inner")


def test_marrow_inner_join(benchmark, tables):
    left, right = tables["ma_left"], tables["ma_right"]
    benchmark(left.join, right, ["k"], None, "inner")


# ---------------------------------------------------------------------------
# Inner join — 50% selectivity (half of right keys match)
# ---------------------------------------------------------------------------


def test_pyarrow_inner_join_half(benchmark, tables_half):
    left, right = tables_half["pa_left"], tables_half["pa_right"]
    benchmark(left.join, right, keys="k", join_type="inner")


def test_polars_inner_join_half(benchmark, tables_half):
    left, right = tables_half["pl_left"], tables_half["pl_right"]
    benchmark(left.join, right, on="k", how="inner")


def test_marrow_inner_join_half(benchmark, tables_half):
    left, right = tables_half["ma_left"], tables_half["ma_right"]
    benchmark(left.join, right, ["k"], None, "inner")


# ---------------------------------------------------------------------------
# Left join
# ---------------------------------------------------------------------------


def test_pyarrow_left_join(benchmark, tables):
    left, right = tables["pa_left"], tables["pa_right"]
    benchmark(left.join, right, keys="k", join_type="left outer")


def test_polars_left_join(benchmark, tables):
    left, right = tables["pl_left"], tables["pl_right"]
    benchmark(left.join, right, on="k", how="left")


def test_marrow_left_join(benchmark, tables):
    left, right = tables["ma_left"], tables["ma_right"]
    benchmark(left.join, right, ["k"], None, "left")


# ---------------------------------------------------------------------------
# Semi join
# ---------------------------------------------------------------------------


def test_pyarrow_semi_join(benchmark, tables):
    left, right = tables["pa_left"], tables["pa_right"]
    benchmark(left.join, right, keys="k", join_type="left semi")


def test_polars_semi_join(benchmark, tables):
    left, right = tables["pl_left"], tables["pl_right"]
    benchmark(left.join, right, on="k", how="semi")


def test_marrow_semi_join(benchmark, tables):
    left, right = tables["ma_left"], tables["ma_right"]
    benchmark(left.join, right, ["k"], None, "semi")
