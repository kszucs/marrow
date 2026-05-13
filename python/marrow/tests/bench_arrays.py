"""Benchmarks for marrow.array() vs PyArrow."""

import pytest
import pyarrow as pa
import marrow as ma


SIZES = [10_000, 100_000]


def _make_data(n):
    return dict(
        int_data=list(range(n)),
        float_data=[float(i) for i in range(n)],
        string_data=[f"item-{i}" for i in range(n)],
        int_data_with_nulls=[i if i % 10 != 0 else None for i in range(n)],
        float_data_with_nulls=[float(i) if i % 10 != 0 else None for i in range(n)],
        string_data_with_nulls=[f"item-{i}" if i % 10 != 0 else None for i in range(n)],
        nested_list_data=[[i, i + 1, i + 2] for i in range(n // 3)],
        struct_data=[{"x": i, "y": float(i), "z": f"s{i}"} for i in range(n // 3)],
        struct_prim_data=[{"a": i, "b": float(i), "c": i * 2} for i in range(n)],
    )


DATA = {n: _make_data(n) for n in SIZES}

struct_prim_type = ma.struct(
    [
        ma.field("a", ma.int64(), True, {}),
        ma.field("b", ma.float64(), True, {}),
        ma.field("c", ma.int64(), True, {}),
    ]
)
struct_prim_pa_type = pa.struct(
    [
        ("a", pa.int64()),
        ("b", pa.float64()),
        ("c", pa.int64()),
    ]
)


@pytest.fixture(params=SIZES, ids=[f"n={n}" for n in SIZES])
def n(request):
    return request.param


# ── int64 ──────────────────────────────────────────────────────────────────


def test_marrow_int64(benchmark, n):
    benchmark(ma.array, DATA[n]["int_data"], type=ma.int64())


def test_pyarrow_int64(benchmark, n):
    benchmark(pa.array, DATA[n]["int_data"], type=pa.int64())


def test_marrow_int64_nulls(benchmark, n):
    benchmark(ma.array, DATA[n]["int_data_with_nulls"], type=ma.int64())


def test_pyarrow_int64_nulls(benchmark, n):
    benchmark(pa.array, DATA[n]["int_data_with_nulls"], type=pa.int64())


# ── float64 ────────────────────────────────────────────────────────────────


def test_marrow_float64(benchmark, n):
    benchmark(ma.array, DATA[n]["float_data"], type=ma.float64())


def test_pyarrow_float64(benchmark, n):
    benchmark(pa.array, DATA[n]["float_data"], type=pa.float64())


def test_marrow_float64_nulls(benchmark, n):
    benchmark(ma.array, DATA[n]["float_data_with_nulls"], type=ma.float64())


def test_pyarrow_float64_nulls(benchmark, n):
    benchmark(pa.array, DATA[n]["float_data_with_nulls"], type=pa.float64())


# ── string ─────────────────────────────────────────────────────────────────


def test_marrow_string(benchmark, n):
    benchmark(ma.array, DATA[n]["string_data"], type=ma.string())


def test_pyarrow_string(benchmark, n):
    benchmark(pa.array, DATA[n]["string_data"], type=pa.string())


def test_marrow_string_nulls(benchmark, n):
    benchmark(ma.array, DATA[n]["string_data_with_nulls"], type=ma.string())


def test_pyarrow_string_nulls(benchmark, n):
    benchmark(pa.array, DATA[n]["string_data_with_nulls"], type=pa.string())


# ── infer (no explicit type) ──────────────────────────────────────────────


def test_marrow_int64_infer(benchmark, n):
    benchmark(ma.array, DATA[n]["int_data"])


def test_pyarrow_int64_infer(benchmark, n):
    benchmark(pa.array, DATA[n]["int_data"])


def test_marrow_string_infer(benchmark, n):
    benchmark(ma.array, DATA[n]["string_data"])


def test_pyarrow_string_infer(benchmark, n):
    benchmark(pa.array, DATA[n]["string_data"])


# ── nested list ────────────────────────────────────────────────────────────


def test_marrow_nested_list(benchmark, n):
    benchmark(ma.array, DATA[n]["nested_list_data"])


def test_pyarrow_nested_list(benchmark, n):
    benchmark(pa.array, DATA[n]["nested_list_data"])


# ── struct ─────────────────────────────────────────────────────────────────


def test_marrow_struct(benchmark, n):
    benchmark(ma.array, DATA[n]["struct_data"])


def test_pyarrow_struct(benchmark, n):
    benchmark(pa.array, DATA[n]["struct_data"])


# ── struct (primitive fields only) ────────────────────────────────────────


def test_marrow_struct_primitives(benchmark, n):
    benchmark(ma.array, DATA[n]["struct_prim_data"], type=struct_prim_type)


def test_pyarrow_struct_primitives(benchmark, n):
    benchmark(pa.array, DATA[n]["struct_prim_data"], type=struct_prim_pa_type)


# ── struct (primitive fields, infer) ──────────────────────────────────────


def test_marrow_struct_primitives_infer(benchmark, n):
    benchmark(ma.array, DATA[n]["struct_prim_data"])


def test_pyarrow_struct_primitives_infer(benchmark, n):
    benchmark(pa.array, DATA[n]["struct_prim_data"])
