"""Benchmarks for marrow cast kernels vs PyArrow and Polars.

Each cast case is benchmarked three ways — ``marrow.compute.cast``,
``pyarrow.compute.cast``, and Polars' ``Series.cast`` — under the same
``group`` so ``pytest-benchmark`` prints them side by side. Run with::

    pixi run -e bench pytest python/marrow/tests/bench_cast.py --benchmark-only

Numeric narrowing / float→int use ``safe=False`` (truncating) so all three
libraries do the same unchecked work; lossless and parse/format casts use the
default checked path.
"""

import pytest
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl

import marrow as ma
from marrow import compute as mc


SIZES = [10_000, 100_000, 1_000_000]


def _ints(n):
    return list(range(n))


def _floats(n):
    return [float(i) * 1.5 for i in range(n)]


def _int_strings(n):
    return [str(i) for i in range(n)]


def _bools(n):
    return [i % 2 == 0 for i in range(n)]


def _urls(n):
    """URL-shaped ASCII — the ClickBench `BYTE_ARRAY` column shape.

    Parquet hands these back as `binary`, and marrow's string kernels are bound
    on `StringLikeType`, so every string query casts. `binary` -> `string` is a
    pure relabel; `safe=True` is what makes it validate, so the safe/unsafe pair
    below isolates the validation cost against pyarrow and polars doing the
    same work."""
    return [
        f"http://example.com/path/segment/{i}?query=value&other={i * 7}#fragment"
        for i in range(n)
    ]


# (id, source array key, marrow target, pyarrow target, polars dtype, safe)
CASES = [
    ("int64->float64", "int64", ma.float64(), pa.float64(), pl.Float64, True),
    ("int32->int64", "int32", ma.int64(), pa.int64(), pl.Int64, True),
    ("float64->int32", "float64", ma.int32(), pa.int32(), pl.Int32, False),
    ("float64->float32", "float64", ma.float32(), pa.float32(), pl.Float32, True),
    ("int64->string", "int64", ma.string(), pa.string(), pl.String, True),
    ("string->int64", "int_string", ma.int64(), pa.int64(), pl.Int64, True),
    ("bool->int8", "bool", ma.int8(), pa.int8(), pl.Int8, True),
    ("binary->string", "binary", ma.string(), pa.string(), pl.String, True),
    (
        "binary->string-unsafe",
        "binary",
        ma.string(),
        pa.string(),
        pl.String,
        False,
    ),
]

_case_params = pytest.mark.parametrize(
    "src,ma_to,pa_to,pl_to,safe",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)


@pytest.fixture(params=SIZES, ids=[f"n={n}" for n in SIZES], scope="session")
def n(request):
    return request.param


@pytest.fixture(scope="session")
def ma_arrays(n):
    return {
        "int64": ma.array(_ints(n), type=ma.int64()),
        "int32": ma.array(_ints(n), type=ma.int32()),
        "float64": ma.array(_floats(n), type=ma.float64()),
        "int_string": ma.array(_int_strings(n), type=ma.string()),
        "bool": ma.array(_bools(n), type=ma.bool_()),
        "binary": mc.cast(ma.array(_urls(n), type=ma.string()), ma.binary()),
    }


@pytest.fixture(scope="session")
def pa_arrays(n):
    return {
        "int64": pa.array(_ints(n), type=pa.int64()),
        "int32": pa.array(_ints(n), type=pa.int32()),
        "float64": pa.array(_floats(n), type=pa.float64()),
        "int_string": pa.array(_int_strings(n), type=pa.string()),
        "bool": pa.array(_bools(n), type=pa.bool_()),
        "binary": pa.array(_urls(n), type=pa.binary()),
    }


@pytest.fixture(scope="session")
def pl_arrays(n):
    return {
        "int64": pl.Series(_ints(n), dtype=pl.Int64),
        "int32": pl.Series(_ints(n), dtype=pl.Int32),
        "float64": pl.Series(_floats(n), dtype=pl.Float64),
        "int_string": pl.Series(_int_strings(n), dtype=pl.String),
        "bool": pl.Series(_bools(n), dtype=pl.Boolean),
        "binary": pl.Series(_urls(n), dtype=pl.String).cast(pl.Binary),
    }


@pytest.mark.benchmark(group="cast")
@_case_params
def test_cast_marrow(benchmark, ma_arrays, src, ma_to, pa_to, pl_to, safe):
    arr = ma_arrays[src]
    benchmark(lambda: mc.cast(arr, ma_to, safe=safe))


@pytest.mark.benchmark(group="cast")
@_case_params
def test_cast_pyarrow(benchmark, pa_arrays, src, ma_to, pa_to, pl_to, safe):
    arr = pa_arrays[src]
    benchmark(lambda: pc.cast(arr, pa_to, safe=safe))


@pytest.mark.benchmark(group="cast")
@_case_params
def test_cast_polars(benchmark, pl_arrays, src, ma_to, pa_to, pl_to, safe):
    s = pl_arrays[src]
    benchmark(lambda: s.cast(pl_to, strict=safe))
