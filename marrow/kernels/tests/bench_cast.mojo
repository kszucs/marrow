"""Benchmarks for cast kernels.

Run with: pixi run -e dev pytest marrow/kernels/tests/bench_cast.mojo --benchmark

Covers the numeric SIMD path (widen / narrow, safe vs unsafe) and the
temporal reinterpret path. String parse/format is builder-based and benchmarked
against PyArrow/Polars in ``python/marrow/tests/bench_cast.py``.
"""

from std.benchmark import BenchMetric, keep

from ...arrays import BinaryArray, DynArray
from ...builders import arange, StringBuilder
from ...dtypes import (
    BinaryType,
    Int32Type,
    Int64Type,
    Float64Type,
    StringType,
    int8,
    int64,
    float64,
    string,
    timestamp,
    second,
    millisecond,
)
from ...kernels.cast import cast, BinaryLikeCast, NumericCast
from ...utils.testing import Benchmark


# ---------------------------------------------------------------------------
# Typed numeric cast — int32 → float64 (widen, lossless, safe)
# ---------------------------------------------------------------------------


def _bench_int32_to_float64(mut b: Benchmark, n: Int) raises:
    var src = arange[Int32Type](0, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(NumericCast.apply[Int32Type, Float64Type, safe=True](src)))

    b.iter(call)
    keep(src)


def bench_int32_to_float64_10k(mut b: Benchmark) raises:
    _bench_int32_to_float64(b, 10_000)


def bench_int32_to_float64_100k(mut b: Benchmark) raises:
    _bench_int32_to_float64(b, 100_000)


def bench_int32_to_float64_1m(mut b: Benchmark) raises:
    _bench_int32_to_float64(b, 1_000_000)


# ---------------------------------------------------------------------------
# Typed numeric cast — int64 → int32 (narrow, unsafe/truncating)
# ---------------------------------------------------------------------------


def _bench_int64_to_int32_unsafe(mut b: Benchmark, n: Int) raises:
    var src = arange[Int64Type](0, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(NumericCast.apply[Int64Type, Int32Type, safe=False](src)))

    b.iter(call)
    keep(src)


def bench_int64_to_int32_unsafe_10k(mut b: Benchmark) raises:
    _bench_int64_to_int32_unsafe(b, 10_000)


def bench_int64_to_int32_unsafe_100k(mut b: Benchmark) raises:
    _bench_int64_to_int32_unsafe(b, 100_000)


def bench_int64_to_int32_unsafe_1m(mut b: Benchmark) raises:
    _bench_int64_to_int32_unsafe(b, 1_000_000)


# ---------------------------------------------------------------------------
# Runtime dispatch — int64 → float64 via the type-erased router
# ---------------------------------------------------------------------------


def _bench_dispatch_int64_to_float64(mut b: Benchmark, n: Int) raises:
    var src: DynArray = arange[Int64Type](0, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(cast(src, float64, safe=True)))

    b.iter(call)
    keep(src)


def bench_dispatch_int64_to_float64_100k(mut b: Benchmark) raises:
    _bench_dispatch_int64_to_float64(b, 100_000)


def bench_dispatch_int64_to_float64_1m(mut b: Benchmark) raises:
    _bench_dispatch_int64_to_float64(b, 1_000_000)


# ---------------------------------------------------------------------------
# Temporal — timestamp[s] → timestamp[ms] (unit upscale)
# ---------------------------------------------------------------------------


def _bench_timestamp_upscale(mut b: Benchmark, n: Int) raises:
    var src = cast(DynArray(arange[Int64Type](0, n)), timestamp(second))
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(cast(src, timestamp(millisecond))))

    b.iter(call)
    keep(src)


def bench_timestamp_upscale_100k(mut b: Benchmark) raises:
    _bench_timestamp_upscale(b, 100_000)


def bench_timestamp_upscale_1m(mut b: Benchmark) raises:
    _bench_timestamp_upscale(b, 1_000_000)


# ---------------------------------------------------------------------------
# Binary → string — the ClickBench shape.
#
# Parquet `BYTE_ARRAY` columns arrive as `binary`, and the string kernels are
# bound on `StringLikeType`, so every string query spells `.cast(string)`.
# `BinaryLikeCast.apply` is a pure relabel when the offset widths match, which
# `binary` → `string` satisfies — so the *entire* cost of these rows is the
# `safe` UTF-8 validation guard. The three cases below separate that out:
#
#   binary → string, safe=True   validating   (the path ClickBench takes)
#   binary → string, safe=False  pure relabel (the floor)
#   string → string, safe=True   pure relabel (control: `bytes_to_text` is
#                                False, so the guard is compiled out entirely)
#
# safe=True minus safe=False *is* the validation cost.
# ---------------------------------------------------------------------------


def _url_binary(n: Int) raises -> BinaryArray:
    """`n` URL-shaped ASCII strings, relabelled to `binary` (a free relabel)."""
    var b = StringBuilder(n)
    for i in range(n):
        b.append(
            String(
                "http://example.com/path/segment/",
                i,
                "?query=value&other=",
                i * 7,
                "#fragment",
            )
        )
    return BinaryLikeCast.apply[StringType, BinaryType, False](b.finish())


def _text_binary(n: Int) raises -> BinaryArray:
    """`n` multi-byte UTF-8 strings, relabelled to `binary`.

    The ASCII fast path cannot carry these, so this row measures the
    non-ASCII branch rather than the branch ClickBench hits."""
    var b = StringBuilder(n)
    for i in range(n):
        b.append(String("Здравствуйте, мир — строка ", i, " ünïcødé"))
    return BinaryLikeCast.apply[StringType, BinaryType, False](b.finish())


def _bench_binary_to_string_safe(mut b: Benchmark, n: Int) raises:
    var src = _url_binary(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(BinaryLikeCast.apply[BinaryType, StringType, True](src)))

    b.iter(call)
    keep(src)


def bench_binary_to_string_safe_10k(mut b: Benchmark) raises:
    _bench_binary_to_string_safe(b, 10_000)


def bench_binary_to_string_safe_100k(mut b: Benchmark) raises:
    _bench_binary_to_string_safe(b, 100_000)


def bench_binary_to_string_safe_1m(mut b: Benchmark) raises:
    _bench_binary_to_string_safe(b, 1_000_000)


def _bench_binary_to_string_unsafe(mut b: Benchmark, n: Int) raises:
    var src = _url_binary(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(BinaryLikeCast.apply[BinaryType, StringType, False](src)))

    b.iter(call)
    keep(src)


def bench_binary_to_string_unsafe_10k(mut b: Benchmark) raises:
    _bench_binary_to_string_unsafe(b, 10_000)


def bench_binary_to_string_unsafe_100k(mut b: Benchmark) raises:
    _bench_binary_to_string_unsafe(b, 100_000)


def bench_binary_to_string_unsafe_1m(mut b: Benchmark) raises:
    _bench_binary_to_string_unsafe(b, 1_000_000)


def _bench_binary_to_string_utf8_safe(mut b: Benchmark, n: Int) raises:
    var src = _text_binary(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(BinaryLikeCast.apply[BinaryType, StringType, True](src)))

    b.iter(call)
    keep(src)


def bench_binary_to_string_utf8_safe_100k(mut b: Benchmark) raises:
    _bench_binary_to_string_utf8_safe(b, 100_000)


def _bench_string_to_string_relabel(mut b: Benchmark, n: Int) raises:
    """Control — `bytes_to_text` is False, so no guard is compiled in at all."""
    var src = _url_binary(n)
    var s = BinaryLikeCast.apply[BinaryType, StringType, False](src)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(BinaryLikeCast.apply[StringType, StringType, True](s)))

    b.iter(call)
    keep(s)
    keep(src)


def bench_string_to_string_relabel_100k(mut b: Benchmark) raises:
    _bench_string_to_string_relabel(b, 100_000)


def bench_string_to_string_relabel_1m(mut b: Benchmark) raises:
    _bench_string_to_string_relabel(b, 1_000_000)


# ---------------------------------------------------------------------------
# Runtime-dispatch equivalent — what `col("URL").cast(ma.string())` actually
# calls, so the row is comparable with the ClickBench end-to-end numbers.
# ---------------------------------------------------------------------------


def _bench_dispatch_binary_to_string(mut b: Benchmark, n: Int) raises:
    var src: DynArray = _url_binary(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(cast(src, string, safe=True)))

    b.iter(call)
    keep(src)


def bench_dispatch_binary_to_string_100k(mut b: Benchmark) raises:
    _bench_dispatch_binary_to_string(b, 100_000)


def bench_dispatch_binary_to_string_1m(mut b: Benchmark) raises:
    _bench_dispatch_binary_to_string(b, 1_000_000)
