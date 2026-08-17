"""Benchmarks for the column hashing kernel.

Run with:
    pixi run bench_mojo -k bench_hashing
    pixi run pytest marrow/kernels/tests/bench_hashing.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...arrays import (
    PrimitiveArray,
    BoolArray,
    Int32Array,
    Int64Array,
    StringArray,
)
from ...builders import (
    PrimitiveBuilder,
    StringBuilder,
    BoolBuilder,
    Int32Builder,
    Int64Builder,
)
from ...dtypes import PrimitiveType, int32, int64, Int32Type, Int64Type
from ...kernels.hashing import (
    AHashKernel,
    HashKernel,
    RapidHashKernel,
    XxHashKernel,
)
from ...utils import AHash64, Hasher, RapidHash64, XxHash64
from ...utils.testing import Benchmark


def _make_int64(n: Int) raises -> Int64Array:
    var b = Int64Builder(capacity=n)
    for i in range(n):
        b.append(Scalar[int64.native](i))
    return b.finish()


def _make_int32(n: Int) raises -> Int32Array:
    var b = Int32Builder(capacity=n)
    for i in range(n):
        b.append(Scalar[int32.native](i))
    return b.finish()


def _make_bool(n: Int) raises -> BoolArray:
    var b = BoolBuilder(capacity=n)
    for i in range(n):
        b.append(Bool(i % 2 == 0))
    return b.finish()


# ---------------------------------------------------------------------------
# int64
# ---------------------------------------------------------------------------


def bench_rapidhash_int64_10k(mut b: Benchmark) raises:
    var keys = _make_int64(10_000)

    @always_inline
    def call() raises {imm}:
        keep(len(RapidHashKernel.apply(keys)))

    b.iter(call)
    keep(keys)


def bench_rapidhash_int64_100k(mut b: Benchmark) raises:
    var keys = _make_int64(100_000)

    @always_inline
    def call() raises {imm}:
        keep(len(RapidHashKernel.apply(keys)))

    b.iter(call)
    keep(keys)


def bench_rapidhash_int64_1m(mut b: Benchmark) raises:
    var keys = _make_int64(1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(RapidHashKernel.apply(keys)))

    b.iter(call)
    keep(keys)


# ---------------------------------------------------------------------------
# int32
# ---------------------------------------------------------------------------


def bench_rapidhash_int32_10k(mut b: Benchmark) raises:
    var keys = _make_int32(10_000)

    @always_inline
    def call() raises {imm}:
        keep(len(RapidHashKernel.apply(keys)))

    b.iter(call)
    keep(keys)


def bench_rapidhash_int32_100k(mut b: Benchmark) raises:
    var keys = _make_int32(100_000)

    @always_inline
    def call() raises {imm}:
        keep(len(RapidHashKernel.apply(keys)))

    b.iter(call)
    keep(keys)


def bench_rapidhash_int32_1m(mut b: Benchmark) raises:
    var keys = _make_int32(1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(RapidHashKernel.apply(keys)))

    b.iter(call)
    keep(keys)


# ---------------------------------------------------------------------------
# bool
# ---------------------------------------------------------------------------


def bench_rapidhash_bool_10k(mut b: Benchmark) raises:
    var keys = _make_bool(10_000)

    @always_inline
    def call() raises {imm}:
        keep(len(RapidHashKernel.apply(keys)))

    b.iter(call)
    keep(keys)


def bench_rapidhash_bool_100k(mut b: Benchmark) raises:
    var keys = _make_bool(100_000)

    @always_inline
    def call() raises {imm}:
        keep(len(RapidHashKernel.apply(keys)))

    b.iter(call)
    keep(keys)


def bench_rapidhash_bool_1m(mut b: Benchmark) raises:
    var keys = _make_bool(1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(RapidHashKernel.apply(keys)))

    b.iter(call)
    keep(keys)


# ---------------------------------------------------------------------------
# The three hashes, same kernel, same data
#
# `HashKernel[H]` resolves `H` at comptime, so each of these is the code a
# hand-written kernel for that one algorithm would generate — the comparison is
# between the *algorithms*, not between a specialised and a generic path.
#
# int64 is the numeric lane path (`H.hash_lanes`); string is the variable-length
# path (`H.hash`), which is scalar per element for all three.
# ---------------------------------------------------------------------------


def _make_strings(n: Int, width: Int) raises -> StringArray:
    var b = StringBuilder(capacity=n)
    for i in range(n):
        var s = String("key-")
        s += String(i)
        while s.byte_length() < width:
            s += "x"
        b.append(s)
    return b.finish()


def _bench_lanes[H: Hasher](mut b: Benchmark, n: Int) raises:
    var keys = _make_int64(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(HashKernel[H].apply(keys))

    b.iter(call)
    keep(len(keys))


def _bench_strings[H: Hasher](mut b: Benchmark, n: Int, width: Int) raises:
    var keys = _make_strings(n, width)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(HashKernel[H].apply(keys))

    b.iter(call)
    keep(len(keys))


# --- numeric lanes, 100k int64 ---


def bench_cmp_lanes_rapidhash_100k(mut b: Benchmark) raises:
    _bench_lanes[RapidHash64](b, 100_000)


def bench_cmp_lanes_xxhash_100k(mut b: Benchmark) raises:
    _bench_lanes[XxHash64](b, 100_000)


def bench_cmp_lanes_ahash_100k(mut b: Benchmark) raises:
    _bench_lanes[AHash64](b, 100_000)


# --- variable-length strings, 100k x 16 bytes (rapidhash's <= 16 branch) ---


def bench_cmp_str16_rapidhash_100k(mut b: Benchmark) raises:
    _bench_strings[RapidHash64](b, 100_000, 16)


def bench_cmp_str16_xxhash_100k(mut b: Benchmark) raises:
    _bench_strings[XxHash64](b, 100_000, 16)


def bench_cmp_str16_ahash_100k(mut b: Benchmark) raises:
    _bench_strings[AHash64](b, 100_000, 16)


# --- longer strings, 100k x 64 bytes (past every short-input fast path) ---


def bench_cmp_str64_rapidhash_100k(mut b: Benchmark) raises:
    _bench_strings[RapidHash64](b, 100_000, 64)


def bench_cmp_str64_xxhash_100k(mut b: Benchmark) raises:
    _bench_strings[XxHash64](b, 100_000, 64)


def bench_cmp_str64_ahash_100k(mut b: Benchmark) raises:
    _bench_strings[AHash64](b, 100_000, 64)
