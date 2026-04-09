"""Benchmarks for rapidhash CPU.

Run with:
    pixi run bench_mojo -k bench_hashing
    pixi run pytest marrow/kernels/tests/bench_hashing.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import PrimitiveArray, BoolArray
from marrow.builders import PrimitiveBuilder, BoolBuilder
from marrow.dtypes import PrimitiveType, int32, int64, Int32Type, Int64Type
from marrow.kernels.hashing import rapidhash
from marrow.testing import BenchSuite, Benchmark


def _make_int64(n: Int) raises -> PrimitiveArray[Int64Type]:
    var b = PrimitiveBuilder[Int64Type](capacity=n)
    for i in range(n):
        b.append(Scalar[int64.native](i))
    return b.finish()


def _make_int32(n: Int) raises -> PrimitiveArray[Int32Type]:
    var b = PrimitiveBuilder[Int32Type](capacity=n)
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
    @parameter
    def call() raises:
        keep(len(rapidhash[Int64Type](keys)))

    b.iter[call]()


def bench_rapidhash_int64_100k(mut b: Benchmark) raises:
    var keys = _make_int64(100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(rapidhash[Int64Type](keys)))

    b.iter[call]()


def bench_rapidhash_int64_1m(mut b: Benchmark) raises:
    var keys = _make_int64(1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(rapidhash[Int64Type](keys)))

    b.iter[call]()


# ---------------------------------------------------------------------------
# int32
# ---------------------------------------------------------------------------


def bench_rapidhash_int32_10k(mut b: Benchmark) raises:
    var keys = _make_int32(10_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(rapidhash[Int32Type](keys)))

    b.iter[call]()


def bench_rapidhash_int32_100k(mut b: Benchmark) raises:
    var keys = _make_int32(100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(rapidhash[Int32Type](keys)))

    b.iter[call]()


def bench_rapidhash_int32_1m(mut b: Benchmark) raises:
    var keys = _make_int32(1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(rapidhash[Int32Type](keys)))

    b.iter[call]()


# ---------------------------------------------------------------------------
# bool
# ---------------------------------------------------------------------------


def bench_rapidhash_bool_10k(mut b: Benchmark) raises:
    var keys = _make_bool(10_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(rapidhash(keys)))

    b.iter[call]()


def bench_rapidhash_bool_100k(mut b: Benchmark) raises:
    var keys = _make_bool(100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(rapidhash(keys)))

    b.iter[call]()


def bench_rapidhash_bool_1m(mut b: Benchmark) raises:
    var keys = _make_bool(1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(rapidhash(keys)))

    b.iter[call]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    BenchSuite.run[__functions_in_module()]()
