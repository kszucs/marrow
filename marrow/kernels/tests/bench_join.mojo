"""Benchmarks for the hash join kernel.

Covers three phases across sizes 10k–1M:
  - build  — build the hash table from the left side
  - probe  — probe + assemble (full join output)
  - full   — hash_join() end-to-end (build + probe + assemble)

Run with: pixi run pytest marrow/kernels/tests/bench_join.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import PrimitiveArray, AnyArray, StructArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int64, struct_, Field
from marrow.expr.relations import JOIN_INNER, JOIN_ALL
from marrow.kernels.join import HashJoin, hash_join
from marrow.testing import BenchSuite, Benchmark


def _make_struct(n: Int) raises -> StructArray:
    """Build a StructArray with columns (k: int64, v: int64), unique keys 0..n."""
    var kb = PrimitiveBuilder[Int64Type](capacity=n)
    var vb = PrimitiveBuilder[Int64Type](capacity=n)
    for i in range(n):
        kb.append(Scalar[int64.native](i))
        vb.append(Scalar[int64.native](i * 10))
    var cols = List[AnyArray]()
    cols.append(kb.finish().to_any())
    cols.append(vb.finish().to_any())
    return StructArray(
        dtype=struct_(Field("k", int64), Field("v", int64)),
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        children=cols^,
    )


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def bench_join_build_10k(mut b: Benchmark) raises:
    var left = _make_struct(10_000)
    var keys = List[Int](0)

    @always_inline
    @parameter
    def call() raises:
        var j = HashJoin()
        j.build(left, keys)
        keep(j.num_left_rows())

    b.iter[call]()


def bench_join_build_100k(mut b: Benchmark) raises:
    var left = _make_struct(100_000)
    var keys = List[Int](0)

    @always_inline
    @parameter
    def call() raises:
        var j = HashJoin()
        j.build(left, keys)
        keep(j.num_left_rows())

    b.iter[call]()


def bench_join_build_1m(mut b: Benchmark) raises:
    var left = _make_struct(1_000_000)
    var keys = List[Int](0)

    @always_inline
    @parameter
    def call() raises:
        var j = HashJoin()
        j.build(left, keys)
        keep(j.num_left_rows())

    b.iter[call]()


# ---------------------------------------------------------------------------
# probe (build once, time probe + assemble)
# ---------------------------------------------------------------------------


def bench_join_probe_10k(mut b: Benchmark) raises:
    var left = _make_struct(10_000)
    var right = _make_struct(10_000)
    var keys = List[Int](0)
    var j = HashJoin()
    j.build(left, keys)

    @always_inline
    @parameter
    def call() raises:
        var r = j.probe(right, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter[call]()


def bench_join_probe_100k(mut b: Benchmark) raises:
    var left = _make_struct(100_000)
    var right = _make_struct(100_000)
    var keys = List[Int](0)
    var j = HashJoin()
    j.build(left, keys)

    @always_inline
    @parameter
    def call() raises:
        var r = j.probe(right, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter[call]()


def bench_join_probe_1m(mut b: Benchmark) raises:
    var left = _make_struct(1_000_000)
    var right = _make_struct(1_000_000)
    var keys = List[Int](0)
    var j = HashJoin()
    j.build(left, keys)

    @always_inline
    @parameter
    def call() raises:
        var r = j.probe(right, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter[call]()


# ---------------------------------------------------------------------------
# full (build + probe + assemble end-to-end)
# ---------------------------------------------------------------------------


def bench_join_full_10k(mut b: Benchmark) raises:
    var left = _make_struct(10_000)
    var right = _make_struct(10_000)
    var keys = List[Int](0)

    @always_inline
    @parameter
    def call() raises:
        var r = hash_join(left, right, keys, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter[call]()


def bench_join_full_100k(mut b: Benchmark) raises:
    var left = _make_struct(100_000)
    var right = _make_struct(100_000)
    var keys = List[Int](0)

    @always_inline
    @parameter
    def call() raises:
        var r = hash_join(left, right, keys, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter[call]()


def bench_join_full_1m(mut b: Benchmark) raises:
    var left = _make_struct(1_000_000)
    var right = _make_struct(1_000_000)
    var keys = List[Int](0)

    @always_inline
    @parameter
    def call() raises:
        var r = hash_join(left, right, keys, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter[call]()


def main() raises:
    BenchSuite.run[__functions_in_module()]()
