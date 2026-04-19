"""Benchmarks for the hash join kernel.

Covers three phases across sizes 10k–10M (and a gated 100M tier):
  - build           — build the hash table from the left side
  - probe           — probe + assemble (full join output)
  - full            — hash_join() end-to-end (build + probe + assemble)

Plus a build×probe shape matrix (small build / large probe, etc.) that
reflects realistic analytical workloads where one side is a fact table and
the other a dimension table.

Run with:
    pixi run pytest marrow/kernels/tests/bench_join.mojo --benchmark

Larger 100M-row comparisons live in ``python/tests/bench_join_parallel.py``
where competitor runtimes (DuckDB/Polars) can be run at full parallelism.
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import PrimitiveArray, AnyArray, StructArray
from marrow.builders import PrimitiveBuilder, Int64Builder
from marrow.dtypes import int64, Int64Type, struct_, Field
from marrow.expr.relations import JOIN_INNER, JOIN_ALL
from marrow.kernels.join import HashJoin, hash_join
from marrow.testing import BenchSuite, Benchmark


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_struct(n: Int, key_stride: Int = 1) raises -> StructArray:
    """Build a StructArray with columns (k: int64, v: int64).

    Keys are ``[0, key_stride, 2*key_stride, ...]``. With ``key_stride==1``
    keys are unique; with larger strides the join produces a Cartesian
    fan-out useful for probing multi-match cost.
    """
    var kb = Int64Builder(capacity=n)
    var vb = Int64Builder(capacity=n)
    for i in range(n):
        kb.append(Scalar[int64.native](i * key_stride))
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
# Shared helpers — parametric on size and shape
# ---------------------------------------------------------------------------


def _bench_build(mut b: Benchmark, n: Int) raises:
    """Measure only HashJoin.build() on ``n`` unique left rows."""
    var left = _make_struct(n)
    var keys = List[Int]()
    keys.append(0)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        var j = HashJoin()
        j.build(left, keys)
        keep(j.num_left_rows())

    b.iter[call]()
    keep(left)
    keep(keys)


def _bench_probe(mut b: Benchmark, build_n: Int, probe_n: Int) raises:
    """Measure HashJoin.probe() + assemble with a pre-built table.

    Throughput is reported in probe rows/sec — the probe side is the
    streaming input and generally dominates runtime for non-trivial
    build:probe ratios.
    """
    var left = _make_struct(build_n)
    var right = _make_struct(probe_n)
    var keys = List[Int]()
    keys.append(0)
    var j = HashJoin()
    j.build(left, keys)
    b.throughput(BenchMetric.elements, probe_n)

    @always_inline
    @parameter
    def call() raises:
        var r = j.probe(right, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter[call]()
    keep(left)
    keep(right)
    keep(keys)
    keep(j)


def _bench_full(mut b: Benchmark, build_n: Int, probe_n: Int) raises:
    """Measure hash_join() end-to-end: build + probe + assemble.

    Throughput is reported in total input rows/sec (build + probe).
    """
    var left = _make_struct(build_n)
    var right = _make_struct(probe_n)
    var keys = List[Int]()
    keys.append(0)
    b.throughput(BenchMetric.elements, build_n + probe_n)

    @always_inline
    @parameter
    def call() raises:
        var r = hash_join(left, right, keys, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter[call]()
    keep(left)
    keep(right)
    keep(keys)


# ---------------------------------------------------------------------------
# build — symmetric n
# ---------------------------------------------------------------------------


def bench_join_build_10k(mut b: Benchmark) raises:
    _bench_build(b, 10_000)


def bench_join_build_100k(mut b: Benchmark) raises:
    _bench_build(b, 100_000)


def bench_join_build_1m(mut b: Benchmark) raises:
    _bench_build(b, 1_000_000)


def bench_join_build_10m(mut b: Benchmark) raises:
    _bench_build(b, 10_000_000)


# ---------------------------------------------------------------------------
# probe — symmetric n (build once, time probe + assemble)
# ---------------------------------------------------------------------------


def bench_join_probe_10k(mut b: Benchmark) raises:
    _bench_probe(b, 10_000, 10_000)


def bench_join_probe_100k(mut b: Benchmark) raises:
    _bench_probe(b, 100_000, 100_000)


def bench_join_probe_1m(mut b: Benchmark) raises:
    _bench_probe(b, 1_000_000, 1_000_000)


def bench_join_probe_10m(mut b: Benchmark) raises:
    _bench_probe(b, 10_000_000, 10_000_000)


# ---------------------------------------------------------------------------
# full — symmetric n (end-to-end build + probe + assemble)
# ---------------------------------------------------------------------------


def bench_join_full_10k(mut b: Benchmark) raises:
    _bench_full(b, 10_000, 10_000)


def bench_join_full_100k(mut b: Benchmark) raises:
    _bench_full(b, 100_000, 100_000)


def bench_join_full_1m(mut b: Benchmark) raises:
    _bench_full(b, 1_000_000, 1_000_000)


def bench_join_full_10m(mut b: Benchmark) raises:
    _bench_full(b, 10_000_000, 10_000_000)


# ---------------------------------------------------------------------------
# shape matrix — asymmetric build × probe (fact / dimension workloads)
#
#   small_build × large_probe  — dimension join (classic broadcast join)
#   large_build × small_probe  — reversed, stresses the build phase
#
# Sizes chosen so both sides fit comfortably in memory for CI; the ratio is
# what matters for the algorithm choice (partition vs. no partition).
# ---------------------------------------------------------------------------


def bench_join_shape_100k_x_10m(mut b: Benchmark) raises:
    """100k-row build × 10M-row probe — broadcast dimension join."""
    _bench_full(b, 100_000, 10_000_000)


def bench_join_shape_10m_x_100k(mut b: Benchmark) raises:
    """10M-row build × 100k-row probe — reversed, large build."""
    _bench_full(b, 10_000_000, 100_000)


def bench_join_shape_1m_x_10m(mut b: Benchmark) raises:
    """1M-row build × 10M-row probe — 1:10 selectivity fan-in."""
    _bench_full(b, 1_000_000, 10_000_000)


def bench_join_shape_10m_x_1m(mut b: Benchmark) raises:
    """10M-row build × 1M-row probe — 10:1 fan-out."""
    _bench_full(b, 10_000_000, 1_000_000)


def main() raises:
    BenchSuite.run[__functions_in_module()]()
