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

Larger 100M-row comparisons live in ``python/marrow/tests/bench_join_parallel.py``
where competitor runtimes (DuckDB/Polars) can be run at full parallelism.
"""

from std.benchmark import BenchMetric, keep

from ...arrays import PrimitiveArray, DynArray, StructArray
from ...builders import PrimitiveBuilder, Int64Builder
from ...dtypes import int64, Int64Type, struct_, Field
from ...kernels.join import JOIN_INNER, JOIN_ALL
from ...kernels.join import HashJoin, hash_join
from ...kernels.aggregate import Fold, SumFold
from ...kernels.core import Groups
from ...execution import ExecContext
from ...utils.testing import Benchmark


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
    var cols = List[DynArray]()
    cols.append(kb.finish().to_dyn())
    cols.append(vb.finish().to_dyn())
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
    def call() raises {imm}:
        var j = HashJoin()
        j.build(left, keys)
        keep(j.num_left_rows())

    b.iter(call)
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
    def call() raises {imm}:
        var r = j.probe(right, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter(call)
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
    def call() raises {imm}:
        var r = hash_join(left, right, keys, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter(call)
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


# ---------------------------------------------------------------------------
# morsel-probe matrix — the shape the plan layer actually produces
#
# `JoinOperator` (marrow/expr/physical.mojo) streams the *probe* side in
# 8192-row morsels against a build side that may be arbitrarily large. Every
# bench above probes with a single big batch, so none of them can see the cost
# of a per-call fan-out decision: they amortize one dispatch over 1M rows,
# where the plan layer pays it ~122 times over 8192 rows each.
#
# Thread count is an explicit axis here because the regression only appears
# when the context resolves to >1 worker: `parallel(1)` and `serial()` both
# take the serial path, so row `t1` is the pre-`auto()` default and the
# baseline every other row must beat.
# ---------------------------------------------------------------------------

comptime _MORSEL = 8192
"""Probe morsel size used by `JoinProcessor` — mirrored here so the kernel
bench reproduces the plan layer's call pattern exactly."""


def _ctx_for(threads: Int) -> ExecContext:
    """`serial()` for one worker, `parallel(n)` otherwise.

    `parallel(1)` would also resolve to one worker, but `serial()` is what the
    pre-`auto()` plan default constructed, so it is what the baseline row
    should measure.
    """
    if threads <= 1:
        return ExecContext.serial()
    else:
        return ExecContext.parallel(threads)


def _bench_probe_morsels(
    mut b: Benchmark, build_n: Int, probe_n: Int, threads: Int
) raises:
    """Build once, then probe `probe_n` rows in `_MORSEL`-row slices.

    This is the regressing shape: the build side is large enough to trip a
    build-side-row-count threshold, while each individual probe call is tiny.
    Throughput is reported over the total probe rows so the number is
    comparable with `_bench_probe`'s single-batch row.
    """
    var left = _make_struct(build_n)
    var right = _make_struct(probe_n)
    var keys = List[Int]()
    keys.append(0)
    var j = HashJoin(_ctx_for(threads))
    j.build(left, keys)
    b.throughput(BenchMetric.elements, probe_n)

    @always_inline
    def call() raises {imm}:
        var off = 0
        while off < probe_n:
            var n = min(_MORSEL, probe_n - off)
            var r = j.probe(right.slice(off, n), keys, JOIN_INNER, JOIN_ALL)
            keep(len(r))
            off += n

    b.iter(call)
    keep(left)
    keep(right)
    keep(keys)
    keep(j)


def _bench_probe_single(
    mut b: Benchmark, build_n: Int, probe_n: Int, threads: Int
) raises:
    """Build once, then probe all `probe_n` rows in one call.

    The counterpart to `_bench_probe_morsels`: same data, same build, same
    thread count, but one big probe instead of many small ones. Parallel
    should genuinely win here, which is what makes it the control that says
    "the parallel probe path itself is not the problem".
    """
    var left = _make_struct(build_n)
    var right = _make_struct(probe_n)
    var keys = List[Int]()
    keys.append(0)
    var j = HashJoin(_ctx_for(threads))
    j.build(left, keys)
    b.throughput(BenchMetric.elements, probe_n)

    @always_inline
    def call() raises {imm}:
        var r = j.probe(right, keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    b.iter(call)
    keep(left)
    keep(right)
    keep(keys)
    keep(j)


# --- the regressing shape: 1M build, 1M probe delivered in 8192-row morsels --


def bench_join_morsel_1m_t1(mut b: Benchmark) raises:
    _bench_probe_morsels(b, 1_000_000, 1_000_000, 1)


def bench_join_morsel_1m_t2(mut b: Benchmark) raises:
    _bench_probe_morsels(b, 1_000_000, 1_000_000, 2)


def bench_join_morsel_1m_t4(mut b: Benchmark) raises:
    _bench_probe_morsels(b, 1_000_000, 1_000_000, 4)


def bench_join_morsel_1m_t8(mut b: Benchmark) raises:
    _bench_probe_morsels(b, 1_000_000, 1_000_000, 8)


# --- same build, one big probe: parallel should win here ---------------------


def bench_join_single_1m_t1(mut b: Benchmark) raises:
    _bench_probe_single(b, 1_000_000, 1_000_000, 1)


def bench_join_single_1m_t2(mut b: Benchmark) raises:
    _bench_probe_single(b, 1_000_000, 1_000_000, 2)


def bench_join_single_1m_t4(mut b: Benchmark) raises:
    _bench_probe_single(b, 1_000_000, 1_000_000, 4)


def bench_join_single_1m_t8(mut b: Benchmark) raises:
    _bench_probe_single(b, 1_000_000, 1_000_000, 8)


# --- small build: must stay serial regardless of the worker budget -----------


def bench_join_morsel_10k_t1(mut b: Benchmark) raises:
    _bench_probe_morsels(b, 10_000, 10_000, 1)


def bench_join_morsel_10k_t8(mut b: Benchmark) raises:
    _bench_probe_morsels(b, 10_000, 10_000, 8)


# --- drift control -----------------------------------------------------------
#
# `SumFold` touches nothing in join.mojo / partition.mojo / hashtable.mojo,
# so this row cannot move for any reason attributable to the fix. Its delta is
# the per-batch drift that must be subtracted from every row above before a
# change is attributed to the code (CLAUDE.md: this box drifts up to ±8% per
# case and ~±5% per batch).


def bench_join_drift_control_sum_1m(mut b: Benchmark) raises:
    var n = 1_000_000
    var vb = Int64Builder(capacity=n)
    for i in range(n):
        vb.append(Scalar[int64.native](i))
    var vals = vb.finish()
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(
            Fold[SumFold, Int64Type]
            .grouped(Groups.single(n), vals.copy())
            .to_dyn()
        )

    b.iter(call)
    keep(vals)
