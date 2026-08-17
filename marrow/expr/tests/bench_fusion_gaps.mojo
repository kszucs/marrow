"""What the string breakers actually cost (Q7.1, B27).

**Read the unit on every row before comparing two numbers.** pytest-benchmark
scales each benchmark independently, so one row reads `Name (time in ns)`, the
next `(time in us)` and the next `(time in ms)`. Comparing the bare figures
across rows reports a 25x speedup where there is a 40x slowdown -- which is
exactly what happened when B27 was first filed, and the filed conclusion was the
opposite of the truth. Strip the ANSI colour codes before reading the table, then compare only
rows whose `Name (time in ...)` header shows the same unit.
"""

from std.benchmark import BenchMetric, keep

from ...utils.testing import Benchmark
from ...builders import StringBuilder, Int32Builder, Int64Builder
from ...dtypes import int64, string, int32
from ...tabular import record_batch, RecordBatch
from ...buffers import Buffer
from ...views import apply
from .test_a1_spike import SpikeColumn
from ...expr.values import Coalesce
from ...expr.builders import col, lit
from ...expr.core import into_array
from ...kernels.string import LengthKernel
from ...arrays import StringArray


def _strings(n: Int) raises -> RecordBatch:
    """`n` short strings of varying length, so the length column is not
    constant and the offsets are genuinely read."""
    var b = StringBuilder(n)
    for i in range(n):
        b.append(String("row", i))
    return record_batch([b.finish().to_dyn()], names=["s"])


def _bench_len_only(mut bm: Benchmark, n: Int) raises:
    var batch = _strings(n)
    bm.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(into_array(col("s", string).length().execute(batch), n).length())

    bm.iter(call)
    keep(batch)


def _bench_len_plus_one(mut bm: Benchmark, n: Int) raises:
    var batch = _strings(n)
    bm.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(
            into_array(
                (col("s", string).length() + lit(1, int32)).execute(batch), n
            ).length()
        )

    bm.iter(call)
    keep(batch)


def bench_fusion_len_only_1m(mut b: Benchmark) raises:
    _bench_len_only(b, 1_000_000)


def bench_fusion_len_plus_one_1m(mut b: Benchmark) raises:
    _bench_len_plus_one(b, 1_000_000)


# ---------------------------------------------------------------------------
# B27 probes — split the 69 ms three ways.
# ---------------------------------------------------------------------------


def bench_b27_probe_bare_column_1m(mut b: Benchmark) raises:
    """Just reading the column through the expression layer."""
    var batch = _strings(1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(into_array(col("s", string).execute(batch), 1_000_000).length())

    b.iter(call)
    keep(batch)


def bench_b27_probe_kernel_only_1m(mut b: Benchmark) raises:
    """`LengthKernel.dispatch` on the array directly, no expression layer."""
    var batch = _strings(1_000_000)
    var arr = batch.columns[0].copy()
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(LengthKernel.dispatch(arr).length())

    b.iter(call)
    keep(arr)
    keep(batch)


def bench_b27_probe_kernel_typed_1m(mut b: Benchmark) raises:
    """`LengthKernel.apply` on the *typed* array — no variant dispatch.

    Against `..._kernel_only_1m` (which calls `dispatch`) this splits the cost
    between the erasure and the loop body.
    """
    var batch = _strings(1_000_000)
    var arr = batch.columns[0].as_string().copy()
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(LengthKernel.apply(arr)))

    b.iter(call)
    keep(arr)
    keep(batch)


def bench_b27_probe_two_lengths_1m(mut b: Benchmark) raises:
    """`s.len() + s.len()` — two Breaker stages, so `prepare` must run
    `LengthKernel` twice. If this lands near the one-length fused case rather
    than near double it, the kernel is not what the fused path is paying for."""
    var batch = _strings(1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(
            into_array(
                (col("s", string).length() + col("s", string).length()).execute(
                    batch
                ),
                1_000_000,
            ).length()
        )

    b.iter(call)
    keep(batch)


def bench_b27_probe_raw_copy_1m(mut b: Benchmark) raises:
    """Scale reference: an O(1) column copy, no computation at all."""
    var batch = _strings(1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(batch.columns[0].copy().length())

    b.iter(call)
    keep(batch)


def bench_b27_probe_plain_fused_add_1m(mut b: Benchmark) raises:
    """`a + 1` over a plain int32 column — a fused pass with **no breaker**.

    The reference Q7.1 needs. `s.len() + 1` pays a breaker slot read per SIMD
    chunk on top of a fused pass; this is the same fused pass without one, so the
    difference is what fusing `StringLength` could actually recover.
    """
    var ib = Int32Builder(1_000_000)
    for i in range(1_000_000):
        ib.append(Int32(i))
    var batch = record_batch([ib.finish().to_dyn()], names=["a"])
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(
            into_array(
                (col("a", int32) + lit(1, int32)).execute(batch), 1_000_000
            ).length()
        )

    b.iter(call)
    keep(batch)


def bench_b28_probe_two_columns_1m(mut b: Benchmark) raises:
    """`a + a` — two column leaves, so two schema lookups per SIMD chunk.

    Against `a + 1` (one lookup) this isolates the per-chunk column resolution
    from everything else in the lane.
    """
    var ib = Int32Builder(1_000_000)
    for i in range(1_000_000):
        ib.append(Int32(i))
    var batch = record_batch([ib.finish().to_dyn()], names=["a"])
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(
            into_array(
                (col("a", int32) + col("a", int32)).execute(batch), 1_000_000
            ).length()
        )

    b.iter(call)
    keep(batch)


def bench_b28_probe_two_literals_1m(mut b: Benchmark) raises:
    """`1 + 1` broadcast — no column leaf at all, so no schema lookup.

    OutShape 0 means this evaluates one lane and splats, so it is not a fair
    throughput comparison; it is here to show the lane machinery itself is
    cheap when nothing resolves a column.
    """
    var ib = Int32Builder(1_000_000)
    for i in range(1_000_000):
        ib.append(Int32(i))
    var batch = record_batch([ib.finish().to_dyn()], names=["a"])
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(
            into_array(
                (col("a", int32) + (lit(1, int32) + lit(2, int32))).execute(
                    batch
                ),
                1_000_000,
            ).length()
        )

    b.iter(call)
    keep(batch)


def bench_b28_probe_hoisted_ideal_1m(mut b: Benchmark) raises:
    """What A1 is aiming at: `a + 1` with the typed view resolved **once**.

    Hand-written to stand in for a fused lane whose state is hoisted out of the
    loop -- the column is looked up, unwrapped and viewed before the pass, and
    the per-chunk body is just a load, an add and a store. Same `views.apply`
    driver the expression layer uses, so the only difference from
    `bench_b27_probe_plain_fused_add_1m` is where the resolution happens.

    This is the number A1 should be judged against.
    """
    var ib = Int32Builder(1_000_000)
    for i in range(1_000_000):
        ib.append(Int32(i))
    var batch = record_batch([ib.finish().to_dyn()], names=["a"])
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {
        imm batch,
    }:
        # Resolved once, outside the lane -- the whole point. `src` is an owned
        # O(1) ref-count bump rather than a `ref` into `batch`: `producer` is a
        # unified closure, and a capture that reaches through two closure layers
        # into an interior reference is not expressible.
        var src = batch.columns[0].as_int32().copy()
        var vals = src.values()
        var out = Buffer.alloc_uninit[int32.native](1_000_000)

        @always_inline
        def producer[
            W: Int
        ](i: Int) {imm vals,} -> SIMD[int32.native, W]:
            return vals.load[W](i) + SIMD[int32.native, W](1)

        apply[int32.native](out.view[int32.native](0, 1_000_000), producer)
        keep(out.to_immutable())

    b.iter(call)
    keep(batch)


def bench_a1_spike_state_lane_1m(mut b: Benchmark) raises:
    """A1's shape, end to end: `a + 1` where the column leaf's state is
    prepared once and the lane just loads from it.

    Compare against `bench_b27_probe_plain_fused_add_1m` (today, 2.00 ms) and
    `bench_b28_probe_hoisted_ideal_1m` (the floor, 65.7 us). If this lands near
    the floor, A1's design is validated on the metric that matters.
    """
    var ib = Int32Builder(1_000_000)
    for i in range(1_000_000):
        ib.append(Int32(i))
    var batch = record_batch([ib.finish().to_dyn()], names=["a"])
    var node = SpikeColumn(0)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        # `prepare` once, outside the lane -- this is the A1 protocol.
        var state = node.prepare(batch)
        var out = Buffer.alloc_uninit[int32.native](1_000_000)

        @always_inline
        def producer[W: Int](i: Int) {imm} -> SIMD[int32.native, W]:
            return node.vectorwise[W](state, i) + SIMD[int32.native, W](1)

        apply[int32.native](out.view[int32.native](0, 1_000_000), producer)
        keep(out.to_immutable())

    b.iter(call)
    # `keep(node)` is mandatory, not tidiness: ASAP destruction can free a
    # captured value after the closure is registered and before it runs, and the
    # "assignment was never used" warning on `node` is exactly the tell CLAUDE.md
    # records for a capture that was not made.
    keep(node)
    keep(batch)


def _cond_batch(n: Int) raises -> RecordBatch:
    """Two int64 columns, `b` all-null in a third of the rows so `coalesce`
    actually has to choose."""
    var ab = Int64Builder(n)
    var bb = Int64Builder(n)
    for i in range(n):
        ab.append(Int64(i))
        if i % 3 == 0:
            bb.append_null()
        else:
            bb.append(Int64(i * 2))
    return record_batch(
        [ab.finish().to_dyn(), bb.finish().to_dyn()], names=["a", "b"]
    )


def bench_fu7_coalesce_fused_1m(mut b: Benchmark) raises:
    """`coalesce(a, b) + 1` over 1M rows — a conditional breaker under a fused
    parent.

    The shape FU-7a was about: the driver asks the breaker for its `state` and
    for its validity, and both used to run the whole selection kernel, so a
    fused pass over `coalesce`/`nullif`/`case_when` did the work twice.
    """
    var batch = _cond_batch(1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(
            into_array(
                (
                    Coalesce(col("a", int64), col("b", int64)) + lit(1, int64)
                ).execute(batch),
                1_000_000,
            ).length()
        )

    b.iter(call)
    keep(batch)
