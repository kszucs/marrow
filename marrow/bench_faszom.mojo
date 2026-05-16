"""Benchmarks: compile-time kernel fusion vs. step-by-step dispatch.

Two expression groups:

3-op: (a + b) == (c + d)
  Fused  — 1 allocation, 1 memory pass
  Dispatch — 3 allocations, 3 memory passes

5-op: (a * b + c) == (d * a - b)
  Fused  — 1 allocation, 1 memory pass
  Dispatch — 5 allocations, 5 memory passes

A 1-op baseline (a == b) is included to confirm the expression-tree
wrapper adds no overhead vs. a direct dispatch call.

Two filter groups demonstrating AOT compiled query advantage:

Group A — simple predicate filter: WHERE a + b > c
  Fused  — 1 pass: predicate SIMD[bool,W] fed directly into compressed_store;
           no BoolArray, no bitmap pack/unpack, 0 intermediate arrays
  Dispatch — 3 passes: ab = a+b, mask = ab>c → BoolArray, filter
             1 intermediate arithmetic array + 1 BoolArray

Group B — compound predicate filter: WHERE a + b > c AND d + e < f
  Fused  — 1 pass: full 5-op compound predicate eval'd per SIMD block, mask
           fed directly into compressed_store; 0 intermediate arrays
  Dispatch — 6 passes: ab=a+b, gt=ab>c, de=d+e, lt=de<f, mask=gt AND lt, filter
             4 intermediate arithmetic arrays + 1 BoolArray (~900 MB at 100M)

At 100M elements with 50% selectivity:
  Group B dispatch materialises ~900 MB of intermediate arrays before any filter runs.

Run with: pixi run pytest marrow/bench_faszom.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import AnyArray, BoolArray, PrimitiveArray
from marrow.builders import arange
from marrow.dtypes import Int32Type
from marrow.faszom import (
    Add,
    AndExpr,
    Column,
    EqExpr,
    FilterExpr,
    GtExpr,
    LtExpr,
    Literal,
    Mul,
    Sub,
    execute,
)
from marrow.kernels.arithmetic import AddKernel, MulKernel, SubKernel
from marrow.kernels.boolean import AndKernel
from marrow.kernels.compare import EqKernel, GtKernel, LtKernel
from marrow.kernels.filter import filter as filter_kernel
from marrow.testing import BenchSuite, Benchmark


# ---------------------------------------------------------------------------
# 1-op baseline: a == b
# Fused and dispatch should be equivalent (both: 1 pass, 1 allocation).
# ---------------------------------------------------------------------------


def bench_fused_eq_1op_100k(mut b: Benchmark) raises:
    var a = arange[Int32Type](0, 100_000)
    var c = arange[Int32Type](0, 100_000)
    var expr = EqExpr(Column(a.copy()), Column(c.copy()))
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, 100_000))

    b.iter[call]()
    keep(a)
    keep(c)
    keep(expr)


def bench_dispatch_eq_1op_100k(mut b: Benchmark) raises:
    var a: AnyArray = arange[Int32Type](0, 100_000)
    var c: AnyArray = arange[Int32Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(EqKernel.dispatch(a, c))

    b.iter[call]()
    keep(a)
    keep(c)


# ---------------------------------------------------------------------------
# 3-op expression: (a + b) == (c + d)
# Fused: 1 allocation.  Dispatch: 3 allocations + 2 extra memory passes.
# ---------------------------------------------------------------------------


def bench_fused_add_eq_100k(mut b: Benchmark) raises:
    var a = arange[Int32Type](0, 100_000)
    var bv = arange[Int32Type](1, 100_001)
    var c = arange[Int32Type](2, 100_002)
    var d = arange[Int32Type](3, 100_003)
    var expr = EqExpr(
        Add(Column(a.copy()), Column(bv.copy())),
        Add(Column(c.copy()), Column(d.copy())),
    )
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, 100_000))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)
    keep(expr)


def bench_dispatch_add_eq_100k(mut b: Benchmark) raises:
    var a: AnyArray = arange[Int32Type](0, 100_000)
    var bv: AnyArray = arange[Int32Type](1, 100_001)
    var c: AnyArray = arange[Int32Type](2, 100_002)
    var d: AnyArray = arange[Int32Type](3, 100_003)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        var ab = AddKernel.dispatch(a, bv)
        var cd = AddKernel.dispatch(c, d)
        keep(EqKernel.dispatch(ab, cd))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)


def bench_fused_add_eq_1m(mut b: Benchmark) raises:
    var a = arange[Int32Type](0, 1_000_000)
    var bv = arange[Int32Type](1, 1_000_001)
    var c = arange[Int32Type](2, 1_000_002)
    var d = arange[Int32Type](3, 1_000_003)
    var expr = EqExpr(
        Add(Column(a.copy()), Column(bv.copy())),
        Add(Column(c.copy()), Column(d.copy())),
    )
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, 1_000_000))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)
    keep(expr)


def bench_dispatch_add_eq_1m(mut b: Benchmark) raises:
    var a: AnyArray = arange[Int32Type](0, 1_000_000)
    var bv: AnyArray = arange[Int32Type](1, 1_000_001)
    var c: AnyArray = arange[Int32Type](2, 1_000_002)
    var d: AnyArray = arange[Int32Type](3, 1_000_003)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        var ab = AddKernel.dispatch(a, bv)
        var cd = AddKernel.dispatch(c, d)
        keep(EqKernel.dispatch(ab, cd))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)


def bench_fused_add_eq_10m(mut b: Benchmark) raises:
    var a = arange[Int32Type](0, 10_000_000)
    var bv = arange[Int32Type](1, 10_000_001)
    var c = arange[Int32Type](2, 10_000_002)
    var d = arange[Int32Type](3, 10_000_003)
    var expr = EqExpr(
        Add(Column(a.copy()), Column(bv.copy())),
        Add(Column(c.copy()), Column(d.copy())),
    )
    b.throughput(BenchMetric.elements, 10_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, 10_000_000))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)
    keep(expr)


def bench_dispatch_add_eq_10m(mut b: Benchmark) raises:
    var a: AnyArray = arange[Int32Type](0, 10_000_000)
    var bv: AnyArray = arange[Int32Type](1, 10_000_001)
    var c: AnyArray = arange[Int32Type](2, 10_000_002)
    var d: AnyArray = arange[Int32Type](3, 10_000_003)
    b.throughput(BenchMetric.elements, 10_000_000)

    @always_inline
    @parameter
    def call() raises:
        var ab = AddKernel.dispatch(a, bv)
        var cd = AddKernel.dispatch(c, d)
        keep(EqKernel.dispatch(ab, cd))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)


def bench_fused_add_eq_100m(mut b: Benchmark) raises:
    var a = arange[Int32Type](0, 100_000_000)
    var bv = arange[Int32Type](1, 100_000_001)
    var c = arange[Int32Type](2, 100_000_002)
    var d = arange[Int32Type](3, 100_000_003)
    var expr = EqExpr(
        Add(Column(a.copy()), Column(bv.copy())),
        Add(Column(c.copy()), Column(d.copy())),
    )
    b.throughput(BenchMetric.elements, 100_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, 100_000_000))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)
    keep(expr)


def bench_dispatch_add_eq_100m(mut b: Benchmark) raises:
    var a: AnyArray = arange[Int32Type](0, 100_000_000)
    var bv: AnyArray = arange[Int32Type](1, 100_000_001)
    var c: AnyArray = arange[Int32Type](2, 100_000_002)
    var d: AnyArray = arange[Int32Type](3, 100_000_003)
    b.throughput(BenchMetric.elements, 100_000_000)

    @always_inline
    @parameter
    def call() raises:
        var ab = AddKernel.dispatch(a, bv)
        var cd = AddKernel.dispatch(c, d)
        keep(EqKernel.dispatch(ab, cd))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)


# ---------------------------------------------------------------------------
# 5-op expression: (a * b + c) == (d * a - b)
# Fused: 1 allocation.  Dispatch: 5 allocations + 4 extra memory passes.
# ---------------------------------------------------------------------------


def bench_fused_mul_add_eq_100k(mut b: Benchmark) raises:
    var a = arange[Int32Type](0, 100_000)
    var bv = arange[Int32Type](1, 100_001)
    var c = arange[Int32Type](2, 100_002)
    var d = arange[Int32Type](3, 100_003)
    var expr = EqExpr(
        Add(Mul(Column(a.copy()), Column(bv.copy())), Column(c.copy())),
        Sub(Mul(Column(d.copy()), Column(a.copy())), Column(bv.copy())),
    )
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, 100_000))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)
    keep(expr)


def bench_dispatch_mul_add_eq_100k(mut b: Benchmark) raises:
    var a: AnyArray = arange[Int32Type](0, 100_000)
    var bv: AnyArray = arange[Int32Type](1, 100_001)
    var c: AnyArray = arange[Int32Type](2, 100_002)
    var d: AnyArray = arange[Int32Type](3, 100_003)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        var ab = MulKernel.dispatch(a, bv)
        var ab_c = AddKernel.dispatch(ab, c)
        var da = MulKernel.dispatch(d, a)
        var da_b = SubKernel.dispatch(da, bv)
        keep(EqKernel.dispatch(ab_c, da_b))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)


def bench_fused_mul_add_eq_1m(mut b: Benchmark) raises:
    var a = arange[Int32Type](0, 1_000_000)
    var bv = arange[Int32Type](1, 1_000_001)
    var c = arange[Int32Type](2, 1_000_002)
    var d = arange[Int32Type](3, 1_000_003)
    var expr = EqExpr(
        Add(Mul(Column(a.copy()), Column(bv.copy())), Column(c.copy())),
        Sub(Mul(Column(d.copy()), Column(a.copy())), Column(bv.copy())),
    )
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, 1_000_000))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)
    keep(expr)


def bench_dispatch_mul_add_eq_1m(mut b: Benchmark) raises:
    var a: AnyArray = arange[Int32Type](0, 1_000_000)
    var bv: AnyArray = arange[Int32Type](1, 1_000_001)
    var c: AnyArray = arange[Int32Type](2, 1_000_002)
    var d: AnyArray = arange[Int32Type](3, 1_000_003)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        var ab = MulKernel.dispatch(a, bv)
        var ab_c = AddKernel.dispatch(ab, c)
        var da = MulKernel.dispatch(d, a)
        var da_b = SubKernel.dispatch(da, bv)
        keep(EqKernel.dispatch(ab_c, da_b))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)


def bench_fused_mul_add_eq_10m(mut b: Benchmark) raises:
    var a = arange[Int32Type](0, 10_000_000)
    var bv = arange[Int32Type](1, 10_000_001)
    var c = arange[Int32Type](2, 10_000_002)
    var d = arange[Int32Type](3, 10_000_003)
    var expr = EqExpr(
        Add(Mul(Column(a.copy()), Column(bv.copy())), Column(c.copy())),
        Sub(Mul(Column(d.copy()), Column(a.copy())), Column(bv.copy())),
    )
    b.throughput(BenchMetric.elements, 10_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, 10_000_000))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)
    keep(expr)


def bench_dispatch_mul_add_eq_10m(mut b: Benchmark) raises:
    var a: AnyArray = arange[Int32Type](0, 10_000_000)
    var bv: AnyArray = arange[Int32Type](1, 10_000_001)
    var c: AnyArray = arange[Int32Type](2, 10_000_002)
    var d: AnyArray = arange[Int32Type](3, 10_000_003)
    b.throughput(BenchMetric.elements, 10_000_000)

    @always_inline
    @parameter
    def call() raises:
        var ab = MulKernel.dispatch(a, bv)
        var ab_c = AddKernel.dispatch(ab, c)
        var da = MulKernel.dispatch(d, a)
        var da_b = SubKernel.dispatch(da, bv)
        keep(EqKernel.dispatch(ab_c, da_b))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)


def bench_fused_mul_add_eq_100m(mut b: Benchmark) raises:
    var a = arange[Int32Type](0, 100_000_000)
    var bv = arange[Int32Type](1, 100_000_001)
    var c = arange[Int32Type](2, 100_000_002)
    var d = arange[Int32Type](3, 100_000_003)
    var expr = EqExpr(
        Add(Mul(Column(a.copy()), Column(bv.copy())), Column(c.copy())),
        Sub(Mul(Column(d.copy()), Column(a.copy())), Column(bv.copy())),
    )
    b.throughput(BenchMetric.elements, 100_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, 100_000_000))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)
    keep(expr)


def bench_dispatch_mul_add_eq_100m(mut b: Benchmark) raises:
    var a: AnyArray = arange[Int32Type](0, 100_000_000)
    var bv: AnyArray = arange[Int32Type](1, 100_000_001)
    var c: AnyArray = arange[Int32Type](2, 100_000_002)
    var d: AnyArray = arange[Int32Type](3, 100_000_003)
    b.throughput(BenchMetric.elements, 100_000_000)

    @always_inline
    @parameter
    def call() raises:
        var ab = MulKernel.dispatch(a, bv)
        var ab_c = AddKernel.dispatch(ab, c)
        var da = MulKernel.dispatch(d, a)
        var da_b = SubKernel.dispatch(da, bv)
        keep(EqKernel.dispatch(ab_c, da_b))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)


# ---------------------------------------------------------------------------
# Group A — simple predicate filter: WHERE a + b > c
# Fused:    1 pass — predicate eval (SIMD[bool,W]) fed directly into
#           compressed_store; no BoolArray, no bitmap pack/unpack
# Dispatch: ab=a+b (pass1), mask=ab>c → BoolArray (pass2), filter (pass3)
#           — 1 intermediate arithmetic array + 1 BoolArray
# ---------------------------------------------------------------------------


def _bench_fused_filter_gt(mut b: Benchmark, n: Int) raises:
    var a = arange[Int32Type](0, n)
    var bv = arange[Int32Type](1, n + 1)
    var c = arange[Int32Type](n // 2, n // 2 + n)  # ~50% selectivity: a+b > c
    var data = arange[Int32Type](0, n)
    var expr = FilterExpr(
        data.copy(),
        GtExpr(Add(Column(a.copy()), Column(bv.copy())), Column(c.copy())),
    )
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, n))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(data)
    keep(expr)


def _bench_dispatch_filter_gt(mut b: Benchmark, n: Int) raises:
    var a: AnyArray = arange[Int32Type](0, n)
    var bv: AnyArray = arange[Int32Type](1, n + 1)
    var c: AnyArray = arange[Int32Type](n // 2, n // 2 + n)
    var data: AnyArray = arange[Int32Type](0, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        var ab = AddKernel.dispatch(a, bv)
        var mask = GtKernel.dispatch(ab, c)
        keep(filter_kernel(data, mask.as_bool().copy()))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(data)


def bench_fused_filter_gt_100k(mut b: Benchmark) raises:
    _bench_fused_filter_gt(b, 100_000)


def bench_dispatch_filter_gt_100k(mut b: Benchmark) raises:
    _bench_dispatch_filter_gt(b, 100_000)


def bench_fused_filter_gt_1m(mut b: Benchmark) raises:
    _bench_fused_filter_gt(b, 1_000_000)


def bench_dispatch_filter_gt_1m(mut b: Benchmark) raises:
    _bench_dispatch_filter_gt(b, 1_000_000)




# ---------------------------------------------------------------------------
# Group B — compound predicate filter: WHERE a + b > c AND d + e < f
# Fused:    1 pass — full compound predicate (5 ops) eval'd per SIMD block,
#           mask fed directly into compressed_store; no intermediate arrays,
#           no BoolArray, no bitmap pack/unpack
# Dispatch: ab=a+b, gt=ab>c, de=d+e, lt=de<f, mask=and_(gt,lt), filter
#           — 4 intermediate arithmetic arrays + 1 BoolArray
# Both conditions give exactly 50% selectivity; combined ~25%.
# ---------------------------------------------------------------------------


def _bench_fused_filter_compound(mut b: Benchmark, n: Int) raises:
    var a = arange[Int32Type](0, n)
    var bv = arange[Int32Type](1, n + 1)
    var c = arange[Int32Type](n // 2, n // 2 + n)      # ~50% for a+b > c
    var d = arange[Int32Type](0, n)
    var ev = arange[Int32Type](1, n + 1)
    var f = arange[Int32Type](n // 2 + 1, n // 2 + 1 + n)  # ~50% for d+e < f
    var data = arange[Int32Type](0, n)
    var expr = FilterExpr(
        data.copy(),
        AndExpr(
            GtExpr(Add(Column(a.copy()), Column(bv.copy())), Column(c.copy())),
            LtExpr(Add(Column(d.copy()), Column(ev.copy())), Column(f.copy())),
        ),
    )
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(execute(expr, n))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)
    keep(ev)
    keep(f)
    keep(data)
    keep(expr)


def _bench_dispatch_filter_compound(mut b: Benchmark, n: Int) raises:
    var a: AnyArray = arange[Int32Type](0, n)
    var bv: AnyArray = arange[Int32Type](1, n + 1)
    var c: AnyArray = arange[Int32Type](n // 2, n // 2 + n)
    var d: AnyArray = arange[Int32Type](0, n)
    var ev: AnyArray = arange[Int32Type](1, n + 1)
    var f: AnyArray = arange[Int32Type](n // 2 + 1, n // 2 + 1 + n)
    var data: AnyArray = arange[Int32Type](0, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        var ab = AddKernel.dispatch(a, bv)
        var gt = GtKernel.dispatch(ab, c)
        var de = AddKernel.dispatch(d, ev)
        var lt = LtKernel.dispatch(de, f)
        var mask = AndKernel.dispatch(gt, lt)
        keep(filter_kernel(data, mask.as_bool().copy()))

    b.iter[call]()
    keep(a)
    keep(bv)
    keep(c)
    keep(d)
    keep(ev)
    keep(f)
    keep(data)


def bench_fused_filter_compound_100k(mut b: Benchmark) raises:
    _bench_fused_filter_compound(b, 100_000)


def bench_dispatch_filter_compound_100k(mut b: Benchmark) raises:
    _bench_dispatch_filter_compound(b, 100_000)


def bench_fused_filter_compound_1m(mut b: Benchmark) raises:
    _bench_fused_filter_compound(b, 1_000_000)


def bench_dispatch_filter_compound_1m(mut b: Benchmark) raises:
    _bench_dispatch_filter_compound(b, 1_000_000)




def main() raises:
    BenchSuite.run[__functions_in_module()]()
