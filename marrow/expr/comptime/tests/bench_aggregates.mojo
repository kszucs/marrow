"""What fusion and erasure each cost, at the same aggregate.

Two comparisons, each holding everything but one variable fixed:

- **fusion** — `sum(qty * price)` grouped. The comptime lane folds lanes out of
  the fused product; the runtime lane must evaluate `qty * price` into a column
  first, then fold it.
- **erasure** — `count_distinct` grouped, which *neither* lane can fuse. Both
  buffer the same columns and call the same kernel, so the only difference left
  is whether the operand is an `EvalOperator[A]` and the kernel a direct call
  (comptime) or a `DynOperator` box and an `AggregateFn` pointer (runtime).

The second pair is the one that says whether typing the buffered operator was
worth it. The first says what fusion is worth, and is the number CLAUDE.md's
14.6x claim comes from.

Run with:
    pixi run -e dev pytest marrow/expr/comptime/tests/bench_aggregates.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ....builders import array
from ....dtypes import int64
from ....tabular import RecordBatch, record_batch
from ....utils.testing import Benchmark
from ...builders import col, lit, table
from ...logical import DynValue
from ...runtime.values import RuntimeValue, column


comptime ROWS = 100_000
comptime GROUPS = 100


def _batch(n: Int, groups: Int) raises -> RecordBatch:
    var g = List[Optional[Int]](capacity=n)
    var qty = List[Optional[Int]](capacity=n)
    var price = List[Optional[Int]](capacity=n)
    for i in range(n):
        g.append(i % groups)
        qty.append(i % 17)
        price.append(100 + (i % 23))
    return record_batch(
        [
            array(g^, int64).to_dyn(),
            array(qty^, int64).to_dyn(),
            array(price^, int64).to_dyn(),
        ],
        names=["g", "qty", "price"],
    )


# ---------------------------------------------------------------------------
# fusion — the same query, one lane folding lanes and the other a column
# ---------------------------------------------------------------------------
def bench_agg_fused_sum_of_product(mut b: Benchmark) raises:
    """`sum(qty * price)` grouped, fused: the product is never materialised."""
    var batch = _batch(ROWS, GROUPS)
    b.throughput(BenchMetric.elements, ROWS)

    @always_inline
    def call() raises {imm}:
        var plan = table(batch.copy()).aggregate(
            [(col("qty", int64) * col("price", int64)).sum().alias("r")],
            [col("g", int64)],
        )
        keep(plan.execute().num_rows())

    b.iter(call)
    keep(batch)


def bench_agg_runtime_sum_of_product(mut b: Benchmark) raises:
    """The same query by name: `qty * price` becomes a column first."""
    var batch = _batch(ROWS, GROUPS)
    b.throughput(BenchMetric.elements, ROWS)

    @always_inline
    def call() raises {imm}:
        var plan = table(batch.copy()).aggregate(
            [
                RuntimeValue("multiply", column("qty"), column("price"))
                .sum()
                .alias("r")
            ],
            [col("g")],
        )
        keep(plan.execute().num_rows())

    b.iter(call)
    keep(batch)


# ---------------------------------------------------------------------------
# erasure — the same aggregate, neither lane able to fuse it
# ---------------------------------------------------------------------------
def bench_agg_typed_distinct(mut b: Benchmark) raises:
    """`count_distinct` grouped through the comptime lane: buffered, but the
    operand is an `EvalOperator[A]` and the kernel a direct call."""
    var batch = _batch(ROWS, GROUPS)
    b.throughput(BenchMetric.elements, ROWS)

    @always_inline
    def call() raises {imm}:
        var plan = table(batch.copy()).aggregate(
            [col("qty", int64).count_distinct().alias("d")],
            [col("g", int64)],
        )
        keep(plan.execute().num_rows())

    b.iter(call)
    keep(batch)


def bench_agg_erased_distinct(mut b: Benchmark) raises:
    """The same aggregate through the runtime lane: a `DynOperator` box and an
    `AggregateFn` pointer. The delta against the case above is the price of
    erasure with buffering held constant."""
    var batch = _batch(ROWS, GROUPS)
    b.throughput(BenchMetric.elements, ROWS)

    @always_inline
    def call() raises {imm}:
        var plan = table(batch.copy()).aggregate(
            [column("qty").count_distinct().alias("d")], [col("g")]
        )
        keep(plan.execute().num_rows())

    b.iter(call)
    keep(batch)
