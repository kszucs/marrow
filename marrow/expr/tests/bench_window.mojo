# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Window functions, by input shape — where the cost of a frame actually is.

Run with:
    pixi run -e dev pytest marrow/expr/tests/bench_window.mojo --benchmark

Four shapes, and the difference between them is the whole measurement:

- **`cumulative`** — `SUM(v) OVER (ORDER BY o)`, SQL's default frame, which
  runs from the partition start to the current row's peer group. The frame
  grows with the row index, so a naive evaluation visits `n(n+1)/2` elements
  and the row-count tiers should scale as the *square* of `n`. This is the
  shape a running accumulator turns linear.
- **`cumulative_ties`** — the same sum over 16 ordering values, so rows form
  16 peer groups and there are only 16 distinct frames. It measures the reuse
  of a repeated frame, and it sorts a different, heavily tied key.
- **`sliding`** — the same sum under `ROWS BETWEEN 9 PRECEDING AND CURRENT
  ROW`. The frame is bounded, so the element visits are already `10n`; what
  remains is the per-row cost of evaluating a frame at all. It is the control
  that separates "the frames are too big" from "each frame is too expensive".
- **`row_number`** — no aggregate, no frame. It pays the same sort, the same
  boundary scan and the same scatter as the other two and nothing else, so it
  is the anchor: this machine drifts up to ~8% per case, and a batch that
  reads as a uniform regression should move this row too. Normalise against
  it before attributing a delta to the frame.

Every shape but `cumulative_ties` sorts the same input and writes back
through the same inverse permutation, so a `cumulative` row and a `sliding` row
of equal `n` differ in the frame only.
"""

from std.benchmark import BenchMetric, keep

from ...builders import Int64Builder
from ...dtypes import int64
from ...tabular import record_batch
from ...utils.testing import Benchmark
from ..builders import col, row_number, table
from ..logical import DynRelation


def _values(n: Int) raises -> List[Int64]:
    """`n` distinct values in scrambled order, for the summed column `v`."""
    var out = List[Int64](capacity=n)
    for i in range(n):
        out.append(Int64((i * 7919) % n))
    return out^


def _rows(n: Int, distinct: Int = 0) raises -> DynRelation:
    """`o` is the ordering key, `v` the summed column.

    `distinct` is how many values `o` takes: `0` means all `n` are distinct, so
    every row is its own peer group and no two rows share a frame. A smaller
    number makes peer groups of `n // distinct` rows, which is what an
    `ORDER BY date` over a year of rows looks like.
    """
    var o = Int64Builder(capacity=n)
    var v = Int64Builder(capacity=n)
    var vals = _values(n)
    for i in range(n):
        o.append(Int64(i if distinct == 0 else i % distinct))
        v.append(vals[i])
    return table(record_batch([o.finish(), v.finish()], names=["o", "v"]))


def _run(mut b: Benchmark, var plan: DynRelation, n: Int) raises:
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(plan.execute().num_rows())

    b.iter(call)
    keep(plan)


def _bench_cumulative(mut b: Benchmark, n: Int, distinct: Int = 0) raises:
    var plan = _rows(n, distinct).with_columns(
        ["s"], [col("v", int64).sum().over(order_by=[col("o", int64)])]
    )
    _run(b, plan^, n)


def bench_window_cumulative_1k(mut b: Benchmark) raises:
    _bench_cumulative(b, 1_000)


def bench_window_cumulative_4k(mut b: Benchmark) raises:
    _bench_cumulative(b, 4_000)


def bench_window_cumulative_16k(mut b: Benchmark) raises:
    _bench_cumulative(b, 16_000)


def bench_window_cumulative_ties_16k(mut b: Benchmark) raises:
    """16,000 rows over 16 ordering values — 1,000 rows to a peer group.

    Every row in a peer group has the same frame under `RANGE`, so this is the
    shape where recomputing per row is recomputing the same answer a thousand
    times.
    """
    _bench_cumulative(b, 16_000, distinct=16)


def _bench_sliding(mut b: Benchmark, n: Int) raises:
    var plan = _rows(n).with_columns(
        ["s"],
        [col("v", int64).sum().over(order_by=[col("o", int64)], rows=(-9, 0))],
    )
    _run(b, plan^, n)


def bench_window_sliding_1k(mut b: Benchmark) raises:
    _bench_sliding(b, 1_000)


def bench_window_sliding_16k(mut b: Benchmark) raises:
    _bench_sliding(b, 16_000)


def _bench_row_number(mut b: Benchmark, n: Int) raises:
    var plan = _rows(n).with_columns(
        ["rn"], [row_number().over(order_by=[col("o", int64)])]
    )
    _run(b, plan^, n)


def bench_window_row_number_1k(mut b: Benchmark) raises:
    _bench_row_number(b, 1_000)


def bench_window_row_number_16k(mut b: Benchmark) raises:
    _bench_row_number(b, 16_000)
