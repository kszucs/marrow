# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Does an inner join gain anything from running below a LEFT or SEMI join?

`(fact ⟕ dim1) ⋈ dim2` and `(fact ⋈ dim2) ⟕ dim1` answer alike; the second
joins the selective `dim2` first, so the LEFT join reads only what survives.
Both orders are written by hand, every join built on its dimension and run
with no rules, so the rows say what the reordering is worth and nothing else.

- `fact`: 1,000,000 rows, `f1` over 100,000 keys and `f2` over 10,000.
- `dim1`: 100,000 unique keys — every fact row matches — and two payload
  columns.
- `dim2` keeps 1% of the fact rows (100 keys) or, as the control, 50% (5,000).

The two `noise` rows run one plan twice: their spread is this run's floor.
"""

from std.benchmark import BenchMetric, keep

from ...utils.testing import Benchmark
from ...arrays import DynArray
from ...builders import array
from ...dtypes import int64
from ...tabular import record_batch
from ...kernels.join import (
    BUILD_RIGHT,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_SEMI,
    JoinKind,
)
from ..builders import table
from ..logical import DynRelation


comptime FACT_ROWS = 1_000_000
comptime DIM1_ROWS = 100_000
comptime SELECTIVE = 100
comptime HALF = 5_000


def _table(
    names: List[String], rows: Int, modulus: List[Int]
) raises -> DynRelation:
    """`rows` rows, column `j` holding `i % modulus[j]`."""
    var columns = List[DynArray](capacity=len(names))
    for j in range(len(names)):
        var values = List[Optional[Int]](capacity=rows)
        for i in range(rows):
            values.append(i % modulus[j])
        columns.append(array(values^, int64).to_dyn())
    return table(record_batch(columns^, names=names.copy()))


def _fact() raises -> DynRelation:
    return _table(["f1", "f2"], FACT_ROWS, [DIM1_ROWS, 10_000])


def _dim1() raises -> DynRelation:
    return _table(["d1", "p1", "p2"], DIM1_ROWS, [DIM1_ROWS, 7, 13])


def _dim2(keys: Int) raises -> DynRelation:
    return _table(["k2"], keys, [keys])


def _written(kind: JoinKind, keys: Int) raises -> DynRelation:
    """`(fact <kind> dim1) ⋈ dim2`: `f2` is column 1 either way."""
    return (
        _fact()
        .join(_dim1(), [0], [0], kind, BUILD_RIGHT)
        .join(_dim2(keys), [1], [0], JOIN_INNER, BUILD_RIGHT)
    )


def _moved(kind: JoinKind, keys: Int) raises -> DynRelation:
    """`(fact ⋈ dim2) <kind> dim1`."""
    return (
        _fact()
        .join(_dim2(keys), [1], [0], JOIN_INNER, BUILD_RIGHT)
        .join(_dim1(), [0], [0], kind, BUILD_RIGHT)
    )


def _run(mut b: Benchmark, plan: DynRelation) raises:
    b.throughput(BenchMetric.elements, FACT_ROWS)

    @always_inline
    def call() raises {imm}:
        keep(plan.execute().num_rows())

    b.iter(call)
    keep(plan)


def bench_attach_left_written_1pct(mut b: Benchmark) raises:
    _run(b, _written(JOIN_LEFT, SELECTIVE))


def bench_attach_left_moved_1pct(mut b: Benchmark) raises:
    _run(b, _moved(JOIN_LEFT, SELECTIVE))


def bench_attach_left_written_50pct(mut b: Benchmark) raises:
    _run(b, _written(JOIN_LEFT, HALF))


def bench_attach_left_moved_50pct(mut b: Benchmark) raises:
    _run(b, _moved(JOIN_LEFT, HALF))


def bench_attach_semi_written_1pct(mut b: Benchmark) raises:
    _run(b, _written(JOIN_SEMI, SELECTIVE))


def bench_attach_semi_moved_1pct(mut b: Benchmark) raises:
    _run(b, _moved(JOIN_SEMI, SELECTIVE))


def bench_attach_noise_a(mut b: Benchmark) raises:
    _run(b, _moved(JOIN_LEFT, HALF))


def bench_attach_noise_b(mut b: Benchmark) raises:
    _run(b, _moved(JOIN_LEFT, HALF))
