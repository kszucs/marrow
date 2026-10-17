# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`RadixPartitioner.repartition`: the rows of several batches split by key hash.

What a parallel group-by's merge pays before it combines: one batch per
worker, each hashed, partitioned and gathered once. Rows are an int64 key and
`width - 1` int64 payload columns, eight batches of 128K rows, split 64 ways.
"""

from std.benchmark import BenchMetric, keep

from ...arrays import DynArray, StructArray
from ...builders import Int64Builder
from ...execution import ExecContext
from ...kernels.partition import RadixPartitioner
from ...tabular import record_batch
from ...utils.testing import Benchmark

comptime BATCHES = 8
comptime ROWS = 131_072


def _batches(width: Int) raises -> List[StructArray]:
    var out = List[StructArray]()
    for b in range(BATCHES):
        var cols = List[DynArray]()
        var names = List[String]()
        for c in range(width):
            var col = Int64Builder(capacity=ROWS)
            for i in range(ROWS):
                col.append(Int64(((b * ROWS + i) * 7919 + c) % 4_000_003))
            cols.append(col.finish())
            names.append(String("c", c))
        out.append(record_batch(cols^, names=names^).to_struct_array())
    return out^


def _bench_repartition(
    mut b: Benchmark, width: Int, var ctx: ExecContext
) raises:
    var batches = _batches(width)
    var keys: List[Int] = [0]
    b.throughput(BenchMetric.elements, BATCHES * ROWS)

    @always_inline
    def call() raises {imm batches, imm keys, imm ctx}:
        keep(len(RadixPartitioner(6, ctx.copy()).repartition(batches, keys)))

    b.iter(call)
    keep(batches)


def bench_repartition_1col_serial(mut b: Benchmark) raises:
    _bench_repartition(b, 1, ExecContext.serial())


def bench_repartition_4col_serial(mut b: Benchmark) raises:
    _bench_repartition(b, 4, ExecContext.serial())


def bench_repartition_4col_parallel(mut b: Benchmark) raises:
    _bench_repartition(b, 4, ExecContext.parallel(BATCHES))
