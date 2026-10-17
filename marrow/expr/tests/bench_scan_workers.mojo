# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""How a Parquet scan scales: a filter and a sum over `ROWS` rows, written
once as 8 row groups of 1,000,000 rows and once as 62 of 131,072, run by
`Driver.dynamic()` on 1 and 12 threads. Few large row groups are fewer
morsels than workers; many small ones are not."""

from std.benchmark import BenchMetric, keep
from std.os.path import join

from ...builders import array
from ...dtypes import Float64Type, Int64Type, float64, int64
from ...engine import Driver
from ...execution import ExecContext
from ..builders import col, lit, scan
from ..logical import DynRelation
from ...io import FileSink
from ...tabular import RecordBatch, Table, record_batch
from ...utils.testing import Benchmark, ScratchDir
from ...formats.parquet.writer import Compression, FileWriter


comptime ROWS = 8_000_000


def _batch() raises -> RecordBatch:
    var g = List[Int64](capacity=ROWS)
    var a = List[Int64](capacity=ROWS)
    var x = List[Float64](capacity=ROWS)
    for i in range(ROWS):
        g.append(Int64((i * 7_919) % 100_000))
        a.append(Int64(i % 1_009))
        x.append(Float64(i % 101) * 0.5)
    return record_batch(
        [
            array[Int64Type](g^, int64).to_dyn(),
            array[Int64Type](a^, int64).to_dyn(),
            array[Float64Type](x^, float64).to_dyn(),
        ],
        names=["g", "a", "x"],
    )


def _bench(mut b: Benchmark, threads: Int, row_group_size: Int) raises:
    """Time lowering and draining the scan on `threads`, over a file of
    `row_group_size`-row groups."""
    with ScratchDir() as dir:
        var path = join(dir, "scan.parquet")
        var batch = _batch()
        var writer = FileWriter(
            FileSink(path),
            batch.schema,
            Compression.SNAPPY,
            row_group_size=row_group_size,
        )
        writer.write(Table.from_batches(batch.schema, [batch.copy()]))
        writer.close()
        var plan = (
            scan(path, batch.schema)
            .filter(col("a", int64) > lit(500, int64))
            .aggregate(aggs=[col("x", float64).sum().alias("s")])
        )
        var ctx = ExecContext.parallel(threads)
        b.throughput(BenchMetric.elements, ROWS)
        b.extra_info("lib", String("w", threads))
        if threads == 12:
            b.extra_info("report", plan.profile(ctx).to_json())

        @always_inline
        def call() raises {imm}:
            var driver = Driver.dynamic(ctx)
            keep(len(driver.collect(plan.lower(driver), plan.schema())))

        b.iter(call)
        keep(plan)


def bench_w1_scan_rg1m(mut b: Benchmark) raises:
    _bench(b, 1, 1_000_000)


def bench_w12_scan_rg1m(mut b: Benchmark) raises:
    _bench(b, 12, 1_000_000)


def bench_w1_scan_rg128k(mut b: Benchmark) raises:
    _bench(b, 1, 131_072)


def bench_w12_scan_rg128k(mut b: Benchmark) raises:
    _bench(b, 12, 131_072)
