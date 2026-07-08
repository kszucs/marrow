"""Benchmarks for the native Parquet reader.

Run with:
    pixi run pytest marrow/parquet/tests/bench_parquet.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep
from std.python import Python

from marrow.testing import BenchSuite, Benchmark
from marrow.parquet import read_table, write_table


def _prepare(path: String, n: Int, compression: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var tbl = pa.table(
        Python.dict(
            a=pa.array(np.arange(n, dtype="int64")),
            b=pa.array(np.arange(n, dtype="float64")),
            c=pa.array(np.arange(n, dtype="int32")),
        )
    )
    pq.write_table(tbl, path, compression=compression)


def _bench_read(mut b: Benchmark, n: Int, compression: String) raises:
    var path = String("/tmp/marrow_bench_read.parquet")
    _prepare(path, n, compression)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(read_table(path))

    b.iter[call]()


def bench_read_snappy_100k(mut b: Benchmark) raises:
    _bench_read(b, 100_000, "snappy")


def bench_read_snappy_1m(mut b: Benchmark) raises:
    _bench_read(b, 1_000_000, "snappy")


def bench_read_uncompressed_1m(mut b: Benchmark) raises:
    _bench_read(b, 1_000_000, "none")


def main() raises:
    BenchSuite.run[__functions_in_module()]()
