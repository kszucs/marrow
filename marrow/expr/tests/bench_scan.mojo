"""Benchmarks for the relational Parquet scan (`ParquetScanProcessor`).

Distinct from `parquet/tests/bench_parquet.mojo`, which benches `read_table` —
the all-row-groups path. This drives the *scan operator*, which decodes one row
group at a time, and so is the only thing that sees the parallelism the
streaming design gives up: `ParquetFile.read` dispatches over the
(row group x leaf) grid, and a per-group read shrinks that grid to (1 x leaf).

The `narrow` cases are the worst case for it (few columns => few grid slots =>
little to parallelize per group); `wide` is the best case.
"""

from std.benchmark import BenchMetric, keep
from std.python import Python

from ...testing import Benchmark
from ...dtypes import Field, int64, field
from ...schema import Schema, schema
from ...expr.relations import ParquetScan, AnyRelation


def _prepare(
    path: String,
    n: Int,
    ncols: Int,
    rgsize: Int,
    compression: String = "none",
) raises:
    """`ncols` int64 columns of `n` rows, in row groups of `rgsize`."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var cols = Python.dict()
    for c in range(ncols):
        cols["c" + String(c)] = pa.array(np.arange(n, dtype="int64") + c)
    pq.write_table(
        pa.table(cols), path, row_group_size=rgsize, compression=compression
    )


def _schema_of(ncols: Int) raises -> Schema:
    var fields = List[Field]()
    for c in range(ncols):
        fields.append(field("c" + String(c), int64))
    return schema(fields^)


def _bench_scan(
    mut b: Benchmark,
    n: Int,
    ncols: Int,
    rgsize: Int,
    compression: String = "none",
) raises:
    var path = String("/tmp/marrow_bench_scan.parquet")
    _prepare(path, n, ncols, rgsize, compression)
    var sch = _schema_of(ncols)
    b.throughput(BenchMetric.elements, n * ncols)

    @always_inline
    @parameter
    def call() raises:
        keep(
            AnyRelation(
                ParquetScan(path=path, schema=Schema(copy=sch))
            ).execute()
        )

    b.iter[call]()
    # Both captures must be kept alive across `iter` — ASAP destruction frees a
    # captured value as soon as its last *syntactic* use passes, which is before
    # the closure ever runs. Keeping only `sch` made every iteration fail
    # immediately and the benchmark reported 17,774 GElems/s.
    keep(sch)
    keep(path)


def bench_scan_narrow_1m_16rg(mut b: Benchmark) raises:
    """2 columns, 16 row groups — least to parallelize inside one group."""
    _bench_scan(b, 1_000_000, 2, 62_500)


def bench_scan_wide_1m_16rg(mut b: Benchmark) raises:
    """16 columns, 16 row groups — a full grid even within one group."""
    _bench_scan(b, 1_000_000, 16, 62_500)


def bench_scan_narrow_1m_1rg(mut b: Benchmark) raises:
    """2 columns, a single row group — per-group and whole-file are the same
    read, so this isolates the plumbing from the parallelism change."""
    _bench_scan(b, 1_000_000, 2, 1_000_000)


def bench_scan_narrow_snappy_1m_16rg(mut b: Benchmark) raises:
    """**The case the streaming design actually costs.** 2 columns and 16 row
    groups, Snappy-compressed so decode is CPU-bound rather than
    memcpy-bound: reading all groups at once dispatches 32 grid slots across
    every core, reading one group at a time dispatches 2. Uncompressed PLAIN
    int64 cannot show this — it is memory-bandwidth-bound, so the extra threads
    buy nothing there either way."""
    _bench_scan(b, 1_000_000, 2, 62_500, "snappy")


def bench_scan_narrow_snappy_1m_1rg(mut b: Benchmark) raises:
    """The control for the case above: same data and codec, one row group, so
    both designs issue the identical read."""
    _bench_scan(b, 1_000_000, 2, 1_000_000, "snappy")


def bench_scan_narrow_100k_16rg(mut b: Benchmark) raises:
    """Scaling probe: 1/10th the rows of `bench_scan_narrow_1m_16rg`."""
    _bench_scan(b, 100_000, 2, 6_250)


def bench_scan_narrow_10m_16rg(mut b: Benchmark) raises:
    """Scaling probe: 10x the rows of `bench_scan_narrow_1m_16rg`."""
    _bench_scan(b, 10_000_000, 2, 625_000)
