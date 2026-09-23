"""Benchmarks for the native Parquet reader.

Run with:
    pixi run pytest marrow/parquet/tests/bench_parquet.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep
from std.python import Python, PythonObject

from ...utils.testing import Benchmark
from ...parquet import ParquetFile, RowSelection, read_table, write_table
from ...parquet.reader import Coverage


def _corpus(n: Int) raises -> PythonObject:
    """The three-column table the read benchmarks share.

    `_prepare_dict` builds its own: it needs low cardinality so pyarrow keeps
    RLE_DICTIONARY, which is the whole point of that corpus."""
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    return pa.table(
        Python.dict(
            a=pa.array(np.arange(n, dtype="int64")),
            b=pa.array(np.arange(n, dtype="float64")),
            c=pa.array(np.arange(n, dtype="int32")),
        )
    )


def _pattern(rows: Int, keep_every: Int = 1, first: Int = -1) -> List[Bool]:
    """Every `keep_every`-th row of the first `first` (all, when negative)."""
    var head = rows if first < 0 else first
    var f = List[Bool](capacity=rows)
    for i in range(rows):
        f.append(i < head and i % keep_every == 0)
    return f^


def _prepare(path: String, n: Int, compression: String) raises:
    var pq = Python.import_module("pyarrow.parquet")
    pq.write_table(_corpus(n), path, compression=compression)


def _prepare_dict(path: String, n: Int) raises:
    """Low-cardinality columns so PyArrow keeps RLE_DICTIONARY throughout —
    exercises the SIMD bit-unpack + dictionary gather path."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var idx = np.arange(n)
    var tbl = pa.table(
        Python.dict(
            a=pa.array(idx % 1000),
            b=pa.array((idx % 777).astype("float64")),
            c=pa.array((idx % 333).astype("int32")),
        )
    )
    pq.write_table(tbl, path, compression="none")


def _bench_read(mut b: Benchmark, n: Int, compression: String) raises:
    var path = String("/tmp/marrow_bench_read.parquet")
    _prepare(path, n, compression)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(read_table(path))

    b.iter(call)


def bench_read_snappy_100k(mut b: Benchmark) raises:
    _bench_read(b, 100_000, "snappy")


def bench_read_snappy_1m(mut b: Benchmark) raises:
    _bench_read(b, 1_000_000, "snappy")


def bench_read_uncompressed_1m(mut b: Benchmark) raises:
    _bench_read(b, 1_000_000, "none")


def bench_read_dict_1m(mut b: Benchmark) raises:
    var path = String("/tmp/marrow_bench_dict.parquet")
    var n = 1_000_000
    _prepare_dict(path, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(read_table(path))

    b.iter(call)
    keep(path)  # keep the captured path alive through the whole benchmark


def _bench_read_small(
    mut b: Benchmark, path: String, compression: String
) raises:
    """Per-*read* set-up cost, isolated.

    A 1,000-row file read over and over: decoding three tiny pages is nearly
    free, so what is left is the fixed cost every `read_table` pays — mmap,
    footer parse, plan, and the codec handles the read allocates for its
    workers. Pair the `snappy` case with the `none` case below: the second
    never touches a compression library, so the difference between the two is
    the codec set-up, and `none` doubles as a drift control for the box."""
    var n = 1_000
    _prepare(path, n, compression)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(read_table(path))

    b.iter(call)
    keep(path)


def _prepare_groups(
    path: String, n: Int, rows_per_group: Int, compression: String = "none"
) raises:
    """Uncompressed, small pages, several row groups — the shape a pushed-down
    predicate reads. Small pages give the page index something to say, and
    `write_page_index` -- not pyarrow's default -- is what makes it say it:
    without one `read` cannot trim its fetch.

    `compression` defaults to none, which keeps the codec out of the
    measurement and is right for pricing the selection itself -- but it is
    also why an uncompressed row cannot show page skipping winning: the decode
    it skips is a memcpy. The snappy sibling is where the feature shows."""
    var pq = Python.import_module("pyarrow.parquet")
    pq.write_table(
        _corpus(n),
        path,
        compression="none",
        row_group_size=rows_per_group,
        data_page_size=64 * 1024,
        write_page_index=True,
    )


def _bench_read_selected(
    mut b: Benchmark,
    path: String,
    var pattern: List[Bool],
    compression: String = "none",
) raises:
    """1M rows under a `RowSelection`, the value a pushed-down predicate builds.

    One selection per row group is shared by all three leaves, so this is where
    the per-(row group x leaf) cost of a selection shows: the flags themselves,
    and the whole-selection answers the decoder asks before it walks a page.
    The flags are built once and the `RowSelection`s inside the timed body, so
    the construction pass is measured rather than hoisted out of it.

    `pattern` is one row group's flags, repeated per group, and the three
    shapes are not interchangeable. `all` skips no page and is the control;
    `eighth` skips none either, since every page holds a selected row, and
    prices the mask; only `prefix` leaves whole pages empty, so only it
    exercises the skip and the trimmed fetch.
    """
    var rows_per_group = len(pattern)
    var num_groups = 4
    var n = rows_per_group * num_groups
    _prepare_groups(path, n, rows_per_group, compression)

    var pf = ParquetFile(path)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        var sels = List[RowSelection](capacity=num_groups)
        for _ in range(num_groups):
            sels.append(RowSelection(pattern.copy()))
        keep(pf.read(row_selections=sels^).num_rows())

    b.iter(call)
    keep(pf)
    keep(pattern)


def _bench_row_selection(mut b: Benchmark, var flags: List[Bool]) raises:
    """One row group's worth of selection handling, with no I/O and no threads.

    `bench_read_selected_*` measures the whole read, and `ParquetFile.read`
    fans its decode out over a dozen workers — which makes it the wrong
    instrument for anything small on a contended machine. This asks the same
    question serially: build the selection a planner builds for one row group,
    then ask of it exactly what three leaf columns' `ColumnReader`s ask — a
    copy each (a refcount bump, since the flags are shared), `last_selected`
    once, then per page a count and, where the page is partly selected, the
    mask -- walked, since `consume_selected` walks it
    and an unread `mask` is a call `-O3` can delete. It also stops where
    `_run_selected` stops, past the last selected row; walking further would
    charge both sides of a comparison for work the reader never does.
    """
    var n = len(flags)
    var page_rows = 8_192
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        var sel = RowSelection(flags.copy())
        var acc = 0
        for _ in range(3):
            var mine = sel.copy()
            var last = mine.last_selected()
            var at = 0
            while at <= last:
                var nv = min(page_rows, n - at)
                if mine.covers(at, nv) == Coverage.SOME:
                    for ref r in mine.runs_in(at, nv):
                        acc += r[1] - r[0]
                at += nv
        keep(acc)

    b.iter(call)
    keep(flags)


def bench_row_selection_scattered_250k(mut b: Benchmark) raises:
    """One row in eight, spread over the whole group — every page is partly
    selected, so every page pays for a mask, and `last_selected` lands within
    a few rows of the end. Same shape as `bench_read_selected_eighth_1m`."""
    _bench_row_selection(b, _pattern(250_000, keep_every=8))


def bench_row_selection_prefix_250k(mut b: Benchmark) raises:
    """The first eighth selected and the rest not — the shape a `limit` or a
    sorted predicate prunes to, and the one where finding `last_selected`
    means walking back over seven eighths of the row group. Same shape as
    `bench_read_selected_prefix_1m`, so the serial and the threaded number
    describe one selection."""
    _bench_row_selection(b, _pattern(250_000, first=250_000 // 8))


def bench_read_selected_all_1m(mut b: Benchmark) raises:
    _bench_read_selected(
        b,
        "/tmp/marrow_bench_selected_all.parquet",
        _pattern(250_000),
    )


def bench_read_selected_eighth_1m(mut b: Benchmark) raises:
    _bench_read_selected(
        b,
        "/tmp/marrow_bench_selected_eighth.parquet",
        _pattern(250_000, keep_every=8),
    )


def bench_read_selected_prefix_1m(mut b: Benchmark) raises:
    """The `limit` shape: an eighth of each group at the front, the rest of its
    pages skipped without decoding and never fetched."""
    _bench_read_selected(
        b,
        "/tmp/marrow_bench_selected_prefix.parquet",
        _pattern(250_000, first=250_000 // 8),
    )


def bench_read_selected_prefix_snappy_1m(mut b: Benchmark) raises:
    """The same shape where skipping a page is worth something.

    Uncompressed, a skipped page saves a memcpy and the bookkeeping costs more
    than that, so the row above cannot show the feature working. Compressed, it
    saves a decompress -- this is the row that moves when page pruning
    improves."""
    _bench_read_selected(
        b,
        "/tmp/marrow_bench_selected_prefix_snappy.parquet",
        _pattern(250_000, first=250_000 // 8),
        "snappy",
    )


def bench_read_small_snappy(mut b: Benchmark) raises:
    _bench_read_small(b, "/tmp/marrow_bench_small_snappy.parquet", "snappy")


def bench_read_small_uncompressed(mut b: Benchmark) raises:
    _bench_read_small(b, "/tmp/marrow_bench_small_none.parquet", "none")
