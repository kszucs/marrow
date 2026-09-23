"""Pruning has to *not read* the bytes it prunes, and only an instrumented
source can say so.

Every other page-skipping case in this package asserts the rows that come back.
That is necessary and it is not sufficient: a reader that decodes every page and
filters afterwards returns exactly the same rows, so those tests pass whether or
not a single byte was saved. The only way to tell the two apart is to watch the
I/O.

`ParquetFile` is generic over `ByteSource`, so the seam is already there —
`_Recorder` wraps a `BufferSource` and records every `(offset, length)` the reader
asks for. A skipped page is then a *provable* claim: its byte range, taken from
the file's own `OffsetIndex`, is disjoint from everything that was read.

The recording goes through an `ArcPointer` because `ByteSource.read_at` takes
`ref self` — the source is borrowed while the reader holds a span into it, so
the tally cannot live in a mutable field.
"""

from std.memory import ArcPointer
from std.os import remove
from std.python import Python, PythonObject
from std.testing import assert_equal, assert_false, assert_true

from ...parquet.format import (
    ColumnChunk,
    ColumnMetaData,
    OffsetIndex,
    PageLocation,
)
from ...parquet.reader import (
    LeafSet,
    ParquetFile,
    RowSelection,
    read_page_index,
    scan_ranges,
)
from ...io import ByteSource, Fetched, BufferSource


comptime Reads = ArcPointer[List[Tuple[Int, Int]]]
"""Every `(offset, length)` the reader asked for, in order."""


struct _Recorder(ByteSource):
    """A `BufferSource` that remembers what was read through it."""

    var _inner: BufferSource
    var _reads: Reads

    def __init__(out self, path: String, var reads: Reads) raises:
        self._inner = BufferSource(path)
        self._reads = reads^

    def size(self) -> Int:
        return self._inner.size()

    def read_at(
        ref self, offset: Int, length: Int
    ) raises -> Span[UInt8, origin_of(self)]:
        self._reads[].append((offset, length))
        return rebind[Span[UInt8, origin_of(self)]](
            self._inner.read_at(offset, length)
        )

    def read_ranges(ref self, ranges: List[Tuple[Int, Int]]) raises -> Fetched:
        # Every range of a batch is recorded individually, so the byte-level
        # assertions below read the same whether the reader asked one range at a
        # time or asked for all of them at once.
        for ref r in ranges:
            self._reads[].append((r[0], r[1]))
        return self._inner.read_ranges(ranges)


def _write_paged(
    path: String, rows: Int, page_rows: Int, group_rows: Int = -1
) raises:
    """`0..rows` ascending in pages of `page_rows`, one row group, page index.

    `write_batch_size` is what fixes the page length — parquet-cpp only checks
    whether a page is full at the end of each write batch, so `data_page_size`
    alone leaves anything under 1024 rows in a single page.
    """
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var builtins = Python.import_module("builtins")
    pq.write_table(
        pa.table(
            Python.dict(a=pa.array(builtins.range(rows), type=pa.int64()))
        ),
        path,
        row_group_size=rows if group_rows < 0 else group_rows,
        data_page_size=1,
        write_batch_size=page_rows,
        write_page_index=True,
        use_dictionary=False,
        compression="none",
    )


def _touched(reads: Reads, after: Int, start: Int, length: Int) -> Bool:
    """Whether any read *after the first `after`* overlaps `[start, start +
    length)`.

    Opening the file is itself a read — the footer tail — and for a file
    smaller than `_FOOTER_READ_SIZE` that tail is the whole file. So every
    question about what decoding fetched has to start counting from the point
    the file was open, which is what `after` is.
    """
    for i in range(after, len(reads[])):
        var lo = reads[][i][0]
        var hi = lo + reads[][i][1]
        if lo < start + length and start < hi:
            return True
    return False


def _page_ranges(path: String, group: Int = 0) raises -> List[Tuple[Int, Int]]:
    """Each data page's `(offset, compressed_size)` for column 0 of `group`."""
    var out = List[Tuple[Int, Int]]()
    var pi = read_page_index(path)
    ref oi = pi[group][0].offset_index.value()
    for ref loc in oi.page_locations:
        out.append((loc.offset, loc.compressed_page_size))
    return out^


def test_opening_a_file_reads_only_its_tail() raises:
    """Opening a file must cost a footer, not the file.

    `ByteSource` exists so the bytes can come from somewhere that is not a
    memory map, and for anywhere-but-local a read is a round trip. This used to
    ask for `read_at(0, size())` to find the footer — downloading the whole
    object to open it, which costs strictly more than every saving underneath
    it puts together.

    The fixture is deliberately large enough that a whole-file read is
    unmistakable: ~800 KB of data against a footer of a few hundred bytes.
    """
    var path = String("/tmp/marrow_pageio_footer.parquet")
    _write_paged(path, 100000, 10000)
    var total = BufferSource(path).size()
    assert_true(total > 400000, "the fixture must dwarf its own footer")

    var reads = Reads(List[Tuple[Int, Int]]())
    var f = ParquetFile[_Recorder, LeafSet.all()](_Recorder(path, reads.copy()))

    var fetched = 0
    for ref r in reads[]:
        fetched += r[1]
    assert_true(
        fetched <= 64 * 1024,
        "opening read " + String(fetched) + " bytes of " + String(total),
    )
    # One round trip, and it is the file's *tail*: a second read of the same
    # bytes is what a naive "measure then parse" does, and it costs a remote
    # source twice.
    assert_equal(len(reads[]), 1, "opening should cost one read")
    assert_equal(reads[][0][0] + reads[][0][1], total, "read the tail")
    assert_equal(f.num_row_groups(), 1)
    remove(path)


def test_only_the_selected_pages_are_fetched() raises:
    """The claim this file exists to make, and it now holds at both ends.

    Seven pages of ten rows, a selection of rows 25-34: pages 2 and 3 are
    fetched and the other five are not. The `OffsetIndex` gives both bounds —
    `first_row_index` says which page holds the first selected row, and
    `compressed_page_size` says where the page holding the last one ends.

    What is still fetched is anything *between* two selected pages, since the
    range is contiguous. A scattered selection therefore pays for its gaps;
    closing that needs a discontiguous span, and so does the dictionary case
    below.
    """
    var path = String("/tmp/marrow_pageio_skip.parquet")
    _write_paged(path, 70, 10)
    var ranges = _page_ranges(path)
    assert_equal(len(ranges), 7, "the fixture must write seven pages")

    var sels = List[RowSelection]()
    sels.append(_sel(70, [(25, 35)]))

    var reads = Reads(List[Tuple[Int, Int]]())
    var f = ParquetFile[_Recorder, LeafSet.all()](_Recorder(path, reads.copy()))
    var opened = len(reads[])
    var got = f.read(row_selections=Optional(sels^))
    assert_equal(got.num_rows(), 10, "only the selected rows come back")

    _assert_fetched(reads, opened, ranges, [2, 3])

    # **The values, not just the count.** The reader walks the whole chunk
    # logically while reading only inside the fetched segments, so its row
    # cursor has to start at the chunk and not at the first byte it was given.
    # When it did start at the first fetched byte, page 2 was treated as row 0
    # and this returned ten rows -- the wrong ten.
    ref c = got.to_batches()[0].columns[0].as_int64()
    for i in range(10):
        assert_equal(Int(c[i].value()), 25 + i, "row " + String(i))
    remove(path)


def test_a_dictionary_page_is_fetched_without_the_pages_around_it() raises:
    """A dictionary-encoded chunk fetches its dictionary and its selected
    pages, and nothing in between.

    The dictionary page is the chunk's first byte and every `RLE_DICTIONARY`
    page is unreadable without it, so it has to be fetched however late the
    selection starts — but it is its *own* range, not a licence to read from
    the chunk's start onward. Data pages 0 and 1 sit between it and the
    selection and are never touched.

    That the values come back correct is the other half of the proof: had the
    dictionary been left out, decoding these pages would not have produced the
    right strings, it would have failed or produced nonsense.
    """
    var path = String("/tmp/marrow_pageio_dict.parquet")
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var a = Python.list()
    for i in range(70):
        a.append(Python.str("k") + Python.str(i % 5))
    pq.write_table(
        pa.table(Python.dict(a=pa.array(a))),
        path,
        row_group_size=70,
        data_page_size=1,
        write_batch_size=10,
        write_page_index=True,
        use_dictionary=True,
        compression="none",
    )
    var ranges = _page_ranges(path)
    assert_equal(len(ranges), 7)

    var sels = List[RowSelection]()
    sels.append(_sel(70, [(25, 35)]))

    var reads = Reads(List[Tuple[Int, Int]]())
    var f = ParquetFile[_Recorder, LeafSet.all()](_Recorder(path, reads.copy()))
    var opened = len(reads[])
    var got = f.read(row_selections=Optional(sels^))
    assert_equal(got.num_rows(), 10)

    _assert_fetched(reads, opened, ranges, [2, 3])

    # The dictionary really was read, and really was used.
    ref c = got.to_batches()[0].columns[0].as_string()
    assert_equal(String(c[0].value()), "k0")
    assert_equal(String(c[9].value()), "k4")
    remove(path)


def _sel(n: Int, runs: List[Tuple[Int, Int]]) -> RowSelection:
    """A selection keeping each half-open `[start, end)` run and nothing else.
    """
    var keep = List[Bool](capacity=n)
    for i in range(n):
        var hit = False
        for ref r in runs:
            if r[0] <= i < r[1]:
                hit = True
                break
        keep.append(hit)
    return RowSelection(keep^)


def _assert_fetched(
    reads: Reads,
    after: Int,
    ranges: List[Tuple[Int, Int]],
    want: List[Int],
) raises:
    """Exactly the pages in `want` were fetched, and no others."""
    for p in range(len(ranges)):
        var expected = False
        for ref w in want:
            if w == p:
                expected = True
                break
        assert_equal(
            _touched(reads, after, ranges[p][0], ranges[p][1]),
            expected,
            "page " + String(p),
        )


def _page_reads(reads: Reads, after: Int, ranges: List[Tuple[Int, Int]]) -> Int:
    """How many reads landed in the column chunk's pages.

    Reads of the page index itself are excluded: `_chunk_offsets` fetches the
    `OffsetIndex` before any page can be located, so it is a real round trip
    but not one of the *data* fetches this is counting.
    """
    var lo = ranges[0][0]
    var hi = ranges[len(ranges) - 1][0] + ranges[len(ranges) - 1][1]
    var n = 0
    for i in range(after, len(reads[])):
        var at = reads[][i][0]
        if lo <= at < hi:
            n += 1
    return n


def test_a_scattered_selection_fetches_one_range_per_run() raises:
    """Disjoint runs of selected pages become disjoint reads, and adjacent
    ones are merged into a single read.

    Every other case here selects one contiguous run, so the merging in
    `_selected_ranges` and its ability to produce *several* ranges were both
    unexercised. Pages 1 and 2 are adjacent and must coalesce; page 5 is
    separated from them by two skipped pages and must not.
    """
    var path = String("/tmp/marrow_pageio_scatter.parquet")
    _write_paged(path, 70, 10)
    var ranges = _page_ranges(path)

    var sels = List[RowSelection]()
    sels.append(_sel(70, [(10, 30), (50, 60)]))

    var reads = Reads(List[Tuple[Int, Int]]())
    var f = ParquetFile[_Recorder, LeafSet.all()](_Recorder(path, reads.copy()))
    var opened = len(reads[])
    var got = f.read(row_selections=Optional(sels^))
    assert_equal(got.num_rows(), 30)

    # Two reads: pages 1+2 as one, page 5 as another. The count is the
    # assertion — three reads would mean the merge never happened, one would
    # mean the gap was fetched.
    assert_equal(
        _page_reads(reads, opened, ranges), 2, "one read per run of pages"
    )
    _assert_fetched(reads, opened, ranges, [1, 2, 5])

    ref c = got.to_batches()[0].columns[0].as_int64()
    assert_equal(Int(c[0].value()), 10)
    assert_equal(Int(c[20].value()), 50)
    assert_equal(Int(c[29].value()), 59)
    remove(path)


def test_a_footer_larger_than_the_speculative_read() raises:
    """A footer past 64 KiB costs a second read, and only then.

    `_read_footer` reads a tail, and takes an exact-size second read when the
    footer did not fit. Nothing reached that branch — real footers are
    kilobytes — so the fixture manufactures one out of many row groups.
    """
    var path = String("/tmp/marrow_pageio_bigfooter.parquet")
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var a = Python.list()
    for i in range(4000):
        a.append(i)
    # One row group per row: the footer carries per-group, per-column metadata,
    # so this is the cheapest way to make it large.
    pq.write_table(
        pa.table(Python.dict(a=pa.array(a, type=pa.int64()))),
        path,
        row_group_size=1,
        compression="none",
    )

    var reads = Reads(List[Tuple[Int, Int]]())
    var f = ParquetFile[_Recorder, LeafSet.all()](_Recorder(path, reads.copy()))
    assert_equal(f.num_row_groups(), 4000)

    var total = BufferSource(path).size()
    assert_true(
        len(reads[]) == 2,
        "a footer over the speculative read costs exactly two reads, got "
        + String(len(reads[])),
    )
    for ref r in reads[]:
        assert_equal(r[0] + r[1], total, "both reads end at the file's end")
    assert_true(
        reads[][1][1] > 64 * 1024,
        "the second read is sized to the footer",
    )
    remove(path)


def test_a_leading_selection_reads_almost_nothing() raises:
    """The `limit`-shaped case, which is where this pays.

    Ten pages of a thousand rows; the first ten rows are wanted. One page's
    worth of bytes is fetched, not ten — the shape of `select … limit 10` once
    a limit reaches the scan as a row range.
    """
    var path = String("/tmp/marrow_pageio_head.parquet")
    _write_paged(path, 10000, 1000)
    var ranges = _page_ranges(path)
    assert_equal(len(ranges), 10)

    var sels = List[RowSelection]()
    sels.append(_sel(10000, [(0, 10)]))

    var fetched, rows = _bytes_after_open(path, Optional(sels^))
    assert_equal(rows, 10)
    var chunk = 0
    for ref r in ranges:
        chunk += r[1]
    assert_true(
        fetched * 4 < chunk,
        "fetched " + String(fetched) + " of " + String(chunk),
    )
    remove(path)


def test_a_skipped_row_group_is_never_fetched() raises:
    """The coarser granularity, and here the saving *is* I/O: a row group left
    out of `row_groups` has its column chunks never requested at all.

    This is what row-group pruning buys and page pruning does not — the loop in
    `read` is over the selected groups, so an unselected group's byte range is
    never handed to the source.
    """
    var path = String("/tmp/marrow_pageio_groups.parquet")
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var a = Python.list()
    for i in range(30):
        a.append(i)
    pq.write_table(
        pa.table(Python.dict(a=pa.array(a, type=pa.int64()))),
        path,
        row_group_size=10,
        write_page_index=True,
        use_dictionary=False,
        compression="none",
    )

    var pi = read_page_index(path)
    var reads = Reads(List[Tuple[Int, Int]]())
    var f = ParquetFile[_Recorder, LeafSet.all()](_Recorder(path, reads.copy()))
    var opened = len(reads[])
    var groups = List[Int]()
    groups.append(1)
    assert_equal(f.read(row_groups=Optional(groups^)).num_rows(), 10)

    for g in range(3):
        ref oi = pi[g][0].offset_index.value()
        ref loc = oi.page_locations[0]
        assert_equal(
            _touched(reads, opened, loc.offset, loc.compressed_page_size),
            g == 1,
            "row group " + String(g),
        )
    remove(path)


# ---------------------------------------------------------------------------
# `scan_ranges` / `OffsetIndex.tiles_chunk`
# ---------------------------------------------------------------------------
#
# The planner's happy path is measured end to end above, by watching the reads.
# Its *fallbacks* are not reachable that way: they exist for a page index whose
# numbers do not describe the file, and a writer marrow would accept does not
# produce one. Tested here directly, because "read the whole chunk" is a
# correctness answer -- a planner that trusted a bad index would hand the
# decoder a range that does not hold the pages it names.


def _loc(offset: Int, size: Int, first_row: Int) -> PageLocation:
    var l = PageLocation()
    l.offset = offset
    l.compressed_page_size = size
    l.first_row_index = first_row
    return l^


def _whole(got: List[Tuple[Int, Int]], length: Int) raises:
    assert_equal(len(got), 1, "a distrusted index must ask for one range")
    assert_equal(got[0][0], 0)
    assert_equal(got[0][1], length, "and that range is the whole chunk")


def _pages(count: Int, size: Int, rows: Int, at: Int) -> OffsetIndex:
    """`count` adjacent pages of `size` bytes and `rows` rows, from `at`."""
    var oi = OffsetIndex()
    for i in range(count):
        oi.page_locations.append(_loc(at + i * size, size, i * rows))
    return oi^


def _chunk(start: Int, length: Int, dictionary: Bool = False) -> ColumnChunk:
    """A column chunk occupying `[start, start+length)`, with or without a
    dictionary page at its head."""
    var md = ColumnMetaData()
    md.data_page_offset = start
    md.total_compressed_size = length
    if dictionary:
        md.dictionary_page_offset = start
    var cc = ColumnChunk()
    cc.meta_data = md^
    return cc^


def test_scan_ranges_plans_runs_and_merges_neighbours() raises:
    """The contract the fallbacks below are a retreat from: one range per run
    of selected pages, adjacent pages merged."""
    var locs = _pages(count=7, size=100, rows=10, at=0)
    var got = scan_ranges(
        _sel(70, [(10, 30), (50, 60)]), locs, _chunk(0, 700), 70
    )
    assert_equal(len(got), 2)
    assert_equal(got[0][0], 100)
    assert_equal(got[0][1], 200, "pages 1 and 2 are one read")
    assert_equal(got[1][0], 500)
    assert_equal(got[1][1], 100)


def test_scan_ranges_stops_after_the_last_selected_row() raises:
    """Page locations ascend, so a page starting past the last selected row
    ends the walk — and every page after it is dead by construction. The
    boundary is `>`, not `>=`: a row that *is* a page's first row keeps that
    page. Without the stop a `limit` selection asks each dead page whether it
    holds anything, and each answers only after reading all of its flags."""
    var locs = _pages(count=7, size=100, rows=10, at=0)

    var got = scan_ranges(_sel(70, [(0, 5)]), locs, _chunk(0, 700), 70)
    assert_equal(len(got), 1, "only the page holding rows 0-4")
    assert_equal(got[0][0], 0)
    assert_equal(got[0][1], 100)

    # last selected row is page 2's first row: the break must not swallow it
    got = scan_ranges(_sel(70, [(5, 21)]), locs, _chunk(0, 700), 70)
    assert_equal(len(got), 1, "pages 0, 1 and 2 merge into one read")
    assert_equal(got[0][0], 0)
    assert_equal(got[0][1], 300, "page 2 is kept, page 3 is not")


def test_scan_ranges_puts_the_dictionary_first() raises:
    """The dictionary page is the bytes before the first data page, and it
    comes back as its own range rather than as a licence to read from the
    chunk's start onward."""
    var locs = _pages(count=7, size=100, rows=10, at=40)
    var got = scan_ranges(
        _sel(70, [(50, 60)]), locs, _chunk(0, 740, dictionary=True), 70
    )
    assert_equal(len(got), 2)
    assert_equal(got[0][0], 0)
    assert_equal(got[0][1], 40, "the dictionary page")
    assert_equal(got[1][0], 540, "and page 5, with nothing in between")


def test_scan_ranges_without_a_page_index_reads_everything() raises:
    """No locations, nothing to plan with."""
    _whole(
        scan_ranges(_sel(70, [(10, 20)]), OffsetIndex(), _chunk(0, 700), 70),
        700,
    )


def test_index_tiles_chunk_refuses_pages_that_overrun_the_group() raises:
    """A page starting past the group's last row: the index does not describe
    this file.

    Note what is *not* malformed — a final page holding fewer rows than the
    ones before it is ordinary, so the check has to be "this page runs past the
    group", not "the pages are uneven". Getting that wrong would refuse to plan
    for most real files.
    """
    var locs = _pages(count=3, size=100, rows=10, at=0)
    assert_false(locs.tiles_chunk(0, 300, 15))


def test_index_tiles_chunk_accepts_a_short_final_page() raises:
    """The other side of that line: 25 rows across three ten-row pages means
    the last one holds five, and the plan is made as usual."""
    var locs = _pages(count=3, size=100, rows=10, at=0)
    var got = scan_ranges(_sel(25, [(20, 25)]), locs, _chunk(0, 300), 25)
    assert_equal(len(got), 1)
    assert_equal(got[0][0], 200, "only the final page")
    assert_equal(got[0][1], 100)


def test_index_tiles_chunk_refuses_a_page_outside_the_chunk() raises:
    """A page whose bytes fall past the chunk's end.

    Checked for every page rather than only the selected ones: the same index
    is handed to `PageReader`, which seeks by these offsets whatever the
    selection asked for, so "nobody wanted that page" is not a reason to accept
    a number that cannot be right.
    """
    var locs = _pages(count=3, size=100, rows=10, at=0)
    locs.page_locations[2] = _loc(600, 100, 20)  # chunk is 300 bytes
    assert_false(locs.tiles_chunk(0, 300, 30))


def test_index_tiles_chunk_refuses_non_increasing_first_rows() raises:
    """Two pages claiming the same first row: one of them spans no rows, and
    the index cannot be walked."""
    var locs = _pages(count=3, size=100, rows=10, at=0)
    locs.page_locations[1] = _loc(100, 100, 0)
    assert_false(locs.tiles_chunk(0, 300, 30))


def test_index_tiles_chunk_requires_the_first_page_at_row_zero() raises:
    """The pages have to start where the row group does."""
    var locs = _pages(count=3, size=100, rows=10, at=0)
    locs.page_locations[0] = _loc(0, 100, 5)
    assert_false(locs.tiles_chunk(0, 300, 35))


def test_index_tiles_chunk_accepts_a_well_formed_index() raises:
    """The control: without one of these, every case above would pass against
    a predicate that answered `False` unconditionally."""
    assert_true(
        _pages(count=3, size=100, rows=10, at=0).tiles_chunk(0, 300, 30)
    )


def test_scan_ranges_refuses_a_dictionary_with_nowhere_to_live() raises:
    """The chunk is said to hold a dictionary page, but the first data page
    starts at the chunk's first byte — so there is no room for one."""
    var locs = _pages(count=3, size=100, rows=10, at=0)
    _whole(
        scan_ranges(
            _sel(30, [(0, 5)]), locs, _chunk(0, 300, dictionary=True), 30
        ),
        300,
    )


def test_scan_ranges_with_nothing_selected_asks_for_nothing() raises:
    """A selection that keeps no row asks for no bytes.

    It used to fall through to the whole-chunk retreat, on the grounds that a
    zero-length fetch is worse than a redundant one. That is true of a *short*
    read and false of this one: `_run_selected` stops before its first page, so
    every byte fetched here goes unread -- free on a memory map, a whole column
    chunk over HTTP. It is reachable too, from the pushdown path: a row group
    can survive statistics pruning and still lose every page.
    """
    var got = scan_ranges(
        _sel(30, List[Tuple[Int, Int]]()),
        _pages(count=3, size=100, rows=10, at=0),
        _chunk(0, 300),
        30,
    )
    assert_equal(len(got), 0, "no rows kept, so no bytes wanted")


def _bytes_after(reads: Reads, after: Int) -> Int:
    """Bytes asked for by the reads past the first `after`.

    `after` excludes the footer, which every read pays and which would flatten
    any ratio asserted on the rest.
    """
    var fetched = 0
    for i in range(after, len(reads[])):
        fetched += reads[][i][1]
    return fetched


def _bytes_after_open(
    path: String, var sels: Optional[List[RowSelection]]
) raises -> Tuple[Int, Int]:
    """`(bytes, rows)` one read asks its source for, footer excluded."""
    var reads = Reads(List[Tuple[Int, Int]]())
    var f = ParquetFile[_Recorder, LeafSet.all()](_Recorder(path, reads.copy()))
    var opened = len(reads[])
    var rows = f.read(row_selections=sels^).num_rows()
    return (_bytes_after(reads, opened), rows)


def test_each_row_group_stops_at_its_own_last_selected_row() raises:
    """Two groups, two different stop rows, and no recorder test before this
    one used more than one row group.

    `read` finds where a selection stops once per row group and hands that to
    `_scan_ranges` for each of the group's leaves. If the value leaked across
    groups -- carried in a loop variable rather than looked up per group -- the
    second group here would stop at the first group's row 499, fetch short, and
    return four hundred rows too few. The asymmetry is the point: group 0 keeps
    its head and group 1 keeps its tail, so one group's answer is wrong for the
    other.
    """
    var path = String("/tmp/marrow_pageio_two_groups.parquet")
    _write_paged(path, 4000, 500, group_rows=2000)

    var sels = List[RowSelection]()
    sels.append(_sel(2000, [(0, 500)]))  # group 0: first page
    sels.append(_sel(2000, [(1500, 2000)]))  # group 1: last page

    var bytes, rows = _bytes_after_open(path, Optional(sels^))
    assert_equal(rows, 1000, "500 from each group, neither truncated")

    # Two pages of eight. The denominator spans both groups: `_page_ranges`
    # answers for one, and charging a two-group fetch against one group's pages
    # is how this assertion first failed.
    var chunk = 0
    for g in range(2):
        for ref r in _page_ranges(path, g):
            chunk += r[1]
    assert_true(
        bytes * 2 < chunk,
        "fetched " + String(bytes) + " of " + String(chunk),
    )
    remove(path)
