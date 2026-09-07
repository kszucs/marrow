"""RowSelection: the unit-level per-row keep/skip within a row group (built from
per-page keep flags and combined with intersect) plus the decode path — reading a
file with a RowSelection must yield exactly the selected rows, and must match a
full read filtered to the same rows. Uses tiny data pages so a selection
genuinely spans skip / keep / partial pages across every builder (primitive,
nullable, string/dict)."""

from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.python import Python, PythonObject
from std.os import remove
from ...parquet import read_table
from ...parquet.reader import ParquetFile, RowSelection
from ...tabular import Table


def _write(tbl: PythonObject) raises -> String:
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_pageskip.parquet")
    # tiny pages -> many pages per chunk; single row group
    pq.write_table(
        tbl,
        path,
        data_page_size=128,
        row_group_size=1000000,
        compression="none",
    )
    return path


def _write_pages(
    tbl: PythonObject,
    page_rows: Int,
    codec: String = "none",
    version: String = "1.0",
) raises -> String:
    """`tbl` written in pages of exactly `page_rows` rows.

    **`write_batch_size` is what sets the page length, not `data_page_size`.**
    parquet-cpp only checks whether the current page is full at the end of each
    write batch, so with the default batch of 1024 a small `data_page_size`
    still yields one page for anything under 1024 rows -- and a page-skipping
    test whose file has one page measures nothing. The cases above get many
    pages only because they write 10,000 rows; the ones below need them at a
    *known* offset, so this names both.

    `codec` and `version` are parameters because page skipping is worth the
    most on a *compressed* file -- a page never decoded is never decompressed --
    and because a v2 data page keeps its rep/def levels outside the compressed
    body, so its length is not the v1 arithmetic.
    """
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_pageskip_boundary.parquet")
    pq.write_table(
        tbl,
        path,
        data_page_size=1,
        write_batch_size=page_rows,
        row_group_size=1000000,
        write_page_index=True,
        use_dictionary=False,
        data_page_version=version,
        compression=codec,
    )
    return path


def _selection(n: Int, keep_from: Int, keep_to: Int) -> RowSelection:
    var s = List[Bool](capacity=n)
    for i in range(n):
        s.append(i >= keep_from and i < keep_to)
    return RowSelection(s^)


def _strided(n: Int, m: Int, r: Int) -> RowSelection:
    # keep rows where i % m < r (scattered across page boundaries)
    var s = List[Bool](capacity=n)
    for i in range(n):
        s.append((i % m) < r)
    return RowSelection(s^)


def _assert_matches_full(
    var got: Table, var full: Table, sel: RowSelection
) raises:
    """`got` (read with `sel`) must equal `full` filtered to the selected rows.
    """
    assert_equal(got.num_rows(), sel.num_selected())
    var gb = got.to_batches()[0].copy()
    var fb = full.to_batches()[0].copy()
    var ncols = gb.num_columns()
    for c in range(ncols):
        var gcol = gb.columns[c].copy()
        var fcol = fb.columns[c].copy()
        var k = 0  # index into the selected (got) rows
        for i in range(sel.total_rows()):
            if sel.selected(i):
                assert_equal(gcol.is_valid(k), fcol.is_valid(i))
                if gcol.is_valid(k):
                    assert_equal(String(gcol[k]), String(fcol[i]))
                k += 1


def _check(tbl: PythonObject, sel: RowSelection) raises:
    var path = _write(tbl)
    var full = read_table(path)
    var rs = List[RowSelection]()
    rs.append(sel.copy())
    var got = read_table(path, row_selections=rs^)
    _assert_matches_full(got^, full^, sel)
    remove(path)


def _col(arr: PythonObject) raises -> PythonObject:
    """A single-column ("c") PyArrow table around `arr`."""
    return Python.import_module("pyarrow").table(Python.dict(c=arr))


def test_contiguous_int() raises:
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    _check(
        _col(pa.array(np.arange(10000), type=pa.int64())),
        _selection(10000, 2500, 7500),
    )


def test_scattered_int() raises:
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    _check(
        _col(pa.array(np.arange(10000), type=pa.int64())),
        _strided(10000, 7, 3),
    )


def test_nullable_int() raises:
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var idx = np.arange(5000)
    _check(
        _col(pa.array(idx, mask=(idx % 4 == 0), type=pa.int64())),
        _strided(5000, 5, 2),
    )


def test_string_dict() raises:
    # low cardinality -> dictionary-encoded data pages
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var vals = np.array(Python.list("red", "green", "blue"))[
        np.arange(6000) % 3
    ]
    _check(_col(pa.array(vals)), _selection(6000, 1000, 4000))


def test_string_plain_nullable() raises:
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var idx = np.arange(4000)
    var vals = np.char.add("v", idx.astype("U"))
    _check(
        _col(pa.array(vals, mask=(idx % 6 == 0), type=pa.string())),
        _strided(4000, 3, 1),
    )


def test_scattered_bool() raises:
    # exercises BoolLeafBuilder's partial-page selected scatter
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var idx = np.arange(6000)
    _check(
        _col(pa.array(idx % 2 == 0, mask=(idx % 9 == 0), type=pa.bool_())),
        _strided(6000, 5, 2),
    )


def test_scattered_float() raises:
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    _check(
        _col(pa.array(np.arange(8000) * 0.25, type=pa.float64())),
        _strided(8000, 7, 3),
    )


def test_scattered_temporal() raises:
    # timestamp column (INT64 storage retagged) under a partial-page selection
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    _check(
        _col(pa.array(np.arange(8000) * 1000, type=pa.timestamp("us"))),
        _selection(8000, 2000, 6000),
    )


def test_scattered_decimal() raises:
    # decimal128 (FIXED_LEN_BYTE_ARRAY) drives DecimalLeafBuilder.place() per row
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var vals = np.char.add(np.arange(4000).astype("U"), ".25")
    _check(
        _col(pa.array(vals).cast(pa.decimal128(12, 2))),
        _strided(4000, 5, 2),
    )


def test_scattered_fixed_size_binary() raises:
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var vals = np.char.zfill((np.arange(5000) % 9999).astype("U"), 4)
    _check(
        _col(pa.array(vals.astype("S4"), type=pa.binary(4))),
        _strided(5000, 6, 3),
    )


def test_selection_across_every_page_boundary_case() raises:
    """Every way a selection can sit against a page edge, in one file.

    Ported from parquet-rs's `test_scan_ranges`
    (`parquet/src/arrow/arrow_reader/selection/ranges.rs`), which pins the same
    property one layer down: which *pages* a per-row selection forces a reader
    to touch. Seven pages of ten rows, and the selection is built so that each
    transition appears exactly once —

    | rows | against the pages |
    |---|---|
    | 0-9 | a whole page skipped, at the start |
    | 10-12, 16-19 | two disjoint runs *inside* one page |
    | 25-29 | a run ending exactly on a page boundary |
    | 30-41 | a whole page skipped, plus part of the next |
    | 42-53 | a run spanning a page boundary |
    | 54-69 | a whole page skipped, at the end |

    The cases above this one sweep contiguous and strided patterns over ~10
    pages, which is good at catching an off-by-one *inside* a page and blind to
    one at its edge: a strided mask keeps something in every page, so no page
    is ever skipped whole. Here pages 0, 3 and 6 must be skipped entirely and
    pages 1, 2, 4 and 5 partially read, and the assertion is on the values.
    """
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var path = _write_pages(_col(pa.array(np.arange(70), type=pa.int64())), 10)

    var f = ParquetFile(path)
    assert_equal(
        len(f.page_bounds()[0][0]),
        7,
        "the fixture must write seven pages or this proves nothing",
    )

    var keep = List[Bool](capacity=70)
    for i in range(70):
        keep.append(
            (10 <= i < 13) or (16 <= i < 20) or (25 <= i < 30) or (42 <= i < 54)
        )
    var sel = RowSelection(keep^)

    var full = read_table(path)
    var rs = List[RowSelection]()
    rs.append(sel.copy())
    var got = read_table(path, row_selections=rs^)
    assert_equal(got.num_rows(), 24)
    _assert_matches_full(got^, full^, sel)
    remove(path)


def _seventy_in_pages_of_ten() raises -> String:
    """`0..69` in seven pages of ten — the shape parquet-rs's `test_scan_ranges`
    reasons over, so the boundary cases below can be read straight across."""
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    return _write_pages(_col(pa.array(np.arange(70), type=pa.int64())), 10)


def _mask(n: Int, runs: List[Tuple[Int, Int]]) -> RowSelection:
    """A selection keeping each half-open `[start, end)` run and nothing else.
    """
    var keep = List[Bool](capacity=n)
    for i in range(n):
        var hit = False
        for ref run in runs:
            if run[0] <= i < run[1]:
                hit = True
                break
        keep.append(hit)
    return RowSelection(keep^)


def _read_selected(path: String, sel: RowSelection) raises:
    """Read `path` under `sel` and assert it equals a full read filtered the
    same way."""
    var full = read_table(path)
    var rs = List[RowSelection]()
    rs.append(sel.copy())
    var got = read_table(path, row_selections=rs^)
    _assert_matches_full(got^, full^, sel)


def test_selection_spilling_one_row_into_the_next_page() raises:
    """The minimal partial page: a run that ends one row past a boundary.

    parquet-rs's `test_scan_ranges` calls this "select to remaining in page and
    first row of next page" — pages 1, 2 and 3 are touched, and page 3
    contributes exactly one row. A reader that rounds a partial page down drops
    that row; one that rounds up returns nine extra.
    """
    var path = _seventy_in_pages_of_ten()
    _read_selected(path, _mask(70, [(10, 13), (16, 20), (25, 31)]))
    remove(path)


def test_selection_running_to_the_last_row_of_the_group() raises:
    """A run that ends on the final row, so the last page is partially read and
    nothing follows it to catch an overrun."""
    var path = _seventy_in_pages_of_ten()
    _read_selected(path, _mask(70, [(10, 13), (42, 70)]))
    remove(path)


def test_selection_of_the_final_page_alone() raises:
    """Everything before the last page skipped — the mirror of the first case
    in the boundary matrix, where the skipped run is a prefix."""
    var path = _seventy_in_pages_of_ten()
    _read_selected(path, _mask(70, [(60, 70)]))
    remove(path)


def test_selection_of_one_row_per_page() raises:
    """Every page partially read and none skipped, which is the case a
    whole-page fast path gets wrong in the opposite direction from a mask that
    skips whole pages."""
    var path = _seventy_in_pages_of_ten()
    _read_selected(
        path,
        _mask(
            70,
            [
                (0, 1),
                (10, 11),
                (20, 21),
                (30, 31),
                (40, 41),
                (50, 51),
                (60, 61),
            ],
        ),
    )
    remove(path)


def test_selection_over_compressed_pages() raises:
    """A skipped page is skipped *before* decompression, and the kept ones
    still decode.

    Every other case here writes uncompressed pages, so the interaction
    between the skip path and `PageReader._body` — which decompresses into a
    reused scratch buffer — was never exercised. Snappy because it is the
    codec marrow opens by default.
    """
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var path = _write_pages(
        _col(pa.array(np.arange(70), type=pa.int64())), 10, "snappy"
    )
    _read_selected(path, _mask(70, [(10, 13), (25, 31), (60, 70)]))
    remove(path)


def test_selection_over_v2_data_pages() raises:
    """A v2 data page puts its rep/def levels outside the compressed body, so
    its length is not simply header plus compressed values. The seek steps by
    the `OffsetIndex`'s size either way; this is what says so."""
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var path = _write_pages(
        _col(pa.array(np.arange(70), type=pa.int64())), 10, "none", "2.0"
    )
    _read_selected(path, _mask(70, [(10, 13), (25, 31), (60, 70)]))
    remove(path)


def test_selection_over_compressed_v2_data_pages() raises:
    """Both at once, which is what a real file tends to be."""
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var path = _write_pages(
        _col(pa.array(np.arange(70), type=pa.int64())), 10, "snappy", "2.0"
    )
    _read_selected(path, _mask(70, [(10, 13), (25, 31), (60, 70)]))
    remove(path)


def test_per_group_selections_are_positional() raises:
    """Three row groups, three *different* selections, applied to the right
    groups.

    `read` takes one selection per entry of `row_groups`, in that order — an
    invariant only its length is checked against. Giving each group a distinct
    pattern is what makes a transposition visible: swapped selections keep the
    right *number* of rows and the wrong ones.
    """
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_pageskip_groups.parquet")
    pq.write_table(
        _col(pa.array(np.arange(30), type=pa.int64())),
        path,
        row_group_size=10,
        data_page_size=1,
        write_batch_size=5,
        write_page_index=True,
        use_dictionary=False,
        compression="none",
    )

    var groups = List[Int]()
    groups.append(0)
    groups.append(1)
    groups.append(2)
    var rs = List[RowSelection]()
    rs.append(_mask(10, [(0, 2)]))  # group 0 -> 0, 1
    rs.append(_mask(10, [(5, 6)]))  # group 1 -> 15
    rs.append(_mask(10, [(7, 10)]))  # group 2 -> 27, 28, 29

    var got = read_table(
        path, row_groups=Optional(groups^), row_selections=Optional(rs^)
    )
    assert_equal(got.num_rows(), 6)
    # One batch per row group, so the values are read across them rather than
    # out of the first — a selection applied to the wrong group would keep the
    # right count and the wrong rows, which is the whole point of this case.
    var seen = List[Int]()
    for ref batch in got.to_batches():
        ref c = batch.columns[0].as_int64()
        for i in range(len(c)):
            seen.append(Int(c[i].value()))
    assert_equal(seen, [0, 1, 15, 27, 28, 29])
    remove(path)


def test_selection_and_column_projection_together() raises:
    """A projection narrows which columns are decoded and a selection narrows
    which rows; both at once must still return aligned columns."""
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_pageskip_project.parquet")
    pq.write_table(
        pa.table(
            Python.dict(
                a=pa.array(np.arange(70), type=pa.int64()),
                b=pa.array(np.arange(70) * 2, type=pa.int64()),
            )
        ),
        path,
        row_group_size=1000000,
        data_page_size=1,
        write_batch_size=10,
        write_page_index=True,
        use_dictionary=False,
        compression="none",
    )

    var sel = _mask(70, [(10, 13), (42, 54)])
    var cols = List[String]()
    cols.append(String("b"))
    var rs = List[RowSelection]()
    rs.append(sel.copy())
    var got = read_table(
        path, columns=Optional(cols^), row_selections=Optional(rs^)
    )
    assert_equal(got.num_rows(), sel.num_selected())
    assert_equal(len(got.schema.fields), 1)
    ref b = got.to_batches()[0].columns[0].as_int64()
    assert_equal(Int(b[0].value()), 20)
    assert_equal(Int(b[3].value()), 84)
    remove(path)


def test_row_selection_on_a_repeated_column_is_refused() raises:
    """A selection the decoder cannot apply is an error, not a silent miss.

    `ColumnReader.decode` picks the flat or the leveled path from the leaf's
    max repetition and only the flat one consults the selection, so a repeated
    column would decode *all* its rows while its flat neighbours decoded a
    subset — columns of different lengths, which is a wrong answer rather than
    a slow one. `read` refuses by name rather than leaving that as a rule every
    caller has to know: a leaf count does not reveal it either, since
    `list<int>` is a single leaf.
    """
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_pageskip_repeated.parquet")
    pq.write_table(
        pa.table(
            Python.dict(
                flat=pa.array(Python.list(1, 2, 3), type=pa.int64()),
                items=pa.array(
                    Python.list(
                        Python.list(1, 2),
                        Python.list(),
                        Python.list(3),
                    ),
                    type=pa.list_(pa.int64()),
                ),
            )
        ),
        path,
        compression="none",
    )

    var rs = List[RowSelection]()
    rs.append(_selection(3, 0, 2))
    with assert_raises(contains="repeated column"):
        _ = read_table(path, row_selections=rs^)

    # The same file reads fine without one -- the refusal is about the
    # selection, not about the file.
    assert_equal(read_table(path).num_rows(), 3)
    remove(path)


def test_select_none() raises:
    # an empty selection reads zero rows
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var path = _write(_col(pa.array(np.arange(2000), type=pa.int64())))
    var s = List[Bool](capacity=2000)
    for _ in range(2000):
        s.append(False)
    var rs = List[RowSelection]()
    rs.append(RowSelection(s^))
    var got = read_table(path, row_selections=rs^)
    assert_equal(got.num_rows(), 0)
    remove(path)


def test_read_row_group_out_of_range() raises:
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var path = _write(_col(pa.array(np.arange(100), type=pa.int64())))
    var groups: List[Int] = [5]  # the file has a single row group
    with assert_raises():
        _ = read_table(path, row_groups=groups^)
    remove(path)


def test_row_selections_count_mismatch() raises:
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var path = _write(_col(pa.array(np.arange(100), type=pa.int64())))
    # two selections but only one (selected) row group
    var rs = List[RowSelection]()
    rs.append(RowSelection.all(100))
    rs.append(RowSelection.all(100))
    with assert_raises():
        _ = read_table(path, row_selections=rs^)
    remove(path)


# ---------------------------------------------------------------------------
# RowSelection unit tests: per-row keep/skip built from per-page keep flags and
# combined with intersect.
# ---------------------------------------------------------------------------


def test_all() raises:
    var s = RowSelection.all(5)
    assert_equal(s.total_rows(), 5)
    assert_equal(s.num_selected(), 5)
    assert_true(s.selects_all())
    assert_true(s.selects_any())


def test_from_pages() raises:
    # 3 pages of 2/3/2 rows; keep pages 0 and 2
    var keep: List[Bool] = [True, False, True]
    var rows: List[Int] = [2, 3, 2]
    var s = RowSelection.from_pages(keep, rows)
    assert_equal(s.total_rows(), 7)
    assert_equal(s.num_selected(), 4)
    assert_true(s.selected(0))
    assert_true(s.selected(1))
    assert_false(s.selected(2))  # page 1 skipped
    assert_false(s.selected(4))
    assert_true(s.selected(5))  # page 2 kept
    assert_false(s.selects_all())
    assert_true(s.selects_any())


def test_intersect() raises:
    var av: List[Bool] = [True, True, False, True]
    var a = RowSelection(av^)
    var bv: List[Bool] = [True, False, True, True]
    var b = RowSelection(bv^)
    var c = a.intersect(b)
    assert_equal(c.num_selected(), 2)  # rows 0 and 3
    assert_true(c.selected(0))
    assert_false(c.selected(1))
    assert_false(c.selected(2))
    assert_true(c.selected(3))


def test_intersect_size_mismatch() raises:
    var a = RowSelection.all(3)
    var b = RowSelection.all(4)
    with assert_raises():
        _ = a.intersect(b)


def test_selected_in() raises:
    var sv: List[Bool] = [True, False, True, True, False]
    var s = RowSelection(sv^)
    assert_equal(s.selected_in(0, 5), 3)
    assert_equal(s.selected_in(1, 2), 1)  # rows 1,2 -> only 2
    assert_equal(s.selected_in(3, 2), 1)  # rows 3,4 -> only 3


def test_none_selected() raises:
    var sv: List[Bool] = [False, False, False]
    var s = RowSelection(sv^)
    assert_false(s.selects_any())
    assert_equal(s.num_selected(), 0)
