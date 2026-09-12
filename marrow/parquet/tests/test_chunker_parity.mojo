"""Parity test: marrow's content-defined page boundaries must be
byte-identical to Arrow C++'s, for the same data and the same chunking
options.

Everything in `test_chunker.mojo` checks marrow against transplanted C++
constants (the gearhash table digest, the mask vectors) or against itself
(the same chunker run twice). None of that can catch a defect shared by
marrow's port and its own tests. This file is the independent oracle: pyarrow
23.0.1 (pinned in this repo) accepts `use_content_defined_chunking=True` on
`pq.write_table`, which drives the real Arrow C++ chunker
(`cpp/src/parquet/chunker_internal.cc`) to produce a reference file. Marrow's
own reader then parses the `OffsetIndex` of *both* files -- pyarrow exposes no
API for it -- so the comparison runs entirely through marrow's read path, but
the reference *bytes* come from pyarrow's writer. The oracle stays
independent even though marrow reads both sides.

One test per level path `ContentDefinedChunker._calculate` dispatches on,
plus the value-hashing dispatch each path exercises:

  - flat required int64  -- no levels hashed at all (`max_def == max_rep == 0`)
  - flat nullable int64  -- def levels hashed, values only when present
  - nullable string      -- same def-levels-only branch, through the
                            binary-like (element-bytes) dispatch instead of
                            the fixed-width one
  - nullable list<int64> -- the nested branch: both def and rep levels
                            hashed, cuts land only at record boundaries, and
                            `value_offset` must track leaf slots through a
                            null element inside a present list

Three settings are load-bearing -- see `marrow/parquet/chunker.mojo`'s module
docstring and `chunker_internal.cc` before touching them:

  - `data_page_size=1 << 30` on the pyarrow side. Arrow C++ also cuts pages on
    an encoder-internal byte estimate marrow does not reproduce, so parity is
    only claimed below that cap.
  - `max_rows_per_page` is left at pyarrow's 20 000 default, the same value as
    `marrow.parquet.writer.MAX_ROWS_PER_PAGE`, so both writers apply the same
    hard cap inside a chunk.
  - `row_group_size` matches on both sides, and every source table is
    single-chunk (one Arrow array per column, built with a single builder and
    wrapped in one `RecordBatch`). Arrow C++ skips the page break after the
    trailing, possibly-incomplete chunk of a `WriteArrow` call, so it can
    carry uninterrupted into the next call; with a single-chunk table, every
    row group is exactly one such call on both sides, so the two agree. A
    multi-chunk source table would not have this property.

pyarrow is called unconditionally -- no keyword probing, no skip. If a future
pyarrow bump drops `use_content_defined_chunking`, this must fail loudly: a
silent skip is how a parity regression would slip through.
"""

from std.testing import assert_equal
from std.python import Python, PythonObject
from std.os import remove
from ...arrays import DynArray
from ...builders import Int64Builder, StringBuilder, ListBuilder, arange
from ...dtypes import Field, Int64Type, DynType, decimal128, decimal256
from ...schema import Schema
from ...tabular import RecordBatch, Table, record_batch
from ...c_data import CArrowArrayStream
from ...kernels.cast import cast
from ...parquet import ParquetFile
from ...parquet.writer import FileWriter
from ...io import FileSink
from ...parquet.chunker import ContentDefinedChunking
from ...parquet.codecs import Compression
from .test_writer import _page_row_counts


# ---------------------------------------------------------------------------
# Bridge + page-boundary reader
# ---------------------------------------------------------------------------


def _to_pyarrow(var t: Table) raises -> PythonObject:
    """Marrow Table -> PyArrow table via the Arrow C stream interface. Both
    writers below consume data that came out of this one conversion, so a
    boundary mismatch cannot be a value mismatch wearing a disguise."""
    var pa = Python.import_module("pyarrow")
    var schema = t.schema.copy()
    var batches = t.to_batches()
    var caps = CArrowArrayStream.from_batches(schema^, batches^).to_pycapsule()
    return pa.RecordBatchReader._import_from_c_capsule(caps).read_all()


def _to_table(var batch: RecordBatch) raises -> Table:
    """Wrap a single RecordBatch into a Table with exactly one chunk per
    column -- the single-chunk property the row-group-boundary parity
    depends on (see the module docstring)."""
    var schema = batch.schema
    var batches = List[RecordBatch]()
    batches.append(batch^)
    return Table.from_batches(schema, batches^)


# ---------------------------------------------------------------------------
# The two writers
# ---------------------------------------------------------------------------


def _write_marrow_cdc(t: Table, path: String, row_group_size: Int) raises:
    var w = FileWriter(
        FileSink(path),
        Compression.UNCOMPRESSED,
        use_dictionary=False,
        content_defined_chunking=ContentDefinedChunking(),
    )
    w.write(t, row_group_size=row_group_size)


def _write_pyarrow_cdc(
    want: PythonObject, path: String, row_group_size: Int
) raises:
    var pq = Python.import_module("pyarrow.parquet")
    pq.write_table(
        want,
        path,
        use_content_defined_chunking=True,
        write_page_index=True,
        compression="none",
        use_dictionary=False,
        data_page_size=1 << 30,
        row_group_size=row_group_size,
    )


def _assert_page_parity(path_marrow: String, path_pyarrow: String) raises:
    """Column-0 page boundaries must match exactly in **every** row group,
    not only row group 0.

    Row group 0 is the one every defect agrees on, so checking it alone checks
    almost nothing. Two live ones hide past it. The chunker's rolling-hash
    state spans a row group and restarts with the next, matching where Arrow
    C++ puts `content_defined_chunker_` (a `ColumnWriterImpl` member, one
    column writer per row group); building one chunker per file instead moves
    every boundary after row group 0. And a nested column's leaf array must be
    narrowed to the row group's own element range before it reaches the writer
    or the chunker -- slicing a `RecordBatch` moves only the container's
    offset, so an unnarrowed child hashes, and writes, row group 0's values
    under row group 1's levels.

    Row group count and page counts are checked first, so a mismatch reports a
    count rather than an index error."""
    var num_rg = ParquetFile(path_marrow).num_row_groups()
    assert_equal(
        num_rg,
        ParquetFile(path_pyarrow).num_row_groups(),
        "row group count differs",
    )
    for rg in range(num_rg):
        var got = _page_row_counts(path_marrow, rg, 0)
        var want = _page_row_counts(path_pyarrow, rg, 0)
        assert_equal(
            len(got),
            len(want),
            "row group "
            + String(rg)
            + " page count differs: marrow "
            + String(len(got))
            + " vs pyarrow "
            + String(len(want)),
        )
        for i in range(len(got)):
            assert_equal(
                got[i],
                want[i],
                "row group "
                + String(rg)
                + " page "
                + String(i)
                + " row count differs: marrow "
                + String(got[i])
                + " vs pyarrow "
                + String(want[i]),
            )


def _check_parity(t: Table, tag: String, row_group_size: Int) raises:
    var path_marrow = "/tmp/marrow_cdc_parity_" + tag + "_marrow.parquet"
    var path_pyarrow = "/tmp/marrow_cdc_parity_" + tag + "_pyarrow.parquet"
    _write_marrow_cdc(t, path_marrow, row_group_size)
    var want = _to_pyarrow(t.copy())
    _write_pyarrow_cdc(want, path_pyarrow, row_group_size)
    _assert_page_parity(path_marrow, path_pyarrow)
    remove(path_marrow)
    remove(path_pyarrow)


# ---------------------------------------------------------------------------
# Table builders -- one per level path
# ---------------------------------------------------------------------------


def _flat_required_int64_batch(n: Int) raises -> RecordBatch:
    """Required (non-nullable) int64: `max_def == max_rep == 0`, so
    `_calculate`'s first branch hashes values only, no levels."""
    var arr = arange[Int64Type](0, n)
    var schema = Schema(fields=[Field("x", arr.type(), nullable=False)])
    var cols = List[DynArray]()
    cols.append(arr^.to_dyn())
    return RecordBatch(schema=schema, columns=cols^)


def _flat_nullable_int64_batch(n: Int) raises -> RecordBatch:
    """Nullable int64: `max_rep == 0`, `max_def == 1` -- `_calculate`'s
    second branch, hashing the def level for every row and the value only
    when present."""
    var b = Int64Builder(capacity=n)
    for i in range(n):
        if i % 7 == 0:
            b.unsafe_append_null()
        else:
            b.unsafe_append(Int64(i))
    return record_batch([b.finish().to_dyn()], names=["x"])


def _nullable_string_batch(n: Int) raises -> RecordBatch:
    """Nullable string: the same def-levels-only branch as the nullable
    int64 case, but through the binary-like (element-bytes) value dispatch
    instead of the fixed-width scalar one."""
    var b = StringBuilder(capacity=n)
    for i in range(n):
        if i % 13 == 0:
            b.append_null()
        else:
            b.append("row-" + String(i))
    return record_batch([b.finish().to_dyn()], names=["x"])


def _nullable_list_int64_batch(n: Int) raises -> RecordBatch:
    """Nullable `list<int64>` with nullable elements: `_calculate`'s nested
    branch (`max_rep >= 1`), hashing both def and rep levels and cutting
    only at record (`rep == 0`) boundaries. Mixes null lists, empty-but-valid
    lists, and lists holding a null element among present ones, so
    `value_offset` must advance on `def >= slot_def` -- including the null
    element -- rather than on a non-null count."""
    var lb = ListBuilder(Int64Builder())
    var next_val: Int64 = 0
    for i in range(n):
        var m = i % 13
        if m == 0:
            lb.append_null()
        elif m == 1:
            lb.append_valid()  # present, zero-length list
        else:
            var k = 1 + (i % 5)
            for j in range(k):
                if (i + j) % 9 == 0:
                    lb.values().as_int64().append_null()
                else:
                    lb.values().as_int64().append(next_val)
                    next_val += 1
            lb.append_valid()
    return record_batch([lb.finish().to_dyn()], names=["x"])


def _flat_nullable_decimal_batch(n: Int, to: DynType) raises -> RecordBatch:
    """Nullable decimal (16- or 32-byte fixed-width leaf, depending on `to`):
    the same def-levels-only branch as the nullable int64 and string cases,
    but through the fixed-width `_roll_scalar` path at a width
    `test_chunker_dispatches_every_leaf_type` only checks does-not-raise,
    never against the C++ reference. `_roll_scalar` shifts the raw
    little-endian integer bytes out of the 128-/256-bit storage; this is the
    one place that assumption could quietly diverge from what C++ hashes."""
    var b = Int64Builder(capacity=n)
    for i in range(n):
        if i % 7 == 0:
            b.unsafe_append_null()
        else:
            b.unsafe_append(Int64(i))
    var dec = cast(b.finish(), to)
    return record_batch([dec^], names=["x"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chunker_parity_flat_required_int64() raises:
    """2M required int64 rows across two row groups: enough bytes that
    several default-size (256 KiB-1 MiB) CDC chunks form, most split further
    by the 20 000-row cap -- so both the content-driven cuts and the
    row-cap-driven ones must agree."""
    var n = 2_000_000
    var t = _to_table(_flat_required_int64_batch(n))
    _check_parity(t, "required_int64", row_group_size=1_000_000)


def test_chunker_parity_flat_nullable_int64() raises:
    var n = 2_000_000
    var t = _to_table(_flat_nullable_int64_batch(n))
    _check_parity(t, "nullable_int64", row_group_size=1_000_000)


def test_chunker_parity_nullable_string() raises:
    var n = 300_000
    var t = _to_table(_nullable_string_batch(n))
    _check_parity(t, "nullable_string", row_group_size=100_000)


def test_chunker_parity_nullable_list_int64() raises:
    var n = 300_000
    var t = _to_table(_nullable_list_int64_batch(n))
    _check_parity(t, "nullable_list_int64", row_group_size=100_000)


def test_chunker_parity_flat_nullable_decimal128() raises:
    var n = 500_000
    var t = _to_table(
        _flat_nullable_decimal_batch(n, decimal128(20, 4).to_dyn())
    )
    _check_parity(t, "nullable_decimal128", row_group_size=250_000)


def test_chunker_parity_flat_nullable_decimal256() raises:
    var n = 500_000
    var t = _to_table(
        _flat_nullable_decimal_batch(n, decimal256(40, 4).to_dyn())
    )
    _check_parity(t, "nullable_decimal256", row_group_size=250_000)
