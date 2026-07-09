"""Cross-compatibility tests: Marrow and PyArrow must read each other's Parquet.

Three shapes, run over a range of types and codecs:

  A. Marrow reads a PyArrow-written file   — the broad read surface.
  B. PyArrow reads a Marrow-written file    — the writer surface.
  C. Marrow round-trips its own file        — the writer surface.

Equality is checked column-by-column — the type string (which is
nullability-agnostic) plus the values (`to_pylist`) — so it tolerates the
chunk-boundary and field-metadata differences that `pa.Table.equals` rejects.
"""

from std.testing import assert_true, assert_equal
from std.python import Python, PythonObject
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table, write_table
from marrow.parquet.compression import Compression
from marrow.tabular import Table
from marrow.c_data import CArrowArrayStream


# ---------------------------------------------------------------------------
# Bridges between the two implementations
# ---------------------------------------------------------------------------


def _to_marrow(py_tbl: PythonObject) raises -> Table:
    """PyArrow table -> Marrow Table via the Arrow C stream interface."""
    var caps = py_tbl.__arrow_c_stream__(Python.none())
    return CArrowArrayStream.from_pycapsule(caps).to_table()


def _to_pyarrow(var t: Table) raises -> PythonObject:
    """Marrow Table -> PyArrow table via the Arrow C stream interface."""
    var pa = Python.import_module("pyarrow")
    var schema = t.schema.copy()
    var batches = t.to_batches()
    var caps = CArrowArrayStream.from_batches(schema^, batches^).to_pycapsule()
    return pa.RecordBatchReader._import_from_c_capsule(caps).read_all()


def _norm_type(t: PythonObject) raises -> String:
    """Normalize Arrow-only type distinctions that Parquet does not preserve, so
    marrow's read and pyarrow's read compare equal: the `large_*` variants
    collapse to their base type (pyarrow only restores `large_string` from its
    private `ARROW:schema` metadata, which marrow ignores), and the LIST element
    field name is arbitrary (marrow calls it `item`, pyarrow `element`)."""
    var s = t.__str__()
    return String(
        s.replace("large_", "").replace("item: ", "").replace("element: ", "")
    )


def _values(col: PythonObject) raises -> PythonObject:
    """Column values as a Python list. `to_pylist()` on nanosecond timestamps /
    times raises without pandas, so those are compared as their raw integers."""
    var pt = Python.import_module("pyarrow.types")
    var pa = Python.import_module("pyarrow")
    if Bool(pt.is_timestamp(col.type)) or Bool(pt.is_time64(col.type)):
        if String(col.type.unit) == "ns":
            return col.cast(pa.int64()).to_pylist()
    return col.to_pylist()


def _assert_equiv(got: PythonObject, want: PythonObject) raises:
    """Assert two PyArrow tables hold the same columns and values, ignoring
    chunk boundaries and field-level metadata (nullability)."""
    var ncols = Int(py=want.num_columns)
    assert_equal(Int(py=got.num_columns), ncols)
    assert_true(Bool(got.column_names == want.column_names), "column names")
    for i in range(ncols):
        var g = got.column(i)
        var w = want.column(i)
        assert_equal(_norm_type(g.type), _norm_type(w.type))
        assert_true(Bool(_values(g) == _values(w)), "values differ")


# ---------------------------------------------------------------------------
# The three shapes
# ---------------------------------------------------------------------------


def _marrow_reads_pyarrow(want: PythonObject, compression: String) raises:
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_iop_a.parquet")
    pq.write_table(want, path, compression=compression)
    # Oracle is PyArrow's *own* read of the same file: Parquet erases some
    # Arrow-only distinctions (large_string -> string, etc.), so both readers
    # should agree with each other, not necessarily with the pre-write table.
    _assert_equiv(_to_pyarrow(read_table(path)), pq.read_table(path))
    remove(path)


def _pyarrow_reads_marrow(want: PythonObject, codec: Compression) raises:
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_iop_b.parquet")
    write_table(_to_marrow(want), path, compression=codec)
    _assert_equiv(pq.read_table(path), want)
    remove(path)


def _marrow_roundtrip(want: PythonObject, codec: Compression) raises:
    var path = String("/tmp/marrow_iop_c.parquet")
    write_table(_to_marrow(want), path, compression=codec)
    _assert_equiv(_to_pyarrow(read_table(path)), want)
    remove(path)


def _all_shapes(want: PythonObject) raises:
    """Full matrix for types the writer supports."""
    _marrow_reads_pyarrow(want, "none")
    _marrow_reads_pyarrow(want, "snappy")
    _marrow_reads_pyarrow(want, "zstd")
    _pyarrow_reads_marrow(want, Compression.UNCOMPRESSED)
    _pyarrow_reads_marrow(want, Compression.SNAPPY)
    _pyarrow_reads_marrow(want, Compression.ZSTD)
    _pyarrow_reads_marrow(want, Compression.LZ4_RAW)
    _marrow_roundtrip(want, Compression.UNCOMPRESSED)
    _marrow_roundtrip(want, Compression.SNAPPY)
    _marrow_roundtrip(want, Compression.LZ4_RAW)


def _read_only(want: PythonObject) raises:
    """Read surface only (writer does not yet emit these types)."""
    _marrow_reads_pyarrow(want, "none")
    _marrow_reads_pyarrow(want, "snappy")
    _marrow_reads_pyarrow(want, "zstd")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_interop_primitives() raises:
    var pa = Python.import_module("pyarrow")
    var t = pa.table(
        Python.dict(
            i32=pa.array(Python.list(1, 2, 3, -4), type=pa.int32()),
            i64=pa.array(Python.list(10, 20, 30, 40), type=pa.int64()),
            f64=pa.array(Python.list(1.5, 2.5, 3.5, 4.5), type=pa.float64()),
            b=pa.array(Python.list(True, False, True, False), type=pa.bool_()),
            s=pa.array(Python.list("a", "bb", "ccc", "dddd")),
        )
    )
    _all_shapes(t)


def test_interop_nullable() raises:
    var pa = Python.import_module("pyarrow")
    var n = Python.none()
    var t = pa.table(
        Python.dict(
            i=pa.array(Python.list(1, n, 3, n, 5), type=pa.int64()),
            f=pa.array(Python.list(n, 2.0, n, 4.0, n), type=pa.float64()),
            b=pa.array(Python.list(True, n, False, n, True), type=pa.bool_()),
            s=pa.array(Python.list("x", n, "z", n, "w")),
        )
    )
    _all_shapes(t)


def test_interop_narrow_ints() raises:
    var pa = Python.import_module("pyarrow")
    var t = pa.table(
        Python.dict(
            i8=pa.array(Python.list(-1, 2, -3), type=pa.int8()),
            i16=pa.array(Python.list(1000, -2000, 3000), type=pa.int16()),
            u8=pa.array(Python.list(1, 200, 3), type=pa.uint8()),
            u16=pa.array(Python.list(1, 2, 60000), type=pa.uint16()),
            u32=pa.array(Python.list(1, 2, 3000000000), type=pa.uint32()),
        )
    )
    _all_shapes(t)


def test_interop_struct() raises:
    var pa = Python.import_module("pyarrow")
    var fields = Python.list(
        pa.field("a", pa.int64()), pa.field("b", pa.string())
    )
    var t = pa.table(
        Python.dict(
            s=pa.array(
                Python.list(
                    Python.dict(a=1, b="one"),
                    Python.dict(a=2, b="two"),
                    Python.dict(a=3, b="three"),
                ),
                type=pa.struct(fields),
            ),
        )
    )
    _all_shapes(t)


def test_interop_empty() raises:
    var pa = Python.import_module("pyarrow")
    var t = pa.table(
        Python.dict(
            i=pa.array(Python.list(), type=pa.int64()),
            s=pa.array(Python.list(), type=pa.string()),
        )
    )
    _all_shapes(t)


def test_interop_dictionary_read() raises:
    # low-cardinality columns -> PyArrow keeps RLE_DICTIONARY, exercising the
    # SIMD bit-unpack + gather read path across every codec.
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var idx = np.arange(5000)
    var t = pa.table(
        Python.dict(
            a=pa.array(idx % 7),
            b=pa.array((idx % 13).astype("float64")),
            c=pa.array(np.where(idx % 3 == 0, "yes", "no")),
        )
    )
    _marrow_reads_pyarrow(t, "none")
    _marrow_reads_pyarrow(t, "snappy")
    _marrow_reads_pyarrow(t, "zstd")
    _marrow_reads_pyarrow(t, "gzip")
    _marrow_reads_pyarrow(t, "lz4")


def test_interop_temporal_read() raises:
    var pa = Python.import_module("pyarrow")
    var dt = Python.import_module("datetime")
    var n = Python.none()
    var t = pa.table(
        Python.dict(
            d=pa.array(
                Python.list(dt.date(2020, 1, 1), n, dt.date(2021, 6, 15)),
                type=pa.date32(),
            ),
            ts_us=pa.array(Python.list(1, 2, n), type=pa.timestamp("us")),
            ts_ns=pa.array(Python.list(10, n, 20), type=pa.timestamp("ns")),
            tm=pa.array(Python.list(5, 6, n), type=pa.time64("us")),
        )
    )
    _read_only(t)


def test_interop_binary_read() raises:
    var t = Python.evaluate(
        "__import__('pyarrow').table({"
        "'b': __import__('pyarrow').array([b'ab', None, b'cd'],"
        " type=__import__('pyarrow').binary()),"
        "'ls': __import__('pyarrow').array(['big', None, 'string'],"
        " type=__import__('pyarrow').large_string())})"
    )
    _read_only(t)


def test_interop_list_read() raises:
    var pa = Python.import_module("pyarrow")
    var n = Python.none()
    var t = pa.table(
        Python.dict(
            li=pa.array(
                Python.list(
                    Python.list(1, 2, 3),
                    Python.list(),
                    n,
                    Python.list(4, 5),
                ),
                type=pa.list_(pa.int64()),
            ),
        )
    )
    _read_only(t)


def main() raises:
    TestSuite.run[__functions_in_module()]()
