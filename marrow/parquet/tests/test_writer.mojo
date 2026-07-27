from std.testing import assert_equal, assert_true, assert_false
from std.math import isnan, isinf
from std.python import Python, PythonObject
from std.os import remove
from ...parquet import (
    read_table,
    write_table,
    read_page_index,
    read_page_bounds,
)
from ...parquet.writer import FileWriter
from ...parquet.codecs import Compression, Encoding
from ...utils import Crc32
from ...tabular import Table
from ...c_data import CArrowArrayStream


def _to_marrow(py: PythonObject) raises -> Table:
    """A PyArrow table -> marrow Table via the Arrow C stream interface."""
    return CArrowArrayStream.from_pycapsule(
        py.__arrow_c_stream__(Python.none())
    ).to_table()


def _one_col(col: PythonObject) raises -> Table:
    """Single-column ("x") marrow Table from a PyArrow array."""
    var pa = Python.import_module("pyarrow")
    return _to_marrow(pa.table({"x": col}))


def _v2_roundtrip(codec: Compression) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var t = _to_marrow(
        pa.table(
            {
                "i": pa.array([1, None, 3, None, 5], type=pa.int64()),
                "s": pa.array(["a", "b", None, "d", "e"]),
            }
        )
    )
    var path = String("/tmp/marrow_v2.parquet")
    var w = FileWriter(codec, version=2)
    w.write(t, path)

    # the file declares format version 2
    var pf = pq.ParquetFile(path)
    assert_equal(Int(py=pf.metadata.format_version[0]), 2)

    # PyArrow reads marrow's v2 pages (validates the DataPageV2 layout)
    var back = pq.read_table(path)
    assert_true(Bool(back.column(0).to_pylist() == [1, None, 3, None, 5]))
    assert_true(Bool(back.column(1).to_pylist() == ["a", "b", None, "d", "e"]))

    # marrow round-trips (exercises the PAGE_DATA_V2 read branch)
    var mback = read_table(path)
    var b = mback.to_batches()[0].copy()
    assert_equal(b.columns[0].copy().as_int64().null_count(), 2)
    assert_equal(b.columns[0].copy().as_int64()[4].value(), 5)
    assert_equal(String(b.columns[1].copy().as_string()[0]), "a")
    remove(path)


def test_write_v2_snappy() raises:
    _v2_roundtrip(Compression.SNAPPY)


def test_write_v2_uncompressed() raises:
    _v2_roundtrip(Compression.UNCOMPRESSED)


def test_multiple_row_groups() raises:
    # 2500 rows, row_group_size 1000 -> 3 row groups
    var pa = Python.import_module("pyarrow")
    var t = _one_col(
        pa.array(Python.import_module("numpy").arange(2500), type=pa.int64())
    )
    var path = String("/tmp/marrow_rg.parquet")
    var w = FileWriter(Compression.SNAPPY)
    w.write(t, path, row_group_size=1000)

    # pyarrow sees 3 row groups
    var pq = Python.import_module("pyarrow.parquet")
    var pf = pq.ParquetFile(path)
    assert_equal(Int(py=pf.metadata.num_row_groups), 3)
    assert_equal(Int(py=pf.metadata.num_rows), 2500)

    # marrow reads all rows back
    var back = read_table(path)
    assert_equal(back.num_rows(), 2500)
    var b = back.to_batches()
    var total = 0
    for ref bat in b:
        total += bat.num_rows()
    assert_equal(total, 2500)
    remove(path)


def test_null_count_statistic() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var t = _one_col(pa.array([1, None, 3, None, 5], type=pa.int64()))
    var path = String("/tmp/marrow_stats.parquet")
    write_table(t, path)
    var pf = pq.ParquetFile(path)
    var stats = pf.metadata.row_group(0).column(0).statistics
    assert_equal(Int(py=stats.null_count), 2)
    remove(path)


# ---------------------------------------------------------------------------
# min/max statistics — PyArrow reads the bounds marrow writes. Covers the
# logical orderings that matter: signed vs unsigned ints, IEEE floats (with
# negatives), and byte-wise string ordering. The writer also emits column_orders
# (TypeDefinedOrder), so PyArrow trusts the unsigned/byte-array bounds.
# ---------------------------------------------------------------------------


def _col_stats(path: String, col: Int) raises -> PythonObject:
    var pq = Python.import_module("pyarrow.parquet")
    return pq.ParquetFile(path).metadata.row_group(0).column(col).statistics


def test_minmax_signed_int() raises:
    var pa = Python.import_module("pyarrow")
    var t = _one_col(pa.array([5, -1, None, 9, -3], type=pa.int64()))
    var path = String("/tmp/marrow_mm_int.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_equal(Int(py=s.min), -3)
    assert_equal(Int(py=s.max), 9)
    assert_equal(Int(py=s.null_count), 1)
    remove(path)


def test_minmax_unsigned_int() raises:
    # unsigned ordering: 3e9 > 1 even though it is negative as signed int32
    var pa = Python.import_module("pyarrow")
    var t = _one_col(pa.array([1, 3000000000, 2], type=pa.uint32()))
    var path = String("/tmp/marrow_mm_uint.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_equal(Int(py=s.min), 1)
    assert_equal(Int(py=s.max), 3000000000)
    remove(path)


def test_minmax_temporal() raises:
    var pa = Python.import_module("pyarrow")
    var dt = Python.import_module("datetime")
    var t = _one_col(pa.array([10, 3, 7, None, 1], type=pa.timestamp("us")))
    var path = String("/tmp/marrow_mm_ts.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    # min value 1us / max value 10us after the epoch
    assert_true(Bool(s.min == dt.datetime(1970, 1, 1, 0, 0, 0, 1)))
    assert_true(Bool(s.max == dt.datetime(1970, 1, 1, 0, 0, 0, 10)))
    assert_equal(Int(py=s.null_count), 1)
    remove(path)


def test_minmax_decimal128() raises:
    var pa = Python.import_module("pyarrow")
    var t = _one_col(
        pa.array(["1.50", "-3.25", "2.00", None, "0.75"]).cast(
            pa.decimal128(5, 2)
        )
    )
    var path = String("/tmp/marrow_mm_dec.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_equal(String(py=s.min), "-3.25")
    assert_equal(String(py=s.max), "2.00")
    remove(path)


def test_minmax_binary() raises:
    var pa = Python.import_module("pyarrow")
    var t = _one_col(
        pa.array(
            Python.list(
                Python.str("zoo").encode(),
                Python.str("abc").encode(),
                Python.none(),
                Python.str("mno").encode(),
            ),
            type=pa.binary(),
        )
    )
    var path = String("/tmp/marrow_mm_bin.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_true(Bool(s.min == Python.str("abc").encode()))
    assert_true(Bool(s.max == Python.str("zoo").encode()))
    remove(path)


def test_minmax_fixed_size_binary() raises:
    var pa = Python.import_module("pyarrow")
    var t = _one_col(
        pa.array(
            Python.list(
                Python.str("yy").encode(),
                Python.str("aa").encode(),
                Python.str("mm").encode(),
                Python.none(),
            ),
            type=pa.binary(2),
        )
    )
    var path = String("/tmp/marrow_mm_fsb.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_true(Bool(s.min == Python.str("aa").encode()))
    assert_true(Bool(s.max == Python.str("yy").encode()))
    remove(path)


def test_minmax_float_with_negatives() raises:
    var pa = Python.import_module("pyarrow")
    var t = _one_col(pa.array([1.5, -2.5, 3.25, None], type=pa.float64()))
    var path = String("/tmp/marrow_mm_float.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_true(Float64(py=s.min) == -2.5)
    assert_true(Float64(py=s.max) == 3.25)
    remove(path)


def test_minmax_string_lexicographic() raises:
    var pa = Python.import_module("pyarrow")
    var t = _one_col(pa.array(["banana", "apple", "cherry", None]))
    var path = String("/tmp/marrow_mm_str.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_equal(String(py=s.min), "apple")
    assert_equal(String(py=s.max), "cherry")
    assert_equal(Int(py=s.null_count), 1)
    remove(path)


def test_minmax_absent_for_all_null() raises:
    # an all-null column carries null_count but no min/max bound
    var pa = Python.import_module("pyarrow")
    var t = _one_col(pa.array([None, None, None], type=pa.int64()))
    var path = String("/tmp/marrow_mm_allnull.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_false(Bool(py=s.has_min_max))
    assert_equal(Int(py=s.null_count), 3)
    remove(path)


def test_narrow_int_roundtrip() raises:
    var pa = Python.import_module("pyarrow")
    var t = _to_marrow(
        pa.table(
            {
                "a": pa.array([-1, 2, -3], type=pa.int8()),
                "b": pa.array([10, 20, 30], type=pa.uint16()),
            }
        )
    )
    var path = String("/tmp/marrow_narrow.parquet")
    write_table(t, path)
    var back = read_table(path)
    var bat = back.to_batches()[0].copy()
    var ca = bat.columns[0].copy()
    assert_equal(ca.as_int8()[0].value(), -1)
    assert_equal(ca.as_int8()[2].value(), -3)
    var cb = bat.columns[1].copy()
    assert_equal(cb.as_uint16()[2].value(), 30)
    remove(path)


def test_write_empty_table() raises:
    # a zero-row table must write a valid file and read back empty
    var pa = Python.import_module("pyarrow")
    var t = _to_marrow(
        pa.table(
            {
                "i": pa.array([], type=pa.int64()),
                "s": pa.array([], type=pa.string()),
            }
        )
    )
    var path = String("/tmp/marrow_empty.parquet")
    write_table(t, path)

    # pyarrow reads it: 0 rows, schema preserved
    var pq = Python.import_module("pyarrow.parquet")
    var pf = pq.ParquetFile(path)
    assert_equal(Int(py=pf.metadata.num_rows), 0)

    var back = read_table(path)
    assert_equal(back.num_rows(), 0)
    assert_equal(back.num_columns(), 2)
    remove(path)


def test_write_all_null_roundtrip() raises:
    var pa = Python.import_module("pyarrow")
    var t = _one_col(pa.array([None, None, None, None, None], type=pa.int64()))
    var path = String("/tmp/marrow_allnull.parquet")
    write_table(t, path)
    var back = read_table(path)
    var c = back.to_batches()[0].copy().columns[0].copy()
    assert_equal(c.as_int64().null_count(), 5)

    # pyarrow agrees the column is entirely null
    var pq = Python.import_module("pyarrow.parquet")
    assert_equal(Int(py=pq.read_table(path).column(0).null_count), 5)
    remove(path)


def test_write_float_special_roundtrip() raises:
    # NaN / +Inf / -Inf must survive marrow write -> read byte-exact
    var pa = Python.import_module("pyarrow")
    var m = Python.import_module("math")
    var t = _one_col(
        pa.array(
            Python.list(m.nan, m.inf, -m.inf, -0.0, 3.25),
            type=pa.float64(),
        )
    )
    var path = String("/tmp/marrow_fspecial.parquet")
    write_table(t, path)
    var back = read_table(path)
    var c = back.to_batches()[0].copy().columns[0].copy()
    ref f = c.as_float64()
    assert_true(isnan(f[0].value()))
    assert_true(isinf(f[1].value()) and f[1].value() > 0)
    assert_true(isinf(f[2].value()) and f[2].value() < 0)
    assert_true(f[4].value() == 3.25)
    remove(path)


# ---------------------------------------------------------------------------
# Nested write — lists and maps (Dremel shredding). Oracle: PyArrow reads back
# marrow's written file and must match the original.
# ---------------------------------------------------------------------------


def _check_write(want: PythonObject, compression: Compression) raises:
    var pq = Python.import_module("pyarrow.parquet")
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_nested_write.parquet")
    write_table(t, path, compression)
    var back = pq.read_table(path)
    assert_true(
        Bool(back.column(0).to_pylist() == want.column(0).to_pylist()),
        "PyArrow read of marrow's write mismatched the original",
    )
    remove(path)


def _list_table(data: PythonObject, dtype: PythonObject) raises -> PythonObject:
    var pa = Python.import_module("pyarrow")
    return pa.table({"v": pa.array(data, type=dtype)})


def test_write_list_int() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table([[1, 2, 3], [], None, [4, 5]], pa.list_(pa.int64())),
        Compression.UNCOMPRESSED,
    )


def test_write_list_int_snappy() raises:
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    # each row is list(range(i % 6)), or None every 7th row
    var data = Python.list()
    for i in range(80):
        if i % 7 == 0:
            data.append(Python.none())
        else:
            data.append(np.arange(i % 6).tolist())
    _check_write(_list_table(data, pa.list_(pa.int32())), Compression.SNAPPY)


def test_write_list_string() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            [["a", "bb"], [], None, ["ccc", None, "d"]], pa.list_(pa.string())
        ),
        Compression.UNCOMPRESSED,
    )


def test_write_list_of_list() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            [[[1, 2], [3]], [], None, [[4], []]],
            pa.list_(pa.list_(pa.int64())),
        ),
        Compression.UNCOMPRESSED,
    )


def test_write_map_string_int() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            [{"a": 1, "b": 2}, {}, None, {"c": 3}],
            pa.map_(pa.string(), pa.int64()),
        ),
        Compression.UNCOMPRESSED,
    )


def test_write_map_nullable_values() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            [{"a": 1, "b": None}, {"c": 3}, None],
            pa.map_(pa.string(), pa.int64()),
        ),
        Compression.SNAPPY,
    )


def test_write_map_int_key() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            [{1: "x", 2: "y"}, None, {3: "z"}],
            pa.map_(pa.int32(), pa.string()),
        ),
        Compression.UNCOMPRESSED,
    )


def test_write_map_multichunk() raises:
    # A map Table with several chunks exercises combine_chunks -> concat ->
    # MapBuilder before the write.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var mt = pa.map_(pa.string(), pa.int64())
    var b1 = pa.RecordBatch.from_arrays(
        [pa.array([{"a": 1}, {}], type=mt)], names=["v"]
    )
    var b2 = pa.RecordBatch.from_arrays(
        [pa.array([None, {"b": 2, "c": 3}], type=mt)], names=["v"]
    )
    var want = pa.Table.from_batches([b1, b2])
    assert_true(Bool(want.column(0).num_chunks > 1))
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_map_multichunk.parquet")
    write_table(t, path, Compression.UNCOMPRESSED)
    var back = pq.read_table(path)
    assert_true(
        Bool(back.column(0).to_pylist() == want.column(0).to_pylist()),
        "multi-chunk map write mismatch",
    )
    remove(path)


def test_write_list_v2() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var want = _list_table([[1, 2], [], None, [3]], pa.list_(pa.int64()))
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_nested_write_v2.parquet")
    var w = FileWriter(Compression.UNCOMPRESSED, version=2)
    w.write(t, path)
    var back = pq.read_table(path)
    assert_true(
        Bool(back.column(0).to_pylist() == want.column(0).to_pylist()),
        "v2 nested write mismatch",
    )
    remove(path)


# ---------------------------------------------------------------------------
# Encodings on write — dictionary (default), delta, byte-stream-split.
# ---------------------------------------------------------------------------


def _col_encodings(path: String, col: Int) raises -> PythonObject:
    var pq = Python.import_module("pyarrow.parquet")
    return pq.ParquetFile(path).metadata.row_group(0).column(col).encodings


def test_write_dictionary_encoding() raises:
    # A low-cardinality column dictionary-encodes by default; pyarrow sees the
    # RLE_DICTIONARY encoding and reads back the exact values.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var strs = Python.list("a", "b", "a", "a", "c", "b") * 20
    var ints = Python.list(1, 2, 1, 1, 3, 2) * 20
    var t = _to_marrow(
        pa.table({"s": pa.array(strs), "i": pa.array(ints, type=pa.int64())})
    )
    var path = String("/tmp/marrow_dict.parquet")
    write_table(
        t, path, Compression.UNCOMPRESSED
    )  # use_dictionary defaults True

    assert_true(
        "DICTIONARY" in String(_col_encodings(path, 0)),
        "string column not dictionary-encoded",
    )
    assert_true(
        "DICTIONARY" in String(_col_encodings(path, 1)),
        "int column not dictionary-encoded",
    )

    var back = pq.read_table(path)
    assert_true(
        Bool(back.column(0).to_pylist() == strs), "dict string values mismatch"
    )
    assert_true(
        Bool(back.column(1).to_pylist() == ints), "dict int values mismatch"
    )
    remove(path)


def test_write_dictionary_disabled_is_plain() raises:
    var pa = Python.import_module("pyarrow")
    var t = _one_col(pa.array(["a", "b", "a", "c"]))
    var path = String("/tmp/marrow_plain.parquet")
    write_table(t, path, Compression.UNCOMPRESSED, use_dictionary=False)
    assert_false(
        "DICTIONARY" in String(_col_encodings(path, 0)),
        "expected PLAIN, got a dictionary",
    )
    remove(path)


def test_write_dictionary_nullable() raises:
    # nulls are placed by the definition levels; only present values are indexed.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var t = _one_col(pa.array(["x", None, "y", "x", None, "y"]))
    var path = String("/tmp/marrow_dict_null.parquet")
    write_table(t, path, Compression.SNAPPY)
    var back = pq.read_table(path)
    assert_true(
        Bool(back.column(0).to_pylist() == ["x", None, "y", "x", None, "y"]),
        "nullable dict values mismatch",
    )
    remove(path)


def _encoding_check(
    col: PythonObject, encoding: Encoding, want_enc: String
) raises:
    """Write a single column ("x") with a forced value encoding, assert PyArrow
    sees that encoding and both PyArrow and marrow read back the values."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var want = pa.table({"x": col})
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_enc.parquet")
    write_table(t, path, Compression.UNCOMPRESSED, encoding=encoding)
    assert_true(
        want_enc in String(_col_encodings(path, 0)),
        "expected " + want_enc + " encoding",
    )
    var back = pq.read_table(path)
    assert_true(
        Bool(back.column(0).to_pylist() == want.column(0).to_pylist()),
        "value mismatch (PyArrow read of marrow's write)",
    )
    # marrow also reads its own output
    var mback = read_table(path)
    var mpa = pa.RecordBatchReader._import_from_c_capsule(
        CArrowArrayStream.from_batches(
            mback.schema.copy(), mback.to_batches()
        ).to_pycapsule()
    ).read_all()
    assert_true(
        Bool(mpa.column(0).to_pylist() == want.column(0).to_pylist()),
        "value mismatch (marrow round-trip)",
    )
    remove(path)


def test_write_delta_binary_packed_int() raises:
    var pa = Python.import_module("pyarrow")
    _encoding_check(
        pa.array([1, 3, 6, 10, 15, 21, -4, 100, None, 7], type=pa.int64()),
        Encoding.DELTA_BINARY_PACKED,
        "DELTA_BINARY_PACKED",
    )


def test_write_delta_int32_narrow() raises:
    var pa = Python.import_module("pyarrow")
    _encoding_check(
        pa.array(Python.import_module("numpy").arange(200), type=pa.int32()),
        Encoding.DELTA_BINARY_PACKED,
        "DELTA_BINARY_PACKED",
    )


def test_write_delta_byte_array_string() raises:
    var pa = Python.import_module("pyarrow")
    _encoding_check(
        pa.array(
            ["apple", "apply", "apricot", None, "banana", "band", "bandana"]
        ),
        Encoding.DELTA_BYTE_ARRAY,
        "DELTA_BYTE_ARRAY",
    )


def test_write_delta_length_byte_array_string() raises:
    var pa = Python.import_module("pyarrow")
    _encoding_check(
        pa.array(["a", "bb", "ccc", "dddd", None, "ef"]),
        Encoding.DELTA_LENGTH_BYTE_ARRAY,
        "DELTA_LENGTH_BYTE_ARRAY",
    )


def test_write_byte_stream_split_float() raises:
    var pa = Python.import_module("pyarrow")
    _encoding_check(
        pa.array([1.5, -2.25, 3.0, None, 100.125, 0.0], type=pa.float64()),
        Encoding.BYTE_STREAM_SPLIT,
        "BYTE_STREAM_SPLIT",
    )


def test_write_byte_stream_split_float32() raises:
    var pa = Python.import_module("pyarrow")
    _encoding_check(
        pa.array([1.0, 2.5, -3.75, 4.0], type=pa.float32()),
        Encoding.BYTE_STREAM_SPLIT,
        "BYTE_STREAM_SPLIT",
    )


def test_write_per_column_encoding() raises:
    # column_encodings overrides per column; other columns keep the default dict.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var want = pa.table(
        {
            "i": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "s": pa.array(["a", "b", "a", "c", "a"]),
        }
    )
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_percol.parquet")
    var enc_map = Dict[String, Encoding]()
    enc_map["i"] = Encoding.DELTA_BINARY_PACKED
    write_table(t, path, Compression.UNCOMPRESSED, column_encodings=enc_map^)
    assert_true(
        "DELTA_BINARY_PACKED" in String(_col_encodings(path, 0)),
        "column i should be delta",
    )
    assert_true(
        "DICTIONARY" in String(_col_encodings(path, 1)),
        "column s should keep the default dictionary",
    )
    var back = pq.read_table(path)
    assert_true(Bool(back.column(0).to_pylist() == want.column(0).to_pylist()))
    assert_true(Bool(back.column(1).to_pylist() == want.column(1).to_pylist()))
    remove(path)


def test_write_dictionary_high_cardinality_falls_back() raises:
    # > 131072 distinct int64 makes the dictionary page exceed 1 MB, so the
    # column falls back to PLAIN instead of a dictionary larger than the data.
    var pa = Python.import_module("pyarrow")
    var t = _one_col(
        pa.array(Python.import_module("numpy").arange(140000), type=pa.int64())
    )
    var path = String("/tmp/marrow_hicard.parquet")
    write_table(t, path, Compression.UNCOMPRESSED)  # dictionary requested
    assert_false(
        "DICTIONARY" in String(_col_encodings(path, 0)),
        "high-cardinality column should fall back to PLAIN",
    )
    var back = read_table(path)
    assert_equal(back.num_rows(), 140000)
    remove(path)


# ---------------------------------------------------------------------------
# Temporal write — date/time/timestamp (incl. nanoseconds and timezone, which
# need the LogicalType union, not just the legacy ConvertedType).
# ---------------------------------------------------------------------------


def test_write_temporal() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var d = Python.import_module("datetime")
    var want = pa.table(
        {
            "date": pa.array(
                [d.date(2020, 1, 1), None, d.date(2021, 6, 15)],
                type=pa.date32(),
            ),
            "ts_ms": pa.array([10, 20, 30], type=pa.timestamp("ms")),
            "ts_us": pa.array([1000, None, 2000], type=pa.timestamp("us")),
            "ts_ns": pa.array([1, 2, 3], type=pa.timestamp("ns")),
            "ts_tz": pa.array(
                [100, 200, 300], type=pa.timestamp("us", tz="UTC")
            ),
            "time_ms": pa.array([1, 2, 3], type=pa.time32("ms")),
            "time_us": pa.array([5, 6, 7], type=pa.time64("us")),
            "time_ns": pa.array([8, 9, 10], type=pa.time64("ns")),
        }
    )
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_temporal.parquet")
    write_table(t, path, Compression.UNCOMPRESSED)

    # PyArrow reads back every type (unit + tz preserved) and every value.
    var back = pq.read_table(path)
    for i in range(Int(py=want.num_columns)):
        assert_true(
            Bool(back.schema.field(i).type == want.schema.field(i).type),
            "temporal type mismatch at column " + String(i),
        )
        assert_true(
            Bool(back.column(i).equals(want.column(i))),
            "temporal value mismatch at column " + String(i),
        )
    remove(path)


# ---------------------------------------------------------------------------
# Decimal + fixed-size-binary write (FIXED_LEN_BYTE_ARRAY).
# ---------------------------------------------------------------------------


def test_write_decimal128() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var dec = Python.import_module("decimal")
    var want = pa.table(
        {
            "d": pa.array(
                [dec.Decimal("1.23"), None, dec.Decimal("-45.67")],
                type=pa.decimal128(10, 2),
            )
        }
    )
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_decimal.parquet")
    write_table(t, path, Compression.UNCOMPRESSED)
    var back = pq.read_table(path)
    assert_true(
        Bool(back.schema.field(0).type == want.schema.field(0).type),
        "decimal type mismatch",
    )
    assert_true(
        Bool(back.column(0).equals(want.column(0))), "decimal value mismatch"
    )
    remove(path)


def test_write_decimal256() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var dec = Python.import_module("decimal")
    var want = pa.table(
        {
            "d": pa.array(
                [dec.Decimal("123456789.123456789"), dec.Decimal("-1")],
                type=pa.decimal256(50, 9),
            )
        }
    )
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_decimal256.parquet")
    write_table(t, path, Compression.UNCOMPRESSED)
    var back = pq.read_table(path)
    assert_true(Bool(back.column(0).equals(want.column(0))), "dec256 mismatch")
    remove(path)


def test_write_fixed_size_binary() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var want = pa.table(
        {
            "b": pa.array(
                Python.list(
                    Python.str("abc").encode(),
                    Python.none(),
                    Python.str("xyz").encode(),
                ),
                type=pa.binary(3),
            )
        }
    )
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_fsb.parquet")
    write_table(t, path, Compression.UNCOMPRESSED)
    var back = pq.read_table(path)
    assert_true(
        Bool(back.schema.field(0).type == want.schema.field(0).type),
        "fixed_size_binary type mismatch",
    )
    assert_true(Bool(back.column(0).equals(want.column(0))), "fsb mismatch")
    remove(path)


def test_write_binary() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var want = pa.table(
        {
            "b": pa.array(
                Python.list(
                    Python.str("abc").encode(),
                    Python.none(),
                    Python.str("xyz").encode(),
                    Python.str("").encode(),
                ),
                type=pa.binary(),
            )
        }
    )
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_binary.parquet")
    write_table(t, path, Compression.UNCOMPRESSED)
    var back = pq.read_table(path)
    assert_true(Bool(back.column(0).equals(want.column(0))), "binary mismatch")
    remove(path)


def test_write_large_binary_and_string() raises:
    # large_binary / large_string carry no distinct Parquet physical type, so
    # they land as BYTE_ARRAY and read back as binary / string — the values must
    # match after a cast.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var want = pa.table(
        {
            "lb": pa.array(
                Python.list(
                    Python.str("hello").encode(),
                    Python.str("world").encode(),
                    Python.none(),
                ),
                type=pa.large_binary(),
            ),
            "ls": pa.array(
                Python.list("aa", "bb", Python.none()), type=pa.large_string()
            ),
        }
    )
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_large.parquet")
    write_table(t, path, Compression.UNCOMPRESSED)
    var back = pq.read_table(path)
    assert_true(
        Bool(back.column(0).cast(pa.large_binary()).equals(want.column(0))),
        "large_binary mismatch",
    )
    assert_true(
        Bool(back.column(1).cast(pa.large_string()).equals(want.column(1))),
        "large_string mismatch",
    )
    remove(path)


def _compression_roundtrip(
    codec: Compression, py_name: String, check_name: Bool = True
) raises:
    """marrow writes with `codec`; PyArrow reads it back, marrow reads it back,
    and marrow reads a PyArrow file written with the same codec."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    # compressible payload so the codec actually kicks in
    var ints = Python.list()
    var strs = Python.list()
    for i in range(500):
        ints.append(i % 17)
        strs.append(Python.str("val_") + Python.str(i % 23))
    var want = pa.table(
        {"i": pa.array(ints, type=pa.int32()), "s": pa.array(strs)}
    )
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_comp.parquet")
    write_table(t, path, codec, use_dictionary=False)

    # PyArrow reads marrow's compressed pages, with the right codec advertised
    # (the deprecated LZ4 code-5 is displayed as "UNKNOWN" by PyArrow, which
    # only names the codecs it still writes — the decompress path still works).
    var pf = pq.ParquetFile(path)
    if check_name:
        assert_equal(
            String(py=pf.metadata.row_group(0).column(0).compression), py_name
        )
    var back = pq.read_table(path)
    assert_true(Bool(back.column(0).equals(want.column(0))), "int mismatch")
    assert_true(Bool(back.column(1).equals(want.column(1))), "str mismatch")

    # marrow round-trips its own file
    var mback = read_table(path)
    assert_equal(mback.num_rows(), 500)

    # marrow reads a PyArrow file written with the same codec
    var pypath = String("/tmp/pyarrow_comp.parquet")
    pq.write_table(
        want, pypath, compression=py_name.lower(), use_dictionary=False
    )
    var mpy = read_table(pypath)
    var b = mpy.to_batches()[0].copy()
    assert_equal(b.columns[0].copy().as_int32()[3].value(), 3)
    remove(path)
    remove(pypath)


def test_compression_gzip() raises:
    _compression_roundtrip(Compression.GZIP, "GZIP")


def test_compression_brotli() raises:
    _compression_roundtrip(Compression.BROTLI, "BROTLI")


def test_compression_lz4() raises:
    # PyArrow displays the deprecated code-5 LZ4 as "UNKNOWN"; skip the name
    # check and rely on the round-trip (its `compression='lz4'` is LZ4_RAW).
    _compression_roundtrip(Compression.LZ4, "LZ4", check_name=False)


def test_compression_zstd() raises:
    _compression_roundtrip(Compression.ZSTD, "ZSTD")


def _page_rows(path: String, col: Int) raises -> Int:
    """Total rows across a column's data pages (must equal the row count)."""
    var pbs = read_page_bounds(path)
    var total = 0
    for p in range(len(pbs[0][col])):
        total += pbs[0][col][p].copy().num_rows
    return total


def test_page_split_flat() raises:
    # 50 000 rows > the 20 000 rows-per-page cap -> multiple data pages
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var ints = Python.list()
    for i in range(50000):
        ints.append(i)
    var want = pa.table(Python.dict(i=pa.array(ints, type=pa.int64())))
    var path = String("/tmp/marrow_split_flat.parquet")
    write_table(_to_marrow(want), path, use_dictionary=False)

    # PyArrow reads every row back
    assert_true(Bool(pq.read_table(path).column(0).equals(want.column(0))))
    # marrow round-trips
    assert_equal(read_table(path).num_rows(), 50000)

    # the chunk is split into >1 page and the pages tile all rows
    var pbs = read_page_bounds(path)
    assert_true(len(pbs[0][0]) > 1)
    assert_equal(_page_rows(path, 0), 50000)
    # per-page bounds are populated; first page starts at value 0
    assert_equal(pbs[0][0][0].copy().min.value().as_int64().value(), 0)

    # OffsetIndex first_row_index is a cumulative, increasing sequence
    var pi = read_page_index(path)
    ref oi = pi[0][0].offset_index.value()
    var expected = 0
    for k in range(len(oi.page_locations)):
        assert_equal(oi.page_locations[k].first_row_index, expected)
        expected += pbs[0][0][k].copy().num_rows
    remove(path)


def test_page_split_dictionary() raises:
    # low-cardinality strings, dictionary-encoded: one shared dictionary page
    # followed by several RLE_DICTIONARY data pages
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var strs = Python.list()
    for i in range(50000):
        strs.append(Python.str("k") + Python.str(i % 100))
    var want = pa.table(Python.dict(s=pa.array(strs)))
    var path = String("/tmp/marrow_split_dict.parquet")
    write_table(_to_marrow(want), path, use_dictionary=True)

    assert_true(Bool(pq.read_table(path).column(0).equals(want.column(0))))
    assert_equal(read_table(path).num_rows(), 50000)
    var pbs = read_page_bounds(path)
    assert_true(len(pbs[0][0]) > 1)
    assert_equal(_page_rows(path, 0), 50000)
    # a single dictionary page precedes the data pages
    var cc = pq.ParquetFile(path).metadata.row_group(0).column(0)
    assert_true(Bool(cc.dictionary_page_offset < cc.data_page_offset))
    remove(path)


def test_writer_float16_roundtrip() raises:
    # float16 is physically FIXED_LEN_BYTE_ARRAY(2) + FLOAT16 logical; verify
    # both encodings round-trip through marrow write and read, incl. nulls and
    # the signed-zero-normalised min/max statistic.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var want = pa.table(
        Python.dict(
            h=pa.array(
                Python.list(1.5, 2.5, 3.0, Python.none(), 4.5, -0.0),
                type=pa.float16(),
            )
        )
    )
    # a marrow float16 Table via PyArrow write -> marrow read (a list-backed
    # PyArrow array's buffer is 64-byte aligned, unlike a numpy-backed one that
    # the C-import would reject).
    var src = String("/tmp/marrow_f16_src.parquet")
    pq.write_table(want, src, use_dictionary=False)
    var mt = read_table(src)
    for use_dict in [False, True]:
        var path = String("/tmp/marrow_f16.parquet")
        write_table(mt, path, use_dictionary=use_dict)
        # pyarrow reads marrow's file back to a halffloat column
        var back = pq.read_table(path)
        assert_true(Bool(back.schema.field(0).type == pa.float16()))
        assert_true(
            Bool(back.column(0).to_pylist() == want.column(0).to_pylist())
        )
        # marrow reads its own file
        assert_equal(read_table(path).num_rows(), 6)
        # statistics: min normalises to -0.0 (0x8000), max is 4.5 (0x4480)
        var s = pq.ParquetFile(path).metadata.row_group(0).column(0).statistics
        assert_true(
            Bool(
                s.min
                == Python.import_module("builtins").bytes(Python.list(0, 0x80))
            )
        )
        assert_true(
            Bool(
                s.max
                == Python.import_module("builtins").bytes(
                    Python.list(0x80, 0x44)
                )
            )
        )
        assert_equal(Int(py=s.null_count), 1)
        remove(path)
    remove(src)


def test_key_value_metadata() raises:
    # user schema metadata round-trips; PyArrow's ARROW:schema is dropped on
    # write (marrow writes its own Parquet schema).
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var md = Python.dict()
    md[Python.str("hello").encode()] = Python.str("world").encode()
    md[Python.str("team").encode()] = Python.str("marrow").encode()
    var sch = pa.schema(Python.list(pa.field("x", pa.int64()))).with_metadata(
        md
    )
    var want = pa.table(
        Python.dict(x=pa.array(Python.list(1, 2, 3))), schema=sch
    )
    var path = String("/tmp/marrow_kv.parquet")
    pq.write_table(want, path)

    # marrow surfaces the file's key/value metadata on the schema
    var t = read_table(path)
    assert_equal(t.schema.metadata["hello"], String("world"))
    assert_equal(t.schema.metadata["team"], String("marrow"))

    # marrow writes it back; PyArrow reads the user keys, ARROW:schema is gone
    var dst = String("/tmp/marrow_kv_out.parquet")
    write_table(t, dst)
    var back = pq.read_table(dst)
    assert_true(
        Bool(
            back.schema.metadata[Python.str("hello").encode()]
            == Python.str("world").encode()
        )
    )
    assert_false(
        Bool(Python.str("ARROW:schema").encode() in back.schema.metadata)
    )
    remove(path)
    remove(dst)


def _decimal_int_backed(patype: PythonObject, phys: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var vals = pa.array(
        Python.list("1.50", "-3.25", "2.00", Python.none(), "0.75")
    ).cast(patype)
    var t = _one_col(vals)
    var a = String("/tmp/marrow_dec_a.parquet")
    write_table(t, a, use_dictionary=False)
    # marrow encodes decimal32/decimal64 with the integer physical type
    assert_equal(
        String(
            py=pq.ParquetFile(a).metadata.row_group(0).column(0).physical_type
        ),
        phys,
    )
    # marrow must read its own int-backed decimal correctly (not as a big-endian
    # FLBA): read -> write -> PyArrow-read must recover the original values.
    var b = String("/tmp/marrow_dec_b.parquet")
    write_table(read_table(a), b, use_dictionary=False)
    assert_true(
        Bool(pq.read_table(b).column(0).to_pylist() == vals.to_pylist())
    )
    remove(a)
    remove(b)


def test_decimal_int_backed_roundtrip() raises:
    var pa = Python.import_module("pyarrow")
    _decimal_int_backed(pa.decimal32(9, 2), "INT32")
    _decimal_int_backed(pa.decimal64(15, 2), "INT64")


def test_distinct_count_statistic() raises:
    # a dictionary-encoded chunk knows its distinct (non-null) value count — the
    # dictionary size — and writes it as Statistics.distinct_count; a PLAIN chunk
    # leaves it absent.
    var pa = Python.import_module("pyarrow")
    var ints = Python.list()
    for i in range(300):
        ints.append(i % 40)
    var t = _one_col(pa.array(ints, type=pa.int64()))

    var dpath = String("/tmp/marrow_distinct.parquet")
    write_table(t, dpath, use_dictionary=True)
    assert_equal(Int(py=_col_stats(dpath, 0).distinct_count), 40)
    remove(dpath)

    var ppath = String("/tmp/marrow_distinct_plain.parquet")
    write_table(t, ppath, use_dictionary=False)
    assert_false(Bool(_col_stats(ppath, 0).distinct_count))  # None
    remove(ppath)


def test_page_checksum() raises:
    # write_page_checksum attaches a CRC-32 to every page. The check vector
    # locks the algorithm; PyArrow's verified read proves the written CRC is
    # correct; marrow reading its own file exercises the verify path.
    assert_equal(Int(Crc32.compute(String("123456789").as_bytes())), 0xCBF43926)

    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var t = _one_col(
        pa.array(Python.import_module("numpy").arange(200), type=pa.int64())
    )
    for ver in [1, 2]:
        var path = String("/tmp/marrow_crc.parquet")
        var w = FileWriter(
            Compression.SNAPPY, version=ver, write_page_checksum=True
        )
        w.write(t, path)
        # PyArrow verifies the checksum -> marrow's CRC matches the spec
        var back = pq.read_table(path, page_checksum_verification=True)
        assert_equal(Int(py=back.num_rows), 200)
        # marrow reads its own checksummed file (runs the verify branch)
        assert_equal(read_table(path).num_rows(), 200)
        remove(path)
