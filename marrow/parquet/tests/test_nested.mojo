# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Nested reconstruction: structs, single-level lists, and arbitrarily nested
lists/structs (any depth, struct-level nulls, encodings inside lists).

The arbitrary-nesting cases give a plain Python data literal plus the Arrow type
built with the `pyarrow` module object, and compare Marrow's read to PyArrow's
own read of the same file.
"""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python, PythonObject
from std.os.path import join
from ...utils.testing import ScratchDir
from ...parquet import (
    ParquetFile,
    read_table,
    write_table,
)
from ...tabular import Table
from ...c_data import CArrowArrayStream
from ... import dtypes as dt
from ...dtypes import FIELD_ID_KEY, Field
from ...schema import Schema
from ...parquet.format import (
    FileMetaData,
    ThriftCompactReader,
    ThriftCompactWriter,
)
from ...parquet.schema import SchemaMapping


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------


def _pa_struct_table() raises -> PythonObject:
    var pa = Python.import_module("pyarrow")
    return pa.table(
        Python.dict(
            s=pa.array(
                Python.list(
                    Python.dict(a=1, x=1.5),
                    Python.dict(a=2, x=2.5),
                    Python.dict(a=3, x=3.5),
                ),
                type=pa.struct(
                    Python.list(
                        pa.field("a", pa.int64()),
                        pa.field("x", pa.float64()),
                    )
                ),
            ),
            k=pa.array(Python.list(10, 20, 30), type=pa.int64()),
        )
    )


def test_read_struct() raises:
    var pq = Python.import_module("pyarrow.parquet")
    with ScratchDir() as dir:
        var path = join(dir, "marrow_nested_read.parquet")
        pq.write_table(_pa_struct_table(), path, compression="snappy")

        var t = read_table(path)
        assert_equal(t.num_rows(), 3)
        assert_equal(t.num_columns(), 2)
        assert_equal(t.column_names()[0], "s")

        var b = t.to_batches()[0].copy()
        var sc = b.columns[0].copy()
        ref st = sc.as_struct()
        assert_equal(len(st.children), 2)
        var ca = st.children[0].copy()
        ref col_a = ca.as_int64()
        assert_equal(col_a[0].value(), 1)
        assert_equal(col_a[2].value(), 3)
        var cx = st.children[1].copy()
        ref col_x = cx.as_float64()
        assert_true(col_x[1].value() == 2.5)


def test_roundtrip_struct_via_pyarrow() raises:
    var pq = Python.import_module("pyarrow.parquet")
    with ScratchDir() as dir:
        var src = join(dir, "marrow_nested_src.parquet")
        pq.write_table(_pa_struct_table(), src, compression="none")

        # marrow read -> marrow write -> pyarrow read
        var t = read_table(src)
        var dst = join(dir, "marrow_nested_dst.parquet")
        write_table(t, dst)

        var back = pq.read_table(dst)
        assert_equal(Int(py=back.num_rows), 3)
        var s_col = back.column("s").to_pylist()
        assert_equal(Int(py=s_col[0]["a"]), 1)
        assert_true(Float64(py=s_col[2]["x"]) == 3.5)
        assert_equal(Int(py=back.column("k").to_pylist()[1]), 20)


# ---------------------------------------------------------------------------
# Single-level lists
# ---------------------------------------------------------------------------


def test_read_list_int() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(
                Python.list(
                    Python.list(1, 2, 3),
                    Python.list(),
                    Python.list(4, 5),
                    Python.list(6),
                ),
                type=pa.list_(pa.int64()),
            ),
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_list_int.parquet")
        pq.write_table(tbl, path, compression="snappy")

        var t = read_table(path)
        assert_equal(t.num_rows(), 4)
        var b = t.to_batches()[0].copy()
        var col = b.columns[0].copy()
        ref lst = col.as_list()
        # offsets: [0,3,3,5,6]
        assert_equal(lst[0].value().length(), 3)
        assert_equal(lst[1].value().length(), 0)  # empty list
        assert_equal(lst[2].value().length(), 2)
        var elem0 = lst[0].value()
        assert_equal(elem0.as_int64()[0].value(), 1)
        assert_equal(elem0.as_int64()[2].value(), 3)


def test_read_list_string_with_nulls() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(
                Python.list(
                    Python.list("a", "bb"),
                    Python.none(),  # null list
                    Python.list("ccc"),
                ),
                type=pa.list_(pa.string()),
            ),
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_list_str.parquet")
        pq.write_table(tbl, path, compression="none")

        var t = read_table(path)
        assert_equal(t.num_rows(), 3)
        var b = t.to_batches()[0].copy()
        var col = b.columns[0].copy()
        ref lst = col.as_list()
        assert_true(lst.is_valid(0))
        assert_false(lst.is_valid(1))  # null list
        assert_true(lst.is_valid(2))
        var e0 = lst[0].value()
        assert_equal(String(e0.as_string()[0]), "a")
        assert_equal(String(e0.as_string()[1]), "bb")


# ---------------------------------------------------------------------------
# Arbitrary nesting — compared against PyArrow's own read
# ---------------------------------------------------------------------------


def _to_pa(var t: Table) raises -> PythonObject:
    var caps = CArrowArrayStream.from_batches(
        t.schema.copy(), t.to_batches()
    ).to_pycapsule()
    var pa = Python.import_module("pyarrow")
    return pa.RecordBatchReader._import_from_c_capsule(caps).read_all()


def _check(
    data: PythonObject,
    dtype: PythonObject,
    compression: String,
    encoding: String = "",
) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var want = pa.table(Python.dict(v=pa.array(data, type=dtype)))
    with ScratchDir() as dir:
        var path = join(dir, "marrow_nested.parquet")
        if encoding != "":
            pq.write_table(
                want,
                path,
                compression=compression,
                use_dictionary=False,
                column_encoding=encoding,
            )
        else:
            pq.write_table(want, path, compression=compression)
        # oracle is PyArrow's own read of the same file
        var got = _to_pa(read_table(path))
        assert_true(
            Bool(
                got.column(0).to_pylist()
                == pq.read_table(path).column(0).to_pylist()
            ),
            "value mismatch",
        )


def _struct(*fields: PythonObject) raises -> PythonObject:
    var pa = Python.import_module("pyarrow")
    var fs = Python.list()
    for f in fields:
        fs.append(f)
    return pa.struct(fs)


def test_list_of_struct() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(Python.dict(a=1, b="x"), Python.dict(a=2, b="y")),
            Python.list(),
            Python.none(),
            Python.list(Python.dict(a=3, b="z")),
        ),
        pa.list_(
            _struct(pa.field("a", pa.int64()), pa.field("b", pa.string()))
        ),
        "none",
    )


def test_list_of_struct_snappy() raises:
    var pa = Python.import_module("pyarrow")
    # row i is [{'x': i, 'y': 2*i}] repeated (i % 3) times
    var data = Python.list()
    for i in range(50):
        var row = Python.list()
        for _ in range(i % 3):
            row.append(Python.dict(x=i, y=i * 2))
        data.append(row)
    _check(
        data,
        pa.list_(_struct(pa.field("x", pa.int32()), pa.field("y", pa.int64()))),
        "snappy",
    )


def test_nullable_struct() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.dict(a=1, b="x"),
            Python.none(),
            Python.dict(a=Python.none(), b="z"),
            Python.dict(a=4, b=Python.none()),
            Python.none(),
        ),
        _struct(pa.field("a", pa.int64()), pa.field("b", pa.string())),
        "none",
    )


def test_nullable_struct_of_struct() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.dict(p=Python.dict(x=1)),
            Python.none(),
            Python.dict(p=Python.none()),
            Python.dict(p=Python.dict(x=4)),
        ),
        _struct(pa.field("p", _struct(pa.field("x", pa.int64())))),
        "snappy",
    )


def test_list_of_list() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(Python.list(1, 2), Python.list(3)),
            Python.list(Python.list(4)),
            Python.none(),
            Python.list(),
        ),
        pa.list_(pa.list_(pa.int64())),
        "none",
    )


def test_list_of_list_of_list() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(Python.list(Python.list(1), Python.list(2, 3))),
            Python.list(Python.list(Python.list(4))),
            Python.list(),
        ),
        pa.list_(pa.list_(pa.list_(pa.int64()))),
        "snappy",
    )


def test_list_of_list_of_struct() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(
                Python.list(Python.dict(a=1)),
                Python.list(Python.dict(a=2), Python.dict(a=3)),
            ),
            Python.list(),
        ),
        pa.list_(pa.list_(_struct(pa.field("a", pa.int64())))),
        "none",
    )


def test_struct_with_list_child() raises:
    # nullable struct whose field is a list; includes a null struct row
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.dict(xs=Python.list(1, 2)),
            Python.dict(xs=Python.list()),
            Python.none(),
        ),
        _struct(pa.field("xs", pa.list_(pa.int64()))),
        "none",
    )


def test_list_of_nullable_struct() raises:
    # struct nulls *inside* a list
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(Python.dict(a=1), Python.none(), Python.dict(a=3)),
            Python.list(),
            Python.none(),
        ),
        pa.list_(_struct(pa.field("a", pa.int64()))),
        "snappy",
    )


def test_list_of_bool() raises:
    # booleans inside a list — the nested path used to lack a bool decoder
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(True, False),
            Python.list(Python.none(), True),
            Python.none(),
            Python.list(),
        ),
        pa.list_(pa.bool_()),
        "none",
    )


def test_list_of_delta_int() raises:
    # a DELTA_BINARY_PACKED-encoded list element — the nested path used to read
    # it as PLAIN and crash; it now shares the flat path's decoders
    var pa = Python.import_module("pyarrow")
    var data = Python.list()
    for i in range(40):
        data.append(Python.list(i, i * 2, i - 5))
    _check(data, pa.list_(pa.int64()), "snappy", "DELTA_BINARY_PACKED")


def test_fsb_delta_byte_array() raises:
    # FIXED_LEN_BYTE_ARRAY under DELTA_BYTE_ARRAY / BYTE_STREAM_SPLIT — flat and
    # nested (list element). PyArrow emits these for fixed_size_binary/decimal.
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.str("abcd").encode(),
            Python.str("efgh").encode(),
            Python.none(),
            Python.str("ijkl").encode(),
        ),
        pa.binary(4),
        "none",
        "DELTA_BYTE_ARRAY",
    )


def test_fsb_byte_stream_split() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.str("abcd").encode(),
            Python.str("efgh").encode(),
            Python.none(),
            Python.str("ijkl").encode(),
        ),
        pa.binary(4),
        "none",
        "BYTE_STREAM_SPLIT",
    )


def test_list_fsb_byte_stream_split() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(
                Python.str("abcd").encode(), Python.str("efgh").encode()
            ),
            Python.list(Python.str("ijkl").encode()),
            Python.none(),
        ),
        pa.list_(pa.binary(4)),
        "none",
        "BYTE_STREAM_SPLIT",
    )


def test_decimal_byte_stream_split() raises:
    var pa = Python.import_module("pyarrow")
    var D = Python.import_module("decimal").Decimal
    _check(
        Python.list(D("1.25"), D("-2.50"), Python.none(), D("3.75")),
        pa.decimal128(9, 2),
        "none",
        "BYTE_STREAM_SPLIT",
    )


def test_list_of_byte_stream_split_float() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(Python.list(1.5, 2.5), Python.list(3.5), Python.none()),
        pa.list_(pa.float64()),
        "none",
        "BYTE_STREAM_SPLIT",
    )


def test_list_of_timestamp() raises:
    # temporal list element — decodes as int64 storage then retags to timestamp
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(1, 2),
            Python.list(3),
            Python.none(),
            Python.list(4, 5, 6),
        ),
        pa.list_(pa.timestamp("us")),
        "snappy",
    )


def test_list_of_date32() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(10, 3), Python.list(7), Python.none(), Python.list()
        ),
        pa.list_(pa.date32()),
        "none",
    )


def test_list_of_decimal128() raises:
    # FIXED_LEN_BYTE_ARRAY decimal list element (big-endian two's complement)
    var pa = Python.import_module("pyarrow")
    var D = Python.import_module("decimal").Decimal
    _check(
        Python.list(
            Python.list(D("1.50"), D("-2.50")),
            Python.none(),
            Python.list(D("3.50")),
        ),
        pa.list_(pa.decimal128(5, 2)),
        "snappy",
    )


def test_list_of_fixed_size_binary() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(Python.str("ab").encode(), Python.str("cd").encode()),
            Python.list(Python.str("ef").encode()),
            Python.none(),
            Python.list(),
        ),
        pa.list_(pa.binary(2)),
        "none",
    )


def test_list_across_many_pages() raises:
    # a list column whose leaf/levels span many data pages within one chunk;
    # the reader must stitch rep/def levels across page boundaries.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    # each row is list(range(i % 5)), or None every 11th row
    var data = Python.list()
    for i in range(3000):
        if i % 11 == 0:
            data.append(Python.none())
        else:
            data.append(np.arange(i % 5).tolist())
    var want = pa.table(
        Python.dict(v=pa.array(data, type=pa.list_(pa.int64())))
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_nested_pages.parquet")
        pq.write_table(
            want,
            path,
            data_page_size=256,
            use_dictionary=False,
            compression="none",
        )
        assert_equal(Int(py=pq.ParquetFile(path).metadata.num_row_groups), 1)

        var got = _to_pa(read_table(path))
        assert_true(
            Bool(
                got.column(0).to_pylist()
                == pq.read_table(path).column(0).to_pylist()
            ),
            "value mismatch",
        )


# ---------------------------------------------------------------------------
# Maps  (physically list<struct<key,value>> — read reconstructs a MapArray)
# ---------------------------------------------------------------------------


def test_map_string_int() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.dict(a=1, b=2),
            Python.dict(),
            Python.none(),
            Python.dict(c=3),
        ),
        pa.map_(pa.string(), pa.int64()),
        "none",
    )


def test_map_int_key_string_value() raises:
    var pa = Python.import_module("pyarrow")
    var m0 = Python.dict()
    m0[1] = "x"
    m0[2] = "y"
    var m2 = Python.dict()
    m2[3] = "z"
    _check(
        Python.list(m0, Python.none(), m2),
        pa.map_(pa.int32(), pa.string()),
        "none",
    )


def test_map_nullable_values() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.dict(a=1, b=Python.none()), Python.dict(c=3), Python.none()
        ),
        pa.map_(pa.string(), pa.int64()),
        "none",
    )


def test_map_snappy_many_rows() raises:
    var pa = Python.import_module("pyarrow")
    # row i is {str(j): 2*j for j in range(i % 4)}
    var data = Python.list()
    for i in range(50):
        var row = Python.dict()
        for j in range(i % 4):
            row[Python.str(String(j))] = j * 2
        data.append(row)
    _check(data, pa.map_(pa.string(), pa.int64()), "snappy")


def test_map_of_list_values() raises:
    # map<string, list<int64>> — a nested value type under the map.
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.dict(a=Python.list(1, 2), b=Python.list()),
            Python.none(),
            Python.dict(c=Python.list(3)),
        ),
        pa.map_(pa.string(), pa.list_(pa.int64())),
        "none",
    )


def test_list_of_map() raises:
    # list<map<string,int64>> — a map nested inside a list.
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(Python.dict(a=1), Python.dict(b=2)),
            Python.list(),
            Python.none(),
            Python.list(Python.dict(c=3)),
        ),
        pa.list_(pa.map_(pa.string(), pa.int64())),
        "none",
    )


def test_page_split_nested() raises:
    # a list column with >20 000 top-level rows must split into multiple pages,
    # each ending on a record boundary (whole lists never cross a page)
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var lists = Python.list()
    for i in range(30000):
        var row = Python.list()
        row.append(i)
        row.append(i + 1)
        lists.append(row)
    var want = pa.table(
        Python.dict(l=pa.array(lists, type=pa.list_(pa.int64())))
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_split_nested.parquet")
        var t = CArrowArrayStream.from_pycapsule(
            want.__arrow_c_stream__(Python.none())
        ).to_table()
        write_table(t, path, use_dictionary=False)

        # PyArrow reads every list back (proves per-page rep/def levels are valid)
        assert_true(Bool(pq.read_table(path).column(0).equals(want.column(0))))
        # marrow round-trips
        var back = _to_pa(read_table(path))
        assert_true(
            Bool(
                back.column(0)
                .combine_chunks()
                .equals(want.column(0).combine_chunks())
            )
        )
        # >1 page, and the pages tile all 30 000 top-level rows
        var pbs = ParquetFile(path).page_bounds()
        assert_true(len(pbs[0][0]) > 1)
        var total = 0
        for p in range(len(pbs[0][0])):
            total += pbs[0][0][p].copy().num_rows
        assert_equal(total, 30000)


def test_write_nullable_struct() raises:
    # a nullable struct with struct-level nulls (and a child null) -> emitted as
    # an OPTIONAL group; PyArrow must read the struct nulls back exactly
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var s = pa.array(
        Python.list(
            Python.dict(a=1, b="x"),
            Python.none(),
            Python.dict(a=3, b=Python.none()),
            Python.none(),
            Python.dict(a=5, b="z"),
        ),
        type=pa.struct(
            Python.list(pa.field("a", pa.int32()), pa.field("b", pa.string()))
        ),
    )
    var want = pa.table(Python.dict(s=s))
    var t = CArrowArrayStream.from_pycapsule(
        want.__arrow_c_stream__(Python.none())
    ).to_table()
    with ScratchDir() as dir:
        var path = join(dir, "marrow_nullable_struct.parquet")
        write_table(t, path, use_dictionary=False)

        var back = pq.read_table(path)
        # struct-level nulls preserved
        assert_true(
            Bool(
                back.column(0).combine_chunks().is_valid().to_pylist()
                == Python.list(True, False, True, False, True)
            )
        )
        # full value equality (incl. the child-null at row 2)
        assert_true(Bool(back.column(0).equals(want.column(0))))
        # marrow round-trips it too
        assert_equal(read_table(path).num_rows(), 5)


def test_list_float16() raises:
    # float16 inside a list exercises the leveled (rep/def) primitive path.
    var pa = Python.import_module("pyarrow")
    _check(
        Python.list(
            Python.list(1.5, 2.5),
            Python.none(),
            Python.list(),
            Python.list(3.0, -4.0, 0.5),
        ),
        pa.list_(pa.float16()),
        "none",
    )


# ---------------------------------------------------------------------------
# Field ids  (`SchemaElement.field_id` <-> `Field.metadata[FIELD_ID_KEY]`)
# ---------------------------------------------------------------------------


def _pa_id(n: Int) raises -> PythonObject:
    var md = Python.dict()
    md[FIELD_ID_KEY] = String(n)
    return md


def _pa_ids_table() raises -> PythonObject:
    """A column, a struct and its children, a list and its element, a map and
    its key and value, each carrying a distinct id."""
    var pa = Python.import_module("pyarrow")
    var schema = pa.schema(
        Python.list(
            pa.field("i", pa.int64(), metadata=_pa_id(1)),
            pa.field(
                "s",
                pa.struct(
                    Python.list(
                        pa.field("a", pa.int64(), metadata=_pa_id(3)),
                        pa.field("x", pa.float64(), metadata=_pa_id(4)),
                    )
                ),
                metadata=_pa_id(2),
            ),
            pa.field(
                "l",
                pa.list_(pa.field("element", pa.int64(), metadata=_pa_id(6))),
                metadata=_pa_id(5),
            ),
            pa.field(
                "m",
                pa.map_(
                    pa.field(
                        "key", pa.string(), nullable=False, metadata=_pa_id(8)
                    ),
                    pa.field("value", pa.int64(), metadata=_pa_id(9)),
                ),
                metadata=_pa_id(7),
            ),
        )
    )
    return pa.Table.from_pydict(
        Python.dict(
            i=Python.list(1, 2),
            s=Python.list(Python.dict(a=1, x=1.5), Python.dict(a=2, x=2.5)),
            l=Python.list(Python.list(1, 2), Python.list()),
            m=Python.list(Python.dict(a=1), Python.dict()),
        ),
        schema=schema,
    )


def _assert_ids(schema: Schema) raises:
    assert_equal(schema.field(name="i").field_id().value(), 1)
    ref s = schema.field(name="s")
    assert_equal(s.field_id().value(), 2)
    assert_equal(s.dtype.as_struct().fields[0].field_id().value(), 3)
    assert_equal(s.dtype.as_struct().fields[1].field_id().value(), 4)
    ref l = schema.field(name="l")
    assert_equal(l.field_id().value(), 5)
    assert_equal(l.dtype.as_list().value_field().field_id().value(), 6)
    ref m = schema.field(name="m")
    assert_equal(m.field_id().value(), 7)
    assert_equal(m.dtype.as_map().key_field().field_id().value(), 8)
    assert_equal(m.dtype.as_map().item_field().field_id().value(), 9)
    # pyarrow writes no id on the `key_value` group
    assert_false(Bool(m.dtype.as_map().entries_field().field_id()))


def _pa_field_id(f: PythonObject) raises -> Int:
    var builtins = Python.import_module("builtins")
    return Int(py=builtins.int(f.metadata[Python.str(FIELD_ID_KEY).encode()]))


def test_field_ids_every_level() raises:
    var pq = Python.import_module("pyarrow.parquet")
    with ScratchDir() as dir:
        var src = join(dir, "marrow_field_ids_src.parquet")
        pq.write_table(_pa_ids_table(), src)
        var t = read_table(src)
        _assert_ids(t.schema)

        # marrow write -> marrow read keeps every id
        var dst = join(dir, "marrow_field_ids_dst.parquet")
        write_table(t, dst)
        var back = read_table(dst)
        _assert_ids(back.schema)
        assert_true(back.schema.fields == t.schema.fields)

        # and pyarrow reads them off marrow's file
        var ps = pq.read_schema(dst)
        assert_equal(_pa_field_id(ps.field("i")), 1)
        assert_equal(_pa_field_id(ps.field("s")), 2)
        assert_equal(_pa_field_id(ps.field("s").type.field("a")), 3)
        assert_equal(_pa_field_id(ps.field("s").type.field("x")), 4)
        assert_equal(_pa_field_id(ps.field("l")), 5)
        assert_equal(_pa_field_id(ps.field("l").type.value_field), 6)
        assert_equal(_pa_field_id(ps.field("m")), 7)
        assert_equal(_pa_field_id(ps.field("m").type.key_field), 8)
        assert_equal(_pa_field_id(ps.field("m").type.item_field), 9)


def test_field_ids_absent_add_no_metadata() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var plain = _pa_ids_table().cast(
        pa.schema(
            Python.list(
                pa.field("i", pa.int64()),
                pa.field(
                    "s",
                    pa.struct(
                        Python.list(
                            pa.field("a", pa.int64()),
                            pa.field("x", pa.float64()),
                        )
                    ),
                ),
                pa.field("l", pa.list_(pa.int64())),
                pa.field("m", pa.map_(pa.string(), pa.int64())),
            )
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_field_ids_absent.parquet")
        pq.write_table(plain, path)
        var schema = read_table(path).schema.copy()
        for ref f in schema.fields:
            assert_equal(len(f.metadata), 0)
        ref s = schema.field(name="s")
        for ref f in s.dtype.as_struct().fields:
            assert_equal(len(f.metadata), 0)
        ref l = schema.field(name="l")
        assert_equal(len(l.dtype.as_list().value_field().metadata), 0)
        ref m = schema.field(name="m")
        assert_equal(len(m.dtype.as_map().entries_field().metadata), 0)
        assert_equal(len(m.dtype.as_map().key_field().metadata), 0)
        assert_equal(len(m.dtype.as_map().item_field().metadata), 0)


def test_field_id_on_map_entries() raises:
    # The map's `key_value` group carries the entries struct's id, which
    # pyarrow cannot set: map the schema to Parquet and back through the
    # footer's Thrift encoding.
    var entries = Field(
        "entries",
        dt.struct_(
            Field(
                "key", dt.string, nullable=False, metadata={FIELD_ID_KEY: "2"}
            ),
            Field("value", dt.int64, metadata={FIELD_ID_KEY: "3"}),
        ),
        nullable=False,
        metadata={FIELD_ID_KEY: "4"},
    )
    var schema = Schema(
        fields=[Field("m", dt.MapType(entries^), metadata={FIELD_ID_KEY: "1"})]
    )
    var meta = FileMetaData()
    meta.schema = SchemaMapping.from_arrow(schema).elements.copy()
    var w = ThriftCompactWriter()
    meta.write(w)
    var r = ThriftCompactReader(Span(w.buf))
    var back = SchemaMapping.from_parquet(FileMetaData.read(r)).schema.copy()
    assert_true(back == schema)
