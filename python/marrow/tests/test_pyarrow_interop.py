"""Test PyArrow ↔ marrow interoperability via Arrow C Data/Stream Interface.

Covers: DataType, Schema, Array, RecordBatch, and Table roundtrips,
plus compute kernel usage with PyArrow-created arrays.
"""

import pyarrow as pa
import pytest
import marrow as ma


INT_TYPES = (pa.int8, pa.int16, pa.int32, pa.int64)
FLOAT_TYPES = (pa.float32, pa.float64)


# ===========================================================================
# DataType roundtrips
# ===========================================================================


def test_datatype_from_pyarrow():
    """PyArrow type → marrow DataType via __arrow_c_schema__."""
    for pa_type, ma_factory in [
        (pa.int8(), ma.int8),
        (pa.int16(), ma.int16),
        (pa.int32(), ma.int32),
        (pa.int64(), ma.int64),
        (pa.uint8(), ma.uint8),
        (pa.uint16(), ma.uint16),
        (pa.uint32(), ma.uint32),
        (pa.uint64(), ma.uint64),
        (pa.float32(), ma.float32),
        (pa.float64(), ma.float64),
        (pa.bool_(), ma.bool_),
        (pa.string(), ma.string),
    ]:
        ma_dt = ma_factory()
        pa_field = pa.field("x", pa_type)
        pa_schema = pa.schema([pa_field])
        ma_schema = ma.schema(pa_schema)
        roundtripped = pa.schema(ma_schema)
        assert roundtripped.field("x").type == pa_type


def test_datatype_dictionary_roundtrip():
    """dictionary(index, value) type round-trips through PyArrow schema."""
    for index_pa_type in [
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
    ]:
        pa_type = pa.dictionary(index_pa_type, pa.string())
        ma_schema = ma.schema(pa.schema([pa.field("d", pa_type)]))
        roundtripped = pa.schema(ma_schema)
        assert roundtripped.field("d").type == pa_type


def test_datatype_nested_roundtrip():
    """Nested types (list, struct) roundtrip through PyArrow."""
    pa_schema = pa.schema(
        [
            pa.field("lst", pa.list_(pa.int32())),
            pa.field(
                "st",
                pa.struct(
                    [
                        pa.field("a", pa.int64()),
                        pa.field("b", pa.float64()),
                    ]
                ),
            ),
        ]
    )
    ma_schema = ma.schema(pa_schema)
    roundtripped = pa.schema(ma_schema)
    assert roundtripped.field("lst").type == pa.list_(pa.int32())
    assert roundtripped.field("st").type.num_fields == 2


# ===========================================================================
# Schema roundtrips
# ===========================================================================


def test_schema_to_pyarrow():
    schema = ma.schema(
        [
            ma.field("x", ma.int32(), True, {}),
            ma.field("y", ma.float64(), True, {}),
            ma.field("s", ma.string(), True, {}),
        ]
    )
    pa_schema = pa.schema(schema)
    assert len(pa_schema) == 3
    assert pa_schema.field("x").type == pa.int32()
    assert pa_schema.field("y").type == pa.float64()
    assert pa_schema.field("s").type == pa.string()


def test_schema_to_pyarrow_nested():
    schema = ma.schema(
        [
            ma.field("lst", ma.list_(ma.int32()), True, {}),
            ma.field(
                "st",
                ma.struct(
                    [
                        ma.field("a", ma.int32(), True, {}),
                        ma.field("b", ma.float64(), True, {}),
                    ]
                ),
                True,
                {},
            ),
        ]
    )
    pa_schema = pa.schema(schema)
    assert len(pa_schema) == 2
    assert str(pa_schema.field("lst").type) == "list<item: int32>"
    assert pa_schema.field("st").type == pa.struct(
        [
            pa.field("a", pa.int32()),
            pa.field("b", pa.float64()),
        ]
    )


def test_schema_from_pyarrow():
    pa_schema = pa.schema([pa.field("x", pa.int32()), pa.field("y", pa.float64())])
    schema = ma.schema(pa_schema)
    assert pa.schema(schema).equals(pa_schema)


def test_schema_from_marrow_schema():
    """Passing a marrow Schema to ma.schema() should return an equal copy."""
    original = ma.schema(
        [
            ma.field("a", ma.int64(), True, {}),
            ma.field("b", ma.string(), True, {}),
        ]
    )
    copy = ma.schema(original)
    assert pa.schema(original).equals(pa.schema(copy))


# ===========================================================================
# Array roundtrips
# ===========================================================================


def test_array_roundtrip_int32():
    arr = ma.array([7, 42, -1], type=ma.int32())
    pyarr = pa.array(arr)
    assert pyarr.type == pa.int32()
    assert pyarr[0].as_py() == 7
    assert pyarr[1].as_py() == 42
    assert pyarr[2].as_py() == -1
    reimported = ma.array(pyarr)
    assert len(reimported) == 3
    assert reimported[0] == 7
    assert reimported[1] == 42
    assert reimported[2] == -1


def test_array_roundtrip_string():
    arr = ma.array(["hello", "world", "mojo"])
    pyarr = pa.array(arr)
    assert pyarr.type == pa.string()
    assert pyarr[0].as_py() == "hello"
    reimported = ma.array(pyarr)
    assert len(reimported) == 3
    assert reimported[0] == "hello"
    assert reimported[2] == "mojo"


def test_array_roundtrip_with_nulls():
    arr = ma.array([1, None, 3, None], type=ma.int64())
    pyarr = pa.array(arr)
    assert pyarr.type == pa.int64()
    assert pyarr[0].as_py() == 1
    assert pyarr[1].as_py() is None
    reimported = ma.array(pyarr)
    assert len(reimported) == 4
    assert reimported.null_count() == 2


def test_array_from_pyarrow_int():
    pa_arr = pa.array([1, 2, 3], type=pa.int64())
    ma_arr = ma.array(pa_arr)
    assert len(ma_arr) == 3
    roundtripped = pa.array(ma_arr)
    assert roundtripped.equals(pa_arr)


def test_array_from_pyarrow_float():
    pa_arr = pa.array([1.5, 2.5, 3.5], type=pa.float64())
    ma_arr = ma.array(pa_arr)
    assert len(ma_arr) == 3
    roundtripped = pa.array(ma_arr)
    assert roundtripped.equals(pa_arr)


def test_array_from_pyarrow_string():
    pa_arr = pa.array(["hello", "world"])
    ma_arr = ma.array(pa_arr)
    assert len(ma_arr) == 2
    roundtripped = pa.array(ma_arr)
    assert roundtripped.equals(pa_arr)


def test_array_from_pyarrow_bool():
    pa_arr = pa.array([True, False, True])
    ma_arr = ma.array(pa_arr)
    assert len(ma_arr) == 3
    roundtripped = pa.array(ma_arr)
    assert roundtripped.equals(pa_arr)


def test_array_from_pyarrow_with_nulls():
    pa_arr = pa.array([1, None, 3], type=pa.int64())
    ma_arr = ma.array(pa_arr)
    assert len(ma_arr) == 3
    roundtripped = pa.array(ma_arr)
    assert roundtripped.equals(pa_arr)


def test_array_from_pyarrow_list():
    pa_arr = pa.array([[1, 2], [3]], type=pa.list_(pa.int32()))
    ma_arr = ma.array(pa_arr)
    assert len(ma_arr) == 2
    roundtripped = pa.array(ma_arr)
    assert roundtripped.equals(pa_arr)


def test_array_from_pyarrow_struct():
    pa_arr = pa.array(
        [{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}],
        type=pa.struct([pa.field("a", pa.int64()), pa.field("b", pa.float64())]),
    )
    ma_arr = ma.array(pa_arr)
    assert len(ma_arr) == 2
    roundtripped = pa.array(ma_arr)
    assert roundtripped.equals(pa_arr)


def test_array_from_pyarrow_dictionary():
    """Import PyArrow dictionary array → Marrow → back to PyArrow."""
    pa_arr = pa.array(["cat", "dog", "cat", "fish", "dog"]).dictionary_encode()
    ma_arr = ma.array(pa_arr)
    assert len(ma_arr) == 5
    assert pa.array(ma_arr).equals(pa_arr)


def test_array_roundtrip_dictionary_with_nulls():
    pa_arr = pa.array(["cat", None, "cat", "fish", None]).dictionary_encode()
    ma_arr = ma.array(pa_arr)
    assert len(ma_arr) == 5
    assert ma_arr.null_count() == 2
    assert pa.array(ma_arr).equals(pa_arr)


def test_array_roundtrip_dictionary_ordered():
    pa_arr = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 0, 2], type=pa.int8()),
        pa.array(["a", "b", "c"]),
        ordered=True,
    )
    ma_arr = ma.array(pa_arr)
    roundtripped = pa.array(ma_arr)
    assert roundtripped.equals(pa_arr)
    assert roundtripped.type.ordered


@pytest.mark.parametrize(
    "index_pa_type",
    [
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
    ],
)
def test_array_roundtrip_dictionary_index_types(index_pa_type):
    pa_arr = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 2, 0], type=index_pa_type),
        pa.array(["x", "y", "z"]),
    )
    assert pa.array(ma.array(pa_arr)).equals(pa_arr)


# ===========================================================================
# RecordBatch roundtrips
# ===========================================================================


def test_record_batch_from_pyarrow():
    pa_rb = pa.record_batch(
        {"x": [1, 2, 3], "y": [4.0, 5.0, 6.0], "z": ["a", "b", "c"]},
        schema=pa.schema(
            [
                pa.field("x", pa.int64()),
                pa.field("y", pa.float64()),
                pa.field("z", pa.string()),
            ]
        ),
    )
    ma_rb = ma.record_batch(pa_rb)
    assert ma_rb.num_rows() == 3
    assert ma_rb.num_columns() == 3
    assert ma_rb.column_names() == ["x", "y", "z"]


def test_record_batch_to_pyarrow():
    ma_rb = ma.record_batch(
        {"x": ma.array([1, 2, 3], type=ma.int64()), "y": ma.array(["a", "b", "c"])}
    )
    pa_rb = pa.record_batch(ma_rb)
    assert pa_rb.num_rows == 3
    assert pa_rb.num_columns == 2
    assert pa_rb.schema.field("x").type == pa.int64()


def test_record_batch_roundtrip():
    ma_rb = ma.record_batch(
        {"a": ma.array([10, 20], type=ma.int32()), "b": ma.array(["x", "y"])}
    )
    pa_rb = pa.record_batch(ma_rb)
    ma_rb2 = ma.record_batch(pa_rb)
    assert ma_rb2.num_rows() == 2
    assert ma_rb2.num_columns() == 2
    assert ma_rb2.column_names() == ["a", "b"]


def test_record_batch_roundtrip_with_nulls():
    batch = ma.record_batch(
        {
            "a": ma.array([1, None, 3], type=ma.int32()),
            "b": ma.array(["x", None, "z"]),
        }
    )
    pa_batch = pa.record_batch(batch)
    assert pa_batch.column("a").to_pylist() == [1, None, 3]
    assert pa_batch.column("b").to_pylist() == ["x", None, "z"]
    ma_batch2 = ma.record_batch(pa_batch)
    assert ma_batch2.num_rows() == 3


def test_record_batch_with_list_column():
    pa_rb = pa.record_batch(
        {"vals": pa.array([[1, 2], [3]], type=pa.list_(pa.int32()))},
    )
    ma_rb = ma.record_batch(pa_rb)
    assert ma_rb.num_rows() == 2
    pa_rb2 = pa.record_batch(ma_rb)
    assert pa_rb2.column("vals").equals(pa_rb.column("vals"))


def test_record_batch_with_dictionary_column():
    pa_rb = pa.record_batch({"cat": pa.array(["x", "y", "x", "z"]).dictionary_encode()})
    ma_rb = ma.record_batch(pa_rb)
    assert ma_rb.num_rows() == 4
    assert pa.record_batch(ma_rb).column("cat").equals(pa_rb.column("cat"))


def test_record_batch_arrow_c_schema():
    batch = ma.record_batch(
        {
            "x": ma.array([1, 2, 3], type=ma.int32()),
            "y": ma.array([4.0, 5.0, 6.0], type=ma.float64()),
            "z": ma.array(["a", "b", "c"]),
        }
    )
    pa_schema = pa.schema(batch)
    assert pa_schema.field("x").type == pa.int32()
    assert pa_schema.field("y").type == pa.float64()
    assert pa_schema.field("z").type == pa.string()


# ===========================================================================
# Table roundtrips
# ===========================================================================


def test_table_from_pyarrow():
    pa_table = pa.table(
        {"x": [1, 2, 3], "y": ["a", "b", "c"]},
        schema=pa.schema(
            [
                pa.field("x", pa.int64()),
                pa.field("y", pa.string()),
            ]
        ),
    )
    ma_table = ma.table(pa_table)
    assert ma_table.num_rows() == 3
    assert ma_table.num_columns() == 2
    assert ma_table.column_names() == ["x", "y"]


def test_table_to_pyarrow():
    ma_table = ma.table(
        {
            "x": ma.array([1, 2, 3], type=ma.int32()),
            "y": ma.array([4.0, 5.0, 6.0], type=ma.float64()),
            "z": ma.array(["a", "b", "c"]),
        }
    )
    pa_table = pa.table(ma_table)
    assert pa_table.num_rows == 3
    assert pa_table.num_columns == 3
    assert pa_table.schema.field("x").type == pa.int32()
    assert pa_table.column("x").to_pylist() == [1, 2, 3]
    assert pa_table.column("z").to_pylist() == ["a", "b", "c"]


def test_table_roundtrip():
    ma_table = ma.table(
        {"a": ma.array([10, 20], type=ma.int32()), "b": ma.array(["x", "y"])}
    )
    pa_table = pa.table(ma_table)
    ma_table2 = ma.table(pa_table)
    assert ma_table2.num_rows() == 2
    assert ma_table2.num_columns() == 2
    assert ma_table2.column_names() == ["a", "b"]


def test_table_roundtrip_with_nulls():
    t = ma.table(
        {
            "a": ma.array([1, None, 3], type=ma.int32()),
            "b": ma.array(["x", None, "z"]),
        }
    )
    pa_table = pa.table(t)
    assert pa_table.column("a").to_pylist() == [1, None, 3]
    assert pa_table.column("b").to_pylist() == ["x", None, "z"]
    ma_table2 = ma.table(pa_table)
    assert ma_table2.num_rows() == 3


def test_table_with_struct_column():
    pa_table = pa.table(
        {
            "s": pa.array(
                [{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}],
                type=pa.struct(
                    [
                        pa.field("a", pa.int64()),
                        pa.field("b", pa.float64()),
                    ]
                ),
            ),
        }
    )
    ma_table = ma.table(pa_table)
    assert ma_table.num_rows() == 2
    pa_table2 = pa.table(ma_table)
    assert pa_table2.column("s").equals(pa_table.column("s"))


def test_table_arrow_c_stream_import():
    pa_table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    t = ma.table(pa_table)
    assert t.num_rows() == 3
    assert t.num_columns() == 2
    assert list(t.column_names()) == ["a", "b"]


# ===========================================================================
# Compute kernels with PyArrow arrays
# ===========================================================================


# ── add ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_add_pyarrow_arrays(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, 2, 3], type=pa_type())
    pa_b = pa.array([10, 20, 30], type=pa_type())
    a = ma.array(pa_a)
    b = ma.array(pa_b)
    result = ma.compute.add(a, b, ctx=None)
    assert len(result) == 3
    assert result.null_count() == 0
    out = pa.array(result)
    assert out[0].as_py() == 11
    assert out[1].as_py() == 22
    assert out[2].as_py() == 33


@pytest.mark.parametrize("pa_type", FLOAT_TYPES)
def test_mojo_add_pyarrow_float(pa_type: pa.DType) -> None:
    pa_a = pa.array([1.0, 2.0, 3.0], type=pa_type())
    pa_b = pa.array([0.5, 1.5, 2.5], type=pa_type())
    a = ma.array(pa_a)
    b = ma.array(pa_b)
    result = ma.compute.add(a, b, ctx=None)
    assert len(result) == 3
    out = pa.array(result)
    assert out[0].as_py() == pytest.approx(1.5)
    assert out[1].as_py() == pytest.approx(3.5)
    assert out[2].as_py() == pytest.approx(5.5)


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_add_pyarrow_nulls_propagate(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, None, 3], type=pa_type())
    pa_b = pa.array([10, 20, 30], type=pa_type())
    result = ma.compute.add(ma.array(pa_a), ma.array(pa_b), ctx=None)
    assert len(result) == 3
    assert result.null_count() == 1
    out = pa.array(result)
    assert out[0].as_py() == 11
    assert out[1].as_py() is None
    assert out[2].as_py() == 33


# ── subtract ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_sub_pyarrow_arrays(pa_type: pa.DType) -> None:
    pa_a = pa.array([10, 20, 30], type=pa_type())
    pa_b = pa.array([1, 2, 3], type=pa_type())
    result = ma.compute.subtract(ma.array(pa_a), ma.array(pa_b), ctx=None)
    assert len(result) == 3
    out = pa.array(result)
    assert out[0].as_py() == 9
    assert out[1].as_py() == 18
    assert out[2].as_py() == 27


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_sub_pyarrow_nulls_propagate(pa_type: pa.DType) -> None:
    pa_a = pa.array([10, None, 30], type=pa_type())
    pa_b = pa.array([1, 2, None], type=pa_type())
    result = ma.compute.subtract(ma.array(pa_a), ma.array(pa_b), ctx=None)
    assert len(result) == 3
    assert result.null_count() == 2


# ── multiply ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_mul_pyarrow_arrays(pa_type: pa.DType) -> None:
    pa_a = pa.array([2, 3, 4], type=pa_type())
    pa_b = pa.array([5, 6, 7], type=pa_type())
    result = ma.compute.multiply(ma.array(pa_a), ma.array(pa_b), ctx=None)
    assert len(result) == 3
    out = pa.array(result)
    assert out[0].as_py() == 10
    assert out[1].as_py() == 18
    assert out[2].as_py() == 28


# ── divide ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_div_pyarrow_arrays(pa_type: pa.DType) -> None:
    pa_a = pa.array([10, 20, 30], type=pa_type())
    pa_b = pa.array([2, 4, 5], type=pa_type())
    result = ma.compute.divide(ma.array(pa_a), ma.array(pa_b), ctx=None)
    assert len(result) == 3
    out = pa.array(result)
    assert out[0].as_py() == 5
    assert out[1].as_py() == 5
    assert out[2].as_py() == 6


# ── aggregates ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_sum_pyarrow_int(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, 2, 3, 4], type=pa_type())
    assert ma.compute.sum(ma.array(pa_a), ctx=None) == 10.0


@pytest.mark.parametrize("pa_type", FLOAT_TYPES)
def test_mojo_sum_pyarrow_float(pa_type: pa.DType) -> None:
    pa_a = pa.array([1.5, 2.5, 3.0], type=pa_type())
    assert ma.compute.sum(ma.array(pa_a), ctx=None) == pytest.approx(7.0)


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_sum_pyarrow_skips_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, None, 3, None], type=pa_type())
    assert ma.compute.sum(ma.array(pa_a), ctx=None) == 4.0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_min_pyarrow(pa_type: pa.DType) -> None:
    pa_a = pa.array([3, 1, 4, 1, 5], type=pa_type())
    assert ma.compute.min(ma.array(pa_a), ctx=None) == 1.0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_max_pyarrow(pa_type: pa.DType) -> None:
    pa_a = pa.array([3, 1, 4, 1, 5], type=pa_type())
    assert ma.compute.max(ma.array(pa_a), ctx=None) == 5.0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_min_pyarrow_skips_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([3, None, 1, None], type=pa_type())
    assert ma.compute.min(ma.array(pa_a), ctx=None) == 1.0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_max_pyarrow_skips_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([3, None, 5, None], type=pa_type())
    assert ma.compute.max(ma.array(pa_a), ctx=None) == 5.0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_product_pyarrow(pa_type: pa.DType) -> None:
    pa_a = pa.array([2, 3, 4], type=pa_type())
    assert ma.compute.product(ma.array(pa_a), ctx=None) == 24.0


# ── filter ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_filter_pyarrow_arrays(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, 2, 3, 4, 5], type=pa_type())
    pa_mask = pa.array([True, False, True, False, True])
    result = ma.compute.filter(ma.array(pa_a), ma.array(pa_mask))
    assert len(result) == 3
    out = pa.array(result)
    assert out[0].as_py() == 1
    assert out[1].as_py() == 3
    assert out[2].as_py() == 5


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_filter_pyarrow_preserves_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, None, 3, None, 5], type=pa_type())
    pa_mask = pa.array([True, True, True, False, True])
    result = ma.compute.filter(ma.array(pa_a), ma.array(pa_mask))
    assert len(result) == 4
    assert result.null_count() == 1


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_filter_pyarrow_all_false(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, 2, 3], type=pa_type())
    pa_mask = pa.array([False, False, False])
    result = ma.compute.filter(ma.array(pa_a), ma.array(pa_mask))
    assert len(result) == 0


# ── pyarrow → mojo compute → pyarrow roundtrip ───────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_pyarrow_to_mojo_compute_to_pyarrow(pa_type: pa.DType) -> None:
    pa_a = pa.array([7, 42, -1], type=pa_type())
    pa_b = pa.array([3, 8, 1], type=pa_type())
    result = ma.compute.add(ma.array(pa_a), ma.array(pa_b), ctx=None)
    out = pa.array(result)
    assert out.type == pa_type()
    assert out[0].as_py() == 10
    assert out[1].as_py() == 50
    assert out[2].as_py() == 0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_pyarrow_to_mojo_compute_to_pyarrow_with_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, None, 3, None], type=pa_type())
    pa_b = pa.array([10, 20, None, 40], type=pa_type())
    result = ma.compute.add(ma.array(pa_a), ma.array(pa_b), ctx=None)
    out = pa.array(result)
    assert out.type == pa_type()
    assert out[0].as_py() == 11
    assert out[1].as_py() is None
    assert out[2].as_py() is None
    assert out[3].as_py() is None


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_pyarrow_to_mojo_drop_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, None, 3, None, 5], type=pa_type())
    result = ma.compute.drop_null(ma.array(pa_a))
    assert len(result) == 3
    assert result.null_count() == 0
    out = pa.array(result)
    assert out[0].as_py() == 1
    assert out[1].as_py() == 3
    assert out[2].as_py() == 5
