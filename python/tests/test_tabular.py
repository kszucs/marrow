"""Tests for RecordBatch Python bindings.

Mirrors PyArrow's RecordBatch test patterns where applicable.
"""

import pyarrow as pa
import pytest
import marrow as ma


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_batch():
    """Return a simple 3-column, 3-row RecordBatch."""
    return ma.record_batch(
        {
            "x": ma.array([1, 2, 3], type=ma.int32()),
            "y": ma.array([4.0, 5.0, 6.0], type=ma.float64()),
            "z": ma.array(["a", "b", "c"]),
        }
    )


# ── Construction ─────────────────────────────────────────────────────────────


def test_record_batch_from_dict():
    batch = make_batch()
    assert type(batch).__name__ == "RecordBatch"
    assert batch.num_rows() == 3
    assert batch.num_columns() == 3


def test_record_batch_from_list_with_names():
    x = ma.array([1, 2], type=ma.int64())
    y = ma.array(["a", "b"])
    batch = ma.record_batch([x, y], names=["x", "y"])
    assert batch.num_rows() == 2
    assert batch.num_columns() == 2


def test_record_batch_from_pyarrow():
    pa_batch = pa.record_batch({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    batch = ma.record_batch(pa_batch)
    assert type(batch).__name__ == "RecordBatch"
    assert batch.num_rows() == 3
    assert batch.num_columns() == 2


def test_record_batch_empty_columns():
    batch = ma.record_batch({})
    assert batch.num_rows() == 0
    assert batch.num_columns() == 0


# ── Properties ───────────────────────────────────────────────────────────────


def test_num_rows():
    assert make_batch().num_rows() == 3


def test_num_columns():
    assert make_batch().num_columns() == 3


def test_shape():
    shape = make_batch().shape()
    assert shape == (3, 3)


def test_column_names():
    names = make_batch().column_names()
    assert list(names) == ["x", "y", "z"]


def test_schema():
    batch = make_batch()
    schema = batch.schema()
    assert type(schema).__name__ == "Schema"


def test_columns():
    cols = make_batch().columns()
    assert len(cols) == 3


def test_str():
    s = str(make_batch())
    assert "RecordBatch" in s
    assert "num_rows=3" in s


# ── Column access ─────────────────────────────────────────────────────────────


def test_column_by_index():
    batch = make_batch()
    col = batch.column(0)
    assert type(col).__name__ == "Int32Array"
    assert len(col) == 3


def test_column_by_name():
    batch = make_batch()
    col = batch.column("y")
    assert type(col).__name__ == "Float64Array"


def test_column_by_name_not_found():
    with pytest.raises(Exception):
        make_batch().column("missing")


# ── Slice ─────────────────────────────────────────────────────────────────────


def test_slice():
    batch = make_batch()
    sliced = batch.slice(1, 2)
    assert sliced.num_rows() == 2
    assert sliced.num_columns() == 3


# ── Equality ─────────────────────────────────────────────────────────────────


def test_equals_same():
    batch = make_batch()
    batch2 = ma.record_batch(
        {
            "x": ma.array([1, 2, 3], type=ma.int32()),
            "y": ma.array([4.0, 5.0, 6.0], type=ma.float64()),
            "z": ma.array(["a", "b", "c"]),
        }
    )
    assert batch.equals(batch2)


def test_equals_different():
    batch = make_batch()
    other = ma.record_batch(
        {
            "x": ma.array([9, 9, 9], type=ma.int32()),
            "y": ma.array([4.0, 5.0, 6.0], type=ma.float64()),
            "z": ma.array(["a", "b", "c"]),
        }
    )
    assert not batch.equals(other)


def test_eq_operator():
    batch = make_batch()
    batch2 = ma.record_batch(
        {
            "x": ma.array([1, 2, 3], type=ma.int32()),
            "y": ma.array([4.0, 5.0, 6.0], type=ma.float64()),
            "z": ma.array(["a", "b", "c"]),
        }
    )
    assert batch.__eq__(batch2)


# ── Select ────────────────────────────────────────────────────────────────────


def test_select_by_index():
    batch = make_batch()
    sub = batch.select([0, 2])
    assert sub.num_columns() == 2
    assert list(sub.column_names()) == ["x", "z"]


def test_select_by_name():
    batch = make_batch()
    sub = batch.select(["z", "x"])
    assert sub.num_columns() == 2
    assert list(sub.column_names()) == ["z", "x"]


def test_select_empty():
    sub = make_batch().select([])
    assert sub.num_columns() == 0


# ── Rename columns ────────────────────────────────────────────────────────────


def test_rename_columns():
    batch = make_batch()
    renamed = batch.rename_columns(["a", "b", "c"])
    assert list(renamed.column_names()) == ["a", "b", "c"]
    assert renamed.num_rows() == 3


def test_rename_columns_wrong_count():
    with pytest.raises(Exception):
        make_batch().rename_columns(["only_one"])


# ── Column mutations (functional — return new batch) ─────────────────────────


def test_add_column():
    batch = make_batch()
    new_col = ma.array([10, 20, 30], type=ma.int64())
    new_field = ma.field("w", ma.int64())
    result = batch.add_column(0, new_field, new_col)
    assert result.num_columns() == 4
    assert list(result.column_names())[0] == "w"


def test_append_column():
    batch = make_batch()
    new_col = ma.array([10, 20, 30], type=ma.int64())
    new_field = ma.field("w", ma.int64())
    result = batch.append_column(new_field, new_col)
    assert result.num_columns() == 4
    assert list(result.column_names())[-1] == "w"


def test_remove_column():
    batch = make_batch()
    result = batch.remove_column(1)
    assert result.num_columns() == 2
    assert list(result.column_names()) == ["x", "z"]


def test_set_column():
    batch = make_batch()
    new_col = ma.array([10, 20, 30], type=ma.int32())
    new_field = ma.field("xx", ma.int32())
    result = batch.set_column(0, new_field, new_col)
    assert result.num_columns() == 3
    assert list(result.column_names())[0] == "xx"


# ── to_pydict / to_pylist ─────────────────────────────────────────────────────


def test_to_pydict():
    batch = ma.record_batch(
        {
            "a": ma.array([1, 2], type=ma.int32()),
            "b": ma.array(["x", "y"]),
        }
    )
    d = batch.to_pydict()
    assert list(d["a"]) == [1, 2]
    assert list(d["b"]) == ["x", "y"]


def test_to_pylist():
    batch = ma.record_batch(
        {
            "a": ma.array([1, 2], type=ma.int32()),
            "b": ma.array(["x", "y"]),
        }
    )
    rows = batch.to_pylist()
    assert len(rows) == 2
    assert rows[0]["a"] == 1
    assert rows[0]["b"] == "x"
    assert rows[1]["a"] == 2
    assert rows[1]["b"] == "y"


# ── Arrow C Data Interface roundtrip ─────────────────────────────────────────


def test_arrow_c_record_batch_roundtrip():
    """Export marrow RecordBatch → import into PyArrow via __arrow_c_record_batch__."""
    batch = make_batch()
    pa_batch = pa.record_batch(batch)
    assert pa_batch.num_rows == 3
    assert pa_batch.num_columns == 3
    assert pa_batch.schema.field("x").type == pa.int32()
    assert pa_batch.schema.field("y").type == pa.float64()
    assert pa_batch.schema.field("z").type == pa.string()
    assert pa_batch.column("x").to_pylist() == [1, 2, 3]
    assert pa_batch.column("z").to_pylist() == ["a", "b", "c"]


def test_arrow_c_schema_roundtrip():
    """Export marrow schema → import into PyArrow via __arrow_c_schema__."""
    batch = make_batch()
    pa_schema = pa.schema(batch)
    assert pa_schema.field("x").type == pa.int32()
    assert pa_schema.field("y").type == pa.float64()
    assert pa_schema.field("z").type == pa.string()


def test_pyarrow_roundtrip_with_nulls():
    batch = ma.record_batch(
        {
            "a": ma.array([1, None, 3], type=ma.int32()),
            "b": ma.array(["x", None, "z"]),
        }
    )
    pa_batch = pa.record_batch(batch)
    assert pa_batch.column("a").to_pylist() == [1, None, 3]
    assert pa_batch.column("b").to_pylist() == ["x", None, "z"]
