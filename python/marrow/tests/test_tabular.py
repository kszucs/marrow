"""Tests for RecordBatch and Table Python bindings.

Mirrors PyArrow's test patterns where applicable.
"""

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
    assert batch.num_rows == 3
    assert batch.num_columns == 3


def test_record_batch_from_list_with_names():
    x = ma.array([1, 2], type=ma.int64())
    y = ma.array(["a", "b"])
    batch = ma.record_batch([x, y], names=["x", "y"])
    assert batch.num_rows == 2
    assert batch.num_columns == 2


def test_record_batch_empty_columns():
    batch = ma.record_batch({})
    assert batch.num_rows == 0
    assert batch.num_columns == 0


# ── Properties ───────────────────────────────────────────────────────────────


def test_num_rows():
    assert make_batch().num_rows == 3


def test_num_columns():
    assert make_batch().num_columns == 3


def test_shape():
    shape = make_batch().shape
    assert shape == (3, 3)


def test_column_names():
    names = make_batch().column_names
    assert list(names) == ["x", "y", "z"]


def test_schema():
    batch = make_batch()
    schema = batch.schema
    assert type(schema).__name__ == "Schema"


def test_columns():
    cols = make_batch().columns
    assert len(cols) == 3


def test_str():
    s = str(make_batch())
    assert "RecordBatch" in s
    assert "num_rows=3" in s


# ── Column access ─────────────────────────────────────────────────────────────


def test_column_by_index():
    batch = make_batch()
    col = batch.column(0)
    assert type(col).__name__ == "Array"
    assert len(col) == 3


def test_column_by_name():
    batch = make_batch()
    col = batch.column("y")
    assert type(col).__name__ == "Array"


def test_column_by_name_not_found():
    with pytest.raises(Exception):
        make_batch().column("missing")


# ── Slice ─────────────────────────────────────────────────────────────────────


def test_slice():
    batch = make_batch()
    sliced = batch.slice(1, 2)
    assert sliced.num_rows == 2
    assert sliced.num_columns == 3


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
    assert batch == batch2


# ── Select ────────────────────────────────────────────────────────────────────


def test_select_by_index():
    batch = make_batch()
    sub = batch.select([0, 2])
    assert sub.num_columns == 2
    assert list(sub.column_names) == ["x", "z"]


def test_select_by_name():
    batch = make_batch()
    sub = batch.select(["z", "x"])
    assert sub.num_columns == 2
    assert list(sub.column_names) == ["z", "x"]


def test_select_empty():
    sub = make_batch().select([])
    assert sub.num_columns == 0


# ── Rename columns ────────────────────────────────────────────────────────────


def test_rename_columns():
    batch = make_batch()
    renamed = batch.rename_columns(["a", "b", "c"])
    assert list(renamed.column_names) == ["a", "b", "c"]
    assert renamed.num_rows == 3


def test_rename_columns_wrong_count():
    with pytest.raises(Exception):
        make_batch().rename_columns(["only_one"])


# ── Column mutations (functional — return new batch) ─────────────────────────


def test_add_column():
    batch = make_batch()
    new_col = ma.array([10, 20, 30], type=ma.int64())
    new_field = ma.field("w", ma.int64(), True, {})
    result = batch.add_column(0, new_field, new_col)
    assert result.num_columns == 4
    assert list(result.column_names)[0] == "w"


def test_append_column():
    batch = make_batch()
    new_col = ma.array([10, 20, 30], type=ma.int64())
    new_field = ma.field("w", ma.int64(), True, {})
    result = batch.append_column(new_field, new_col)
    assert result.num_columns == 4
    assert list(result.column_names)[-1] == "w"


def test_remove_column():
    batch = make_batch()
    result = batch.remove_column(1)
    assert result.num_columns == 2
    assert list(result.column_names) == ["x", "z"]


def test_set_column():
    batch = make_batch()
    new_col = ma.array([10, 20, 30], type=ma.int32())
    new_field = ma.field("xx", ma.int32(), True, {})
    result = batch.set_column(0, new_field, new_col)
    assert result.num_columns == 3
    assert list(result.column_names)[0] == "xx"


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


# ===========================================================================
# Table tests
# ===========================================================================


def make_table():
    """Return a simple 3-column, 3-row Table."""
    return ma.table(
        {
            "x": ma.array([1, 2, 3], type=ma.int32()),
            "y": ma.array([4.0, 5.0, 6.0], type=ma.float64()),
            "z": ma.array(["a", "b", "c"]),
        }
    )


# ── Construction ─────────────────────────────────────────────────────────────


def test_table_from_dict():
    t = make_table()
    assert type(t).__name__ == "Table"
    assert t.num_rows == 3
    assert t.num_columns == 3


def test_table_from_list_with_names():
    x = ma.array([1, 2], type=ma.int64())
    y = ma.array(["a", "b"])
    t = ma.table([x, y], names=["x", "y"])
    assert t.num_rows == 2
    assert t.num_columns == 2


# ── Properties ───────────────────────────────────────────────────────────────


def test_table_shape():
    assert make_table().shape == (3, 3)


def test_table_column_names():
    assert list(make_table().column_names) == ["x", "y", "z"]


def test_table_schema():
    t = make_table()
    schema = t.schema
    assert type(schema).__name__ == "Schema"


def test_table_columns():
    cols = make_table().columns
    assert len(cols) == 3


def test_table_str():
    s = str(make_table())
    assert "Table" in s
    assert "num_rows=3" in s


# ── Column access ────────────────────────────────────────────────────────────


def test_table_column_by_index():
    """A table column is a ``ChunkedArray`` — what the table actually holds.

    It used to answer an ``Array``, because ``ChunkedArray`` was not a
    registered type, so ``column()`` combined the chunks and handed back a
    copy whose shape said nothing about the table's."""
    t = make_table()
    col = t.column(0)
    assert type(col).__name__ == "ChunkedArray"
    assert len(col) == 3


def test_table_column_by_name():
    t = make_table()
    col = t.column("y")
    assert type(col).__name__ == "ChunkedArray"


def test_table_column_by_name_not_found():
    with pytest.raises(Exception):
        make_table().column("missing")


# ── Equality ─────────────────────────────────────────────────────────────────


def test_table_equals():
    t1 = make_table()
    t2 = make_table()
    assert t1.equals(t2)
    assert t1 == t2


def test_table_not_equals():
    t = make_table()
    other = ma.table(
        {
            "x": ma.array([9, 9, 9], type=ma.int32()),
            "y": ma.array([4.0, 5.0, 6.0], type=ma.float64()),
            "z": ma.array(["a", "b", "c"]),
        }
    )
    assert not t.equals(other)


# ── to_batches ───────────────────────────────────────────────────────────────


def test_table_to_batches():
    t = make_table()
    batches = t.to_batches()
    assert len(batches) >= 1
    total_rows = sum(b.num_rows for b in batches)
    assert total_rows == 3


# ── to_pydict / to_pylist ───────────────────────────────────────────────────


def test_table_to_pydict():
    t = ma.table(
        {
            "a": ma.array([1, 2], type=ma.int32()),
            "b": ma.array(["x", "y"]),
        }
    )
    d = t.to_pydict()
    assert list(d["a"]) == [1, 2]
    assert list(d["b"]) == ["x", "y"]


def test_table_to_pylist():
    t = ma.table(
        {
            "a": ma.array([1, 2], type=ma.int32()),
            "b": ma.array(["x", "y"]),
        }
    )
    rows = t.to_pylist()
    assert len(rows) == 2
    assert rows[0]["a"] == 1
    assert rows[1]["b"] == "y"


# ── ChunkedArray, Table verbs, constructors ────────────────────────────────


def _chunked_table():
    """Two batches, so the table's columns really are chunked."""
    a = ma.RecordBatch.from_pydict({"n": [1, 2], "s": ["a", "b"]})
    b = ma.RecordBatch.from_pydict({"n": [3], "s": ["c"]})
    return ma.Table.from_batches([a, b])


def test_chunked_array_reports_its_chunks():
    col = _chunked_table().column("n")
    assert isinstance(col, ma.ChunkedArray)
    assert col.num_chunks == 2
    assert len(col) == 3
    assert col.to_pylist() == [1, 2, 3]
    assert [c.to_pylist() for c in col.chunks] == [[1, 2], [3]]


def test_chunked_array_indexes_across_chunks():
    col = _chunked_table().column("n")
    assert [col[i].as_py() for i in range(3)] == [1, 2, 3]
    assert col[-1].as_py() == 3


def test_chunked_array_combine_chunks():
    col = _chunked_table().column("n")
    combined = col.combine_chunks()
    assert isinstance(combined, ma.Array)
    assert combined.to_pylist() == [1, 2, 3]


def test_chunked_array_factory():
    col = ma.chunked_array([ma.array([1, 2]), ma.array([3])])
    assert col.num_chunks == 2
    assert col.to_pylist() == [1, 2, 3]


def test_table_verbs_match_record_batch():
    t = _chunked_table()
    assert t.select(["n"]).column_names == ["n"]
    assert t.drop(["s"]).column_names == ["n"]
    assert t.slice(1, 2).to_pydict() == {"n": [2, 3], "s": ["b", "c"]}
    assert t.filter([True, False, True]).to_pydict() == {
        "n": [1, 3],
        "s": ["a", "c"],
    }
    assert t.take([2, 0]).to_pydict() == {"n": [3, 1], "s": ["c", "a"]}
    assert t.rename_columns(["x", "y"]).column_names == ["x", "y"]


def test_table_from_pydict_and_from_pylist():
    assert ma.Table.from_pydict({"a": [1, 2]}).to_pydict() == {"a": [1, 2]}
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert ma.Table.from_pylist(rows).to_pylist() == rows


def test_from_pylist_fills_a_missing_key_with_null():
    """A ragged list keeps every key any row mentions, in first-seen order."""
    out = ma.Table.from_pylist([{"a": 1}, {"a": 2, "b": 9}])
    assert out.column_names == ["a", "b"]
    assert out.to_pydict() == {"a": [1, 2], "b": [None, 9]}


def test_concat_tables_keeps_one_chunk_per_batch():
    t = ma.Table.from_pydict({"a": [1, 2]})
    out = ma.concat_tables([t, t])
    assert out.num_rows == 4
    assert out.column(0).num_chunks == 2
    assert out.column(0).to_pylist() == [1, 2, 1, 2]


def test_concat_arrays():
    out = ma.concat_arrays([ma.array([1, 2]), ma.array([3])])
    assert out.to_pylist() == [1, 2, 3]


def test_take_accepts_any_integer_index_type():
    """`array([2, 0])` infers int64; `as_int32()` asserts rather than casts, so
    this used to abort the process instead of raising."""
    t = ma.Table.from_pydict({"a": [1, 2, 3]})
    assert t.take([2, 0]).to_pydict() == {"a": [3, 1]}
    assert t.take(ma.array([2, 0], type=ma.int32())).to_pydict() == {"a": [3, 1]}


def test_metadata_verbs_keep_the_tables_chunking():
    """`select` and friends are column-shaped, so they map across the chunks
    rather than combining them — a chunked table stays chunked."""
    t = _chunked_table()
    assert t.column(0).num_chunks == 2
    for out in (
        t.select(["n"]),
        t.drop(["s"]),
        t.rename_columns(["x", "y"]),
        t.remove_column(1),
    ):
        assert out.column(0).num_chunks == 2


def test_row_shaped_verbs_index_against_the_whole_table():
    """`slice`, `filter` and `take` take arguments indexed against the table,
    not against a chunk — handing each chunk the same argument would answer a
    different question per chunk."""
    t = _chunked_table()  # n = [1, 2] + [3]
    assert t.slice(1, 2).to_pydict() == {"n": [2, 3], "s": ["b", "c"]}
    assert t.filter([False, True, True]).to_pydict() == {
        "n": [2, 3],
        "s": ["b", "c"],
    }
    assert t.take([2, 0]).to_pydict() == {"n": [3, 1], "s": ["c", "a"]}


def test_appending_a_column_to_a_chunked_table():
    t = _chunked_table()
    out = t.append_column(ma.field("z", ma.int64()), ma.array([7, 8, 9]))
    assert out.to_pydict()["z"] == [7, 8, 9]
