"""Tests for the name-resolved column handles in ``marrow.expr.relations``.

- ``Table[Tbl]()`` — the column-access handle over a plain schema struct of
  dtype-tag fields. ``t.a`` reflects each field's dtype into a
  ``NumericColumn[T]`` / ``StringColumn`` leaf (columns carry only a runtime
  ``name``; position is resolved by name against the batch); string fields
  dispatch to ``StringColumn`` automatically.
- ``col(name, dtype)`` — the schema-struct-free, polars-style by-name factory
  producing the same leaves.

(The fully-typed ``Project[*Es]``/``Filter`` layer was removed once the erased
fat-node relations superseded it; execution is tested in ``test_plan`` /
``test_streaming``.)
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64, string
from marrow.tabular import RecordBatch, record_batch
from marrow.expr.values import Table
from marrow.expr.values import Add, Greater, col


struct _Orders:
    """Two int64 columns and one string column, declared in that order."""

    var a: Int64Type
    var b: Int64Type
    var name: StringType


def _make_batch() raises -> RecordBatch:
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var s = array(["x", "y", "z"])
    return record_batch(
        [a.copy(), b.copy(), s.copy()], names=["a", "b", "name"]
    )


# ---------------------------------------------------------------------------
# Table[Tbl]() — column-access handle
# ---------------------------------------------------------------------------


def test_column_name_matches_declared_field() raises:
    """Table[Tbl]().<name> carries the field name; position is resolved by name
    against the batch schema (columns store no index)."""
    var t = Table[_Orders]()
    assert_equal(t.a.field_name(), "a")
    assert_equal(t.b.field_name(), "b")
    assert_equal(t.name.field_name(), "name")


def test_column_executes_without_runtime_schema() raises:
    """NumericColumn resolves data by name."""
    var t = Table[_Orders]()
    var batch = _make_batch()

    ref result = t.a.execute(batch).to_any().as_int64()
    assert_true(result.copy() == array([1, 2, 3], int64))

    ref result_b = t.b.execute(batch).to_any().as_int64()
    assert_true(result_b.copy() == array([10, 20, 30], int64))


def test_named_column_add_fuses() raises:
    """Fully-typed columns compose with the Add fusion node."""
    var t = Table[_Orders]()
    var batch = _make_batch()

    var added = Add(t.a, t.b)
    ref result = added.execute(batch).to_any().as_int64()
    assert_true(result.copy() == array([11, 22, 33], int64))


def test_named_column_gt_fuses() raises:
    """Fully-typed columns compose with the Greater comparison node."""
    var t = Table[_Orders]()
    var batch = _make_batch()

    var pred = Greater(t.a, t.b)
    var result = pred.execute(batch)
    assert_true(result[0].value() == False)
    assert_true(result[1].value() == False)
    assert_true(result[2].value() == False)


def test_named_column_write_to() raises:
    """NumericColumn.write_to() displays the compile-time name."""
    var t = Table[_Orders]()
    assert_equal(String(t.a), "Col[a]")
    assert_equal(String(t.b), "Col[b]")


def test_named_string_column_executes() raises:
    """A string-typed field dispatches to StringColumn and resolves by name."""
    var t = Table[_Orders]()
    var batch = _make_batch()

    var result = t.name.execute(batch)
    assert_equal(result[0].to_string(), "x")
    assert_equal(result[1].to_string(), "y")
    assert_equal(result[2].to_string(), "z")


def test_named_string_column_write_to() raises:
    """StringColumn.write_to() displays the compile-time name."""
    var t = Table[_Orders]()
    assert_equal(String(t.name), "StrCol[name]")


# ---------------------------------------------------------------------------
# col(name, dtype) — schema-struct-free by-name factory
# ---------------------------------------------------------------------------


def test_col_resolves_numeric_by_name() raises:
    """A numeric col resolves its position by name against the batch."""
    var batch = _make_batch()
    ref result = col("a", int64).execute(batch).to_any().as_int64()
    assert_true(result.copy() == array([1, 2, 3], int64))


def test_col_resolves_string_by_name() raises:
    """A string col dispatches to StringColumn and resolves by name."""
    var batch = _make_batch()
    var result = col("name", string).execute(batch)
    assert_equal(result[0].to_string(), "x")
    assert_equal(result[2].to_string(), "z")


def main() raises:
    TestSuite.run[__functions_in_module()]()
