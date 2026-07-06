"""Tests for named comptime-typed expression nodes (``marrow.aot.relations``).

Mirrors ``dyn/tests/test_aot_injection.mojo``'s coverage of
``values.Column``/``Add``, but for the fully-typed
``table.Column[Tbl, name, T]`` variant used by the AOT relational layer
(``docs/aot-relations-design.md``). Unlike
``values.Column[T]``, there is no runtime ``index`` field and no runtime
``Schema`` lookup anywhere — ``index`` is a ``comptime`` constant derived by
reflecting ``name``'s position on ``Tbl`` (here, the enclosing ``Orders``
struct itself). Confirms:
- ``t.a.index`` is the correct compile-time constant for each declared field.
- Fused execution (direct, via ``Add``, via ``Gt``) is identical to the
  unnamed ``values.Column``.
- ``StringColumn[Tbl, name]`` resolves and executes correctly.

A name that doesn't exist on ``Tbl`` is a compile error (``struct 'X' has no
field named 'y'``), not a runtime exception — there is no
``compile_fail``-style test harness in this project, so that behavior is
documented here rather than asserted by a test.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import Int64Array
from marrow.builders import array
from marrow.dtypes import Int64Type, int64
from marrow.tabular import RecordBatch, record_batch
from marrow.aot.relations import Column, StringColumn, Table
from marrow.aot.values import Add, Gt


struct _Orders(Table):
    """Test fixture: a fully-typed table with two int64 columns and one
    string column, declared in that order.
    """

    var a: Column[_Orders, "a", Int64Type]
    var b: Column[_Orders, "b", Int64Type]
    var name: StringColumn[_Orders, "name"]

    def __init__(out self):
        self.a = {}
        self.b = {}
        self.name = {}


def _make_batch() raises -> RecordBatch:
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var s = array(["x", "y", "z"])
    return record_batch(
        [a.copy(), b.copy(), s.copy()], names=["a", "b", "name"]
    )


def test_column_index_is_compile_time_constant() raises:
    """Column[Tbl, name, T].index matches Tbl's declared field order."""
    var t = _Orders()
    assert_equal(t.a.index, 0)
    assert_equal(t.b.index, 1)
    assert_equal(t.name.index, 2)


def test_column_executes_without_runtime_schema() raises:
    """Column[Tbl, name, T].execute resolves data via its baked-in index."""
    var t = _Orders()
    var batch = _make_batch()

    ref result = t.a.execute(batch).to_any().as_int64()
    assert_true(result.copy() == array([1, 2, 3], int64))

    ref result_b = t.b.execute(batch).to_any().as_int64()
    assert_true(result_b.copy() == array([10, 20, 30], int64))


def test_named_column_add_fuses() raises:
    """Fully-typed columns compose with the existing Add fusion node."""
    var t = _Orders()
    var batch = _make_batch()

    var added = Add(t.a, t.b)
    ref result = added.execute(batch).to_any().as_int64()
    assert_true(result.copy() == array([11, 22, 33], int64))


def test_named_column_gt_fuses() raises:
    """Fully-typed columns compose with the Gt comparison node."""
    var t = _Orders()
    var batch = _make_batch()

    var pred = Gt(t.a, t.b)
    var result = pred.execute(batch)
    assert_true(result[0].value() == False)
    assert_true(result[1].value() == False)
    assert_true(result[2].value() == False)


def test_named_column_write_to() raises:
    """Column[Tbl, name, T].write_to() displays the compile-time name."""
    var t = _Orders()
    assert_equal(String(t.a), "Col[a]")
    assert_equal(String(t.b), "Col[b]")


def test_named_string_column_executes() raises:
    """StringColumn[Tbl, name] resolves and executes via its baked-in index.
    """
    var t = _Orders()
    var batch = _make_batch()

    var result = t.name.execute(batch)
    assert_equal(result[0].to_string(), "x")
    assert_equal(result[1].to_string(), "y")
    assert_equal(result[2].to_string(), "z")


def test_named_string_column_write_to() raises:
    """StringColumn[Tbl, name].write_to() displays the compile-time name."""
    var t = _Orders()
    assert_equal(String(t.name), "StrCol[name]")


def main() raises:
    TestSuite.run[__functions_in_module()]()
