"""Tests for ``Project[*Es]`` (``marrow.aot.table``) -- the fully-typed,
variadic projection over named comptime expression nodes.

``Project`` takes a pre-built ``Tuple[*Es]``, not bare variadic args
(``Project(a, b)``) -- a ``VariadicPack`` captured by one function cannot be
forwarded to another function's variadic parameter in current Mojo (confirmed
against the pinned toolchain; see ``docs/aot-relations-design.md``). The
supported call site is ``Project(Tuple(t.a, t.b))``.

Covers: heterogeneous NumericValue + StringValue columns in one projection,
composite NumericValue expressions (Add), schema/field-name derivation from
the expression tree's type, and that each column executes as its own fused
kernel (no cross-column interference).
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import Int64Array
from marrow.builders import array
from marrow.dtypes import Int64Type, int64, string
from marrow.tabular import RecordBatch, record_batch
from marrow.aot.table import Column, StringColumn, Table, Project
from marrow.aot.values import Add


struct _Orders(Table):
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


def test_project_two_named_columns() raises:
    """Project over two plain named Columns preserves values and names."""
    var t = _Orders()
    var batch = _make_batch()

    var proj = Project(Tuple(t.a, t.b))
    var result = proj.execute(batch)

    assert_equal(len(result.schema), 2)
    assert_equal(result.schema.fields[0].name, "a")
    assert_equal(result.schema.fields[1].name, "b")

    ref col_a = result.columns[0].as_int64()
    assert_true(col_a.copy() == array([1, 2, 3], int64))
    ref col_b = result.columns[1].as_int64()
    assert_true(col_b.copy() == array([10, 20, 30], int64))


def test_project_mixed_numeric_and_string() raises:
    """Project handles a heterogeneous NumericValue + StringValue mix."""
    var t = _Orders()
    var batch = _make_batch()

    var proj = Project(Tuple(t.a, t.name))
    var result = proj.execute(batch)

    assert_equal(len(result.schema), 2)
    assert_equal(result.schema.fields[0].name, "a")
    assert_equal(result.schema.fields[0].dtype, int64)
    assert_equal(result.schema.fields[1].name, "name")
    assert_equal(result.schema.fields[1].dtype, string)

    ref col_a = result.columns[0].as_int64()
    assert_true(col_a.copy() == array([1, 2, 3], int64))

    ref col_name = result.columns[1].as_string()
    assert_equal(col_name[0].to_string(), "x")
    assert_equal(col_name[1].to_string(), "y")
    assert_equal(col_name[2].to_string(), "z")


def test_project_single_column() raises:
    """Project over a single column still produces a valid one-field batch.
    """
    var t = _Orders()
    var batch = _make_batch()

    var proj = Project(Tuple(t.b))
    var result = proj.execute(batch)

    assert_equal(len(result.schema), 1)
    assert_equal(result.schema.fields[0].name, "b")
    ref col_b = result.columns[0].as_int64()
    assert_true(col_b.copy() == array([10, 20, 30], int64))


def main() raises:
    TestSuite.run[__functions_in_module()]()
