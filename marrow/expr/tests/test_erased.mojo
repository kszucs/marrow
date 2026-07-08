"""Tests for the universal value box ``AnyValue`` in ``marrow.expr.values``.

``AnyValue`` wraps any ``Value`` node behind a ``to_array(batch)`` trampoline:
- fused comptime nodes (named columns, ``Greater``/``Less``/``Equal``) — the AOT path;
- the runtime ``DynValue`` interpreter (built via ``col()`` + operators).

The load-bearing property is that fused and interpreted values interchange
through the one box, and a heterogeneous list can hold both. Relational
execution over ``List[AnyValue]`` is covered by ``test_streaming``/``test_plan``.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64
from marrow.tabular import RecordBatch, record_batch
from marrow.expr.values import Table
from marrow.expr.values import AnyValue
from marrow.expr.dynamic import col


struct _Orders:
    var a: Int64Type
    var b: Int64Type
    var name: StringType


def _batch() raises -> RecordBatch:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var s = array(["p", "q", "r", "s", "t"])
    return record_batch(
        [a.copy(), b.copy(), s.copy()], names=["a", "b", "name"]
    )


def test_box_fused_column() raises:
    """A fused named column boxed into AnyValue resolves by name."""
    var t = Table[_Orders]()
    var v = AnyValue(t.a)
    assert_equal(v.name(), "a")
    assert_true(
        v.to_array(_batch()).as_int64().copy() == array([1, 5, 3, 8, 2], int64)
    )


def test_box_dynvalue_arithmetic() raises:
    """A DynValue (col()+col()) boxed into AnyValue interprets over the batch.
    """
    var v = AnyValue(col("a") + col("b"))
    assert_true(
        v.to_array(_batch()).as_int64().copy() == array([5, 9, 7, 12, 6], int64)
    )


def test_box_dynvalue_resolves_by_name() raises:
    """A DynValue col() boxed into AnyValue resolves by name."""
    var v = AnyValue(col("a"))
    assert_true(
        v.to_array(_batch()).as_int64().copy() == array([1, 5, 3, 8, 2], int64)
    )


def test_fused_and_dynvalue_interchange() raises:
    """A heterogeneous list holds a fused *and* an interpreter value; both run
    through the same to_array — fused-vs-interpreted is which node you boxed."""
    var t = Table[_Orders]()
    var values = List[AnyValue]()
    values.append(AnyValue(t.a))  # fused
    values.append(AnyValue(col("b")))  # interpreter
    var batch = _batch()
    assert_true(
        values[0].to_array(batch).as_int64().copy()
        == array([1, 5, 3, 8, 2], int64)
    )
    assert_true(
        values[1].to_array(batch).as_int64().copy()
        == array([4, 4, 4, 4, 4], int64)
    )


def test_write_to_delegates() raises:
    """AnyValue.write_to delegates to the boxed node."""
    var t = Table[_Orders]()
    assert_equal(String(AnyValue(t.a)), "Col[a]")


def main() raises:
    TestSuite.run[__functions_in_module()]()
