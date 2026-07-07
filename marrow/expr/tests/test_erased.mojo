"""Tests for the erased/unified relational layer in ``marrow.expr.erased``.

Covers the two halves of the ``marrow.expr`` unification prototype
(``docs/expr-unification-plan.md``):

- ``AnyValue`` + self-executing runtime ``Project``/``Filter`` — a walkable,
  rewritable plan tree over **fused-only** value boxes. Same result as the typed
  ``marrow.expr.relations`` layer, at the same (measured) binary size.
- ``DynValue`` — the type-erased tag-interpreter node (the reworked ``Expr``),
  boxed into the *same* ``AnyValue``. The load-bearing property is that fused
  and interpreted values interchange: a plan built from either produces the same
  result, and a single ``Project`` can hold both at once.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64
from marrow.tabular import RecordBatch, record_batch
from marrow.expr.relations import Table
from marrow.expr.values import Gt
from marrow.expr.erased import AnyValue, DynValue, Project, ADD, GT


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


# ---------------------------------------------------------------------------
# Fused path — AnyValue over the fused named columns
# ---------------------------------------------------------------------------


def test_fused_project() raises:
    """Project over boxed fused columns assembles the right RecordBatch."""
    var t = Table[_Orders]()
    var batch = _batch()
    var exprs = List[AnyValue]()
    exprs.append(AnyValue(t.a))
    exprs.append(AnyValue(t.name))
    var result = Project(exprs^).execute(batch)

    assert_equal(result.num_rows(), 5)
    assert_equal(len(result.columns), 2)
    assert_equal(result.schema.fields[0].name, "a")
    assert_equal(result.schema.fields[1].name, "name")
    assert_true(result.columns[0].as_int64().copy() == array([1, 5, 3, 8, 2], int64))


def test_fused_filter() raises:
    """SELECT a, name WHERE a > b over the fused path keeps rows 5 and 8."""
    var t = Table[_Orders]()
    var batch = _batch()
    var exprs = List[AnyValue]()
    exprs.append(AnyValue(t.a))
    exprs.append(AnyValue(t.name))
    var result = Project(exprs^).filter(AnyValue(Gt(t.a, t.b))).execute(batch)

    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].as_int64().copy() == array([5, 8], int64))


# ---------------------------------------------------------------------------
# Interpreter path — DynValue boxed into the same AnyValue
# ---------------------------------------------------------------------------


def test_dynvalue_arithmetic() raises:
    """DynValue interprets a binary op over LOAD children."""
    var batch = _batch()
    var expr = DynValue.binary(
        ADD, AnyValue(DynValue.load(0, "a")), AnyValue(DynValue.load(1, "b"))
    )
    ref result = expr.to_array(batch).as_int64()
    assert_true(result.copy() == array([5, 9, 7, 12, 6], int64))


def test_dynvalue_load_resolves_by_name() raises:
    """A DynValue LOAD resolves by name (matching the fused columns), so a wrong
    index is overridden by the name."""
    var batch = _batch()
    var expr = DynValue.load(99, "a")  # bogus index, name wins
    ref result = expr.to_array(batch).as_int64()
    assert_true(result.copy() == array([1, 5, 3, 8, 2], int64))


def test_dynvalue_matches_fused() raises:
    """The same query built from DynValue interpreter nodes produces the same
    result as the fused path — fused and interpreted values interchange."""
    var batch = _batch()
    var exprs = List[AnyValue]()
    exprs.append(AnyValue(DynValue.load(0, "a")))
    exprs.append(AnyValue(DynValue.load(2, "name")))
    var pred = AnyValue(
        DynValue.gt(
            AnyValue(DynValue.load(0, "a")), AnyValue(DynValue.load(1, "b"))
        )
    )
    var result = Project(exprs^).filter(pred^).execute(batch)

    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].as_int64().copy() == array([5, 8], int64))


def test_cobox_fused_and_dynvalue() raises:
    """A single Project holds a fused column *and* an interpreter column — the
    box is shared, the choice is per-node."""
    var t = Table[_Orders]()
    var batch = _batch()
    var exprs = List[AnyValue]()
    exprs.append(AnyValue(t.a))  # fused
    exprs.append(AnyValue(DynValue.load(1, "b")))  # interpreter
    var result = Project(exprs^).execute(batch)

    assert_equal(result.num_rows(), 5)
    assert_true(result.columns[0].as_int64().copy() == array([1, 5, 3, 8, 2], int64))
    assert_true(result.columns[1].as_int64().copy() == array([4, 4, 4, 4, 4], int64))


def main() raises:
    TestSuite.run[__functions_in_module()]()
