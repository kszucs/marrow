"""The nodes that consume a list column.

`ListValue` declares no lane — a list element is a whole sub-array — so every
list operation is a node of *another* family that binds the column and produces
a fixed-width result. `ListLength` is the numeric instance and
`ArrayContains` the boolean one; what these cases pin is that the family
crossing works and that the kernels' null rules survive it.
"""

from std.testing import assert_equal, assert_true

from ...builders import array_contains, array_length, col, lit
from ....arrays import BoolArray
from ...bindings import Bindings
from ....builders import BoolBuilder, Int64Builder, ListBuilder, array
from ....dtypes import int64, int32, list_
from ....tabular import RecordBatch, record_batch
from ..core import ComptimeValue


def _lists() raises -> RecordBatch:
    """`xs` = [[1, 2, 3], [4], [], null], `needle` = [2, 4, 1, 1]."""
    var ints = Int64Builder()
    var lists = ListBuilder(ints^)
    var child_any = lists.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    child.append(3)
    lists.append_valid()
    child.append(4)
    lists.append_valid()
    lists.append_valid()
    lists.append_null()
    return record_batch(
        [lists.finish().to_dyn(), array([2, 4, 1, 1], int64).to_dyn()],
        names=["xs", "needle"],
    )


def _nested_bools(codes: List[Int]) raises -> BoolArray:
    """`1` TRUE, `0` FALSE, `-1` NULL."""
    var b = BoolBuilder(capacity=len(codes))
    for ref c in codes:
        if c < 0:
            b.append_null()
        else:
            b.append(c == 1)
    return b.finish()


def _nested_eval(v: Some[ComptimeValue], b: RecordBatch) raises -> BoolArray:
    return (
        v.evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .as_bool()
        .copy()
    )


def test_array_contains_searches_each_row_with_its_own_value() raises:
    """The search value is a column, not a constant: row `i` looks for
    `needle[i]` in `xs[i]`."""
    var b = _lists()
    var v = array_contains(col("xs", list_(int64)), col("needle", int64))
    assert_true(_nested_eval(v, b) == _nested_bools([1, 1, 0, -1]))


def test_array_contains_is_null_exactly_where_the_list_is() raises:
    """`ArrayContainsKernel` propagates the list's validity and nothing else,
    and `ColumnBound` reads that answer back rather than re-deriving it — an
    empty list is FALSE, a null list is NULL."""
    var b = _lists()
    var v = array_contains(col("xs", list_(int64)), col("needle", int64))
    var got = _nested_eval(v, b)
    assert_equal(got.null_count(), 1)
    assert_true(got.is_null(3))
    assert_true(not got[2].value())


def test_array_contains_takes_a_literal_search_value() raises:
    """A constant needle is `lit`, which stays `Shape.scalar` until `evaluate`
    broadcasts it — so the node needs no scalar special case."""
    var b = _lists()
    var v = array_contains(col("xs", list_(int64)), lit(4, int64))
    assert_true(_nested_eval(v, b) == _nested_bools([0, 1, 0, -1]))


def test_array_contains_composes_with_a_boolean_connective() raises:
    """It is an ordinary `BoolValue`, so it is an ordinary operand of `AND`."""
    var b = _lists()
    var v = array_contains(col("xs", list_(int64)), lit(1, int64)) & (
        array_length(col("xs", list_(int64))) > lit(2, int32)
    )
    assert_true(_nested_eval(v, b) == _nested_bools([1, 0, 0, -1]))


def test_array_contains_reports_both_operands_columns() raises:
    """Projection pushdown reads this list, so a two-operand node has to merge
    rather than forward one side."""
    var v = array_contains(col("xs", list_(int64)), col("needle", int64))
    var cols = v.columns()
    assert_equal(len(cols), 2)
    assert_equal(cols[0], "xs")
    assert_equal(cols[1], "needle")
    assert_equal(v.name(), "")
