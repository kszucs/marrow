"""The list family — a materialize-only list column feeding fixed-width breakers:
`length` (-> int32) and `contains` (-> bool). Both delegate to kernels.nested and
compose with the numeric/bool lanes above them (`length(l) + 1`, `contains(l,x) & ...`).
"""

from std.testing import assert_true

from marrow.testing import TestSuite
from marrow.builders import array, ListBuilder, Int64Builder
from marrow.dtypes import int64, int32, ListType
from marrow.tabular import record_batch, RecordBatch
from marrow.expr.lane import (
    col,
    lit,
    Add,
    ListColumn,
    ListLength,
    ListContains,
    into_array,
)


def _list_batch() raises -> RecordBatch:
    # list<int64> column: [[10, 20, 30], [40, 50]]
    var lb = ListBuilder(Int64Builder(capacity=8))
    var child_any = lb.values()
    ref child = child_any.as_int64()
    child.append(10)
    child.append(20)
    child.append(30)
    lb.append_valid()
    child.append(40)
    child.append(50)
    lb.append_valid()
    return record_batch([lb.finish()], names=["l"])


def test_list_length() raises:
    # length([[10,20,30],[40,50]]) = [3, 2]
    var cv = (ListLength(ListColumn[ListType]("l"))).execute(_list_batch())
    assert_true(into_array(cv, 2) == array([3, 2], int32).to_any())


def test_list_length_fuses_above() raises:
    # length(l) + 1 = [4, 3] — the breaker feeds the fused numeric lane
    var cv = (Add(ListLength(ListColumn[ListType]("l")), lit(1, int32))).execute(_list_batch())
    assert_true(into_array(cv, 2) == array([4, 3], int32).to_any())


def test_list_contains() raises:
    # 20 in [10,20,30] = T ; 20 in [40,50] = F  ->  [T, F]
    var cv = (ListContains(ListColumn[ListType]("l"), lit(20, int64))).execute(_list_batch())
    assert_true(into_array(cv, 2) == array([True, False]).to_any())


def main() raises:
    TestSuite.run[__functions_in_module()]()
