"""Fused aggregates, and the cases a single-batch test would miss."""

from std.testing import assert_equal, assert_true

from ....arrays import Int32Array
from ....builders import array, arange
from ....dtypes import Int32Type, Int64Type, int32, int64
from ....tabular import RecordBatch, record_batch
from ...core import DynAggValue
from ..aggregates import Max, Mean, Min, Sum
from ..leaves import Column, Literal
from ..numeric import Mul


def _b(var v: List[Optional[Int]]) raises -> RecordBatch:
    return record_batch([array(v^, int64).copy()], names=["a"])


def _groups(var g: List[Optional[Int]]) raises -> Int32Array:
    return array(g^, int32).copy()


def test_fused_sum_folds_across_morsels() raises:
    """One state, several batches — what `to_state` exists for."""
    var s = Sum(Column[Int64Type]("a"), "total").to_state(False)
    s.update(_b([1, 2]), _groups(List[Optional[Int]]()), 1)
    s.update(_b([3, 4]), _groups(List[Optional[Int]]()), 1)
    s.update(_b([5]), _groups(List[Optional[Int]]()), 1)
    assert_true(s.finish(1) == array([15], int64))


def test_fused_sum_skips_nulls() raises:
    var s = Sum(Column[Int64Type]("a"), "t").to_state(False)
    s.update(_b([1, None, 3]), _groups(List[Optional[Int]]()), 1)
    assert_true(s.finish(1) == array([4], int64))


def test_min_max_expose_null_blindness() raises:
    """`sum` can be silently right over a null whose payload is 0; `min` cannot.
    These are the cases that prove the lane mask is applied."""
    var lo = Min(Column[Int64Type]("a"), "lo").to_state(False)
    var hi = Max(Column[Int64Type]("a"), "hi").to_state(False)
    lo.update(_b([5, None, 9]), _groups(List[Optional[Int]]()), 1)
    hi.update(_b([5, None, 9]), _groups(List[Optional[Int]]()), 1)
    assert_true(lo.finish(1) == array([5], int64))
    assert_true(hi.finish(1) == array([9], int64))


def test_fused_sum_over_no_rows_is_null() raises:
    """R10, and the live out-of-bounds this design was blocked on: `AggState`
    only grew in `update`, so an aggregate that never updated read a slot that
    did not exist — a crash under ASSERT=all, a silent bad read otherwise."""
    var s = Sum(Column[Int64Type]("a"), "t").to_state(False)
    assert_true(s.finish(1).as_int64().is_null(0))


def test_fused_sum_over_empty_batch_is_null() raises:
    var s = Sum(Column[Int64Type]("a"), "t").to_state(False)
    s.update(_b(List[Optional[Int]]()), _groups(List[Optional[Int]]()), 1)
    assert_true(s.finish(1).as_int64().is_null(0))


def test_fused_sum_over_all_nulls_is_null() raises:
    var s = Sum(Column[Int64Type]("a"), "t").to_state(False)
    s.update(_b([None, None]), _groups(List[Optional[Int]]()), 1)
    assert_true(s.finish(1).as_int64().is_null(0))


def test_a_fused_subtree_never_materialises() raises:
    """`sum(a * 2)` — the input is a fused node, so the fold reads its lane."""
    var s = Sum(
        Mul(Column[Int64Type]("a"), Literal[Int64Type](2)), "t"
    ).to_state(False)
    s.update(_b([1, 2, 3]), _groups(List[Optional[Int]]()), 1)
    assert_true(s.finish(1) == array([12], int64))


def test_a_ragged_tail_stays_in_bounds() raises:
    """The row count is not a multiple of the SIMD width. A body without a
    scalar tail
    reads past the view and aborts the process."""
    var b = record_batch([arange[Int64Type](0, 1003).to_dyn()], names=["a"])
    var s = Sum(Column[Int64Type]("a"), "t").to_state(False)
    s.update(b, _groups(List[Optional[Int]]()), 1)
    assert_true(s.finish(1) == array([1003 * 1002 // 2], int64))


def test_sum_widens_to_the_accumulator_type() raises:
    var b = record_batch([array([1, 2, 3], int32).copy()], names=["a"])
    var agg = Sum(Column[Int32Type]("a"), "t")
    var s = agg.to_state(False)
    s.update(b, _groups(List[Optional[Int]]()), 1)
    var got = s.finish(1)
    assert_true(got.dtype() == agg.dtype(b.schema))
    assert_true(got == array([6], int64))


def test_grouped_folds_into_slots() raises:
    var s = Sum(Column[Int64Type]("a"), "t").to_state(True)
    s.update(_b([1, 2, 3, 4]), _groups([0, 1, 0, 1]), 2)
    s.update(_b([10, 20]), _groups([1, 0]), 2)
    assert_true(s.finish(2) == array([1 + 3 + 20, 2 + 4 + 10], int64))


def test_grouped_skips_nulls_per_group() raises:
    var s = Min(Column[Int64Type]("a"), "t").to_state(True)
    s.update(_b([5, None, 1, 9]), _groups([0, 0, 1, 1]), 2)
    assert_true(s.finish(2) == array([5, 1], int64))


def test_mean_uses_the_valid_count_as_divisor() raises:
    """The count is not bookkeeping: it is `finalize`'s divisor, and a null
    must not be in it."""
    var s = Mean(Column[Int64Type]("a"), "t").to_state(False)
    s.update(_b([1, None, 5]), _groups(List[Optional[Int]]()), 1)
    assert_equal(String(s.finish(1).as_float64()[0]), "3.0")


def test_erasure_answers_as_the_value_it_holds() raises:
    var b = _b([1, 2, 3])
    var agg = Sum(Column[Int64Type]("a"), "total")
    var boxed = DynAggValue(agg.copy())
    assert_equal(boxed.name(), "total")
    assert_equal(boxed.columns()[0], "a")
    assert_true(boxed.dtype(b.schema) == agg.dtype(b.schema))
    var s = boxed.to_state(False)
    s.update(b, _groups(List[Optional[Int]]()), 1)
    assert_true(s.finish(1) == array([6], int64))
