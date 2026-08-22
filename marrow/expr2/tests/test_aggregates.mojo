"""Reductions: the pure description, and the fold it starts.

The claims worth pinning are the ones a single-batch test would miss —
accumulating **across** morsels, and what an empty or all-null input yields.
Both are where an aggregate differs from a value, and both are wrong by default
if the accumulator is built per batch instead of per query.
"""

from std.testing import assert_equal, assert_true

from ...builders import array
from ...dtypes import Int32Type, Int64Type, int32, int64
from ...tabular import RecordBatch, record_batch
from ..aggregates import DynReduction
from ..`comptime`.leaves import Column, Literal
from ..`comptime`.numeric import Mul
from ..`comptime`.reductions import Max, Mean, Min, Sum


def _batch(var values: List[Optional[Int]]) raises -> RecordBatch:
    return record_batch([array(values^, int64).copy()], names=["a"])


def test_a_reduction_folds_across_morsels() raises:
    """One accumulator, several batches — the case a per-batch kernel call
    cannot express and the reason `to_accumulator` exists at all."""
    var acc = Sum(Column[Int64Type]("a"), "total").to_accumulator()
    acc.update(_batch([1, 2]))
    acc.update(_batch([3, 4]))
    acc.update(_batch([5]))
    assert_equal(String(acc.finish()), "15")


def test_a_reduction_skips_nulls() raises:
    var acc = Sum(Column[Int64Type]("a"), "total").to_accumulator()
    acc.update(_batch([1, None, 3]))
    assert_equal(String(acc.finish()), "4")


def test_a_reduction_over_no_rows_is_null() raises:
    """SQL's answer, not the kernel's identity: `SUM` of nothing is NULL, not
    zero. A fold that seeded with `identity` and never checked would say 0."""
    var acc = Sum(Column[Int64Type]("a"), "total").to_accumulator()
    acc.update(_batch(List[Optional[Int]]()))
    assert_true(acc.finish().is_null())


def test_a_reduction_over_all_nulls_is_null() raises:
    var acc = Sum(Column[Int64Type]("a"), "total").to_accumulator()
    acc.update(_batch([None, None]))
    assert_true(acc.finish().is_null())


def test_sum_widens_to_the_accumulator_type() raises:
    """`sum(int32)` is `int64`. The rule lives in `AggKernel.AccType`; this
    pins that the expression layer reports the same thing it produces, which
    is what a plan's schema is built from."""
    var b = record_batch([array([1, 2, 3], int32).copy()], names=["a"])
    var agg = Sum(Column[Int32Type]("a"), "total")
    assert_true(agg.dtype(b.schema) == DynReduction(agg.copy()).dtype(b.schema))
    var acc = agg.to_accumulator()
    acc.update(b)
    assert_true(acc.finish().type() == agg.dtype(b.schema))


def test_a_reduction_folds_a_fused_input() raises:
    """`sum(a * 2)` never materialises `a * 2`: the subtree evaluates into the
    accumulator, which is the point of taking a `NumericValue` rather than a
    column name."""
    var acc = Sum(
        Mul(Column[Int64Type]("a"), Literal[Int64Type](2)), "total"
    ).to_accumulator()
    acc.update(_batch([1, 2, 3]))
    assert_equal(String(acc.finish()), "12")


def test_min_max_and_mean() raises:
    var mn = Min(Column[Int64Type]("a"), "lo").to_accumulator()
    var mx = Max(Column[Int64Type]("a"), "hi").to_accumulator()
    var av = Mean(Column[Int64Type]("a"), "avg").to_accumulator()
    for ref b in [_batch([5, 1]), _batch([9, 3])]:
        mn.update(b)
        mx.update(b)
        av.update(b)
    assert_equal(String(mn.finish()), "1")
    assert_equal(String(mx.finish()), "9")
    assert_equal(String(av.finish()), "4.5")


def test_erasure_preserves_what_a_reduction_reports() raises:
    """A boxed reduction must answer exactly as the value it holds — the same
    claim `DynValue` carries, and the reason both exist."""
    var b = _batch([1, 2, 3])
    var agg = Sum(Column[Int64Type]("a"), "total")
    var boxed = DynReduction(agg.copy())
    assert_equal(boxed.name(), agg.name())
    assert_equal(boxed.name(), "total")
    assert_equal(len(boxed.columns()), 1)
    assert_equal(boxed.columns()[0], "a")
    assert_true(boxed.dtype(b.schema) == agg.dtype(b.schema))
    var acc = boxed.to_accumulator()
    acc.update(b)
    assert_equal(String(acc.finish()), "6")
