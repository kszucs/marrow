"""Fused aggregates, and the cases a single-batch test would miss."""

from std.testing import assert_equal, assert_true

from ...builders import col, count_star, lit
from ....arrays import Int32Array
from ....builders import array, arange
from ....dtypes import Int32Type, Int64Type, int32, int64
from ....tabular import RecordBatch, record_batch
from ....kernels.core import Groups
from ...logical import DynValue
from ...physical import Morsel
from ..aggregates import Count, Max, Mean, Min, Product, Sum
from ..leaves import Column, Literal
from ..numeric import Mul


def _b(var v: List[Optional[Int]]) raises -> RecordBatch:
    return record_batch([array(v^, int64).copy()], names=["a"])


def _groups(var g: List[Optional[Int]]) raises -> Int32Array:
    return array(g^, int32).copy()


def _m(var batch: RecordBatch, var ids: Int32Array, n: Int) raises -> Morsel:
    """A morsel carries its grouping, which is what lets a fold be an
    `Operator` rather than needing a trait of its own."""
    return Morsel(batch.to_struct_array(), Groups(ids^, n))


def test_fused_sum_folds_across_morsels() raises:
    """One state, several batches — what `to_state` exists for."""
    var s = col("a", int64).sum().alias("total").to_operator(False)
    _ = s.push(_m(_b([1, 2]), _groups(List[Optional[Int]]()), 1))
    _ = s.push(_m(_b([3, 4]), _groups(List[Optional[Int]]()), 1))
    _ = s.push(_m(_b([5]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([15], int64))


def test_fused_sum_skips_nulls() raises:
    var s = col("a", int64).sum().alias("t").to_operator(False)
    _ = s.push(_m(_b([1, None, 3]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([4], int64))


def test_min_max_expose_null_blindness() raises:
    """`sum` can be silently right over a null whose payload is 0; `min` cannot.
    These are the cases that prove the lane mask is applied."""
    var lo = col("a", int64).min().alias("lo").to_operator(False)
    var hi = col("a", int64).max().alias("hi").to_operator(False)
    _ = lo.push(_m(_b([5, None, 9]), _groups(List[Optional[Int]]()), 1))
    _ = hi.push(_m(_b([5, None, 9]), _groups(List[Optional[Int]]()), 1))
    assert_true(lo.drain().value().to_array(1) == array([5], int64))
    assert_true(hi.drain().value().to_array(1) == array([9], int64))


def test_fused_sum_over_no_rows_is_null() raises:
    """R10, and the live out-of-bounds this design was blocked on: `AggState`
    only grew in `update`, so an aggregate that never updated read a slot that
    did not exist — a crash under ASSERT=all, a silent bad read otherwise."""
    var s = col("a", int64).sum().alias("t").to_operator(False)
    assert_true(s.drain().value().to_array(1).as_int64().is_null(0))


def test_fused_sum_over_empty_batch_is_null() raises:
    var s = col("a", int64).sum().alias("t").to_operator(False)
    _ = s.push(_m(_b(List[Optional[Int]]()), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1).as_int64().is_null(0))


def test_fused_sum_over_all_nulls_is_null() raises:
    var s = col("a", int64).sum().alias("t").to_operator(False)
    _ = s.push(_m(_b([None, None]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1).as_int64().is_null(0))


def test_a_fused_subtree_never_materialises() raises:
    """`sum(a * 2)` — the input is a fused node, so the fold reads its lane."""
    var s = (
        (col("a", int64) * lit(2, int64)).sum().alias("t").to_operator(False)
    )
    _ = s.push(_m(_b([1, 2, 3]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([12], int64))


def test_a_ragged_tail_stays_in_bounds() raises:
    """The row count is not a multiple of the SIMD width. A body without a
    scalar tail
    reads past the view and aborts the process."""
    var b = record_batch([arange[Int64Type](0, 1003).to_dyn()], names=["a"])
    var s = col("a", int64).sum().alias("t").to_operator(False)
    _ = s.push(_m(b.copy(), _groups(List[Optional[Int]]()), 1))
    assert_true(
        s.drain().value().to_array(1) == array([1003 * 1002 // 2], int64)
    )


def test_sum_widens_to_the_accumulator_type() raises:
    var b = record_batch([array([1, 2, 3], int32).copy()], names=["a"])
    var agg = col("a", int32).sum().alias("t")
    var s = agg.to_operator(False)
    _ = s.push(_m(b.copy(), _groups(List[Optional[Int]]()), 1))
    var got = s.drain().value().to_array(1)
    assert_true(got.dtype() == agg.dtype(b.schema))
    assert_true(got == array([6], int64))


def test_grouped_folds_into_slots() raises:
    var s = col("a", int64).sum().alias("t").to_operator(True)
    _ = s.push(_m(_b([1, 2, 3, 4]), _groups([0, 1, 0, 1]), 2))
    _ = s.push(_m(_b([10, 20]), _groups([1, 0]), 2))
    assert_true(
        s.drain().value().to_array(1) == array([1 + 3 + 20, 2 + 4 + 10], int64)
    )


def test_grouped_skips_nulls_per_group() raises:
    var s = col("a", int64).min().alias("t").to_operator(True)
    _ = s.push(_m(_b([5, None, 1, 9]), _groups([0, 0, 1, 1]), 2))
    assert_true(s.drain().value().to_array(1) == array([5, 1], int64))


def test_mean_uses_the_valid_count_as_divisor() raises:
    """The count is not bookkeeping: it is `finalize`'s divisor, and a null
    must not be in it."""
    var s = col("a", int64).mean().alias("t").to_operator(False)
    _ = s.push(_m(_b([1, None, 5]), _groups(List[Optional[Int]]()), 1))
    assert_equal(String(s.drain().value().to_array(1).as_float64()[0]), "3.0")


def test_erasure_answers_as_the_value_it_holds() raises:
    var b = _b([1, 2, 3])
    var agg = col("a", int64).sum().alias("total")
    var boxed = agg.copy()
    assert_equal(boxed.name(), "total")
    assert_equal(boxed.columns()[0], "a")
    assert_true(boxed.dtype(b.schema) == agg.dtype(b.schema))
    var s = boxed.to_operator(False)
    _ = s.push(_m(b.copy(), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([6], int64))


def test_a_fold_reports_spent_on_a_second_drain() raises:
    """`drain` is repeatable, so a fold must be able to say "nothing left".

    The driver calls `drain` until it answers `None`. A fold that answered
    `Some` every time would spin `while True: drain()` forever — it is only
    safe today because `AggregateOperator` happens to call it once, and
    "happens to" is not a contract.
    """
    var s = col("a", int64).sum().alias("t").to_operator(False)
    _ = s.push(_m(_b([1, 2]), _groups(List[Optional[Int]]()), 1))
    assert_true(Bool(s.drain()))
    assert_true(not Bool(s.drain()))


def test_product_folds() raises:
    """`Product` had no test at all — found by auditing public names against
    test references."""
    var s = Product(col("a", int64), "p").to_operator(False)
    _ = s.push(_m(_b([2, 3, 4]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([24], int64))


def test_count_skips_nulls() raises:
    """`COUNT(x)` is the *valid* count, which is what separates it from
    `COUNT(*)` on any nullable column."""
    var s = col("a", int64).count().to_operator(False)
    _ = s.push(_m(_b([1, None, 3, None, 5]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([3], int64))


def test_count_star_counts_every_row_including_nulls() raises:
    """`count_star()` is `CountKernel` over a literal, and a literal is valid
    on every row — so the valid-count of a constant column is the row count.
    The trick is only correct if it survives a *nullable* input column, which
    is the whole point of testing it against one.
    """
    var s = count_star().to_operator(False)
    _ = s.push(_m(_b([1, None, 3, None, 5]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([5], int64))


def test_count_and_count_star_disagree_on_a_nullable_column() raises:
    """Stated as one case because the two are constantly confused, and a test
    that pins each separately does not show that they must differ."""
    var batch = _b([1, None, 3])
    var ids = _groups(List[Optional[Int]]())

    var counted = col("a", int64).count().to_operator(False)
    _ = counted.push(_m(batch.copy(), ids.copy(), 1))

    var starred = count_star().to_operator(False)
    _ = starred.push(_m(batch.copy(), ids.copy(), 1))

    assert_true(counted.drain().value().to_array(1) == array([2], int64))
    assert_true(starred.drain().value().to_array(1) == array([3], int64))


def test_count_is_named_and_aliasable() raises:
    """`count_star()` arrives pre-aliased; `.count()` takes the kernel's name
    until something renames it."""
    assert_equal(col("a", int64).count().name(), "count")
    assert_equal(count_star().name(), "count_star")
    assert_equal(col("a", int64).count().alias("n").name(), "n")
