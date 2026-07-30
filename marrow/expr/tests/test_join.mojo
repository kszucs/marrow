"""Executor-level tests for join plan nodes."""

from std.testing import assert_equal, assert_true

from ...arrays import DynArray
from ...builders import array, PrimitiveBuilder, Int64Builder
from ...dtypes import int64, float64, Int64Type
from ...tabular import record_batch, RecordBatch
from ...expr import (
    DynValue,
    col,
    lit,
    in_memory_table,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_ALL,
    JOIN_ANY,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _batch(k: List[Int], v: List[Int]) raises -> RecordBatch:
    var a = Int64Builder(capacity=len(k))
    var b = Int64Builder(capacity=len(v))
    for x in k:
        a.append(Scalar[int64.native](x))
    for x in v:
        b.append(Scalar[int64.native](x))
    var cols = List[DynArray]()
    cols.append(a.finish().to_dyn())
    cols.append(b.finish().to_dyn())
    return record_batch(cols^, names=["k", "v"])


def _keys(v: List[Int]) -> List[DynValue]:
    var r = List[DynValue]()
    r.append(col("k"))
    return r^


# ---------------------------------------------------------------------------
# execute — INNER join
# ---------------------------------------------------------------------------


def test_execute_inner_join_basic() raises:
    """End-to-end inner join via the expression/executor pipeline."""
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    lk.append(3)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    lv.append(30)
    var left_batch = _batch(lk, lv)

    var rk = List[Int]()
    rk.append(2)
    rk.append(3)
    rk.append(4)
    var rv = List[Int]()
    rv.append(200)
    rv.append(300)
    rv.append(400)
    var right_batch = _batch(rk, rv)

    var left_plan = in_memory_table(left_batch)
    var right_plan = in_memory_table(right_batch)

    var keys_left = [col("k")]
    var keys_right = [col("k")]

    var result = left_plan.join(right_plan, keys_left, keys_right).execute()
    assert_equal(result.num_rows(), 2)
    assert_equal(result.num_columns(), 4)  # k, v, k_right, v_right


def test_execute_inner_join_no_matches() raises:
    """Inner join with no matching keys returns empty result."""
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    var left_batch = _batch(lk, lv)

    var rk = List[Int]()
    rk.append(3)
    rk.append(4)
    var rv = List[Int]()
    rv.append(30)
    rv.append(40)
    var right_batch = _batch(rk, rv)

    var left_plan = in_memory_table(left_batch)
    var right_plan = in_memory_table(right_batch)

    var keys_left = [col("k")]
    var keys_right = [col("k")]

    var result = left_plan.join(right_plan, keys_left, keys_right).execute()
    assert_equal(result.num_rows(), 0)


# ---------------------------------------------------------------------------
# execute — LEFT join
# ---------------------------------------------------------------------------


def test_execute_left_join() raises:
    """Left join keeps all left rows, nulls for unmatched right side."""
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    lk.append(3)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    lv.append(30)
    var left_batch = _batch(lk, lv)

    var rk = List[Int]()
    rk.append(2)
    var rv = List[Int]()
    rv.append(200)
    var right_batch = _batch(rk, rv)

    var left_plan = in_memory_table(left_batch)
    var right_plan = in_memory_table(right_batch)

    var keys_left = [col("k")]
    var keys_right = [col("k")]

    var result = (
        left_plan.join(right_plan, keys_left, keys_right, how=JOIN_LEFT)
    ).execute()
    assert_equal(result.num_rows(), 3)
    assert_equal(result.columns[2].null_count(), 2)  # 2 unmatched right nulls


# ---------------------------------------------------------------------------
# execute — SEMI join
# ---------------------------------------------------------------------------


def test_execute_semi_join() raises:
    """Semi join returns only left rows that have a match, left columns only."""
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    lk.append(3)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    lv.append(30)
    var left_batch = _batch(lk, lv)

    var rk = List[Int]()
    rk.append(2)
    rk.append(2)  # duplicate — semi should deduplicate to 1 output row
    var rv = List[Int]()
    rv.append(200)
    rv.append(201)
    var right_batch = _batch(rk, rv)

    var left_plan = in_memory_table(left_batch)
    var right_plan = in_memory_table(right_batch)

    var keys_left = [col("k")]
    var keys_right = [col("k")]

    var result = (
        left_plan.join(right_plan, keys_left, keys_right, how=JOIN_SEMI)
    ).execute()
    assert_equal(result.num_rows(), 1)
    assert_equal(result.num_columns(), 2)  # left columns only


# ---------------------------------------------------------------------------
# execute — ANTI join
# ---------------------------------------------------------------------------


def test_execute_anti_join() raises:
    """Anti join returns left rows with no match."""
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    lk.append(3)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    lv.append(30)
    var left_batch = _batch(lk, lv)

    var rk = List[Int]()
    rk.append(2)
    var rv = List[Int]()
    rv.append(200)
    var right_batch = _batch(rk, rv)

    var left_plan = in_memory_table(left_batch)
    var right_plan = in_memory_table(right_batch)

    var keys_left = [col("k")]
    var keys_right = [col("k")]

    var result = (
        left_plan.join(right_plan, keys_left, keys_right, how=JOIN_ANTI)
    ).execute()
    assert_equal(result.num_rows(), 2)
    assert_equal(result.num_columns(), 2)  # left columns only
    ref k = result.columns[0].as_int64()
    assert_equal(k[0].value(), Scalar[int64.native](1))
    assert_equal(k[1].value(), Scalar[int64.native](3))


# ---------------------------------------------------------------------------
# execute — join then filter pipeline
# ---------------------------------------------------------------------------


def test_join_then_filter() raises:
    """Join followed by a filter on the result."""
    # left: (k=1,v=10), (k=2,v=20), (k=3,v=30)
    # right: (k=1,v=100), (k=2,v=200), (k=3,v=300)
    # inner join → 3 rows; filter v > 10 → 2 rows
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    lk.append(3)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    lv.append(30)
    var left_batch = _batch(lk, lv)

    var rk = List[Int]()
    rk.append(1)
    rk.append(2)
    rk.append(3)
    var rv = List[Int]()
    rv.append(100)
    rv.append(200)
    rv.append(300)
    var right_batch = _batch(rk, rv)

    var left_plan = in_memory_table(left_batch)
    var right_plan = in_memory_table(right_batch)

    var keys_left = [col("k")]
    var keys_right = [col("k")]

    # After join: columns are k(0), v(1), k_right(2), v_right(3)
    var result = (
        left_plan.join(right_plan, keys_left, keys_right).filter(
            col("v") > lit[Int64Type](10)
        )
    ).execute()
    assert_equal(result.num_rows(), 2)


def test_join_plan_is_reusable() raises:
    """The join's built hash index is reset on clone, so re-executing the same
    plan rebuilds and yields the same result (no single-use state)."""
    var left_batch = _batch([1, 2, 3], [10, 20, 30])
    var right_batch = _batch([2, 3, 4], [200, 300, 400])
    var plan = in_memory_table(left_batch).join(
        in_memory_table(right_batch), [col("k")], [col("k")]
    )
    var r1 = plan.execute()
    var r2 = plan.execute()
    assert_equal(r1.num_rows(), 2)
    assert_equal(r2.num_rows(), 2)
