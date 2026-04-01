"""Tests for the hash join kernel."""

from std.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true, assert_false, TestSuite

from marrow.arrays import AnyArray, PrimitiveArray, StringArray, StructArray
from marrow.builders import array, PrimitiveBuilder, StringBuilder
from marrow.dtypes import int32, int64, uint64, float64, string, Field
from marrow.tabular import record_batch
from marrow.kernels.filter import take
from marrow.kernels.join import hash_join
from marrow.expr.relations import (
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_ALL,
    JOIN_ANY,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _int32_struct(col0: List[Int], col1: List[Int]) raises -> StructArray:
    var a = PrimitiveBuilder[Int32Type](capacity=len(col0))
    var b = PrimitiveBuilder[Int32Type](capacity=len(col1))
    for v in col0:
        a.append(Scalar[int32.native](v))
    for v in col1:
        b.append(Scalar[int32.native](v))
    var cols = List[AnyArray]()
    cols.append(a.finish().to_any())
    cols.append(b.finish().to_any())
    return record_batch(cols^, names=["k", "v"]).to_struct_array()


def _left_on() -> List[Int]:
    var l = List[Int]()
    l.append(0)
    return l^


def _right_on() -> List[Int]:
    var r = List[Int]()
    r.append(0)
    return r^


# ---------------------------------------------------------------------------
# take — standalone tests
# ---------------------------------------------------------------------------


def test_take_primitive_basic() raises:
    """Gather elements from a primitive array at given indices."""
    var a: AnyArray = array[Int32Type]([10, 20, 30, 40])
    var result = take(a.copy(), array[Int32Type]([2, 0, 3]))
    ref r = result.as_primitive[Int32Type]()
    assert_equal(r[0], Scalar[int32.native](30))
    assert_equal(r[1], Scalar[int32.native](10))
    assert_equal(r[2], Scalar[int32.native](40))


def test_take_null_index_produces_null() raises:
    """Null index in take produces a null output element."""
    var a: AnyArray = array[Int32Type]([10, 20, 30])
    var idx = PrimitiveBuilder[Int32Type](capacity=2)
    idx.append_null()
    idx.append(Scalar[int32.native](1))
    var result = take(a.copy(), idx.finish())
    assert_equal(result.null_count(), 1)
    assert_false(result.is_valid(0))
    assert_true(result.is_valid(1))


# ---------------------------------------------------------------------------
# hash_join — INNER join
# ---------------------------------------------------------------------------


def test_inner_join_basic() raises:
    """Basic inner join: matching rows from both sides."""
    # left: (k=1,v=10), (k=2,v=20), (k=3,v=30)
    # right: (k=2,v=200), (k=3,v=300), (k=4,v=400)
    # expected: (k=2,v=20,k_right=2,v_right=200), (k=3,v=30,k_right=3,v_right=300)
    var left_keys = List[Int]()
    left_keys.append(1)
    left_keys.append(2)
    left_keys.append(3)
    var left_vals = List[Int]()
    left_vals.append(10)
    left_vals.append(20)
    left_vals.append(30)
    var left = _int32_struct(left_keys, left_vals)

    var right_keys = List[Int]()
    right_keys.append(2)
    right_keys.append(3)
    right_keys.append(4)
    var right_vals = List[Int]()
    right_vals.append(200)
    right_vals.append(300)
    right_vals.append(400)
    var right = _int32_struct(right_keys, right_vals)

    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(len(result), 2)
    assert_equal(len(result.children), 4)  # k, v, k_right, v_right


def test_inner_join_no_matches() raises:
    """Inner join with no matching keys returns empty batch."""
    var left_keys = List[Int]()
    left_keys.append(1)
    left_keys.append(2)
    var left_vals = List[Int]()
    left_vals.append(10)
    left_vals.append(20)
    var left = _int32_struct(left_keys, left_vals)

    var right_keys = List[Int]()
    right_keys.append(3)
    right_keys.append(4)
    var right_vals = List[Int]()
    right_vals.append(30)
    right_vals.append(40)
    var right = _int32_struct(right_keys, right_vals)

    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(len(result), 0)


def test_inner_join_duplicate_keys_cartesian() raises:
    """Inner join with duplicate keys on both sides produces Cartesian product.
    """
    # left: (1,a), (1,b)
    # right: (1,x), (1,y)
    # expected 4 rows: (1,a,1,x), (1,a,1,y), (1,b,1,x), (1,b,1,y)
    var lk = List[Int]()
    lk.append(1)
    lk.append(1)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    var left = _int32_struct(lk, lv)

    var rk = List[Int]()
    rk.append(1)
    rk.append(1)
    var rv = List[Int]()
    rv.append(100)
    rv.append(200)
    var right = _int32_struct(rk, rv)

    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(len(result), 4)


def test_inner_join_empty_left() raises:
    """Inner join with empty left side returns empty batch."""
    var left_keys = List[Int]()
    var left_vals = List[Int]()
    var left = _int32_struct(left_keys, left_vals)

    var rk = List[Int]()
    rk.append(1)
    var rv = List[Int]()
    rv.append(10)
    var right = _int32_struct(rk, rv)

    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(len(result), 0)


def test_inner_join_empty_right() raises:
    """Inner join with empty right side returns empty batch."""
    var lk = List[Int]()
    lk.append(1)
    var lv = List[Int]()
    lv.append(10)
    var left = _int32_struct(lk, lv)

    var right_keys = List[Int]()
    var right_vals = List[Int]()
    var right = _int32_struct(right_keys, right_vals)

    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(len(result), 0)


# ---------------------------------------------------------------------------
# hash_join — LEFT join
# ---------------------------------------------------------------------------


def test_left_join_unmatched_left_rows() raises:
    """Left join: unmatched left rows appear with nulls for right columns."""
    # left: (1,10), (2,20), (3,30)
    # right: (2,200)
    # expected: (1,10,null,null), (2,20,2,200), (3,30,null,null)
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    lk.append(3)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    lv.append(30)
    var left = _int32_struct(lk, lv)

    var rk = List[Int]()
    rk.append(2)
    var rv = List[Int]()
    rv.append(200)
    var right = _int32_struct(rk, rv)

    var result = hash_join(left, right, _left_on(), _right_on(), kind=JOIN_LEFT)
    assert_equal(len(result), 3)
    # Right side columns have nulls for unmatched rows.
    assert_equal(result.children[2].null_count(), 2)
    assert_equal(result.children[3].null_count(), 2)


# ---------------------------------------------------------------------------
# hash_join — RIGHT join
# ---------------------------------------------------------------------------


def test_right_join_unmatched_right_rows() raises:
    """Right join: unmatched right rows appear with nulls for left columns."""
    # left: (2,20)
    # right: (1,100), (2,200), (3,300)
    # expected: (null,null,1,100), (2,20,2,200), (null,null,3,300)
    var lk = List[Int]()
    lk.append(2)
    var lv = List[Int]()
    lv.append(20)
    var left = _int32_struct(lk, lv)

    var rk = List[Int]()
    rk.append(1)
    rk.append(2)
    rk.append(3)
    var rv = List[Int]()
    rv.append(100)
    rv.append(200)
    rv.append(300)
    var right = _int32_struct(rk, rv)

    var result = hash_join(
        left, right, _left_on(), _right_on(), kind=JOIN_RIGHT
    )
    assert_equal(len(result), 3)
    # Left side columns have nulls for unmatched rows.
    assert_equal(result.children[0].null_count(), 2)
    assert_equal(result.children[1].null_count(), 2)


# ---------------------------------------------------------------------------
# hash_join — FULL OUTER join
# ---------------------------------------------------------------------------


def test_full_outer_join() raises:
    """Full outer join emits all rows from both sides, nulls for non-matches."""
    # left: (1,10), (2,20)
    # right: (2,200), (3,300)
    # expected: (1,10,null,null), (2,20,2,200), (null,null,3,300) — 3 rows
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    var left = _int32_struct(lk, lv)

    var rk = List[Int]()
    rk.append(2)
    rk.append(3)
    var rv = List[Int]()
    rv.append(200)
    rv.append(300)
    var right = _int32_struct(rk, rv)

    var result = hash_join(left, right, _left_on(), _right_on(), kind=JOIN_FULL)
    assert_equal(len(result), 3)
    assert_equal(len(result.children), 4)


# ---------------------------------------------------------------------------
# hash_join — SEMI join
# ---------------------------------------------------------------------------


def test_semi_join_basic() raises:
    """Semi join: left rows with at least one match; left columns only."""
    # left: (1,10), (2,20), (3,30)
    # right: (2,200), (2,201)
    # expected: (2,20) — only 1 row even though right has 2 matches
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    lk.append(3)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    lv.append(30)
    var left = _int32_struct(lk, lv)

    var rk = List[Int]()
    rk.append(2)
    rk.append(2)
    var rv = List[Int]()
    rv.append(200)
    rv.append(201)
    var right = _int32_struct(rk, rv)

    var result = hash_join(left, right, _left_on(), _right_on(), kind=JOIN_SEMI)
    assert_equal(len(result), 1)
    assert_equal(len(result.children), 2)  # left columns only
    ref k = result.children[0].as_primitive[Int32Type]()
    assert_equal(k[0], Scalar[int32.native](2))


# ---------------------------------------------------------------------------
# hash_join — ANTI join
# ---------------------------------------------------------------------------


def test_anti_join_basic() raises:
    """Anti join: left rows with no match; left columns only."""
    # left: (1,10), (2,20), (3,30)
    # right: (2,200)
    # expected: (1,10), (3,30)
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    lk.append(3)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    lv.append(30)
    var left = _int32_struct(lk, lv)

    var rk = List[Int]()
    rk.append(2)
    var rv = List[Int]()
    rv.append(200)
    var right = _int32_struct(rk, rv)

    var result = hash_join(left, right, _left_on(), _right_on(), kind=JOIN_ANTI)
    assert_equal(len(result), 2)
    assert_equal(len(result.children), 2)  # left columns only
    ref k = result.children[0].as_primitive[Int32Type]()
    assert_equal(k[0], Scalar[int32.native](1))
    assert_equal(k[1], Scalar[int32.native](3))


# ---------------------------------------------------------------------------
# hash_join — ANY strictness
# ---------------------------------------------------------------------------


def test_any_strictness_deduplicates() raises:
    """JOIN_ANY: at most one output row per build row, no Cartesian explosion.
    """
    # left: (1,10), (1,20)  ← two rows with key=1
    # right: (1,100), (1,200)
    # With JOIN_ALL: 4 rows (Cartesian)
    # With JOIN_ANY: at most 2 rows (one per build row)
    var lk = List[Int]()
    lk.append(1)
    lk.append(1)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    var left = _int32_struct(lk, lv)

    var rk = List[Int]()
    rk.append(1)
    rk.append(1)
    var rv = List[Int]()
    rv.append(100)
    rv.append(200)
    var right = _int32_struct(rk, rv)

    var result_all = hash_join(
        left,
        right,
        _left_on(),
        _right_on(),
        kind=JOIN_INNER,
        strictness=JOIN_ALL,
    )
    assert_equal(len(result_all), 4)

    var result_any = hash_join(
        left,
        right,
        _left_on(),
        _right_on(),
        kind=JOIN_INNER,
        strictness=JOIN_ANY,
    )
    assert_true(len(result_any) <= 2)


# ---------------------------------------------------------------------------
# hash_join — string keys
# ---------------------------------------------------------------------------


def test_inner_join_string_keys() raises:
    """Inner join works with string key columns."""
    var lb = StringBuilder(3)
    lb.append("a")
    lb.append("b")
    lb.append("c")
    var lv_b = PrimitiveBuilder[Int32Type](capacity=3)
    lv_b.append(Scalar[int32.native](1))
    lv_b.append(Scalar[int32.native](2))
    lv_b.append(Scalar[int32.native](3))
    var lcols = List[AnyArray]()
    lcols.append(lb.finish().to_any())
    lcols.append(lv_b.finish().to_any())
    var left = record_batch(lcols^, names=["k", "v"]).to_struct_array()

    var rb = StringBuilder(2)
    rb.append("b")
    rb.append("c")
    var rv_b = PrimitiveBuilder[Int32Type](capacity=2)
    rv_b.append(Scalar[int32.native](20))
    rv_b.append(Scalar[int32.native](30))
    var rcols = List[AnyArray]()
    rcols.append(rb.finish().to_any())
    rcols.append(rv_b.finish().to_any())
    var right = record_batch(rcols^, names=["k", "v"]).to_struct_array()

    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(len(result), 2)


# ---------------------------------------------------------------------------
# hash_join — multi-key join
# ---------------------------------------------------------------------------


def test_inner_join_multi_key() raises:
    """Inner join on two key columns produces correct matches."""
    # left: (a=1,b=10,v=100), (a=1,b=20,v=200), (a=2,b=10,v=300)
    # right: (a=1,b=10,v=1000), (a=2,b=30,v=2000)
    # expected: only (a=1,b=10) matches → 1 row
    var la = PrimitiveBuilder[Int32Type](capacity=3)
    la.append(Scalar[int32.native](1))
    la.append(Scalar[int32.native](1))
    la.append(Scalar[int32.native](2))
    var lb2 = PrimitiveBuilder[Int32Type](capacity=3)
    lb2.append(Scalar[int32.native](10))
    lb2.append(Scalar[int32.native](20))
    lb2.append(Scalar[int32.native](10))
    var lv2 = PrimitiveBuilder[Int32Type](capacity=3)
    lv2.append(Scalar[int32.native](100))
    lv2.append(Scalar[int32.native](200))
    lv2.append(Scalar[int32.native](300))
    var lcols = List[AnyArray]()
    lcols.append(la.finish().to_any())
    lcols.append(lb2.finish().to_any())
    lcols.append(lv2.finish().to_any())
    var left = record_batch(lcols^, names=["a", "b", "v"]).to_struct_array()

    var ra = PrimitiveBuilder[Int32Type](capacity=2)
    ra.append(Scalar[int32.native](1))
    ra.append(Scalar[int32.native](2))
    var rb2 = PrimitiveBuilder[Int32Type](capacity=2)
    rb2.append(Scalar[int32.native](10))
    rb2.append(Scalar[int32.native](30))
    var rv2 = PrimitiveBuilder[Int32Type](capacity=2)
    rv2.append(Scalar[int32.native](1000))
    rv2.append(Scalar[int32.native](2000))
    var rcols = List[AnyArray]()
    rcols.append(ra.finish().to_any())
    rcols.append(rb2.finish().to_any())
    rcols.append(rv2.finish().to_any())
    var right = record_batch(rcols^, names=["a", "b", "v"]).to_struct_array()

    var left_on = List[Int]()
    left_on.append(0)
    left_on.append(1)
    var right_on = List[Int]()
    right_on.append(0)
    right_on.append(1)

    var result = hash_join(left, right, left_on, right_on)
    assert_equal(len(result), 1)


# ---------------------------------------------------------------------------
# hash_join — output schema collision resolution
# ---------------------------------------------------------------------------


def test_output_schema_column_name_collision() raises:
    """Colliding column names get _right suffix in output schema."""
    # Both sides have columns named "k" and "v"
    var lk = List[Int]()
    lk.append(1)
    var lv = List[Int]()
    lv.append(10)
    var left = _int32_struct(lk, lv)

    var rk = List[Int]()
    rk.append(1)
    var rv = List[Int]()
    rv.append(100)
    var right = _int32_struct(rk, rv)

    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(result.dtype.fields[0].name, "k")
    assert_equal(result.dtype.fields[1].name, "v")
    assert_equal(result.dtype.fields[2].name, "k_right")
    assert_equal(result.dtype.fields[3].name, "v_right")


# ---------------------------------------------------------------------------
# hash_join — collision correctness
# ---------------------------------------------------------------------------


def _constant_hash(
    keys: StructArray,
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[UInt64Type]:
    """Degenerate hash function: all keys map to the same hash.

    Forces every key into a single bucket — without key equality checks,
    an inner join would produce N×M rows (all-pairs). With equality checks,
    only actual matching keys produce output.
    """
    var n = len(keys)
    var b = PrimitiveBuilder[UInt64Type](capacity=n)
    for _ in range(n):
        b.unsafe_append(Scalar[uint64.native](42))
    return b.finish()


def test_collision_inner_join() raises:
    """With all hashes colliding, key equality filters to correct matches."""
    from marrow.kernels.join import HashJoin

    # left: k=[1,2,3], v=[10,20,30]
    # right: k=[2,3,4], v=[100,200,300]
    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    lk.append(3)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    lv.append(30)
    var left = _int32_struct(lk, lv)

    var rk = List[Int]()
    rk.append(2)
    rk.append(3)
    rk.append(4)
    var rv = List[Int]()
    rv.append(100)
    rv.append(200)
    rv.append(300)
    var right = _int32_struct(rk, rv)

    # Use degenerate hash — all keys hash to 42.
    var join = HashJoin[_constant_hash]()
    join.build(left, _left_on())
    var result = join.probe(right, _right_on(), JOIN_INNER, JOIN_ALL)

    # Only k=2 and k=3 match → 2 result rows.
    # Without equality check this would be 3×3 = 9 (WRONG).
    assert_equal(len(result), 2)


def test_collision_left_join() raises:
    """With all hashes colliding, left join produces correct unmatched rows."""
    from marrow.kernels.join import HashJoin

    var lk = List[Int]()
    lk.append(1)
    lk.append(2)
    var lv = List[Int]()
    lv.append(10)
    lv.append(20)
    var left = _int32_struct(lk, lv)

    var rk = List[Int]()
    rk.append(2)
    rk.append(3)
    var rv = List[Int]()
    rv.append(100)
    rv.append(200)
    var right = _int32_struct(rk, rv)

    var join = HashJoin[_constant_hash]()
    join.build(left, _left_on())
    var result = join.probe(right, _right_on(), JOIN_LEFT, JOIN_ALL)

    # k=1 unmatched (left), k=2 matched → 2 result rows.
    assert_equal(len(result), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
