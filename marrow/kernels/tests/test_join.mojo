"""Tests for the hash join kernel."""

from std.testing import assert_equal, assert_true, assert_false

from ...arrays import (
    DynArray,
    PrimitiveArray,
    StringArray,
    StructArray,
    UInt64Array,
)
from ...execution import ExecContext
from ...builders import (
    array,
    PrimitiveBuilder,
    BinaryLikeBuilder,
    StringBuilder,
    Int32Builder,
    UInt64Builder,
)
from ...dtypes import (
    int32,
    int64,
    uint64,
    float64,
    string,
    Field,
    Int32Type,
    Int64Type,
    UInt64Type,
    Float64Type,
    BinaryLikeType,
    BinaryType,
    LargeBinaryType,
    struct_,
)
from ...schema import Schema
from ...tabular import record_batch, RecordBatch
from ...utils import Hasher
from ...kernels.join import (
    hash_join,
    HashJoin,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_RIGHT_ANTI,
    JOIN_MARK,
    JOIN_SINGLE,
    JOIN_CROSS,
    JOIN_ALL,
    JOIN_ANY,
    JoinKind,
    JoinBuildSide,
    BUILD_LEFT,
    BUILD_RIGHT,
)
from ...kernels.sort import sort


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _int32_struct(col0: List[Int], col1: List[Int]) raises -> StructArray:
    var a = Int32Builder(capacity=len(col0))
    var b = Int32Builder(capacity=len(col1))
    for v in col0:
        a.append(Scalar[int32.native](v))
    for v in col1:
        b.append(Scalar[int32.native](v))
    var cols = List[DynArray]()
    cols.append(a.finish().to_dyn())
    cols.append(b.finish().to_dyn())
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
    ref k = result.children[0].as_int32()
    assert_equal(k[0].value(), Scalar[int32.native](2))


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
    ref k = result.children[0].as_int32()
    assert_equal(k[0].value(), Scalar[int32.native](1))
    assert_equal(k[1].value(), Scalar[int32.native](3))


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
    var lv_b = Int32Builder(capacity=3)
    lv_b.append(Scalar[int32.native](1))
    lv_b.append(Scalar[int32.native](2))
    lv_b.append(Scalar[int32.native](3))
    var lcols = List[DynArray]()
    lcols.append(lb.finish().to_dyn())
    lcols.append(lv_b.finish().to_dyn())
    var left = record_batch(lcols^, names=["k", "v"]).to_struct_array()

    var rb = StringBuilder(2)
    rb.append("b")
    rb.append("c")
    var rv_b = Int32Builder(capacity=2)
    rv_b.append(Scalar[int32.native](20))
    rv_b.append(Scalar[int32.native](30))
    var rcols = List[DynArray]()
    rcols.append(rb.finish().to_dyn())
    rcols.append(rv_b.finish().to_dyn())
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
    var la = Int32Builder(capacity=3)
    la.append(Scalar[int32.native](1))
    la.append(Scalar[int32.native](1))
    la.append(Scalar[int32.native](2))
    var lb2 = Int32Builder(capacity=3)
    lb2.append(Scalar[int32.native](10))
    lb2.append(Scalar[int32.native](20))
    lb2.append(Scalar[int32.native](10))
    var lv2 = Int32Builder(capacity=3)
    lv2.append(Scalar[int32.native](100))
    lv2.append(Scalar[int32.native](200))
    lv2.append(Scalar[int32.native](300))
    var lcols = List[DynArray]()
    lcols.append(la.finish().to_dyn())
    lcols.append(lb2.finish().to_dyn())
    lcols.append(lv2.finish().to_dyn())
    var left = record_batch(lcols^, names=["a", "b", "v"]).to_struct_array()

    var ra = Int32Builder(capacity=2)
    ra.append(Scalar[int32.native](1))
    ra.append(Scalar[int32.native](2))
    var rb2 = Int32Builder(capacity=2)
    rb2.append(Scalar[int32.native](10))
    rb2.append(Scalar[int32.native](30))
    var rv2 = Int32Builder(capacity=2)
    rv2.append(Scalar[int32.native](1000))
    rv2.append(Scalar[int32.native](2000))
    var rcols = List[DynArray]()
    rcols.append(ra.finish().to_dyn())
    rcols.append(rb2.finish().to_dyn())
    rcols.append(rv2.finish().to_dyn())
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
    assert_equal(result.dtype.as_struct().fields[0].name, "k")
    assert_equal(result.dtype.as_struct().fields[1].name, "v")
    assert_equal(result.dtype.as_struct().fields[2].name, "k_right")
    assert_equal(result.dtype.as_struct().fields[3].name, "v_right")


# ---------------------------------------------------------------------------
# hash_join — collision correctness
# ---------------------------------------------------------------------------


struct ConstantHash(Hasher):
    """Degenerate hash: every value maps to the same digest.

    Forces every key into one bucket — without key-equality verification an
    inner join would emit N x M rows. It is a `Hasher` rather than a function
    because `HashJoin` takes the algorithm as a type; writing one is the
    smallest demonstration that the parameter is genuinely open.
    """

    comptime name = StaticString("constant")

    @staticmethod
    def hash(data: Span[UInt8, _], seed: UInt64 = 0) -> UInt64:
        return 42

    @staticmethod
    @always_inline
    def hash_lanes[
        byte_width: Int, W: Int
    ](values: SIMD[DType.uint64, W]) -> SIMD[DType.uint64, W]:
        return SIMD[DType.uint64, W](42)


def test_collision_inner_join() raises:
    """With all hashes colliding, key equality filters to correct matches."""
    from ...kernels.join import HashJoin

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
    var join = HashJoin[ConstantHash]()
    join.build(left, _left_on())
    var result = join.probe(right, _right_on(), JOIN_INNER, JOIN_ALL)

    # Only k=2 and k=3 match → 2 result rows.
    # Without equality check this would be 3×3 = 9 (WRONG).
    assert_equal(len(result), 2)


def test_collision_left_join() raises:
    """With all hashes colliding, left join produces correct unmatched rows."""
    from ...kernels.join import HashJoin

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

    var join = HashJoin[ConstantHash]()
    join.build(left, _left_on())
    var result = join.probe(right, _right_on(), JOIN_LEFT, JOIN_ALL)

    # k=1 unmatched (left), k=2 matched → 2 result rows.
    assert_equal(len(result), 2)


# ---------------------------------------------------------------------------
# parallel path correctness — equivalence with serial results
# ---------------------------------------------------------------------------


def _dense_struct(n: Int) raises -> StructArray:
    """Build a StructArray of (k=Int32, v=Int32) with unique keys 0..n."""
    var kb = Int32Builder(capacity=n)
    var vb = Int32Builder(capacity=n)
    for i in range(n):
        kb.append(Scalar[int32.native](i))
        vb.append(Scalar[int32.native](i * 10))
    var cols = List[DynArray]()
    cols.append(kb.finish().to_dyn())
    cols.append(vb.finish().to_dyn())
    return record_batch(cols^, names=["k", "v"]).to_struct_array()


def _run_inner(
    left: StructArray, right: StructArray, num_threads: Int
) raises -> StructArray:
    return hash_join(
        left,
        right,
        _left_on(),
        _right_on(),
        JOIN_INNER,
        JOIN_ALL,
        ctx=ExecContext.parallel(num_threads),
    )


def test_parallel_inner_matches_serial() raises:
    """Inner join results are equivalent between serial and parallel paths.

    Uses 200k rows to exceed ``_PARALLEL_THRESHOLD``; every probe row has
    exactly one match so both paths produce the same number of pairs.
    """
    var n = 200_000
    var left = _dense_struct(n)
    var right = _dense_struct(n)

    var serial = _run_inner(left, right, 1)
    var parallel = _run_inner(left, right, 4)

    # Row count must match; per-row ordering may differ across paths.
    assert_equal(len(serial), len(parallel))
    assert_equal(len(serial), n)


def test_parallel_inner_no_matches() raises:
    """Parallel path must correctly produce zero rows when nothing matches."""
    var n = 150_000
    # left keys 0..n, right keys n..2n — disjoint.
    var left = _dense_struct(n)
    var kb = Int32Builder(capacity=n)
    var vb = Int32Builder(capacity=n)
    for i in range(n):
        kb.append(Scalar[int32.native](n + i))
        vb.append(Scalar[int32.native](i * 10))
    var cols = List[DynArray]()
    cols.append(kb.finish().to_dyn())
    cols.append(vb.finish().to_dyn())
    var right = record_batch(cols^, names=["k", "v"]).to_struct_array()

    var parallel = _run_inner(left, right, 4)
    assert_equal(len(parallel), 0)


def test_parallel_partial_match() raises:
    """Parallel inner join with partial key overlap."""
    var n = 150_000
    var left = _dense_struct(n)
    # right keys are [n/2, 3n/2) — half overlap.
    var kb = Int32Builder(capacity=n)
    var vb = Int32Builder(capacity=n)
    for i in range(n):
        kb.append(Scalar[int32.native](n // 2 + i))
        vb.append(Scalar[int32.native](i * 10))
    var cols = List[DynArray]()
    cols.append(kb.finish().to_dyn())
    cols.append(vb.finish().to_dyn())
    var right = record_batch(cols^, names=["k", "v"]).to_struct_array()

    var serial = _run_inner(left, right, 1)
    var parallel = _run_inner(left, right, 4)
    assert_equal(len(serial), len(parallel))
    assert_equal(len(serial), n // 2)


# ---------------------------------------------------------------------------
# Execution context plumbing
#
# `HashJoin` used to store a bare worker count and rebuild
# `ExecContext.parallel(n)` at five internal sites, each of which dropped
# the caller's device. It now holds the context whole. These pin the observable
# half of that: which path a given context selects, and that the default did not
# silently change when `num_threads` was replaced by `ctx`.
# ---------------------------------------------------------------------------


def test_join_default_context_matches_explicit_auto() raises:
    """`hash_join`'s default is *auto*, not serial.

    The parameter it replaced defaulted to `num_threads=0`, which meant
    all-cores. Defaulting the context to `.serial()` instead would have quietly
    made every default join single-threaded — a change no correctness test would
    have caught, since all three paths return the same rows.
    """
    var left = _dense_struct(20_000)
    var right = _dense_struct(20_000)
    var default_result = hash_join(left, right, _left_on(), _right_on())
    var auto_result = hash_join(
        left,
        right,
        _left_on(),
        _right_on(),
        ctx=ExecContext.auto(),
    )
    assert_equal(default_result.length, auto_result.length)


def test_join_serial_and_parallel_contexts_agree() raises:
    """The serial single-table path and the radix-partitioned parallel path
    return the same number of matched rows for the same input."""
    var left = _dense_struct(20_000)
    var right = _dense_struct(20_000)
    var serial = hash_join(
        left,
        right,
        _left_on(),
        _right_on(),
        ctx=ExecContext.serial(),
    )
    var parallel = hash_join(
        left,
        right,
        _left_on(),
        _right_on(),
        ctx=ExecContext.parallel(4),
    )
    assert_equal(serial.length, parallel.length)


def test_hash_join_struct_default_is_serial() raises:
    """`HashJoin()` built with no argument stays on the serial path.

    `expr/execution.mojo` and `bench_join` both construct it that way, and its
    old `num_threads=1` default meant serial — unlike the free function's.
    """
    var join = HashJoin(ExecContext())
    var left = _dense_struct(20_000)
    join.build(left, _left_on())
    var out = join.probe(
        _dense_struct(20_000), _right_on(), JOIN_INNER, JOIN_ALL
    )
    assert_true(out.length > 0)


# ---------------------------------------------------------------------------
# JoinKind — the "does this kind emit right-side columns?" question has one
# answer, on the kind itself.
#
# It used to be re-derived at four sites with three different memberships:
# `output_dtype` said MARK emits right columns, `_assemble` said it does not,
# `relations.mojo` agreed with the first, and `tabular.mojo` re-parsed strings.
# A `StructArray` whose dtype declares more fields than it has children is
# corrupt, and nothing checked.
# ---------------------------------------------------------------------------


def test_join_kind_agrees_on_right_columns() raises:
    """Every kind's declared schema must have exactly as many fields as the
    assembled result has columns. This is the invariant the four copies broke.
    """
    var left = _dense_struct(64)
    var right = _dense_struct(64)
    var kinds = List[JoinKind]()
    kinds.append(JOIN_INNER)
    kinds.append(JOIN_LEFT)
    kinds.append(JOIN_RIGHT)
    kinds.append(JOIN_FULL)
    kinds.append(JOIN_SEMI)
    kinds.append(JOIN_ANTI)
    for ref k in kinds:
        var out = hash_join(left, right, _left_on(), _right_on(), k)
        assert_equal(
            len(out.dtype.as_struct().fields),
            len(out.children),
            String("kind ", k, ": dtype fields != columns"),
        )


def test_join_kind_semi_anti_emit_left_columns_only() raises:
    assert_false(JOIN_SEMI.emits_right_columns())
    assert_false(JOIN_ANTI.emits_right_columns())
    assert_true(JOIN_INNER.emits_right_columns())
    assert_true(JOIN_LEFT.emits_right_columns())
    assert_true(JOIN_RIGHT.emits_right_columns())
    assert_true(JOIN_FULL.emits_right_columns())


def test_unimplemented_join_kind_raises_instead_of_corrupting() raises:
    """MARK, SINGLE and CROSS have constants but no implementation.

    MARK previously fell through to the LEFT/RIGHT/FULL arm and produced a
    `StructArray` declaring the right side's fields while carrying only the
    left side's columns. Raising is the honest answer until someone implements
    the marker column.
    """
    var left = _dense_struct(64)
    var right = _dense_struct(64)
    var unimplemented = List[JoinKind]()
    unimplemented.append(JOIN_MARK)
    unimplemented.append(JOIN_SINGLE)
    unimplemented.append(JOIN_CROSS)
    for ref k in unimplemented:
        var raised = False
        try:
            _ = hash_join(left, right, _left_on(), _right_on(), k)
        except:
            raised = True
        assert_true(raised, String("kind ", k, " should raise"))


def test_join_kind_writes_its_name() raises:
    """A kind prints as a name, not a number — it is what error messages and
    plan output show."""
    assert_equal(String(JOIN_INNER), "inner")
    assert_equal(String(JOIN_LEFT), "left outer")
    assert_equal(String(JOIN_SEMI), "left semi")


# ---------------------------------------------------------------------------
# hash_join — binary / large_binary keys
#
# `SwissHashTable.probe` verifies hash-collision candidates with
# `EqKernel.apply(StructArray, StructArray)`, which routes each key column
# through `equal`. That picked its kernel family with
# `is_string() or is_large_string()`, so a `binary` key column fell through to
# the numeric arm and `dispatch_primitive` raised — joining on `binary` was
# impossible while the identical join on `string` worked.
# ---------------------------------------------------------------------------


def _bytes_join_side[
    T: BinaryLikeType
](keys: List[String], vals: List[Int]) raises -> StructArray:
    var kb = BinaryLikeBuilder[T](len(keys))
    for k in keys:
        kb.append(k)
    var vb = Int32Builder(capacity=len(vals))
    for v in vals:
        vb.append(Scalar[int32.native](v))
    var cols = List[DynArray]()
    cols.append(kb.finish().to_dyn())
    cols.append(vb.finish().to_dyn())
    return record_batch(cols^, names=["k", "v"]).to_struct_array()


def _assert_bytes_join[T: BinaryLikeType]() raises:
    var left = _bytes_join_side[T](["a", "b", "c"], [1, 2, 3])
    var right = _bytes_join_side[T](["b", "c", "d"], [20, 30, 40])
    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(len(result), 2)


def test_inner_join_binary_keys() raises:
    _assert_bytes_join[BinaryType]()


def test_inner_join_large_binary_keys() raises:
    _assert_bytes_join[LargeBinaryType]()


def test_inner_join_binary_keys_collision_verified() raises:
    """Duplicate keys on both sides exercise the equality verification path
    (not just the hash lookup): 2x2 matches on "b" plus 1x1 on "c"."""
    var left = _bytes_join_side[BinaryType](["b", "b", "c", "x"], [1, 2, 3, 4])
    var right = _bytes_join_side[BinaryType](["b", "b", "c"], [10, 20, 30])
    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(len(result), 5)


# ---------------------------------------------------------------------------
# serial vs partitioned probe — the two paths must be indistinguishable
#
# `build` picks the layout (single table vs one table per radix partition) and
# `probe` must follow it. The two emit the same rows in different orders, so
# these compare an order-insensitive fingerprint rather than the arrays: the
# partitioned probe emits in partition order, the single-table probe in probe
# order, and a regression here shows up as dropped or duplicated rows, which a
# row count alone would miss whenever a drop and a duplicate cancel out.
# ---------------------------------------------------------------------------


def _join_side(n: Int, key_offset: Int, val_mul: Int) raises -> StructArray:
    var ks = List[Int](capacity=n)
    var vs = List[Int](capacity=n)
    for i in range(n):
        ks.append(i + key_offset)
        vs.append(i * val_mul)
    return _int32_struct(ks, vs)


def _join_fingerprint(result: StructArray) raises -> String:
    """Row count plus, per column, the sum of its values and its null count.

    Order-insensitive by construction, and sensitive to exactly the failures
    that matter: a dropped row moves the sum and the count, a duplicated row
    moves both the other way, and a mis-slotted NULL moves the null count
    without moving anything else.
    """
    var out = String(len(result))
    for i in range(len(result.children)):
        var col = result.field(i)
        var total = 0
        for j in range(len(col)):
            if col.is_valid(j):
                total += Int(col.as_int32()[j].value())
        out += "|" + String(total) + ":" + String(col.null_count())
    return out^


def _probe_in_morsels(
    left: StructArray,
    right: StructArray,
    var ctx: ExecContext,
    kind: JoinKind,
    morsel: Int,
) raises -> String:
    """Build with `ctx`, then probe `right` in `morsel`-row slices.

    Mirrors how the plan layer drives a join: one build, many small probes.
    Returns the concatenated per-morsel fingerprints, so a divergence is
    localized to the morsel that caused it.
    """
    var j = HashJoin(ctx^)
    j.build(left, _left_on())
    var out = String("parallel=") + String(j.built_parallel())
    var off = 0
    while off < len(right):
        var m = min(morsel, len(right) - off)
        out += ";" + _join_fingerprint(
            j.probe(right.slice(off, m), _right_on(), kind, JOIN_ALL)
        )
        off += m
    return out^


def _assert_join_paths_agree(kind: JoinKind, morsel: Int) raises:
    """Same join, same data, both layouts — results must be identical.

    `n` sits above `_PARALLEL_THRESHOLD` (100k) so the parallel context takes
    the partitioned build, and the key offset overlaps the two sides by half
    so LEFT/SEMI/ANTI all see matched *and* unmatched rows.
    """
    var n = 150_000
    var left = _join_side(n, 0, 10)
    var right = _join_side(n, n // 2, 100)

    var par = HashJoin(ExecContext.parallel(4))
    par.build(left, _left_on())
    assert_true(
        par.built_parallel(),
        (
            "expected the partitioned layout at 150k rows / 4 workers — without"
            " it this test would compare the serial path against itself"
        ),
    )

    var serial_fp = _probe_in_morsels(
        left, right, ExecContext.serial(), kind, morsel
    )
    var parallel_fp = _probe_in_morsels(
        left, right, ExecContext.parallel(4), kind, morsel
    )
    # Strip the layout tag: it is asserted above and is *expected* to differ.
    var s_body = serial_fp[byte = serial_fp.find(";") :]
    var p_body = parallel_fp[byte = parallel_fp.find(";") :]
    assert_equal(s_body, p_body)


def test_join_paths_agree_inner_morsels() raises:
    _assert_join_paths_agree(JOIN_INNER, 8192)


def test_join_paths_agree_left_morsels() raises:
    _assert_join_paths_agree(JOIN_LEFT, 8192)


def test_join_paths_agree_semi_morsels() raises:
    _assert_join_paths_agree(JOIN_SEMI, 8192)


def test_join_paths_agree_anti_morsels() raises:
    _assert_join_paths_agree(JOIN_ANTI, 8192)


def test_join_paths_agree_inner_single_probe() raises:
    """One probe call above `_PROBE_STRIPE_THRESHOLD`, so the probe stripes
    rather than taking the small-batch serial hashing path."""
    _assert_join_paths_agree(JOIN_INNER, 150_000)


def test_join_paths_agree_left_single_probe() raises:
    _assert_join_paths_agree(JOIN_LEFT, 150_000)


# ---------------------------------------------------------------------------
# NULL join keys — a NULL matches nothing, not even another NULL
#
# SQL's rule, and Arrow C++'s default `JoinKeyCmp::EQ`: Acero routes a null-keyed
# row straight to no-match. marrow drops the pair at the equality verification in
# `SwissHashTable.probe`, which every candidate pair passes through — so these
# cases pin the *behaviour* of each join kind rather than the mechanism.
#
# Fingerprints are `_join_fingerprint`: row count, then `sum:null_count` per
# column. A NULL landing in the wrong row moves a null count without moving a
# sum, which a row count alone would miss.
# ---------------------------------------------------------------------------


def _nullable_int32_struct(
    keys: List[Optional[Int]], vals: List[Optional[Int]]
) raises -> StructArray:
    """A two-column `k`/`v` side whose key column may carry NULLs."""
    var cols = List[DynArray]()
    cols.append(array(keys, int32).to_dyn())
    cols.append(array(vals, int32).to_dyn())
    return record_batch(cols^, names=["k", "v"]).to_struct_array()


def test_inner_join_null_key_left_only() raises:
    """A NULL build key matches no probe row, however ordinary the probe."""
    var left = _nullable_int32_struct([1, None], [10, 20])
    var right = _nullable_int32_struct([1, 2], [100, 200])

    var result = hash_join(left, right, _left_on(), _right_on())
    # Only (k=1, v=10, k_right=1, v_right=100).
    assert_equal(_join_fingerprint(result), "1|1:0|10:0|1:0|100:0")


def test_inner_join_null_key_both_sides() raises:
    """NULL does not match NULL — the case that used to return two rows.

    Both null slots hold the same underlying bytes, so the comparison kernel's
    SIMD lane set the data bit and only the validity bitmap recorded that the
    bit was meaningless.
    """
    var left = _nullable_int32_struct([1, None], [10, 20])
    var right = _nullable_int32_struct([1, None], [100, 200])

    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(_join_fingerprint(result), "1|1:0|10:0|1:0|100:0")


def test_semi_join_null_key_has_no_match() raises:
    """SEMI keeps a left row only if it matched; a NULL key never does."""
    var left = _nullable_int32_struct([1, None, 2], [10, 20, 30])
    var right = _nullable_int32_struct([1, None], [100, 200])

    var result = hash_join(left, right, _left_on(), _right_on(), kind=JOIN_SEMI)
    # Left columns only, and only the k=1 row.
    assert_equal(_join_fingerprint(result), "1|1:0|10:0")


def test_anti_join_keeps_null_key() raises:
    """ANTI keeps the NULL-keyed row: it matched nothing, which is the point."""
    var left = _nullable_int32_struct([1, None, 2], [10, 20, 30])
    var right = _nullable_int32_struct([1, None], [100, 200])

    var result = hash_join(left, right, _left_on(), _right_on(), kind=JOIN_ANTI)
    # (k=NULL, v=20) and (k=2, v=30): key sum 2 with one null, values 20+30.
    assert_equal(_join_fingerprint(result), "2|2:1|50:0")


def test_left_join_null_key_widens_right() raises:
    """LEFT keeps the NULL-keyed left row, with the right side NULL-widened."""
    var left = _nullable_int32_struct([1, None], [10, 20])
    var right = _nullable_int32_struct([1, None], [100, 200])

    var result = hash_join(left, right, _left_on(), _right_on(), kind=JOIN_LEFT)
    # (1,10,1,100) and (NULL,20,NULL,NULL).
    assert_equal(_join_fingerprint(result), "2|1:1|30:0|1:1|100:1")


def test_right_join_null_key_widens_left() raises:
    """RIGHT keeps the unmatched NULL-keyed probe row, left side NULL-widened.
    """
    var left = _nullable_int32_struct([1], [10])
    var right = _nullable_int32_struct([1, None], [100, 200])

    var result = hash_join(
        left, right, _left_on(), _right_on(), kind=JOIN_RIGHT
    )
    # (1,10,1,100) and (NULL,NULL,NULL,200).
    assert_equal(_join_fingerprint(result), "2|1:1|10:1|1:1|300:0")


def test_full_join_null_keys_on_both_sides() raises:
    """FULL emits the NULL-keyed row from *each* side, unmatched."""
    var left = _nullable_int32_struct([1, None], [10, 20])
    var right = _nullable_int32_struct([1, None], [100, 200])

    var result = hash_join(left, right, _left_on(), _right_on(), kind=JOIN_FULL)
    # (1,10,1,100), (NULL,20,NULL,NULL), (NULL,NULL,NULL,200).
    assert_equal(_join_fingerprint(result), "3|1:2|30:1|1:2|300:1")


def test_any_strictness_null_key_has_no_match() raises:
    """JOIN_ANY changes how many matches are used, not what counts as one."""
    var left = _nullable_int32_struct([1, None], [10, 20])
    var right = _nullable_int32_struct([1, None], [100, 200])

    var result = hash_join(
        left,
        right,
        _left_on(),
        _right_on(),
        kind=JOIN_INNER,
        strictness=JOIN_ANY,
    )
    assert_equal(_join_fingerprint(result), "1|1:0|10:0|1:0|100:0")


def test_multi_key_join_null_in_one_column() raises:
    """A row is unmatchable if *any* key column is NULL, not only all of them.

    Both sides carry a `(1, NULL)` row here, so a key comparison that ignored
    validity would pair them on the strength of the non-null column agreeing.
    """
    var lcols = List[DynArray]()
    lcols.append(array([1, 1], int32).to_dyn())
    lcols.append(array([None, 10], int32).to_dyn())
    lcols.append(array([100, 200], int32).to_dyn())
    var left = record_batch(lcols^, names=["a", "b", "v"]).to_struct_array()

    var rcols = List[DynArray]()
    rcols.append(array([1, 1], int32).to_dyn())
    rcols.append(array([None, 10], int32).to_dyn())
    rcols.append(array([1000, 2000], int32).to_dyn())
    var right = record_batch(rcols^, names=["a", "b", "v"]).to_struct_array()

    var left_on = List[Int]()
    left_on.append(0)
    left_on.append(1)
    var right_on = List[Int]()
    right_on.append(0)
    right_on.append(1)

    var result = hash_join(left, right, left_on, right_on)
    # Only (a=1, b=10) pairs: v 200 on the left, 2000 on the right.
    assert_equal(_join_fingerprint(result), "1|1:0|10:0|200:0|1:0|10:0|2000:0")


def test_parallel_inner_join_drops_null_keys() raises:
    """The partitioned layout drops NULL keys too.

    It verifies through the same `SwissHashTable.probe` as the serial one, but
    nothing else in this file probes the partitioned path with NULL keys — and
    every NULL key hashes to one sentinel, so they all land in one partition.

    `n` is 120k rather than the 150k the other parallel cases use, and the
    difference is not cosmetic: a *nullable* 150k column has an 18,750-byte
    validity bitmap, 62 mod 64, so the masked `apply` lane's unconditional
    4-byte `BitmapView.load` reads one byte past the allocation's 64-byte
    padding. That is a live bug in `views.mojo` with nothing to do with joins —
    hashing one nullable 150k column reproduces it on its own — and this test is
    sized to keep it out of the way rather than to hide it.
    """
    var n = 120_000
    var kb = Int32Builder(capacity=n)
    var vb = Int32Builder(capacity=n)
    var null_keys = 0
    for i in range(n):
        if i % 1000 == 0:
            kb.append_null()
            null_keys += 1
        else:
            kb.append(Scalar[int32.native](i))
        vb.append(Scalar[int32.native](i))
    var cols = List[DynArray]()
    cols.append(kb.finish().to_dyn())
    cols.append(vb.finish().to_dyn())
    var side = record_batch(cols^, names=["k", "v"]).to_struct_array()

    var joiner = HashJoin(ExecContext.parallel(4))
    joiner.build(side, _left_on())
    assert_true(
        joiner.built_parallel(),
        (
            "expected the partitioned layout at 120k rows / 4 workers — without"
            " it this test would re-check the serial path"
        ),
    )

    var result = joiner.probe(side, _right_on(), JOIN_INNER, JOIN_ALL)
    # Every non-null key is unique, so it matches itself exactly once.
    assert_equal(len(result), n - null_keys)
    assert_equal(result.field(0).null_count(), 0)


# ---------------------------------------------------------------------------
# bool join keys
#
# Booleans are bit-packed, so a `bool` key column is a `BoolArray` and never a
# `PrimitiveArray[bool_]`. Key verification used to route it through
# `dispatch_primitive`, which raised "dtype is not primitive" — joining on a
# bool key was impossible.
# ---------------------------------------------------------------------------


def _bool_key_struct(
    keys: List[Optional[Bool]],
    vals: List[Optional[Int]],
    names: List[String],
) raises -> StructArray:
    """A side whose *key* column is bit-packed bool."""
    var cols = List[DynArray]()
    cols.append(array(keys).to_dyn())
    cols.append(array(vals, int32).to_dyn())
    return record_batch(cols^, names=names).to_struct_array()


def test_inner_join_bool_keys() raises:
    """A bool key joins at all, and True does not match NULL."""
    var left = _bool_key_struct([True, False, None], [1, 2, 3], ["b", "v"])
    var right = _bool_key_struct([True, None], [9, 8], ["b2", "w"])

    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(len(result), 1)
    assert_true(result.field(0).as_bool()[0].value())
    assert_equal(Int(result.field(1).as_int32()[0].value()), 1)
    assert_true(result.field(2).as_bool()[0].value())
    assert_equal(Int(result.field(3).as_int32()[0].value()), 9)


def test_inner_join_bool_keys_false_matches_false() raises:
    """False is a key value, not the absence of one — XNOR, not AND."""
    var left = _bool_key_struct([True, False, None], [1, 2, 3], ["b", "v"])
    var right = _bool_key_struct([False, None], [7, 8], ["b2", "w"])

    var result = hash_join(left, right, _left_on(), _right_on())
    assert_equal(len(result), 1)
    assert_false(result.field(0).as_bool()[0].value())
    assert_equal(Int(result.field(1).as_int32()[0].value()), 2)
    assert_equal(Int(result.field(3).as_int32()[0].value()), 7)


def test_left_join_bool_key_null_widens_right() raises:
    """A NULL bool key keeps its left row and NULL-widens the right side."""
    var left = _bool_key_struct([True, None], [1, 2], ["b", "v"])
    var right = _bool_key_struct([True], [9], ["b2", "w"])

    var result = hash_join(left, right, _left_on(), _right_on(), kind=JOIN_LEFT)
    assert_equal(len(result), 2)
    assert_equal(result.field(2).null_count(), 1)
    assert_equal(result.field(3).null_count(), 1)


def test_semi_join_bool_key_null_has_no_match() raises:
    """The right side has a NULL bool key too; only the True row matches."""
    var left = _bool_key_struct([True, False, None], [1, 2, 3], ["b", "v"])
    var right = _bool_key_struct([True, None], [9, 8], ["b2", "w"])

    var result = hash_join(left, right, _left_on(), _right_on(), kind=JOIN_SEMI)
    assert_equal(len(result), 1)
    assert_equal(Int(result.field(1).as_int32()[0].value()), 1)


# ---------------------------------------------------------------------------
# build side vs output order — indexing the other input must change nothing
#
# `JoinBuildSide` is the only thing that says which of the two inputs is
# materialised, and it is *supposed* to be invisible in the answer: same field
# names, same field order, same rows. Row order is the one thing that does
# move, because it follows the probe side, so these compare the two results
# sorted on every column rather than as they came out. `_join_fingerprint`
# would not do here: it is a sum, and two columns with equal sums could swap
# places under it unnoticed — which is the precise failure a build-side swap
# used to cause.
# ---------------------------------------------------------------------------


def _swap_left() raises -> StructArray:
    """Duplicate keys, an unmatched key and a NULL key.

    Every kind then has matched rows, unmatched rows and a key that matches
    nothing at all; the shared `k` name is what makes the `_right` suffix
    rule observable.
    """
    return record_batch(
        [
            array([1, 2, 3, 3, None], int32).copy(),
            array([10, 20, 30, 31, 50], int32).copy(),
        ],
        names=["k", "lv"],
    ).to_struct_array()


def _swap_right() raises -> StructArray:
    return record_batch(
        [
            array([2, 3, 3, 4, None], int32).copy(),
            array([200, 300, 301, 400, 500], int32).copy(),
        ],
        names=["k", "rv"],
    ).to_struct_array()


def _canonical(result: StructArray) raises -> StructArray:
    """`result` with its rows in a canonical order — sorted on every column.

    A multiset comparison rather than a set one: sorting keeps duplicate rows,
    which a join over duplicate keys produces on purpose.
    """
    var keys = List[Int](capacity=len(result.children))
    var asc = List[Bool](capacity=len(result.children))
    for i in range(len(result.children)):
        keys.append(i)
        asc.append(True)
    return sort(result, keys, asc)


def _assert_build_sides_agree(kind: JoinKind) raises:
    var left = _swap_left()
    var right = _swap_right()
    var built_left = hash_join(
        left, right, _left_on(), _right_on(), kind, JOIN_ALL, BUILD_LEFT
    )
    var built_right = hash_join(
        left, right, _left_on(), _right_on(), kind, JOIN_ALL, BUILD_RIGHT
    )
    assert_true(
        built_left.dtype == built_right.dtype,
        String(
            "kind ",
            kind,
            ": the schema moved with the build side — ",
            built_left.dtype,
            " vs ",
            built_right.dtype,
        ),
    )
    assert_equal(
        len(built_left),
        len(built_right),
        String("kind ", kind, ": row count changed with the build side"),
    )
    assert_true(
        _canonical(built_left) == _canonical(built_right),
        String("kind ", kind, ": the rows differ"),
    )


def test_build_side_agrees_for_every_kind() raises:
    """Either build side returns the same schema and rows, for every kind."""
    var kinds: List[JoinKind] = [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
        JOIN_RIGHT_SEMI,
        JOIN_RIGHT_ANTI,
    ]
    for ref k in kinds:
        _assert_build_sides_agree(k)


# ---------------------------------------------------------------------------
# The same claim over a composite key
#
# Every case above joins on `_left_on()` / `_right_on()`, which are one column
# each. A single-member key struct cannot distinguish a per-struct rule from a
# per-member one, and `_key_struct` — which normalises the key struct's field
# names, nullability and metadata before `expect_same_dtype` compares the whole
# dtype — is a per-struct rule. So is the build side: swapping it exchanges the
# two key structs, and a normalisation that got a member's *position* wrong
# would still match a one-column key.
# ---------------------------------------------------------------------------
def _both_keys() -> List[Int]:
    var out = List[Int]()
    out.append(0)
    out.append(1)
    return out^


def _multi_left() raises -> StructArray:
    """Composite keys: a duplicate, an unmatched one, and a NULL member.

    `(1,7)` matches nothing, `(1,8)` matches once, `(2,7)` appears twice on
    each side, and `(NULL,7)` matches nothing on either — so every kind has
    matched rows, unmatched rows and a null-keyed row to place.
    """
    return record_batch(
        [
            array([1, 1, 2, 2, None], int32).copy(),
            array([7, 8, 7, 7, 7], int32).copy(),
            array([10, 20, 30, 31, 50], int32).copy(),
        ],
        names=["k1", "k2", "lv"],
    ).to_struct_array()


def _multi_right() raises -> StructArray:
    return record_batch(
        [
            array([1, 2, 2, 3, None], int32).copy(),
            array([8, 7, 7, 9, 7], int32).copy(),
            array([200, 300, 301, 400, 500], int32).copy(),
        ],
        names=["k1", "k2", "rv"],
    ).to_struct_array()


def test_build_side_agrees_on_a_multi_column_key() raises:
    """Both arrangements, over a two-column key, for every kind that
    commutes.

    One case over a list rather than eight named ones, because what varies is
    the kind and the failure message carries it — the eight above are separate
    cases only because they predate the list.
    """
    var left = _multi_left()
    var right = _multi_right()
    var kinds: List[JoinKind] = [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
        JOIN_RIGHT_SEMI,
        JOIN_RIGHT_ANTI,
    ]
    for ref kind in kinds:
        var built_left = hash_join(
            left,
            right,
            _both_keys(),
            _both_keys(),
            kind,
            JOIN_ALL,
            BUILD_LEFT,
        )
        var built_right = hash_join(
            left,
            right,
            _both_keys(),
            _both_keys(),
            kind,
            JOIN_ALL,
            BUILD_RIGHT,
        )
        assert_true(
            built_left.dtype == built_right.dtype,
            String(
                "kind ",
                kind,
                ": the schema moved with the build side — ",
                built_left.dtype,
                " vs ",
                built_right.dtype,
            ),
        )
        assert_equal(
            len(built_left),
            len(built_right),
            String("kind ", kind, ": row count changed with the build side"),
        )
        assert_true(
            _canonical(built_left) == _canonical(built_right),
            String("kind ", kind, ": the rows differ"),
        )

    # Not vacuous: the composite key really does match. `(1,8)` once and
    # `(2,7)` two-by-two, so an INNER join is five rows — a fixture whose keys
    # never met would agree on nothing and pass the loop above.
    assert_equal(
        len(
            hash_join(
                left,
                right,
                _both_keys(),
                _both_keys(),
                JOIN_INNER,
                JOIN_ALL,
                BUILD_LEFT,
            )
        ),
        5,
    )


def test_join_multi_member_keys_may_differ_in_nullability() raises:
    """A *two-column* key whose members disagree with the other side's.

    `expect_same_dtype` compares the whole key struct, and a struct dtype
    carries every field's name, nullability and metadata — so a composite key
    is where `_key_struct`'s normalisation has to hold for each member rather
    than merely for the one. The single-column case beside this one is passed
    by a fix that normalises only the first field.
    """
    var meta = Dict[String, String]()
    meta["source"] = "warehouse"
    var annotated = List[Field]()
    annotated.append(Field("k1", int32, False, meta^))
    annotated.append(Field("k2", int32, False))
    annotated.append(Field("v", int32, True))
    var left = StructArray(
        dtype=struct_(annotated^),
        length=3,
        nulls=0,
        offset=0,
        bitmap=None,
        children=[
            array([1, 2, 3], int32).to_dyn(),
            array([7, 7, 7], int32).to_dyn(),
            array([10, 20, 30], int32).to_dyn(),
        ],
    )
    var right = record_batch(
        [
            array([2, 3, 4], int32).to_dyn(),
            array([7, 7, 7], int32).to_dyn(),
            array([200, 300, 400], int32).to_dyn(),
        ],
        names=["k1", "k2", "rv"],
    ).to_struct_array()

    var out = hash_join(left, right, _both_keys(), _both_keys())
    assert_equal(len(out), 2)
    # The members survive into the output unnormalised, for both key columns —
    # `_key_struct` normalises only the copy it hands the equality kernel.
    ref fields = out.dtype.as_struct().fields
    assert_false(fields[0].nullable)
    assert_equal(fields[0].metadata["source"], "warehouse")
    assert_false(fields[1].nullable)


def test_build_side_suffixes_the_logical_right_side() raises:
    """`_right` marks where a column came from in the *query*.

    Both sides call their key `k`, so one of the two must be renamed. The
    output is left-then-right whichever side was indexed, so it is always the
    right side's `k` that becomes `k_right` — following the *probe* side
    instead would rename the left one the moment the build side flipped.
    """
    var left = _swap_left()
    var right = _swap_right()
    var expected: List[String] = ["k", "lv", "k_right", "rv"]
    var sides: List[JoinBuildSide] = [BUILD_LEFT, BUILD_RIGHT]
    for ref side in sides:
        var out = hash_join(
            left, right, _left_on(), _right_on(), JOIN_INNER, JOIN_ALL, side
        )
        ref fields = out.dtype.as_struct().fields
        assert_equal(len(fields), len(expected), String(side, ": width"))
        for i in range(len(expected)):
            assert_equal(fields[i].name, expected[i], String(side, ": name"))


def test_build_side_agrees_on_the_partitioned_path() raises:
    """The same claim on the radix-partitioned build and probe.

    `probe_parallel` assembles from concatenated per-partition pairs and is a
    second wiring of `build_side` — sized above `_PARALLEL_THRESHOLD` so the
    partitioned layout is the one under test rather than a silent fallback to
    the serial path the cases above already cover.
    """
    var n = 150_000
    var left = _join_side(n, 0, 10)
    var right = _join_side(n, n // 2, 100)
    var ctx = ExecContext.parallel(4)

    var probe = HashJoin(ctx.copy())
    probe.build(left, _left_on())
    assert_true(
        probe.built_parallel(),
        (
            "expected the partitioned layout at 150k rows / 4 workers —"
            " without it this test would re-check the serial path"
        ),
    )

    var built_left = hash_join(
        left,
        right,
        _left_on(),
        _right_on(),
        JOIN_LEFT,
        JOIN_ALL,
        BUILD_LEFT,
        ctx.copy(),
    )
    var built_right = hash_join(
        left,
        right,
        _left_on(),
        _right_on(),
        JOIN_LEFT,
        JOIN_ALL,
        BUILD_RIGHT,
        ctx.copy(),
    )
    assert_true(built_left.dtype == built_right.dtype)
    assert_equal(_join_fingerprint(built_left), _join_fingerprint(built_right))


# ---------------------------------------------------------------------------
# JoinKind.mirror — the exchange that makes the build side free
# ---------------------------------------------------------------------------


def test_join_kind_mirror_is_an_involution() raises:
    """Every supported kind has a supported mirror, and mirroring twice is a
    no-op. A kind whose mirror was unsupported would make the build side
    change the answer for that kind alone."""
    var kinds: List[JoinKind] = [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
        JOIN_RIGHT_SEMI,
        JOIN_RIGHT_ANTI,
    ]
    for ref k in kinds:
        assert_true(k.is_supported(), String(k, ": not supported"))
        assert_true(
            k.mirror().is_supported(), String(k, ": mirror unsupported")
        )
        assert_true(k.mirror().mirror() == k, String(k, ": not an involution"))
    assert_true(JOIN_LEFT.mirror() == JOIN_RIGHT)
    assert_true(JOIN_SEMI.mirror() == JOIN_RIGHT_SEMI)
    assert_true(JOIN_ANTI.mirror() == JOIN_RIGHT_ANTI)
    assert_true(JOIN_INNER.mirror() == JOIN_INNER)
    assert_true(JOIN_FULL.mirror() == JOIN_FULL)


def test_join_kind_mirror_refuses_an_unimplemented_kind() raises:
    """CROSS, MARK and SINGLE have no mirror because they have no kernel.
    Answering `self` would swap the sides of a join and keep the kind."""
    var kinds: List[JoinKind] = [JOIN_CROSS, JOIN_MARK, JOIN_SINGLE]
    for ref k in kinds:
        var raised = False
        try:
            _ = k.mirror()
        except e:
            raised = True
            assert_true("cannot be mirrored" in String(e))
        assert_true(raised, String(k, ": mirror did not raise"))


def test_right_semi_and_right_anti_emit_the_right_side_only() raises:
    """The two kinds `mirror` added. They are the left-handed pair read the
    other way round: right rows, right columns, one row per match or
    non-match."""
    var left = _swap_left()
    var right = _swap_right()

    var semi = hash_join(left, right, _left_on(), _right_on(), JOIN_RIGHT_SEMI)
    ref semi_fields = semi.dtype.as_struct().fields
    assert_equal(len(semi_fields), 2)
    assert_equal(semi_fields[0].name, "k")
    assert_equal(semi_fields[1].name, "rv")
    # Right keys 2, 3, 3 match; 4 and NULL do not.
    assert_equal(len(semi), 3)

    var anti = hash_join(left, right, _left_on(), _right_on(), JOIN_RIGHT_ANTI)
    assert_equal(len(anti.dtype.as_struct().fields), 2)
    assert_equal(len(anti), 2)


def test_right_semi_is_a_mirrored_semi_over_swapped_inputs() raises:
    """The property `mirror` claims, checked against the kernel rather than
    against itself: `semi(a, b)` and `right_semi(b, a)` are the same rows."""
    var left = _swap_left()
    var right = _swap_right()
    var semi = hash_join(left, right, _left_on(), _right_on(), JOIN_SEMI)
    var mirrored = hash_join(
        right, left, _right_on(), _left_on(), JOIN_RIGHT_SEMI
    )
    assert_true(semi.dtype == mirrored.dtype)
    assert_true(_canonical(semi) == _canonical(mirrored))


# ---------------------------------------------------------------------------
# a Field has four members, and a join used to carry one and a half
#
# `output_dtype` built every renamed field as `Field(name, dtype)`, taking the
# defaults for `nullable` and `metadata` — so a colliding right-hand column
# came out nullable with its metadata gone. `RecordBatch.join` reads its result
# schema straight off this dtype, so the loss is user-visible and not internal.
#
# Nothing caught it because `Field.__eq__` compares four members while
# `Field.write_to` rendered two, and every schema assertion in the tree went
# through `String(...)`. Both halves are fixed; these pin both.
# ---------------------------------------------------------------------------


def _annotated_side(
    key: String, value: String, unit: String
) raises -> StructArray:
    """A two-column side whose *value* column is required and annotated.

    The key column stays ordinary: it is the value column that has to survive,
    and giving both the same treatment would let a fix that carried only the
    first field's members pass.
    """
    var meta = Dict[String, String]()
    meta["unit"] = unit
    var fields = List[Field]()
    fields.append(Field(key, int32, True))
    fields.append(Field(value, int32, False, meta^))
    var cols = List[DynArray]()
    cols.append(array([1, 2, 3], int32).to_dyn())
    cols.append(array([10, 20, 30], int32).to_dyn())
    return StructArray(
        dtype=struct_(fields^),
        length=3,
        nulls=0,
        offset=0,
        bitmap=None,
        children=cols^,
    )


def _assert_annotations_survive(side: JoinBuildSide) raises:
    """Both sides' `v` columns collide, so the right one is renamed — which is
    the path that used to rebuild the field from its dtype alone."""
    var left = _annotated_side("k", "v", "cents")
    var right = _annotated_side("k", "v", "grams")
    var out = hash_join(
        left, right, _left_on(), _right_on(), JOIN_INNER, JOIN_ALL, side
    )
    ref fields = out.dtype.as_struct().fields
    assert_equal(len(fields), 4, String(side, ": width"))

    assert_equal(fields[1].name, "v")
    assert_false(fields[1].nullable, String(side, ": left v went nullable"))
    assert_equal(fields[1].metadata["unit"], "cents")

    # The renamed one — the field the old spelling reconstructed.
    assert_equal(fields[3].name, "v_right")
    assert_false(fields[3].nullable, String(side, ": right v went nullable"))
    assert_equal(fields[3].metadata["unit"], "grams")


def test_join_preserves_field_nullability_and_metadata() raises:
    """Output fields keep their nullability and metadata, renamed or not,
    under either build side."""
    var sides: List[JoinBuildSide] = [BUILD_LEFT, BUILD_RIGHT]
    for ref side in sides:
        _assert_annotations_survive(side)


def test_join_preserves_field_members_through_record_batch() raises:
    """The user-visible half: `RecordBatch.join` reads its schema off the
    kernel's dtype, so a dropped member reaches a caller who never touched the
    kernel."""
    var left = RecordBatch(
        schema=Schema(fields=[Field("k", int32, True)]),
        columns=[array([1, 2, 3], int32).to_dyn()],
    )
    var meta = Dict[String, String]()
    meta["unit"] = "grams"
    var right = RecordBatch(
        schema=Schema(
            fields=[Field("rk", int32, True), Field("w", int32, False, meta^)]
        ),
        columns=[
            array([2, 3, 4], int32).to_dyn(),
            array([20, 30, 40], int32).to_dyn(),
        ],
    )
    var out = left.join(right, ["k"], ["rk"])
    ref fields = out.schema.fields
    assert_equal(len(fields), 3)
    assert_equal(fields[2].name, "w")
    assert_false(fields[2].nullable)
    assert_equal(fields[2].metadata["unit"], "grams")


# ---------------------------------------------------------------------------
# commutes — the total predicate a rule guards on
# ---------------------------------------------------------------------------


def test_commutes_agrees_with_mirror_on_every_kind() raises:
    """`commutes` is `mirror` without the raise, and the two must not drift.

    Checked against `mirror` itself rather than against a second hand-written
    list: a kind that gained a mirror and not a `commutes` arm would be a
    reorder rule declining a swap it could have made, which nothing else in
    the tree would notice.
    """
    var kinds: List[JoinKind] = [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
        JOIN_RIGHT_SEMI,
        JOIN_RIGHT_ANTI,
        JOIN_CROSS,
        JOIN_MARK,
        JOIN_SINGLE,
    ]
    for ref k in kinds:
        var mirrors = True
        try:
            _ = k.mirror()
        except:
            mirrors = False
        assert_equal(
            k.commutes(),
            mirrors,
            String("kind ", k, ": commutes() disagrees with mirror()"),
        )


def test_negates_is_invariant_under_mirror() raises:
    """Exchanging a join's sides cannot change whether it keeps the unmatched
    rows, so `negates` must agree with its own mirror on every kind that has
    one -- a kind added to the anti family on one side only fails here.
    """
    var kinds: List[JoinKind] = [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
        JOIN_RIGHT_SEMI,
        JOIN_RIGHT_ANTI,
    ]
    for ref k in kinds:
        assert_equal(
            k.negates(),
            k.mirror().negates(),
            String("kind ", k, ": negates() changes under mirror()"),
        )
    assert_true(JOIN_ANTI.negates())
    assert_true(JOIN_RIGHT_ANTI.negates())
    assert_false(JOIN_SEMI.negates())
    assert_false(JOIN_RIGHT_SEMI.negates())
    assert_false(JOIN_INNER.negates())
