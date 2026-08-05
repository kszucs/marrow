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
)
from ...tabular import record_batch
from ...kernels.join import (
    hash_join,
    HashJoin,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_MARK,
    JOIN_SINGLE,
    JOIN_CROSS,
    JOIN_ALL,
    JOIN_ANY,
    JoinKind,
)


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


def _constant_hash(
    keys: StructArray,
    ctx: ExecContext = ExecContext.serial(),
) raises -> UInt64Array:
    """Degenerate hash function: all keys map to the same hash.

    Forces every key into a single bucket — without key equality checks,
    an inner join would produce N×M rows (all-pairs). With equality checks,
    only actual matching keys produce output.
    """
    var n = len(keys)
    var b = UInt64Builder(capacity=n)
    for _ in range(n):
        b.unsafe_append(Scalar[uint64.native](42))
    return b.finish()


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
    var join = HashJoin[_constant_hash]()
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

    var join = HashJoin[_constant_hash]()
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
