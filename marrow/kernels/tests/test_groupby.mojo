from std.testing import assert_equal, assert_true, assert_false

from ...utils.testing import assert_values_equal

from ...arrays import (
    DynArray,
    PrimitiveArray,
    StringArray,
    StructArray,
)
from ...builders import (
    array,
    PrimitiveBuilder,
    BinaryLikeBuilder,
    StringBuilder,
    Int32Builder,
    Int64Builder,
    Date32Builder,
    Float64Builder,
)
from ...tabular import RecordBatch
from ...dtypes import (
    int8,
    int32,
    int64,
    uint8,
    uint32,
    float64,
    bool_,
    string,
    date32,
    Field,
    struct_,
    Int8Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt32Type,
    Float64Type,
    DynType,
    BinaryLikeType,
    BinaryType,
    LargeBinaryType,
    StringType,
    LargeStringType,
)
from ...arrays import Int32Array
from ...execution import ExecContext
from ...kernels.core import Groups
from ...kernels.groupby import (
    GroupBy,
    Grouping,
    HashGrouping,
    ScalarGrouping,
    GroupedColumns,
    GROUP_SERIAL,
    GROUP_RADIX,
    GROUP_THREAD_LOCAL,
)
from ...exprold.aggregates import (
    Sum,
    Min,
    Max,
    Count,
    Mean,
    CountDistinct,
    ApproxCountDistinct,
)
from ...kernels.aggregate import (
    SumKernel,
    MeanKernel,
    CountKernel,
    CountAgg,
    NumericAgg,
)


from ...kernels.distinct import count_distinct_grouped


# An aggregate is an `Aggregation` — a kernel already bound to its input type.
# `GroupBy(keys).aggregate[A]` takes one directly (the AOT path); `apply[F]`
# resolves the column's dtype to it first (the runtime-dtype path).
# (The runtime, string/plan-driven multi-aggregate path is covered by the
# expression-layer tests in `marrow/exprold/tests/test_streaming.mojo`.)


# ---------------------------------------------------------------------------
# group_by — sum
# ---------------------------------------------------------------------------


def test_groupby_sum_basic() raises:
    """Sum aggregation: [1,2,1,3,2] keys, [10,20,30,40,50] values."""
    var keys: DynArray = array([1, 2, 1, 3, 2], int32)
    var vals: DynArray = array([10, 20, 30, 40, 50], int32)
    var result = GroupBy(keys).apply[Sum](vals)

    # 3 groups: key=1 (sum=40), key=2 (sum=70), key=3 (sum=40)
    assert_equal(result.num_rows(), 3)

    # Key column in encounter order.
    ref k = result.keys[0].as_int32()
    assert_equal(k[0].value(), 1)
    assert_equal(k[1].value(), 2)
    assert_equal(k[2].value(), 3)

    # Sum column (int64 — integer input produces integer output).
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 40)  # 10 + 30
    assert_equal(s[1].value(), 70)  # 20 + 50
    assert_equal(s[2].value(), 40)  # 40


def test_groupby_sum_all_same_key() raises:
    var keys: DynArray = array([5, 5, 5], int32)
    var vals: DynArray = array([1, 2, 3], int32)
    var result = GroupBy(keys).apply[Sum](vals)
    assert_equal(result.num_rows(), 1)
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 6)


# ---------------------------------------------------------------------------
# group_by — min / max
# ---------------------------------------------------------------------------


def test_groupby_min() raises:
    var keys: DynArray = array([1, 2, 1, 2], int32)
    var vals: DynArray = array([30, 10, 20, 40], int32)
    var result = GroupBy(keys).apply[Min](vals)
    # min preserves the input dtype (PyArrow-correct), so int32 in -> int32 out.
    ref m = result.aggregates[0].as_int32()
    assert_equal(m[0].value(), 20)  # min(30, 20)
    assert_equal(m[1].value(), 10)  # min(10, 40)


def test_groupby_max() raises:
    var keys: DynArray = array([1, 2, 1, 2], int32)
    var vals: DynArray = array([30, 10, 20, 40], int32)
    var result = GroupBy(keys).apply[Max](vals)
    # max preserves the input dtype (PyArrow-correct), so int32 in -> int32 out.
    ref m = result.aggregates[0].as_int32()
    assert_equal(m[0].value(), 30)  # max(30, 20)
    assert_equal(m[1].value(), 40)  # max(10, 40)


def test_groupby_sum_int64_precision() raises:
    """Sum of int64 values above 2**53 must not lose precision via float64."""
    var keys: DynArray = array([1, 1], int32)
    var vals: DynArray = array([9_007_199_254_740_993, 1], int64)
    var result = GroupBy(keys).apply[Sum](vals)
    assert_equal(result.num_rows(), 1)
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 9_007_199_254_740_994)


def test_groupby_min_int64_precision() raises:
    """Min over int64 values above 2**53 must stay exact."""
    var keys: DynArray = array([1, 1], int32)
    var vals: DynArray = array(
        [9_007_199_254_740_993, 9_007_199_254_740_995], int64
    )
    var result = GroupBy(keys).apply[Min](vals)
    ref m = result.aggregates[0].as_int64()
    assert_equal(m[0].value(), 9_007_199_254_740_993)


def test_groupby_max_int64_precision() raises:
    """Max over int64 values above 2**53 must stay exact."""
    var keys: DynArray = array([1, 1], int32)
    var vals: DynArray = array(
        [9_007_199_254_740_993, 9_007_199_254_740_995], int64
    )
    var result = GroupBy(keys).apply[Max](vals)
    ref m = result.aggregates[0].as_int64()
    assert_equal(m[0].value(), 9_007_199_254_740_995)


def test_groupby_sum_uint8_widens_to_int64() raises:
    """Integer sum widens to an int64 accumulator (no overflow for small ints).
    """
    var keys: DynArray = array([1, 1], int32)
    var vals: DynArray = array([100, 50], uint8)
    var result = GroupBy(keys).apply[Sum](vals)
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 150)


# ---------------------------------------------------------------------------
# group_by — count
# ---------------------------------------------------------------------------


def test_groupby_count() raises:
    var keys: DynArray = array([1, 2, 1, 3, 2], int32)
    var vals: DynArray = array([10, 20, 30, 40, 50], int32)
    var result = GroupBy(keys).apply[Count](vals)
    ref c = result.aggregates[0].as_int64()
    assert_equal(c[0].value(), 2)  # key=1: 2 rows
    assert_equal(c[1].value(), 2)  # key=2: 2 rows
    assert_equal(c[2].value(), 1)  # key=3: 1 row


# ---------------------------------------------------------------------------
# group_by — mean
# ---------------------------------------------------------------------------


def test_groupby_mean() raises:
    var keys: DynArray = array([1, 2, 1, 2], int32)
    var vals: DynArray = array([10, 20, 30, 40], int32)
    var result = GroupBy(keys).apply[Mean](vals)
    ref m = result.aggregates[0].as_float64()
    assert_equal(m[0].value(), 20.0)  # (10+30)/2
    assert_equal(m[1].value(), 30.0)  # (20+40)/2


def test_groupby_sum_float64_preserved() raises:
    """Float64 input to sum still produces a float64 result (regression guard).
    """
    var keys: DynArray = array([1, 1], int32)
    var vals: DynArray = array([1.5, 2.5], float64)
    var result = GroupBy(keys).apply[Sum](vals)
    ref s = result.aggregates[0].as_float64()
    assert_equal(s[0].value(), 4.0)


# ---------------------------------------------------------------------------
# group_by — null handling
# ---------------------------------------------------------------------------


def test_groupby_null_keys() raises:
    """Null keys form their own group."""
    var keys: DynArray = array([1, None, 2, None, 1], int32)
    var vals: DynArray = array([10, 20, 30, 40, 50], int32)
    var result = GroupBy(keys).apply[Sum](vals)
    assert_equal(result.num_rows(), 3)
    # Group order: 1, null, 2
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 60)  # key=1: 10+50
    assert_equal(s[1].value(), 60)  # key=null: 20+40
    assert_equal(s[2].value(), 30)  # key=2: 30


def test_groupby_null_values_skipped() raises:
    """Null values are skipped in aggregation."""
    var keys: DynArray = array([1, 1, 1], int32)
    var vals: DynArray = array([10, None, 30], int32)
    var result = GroupBy(keys).apply[Sum](vals)
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 40)  # 10 + 30 (null skipped)


def test_groupby_count_skips_nulls() raises:
    """Count only counts non-null values."""
    var keys: DynArray = array([1, 1, 1], int32)
    var vals: DynArray = array([10, None, 30], int32)
    var result = GroupBy(keys).apply[Count](vals)
    ref c = result.aggregates[0].as_int64()
    assert_equal(c[0].value(), 2)  # 2 non-null values


# ---------------------------------------------------------------------------
# group_by — string keys
# ---------------------------------------------------------------------------


def test_groupby_string_key() raises:
    var b = StringBuilder(4)
    b.append("a")
    b.append("b")
    b.append("a")
    b.append("b")
    var keys: DynArray = b.finish()
    var vals: DynArray = array([10, 20, 30, 40], int32)
    var result = GroupBy(keys).apply[Sum](vals)
    assert_equal(result.num_rows(), 2)
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 40)  # "a": 10+30
    assert_equal(s[1].value(), 60)  # "b": 20+40


# ---------------------------------------------------------------------------
# group_by — multi-key (StructArray)
# ---------------------------------------------------------------------------


def test_groupby_multikey() raises:
    var a: DynArray = array([1, 1, 2, 2], int32)
    var b: DynArray = array([10, 20, 10, 20], int32)
    var children = List[DynArray]()
    children.append(a.copy())
    children.append(b.copy())
    var keys = StructArray(
        dtype=struct_(
            Field("a", a.dtype().copy()), Field("b", b.dtype().copy())
        ),
        length=4,
        nulls=0,
        offset=0,
        bitmap=None,
        children=children^,
    )
    var vals: DynArray = array([1, 2, 3, 4], int32)
    var result = GroupBy(keys).apply[Sum](vals)
    assert_equal(result.num_rows(), 4)  # 4 unique combos


# ---------------------------------------------------------------------------
# group_by — empty input
# ---------------------------------------------------------------------------


def test_groupby_empty() raises:
    var keys: DynArray = array(int32)
    var vals: DynArray = array(int32)
    var result = GroupBy(keys).apply[Sum](vals)
    assert_equal(result.num_rows(), 0)


# ---------------------------------------------------------------------------
# group_by — bool key (identity hash path)
# ---------------------------------------------------------------------------


def test_groupby_bool_key() raises:
    var keys: DynArray = array([True, False, True, False, True])
    var vals: DynArray = array([1, 2, 3, 4, 5], int32)
    var result = GroupBy(keys).apply[Sum](vals)
    assert_equal(result.num_rows(), 2)
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 9)  # True: 1+3+5
    assert_equal(s[1].value(), 6)  # False: 2+4


# ---------------------------------------------------------------------------
# group_by — reusing one grouper across kernels (shared keys)
# ---------------------------------------------------------------------------


def test_groupby_sum_and_count_share_keys() raises:
    """Two typed aggregates over the same keys agree on group order/count —
    the building block the expression layer composes for multi-aggregate GROUP
    BY."""
    var keys: DynArray = array([1, 2, 1, 2], int32)
    var vals: DynArray = array([10, 20, 30, 40], int32)

    var sums = GroupBy(keys).apply[Sum](vals)
    var counts = GroupBy(keys).apply[Count](vals)

    assert_equal(sums.num_rows(), 2)
    assert_equal(counts.num_rows(), 2)
    ref s = sums.aggregates[0].as_int64()
    assert_equal(s[0].value(), 40)  # key=1: 10+30
    assert_equal(s[1].value(), 60)  # key=2: 20+40
    ref c = counts.aggregates[0].as_int64()
    assert_equal(c[0].value(), 2)
    assert_equal(c[1].value(), 2)


# ---------------------------------------------------------------------------
# group_by — parallel path matches serial
# ---------------------------------------------------------------------------


def _assert_matches_expected(result: GroupedColumns) raises:
    """Both parallel paths emit group order by hash, so verify the group count,
    the order-independent total, and each group's key↔sum association. Keys are
    `i % 50` over `i` in 0..2999, so group k holds {k, k+50, ..., k+50*59} and
    its sum is 60*k + 88500."""
    assert_equal(result.num_rows(), 50)
    assert_equal(
        SumKernel.reduce(result.aggregates[0].as_int64()).value(), 4498500
    )
    ref pk = result.keys[0].as_int32()
    ref ps = result.aggregates[0].as_int64()
    for i in range(result.num_rows()):
        var k = Int(pk[i].value())
        assert_equal(ps[i].value(), Int64(60 * k + 88500))


def test_groupby_parallel_matches_serial() raises:
    """Both the radix and thread-local parallel paths produce the same groups
    and sums as the serial path (group order differs — by hash — so compare the
    group count, the total, and the per-group key↔sum association)."""
    var kb = Int32Builder(3000)
    var vb = Int32Builder(3000)
    for i in range(3000):
        kb.append(Scalar[int32.native](i % 50))
        vb.append(Scalar[int32.native](i))
    var keys: DynArray = kb.finish()
    var vals: DynArray = vb.finish()

    var children = List[DynArray]()
    children.append(keys.copy())
    var kd = keys.to_data()
    var sa = StructArray(
        dtype=struct_(Field("k", kd.dtype.copy())),
        length=kd.length,
        nulls=kd.nulls,
        offset=kd.offset,
        bitmap=kd.bitmap,
        children=children^,
    )

    comptime SumInt32 = NumericAgg[SumKernel, Int32Type]
    ref typed = vals.as_int32()
    var ctx = ExecContext.parallel(4)
    _assert_matches_expected(
        GroupBy(sa, ctx, GROUP_SERIAL).aggregate[SumInt32](typed)
    )
    _assert_matches_expected(
        GroupBy(sa, ctx, GROUP_RADIX).aggregate[SumInt32](typed)
    )
    _assert_matches_expected(
        GroupBy(sa, ctx, GROUP_THREAD_LOCAL).aggregate[SumInt32](typed)
    )


def _mean_for_key(result: GroupedColumns, key: Int) raises -> Optional[Float64]:
    """The mean-aggregate value for `key` in a group_by result (None if the
    group's output is null — i.e. all values were null)."""
    ref k = result.keys[0].as_int32()
    ref v = result.aggregates[0].as_float64()
    for i in range(result.num_rows()):
        if Int(k[i].value()) == key:
            return None if not v.is_valid(i) else Optional(v[i].value())
    return None


def test_groupby_thread_local_mean_nulls_match_serial() raises:
    """The thread-local merge folds partial `(Σsum, Σcount)` per group and
    finalizes once — so `mean` across chunks (with nulls, and an all-null group)
    must match the serial path exactly. Group 3 is entirely null → null output;
    the merge must carry that through the per-thread count."""
    var kb = Int32Builder(4000)
    var vb = PrimitiveBuilder[Float64Type](4000)
    for i in range(4000):
        var g = i % 4
        kb.append(Scalar[int32.native](g))
        # Group 3 is all-null; elsewhere null every 7th row.
        if g == 3 or i % 7 == 0:
            vb.append_null()
        else:
            vb.append(Scalar[float64.native](Float64(i)))
    var keys: DynArray = kb.finish()
    var vals: DynArray = vb.finish()

    var children = List[DynArray]()
    children.append(keys.copy())
    var kd = keys.to_data()
    var sa = StructArray(
        dtype=struct_(Field("k", kd.dtype.copy())),
        length=kd.length,
        nulls=kd.nulls,
        offset=kd.offset,
        bitmap=kd.bitmap,
        children=children^,
    )

    comptime MeanFloat64 = NumericAgg[MeanKernel, Float64Type]
    ref typed = vals.as_float64()
    var ctx = ExecContext.parallel(4)
    var serial = GroupBy(sa, ctx, GROUP_SERIAL).aggregate[MeanFloat64](typed)
    var threaded = GroupBy(sa, ctx, GROUP_THREAD_LOCAL).aggregate[MeanFloat64](
        typed
    )
    assert_equal(serial.num_rows(), 4)
    assert_equal(threaded.num_rows(), 4)
    for key in range(4):
        var a = _mean_for_key(serial, key)
        var b = _mean_for_key(threaded, key)
        assert_equal(a.__bool__(), b.__bool__())
        if a:
            assert_true(abs(a.value() - b.value()) < 1e-9)
    assert_false(_mean_for_key(threaded, 3).__bool__())  # all-null group


# ---------------------------------------------------------------------------
# group_by — count_distinct / approx_count_distinct
# ---------------------------------------------------------------------------


def test_groupby_count_distinct_basic() raises:
    # key=1 sees values {10, 10, 20} -> 2 distinct; key=2 sees {30, 30} -> 1.
    var keys: DynArray = array([1, 1, 1, 2, 2], int32)
    var vals: DynArray = array([10, 10, 20, 30, 30], int32)
    var result = GroupBy(keys).apply[CountDistinct](vals)
    assert_equal(result.num_rows(), 2)
    var k = result.keys[0].as_int32().copy()
    var c = result.aggregates[0].as_int64().copy()
    assert_equal(k[0].value(), 1)
    assert_equal(c[0].value(), 2)
    assert_equal(k[1].value(), 2)
    assert_equal(c[1].value(), 1)


def test_groupby_count_distinct_excludes_nulls() raises:
    var keys: DynArray = array([1, 1, 1, 1], int32)
    var vb = Int32Builder(4)
    vb.append(5)
    vb.append_null()
    vb.append(5)
    vb.append_null()
    var result = GroupBy(keys).apply[CountDistinct](vb.finish())
    assert_equal(result.num_rows(), 1)
    ref c = result.aggregates[0].as_int64()
    assert_equal(c[0].value(), 1)  # only {5} counts


def test_groupby_count_distinct_all_null_group() raises:
    var keys: DynArray = array([7, 7], int32)
    var vb = Int32Builder(2)
    vb.append_null()
    vb.append_null()
    var result = GroupBy(keys).apply[CountDistinct](vb.finish())
    ref c = result.aggregates[0].as_int64()
    assert_equal(c[0].value(), 0)


def test_groupby_approx_count_distinct_matches_exact_small() raises:
    # small per-group cardinalities → linear counting is near-exact.
    var kb = Int32Builder(3000)
    var vb = Int32Builder(3000)
    for i in range(3000):
        kb.append(Int32(i % 3))  # 3 groups
        vb.append(Int32(i % 300))  # up to 100 distinct per group
    var keys: DynArray = kb.finish()
    var vals: DynArray = vb.finish()
    var result = GroupBy(keys).apply[ApproxCountDistinct](vals)
    assert_equal(result.num_rows(), 3)
    ref c = result.aggregates[0].as_int64()
    for g in range(3):
        # p=11 sketch → ~2.3% standard error; allow a few-sigma band.
        assert_true(abs(c[g].value() - 100) <= 10)


def _assert_all_distinct_10(result: GroupedColumns) raises:
    """10 groups, each with exactly 10 distinct values (see the pattern below).
    """
    assert_equal(result.num_rows(), 10)
    ref c = result.aggregates[0].as_int64()
    for i in range(10):
        assert_equal(c[i].value(), 10)


def test_groupby_count_distinct_radix_matches_serial() raises:
    """The radix-partition-parallel path computes per-group distinct counts
    (each partition's groups are disjoint, so counts concatenate — no merge) and
    agrees with the serial path. keys = i%10, values = i%100, so within group k
    the values are exactly {v in 0..99 : v ≡ k (mod 10)} → 10 distinct each."""
    var kb = Int32Builder(3000)
    var vb = Int32Builder(3000)
    for i in range(3000):
        kb.append(Scalar[int32.native](i % 10))
        vb.append(Scalar[int32.native](i % 100))
    var keys: DynArray = kb.finish()
    var vals: DynArray = vb.finish()

    var children = List[DynArray]()
    children.append(keys.copy())
    var kd = keys.to_data()
    var sa = StructArray(
        dtype=struct_(Field("k", kd.dtype.copy())),
        length=kd.length,
        nulls=kd.nulls,
        offset=kd.offset,
        bitmap=kd.bitmap,
        children=children^,
    )
    var values = List[DynArray]()
    values.append(vals.copy())

    def exact(_j: Int, groups: Groups, col: DynArray) raises {imm} -> DynArray:
        return count_distinct_grouped(groups, col)

    var ctx = ExecContext.parallel(4)
    _assert_all_distinct_10(
        GroupBy(sa, ctx, GROUP_SERIAL).aggregate_columns(values, exact)
    )
    _assert_all_distinct_10(
        GroupBy(sa, ctx, GROUP_RADIX).aggregate_columns(values, exact)
    )


# ---------------------------------------------------------------------------
# group_by — min / max over strings (bytewise) and temporal (dtype preserved)
# ---------------------------------------------------------------------------


def test_groupby_min_max_string() raises:
    # group 1: {"b", "c"} -> min "b", max "c"; group 2: {"a", "z"} -> "a"/"z".
    var keys: DynArray = array([1, 2, 1, 2], int32)
    var sb = StringBuilder(4)
    sb.append("b")
    sb.append("a")
    sb.append("c")
    sb.append("z")
    var vals: DynArray = sb.finish()

    var mn = GroupBy(keys).apply[Min](vals)
    assert_equal(mn.num_rows(), 2)
    ref smn = mn.aggregates[0].as_string()
    assert_equal(smn[0].to_string(), "b")
    assert_equal(smn[1].to_string(), "a")

    var mx = GroupBy(keys).apply[Max](vals)
    ref smx = mx.aggregates[0].as_string()
    assert_equal(smx[0].to_string(), "c")
    assert_equal(smx[1].to_string(), "z")


def test_groupby_min_string_skips_nulls() raises:
    var keys: DynArray = array([1, 1, 1], int32)
    var sb = StringBuilder(3)
    sb.append("m")
    sb.append_null()
    sb.append("a")
    var result = GroupBy(keys).apply[Min](sb.finish())
    ref s = result.aggregates[0].as_string()
    assert_equal(s[0].to_string(), "a")


def test_groupby_min_string_all_null_group() raises:
    var keys: DynArray = array([9, 9], int32)
    var sb = StringBuilder(2)
    sb.append_null()
    sb.append_null()
    var result = GroupBy(keys).apply[Min](sb.finish())
    assert_false(result.aggregates[0].as_string().is_valid(0))


def test_groupby_min_max_date32() raises:
    # group 1: {19000, 18500} -> min 18500 / max 19000; group 2: {18800}.
    var keys: DynArray = array([1, 1, 2], int32)
    var b = Date32Builder(date32(), 3)
    b.append(Scalar[int32.native](19000))
    b.append(Scalar[int32.native](18500))
    b.append(Scalar[int32.native](18800))
    var vals: DynArray = b.finish()

    var mn = GroupBy(keys).apply[Min](vals)
    ref dmn = mn.aggregates[0].as_date32()
    assert_equal(dmn[0].value(), 18500)
    assert_equal(dmn[1].value(), 18800)

    var mx = GroupBy(keys).apply[Max](vals)
    ref dmx = mx.aggregates[0].as_date32()
    assert_equal(dmx[0].value(), 19000)
    assert_equal(dmx[1].value(), 18800)


def test_groupby_min_date32_all_null_group() raises:
    var keys: DynArray = array([1, 1], int32)
    var b = Date32Builder(date32(), 2)
    b.append_null()
    b.append_null()
    var result = GroupBy(keys).apply[Min](b.finish())
    assert_false(result.aggregates[0].as_date32().is_valid(0))


# ---------------------------------------------------------------------------
# group_by — count_distinct over strings (first-class grouped aggregate)
# ---------------------------------------------------------------------------


def test_groupby_count_distinct_string() raises:
    # group 1: {"x"} -> 1 distinct; group 2: {"y", "z"} -> 2 distinct.
    var keys: DynArray = array([1, 1, 2, 2], int32)
    var sb = StringBuilder(4)
    sb.append("x")
    sb.append("x")
    sb.append("y")
    sb.append("z")
    var result = GroupBy(keys).apply[CountDistinct](sb.finish())
    assert_equal(result.num_rows(), 2)
    ref c = result.aggregates[0].as_int64()
    assert_equal(c[0].value(), 1)
    assert_equal(c[1].value(), 2)


# ---------------------------------------------------------------------------
# group_by — count's two grouped implementations must agree (Q7.3)
# ---------------------------------------------------------------------------


def test_grouped_count_implementations_agree_on_nulls() raises:
    """`CountAgg` and `NumericAgg[CountKernel, _]` must be interchangeable.

    Q7.3 is a choice between them, so any disagreement here makes that choice a
    correctness question rather than a performance one. `count` excludes nulls
    (SQL `COUNT(x)`), and group 2 -- all-null, so zero *valid* rows -- counts 0,
    never null.
    """
    var kb = Int32Builder(9)
    for i in range(6):
        kb.append(Scalar[int32.native](i % 2))
    for _ in range(3):
        kb.append(Scalar[int32.native](2))
    var keys: DynArray = kb.finish()

    var vb = Float64Builder(9)
    vb.append(Scalar[float64.native](Float64(1)))
    vb.append_null()
    vb.append(Scalar[float64.native](Float64(3)))
    vb.append_null()
    vb.append(Scalar[float64.native](Float64(5)))
    vb.append(Scalar[float64.native](Float64(6)))
    vb.append_null()
    vb.append_null()
    vb.append_null()
    var vals: DynArray = vb.finish()

    var via_state = GroupBy(keys).aggregate[
        NumericAgg[CountKernel, Float64Type]
    ](NumericAgg[CountKernel, Float64Type].from_any(vals))
    var via_scan = GroupBy(keys).aggregate[CountAgg](CountAgg.from_any(vals))
    # Two implementations, so two independently built layouts — the question
    # is whether they agree on *values*, not on buffer offsets.
    assert_values_equal(via_state.keys[0], via_scan.keys[0])
    assert_values_equal(via_state.aggregates[0], via_scan.aggregates[0])

    ref gk = via_scan.keys[0].as_int32()
    ref gc = via_scan.aggregates[0].as_int64()
    for i in range(via_scan.num_rows()):
        if Int(gk[i].value()) == 2:
            assert_true(gc.is_valid(i))
            assert_equal(gc[i].value(), Int64(0))


# ---------------------------------------------------------------------------
# group_by — `binarylike` keys across all three strategies
#
# Regression: the thread-local strategy is the only one that materializes its
# unique keys through a `DynBuilder` (`HashGrouper._register_new_groups`); the
# serial and radix strategies gather theirs with `take`. `BinaryLikeBuilder`'s
# erased `extend` used to pick the *source* array type from the builder's own
# offset width, naming `BinaryLikeArray[StringType]` for any 32-bit-offset
# builder — so a `binary` key column aborted the process inside the worker
# threads. Forcing the strategy keeps these small: the row-count/cardinality
# heuristic would otherwise need 200k rows to reach thread-local.
# ---------------------------------------------------------------------------


def _bytes_key_column[
    T: BinaryLikeType
](n: Int, groups: Int) raises -> DynArray:
    """`n` rows of `binarylike` keys cycling through `groups` distinct values.
    """
    var kb = BinaryLikeBuilder[T](n)
    for i in range(n):
        kb.append("k" + String(i % groups))
    var out: DynArray = kb.finish()
    return out^


def _check_bytes_keys[T: BinaryLikeType](strategy: UInt8) raises:
    """Group 3000 rows by a `binarylike` key under a forced `strategy`.

    Row `i` carries value `i` and lands in group `i % 50`, so group `g` sums to
    `88500 + 60 * g` — the same arithmetic
    `test_groupby_parallel_matches_serial` checks for int32 keys. Asserting the
    per-group key↔sum *association* (not just the group count) is what catches
    a wrong-typed key builder that yields the right shape and garbage bytes.
    """
    comptime N = 3000
    comptime G = 50
    var keys = _bytes_key_column[T](N, G)
    var vb = Int64Builder(N)
    for i in range(N):
        vb.append(Scalar[int64.native](i))
    var vals: DynArray = vb.finish()

    var ctx = ExecContext.parallel(4)
    var result = GroupBy(keys, ctx, strategy).apply[Sum](vals)

    assert_equal(result.num_rows(), G)
    ref ks = result.keys[0].as_binary_like[T]()
    ref ss = result.aggregates[0].as_int64()
    for g in range(G):
        var want = "k" + String(g)
        var found = False
        for i in range(result.num_rows()):
            if ks[i].to_string() == want:
                assert_true(ss.is_valid(i), "null sum for group " + want)
                assert_equal(ss[i].value(), Int64(88500 + 60 * g))
                found = True
                break
        assert_true(found, "missing group " + want)


def test_groupby_binary_keys_serial() raises:
    _check_bytes_keys[BinaryType](GROUP_SERIAL)


def test_groupby_binary_keys_thread_local() raises:
    _check_bytes_keys[BinaryType](GROUP_THREAD_LOCAL)


def test_groupby_binary_keys_radix() raises:
    _check_bytes_keys[BinaryType](GROUP_RADIX)


def test_groupby_large_binary_keys_serial() raises:
    _check_bytes_keys[LargeBinaryType](GROUP_SERIAL)


def test_groupby_large_binary_keys_thread_local() raises:
    _check_bytes_keys[LargeBinaryType](GROUP_THREAD_LOCAL)


def test_groupby_large_binary_keys_radix() raises:
    _check_bytes_keys[LargeBinaryType](GROUP_RADIX)


def test_groupby_large_string_keys_thread_local() raises:
    """`large_string` shares the 64-bit-offset arm the old code reached by
    accident; it must keep working now that dispatch is on the source dtype."""
    _check_bytes_keys[LargeStringType](GROUP_THREAD_LOCAL)


def test_groupby_string_keys_thread_local() raises:
    """The case that always worked — the only one the offset-width shortcut
    happened to name correctly. Guards against fixing binary by breaking it."""
    _check_bytes_keys[StringType](GROUP_THREAD_LOCAL)


# ---------------------------------------------------------------------------
# Grouping — the placement axis
# ---------------------------------------------------------------------------
def _int_key(values: List[Int]) raises -> List[DynArray]:
    var b = Int64Builder(len(values))
    for v in values:
        b.append(Int64(v))
    var cols = List[DynArray]()
    cols.append(b.finish().to_dyn())
    return cols^


def test_scalar_grouping_allocates_no_ids() raises:
    """One implicit slot, and deliberately no zero vector to say so.

    A fold whose `scatters` is False never reads the ids, so materialising one
    `Int32` per row to communicate a constant is exactly the cost this
    conformer exists to avoid — measured at 14.6x against the register fold.
    """
    var g = ScalarGrouping()
    assert_true(not ScalarGrouping.scatters)
    var groups = g.assign(_int_key([1, 2, 3, 4]), 4)
    assert_equal(groups.num_groups, 1)
    assert_equal(len(groups.ids), 0)  # not four zeros
    assert_equal(g.num_groups(), 1)
    assert_equal(len(g.key_columns(List[Field]())), 0)


def test_hash_grouping_assigns_dense_ids_in_first_appearance_order() raises:
    var g = HashGrouping()
    assert_true(HashGrouping.scatters)
    var groups = g.assign(_int_key([7, 9, 7, 9, 7]), 5)
    assert_equal(groups.num_groups, 2)
    assert_true(groups.ids == array([0, 1, 0, 1, 0], int32))
    assert_equal(g.num_groups(), 2)


def test_hash_grouping_ids_stay_stable_across_batches() raises:
    """The property every accumulator depends on.

    A state that folded batch N keeps its slots when N+1 introduces new groups;
    ids already handed out are never renumbered, so `AggState._grow` can extend
    in place rather than rebuild a fold.
    """
    var g = HashGrouping()
    var first = g.assign(_int_key([7, 9]), 2)
    assert_equal(first.num_groups, 2)

    var second = g.assign(_int_key([9, 4, 7]), 3)
    assert_equal(second.num_groups, 3)
    # 9 and 7 keep the ids they were given in the first batch; only 4 is new.
    assert_true(second.ids == array([1, 2, 0], int32))

    var keys = g.key_columns([Field("k", int64)])
    assert_equal(len(keys), 1)
    assert_true(keys[0].as_int64() == array([7, 9, 4], int64))


def test_a_grouping_is_usable_through_the_trait() raises:
    """Placement composes as a comptime type parameter, not a branch.

    This is what lets `PartitionGrouping` and a sorted or radix placement land
    as conformers rather than as further branches inside `GroupBy`.
    """

    def slots[G: Grouping](var g: G) raises -> Int:
        var groups = g.assign(_int_key([5, 5, 6]), 3)
        return groups.num_groups

    assert_equal(slots(ScalarGrouping()), 1)
    assert_equal(slots(HashGrouping()), 2)
