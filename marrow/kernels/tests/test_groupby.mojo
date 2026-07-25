from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite

from marrow.arrays import (
    AnyArray,
    PrimitiveArray,
    StringArray,
    StructArray,
)
from marrow.builders import (
    array,
    PrimitiveBuilder,
    StringBuilder,
    Int32Builder,
    Date32Builder,
)
from marrow.tabular import RecordBatch
from marrow.dtypes import (
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
    AnyDataType,
)
from marrow.arrays import Int32Array
from marrow.kernels.groupby import GroupBy
from marrow.kernels.aggregate import (
    SumKernel,
    MinKernel,
    MaxKernel,
    CountKernel,
    MeanKernel,
)
from marrow.kernels.distinct import count_distinct_grouped


# Aggregates are typed kernels: `GroupBy(keys).aggregate[K]` (or the `.sum` /
# `.min` / … shorthands) picks the kernel at compile time.
# (The runtime, string/plan-driven multi-aggregate path is covered by the
# expression-layer tests in `marrow/expr/tests/test_streaming.mojo`.)


# ---------------------------------------------------------------------------
# group_by — sum
# ---------------------------------------------------------------------------


def test_groupby_sum_basic() raises:
    """Sum aggregation: [1,2,1,3,2] keys, [10,20,30,40,50] values."""
    var keys: AnyArray = array([1, 2, 1, 3, 2], int32)
    var vals: AnyArray = array([10, 20, 30, 40, 50], int32)
    var result = GroupBy(keys).sum(vals)

    # 3 groups: key=1 (sum=40), key=2 (sum=70), key=3 (sum=40)
    assert_equal(result.num_rows(), 3)
    assert_equal(result.num_columns(), 2)  # key + sum

    # Key column in encounter order.
    ref k = result.columns[0].as_int32()
    assert_equal(k[0].value(), 1)
    assert_equal(k[1].value(), 2)
    assert_equal(k[2].value(), 3)

    # Sum column (int64 — integer input produces integer output).
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 40)  # 10 + 30
    assert_equal(s[1].value(), 70)  # 20 + 50
    assert_equal(s[2].value(), 40)  # 40


def test_groupby_sum_all_same_key() raises:
    var keys: AnyArray = array([5, 5, 5], int32)
    var vals: AnyArray = array([1, 2, 3], int32)
    var result = GroupBy(keys).sum(vals)
    assert_equal(result.num_rows(), 1)
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 6)


# ---------------------------------------------------------------------------
# group_by — min / max
# ---------------------------------------------------------------------------


def test_groupby_min() raises:
    var keys: AnyArray = array([1, 2, 1, 2], int32)
    var vals: AnyArray = array([30, 10, 20, 40], int32)
    var result = GroupBy(keys).min(vals)
    # min preserves the input dtype (PyArrow-correct), so int32 in -> int32 out.
    assert_true(result.schema.fields[1].dtype == AnyDataType(int32))
    ref m = result.columns[1].as_int32()
    assert_equal(m[0].value(), 20)  # min(30, 20)
    assert_equal(m[1].value(), 10)  # min(10, 40)


def test_groupby_max() raises:
    var keys: AnyArray = array([1, 2, 1, 2], int32)
    var vals: AnyArray = array([30, 10, 20, 40], int32)
    var result = GroupBy(keys).max(vals)
    # max preserves the input dtype (PyArrow-correct), so int32 in -> int32 out.
    assert_true(result.schema.fields[1].dtype == AnyDataType(int32))
    ref m = result.columns[1].as_int32()
    assert_equal(m[0].value(), 30)  # max(30, 20)
    assert_equal(m[1].value(), 40)  # max(10, 40)


def test_groupby_sum_int64_precision() raises:
    """Sum of int64 values above 2**53 must not lose precision via float64."""
    var keys: AnyArray = array([1, 1], int32)
    var vals: AnyArray = array([9_007_199_254_740_993, 1], int64)
    var result = GroupBy(keys).sum(vals)
    assert_equal(result.num_rows(), 1)
    assert_true(result.schema.fields[1].dtype == AnyDataType(int64))
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 9_007_199_254_740_994)


def test_groupby_min_int64_precision() raises:
    """Min over int64 values above 2**53 must stay exact."""
    var keys: AnyArray = array([1, 1], int32)
    var vals: AnyArray = array(
        [9_007_199_254_740_993, 9_007_199_254_740_995], int64
    )
    var result = GroupBy(keys).min(vals)
    assert_true(result.schema.fields[1].dtype == AnyDataType(int64))
    ref m = result.columns[1].as_int64()
    assert_equal(m[0].value(), 9_007_199_254_740_993)


def test_groupby_max_int64_precision() raises:
    """Max over int64 values above 2**53 must stay exact."""
    var keys: AnyArray = array([1, 1], int32)
    var vals: AnyArray = array(
        [9_007_199_254_740_993, 9_007_199_254_740_995], int64
    )
    var result = GroupBy(keys).max(vals)
    assert_true(result.schema.fields[1].dtype == AnyDataType(int64))
    ref m = result.columns[1].as_int64()
    assert_equal(m[0].value(), 9_007_199_254_740_995)


def test_groupby_sum_uint8_widens_to_int64() raises:
    """Integer sum widens to an int64 accumulator (no overflow for small ints).
    """
    var keys: AnyArray = array([1, 1], int32)
    var vals: AnyArray = array([100, 50], uint8)
    var result = GroupBy(keys).sum(vals)
    assert_true(result.schema.fields[1].dtype == AnyDataType(int64))
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 150)


# ---------------------------------------------------------------------------
# group_by — count
# ---------------------------------------------------------------------------


def test_groupby_count() raises:
    var keys: AnyArray = array([1, 2, 1, 3, 2], int32)
    var vals: AnyArray = array([10, 20, 30, 40, 50], int32)
    var result = GroupBy(keys).count(vals)
    ref c = result.columns[1].as_int64()
    assert_equal(c[0].value(), 2)  # key=1: 2 rows
    assert_equal(c[1].value(), 2)  # key=2: 2 rows
    assert_equal(c[2].value(), 1)  # key=3: 1 row


# ---------------------------------------------------------------------------
# group_by — mean
# ---------------------------------------------------------------------------


def test_groupby_mean() raises:
    var keys: AnyArray = array([1, 2, 1, 2], int32)
    var vals: AnyArray = array([10, 20, 30, 40], int32)
    var result = GroupBy(keys).mean(vals)
    ref m = result.columns[1].as_float64()
    assert_equal(m[0].value(), 20.0)  # (10+30)/2
    assert_equal(m[1].value(), 30.0)  # (20+40)/2


def test_groupby_sum_float64_preserved() raises:
    """Float64 input to sum still produces a float64 result (regression guard).
    """
    var keys: AnyArray = array([1, 1], int32)
    var vals: AnyArray = array([1.5, 2.5], float64)
    var result = GroupBy(keys).sum(vals)
    assert_true(result.schema.fields[1].dtype == AnyDataType(float64))
    ref s = result.columns[1].as_float64()
    assert_equal(s[0].value(), 4.0)


# ---------------------------------------------------------------------------
# group_by — null handling
# ---------------------------------------------------------------------------


def test_groupby_null_keys() raises:
    """Null keys form their own group."""
    var keys: AnyArray = array([1, None, 2, None, 1], int32)
    var vals: AnyArray = array([10, 20, 30, 40, 50], int32)
    var result = GroupBy(keys).sum(vals)
    assert_equal(result.num_rows(), 3)
    # Group order: 1, null, 2
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 60)  # key=1: 10+50
    assert_equal(s[1].value(), 60)  # key=null: 20+40
    assert_equal(s[2].value(), 30)  # key=2: 30


def test_groupby_null_values_skipped() raises:
    """Null values are skipped in aggregation."""
    var keys: AnyArray = array([1, 1, 1], int32)
    var vals: AnyArray = array([10, None, 30], int32)
    var result = GroupBy(keys).sum(vals)
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 40)  # 10 + 30 (null skipped)


def test_groupby_count_skips_nulls() raises:
    """Count only counts non-null values."""
    var keys: AnyArray = array([1, 1, 1], int32)
    var vals: AnyArray = array([10, None, 30], int32)
    var result = GroupBy(keys).count(vals)
    ref c = result.columns[1].as_int64()
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
    var keys: AnyArray = b.finish()
    var vals: AnyArray = array([10, 20, 30, 40], int32)
    var result = GroupBy(keys).sum(vals)
    assert_equal(result.num_rows(), 2)
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 40)  # "a": 10+30
    assert_equal(s[1].value(), 60)  # "b": 20+40


# ---------------------------------------------------------------------------
# group_by — multi-key (StructArray)
# ---------------------------------------------------------------------------


def test_groupby_multikey() raises:
    var a: AnyArray = array([1, 1, 2, 2], int32)
    var b: AnyArray = array([10, 20, 10, 20], int32)
    var children = List[AnyArray]()
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
    var vals: AnyArray = array([1, 2, 3, 4], int32)
    var result = GroupBy(keys).sum(vals)
    assert_equal(result.num_rows(), 4)  # 4 unique combos


# ---------------------------------------------------------------------------
# group_by — empty input
# ---------------------------------------------------------------------------


def test_groupby_empty() raises:
    var keys: AnyArray = array(int32)
    var vals: AnyArray = array(int32)
    var result = GroupBy(keys).sum(vals)
    assert_equal(result.num_rows(), 0)


# ---------------------------------------------------------------------------
# group_by — bool key (identity hash path)
# ---------------------------------------------------------------------------


def test_groupby_bool_key() raises:
    var keys: AnyArray = array([True, False, True, False, True])
    var vals: AnyArray = array([1, 2, 3, 4, 5], int32)
    var result = GroupBy(keys).sum(vals)
    assert_equal(result.num_rows(), 2)
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 9)  # True: 1+3+5
    assert_equal(s[1].value(), 6)  # False: 2+4


# ---------------------------------------------------------------------------
# group_by — reusing one grouper across kernels (shared keys)
# ---------------------------------------------------------------------------


def test_groupby_sum_and_count_share_keys() raises:
    """Two typed aggregates over the same keys agree on group order/count —
    the building block the expression layer composes for multi-aggregate GROUP
    BY."""
    var keys: AnyArray = array([1, 2, 1, 2], int32)
    var vals: AnyArray = array([10, 20, 30, 40], int32)

    var sums = GroupBy(keys).sum(vals)
    var counts = GroupBy(keys).count(vals)

    assert_equal(sums.num_rows(), 2)
    assert_equal(counts.num_rows(), 2)
    ref s = sums.columns[1].as_int64()
    assert_equal(s[0].value(), 40)  # key=1: 10+30
    assert_equal(s[1].value(), 60)  # key=2: 20+40
    ref c = counts.columns[1].as_int64()
    assert_equal(c[0].value(), 2)
    assert_equal(c[1].value(), 2)


# ---------------------------------------------------------------------------
# group_by — parallel path matches serial
# ---------------------------------------------------------------------------


def _assert_matches_expected(result: RecordBatch) raises:
    """Both parallel paths emit group order by hash, so verify the group count,
    the order-independent total, and each group's key↔sum association. Keys are
    `i % 50` over `i` in 0..2999, so group k holds {k, k+50, ..., k+50*59} and
    its sum is 60*k + 88500."""
    assert_equal(result.num_rows(), 50)
    assert_equal(
        SumKernel.dispatch(result.column(1)).as_int64().value(), 4498500
    )
    ref pk = result.column(0).as_int32()
    ref ps = result.column(1).as_int64()
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
    var keys: AnyArray = kb.finish()
    var vals: AnyArray = vb.finish()

    var children = List[AnyArray]()
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

    _assert_matches_expected(GroupBy._serial[SumKernel](sa, vals))
    _assert_matches_expected(GroupBy._radix[SumKernel](sa, vals, 4))
    _assert_matches_expected(GroupBy._thread_local[SumKernel](sa, vals, 4))


def _mean_for_key(result: RecordBatch, key: Int) raises -> Optional[Float64]:
    """The mean-aggregate value for `key` in a group_by result (None if the
    group's output is null — i.e. all values were null)."""
    ref k = result.column(0).as_int32()
    ref v = result.column(1).as_float64()
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
    var keys: AnyArray = kb.finish()
    var vals: AnyArray = vb.finish()

    var children = List[AnyArray]()
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

    var serial = GroupBy._serial[MeanKernel](sa, vals)
    var threaded = GroupBy._thread_local[MeanKernel](sa, vals, 4)
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
    var keys: AnyArray = array([1, 1, 1, 2, 2], int32)
    var vals: AnyArray = array([10, 10, 20, 30, 30], int32)
    var result = GroupBy(keys).count_distinct(vals)
    assert_equal(result.num_rows(), 2)
    assert_equal(result.num_columns(), 2)
    assert_true(result.schema.fields[1].name == "count_distinct")
    var k = result.columns[0].as_int32().copy()
    var c = result.columns[1].as_int64().copy()
    assert_equal(k[0].value(), 1)
    assert_equal(c[0].value(), 2)
    assert_equal(k[1].value(), 2)
    assert_equal(c[1].value(), 1)


def test_groupby_count_distinct_excludes_nulls() raises:
    var keys: AnyArray = array([1, 1, 1, 1], int32)
    var vb = Int32Builder(4)
    vb.append(5)
    vb.append_null()
    vb.append(5)
    vb.append_null()
    var result = GroupBy(keys).count_distinct(vb.finish())
    assert_equal(result.num_rows(), 1)
    ref c = result.columns[1].as_int64()
    assert_equal(c[0].value(), 1)  # only {5} counts


def test_groupby_count_distinct_all_null_group() raises:
    var keys: AnyArray = array([7, 7], int32)
    var vb = Int32Builder(2)
    vb.append_null()
    vb.append_null()
    var result = GroupBy(keys).count_distinct(vb.finish())
    ref c = result.columns[1].as_int64()
    assert_equal(c[0].value(), 0)


def test_groupby_approx_count_distinct_matches_exact_small() raises:
    # small per-group cardinalities → linear counting is near-exact.
    var kb = Int32Builder(3000)
    var vb = Int32Builder(3000)
    for i in range(3000):
        kb.append(Int32(i % 3))  # 3 groups
        vb.append(Int32(i % 300))  # up to 100 distinct per group
    var keys: AnyArray = kb.finish()
    var vals: AnyArray = vb.finish()
    var result = GroupBy(keys).approx_count_distinct(vals)
    assert_equal(result.num_rows(), 3)
    ref c = result.columns[1].as_int64()
    for g in range(3):
        # p=11 sketch → ~2.3% standard error; allow a few-sigma band.
        assert_true(abs(c[g].value() - 100) <= 10)


def _assert_all_distinct_10(result: RecordBatch) raises:
    """10 groups, each with exactly 10 distinct values (see the pattern below).
    """
    assert_equal(result.num_rows(), 10)
    ref c = result.column(1).as_int64()
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
    var keys: AnyArray = kb.finish()
    var vals: AnyArray = vb.finish()

    var children = List[AnyArray]()
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
    var values = List[AnyArray]()
    values.append(vals.copy())
    var names = List[String]()
    names.append("count_distinct")

    @parameter
    def exact(
        _j: Int, gids: Int32Array, col: AnyArray, ng: Int
    ) raises -> AnyArray:
        return count_distinct_grouped(gids, col, ng)

    _assert_all_distinct_10(GroupBy._serial_columns[exact](sa, values, names))
    _assert_all_distinct_10(GroupBy._radix_columns[exact](sa, values, names, 4))


# ---------------------------------------------------------------------------
# group_by — min / max over strings (bytewise) and temporal (dtype preserved)
# ---------------------------------------------------------------------------


def test_groupby_min_max_string() raises:
    # group 1: {"b", "c"} -> min "b", max "c"; group 2: {"a", "z"} -> "a"/"z".
    var keys: AnyArray = array([1, 2, 1, 2], int32)
    var sb = StringBuilder(4)
    sb.append("b")
    sb.append("a")
    sb.append("c")
    sb.append("z")
    var vals: AnyArray = sb.finish()

    var mn = GroupBy(keys).min(vals)
    assert_equal(mn.num_rows(), 2)
    assert_true(mn.schema.fields[1].dtype == string.to_any())
    ref smn = mn.columns[1].as_string()
    assert_equal(smn[0].to_string(), "b")
    assert_equal(smn[1].to_string(), "a")

    var mx = GroupBy(keys).max(vals)
    assert_true(mx.schema.fields[1].dtype == string.to_any())
    ref smx = mx.columns[1].as_string()
    assert_equal(smx[0].to_string(), "c")
    assert_equal(smx[1].to_string(), "z")


def test_groupby_min_string_skips_nulls() raises:
    var keys: AnyArray = array([1, 1, 1], int32)
    var sb = StringBuilder(3)
    sb.append("m")
    sb.append_null()
    sb.append("a")
    var result = GroupBy(keys).min(sb.finish())
    ref s = result.columns[1].as_string()
    assert_equal(s[0].to_string(), "a")


def test_groupby_min_string_all_null_group() raises:
    var keys: AnyArray = array([9, 9], int32)
    var sb = StringBuilder(2)
    sb.append_null()
    sb.append_null()
    var result = GroupBy(keys).min(sb.finish())
    assert_false(result.columns[1].as_string().is_valid(0))


def test_groupby_min_max_date32() raises:
    # group 1: {19000, 18500} -> min 18500 / max 19000; group 2: {18800}.
    var keys: AnyArray = array([1, 1, 2], int32)
    var b = Date32Builder(date32(), 3)
    b.append(Scalar[int32.native](19000))
    b.append(Scalar[int32.native](18500))
    b.append(Scalar[int32.native](18800))
    var vals: AnyArray = b.finish()

    var mn = GroupBy(keys).min(vals)
    assert_true(mn.schema.fields[1].dtype == date32().to_any())  # preserved
    ref dmn = mn.columns[1].as_date32()
    assert_equal(dmn[0].value(), 18500)
    assert_equal(dmn[1].value(), 18800)

    var mx = GroupBy(keys).max(vals)
    assert_true(mx.schema.fields[1].dtype == date32().to_any())
    ref dmx = mx.columns[1].as_date32()
    assert_equal(dmx[0].value(), 19000)
    assert_equal(dmx[1].value(), 18800)


def test_groupby_min_date32_all_null_group() raises:
    var keys: AnyArray = array([1, 1], int32)
    var b = Date32Builder(date32(), 2)
    b.append_null()
    b.append_null()
    var result = GroupBy(keys).min(b.finish())
    assert_false(result.columns[1].as_date32().is_valid(0))


# ---------------------------------------------------------------------------
# group_by — count_distinct over strings (first-class grouped aggregate)
# ---------------------------------------------------------------------------


def test_groupby_count_distinct_string() raises:
    # group 1: {"x"} -> 1 distinct; group 2: {"y", "z"} -> 2 distinct.
    var keys: AnyArray = array([1, 1, 2, 2], int32)
    var sb = StringBuilder(4)
    sb.append("x")
    sb.append("x")
    sb.append("y")
    sb.append("z")
    var result = GroupBy(keys).count_distinct(sb.finish())
    assert_equal(result.num_rows(), 2)
    assert_true(result.schema.fields[1].name == "count_distinct")
    ref c = result.columns[1].as_int64()
    assert_equal(c[0].value(), 1)
    assert_equal(c[1].value(), 2)


def main() raises:
    TestSuite.run[__functions_in_module()]()
