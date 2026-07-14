from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite

from marrow.arrays import (
    AnyArray,
    PrimitiveArray,
    StringArray,
    StructArray,
)
from marrow.builders import array, PrimitiveBuilder, StringBuilder
from marrow.dtypes import (
    int8,
    int32,
    int64,
    uint8,
    uint32,
    float64,
    bool_,
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
from marrow.kernels.groupby import group_by
from marrow.kernels.aggregate import (
    SumKernel,
    MinKernel,
    MaxKernel,
    CountKernel,
    MeanKernel,
)


# Aggregates are typed kernels: `group_by[K]` picks the kernel at compile time.
# (The runtime, string/plan-driven multi-aggregate path is covered by the
# expression-layer tests in `marrow/expr/tests/test_streaming.mojo`.)


# ---------------------------------------------------------------------------
# group_by — sum
# ---------------------------------------------------------------------------


def test_groupby_sum_basic() raises:
    """Sum aggregation: [1,2,1,3,2] keys, [10,20,30,40,50] values."""
    var keys: AnyArray = array([1, 2, 1, 3, 2], int32)
    var vals: AnyArray = array([10, 20, 30, 40, 50], int32)
    var result = group_by[SumKernel](keys, vals)

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
    var result = group_by[SumKernel](keys, vals)
    assert_equal(result.num_rows(), 1)
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 6)


# ---------------------------------------------------------------------------
# group_by — min / max
# ---------------------------------------------------------------------------


def test_groupby_min() raises:
    var keys: AnyArray = array([1, 2, 1, 2], int32)
    var vals: AnyArray = array([30, 10, 20, 40], int32)
    var result = group_by[MinKernel](keys, vals)
    # min preserves the input dtype (PyArrow-correct), so int32 in -> int32 out.
    assert_true(result.schema.fields[1].dtype == AnyDataType(int32))
    ref m = result.columns[1].as_int32()
    assert_equal(m[0].value(), 20)  # min(30, 20)
    assert_equal(m[1].value(), 10)  # min(10, 40)


def test_groupby_max() raises:
    var keys: AnyArray = array([1, 2, 1, 2], int32)
    var vals: AnyArray = array([30, 10, 20, 40], int32)
    var result = group_by[MaxKernel](keys, vals)
    # max preserves the input dtype (PyArrow-correct), so int32 in -> int32 out.
    assert_true(result.schema.fields[1].dtype == AnyDataType(int32))
    ref m = result.columns[1].as_int32()
    assert_equal(m[0].value(), 30)  # max(30, 20)
    assert_equal(m[1].value(), 40)  # max(10, 40)


def test_groupby_sum_int64_precision() raises:
    """Sum of int64 values above 2**53 must not lose precision via float64."""
    var keys: AnyArray = array([1, 1], int32)
    var vals: AnyArray = array([9_007_199_254_740_993, 1], int64)
    var result = group_by[SumKernel](keys, vals)
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
    var result = group_by[MinKernel](keys, vals)
    assert_true(result.schema.fields[1].dtype == AnyDataType(int64))
    ref m = result.columns[1].as_int64()
    assert_equal(m[0].value(), 9_007_199_254_740_993)


def test_groupby_max_int64_precision() raises:
    """Max over int64 values above 2**53 must stay exact."""
    var keys: AnyArray = array([1, 1], int32)
    var vals: AnyArray = array(
        [9_007_199_254_740_993, 9_007_199_254_740_995], int64
    )
    var result = group_by[MaxKernel](keys, vals)
    assert_true(result.schema.fields[1].dtype == AnyDataType(int64))
    ref m = result.columns[1].as_int64()
    assert_equal(m[0].value(), 9_007_199_254_740_995)


def test_groupby_sum_uint8_widens_to_int64() raises:
    """Integer sum widens to an int64 accumulator (no overflow for small ints).
    """
    var keys: AnyArray = array([1, 1], int32)
    var vals: AnyArray = array([100, 50], uint8)
    var result = group_by[SumKernel](keys, vals)
    assert_true(result.schema.fields[1].dtype == AnyDataType(int64))
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 150)


# ---------------------------------------------------------------------------
# group_by — count
# ---------------------------------------------------------------------------


def test_groupby_count() raises:
    var keys: AnyArray = array([1, 2, 1, 3, 2], int32)
    var vals: AnyArray = array([10, 20, 30, 40, 50], int32)
    var result = group_by[CountKernel](keys, vals)
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
    var result = group_by[MeanKernel](keys, vals)
    ref m = result.columns[1].as_float64()
    assert_equal(m[0].value(), 20.0)  # (10+30)/2
    assert_equal(m[1].value(), 30.0)  # (20+40)/2


def test_groupby_sum_float64_preserved() raises:
    """Float64 input to sum still produces a float64 result (regression guard).
    """
    var keys: AnyArray = array([1, 1], int32)
    var vals: AnyArray = array([1.5, 2.5], float64)
    var result = group_by[SumKernel](keys, vals)
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
    var result = group_by[SumKernel](keys, vals)
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
    var result = group_by[SumKernel](keys, vals)
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 40)  # 10 + 30 (null skipped)


def test_groupby_count_skips_nulls() raises:
    """Count only counts non-null values."""
    var keys: AnyArray = array([1, 1, 1], int32)
    var vals: AnyArray = array([10, None, 30], int32)
    var result = group_by[CountKernel](keys, vals)
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
    var result = group_by[SumKernel](keys, vals)
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
    var result = group_by[SumKernel](keys, vals)
    assert_equal(result.num_rows(), 4)  # 4 unique combos


# ---------------------------------------------------------------------------
# group_by — empty input
# ---------------------------------------------------------------------------


def test_groupby_empty() raises:
    var keys: AnyArray = array(int32)
    var vals: AnyArray = array(int32)
    var result = group_by[SumKernel](keys, vals)
    assert_equal(result.num_rows(), 0)


# ---------------------------------------------------------------------------
# group_by — bool key (identity hash path)
# ---------------------------------------------------------------------------


def test_groupby_bool_key() raises:
    var keys: AnyArray = array([True, False, True, False, True])
    var vals: AnyArray = array([1, 2, 3, 4, 5], int32)
    var result = group_by[SumKernel](keys, vals)
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

    var sums = group_by[SumKernel](keys, vals)
    var counts = group_by[CountKernel](keys, vals)

    assert_equal(sums.num_rows(), 2)
    assert_equal(counts.num_rows(), 2)
    ref s = sums.columns[1].as_int64()
    assert_equal(s[0].value(), 40)  # key=1: 10+30
    assert_equal(s[1].value(), 60)  # key=2: 20+40
    ref c = counts.columns[1].as_int64()
    assert_equal(c[0].value(), 2)
    assert_equal(c[1].value(), 2)


def main() raises:
    TestSuite.run[__functions_in_module()]()
