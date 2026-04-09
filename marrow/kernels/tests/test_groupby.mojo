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
)
from marrow.kernels.groupby import groupby


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _values(ref a: AnyArray) -> List[AnyArray]:
    var v = List[AnyArray]()
    v.append(a.copy())
    return v^


def _aggs(s: String) -> List[String]:
    var a = List[String]()
    a.append(s)
    return a^


# ---------------------------------------------------------------------------
# groupby — sum
# ---------------------------------------------------------------------------


def test_groupby_sum_basic() raises:
    """Sum aggregation: [1,2,1,3,2] keys, [10,20,30,40,50] values."""
    var keys = AnyArray(array[Int32Type]([1, 2, 1, 3, 2]))
    var vals = AnyArray(array[Int32Type]([10, 20, 30, 40, 50]))
    var result = groupby(keys, _values(vals), _aggs("sum"))

    # 3 groups: key=1 (sum=40), key=2 (sum=70), key=3 (sum=40)
    assert_equal(result.num_rows(), 3)
    assert_equal(result.num_columns(), 2)  # key + sum

    # Key column in encounter order.
    ref k = result.columns[0].as_primitive[Int32Type]()
    assert_equal(k[0], 1)
    assert_equal(k[1], 2)
    assert_equal(k[2], 3)

    # Sum column (float64).
    ref s = result.columns[1].as_primitive[Float64Type]()
    assert_equal(s[0], 40.0)  # 10 + 30
    assert_equal(s[1], 70.0)  # 20 + 50
    assert_equal(s[2], 40.0)  # 40


def test_groupby_sum_all_same_key() raises:
    var keys = AnyArray(array[Int32Type]([5, 5, 5]))
    var vals = AnyArray(array[Int32Type]([1, 2, 3]))
    var result = groupby(keys, _values(vals), _aggs("sum"))
    assert_equal(result.num_rows(), 1)
    ref s = result.columns[1].as_primitive[Float64Type]()
    assert_equal(s[0], 6.0)


# ---------------------------------------------------------------------------
# groupby — min / max
# ---------------------------------------------------------------------------


def test_groupby_min() raises:
    var keys = AnyArray(array[Int32Type]([1, 2, 1, 2]))
    var vals = AnyArray(array[Int32Type]([30, 10, 20, 40]))
    var result = groupby(keys, _values(vals), _aggs("min"))
    ref m = result.columns[1].as_primitive[Float64Type]()
    assert_equal(m[0], 20.0)  # min(30, 20)
    assert_equal(m[1], 10.0)  # min(10, 40)


def test_groupby_max() raises:
    var keys = AnyArray(array[Int32Type]([1, 2, 1, 2]))
    var vals = AnyArray(array[Int32Type]([30, 10, 20, 40]))
    var result = groupby(keys, _values(vals), _aggs("max"))
    ref m = result.columns[1].as_primitive[Float64Type]()
    assert_equal(m[0], 30.0)  # max(30, 20)
    assert_equal(m[1], 40.0)  # max(10, 40)


# ---------------------------------------------------------------------------
# groupby — count
# ---------------------------------------------------------------------------


def test_groupby_count() raises:
    var keys = AnyArray(array[Int32Type]([1, 2, 1, 3, 2]))
    var vals = AnyArray(array[Int32Type]([10, 20, 30, 40, 50]))
    var result = groupby(keys, _values(vals), _aggs("count"))
    ref c = result.columns[1].as_primitive[Int64Type]()
    assert_equal(c[0], 2)  # key=1: 2 rows
    assert_equal(c[1], 2)  # key=2: 2 rows
    assert_equal(c[2], 1)  # key=3: 1 row


# ---------------------------------------------------------------------------
# groupby — mean
# ---------------------------------------------------------------------------


def test_groupby_mean() raises:
    var keys = AnyArray(array[Int32Type]([1, 2, 1, 2]))
    var vals = AnyArray(array[Int32Type]([10, 20, 30, 40]))
    var result = groupby(keys, _values(vals), _aggs("mean"))
    ref m = result.columns[1].as_primitive[Float64Type]()
    assert_equal(m[0], 20.0)  # (10+30)/2
    assert_equal(m[1], 30.0)  # (20+40)/2


# ---------------------------------------------------------------------------
# groupby — null handling
# ---------------------------------------------------------------------------


def test_groupby_null_keys() raises:
    """Null keys form their own group."""
    var keys = AnyArray(array[Int32Type]([1, None, 2, None, 1]))
    var vals = AnyArray(array[Int32Type]([10, 20, 30, 40, 50]))
    var result = groupby(keys, _values(vals), _aggs("sum"))
    assert_equal(result.num_rows(), 3)
    # Group order: 1, null, 2
    ref s = result.columns[1].as_primitive[Float64Type]()
    assert_equal(s[0], 60.0)  # key=1: 10+50
    assert_equal(s[1], 60.0)  # key=null: 20+40
    assert_equal(s[2], 30.0)  # key=2: 30


def test_groupby_null_values_skipped() raises:
    """Null values are skipped in aggregation."""
    var keys = AnyArray(array[Int32Type]([1, 1, 1]))
    var vals = AnyArray(array[Int32Type]([10, None, 30]))
    var result = groupby(keys, _values(vals), _aggs("sum"))
    ref s = result.columns[1].as_primitive[Float64Type]()
    assert_equal(s[0], 40.0)  # 10 + 30 (null skipped)


def test_groupby_count_skips_nulls() raises:
    """Count only counts non-null values."""
    var keys = AnyArray(array[Int32Type]([1, 1, 1]))
    var vals = AnyArray(array[Int32Type]([10, None, 30]))
    var result = groupby(keys, _values(vals), _aggs("count"))
    ref c = result.columns[1].as_primitive[Int64Type]()
    assert_equal(c[0], 2)  # 2 non-null values


# ---------------------------------------------------------------------------
# groupby — string keys
# ---------------------------------------------------------------------------


def test_groupby_string_key() raises:
    var b = StringBuilder(4)
    b.append("a")
    b.append("b")
    b.append("a")
    b.append("b")
    var keys = AnyArray(b.finish())
    var vals = AnyArray(array[Int32Type]([10, 20, 30, 40]))
    var result = groupby(keys, _values(vals), _aggs("sum"))
    assert_equal(result.num_rows(), 2)
    ref s = result.columns[1].as_primitive[Float64Type]()
    assert_equal(s[0], 40.0)  # "a": 10+30
    assert_equal(s[1], 60.0)  # "b": 20+40


# ---------------------------------------------------------------------------
# groupby — multi-key (StructArray)
# ---------------------------------------------------------------------------


def test_groupby_multikey() raises:
    var a = AnyArray(array[Int32Type]([1, 1, 2, 2]))
    var b = AnyArray(array[Int32Type]([10, 20, 10, 20]))
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
    var vals = AnyArray(array[Int32Type]([1, 2, 3, 4]))
    var result = groupby(keys, _values(vals), _aggs("sum"))
    assert_equal(result.num_rows(), 4)  # 4 unique combos


# ---------------------------------------------------------------------------
# groupby — empty input
# ---------------------------------------------------------------------------


def test_groupby_empty() raises:
    var keys = AnyArray(array[Int32Type]())
    var vals = AnyArray(array[Int32Type]())
    var result = groupby(keys, _values(vals), _aggs("sum"))
    assert_equal(result.num_rows(), 0)


# ---------------------------------------------------------------------------
# groupby — bool key (identity hash path)
# ---------------------------------------------------------------------------


def test_groupby_bool_key() raises:
    var keys = AnyArray(array([True, False, True, False, True]))
    var vals = AnyArray(array[Int32Type]([1, 2, 3, 4, 5]))
    var result = groupby(keys, _values(vals), _aggs("sum"))
    assert_equal(result.num_rows(), 2)
    ref s = result.columns[1].as_primitive[Float64Type]()
    assert_equal(s[0], 9.0)  # True: 1+3+5
    assert_equal(s[1], 6.0)  # False: 2+4


# ---------------------------------------------------------------------------
# groupby — multiple aggregations
# ---------------------------------------------------------------------------


def test_groupby_multiple_aggs() raises:
    var keys = AnyArray(array[Int32Type]([1, 2, 1, 2]))

    var vals = List[AnyArray]()
    var v = AnyArray(array[Int32Type]([10, 20, 30, 40]))
    vals.append(v.copy())
    vals.append(v.copy())

    var aggs = List[String]()
    aggs.append("sum")
    aggs.append("count")

    var result = groupby(keys, vals, aggs)
    assert_equal(result.num_columns(), 3)  # key + sum + count

    ref s = result.columns[1].as_primitive[Float64Type]()
    assert_equal(s[0], 40.0)  # sum for key=1
    assert_equal(s[1], 60.0)  # sum for key=2

    ref c = result.columns[2].as_primitive[Int64Type]()
    assert_equal(c[0], 2)  # count for key=1
    assert_equal(c[1], 2)  # count for key=2


def main() raises:
    TestSuite.run[__functions_in_module()]()
