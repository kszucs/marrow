from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite

from marrow.arrays import AnyArray, PrimitiveArray
from marrow.scalars import AnyScalar
from marrow.builders import (
    array,
    nulls,
    PrimitiveBuilder,
    Int32Builder,
    StringBuilder,
    Date32Builder,
    TimestampBuilder,
)
from marrow.dtypes import (
    int32,
    int64,
    float64,
    string,
    date32,
    timestamp,
    second,
    Int32Type,
    Int64Type,
)
from marrow.expr.aggregates import (
    Sum,
    Mean,
    Min,
    Max,
)
from marrow.kernels.aggregate import (
    Aggregation,
    AggFunction,
    SumKernel,
    MeanKernel,
    MinKernel,
    MaxKernel,
    CountKernel,
)


def whole[F: AggFunction](value: AnyArray) raises -> AnyScalar:
    """Whole-table aggregate of an erased column: resolve the column's dtype to
    the `Aggregation` implementing `F`, run its one-row `whole`, read the value.
    """
    var box = List[AnyArray]()

    @parameter
    def run[A: Aggregation]() raises:
        box.append(A.whole(A.from_any(value)).to_any())

    F.resolve[run](value.dtype())
    return box[0][0]


def test_sumtyped() raises:
    var a = array([1, 2, 3, 4, 5], int64)
    var result = SumKernel.apply[Int64Type](a)
    assert_equal(result.value(), 15)


def test_sumwith_nulls() raises:
    """Sum skips null values."""
    var a = Int32Builder(3)
    a.append(10)
    a.append(20)
    a.append_null()  # index 2 is null
    var result = SumKernel.apply[Int32Type](a.finish())
    assert_equal(result.value(), 30)


def test_sumall_nulls() raises:
    var a = nulls(5, int64)
    var result = SumKernel.apply[Int64Type](a)
    assert_equal(result.value(), 0)


def test_sumempty() raises:
    var a = array(int32)
    var result = SumKernel.apply[Int32Type](a)
    assert_equal(result.value(), 0)


def test_sumuntyped() raises:
    var a: AnyArray = array([1, 2, 3], int64)
    var result = whole[Sum](a)
    assert_equal(result.as_int64().value(), 6)


def test_reduce_typed_sum_widens() raises:
    """Typed `reduce[V]` on a narrow int returns a widened int64 scalar directly
    (no erased `AnyScalar`)."""
    var result = SumKernel.reduce(array([1, 2, 3, 4, 5], int32))
    assert_true(result.type() == int64)
    assert_equal(result.value(), 15)


def test_reduce_typed_min_keeps_type() raises:
    """`min`/`max` keep the operand dtype through the typed path."""
    var result = MinKernel.reduce(array([4, 1, 3, 2], int32))
    assert_true(result.type() == int32)
    assert_equal(result.value(), 1)


def test_reduce_typed_mean_float() raises:
    var result = MeanKernel.reduce(array([1, 2, 3, 4], int32))
    assert_true(result.type() == float64)
    assert_equal(result.value(), 2.5)


def test_reduce_typed_count() raises:
    var b = Int32Builder(3)
    b.append(1)
    b.append_null()
    b.append(3)
    var result = CountKernel.reduce(b.finish())
    assert_true(result.type() == int64)
    assert_equal(result.value(), 2)


def test_mean_int() raises:
    """Mean of integers is a float64 scalar."""
    var a: AnyArray = array([1, 2, 3, 4], int32)
    var result = whole[Mean](a)
    assert_true(result.type() == float64)
    assert_equal(result.as_float64().value(), 2.5)


def test_mean_float() raises:
    var a: AnyArray = array([1.0, 2.0, 6.0], float64)
    var result = whole[Mean](a)
    assert_equal(result.as_float64().value(), 3.0)


def test_mean_skips_nulls() raises:
    """Nulls are excluded from both the sum and the divisor."""
    var a = Int32Builder(4)
    a.append(10)
    a.append(20)
    a.append_null()
    a.append(30)
    var arr: AnyArray = a.finish()
    var result = whole[Mean](arr)
    assert_equal(result.as_float64().value(), 20.0)  # (10+20+30)/3


def test_mean_all_null_is_null() raises:
    var a: AnyArray = nulls(4, int64)
    var result = whole[Mean](a)
    assert_false(result.is_valid())


def test_mean_empty_is_null() raises:
    var a: AnyArray = array(int32)
    var result = whole[Mean](a)
    assert_false(result.is_valid())


# ---------------------------------------------------------------------------
# whole-table min / max over strings (lexicographic / bytewise)
# ---------------------------------------------------------------------------


def _strings(var items: List[String]) raises -> AnyArray:
    var b = StringBuilder(len(items))
    for it in items:
        b.append(it)
    return b.finish()


def test_min_string() raises:
    var a = _strings(["banana", "apple", "cherry"])
    var r = whole[Min](a)
    assert_true(r.type() == string)
    assert_equal(r.as_string().to_string(), "apple")


def test_max_string() raises:
    var a = _strings(["banana", "apple", "cherry"])
    var r = whole[Max](a)
    assert_true(r.type() == string)
    assert_equal(r.as_string().to_string(), "cherry")


def test_min_string_skips_nulls() raises:
    var b = StringBuilder(4)
    b.append("m")
    b.append_null()
    b.append("a")
    b.append_null()
    var a: AnyArray = b.finish()
    assert_equal(whole[Min](a).as_string().to_string(), "a")
    assert_equal(whole[Max](a).as_string().to_string(), "m")


def test_min_string_all_null_is_null() raises:
    var b = StringBuilder(2)
    b.append_null()
    b.append_null()
    var a: AnyArray = b.finish()
    assert_false(whole[Min](a).is_valid())
    assert_false(whole[Max](a).is_valid())


def test_min_string_empty_is_null() raises:
    var b = StringBuilder(0)
    var a: AnyArray = b.finish()
    assert_false(whole[Min](a).is_valid())


# ---------------------------------------------------------------------------
# whole-table min / max over temporal (integer backing, dtype preserved)
# ---------------------------------------------------------------------------


def _date32(var days: List[Int]) raises -> AnyArray:
    var b = Date32Builder(date32(), len(days))
    for d in days:
        b.append(Scalar[int32.native](d))
    return b.finish()


def test_min_max_date32() raises:
    var a = _date32([19000, 18500, 19000, 18800])
    var mn = whole[Min](a)
    var mx = whole[Max](a)
    assert_true(mn.type() == date32().to_any())  # dtype preserved
    assert_true(mx.type() == date32().to_any())
    assert_equal(mn.as_date32().value(), 18500)
    assert_equal(mx.as_date32().value(), 19000)


def test_min_max_date32_skips_nulls() raises:
    var b = Date32Builder(date32(), 3)
    b.append(Scalar[int32.native](19000))
    b.append_null()
    b.append(Scalar[int32.native](18500))
    var a: AnyArray = b.finish()
    assert_equal(whole[Min](a).as_date32().value(), 18500)
    assert_equal(whole[Max](a).as_date32().value(), 19000)


def test_min_max_date32_all_null_is_null() raises:
    var b = Date32Builder(date32(), 2)
    b.append_null()
    b.append_null()
    var a: AnyArray = b.finish()
    assert_false(whole[Min](a).is_valid())
    assert_false(whole[Max](a).is_valid())


def test_min_max_timestamp_preserves_unit_tz() raises:
    var b = TimestampBuilder(timestamp(second, "UTC"), 3)
    b.append(Scalar[int64.native](3000))
    b.append(Scalar[int64.native](1000))
    b.append(Scalar[int64.native](2000))
    var a: AnyArray = b.finish()
    var mn = whole[Min](a)
    assert_true(mn.type() == timestamp(second, "UTC").to_any())
    assert_equal(mn.as_timestamp().value(), 1000)
    assert_equal(whole[Max](a).as_timestamp().value(), 3000)


def main() raises:
    TestSuite.run[__functions_in_module()]()
