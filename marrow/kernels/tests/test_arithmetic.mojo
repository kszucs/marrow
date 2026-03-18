from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
    TestSuite,
)

from marrow.arrays import array, arange, Array, PrimitiveArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, int64, float32, float64
from marrow.kernels.arithmetic import (
    add,
    sub,
    mul,
    div,
    floordiv,
    mod,
    min_,
    max_,
    neg,
    abs_,
)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def test_add_typed() raises:
    var a = array[int32]([1, 2, 3, 4])
    var b = array[int32]([10, 20, 30, 40])
    var result = add[int32](a, b)
    assert_equal(len(result), 4)
    assert_equal(result[0], 11)
    assert_equal(result[1], 22)
    assert_equal(result[2], 33)
    assert_equal(result[3], 44)


def test_add_with_nulls() raises:
    """Nulls propagate: null + valid = null."""
    var a = PrimitiveBuilder[int32](3)
    a.append(1)
    a.append(2)
    a.append_null()

    var b = array[int32]([10, 20, 30])
    var result = add[int32](a.finish_typed(), b)
    assert_equal(len(result), 3)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], 11)
    assert_equal(result[1], 22)


def test_add_length_mismatch() raises:
    var a = array[int32]([1, 2])
    var b = array[int32]([1, 2, 3])
    with assert_raises():
        _ = add[int32](a, b)


def test_add_untyped() raises:
    var a = Array(array[int64]([1, 2, 3]))
    var b = Array(array[int64]([4, 5, 6]))
    var result = add(a, b)
    assert_equal(result.length, 3)
    var typed = result.as_primitive[int64]()
    assert_equal(typed[0], 5)
    assert_equal(typed[1], 7)
    assert_equal(typed[2], 9)


def test_add_empty() raises:
    var a = array[int32]()
    var b = array[int32]()
    var result = add[int32](a, b)
    assert_equal(len(result), 0)


def test_add_float64() raises:
    var a = array[float64]([1, 2, 3, 4])
    var b = array[float64]([10, 20, 30, 40])
    var result = add[float64](a, b)
    assert_equal(len(result), 4)
    assert_true(result[0] == 11)
    assert_true(result[1] == 22)
    assert_true(result[2] == 33)
    assert_true(result[3] == 44)


def test_add_large_array() raises:
    """Exercise the SIMD fast path with an array larger than SIMD width."""
    var a = arange[int32](0, 1000)
    var b = arange[int32](0, 1000)
    var result = add[int32](a, b)
    assert_equal(len(result), 1000)
    assert_equal(result[0], 0)
    assert_equal(result[499], 998)
    assert_equal(result[999], 1998)


# ---------------------------------------------------------------------------
# sub
# ---------------------------------------------------------------------------


def test_sub_typed() raises:
    var a = array[int32]([10, 20, 30, 40])
    var b = array[int32]([1, 2, 3, 4])
    var result = sub[int32](a, b)
    assert_equal(result[0], 9)
    assert_equal(result[1], 18)
    assert_equal(result[2], 27)
    assert_equal(result[3], 36)


def test_sub_with_nulls() raises:
    var a = PrimitiveBuilder[int32](3)
    a.append(10)
    a.append(20)
    a.append_null()

    var b = array[int32]([1, 2, 3])
    var result = sub[int32](a.finish_typed(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], 9)
    assert_equal(result[1], 18)


def test_sub_untyped() raises:
    var a = Array(array[int64]([10, 20, 30]))
    var b = Array(array[int64]([1, 2, 3]))
    var result = sub(a, b)
    var typed = result.as_primitive[int64]()
    assert_equal(typed[0], 9)
    assert_equal(typed[1], 18)
    assert_equal(typed[2], 27)


# ---------------------------------------------------------------------------
# mul
# ---------------------------------------------------------------------------


def test_mul_typed() raises:
    var a = array[int32]([2, 3, 4, 5])
    var b = array[int32]([10, 10, 10, 10])
    var result = mul[int32](a, b)
    assert_equal(result[0], 20)
    assert_equal(result[1], 30)
    assert_equal(result[2], 40)
    assert_equal(result[3], 50)


def test_mul_with_nulls() raises:
    var a = PrimitiveBuilder[int32](3)
    a.append(2)
    a.append(3)
    a.append_null()

    var b = array[int32]([10, 10, 10])
    var result = mul[int32](a.finish_typed(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], 20)
    assert_equal(result[1], 30)


def test_mul_large_array() raises:
    var a = arange[int32](0, 1000)
    var b = arange[int32](0, 1000)
    var result = mul[int32](a, b)
    assert_equal(result[0], 0)
    assert_equal(result[10], 100)
    assert_equal(result[31], 961)


# ---------------------------------------------------------------------------
# div
# ---------------------------------------------------------------------------


def test_div_typed() raises:
    var a = array[float64]([10, 20, 30])
    var b = array[float64]([2, 4, 5])
    var result = div[float64](a, b)
    assert_true(result[0] == 5.0)
    assert_true(result[1] == 5.0)
    assert_true(result[2] == 6.0)


def test_div_with_nulls() raises:
    var a = PrimitiveBuilder[float64](3)
    a.append(10)
    a.append(20)
    a.append_null()

    var b = array[float64]([2, 4, 5])
    var result = div[float64](a.finish_typed(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))


# ---------------------------------------------------------------------------
# floordiv
# ---------------------------------------------------------------------------


def test_floordiv_typed() raises:
    var a = array[int32]([10, 20, 7, 15])
    var b = array[int32]([3, 7, 3, 4])
    var result = floordiv[int32](a, b)
    assert_equal(result[0], 3)
    assert_equal(result[1], 2)
    assert_equal(result[2], 2)
    assert_equal(result[3], 3)


# ---------------------------------------------------------------------------
# mod
# ---------------------------------------------------------------------------


def test_mod_typed() raises:
    var a = array[int32]([10, 20, 7, 15])
    var b = array[int32]([3, 7, 3, 4])
    var result = mod[int32](a, b)
    assert_equal(result[0], 1)
    assert_equal(result[1], 6)
    assert_equal(result[2], 1)
    assert_equal(result[3], 3)


# ---------------------------------------------------------------------------
# min_ / max_
# ---------------------------------------------------------------------------


def test_min_typed() raises:
    var a = array[int32]([1, 5, 3, 8])
    var b = array[int32]([4, 2, 3, 6])
    var result = min_[int32](a, b)
    assert_equal(result[0], 1)
    assert_equal(result[1], 2)
    assert_equal(result[2], 3)
    assert_equal(result[3], 6)


def test_max_typed() raises:
    var a = array[int32]([1, 5, 3, 8])
    var b = array[int32]([4, 2, 3, 6])
    var result = max_[int32](a, b)
    assert_equal(result[0], 4)
    assert_equal(result[1], 5)
    assert_equal(result[2], 3)
    assert_equal(result[3], 8)


def test_min_with_nulls() raises:
    var a = PrimitiveBuilder[int32](3)
    a.append(1)
    a.append(5)
    a.append_null()

    var b = array[int32]([4, 2, 3])
    var result = min_[int32](a.finish_typed(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], 1)
    assert_equal(result[1], 2)


# ---------------------------------------------------------------------------
# neg
# ---------------------------------------------------------------------------


def test_neg_typed() raises:
    var a = array[int32]([1, -2, 0, 4])
    var result = neg[int32](a)
    assert_equal(result[0], -1)
    assert_equal(result[1], 2)
    assert_equal(result[2], 0)
    assert_equal(result[3], -4)


def test_neg_with_nulls() raises:
    var a = PrimitiveBuilder[int32](3)
    a.append(1)
    a.append(-2)
    a.append_null()

    var result = neg[int32](a.finish_typed())
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], -1)
    assert_equal(result[1], 2)


def test_neg_large_array() raises:
    """Exercise the SIMD fast path."""
    var a = arange[int32](0, 1000)
    var result = neg[int32](a)
    assert_equal(result[0], 0)
    assert_equal(result[1], -1)
    assert_equal(result[999], -999)


# ---------------------------------------------------------------------------
# abs_
# ---------------------------------------------------------------------------


def test_abs_typed() raises:
    var a = array[int32]([-3, 0, 4, -1])
    var result = abs_[int32](a)
    assert_equal(result[0], 3)
    assert_equal(result[1], 0)
    assert_equal(result[2], 4)
    assert_equal(result[3], 1)


def test_abs_with_nulls() raises:
    var a = PrimitiveBuilder[int32](3)
    a.append(-3)
    a.append(4)
    a.append_null()

    var result = abs_[int32](a.finish_typed())
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], 3)
    assert_equal(result[1], 4)


def test_abs_large_array() raises:
    """Exercise the SIMD fast path."""
    var a = arange[int32](-500, 500)
    var result = abs_[int32](a)
    assert_equal(result[0], 500)
    assert_equal(result[500], 0)
    assert_equal(result[999], 499)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
