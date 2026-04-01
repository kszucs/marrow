from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
    TestSuite,
)

from marrow.arrays import AnyArray, PrimitiveArray
from marrow.builders import array, arange, PrimitiveBuilder
from marrow.dtypes import int32, int64, float32, float64, Int32Type, Int64Type, Float32Type, Float64Type
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
    sign,
    pow_,
    sqrt,
    exp,
    exp2,
    log,
    log2,
    log10,
    log1p,
    floor,
    ceil,
    trunc,
    round,
    sin,
    cos,
)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def test_add_typed() raises:
    var a = array[Int32Type]([1, 2, 3, 4])
    var b = array[Int32Type]([10, 20, 30, 40])
    var result = add[Int32Type](a, b)
    assert_equal(len(result), 4)
    assert_equal(result[0], 11)
    assert_equal(result[1], 22)
    assert_equal(result[2], 33)
    assert_equal(result[3], 44)


def test_add_with_nulls() raises:
    """Nulls propagate: null + valid = null."""
    var a = PrimitiveBuilder[Int32Type](3)
    a.append(1)
    a.append(2)
    a.append_null()

    var b = array[Int32Type]([10, 20, 30])
    var result = add[Int32Type](a.finish(), b)
    assert_equal(len(result), 3)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], 11)
    assert_equal(result[1], 22)


def test_add_length_mismatch() raises:
    var a = array[Int32Type]([1, 2])
    var b = array[Int32Type]([1, 2, 3])
    with assert_raises():
        _ = add[Int32Type](a, b)


def test_add_untyped() raises:
    var a = AnyArray(array[Int64Type]([1, 2, 3]))
    var b = AnyArray(array[Int64Type]([4, 5, 6]))
    var result = add(a, b)
    assert_equal(result.length(), 3)
    ref typed = result.as_primitive[Int64Type]()
    assert_equal(typed[0], 5)
    assert_equal(typed[1], 7)
    assert_equal(typed[2], 9)


def test_add_empty() raises:
    var a = array[Int32Type]()
    var b = array[Int32Type]()
    var result = add[Int32Type](a, b)
    assert_equal(len(result), 0)


def test_add_float64() raises:
    var a = array[Float64Type]([1, 2, 3, 4])
    var b = array[Float64Type]([10, 20, 30, 40])
    var result = add[Float64Type](a, b)
    assert_equal(len(result), 4)
    assert_true(result[0] == 11)
    assert_true(result[1] == 22)
    assert_true(result[2] == 33)
    assert_true(result[3] == 44)


def test_add_large_array() raises:
    """Exercise the SIMD fast path with an array larger than SIMD width."""
    var a = arange[Int32Type](0, 1000)
    var b = arange[Int32Type](0, 1000)
    var result = add[Int32Type](a, b)
    assert_equal(len(result), 1000)
    assert_equal(result[0], 0)
    assert_equal(result[499], 998)
    assert_equal(result[999], 1998)


# ---------------------------------------------------------------------------
# sub
# ---------------------------------------------------------------------------


def test_sub_typed() raises:
    var a = array[Int32Type]([10, 20, 30, 40])
    var b = array[Int32Type]([1, 2, 3, 4])
    var result = sub[Int32Type](a, b)
    assert_equal(result[0], 9)
    assert_equal(result[1], 18)
    assert_equal(result[2], 27)
    assert_equal(result[3], 36)


def test_sub_with_nulls() raises:
    var a = PrimitiveBuilder[Int32Type](3)
    a.append(10)
    a.append(20)
    a.append_null()

    var b = array[Int32Type]([1, 2, 3])
    var result = sub[Int32Type](a.finish(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], 9)
    assert_equal(result[1], 18)


def test_sub_untyped() raises:
    var a = AnyArray(array[Int64Type]([10, 20, 30]))
    var b = AnyArray(array[Int64Type]([1, 2, 3]))
    var result = sub(a, b)
    ref typed = result.as_primitive[Int64Type]()
    assert_equal(typed[0], 9)
    assert_equal(typed[1], 18)
    assert_equal(typed[2], 27)


# ---------------------------------------------------------------------------
# mul
# ---------------------------------------------------------------------------


def test_mul_typed() raises:
    var a = array[Int32Type]([2, 3, 4, 5])
    var b = array[Int32Type]([10, 10, 10, 10])
    var result = mul[Int32Type](a, b)
    assert_equal(result[0], 20)
    assert_equal(result[1], 30)
    assert_equal(result[2], 40)
    assert_equal(result[3], 50)


def test_mul_with_nulls() raises:
    var a = PrimitiveBuilder[Int32Type](3)
    a.append(2)
    a.append(3)
    a.append_null()

    var b = array[Int32Type]([10, 10, 10])
    var result = mul[Int32Type](a.finish(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], 20)
    assert_equal(result[1], 30)


def test_mul_large_array() raises:
    var a = arange[Int32Type](0, 1000)
    var b = arange[Int32Type](0, 1000)
    var result = mul[Int32Type](a, b)
    assert_equal(result[0], 0)
    assert_equal(result[10], 100)
    assert_equal(result[31], 961)


# ---------------------------------------------------------------------------
# div
# ---------------------------------------------------------------------------


def test_div_typed() raises:
    var a = array[Float64Type]([10, 20, 30])
    var b = array[Float64Type]([2, 4, 5])
    var result = div[Float64Type](a, b)
    assert_true(result[0] == 5.0)
    assert_true(result[1] == 5.0)
    assert_true(result[2] == 6.0)


def test_div_with_nulls() raises:
    var a = PrimitiveBuilder[Float64Type](3)
    a.append(10)
    a.append(20)
    a.append_null()

    var b = array[Float64Type]([2, 4, 5])
    var result = div[Float64Type](a.finish(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))


# ---------------------------------------------------------------------------
# floordiv
# ---------------------------------------------------------------------------


def test_floordiv_typed() raises:
    var a = array[Int32Type]([10, 20, 7, 15])
    var b = array[Int32Type]([3, 7, 3, 4])
    var result = floordiv[Int32Type](a, b)
    assert_equal(result[0], 3)
    assert_equal(result[1], 2)
    assert_equal(result[2], 2)
    assert_equal(result[3], 3)


# ---------------------------------------------------------------------------
# mod
# ---------------------------------------------------------------------------


def test_mod_typed() raises:
    var a = array[Int32Type]([10, 20, 7, 15])
    var b = array[Int32Type]([3, 7, 3, 4])
    var result = mod[Int32Type](a, b)
    assert_equal(result[0], 1)
    assert_equal(result[1], 6)
    assert_equal(result[2], 1)
    assert_equal(result[3], 3)


# ---------------------------------------------------------------------------
# min_ / max_
# ---------------------------------------------------------------------------


def test_min_typed() raises:
    var a = array[Int32Type]([1, 5, 3, 8])
    var b = array[Int32Type]([4, 2, 3, 6])
    var result = min_[Int32Type](a, b)
    assert_equal(result[0], 1)
    assert_equal(result[1], 2)
    assert_equal(result[2], 3)
    assert_equal(result[3], 6)


def test_max_typed() raises:
    var a = array[Int32Type]([1, 5, 3, 8])
    var b = array[Int32Type]([4, 2, 3, 6])
    var result = max_[Int32Type](a, b)
    assert_equal(result[0], 4)
    assert_equal(result[1], 5)
    assert_equal(result[2], 3)
    assert_equal(result[3], 8)


def test_min_with_nulls() raises:
    var a = PrimitiveBuilder[Int32Type](3)
    a.append(1)
    a.append(5)
    a.append_null()

    var b = array[Int32Type]([4, 2, 3])
    var result = min_[Int32Type](a.finish(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], 1)
    assert_equal(result[1], 2)


# ---------------------------------------------------------------------------
# neg
# ---------------------------------------------------------------------------


def test_neg_typed() raises:
    var a = array[Int32Type]([1, -2, 0, 4])
    var result = neg[Int32Type](a)
    assert_equal(result[0], -1)
    assert_equal(result[1], 2)
    assert_equal(result[2], 0)
    assert_equal(result[3], -4)


def test_neg_with_nulls() raises:
    var a = PrimitiveBuilder[Int32Type](3)
    a.append(1)
    a.append(-2)
    a.append_null()

    var result = neg[Int32Type](a.finish())
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], -1)
    assert_equal(result[1], 2)


def test_neg_large_array() raises:
    """Exercise the SIMD fast path."""
    var a = arange[Int32Type](0, 1000)
    var result = neg[Int32Type](a)
    assert_equal(result[0], 0)
    assert_equal(result[1], -1)
    assert_equal(result[999], -999)


# ---------------------------------------------------------------------------
# abs_
# ---------------------------------------------------------------------------


def test_abs_typed() raises:
    var a = array[Int32Type]([-3, 0, 4, -1])
    var result = abs_[Int32Type](a)
    assert_equal(result[0], 3)
    assert_equal(result[1], 0)
    assert_equal(result[2], 4)
    assert_equal(result[3], 1)


def test_abs_with_nulls() raises:
    var a = PrimitiveBuilder[Int32Type](3)
    a.append(-3)
    a.append(4)
    a.append_null()

    var result = abs_[Int32Type](a.finish())
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0], 3)
    assert_equal(result[1], 4)


def test_abs_large_array() raises:
    """Exercise the SIMD fast path."""
    var a = arange[Int32Type](-500, 500)
    var result = abs_[Int32Type](a)
    assert_equal(result[0], 500)
    assert_equal(result[500], 0)
    assert_equal(result[999], 499)


# ---------------------------------------------------------------------------
# sign
# ---------------------------------------------------------------------------


def test_sign_typed() raises:
    var a = array[Int32Type]([-3, 0, 5, -1])
    var result = sign[Int32Type](a)
    assert_equal(result[0], -1)
    assert_equal(result[1], 0)
    assert_equal(result[2], 1)
    assert_equal(result[3], -1)


def test_sign_with_nulls() raises:
    var a = PrimitiveBuilder[Int32Type](3)
    a.append(-3)
    a.append_null()
    a.append(5)
    var result = sign[Int32Type](a.finish())
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_equal(result[0], -1)
    assert_equal(result[2], 1)


def test_sign_runtime_typed() raises:
    var a = array[Int32Type]([-3, 0, 5])
    var result = sign(AnyArray(a^))
    ref r = result.as_primitive[Int32Type]()
    assert_equal(r[0], -1)
    assert_equal(r[1], 0)
    assert_equal(r[2], 1)


# ---------------------------------------------------------------------------
# sqrt
# ---------------------------------------------------------------------------


def test_sqrt_typed() raises:
    var a = array[Float32Type]([4.0, 9.0, 16.0, 25.0])
    var result = sqrt[Float32Type](a)
    assert_equal(result[0], 2.0)
    assert_equal(result[1], 3.0)
    assert_equal(result[2], 4.0)
    assert_equal(result[3], 5.0)


def test_sqrt_with_nulls() raises:
    var a = PrimitiveBuilder[Float32Type](3)
    a.append(4.0)
    a.append_null()
    a.append(9.0)
    var result = sqrt[Float32Type](a.finish())
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_equal(result[0], 2.0)
    assert_equal(result[2], 3.0)


def test_sqrt_runtime_typed() raises:
    var a = array[Float64Type]([1.0, 4.0, 9.0])
    var result = sqrt(AnyArray(a^))
    ref r = result.as_primitive[Float64Type]()
    assert_equal(r[0], 1.0)
    assert_equal(r[1], 2.0)
    assert_equal(r[2], 3.0)


# ---------------------------------------------------------------------------
# exp / exp2
# ---------------------------------------------------------------------------


def test_exp_typed() raises:
    var a = array[Float32Type]([0.0, 1.0])
    var result = exp[Float32Type](a)
    assert_equal(result[0], 1.0)
    assert_true(result[1].value() > 2.718 and result[1].value() < 2.719)


def test_exp2_typed() raises:
    var a = array[Float32Type]([0.0, 1.0, 2.0, 3.0])
    var result = exp2[Float32Type](a)
    assert_equal(result[0], 1.0)
    assert_equal(result[1], 2.0)
    assert_equal(result[2], 4.0)
    assert_equal(result[3], 8.0)


# ---------------------------------------------------------------------------
# log / log2 / log10 / log1p
# ---------------------------------------------------------------------------


def test_log_typed() raises:
    var a = array[Float32Type]([1.0, 2.718282])
    var result = log[Float32Type](a)
    assert_equal(result[0], 0.0)
    assert_true(result[1].value() > 0.999 and result[1].value() < 1.001)


def test_log2_typed() raises:
    var a = array[Float32Type]([1.0, 2.0, 4.0, 8.0])
    var result = log2[Float32Type](a)
    assert_equal(result[0], 0.0)
    assert_equal(result[1], 1.0)
    assert_equal(result[2], 2.0)
    assert_equal(result[3], 3.0)


def test_log10_typed() raises:
    var a = array[Float32Type]([1.0, 10.0, 100.0])
    var result = log10[Float32Type](a)
    assert_equal(result[0], 0.0)
    assert_equal(result[1], 1.0)
    assert_equal(result[2], 2.0)


def test_log1p_typed() raises:
    var a = array[Float32Type]([0.0])
    var result = log1p[Float32Type](a)
    assert_equal(result[0], 0.0)


# ---------------------------------------------------------------------------
# floor / ceil / trunc / round
# ---------------------------------------------------------------------------


def test_floor_typed() raises:
    var a = array[Float32Type]([1.7, -1.7, 2.0])
    var result = floor[Float32Type](a)
    assert_equal(result[0], 1.0)
    assert_equal(result[1], -2.0)
    assert_equal(result[2], 2.0)


def test_ceil_typed() raises:
    var a = array[Float32Type]([1.2, -1.2, 2.0])
    var result = ceil[Float32Type](a)
    assert_equal(result[0], 2.0)
    assert_equal(result[1], -1.0)
    assert_equal(result[2], 2.0)


def test_trunc_typed() raises:
    var a = array[Float32Type]([1.9, -1.9, 2.0])
    var result = trunc[Float32Type](a)
    assert_equal(result[0], 1.0)
    assert_equal(result[1], -1.0)
    assert_equal(result[2], 2.0)


def test_round_typed() raises:
    var a = array[Float32Type]([1.4, 1.6, 2.0, -1.6])
    var result = round[Float32Type](a)
    assert_equal(result[0], 1.0)
    assert_equal(result[1], 2.0)
    assert_equal(result[2], 2.0)
    assert_equal(result[3], -2.0)


def test_floor_with_nulls() raises:
    var a = PrimitiveBuilder[Float32Type](3)
    a.append(1.7)
    a.append_null()
    a.append(-1.7)
    var result = floor[Float32Type](a.finish())
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_equal(result[0], 1.0)
    assert_equal(result[2], -2.0)


# ---------------------------------------------------------------------------
# sin / cos
# ---------------------------------------------------------------------------


def test_sin_typed() raises:
    var a = array[Float64Type]([0.0])
    var result = sin[Float64Type](a)
    assert_equal(result[0], 0.0)


def test_cos_typed() raises:
    var a = array[Float64Type]([0.0])
    var result = cos[Float64Type](a)
    assert_equal(result[0], 1.0)


# ---------------------------------------------------------------------------
# pow_
# ---------------------------------------------------------------------------


def test_pow_typed() raises:
    var a = array[Float32Type]([2.0, 3.0, 4.0])
    var b = array[Float32Type]([3.0, 2.0, 0.5])
    var result = pow_[Float32Type](a, b)
    assert_equal(result[0], 8.0)
    assert_equal(result[1], 9.0)
    assert_equal(result[2], 2.0)


def test_pow_with_nulls() raises:
    var a = PrimitiveBuilder[Float32Type](3)
    a.append(2.0)
    a.append_null()
    a.append(4.0)
    var b = array[Float32Type]([3.0, 2.0, 0.5])
    var result = pow_[Float32Type](a.finish(), b)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_equal(result[0], 8.0)
    assert_equal(result[2], 2.0)


def test_pow_runtime_typed() raises:
    var a = array[Float64Type]([2.0, 3.0])
    var b = array[Float64Type]([3.0, 2.0])
    var result = pow_(AnyArray(a^), AnyArray(b^))
    ref r = result.as_primitive[Float64Type]()
    assert_equal(r[0], 8.0)
    assert_equal(r[1], 9.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
