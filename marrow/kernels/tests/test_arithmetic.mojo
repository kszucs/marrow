from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from marrow.testing import TestSuite

from marrow.arrays import AnyArray, PrimitiveArray
from marrow.builders import (
    array,
    arange,
    PrimitiveBuilder,
    Int32Builder,
    Float32Builder,
    Float64Builder,
)
from marrow.dtypes import (
    int32,
    int64,
    float32,
    float64,
    Int32Type,
    Int64Type,
    Float32Type,
    Float64Type,
)
from marrow.kernels.arithmetic import (
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    FloordivKernel,
    ModKernel,
    MinKernel,
    MaxKernel,
    NegKernel,
    AbsKernel,
    SignKernel,
    PowKernel,
    SqrtKernel,
    ExpKernel,
    Exp2Kernel,
    LogKernel,
    Log2Kernel,
    Log10Kernel,
    Log1pKernel,
    FloorKernel,
    CeilKernel,
    TruncKernel,
    RoundKernel,
    SinKernel,
    CosKernel,
)
from marrow.kernels.execution import ExecutionContext


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def test_add_typed() raises:
    var a = array([1, 2, 3, 4], int32)
    var b = array([10, 20, 30, 40], int32)
    var result = AddKernel.apply[Int32Type](a, b)
    assert_equal(len(result), 4)
    assert_equal(result[0].value(), 11)
    assert_equal(result[1].value(), 22)
    assert_equal(result[2].value(), 33)
    assert_equal(result[3].value(), 44)


def test_add_with_nulls() raises:
    """Nulls propagate: null + valid = null."""
    var a = Int32Builder(3)
    a.append(1)
    a.append(2)
    a.append_null()

    var b = array([10, 20, 30], int32)
    var result = AddKernel.apply[Int32Type](a.finish(), b)
    assert_equal(len(result), 3)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0].value(), 11)
    assert_equal(result[1].value(), 22)


def test_add_length_mismatch() raises:
    var a = array([1, 2], int32)
    var b = array([1, 2, 3], int32)
    with assert_raises():
        _ = AddKernel.apply[Int32Type](a, b)


def test_add_untyped() raises:
    var a: AnyArray = array([1, 2, 3], int64)
    var b: AnyArray = array([4, 5, 6], int64)
    var result = AddKernel.dispatch(a, b)
    assert_equal(result.length(), 3)
    ref typed = result.as_int64()
    assert_equal(typed[0].value(), 5)
    assert_equal(typed[1].value(), 7)
    assert_equal(typed[2].value(), 9)


def test_add_empty() raises:
    var a = array(int32)
    var b = array(int32)
    var result = AddKernel.apply[Int32Type](a, b)
    assert_equal(len(result), 0)


def test_add_float64() raises:
    var a = array([1, 2, 3, 4], float64)
    var b = array([10, 20, 30, 40], float64)
    var result = AddKernel.apply[Float64Type](a, b)
    assert_equal(len(result), 4)
    assert_true(result[0].value() == 11)
    assert_true(result[1].value() == 22)
    assert_true(result[2].value() == 33)
    assert_true(result[3].value() == 44)


def test_add_large_array() raises:
    """Exercise the SIMD fast path with an array larger than SIMD width."""
    var a = arange[Int32Type](0, 1000)
    var b = arange[Int32Type](0, 1000)
    var result = AddKernel.apply[Int32Type](a, b)
    assert_equal(len(result), 1000)
    assert_equal(result[0].value(), 0)
    assert_equal(result[499].value(), 998)
    assert_equal(result[999].value(), 1998)


# ---------------------------------------------------------------------------
# subtract
# ---------------------------------------------------------------------------


def test_sub_typed() raises:
    var a = array([10, 20, 30, 40], int32)
    var b = array([1, 2, 3, 4], int32)
    var result = SubKernel.apply[Int32Type](a, b)
    assert_equal(result[0].value(), 9)
    assert_equal(result[1].value(), 18)
    assert_equal(result[2].value(), 27)
    assert_equal(result[3].value(), 36)


def test_sub_with_nulls() raises:
    var a = Int32Builder(3)
    a.append(10)
    a.append(20)
    a.append_null()

    var b = array([1, 2, 3], int32)
    var result = SubKernel.apply[Int32Type](a.finish(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0].value(), 9)
    assert_equal(result[1].value(), 18)


def test_sub_untyped() raises:
    var a: AnyArray = array([10, 20, 30], int64)
    var b: AnyArray = array([1, 2, 3], int64)
    var result = SubKernel.dispatch(a, b)
    ref typed = result.as_int64()
    assert_equal(typed[0].value(), 9)
    assert_equal(typed[1].value(), 18)
    assert_equal(typed[2].value(), 27)


# ---------------------------------------------------------------------------
# multiply
# ---------------------------------------------------------------------------


def test_mul_typed() raises:
    var a = array([2, 3, 4, 5], int32)
    var b = array([10, 10, 10, 10], int32)
    var result = MulKernel.apply[Int32Type](a, b)
    assert_equal(result[0].value(), 20)
    assert_equal(result[1].value(), 30)
    assert_equal(result[2].value(), 40)
    assert_equal(result[3].value(), 50)


def test_mul_with_nulls() raises:
    var a = Int32Builder(3)
    a.append(2)
    a.append(3)
    a.append_null()

    var b = array([10, 10, 10], int32)
    var result = MulKernel.apply[Int32Type](a.finish(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0].value(), 20)
    assert_equal(result[1].value(), 30)


def test_mul_large_array() raises:
    var a = arange[Int32Type](0, 1000)
    var b = arange[Int32Type](0, 1000)
    var result = MulKernel.apply[Int32Type](a, b)
    assert_equal(result[0].value(), 0)
    assert_equal(result[10].value(), 100)
    assert_equal(result[31].value(), 961)


# ---------------------------------------------------------------------------
# divide
# ---------------------------------------------------------------------------


def test_div_typed() raises:
    var a = array([10, 20, 30], float64)
    var b = array([2, 4, 5], float64)
    var result = DivKernel.apply[Float64Type](a, b)
    assert_true(result[0].value() == 5.0)
    assert_true(result[1].value() == 5.0)
    assert_true(result[2].value() == 6.0)


def test_div_with_nulls() raises:
    var a = Float64Builder(3)
    a.append(10)
    a.append(20)
    a.append_null()

    var b = array([2, 4, 5], float64)
    var result = DivKernel.apply[Float64Type](a.finish(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))


# ---------------------------------------------------------------------------
# floordiv
# ---------------------------------------------------------------------------


def test_floordiv_typed() raises:
    var a = array([10, 20, 7, 15], int32)
    var b = array([3, 7, 3, 4], int32)
    var result = FloordivKernel.apply[Int32Type](a, b)
    assert_equal(result[0].value(), 3)
    assert_equal(result[1].value(), 2)
    assert_equal(result[2].value(), 2)
    assert_equal(result[3].value(), 3)


# ---------------------------------------------------------------------------
# mod
# ---------------------------------------------------------------------------


def test_mod_typed() raises:
    var a = array([10, 20, 7, 15], int32)
    var b = array([3, 7, 3, 4], int32)
    var result = ModKernel.apply[Int32Type](a, b)
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 6)
    assert_equal(result[2].value(), 1)
    assert_equal(result[3].value(), 3)


# ---------------------------------------------------------------------------
# min_element_wise / max_element_wise
# ---------------------------------------------------------------------------


def test_min_typed() raises:
    var a = array([1, 5, 3, 8], int32)
    var b = array([4, 2, 3, 6], int32)
    var result = MinKernel.apply[Int32Type](a, b)
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 2)
    assert_equal(result[2].value(), 3)
    assert_equal(result[3].value(), 6)


def test_max_typed() raises:
    var a = array([1, 5, 3, 8], int32)
    var b = array([4, 2, 3, 6], int32)
    var result = MaxKernel.apply[Int32Type](a, b)
    assert_equal(result[0].value(), 4)
    assert_equal(result[1].value(), 5)
    assert_equal(result[2].value(), 3)
    assert_equal(result[3].value(), 8)


def test_min_with_nulls() raises:
    var a = Int32Builder(3)
    a.append(1)
    a.append(5)
    a.append_null()

    var b = array([4, 2, 3], int32)
    var result = MinKernel.apply[Int32Type](a.finish(), b)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 2)


# ---------------------------------------------------------------------------
# neg
# ---------------------------------------------------------------------------


def test_neg_typed() raises:
    var a = array([1, -2, 0, 4], int32)
    var result = NegKernel.apply[Int32Type](a)
    assert_equal(result[0].value(), -1)
    assert_equal(result[1].value(), 2)
    assert_equal(result[2].value(), 0)
    assert_equal(result[3].value(), -4)


def test_neg_with_nulls() raises:
    var a = Int32Builder(3)
    a.append(1)
    a.append(-2)
    a.append_null()

    var result = NegKernel.apply[Int32Type](a.finish())
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0].value(), -1)
    assert_equal(result[1].value(), 2)


def test_neg_large_array() raises:
    """Exercise the SIMD fast path."""
    var a = arange[Int32Type](0, 1000)
    var result = NegKernel.apply[Int32Type](a)
    assert_equal(result[0].value(), 0)
    assert_equal(result[1].value(), -1)
    assert_equal(result[999].value(), -999)


# ---------------------------------------------------------------------------
# abs_
# ---------------------------------------------------------------------------


def test_abs_typed() raises:
    var a = array([-3, 0, 4, -1], int32)
    var result = AbsKernel.apply[Int32Type](a)
    assert_equal(result[0].value(), 3)
    assert_equal(result[1].value(), 0)
    assert_equal(result[2].value(), 4)
    assert_equal(result[3].value(), 1)


def test_abs_with_nulls() raises:
    var a = Int32Builder(3)
    a.append(-3)
    a.append(4)
    a.append_null()

    var result = AbsKernel.apply[Int32Type](a.finish())
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result[0].value(), 3)
    assert_equal(result[1].value(), 4)


def test_abs_large_array() raises:
    """Exercise the SIMD fast path."""
    var a = arange[Int32Type](-500, 500)
    var result = AbsKernel.apply[Int32Type](a)
    assert_equal(result[0].value(), 500)
    assert_equal(result[500].value(), 0)
    assert_equal(result[999].value(), 499)


# ---------------------------------------------------------------------------
# sign
# ---------------------------------------------------------------------------


def test_sign_typed() raises:
    var a = array([-3, 0, 5, -1], int32)
    var result = SignKernel.apply[Int32Type](a)
    assert_equal(result[0].value(), -1)
    assert_equal(result[1].value(), 0)
    assert_equal(result[2].value(), 1)
    assert_equal(result[3].value(), -1)


def test_sign_with_nulls() raises:
    var a = Int32Builder(3)
    a.append(-3)
    a.append_null()
    a.append(5)
    var result = SignKernel.apply[Int32Type](a.finish())
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_equal(result[0].value(), -1)
    assert_equal(result[2].value(), 1)


def test_sign_runtime_typed() raises:
    var a = array([-3, 0, 5], int32)
    var result = SignKernel.dispatch(a^)
    ref typed = result.as_int32()
    assert_equal(typed[0].value(), -1)
    assert_equal(typed[1].value(), 0)
    assert_equal(typed[2].value(), 1)


# ---------------------------------------------------------------------------
# sqrt
# ---------------------------------------------------------------------------


def test_sqrt_typed() raises:
    var a = array([4.0, 9.0, 16.0, 25.0], float32)
    var result = SqrtKernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 2.0)
    assert_equal(result[1].value(), 3.0)
    assert_equal(result[2].value(), 4.0)
    assert_equal(result[3].value(), 5.0)


def test_sqrt_with_nulls() raises:
    var a = Float32Builder(3)
    a.append(4.0)
    a.append_null()
    a.append(9.0)
    var result = SqrtKernel.apply[Float32Type](a.finish())
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_equal(result[0].value(), 2.0)
    assert_equal(result[2].value(), 3.0)


def test_sqrt_runtime_typed() raises:
    var a = array([1.0, 4.0, 9.0], float64)
    var result = SqrtKernel.dispatch(a^)
    ref typed = result.as_float64()
    assert_equal(typed[0].value(), 1.0)
    assert_equal(typed[1].value(), 2.0)
    assert_equal(typed[2].value(), 3.0)


# ---------------------------------------------------------------------------
# exp / exp2
# ---------------------------------------------------------------------------


def test_exp_typed() raises:
    var a = array([0.0, 1.0], float32)
    var result = ExpKernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 1.0)
    assert_true(result[1].value() > 2.718 and result[1].value() < 2.719)


def test_exp2_typed() raises:
    var a = array([0.0, 1.0, 2.0, 3.0], float32)
    var result = Exp2Kernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 1.0)
    assert_equal(result[1].value(), 2.0)
    assert_equal(result[2].value(), 4.0)
    assert_equal(result[3].value(), 8.0)


# ---------------------------------------------------------------------------
# log / log2 / log10 / log1p
# ---------------------------------------------------------------------------


def test_log_typed() raises:
    var a = array([1.0, 2.718282], float32)
    var result = LogKernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 0.0)
    assert_true(result[1].value() > 0.999 and result[1].value() < 1.001)


def test_log2_typed() raises:
    var a = array([1.0, 2.0, 4.0, 8.0], float32)
    var result = Log2Kernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 0.0)
    assert_equal(result[1].value(), 1.0)
    assert_equal(result[2].value(), 2.0)
    assert_equal(result[3].value(), 3.0)


def test_log10_typed() raises:
    var a = array([1.0, 10.0, 100.0], float32)
    var result = Log10Kernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 0.0)
    assert_equal(result[1].value(), 1.0)
    assert_equal(result[2].value(), 2.0)


def test_log1p_typed() raises:
    var a = array([0.0], float32)
    var result = Log1pKernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 0.0)


# ---------------------------------------------------------------------------
# floor / ceil / trunc / round
# ---------------------------------------------------------------------------


def test_floor_typed() raises:
    var a = array([1.7, -1.7, 2.0], float32)
    var result = FloorKernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 1.0)
    assert_equal(result[1].value(), -2.0)
    assert_equal(result[2].value(), 2.0)


def test_ceil_typed() raises:
    var a = array([1.2, -1.2, 2.0], float32)
    var result = CeilKernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 2.0)
    assert_equal(result[1].value(), -1.0)
    assert_equal(result[2].value(), 2.0)


def test_trunc_typed() raises:
    var a = array([1.9, -1.9, 2.0], float32)
    var result = TruncKernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 1.0)
    assert_equal(result[1].value(), -1.0)
    assert_equal(result[2].value(), 2.0)


def test_round_typed() raises:
    var a = array([1.4, 1.6, 2.0, -1.6], float32)
    var result = RoundKernel.apply[Float32Type](a)
    assert_equal(result[0].value(), 1.0)
    assert_equal(result[1].value(), 2.0)
    assert_equal(result[2].value(), 2.0)
    assert_equal(result[3].value(), -2.0)


def test_floor_with_nulls() raises:
    var a = Float32Builder(3)
    a.append(1.7)
    a.append_null()
    a.append(-1.7)
    var result = FloorKernel.apply[Float32Type](a.finish())
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_equal(result[0].value(), 1.0)
    assert_equal(result[2].value(), -2.0)


# ---------------------------------------------------------------------------
# sin / cos
# ---------------------------------------------------------------------------


def test_sin_typed() raises:
    var a = array([0.0], float64)
    var result = SinKernel.apply[Float64Type](a)
    assert_equal(result[0].value(), 0.0)


def test_cos_typed() raises:
    var a = array([0.0], float64)
    var result = CosKernel.apply[Float64Type](a)
    assert_equal(result[0].value(), 1.0)


# ---------------------------------------------------------------------------
# pow_
# ---------------------------------------------------------------------------


def test_pow_typed() raises:
    var a = array([2.0, 3.0, 4.0], float32)
    var b = array([3.0, 2.0, 0.5], float32)
    var result = PowKernel.apply[Float32Type](a, b)
    assert_equal(result[0].value(), 8.0)
    assert_equal(result[1].value(), 9.0)
    assert_equal(result[2].value(), 2.0)


def test_pow_with_nulls() raises:
    var a = Float32Builder(3)
    a.append(2.0)
    a.append_null()
    a.append(4.0)
    var b = array([3.0, 2.0, 0.5], float32)
    var result = PowKernel.apply[Float32Type](a.finish(), b)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_equal(result[0].value(), 8.0)
    assert_equal(result[2].value(), 2.0)


def test_pow_runtime_typed() raises:
    var a = array([2.0, 3.0], float64)
    var b = array([3.0, 2.0], float64)
    var result = PowKernel.dispatch(a^, b^)
    ref typed = result.as_float64()
    assert_equal(typed[0].value(), 8.0)
    assert_equal(typed[1].value(), 9.0)


# ---------------------------------------------------------------------------
# ExecutionContext dispatch — serial / parallel / auto must agree
# ---------------------------------------------------------------------------


def _assert_add_matches_reference(ctx: ExecutionContext) raises:
    """Run ``add`` over a 100k-element input with ``ctx`` and assert the
    output equals the analytic reference (a[i]+b[i] == 2*i)."""
    comptime N = 100_000
    var a = arange[Int32Type](0, N)
    var b = arange[Int32Type](0, N)
    var result = AddKernel.apply[Int32Type](a, b, ctx)
    assert_equal(len(result), N)
    assert_equal(result[0].value(), 0)
    assert_equal(result[1].value(), 2)
    assert_equal(result[N - 1].value(), 2 * (N - 1))
    # Spot-check across the range to ensure no chunk got skipped or
    # double-summed by the parallel path.
    assert_equal(result[N // 2].value(), 2 * (N // 2))
    assert_equal(result[N // 3].value(), 2 * (N // 3))
    assert_equal(result[N // 4].value(), 2 * (N // 4))


def test_add_dispatch_serial() raises:
    """Forced serial CPU dispatch produces correct results."""
    _assert_add_matches_reference(ExecutionContext.serial())


def test_add_dispatch_parallel_2() raises:
    """Forced 2-worker parallel CPU dispatch produces correct results."""
    _assert_add_matches_reference(ExecutionContext.parallel(2))


def test_add_dispatch_parallel_4() raises:
    """Forced 4-worker parallel CPU dispatch produces correct results."""
    _assert_add_matches_reference(ExecutionContext.parallel(4))


def test_add_dispatch_auto() raises:
    """Auto CPU dispatch (threshold gated) produces correct results.

    With N=100_000 (>= the 32_768 default min_parallel_size) auto should
    pick the parallel path.
    """
    _assert_add_matches_reference(ExecutionContext.auto())


def test_add_dispatch_auto_small() raises:
    """Auto CPU dispatch on a sub-threshold input falls back to serial.

    Below ``min_parallel_size`` auto must not stripe (would dominate the
    compute with dispatch overhead) — output still correct.
    """
    var a = arange[Int32Type](0, 100)
    var b = arange[Int32Type](0, 100)
    var result = AddKernel.apply[Int32Type](a, b, ExecutionContext.auto())
    assert_equal(len(result), 100)
    assert_equal(result[0].value(), 0)
    assert_equal(result[50].value(), 100)
    assert_equal(result[99].value(), 198)


def test_add_dispatch_parallel_small() raises:
    """Forced parallel on a small input still stripes (no threshold check).

    Tests that ``parallel(N)`` is forced — bypasses the size threshold —
    while still producing the correct result.
    """
    var a = arange[Int32Type](0, 100)
    var b = arange[Int32Type](0, 100)
    var result = AddKernel.apply[Int32Type](a, b, ExecutionContext.parallel(8))
    assert_equal(len(result), 100)
    assert_equal(result[0].value(), 0)
    assert_equal(result[50].value(), 100)
    assert_equal(result[99].value(), 198)


def main() raises:
    TestSuite.run[__functions_in_module()]()
