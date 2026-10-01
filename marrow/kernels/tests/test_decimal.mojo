# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Decimal arithmetic, text casts and aggregates.

Every expected value and result type here was produced by `pyarrow.compute`
23.0 (2026-09-27), whose decimal rules these kernels implement. Values are
compared as *text*, so the scale is asserted along with the number: `4.50` and
`4.5` are the same number and different answers.
"""

from std.testing import assert_equal, assert_raises, assert_true

from ...arrays import DynArray, Int32Array
from ...builders import array
from ...dtypes import (
    Decimal128Type,
    DynType,
    decimal32,
    decimal128,
    decimal256,
    int32,
    string,
)
from ...kernels.aggregate import (
    AggKernel,
    DecimalMeanFold,
    DecimalSumFold,
    Fold,
    MaxFold,
)
from ...kernels.cast import cast
from ...kernels.decimal import (
    DecimalAddKernel,
    DecimalDivKernel,
    DecimalMulKernel,
    DecimalPromotion,
    DecimalSubKernel,
)
from ...kernels.groupby import Groups


def _dec(values: List[Optional[String]], dt: DynType) raises -> DynArray:
    """A decimal column spelled as text, through the string -> decimal cast."""
    return cast(array(values), dt)


def _text(values: DynArray) raises -> List[Optional[String]]:
    """`values` rendered through the decimal -> string cast."""
    var s = cast(values, string).as_string().copy()
    var out = List[Optional[String]]()
    for i in range(len(s)):
        if s.is_valid(i):
            out.append(String(s.unsafe_get(UInt(i))))
        else:
            out.append(None)
    return out^


def _assert_text(values: DynArray, expected: List[Optional[String]]) raises:
    var got = _text(values)
    assert_equal(len(got), len(expected))
    for i in range(len(got)):
        if expected[i]:
            assert_true(
                Bool(got[i]), String(t"row {i}: expected a value, got null")
            )
            assert_equal(got[i].value(), expected[i].value())
        else:
            assert_true(not got[i], String(t"row {i}: expected null"))


def _a() raises -> DynArray:
    return _dec(["1.50", None, "-1.25"], decimal128(10, 2))


def _b() raises -> DynArray:
    return _dec(["3.0", "1.0", "0.0"], decimal128(5, 1))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


def test_decimal_promotion_add() raises:
    var promo = DecimalPromotion.add(decimal128(10, 2), decimal128(5, 1))
    assert_true(promo.output == decimal128(11, 2).to_dyn())
    assert_true(promo.right == decimal128(6, 2).to_dyn())
    assert_equal(promo.left_up, 0)
    assert_equal(promo.right_up, 1)


def test_decimal_promotion_multiply() raises:
    var promo = DecimalPromotion.multiply(decimal128(10, 2), decimal128(5, 1))
    assert_true(promo.output == decimal128(16, 3).to_dyn())


def test_decimal_promotion_divide() raises:
    var promo = DecimalPromotion.divide(decimal128(10, 2), decimal128(5, 1))
    assert_true(promo.output == decimal128(16, 7).to_dyn())
    var by_int = DecimalPromotion.divide(decimal128(10, 2), int32)
    assert_true(by_int.output == decimal128(21, 13).to_dyn())


def test_decimal_promotion_integer_operand() raises:
    var promo = DecimalPromotion.add(decimal128(10, 2), int32)
    assert_true(promo.output == decimal128(13, 2).to_dyn())


def test_decimal_promotion_narrow_widths_promote() raises:
    var promo = DecimalPromotion.add(decimal32(5, 1), decimal32(5, 1))
    assert_true(promo.output == decimal128(6, 1).to_dyn())


def test_decimal_promotion_wide() raises:
    var promo = DecimalPromotion.add(decimal256(40, 1), decimal128(10, 2))
    assert_true(promo.output == decimal256(42, 2).to_dyn())


def test_decimal_promotion_precision_overflow() raises:
    with assert_raises(contains="out of range [1, 38]: 39"):
        _ = DecimalPromotion.add(decimal128(38, 2), decimal128(38, 2))
    with assert_raises(contains="out of range [1, 38]: 41"):
        _ = DecimalPromotion.multiply(decimal128(20, 1), decimal128(20, 1))


def test_decimal_promotion_compare_moves_to_256() raises:
    var promo = DecimalPromotion.compare(decimal128(38, 0), decimal128(38, 10))
    assert_true(promo.left == decimal256(48, 10).to_dyn())


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_decimal_add_aligns_scales() raises:
    var r = DecimalAddKernel.dispatch(_a(), _b())
    assert_true(r.dtype() == decimal128(11, 2).to_dyn())
    _assert_text(r, ["4.50", None, "-1.25"])


def test_decimal_subtract() raises:
    _assert_text(
        DecimalSubKernel.dispatch(_a(), _b()), ["-1.50", None, "-1.25"]
    )


def test_decimal_multiply_adds_scales() raises:
    var r = DecimalMulKernel.dispatch(_a(), _b())
    assert_true(r.dtype() == decimal128(16, 3).to_dyn())
    _assert_text(r, ["4.500", None, "0.000"])


def test_decimal_divide_truncates() raises:
    var three = _dec(["3.0", "3.0", "3.0"], decimal128(5, 1))
    var r = DecimalDivKernel.dispatch(_a(), three)
    assert_true(r.dtype() == decimal128(16, 7).to_dyn())
    _assert_text(r, ["0.5000000", None, "-0.4166666"])


def test_decimal_integer_operands() raises:
    var i: DynArray = array([3, 4, 5], int32)
    _assert_text(DecimalAddKernel.dispatch(_a(), i), ["4.50", None, "3.75"])
    _assert_text(DecimalMulKernel.dispatch(_a(), i), ["4.50", None, "-6.25"])
    var q = DecimalDivKernel.dispatch(_a(), i)
    assert_true(q.dtype() == decimal128(21, 13).to_dyn())
    _assert_text(q, ["0.5000000000000", None, "-0.2500000000000"])


def test_decimal256_add() raises:
    var w = _dec(["1.5", None, "10"], decimal256(40, 1))
    var r = DecimalAddKernel.dispatch(w, _a())
    assert_true(r.dtype() == decimal256(42, 2).to_dyn())
    _assert_text(r, ["3.00", None, "8.75"])


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def test_decimal_to_string_keeps_scale() raises:
    _assert_text(
        _dec(["-0.05", "0", "12", None], decimal128(10, 2)),
        ["-0.05", "0.00", "12.00", None],
    )
    _assert_text(_dec(["12"], decimal128(10, 0)), ["12"])


def test_string_to_decimal_grammar() raises:
    _assert_text(
        _dec(
            ["+1.5", ".5", "5.", "1e2", "1.5e-1", "-0", "1.50000"],
            decimal128(10, 2),
        ),
        ["1.50", "0.50", "5.00", "100.00", "0.15", "0.00", "1.50"],
    )


def test_string_to_decimal_rejects() raises:
    with assert_raises(contains="data loss"):
        _ = _dec(["1.555"], decimal128(10, 2))
    with assert_raises(contains="not a decimal number"):
        _ = _dec([" 1.5"], decimal128(10, 2))
    with assert_raises(contains="does not fit its precision"):
        _ = _dec(["12345678901"], decimal128(10, 0))


def test_string_to_decimal_unsafe_nulls() raises:
    var r = cast(array(["1.555", "2.5", "x"]), decimal128(10, 2), safe=False)
    _assert_text(r, [None, "2.50", None])


def test_decimal_text_extremes() raises:
    var widest = String("-")
    for _ in range(38):
        widest += "9"
    _assert_text(_dec([widest], decimal128(38, 0)), [widest])
    _assert_text(_dec(["0.0000000001"], decimal256(76, 10)), ["0.0000000001"])


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def _whole[A: AggKernel](value: DynArray) raises -> DynArray:
    return A.grouped(
        Groups.single(len(value)), A.InArray(value.to_data())
    ).to_dyn()


def _column() raises -> DynArray:
    return _dec(["1.50", None, "-1.25", "0.01"], decimal128(10, 2))


def test_decimal_sum_widens_precision_keeps_scale() raises:
    var r = _whole[Fold[DecimalSumFold, Decimal128Type]](_column())
    assert_true(r.dtype() == decimal128(38, 2).to_dyn())
    _assert_text(r, ["0.26"])


def test_decimal_mean_rounds_half_away_from_zero() raises:
    _assert_text(
        _whole[Fold[DecimalMeanFold, Decimal128Type]](_column()), ["0.09"]
    )
    var pos = _dec(["0.01", "0.02", "0.02"], decimal128(10, 2))
    _assert_text(_whole[Fold[DecimalMeanFold, Decimal128Type]](pos), ["0.02"])
    var neg = _dec(["-0.01", "-0.02", "-0.02"], decimal128(10, 2))
    _assert_text(_whole[Fold[DecimalMeanFold, Decimal128Type]](neg), ["-0.02"])


def test_decimal_max_keeps_type() raises:
    var r = _whole[Fold[MaxFold, Decimal128Type]](_column())
    assert_true(r.dtype() == decimal128(10, 2).to_dyn())
    _assert_text(r, ["1.50"])


def test_decimal_sum_grouped() raises:
    var ids: Int32Array = array([0, 0, 1, 1], int32)
    comptime A = Fold[DecimalSumFold, Decimal128Type]
    var r = A.grouped(Groups(ids^, 2), A.InArray(_column().to_data())).to_dyn()
    _assert_text(r, ["1.50", "-1.24"])
