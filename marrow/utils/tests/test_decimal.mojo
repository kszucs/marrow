# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`float_to_decimal` — the exact unscaled integer of a double — and decimal
text."""

from std.testing import assert_equal, assert_false, assert_raises
from std.utils.numerics import inf, nan

from ..decimal import (
    decimal_bound,
    float_to_decimal,
    format_decimal,
    parse_decimal,
)


def test_exact_float_to_decimal_keeps_every_digit() raises:
    # Past 2^53 / 10^4, where a float64 product would round.
    assert_equal(
        float_to_decimal(14411518807587.0, 4, 20).value(),
        Int256(144115188075870000),
    )
    # 0.1 is not 1/10: at scale 50 its binary expansion shows.
    assert_equal(
        float_to_decimal(0.1, 50, 76).value(),
        Int256(10000000000000000555111512312578270211815834045410),
    )
    assert_equal(
        float_to_decimal(1e38, 0, 38).value(),
        Int256(99999999999999997748809823456034029568),
    )
    assert_equal(float_to_decimal(-0.375, 4, 10).value(), Int256(-3750))


def test_exact_float_to_decimal_ties_to_even() raises:
    var cases: List[Tuple[Float64, Int]] = [
        (0.5, 0),
        (1.5, 2),
        (2.5, 2),
        (-0.5, 0),
        (-2.5, -2),
    ]
    for c in cases:
        assert_equal(float_to_decimal(c[0], 0, 10).value(), Int256(c[1]))
    assert_equal(float_to_decimal(0.125, 2, 10).value(), Int256(12))
    assert_equal(float_to_decimal(0.375, 2, 10).value(), Int256(38))


def test_exact_float_to_decimal_tiny_values_round_to_zero() raises:
    assert_equal(float_to_decimal(5e-324, 4, 10).value(), Int256(0))
    assert_equal(float_to_decimal(-0.0, 4, 10).value(), Int256(0))
    assert_equal(float_to_decimal(0.001, 2, 7).value(), Int256(0))


def test_exact_float_to_decimal_refuses_what_does_not_fit() raises:
    assert_false(float_to_decimal(1000.0, 2, 5))
    assert_false(float_to_decimal(0.05, 7, 5))
    assert_false(float_to_decimal(nan[DType.float64](), 2, 10))
    assert_false(float_to_decimal(inf[DType.float64](), 2, 10))
    assert_false(float_to_decimal(-inf[DType.float64](), 2, 10))
    # 99999.995 is just below the tie, so it rounds down and fits.
    assert_equal(float_to_decimal(99999.995, 2, 7).value(), Int256(9999999))


def test_format_decimal_keeps_every_scale_digit() raises:
    assert_equal(format_decimal(Int64(150), 2), "1.50")
    assert_equal(format_decimal(Int64(-5), 2), "-0.05")
    assert_equal(format_decimal(Int64(42), 0), "42")
    # The most negative value: its magnitude does not fit, its digits do.
    assert_equal(format_decimal(Int8(-128), 1), "-12.8")


def test_parse_decimal_rescales_exactly() raises:
    var bound = decimal_bound[DType.int64](10)
    assert_equal(parse_decimal[DType.int64]("+1.5", 10, bound, 2), 150)
    assert_equal(parse_decimal[DType.int64](".5", 10, bound, 2), 50)
    assert_equal(parse_decimal[DType.int64]("1.5e-1", 10, bound, 2), 15)
    assert_equal(parse_decimal[DType.int64]("-12", 10, bound, 0), -12)
    with assert_raises(contains="data loss"):
        _ = parse_decimal[DType.int64]("1.555", 10, bound, 2)
    with assert_raises(contains="not a decimal number"):
        _ = parse_decimal[DType.int64](" 1", 10, bound, 0)
    with assert_raises(contains="does not fit its precision"):
        _ = parse_decimal[DType.int64](
            "100000", 10, decimal_bound[DType.int64](5), 0
        )


def test_decimal_bound() raises:
    assert_equal(decimal_bound[DType.int64](3), 1000)
    assert_equal(decimal_bound[DType.int64](0), 0)
