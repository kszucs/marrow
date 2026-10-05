# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`float_to_decimal` — the exact unscaled integer of a double."""

from std.testing import assert_equal, assert_false
from std.utils.numerics import inf, nan

from ..decimal import float_to_decimal


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
