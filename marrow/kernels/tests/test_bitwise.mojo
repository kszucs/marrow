# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bitwise and shift kernels, against PyArrow's and Arrow
C++'s answers."""

from std.testing import assert_raises, assert_true

from ...arrays import DynArray
from ...builders import array
from ...dtypes import float64, int8, int16, int64, uint8
from ...kernels.bitwise import (
    BitAndKernel,
    BitNotKernel,
    BitOrKernel,
    BitXorKernel,
    ShiftLeftKernel,
    ShiftRightKernel,
)


def test_bitwise_follows_pyarrow() raises:
    var x = array([15, -1, None, 5, -128], int8)
    var y = array([60, 7, 1, None, 1], int8)
    assert_true(BitAndKernel.apply(x, y) == array([12, 7, None, None, 0], int8))
    assert_true(
        BitOrKernel.apply(x, y) == array([63, -1, None, None, -127], int8)
    )
    assert_true(
        BitXorKernel.apply(x, y) == array([51, -8, None, None, -127], int8)
    )
    assert_true(BitNotKernel.apply(x) == array([-16, 0, None, -6, 127], int8))
    assert_true(
        BitNotKernel.apply(array([0, 85, 255, None], uint8))
        == array([255, 170, 0, None], uint8)
    )


def test_shift_left_follows_arrow() raises:
    """An amount below zero, or not below the precision (7 for `int8`, 8 for
    `uint8`), leaves the value as it is; a shift into the sign bit wraps."""
    var ones = array([1, 1, 1, 1, 1, 1], int8)
    assert_true(
        ShiftLeftKernel.apply(ones, array([0, 1, 6, 7, 8, -1], int8))
        == array([1, 2, 64, 1, 1, 1], int8)
    )
    var unsigned = array([1, 1, 1, 1, 1, 1], uint8)
    assert_true(
        ShiftLeftKernel.apply(unsigned, array([0, 1, 6, 7, 8, 255], uint8))
        == array([1, 2, 64, 128, 1, 1], uint8)
    )
    assert_true(
        ShiftLeftKernel.apply(
            array([1, 1, 1], int64), array([62, 63, 64], int64)
        )
        == array([4611686018427387904, 1, 1], int64)
    )
    assert_true(
        ShiftLeftKernel.apply(array([-1, 64, 3], int8), array([1, 1, 6], int8))
        == array([-2, -128, -64], int8)
    )


def test_shift_right_is_arithmetic_when_signed() raises:
    assert_true(
        ShiftRightKernel.apply(
            array([-128, -128, -128, -128, -1], int8),
            array([1, 6, 7, 8, 3], int8),
        )
        == array([-64, -2, -128, -128, -1], int8)
    )
    assert_true(
        ShiftRightKernel.apply(
            array([255, 255, 255, 255, 7], uint8), array([1, 6, 7, 8, 3], uint8)
        )
        == array([127, 3, 1, 255, 0], uint8)
    )


def test_bitwise_dispatch_rejects_non_integers() raises:
    var floats: DynArray = array([1.0], float64)
    with assert_raises(contains="unsupported dtype"):
        _ = BitAndKernel.dispatch(floats, floats)
    with assert_raises(contains="unsupported dtype"):
        _ = ShiftLeftKernel.dispatch(floats, floats)
    with assert_raises(contains="unsupported dtype"):
        _ = BitNotKernel.dispatch(floats)
    var bools: DynArray = array([True])
    with assert_raises(contains="TypeError: dtype is not numeric"):
        _ = BitOrKernel.dispatch(bools, bools)
    with assert_raises(contains="dtype mismatch"):
        _ = BitXorKernel.dispatch(array([1], int8), array([1], int16))
