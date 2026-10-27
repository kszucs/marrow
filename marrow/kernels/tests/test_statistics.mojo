# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`BoundsKernel`: a column's least and greatest valid value, NaN skipped."""

from std.math import isnan, nan
from std.testing import assert_equal, assert_false, assert_true

from ...arrays import PrimitiveArray
from ...builders import (
    Decimal128Builder,
    Int32Builder,
    Int64Builder,
    TimestampBuilder,
    array,
)
from ...dtypes import (
    PrimitiveType,
    decimal128,
    float64,
    int32,
    int64,
    second,
    timestamp,
)
from ...kernels.statistics import BoundsKernel


def _reference[
    T: PrimitiveType
](values: PrimitiveArray[T]) -> Tuple[
    Optional[Scalar[T.native]], Optional[Scalar[T.native]]
]:
    """The bounds by a scalar scan of the valid, non-NaN values."""
    var least: Optional[Scalar[T.native]] = None
    var greatest: Optional[Scalar[T.native]] = None
    for i in range(len(values)):
        if values.is_valid(i):
            var v = values.unsafe_get(i)
            var skip = False
            comptime if T.native.is_floating_point():
                skip = isnan(v)
            if not skip:
                if not least or v < least.value():
                    least = v
                if not greatest or v > greatest.value():
                    greatest = v
    return (least, greatest)


def _assert_bounds[T: PrimitiveType](values: PrimitiveArray[T]) raises:
    var got = BoundsKernel.apply(values)
    var expected = _reference(values)
    assert_equal(Bool(got[0]), Bool(expected[0]), "least present")
    assert_equal(Bool(got[1]), Bool(expected[1]), "greatest present")
    if expected[0]:
        assert_equal(got[0].value(), expected[0].value(), "least")
        assert_equal(got[1].value(), expected[1].value(), "greatest")


def test_bounds_skip_nulls_and_nan() raises:
    var n = nan[DType.float64]()
    var values = array([n, 3.0, None, -1.0, n], float64)
    var got = BoundsKernel.apply(values)
    assert_equal(got[0].value(), -1.0)
    assert_equal(got[1].value(), 3.0)


def test_bounds_all_nan_or_all_null_is_absent() raises:
    var n = nan[DType.float64]()
    var all_nan = BoundsKernel.apply(array([n, n], float64))
    assert_false(Bool(all_nan[0]))
    assert_false(Bool(all_nan[1]))
    var nulls = Int64Builder(3)
    nulls.append_nulls(3)
    var all_null = BoundsKernel.apply(nulls.finish())
    assert_false(Bool(all_null[0]))
    assert_false(Bool(all_null[1]))
    var empty = BoundsKernel.apply(array(int64))
    assert_false(Bool(empty[0]))
    assert_false(Bool(empty[1]))


def test_bounds_signed_zero_equals_zero() raises:
    # Which zero bounds `-0.0` and `0.0` is the target's choice: compare by
    # value.
    var z = Float64(0)
    var nz = -z
    for zeros in [array([z, nz], float64), array([nz, z], float64)]:
        var got = BoundsKernel.apply(zeros)
        assert_equal(got[0].value(), 0.0)
        assert_equal(got[1].value(), 0.0)


def test_bounds_integer_extremes() raises:
    """A value equal to a fold's identity is still a bound."""
    var top = Int64Builder(1)
    top.append(Int64.MAX)
    var top_bounds = BoundsKernel.apply(top.finish())
    assert_equal(top_bounds[0].value(), Int64.MAX)
    assert_equal(top_bounds[1].value(), Int64.MAX)
    var low = Int64Builder(2)
    low.append(Int64.MIN)
    low.append(5)
    var bottom = BoundsKernel.apply(low.finish())
    assert_equal(bottom[0].value(), Int64.MIN)
    assert_equal(bottom[1].value(), 5)


def test_bounds_sliced_with_nulls() raises:
    var b = Int32Builder(100)
    for i in range(100):
        if i % 3 == 0:
            b.append_null()
        else:
            b.append(Int32((i * 37) % 101 - 50))
    var values = b.finish()
    _assert_bounds(values.slice(1))
    _assert_bounds(values.slice(1, 7))
    _assert_bounds(values.slice(3, 1))  # one null row: no bounds


def test_bounds_decimal_and_temporal() raises:
    var d = Decimal128Builder(decimal128(38, 2), 6)
    for v in [Int128(250), Int128(-12), Int128(0), Int128(99999)]:
        d.append(v)
    d.append_null()
    d.append(Int128(-77777))
    _assert_bounds(d.finish())

    var t = TimestampBuilder(timestamp(second, "UTC"), 5)
    t.append(Int64(3000))
    t.append_null()
    t.append(Int64(1000))
    t.append(Int64(2000))
    t.append(Int64(-500))
    _assert_bounds(t.finish())
