# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Decimal arithmetic, comparison and aggregates in the runtime lane.

The runtime lane follows Arrow C++ where the comptime lane cannot: an integer
operand meets a decimal as a scale-0 decimal, a float operand turns the whole
operation into `float64`, and a comparison whose aligned scales need more than
38 digits moves to decimal256 instead of raising. Values are
`pyarrow.compute` 23.0's (2026-09-27), except the zero divisor — NULL, as `//`.
"""

from std.testing import assert_equal, assert_true

from ...builders import col, table
from ...logical import DynRelation, DynValue
from ....arrays import DynArray
from ....builders import array
from ....dtypes import DynType, decimal128, decimal256, float64, int64, string
from ....kernels.cast import cast
from ....tabular import record_batch


def _dec(values: List[Optional[String]], dt: DynType) raises -> DynArray:
    return cast(array(values), dt)


def _prices() raises -> DynRelation:
    return table(
        record_batch(
            [
                _dec(["1.50", None, "-1.25", "0.01"], decimal128(10, 2)),
                _dec(["3.0", "1.0", "0.0", "2.0"], decimal128(5, 1)),
                array([3, 4, 5, 6], int64).to_dyn(),
                array([1.0, 2.0, 3.0, 4.0], float64).to_dyn(),
                _dec(["1", "1", "1", "1"], decimal128(38, 0)),
            ],
            names=["a", "b", "i", "f", "w"],
        )
    )


def _assert_text(column: DynArray, expected: List[Optional[String]]) raises:
    var s = cast(column, string).as_string().copy()
    assert_equal(len(s), len(expected))
    for i in range(len(s)):
        if expected[i]:
            assert_true(s.is_valid(i), String(t"row {i}: got null"))
            assert_equal(String(s.unsafe_get(UInt(i))), expected[i].value())
        else:
            assert_true(not s.is_valid(i), String(t"row {i}: expected null"))


def test_runtime_decimal_arithmetic() raises:
    var out = (
        _prices()
        .project(
            ["add", "mul", "div"],
            [col("a") + col("b"), col("a") * col("b"), col("a") / col("b")],
        )
        .execute()
    )
    assert_true(out.columns[0].dtype() == DynType(decimal128(11, 2)))
    assert_true(out.columns[1].dtype() == DynType(decimal128(16, 3)))
    assert_true(out.columns[2].dtype() == DynType(decimal128(16, 7)))
    _assert_text(out.columns[0], ["4.50", None, "-1.25", "2.01"])
    _assert_text(out.columns[1], ["4.500", None, "0.000", "0.020"])
    _assert_text(out.columns[2], ["0.5000000", None, None, "0.0050000"])


def test_runtime_decimal_integer_and_float_operands() raises:
    var out = (
        _prices()
        .project(["di", "df"], [col("a") * col("i"), col("a") + col("f")])
        .execute()
    )
    assert_true(out.columns[0].dtype() == DynType(decimal128(30, 2)))
    _assert_text(out.columns[0], ["4.50", None, "-6.25", "0.06"])
    assert_true(
        out.columns[1].as_float64()
        == array([2.5, None, 1.75, 0.01 + 4.0], float64)
    )


def test_runtime_decimal_compare() raises:
    """`a >= w` aligns a `decimal128(10, 2)` with a `decimal128(38, 0)` at
    40 digits, which Arrow compares in decimal256."""
    var out = (
        _prices()
        .project(
            ["lt", "wide"],
            [col("a") < col("b"), col("a") >= col("w")],
        )
        .execute()
    )
    assert_true(out.columns[0].as_bool() == array([True, None, True, True]))
    assert_true(out.columns[1].as_bool() == array([True, None, False, False]))


def test_runtime_decimal_aggregates() raises:
    var out = (
        _prices()
        .aggregate(
            [col("a").sum(), col("a").mean(), col("a").max()],
            List[DynValue](),
        )
        .execute()
    )
    assert_true(out.columns[0].dtype() == DynType(decimal128(38, 2)))
    _assert_text(out.columns[0], ["0.26"])
    _assert_text(out.columns[1], ["0.09"])
    _assert_text(out.columns[2], ["1.50"])


def test_runtime_decimal256_sum() raises:
    var out = (
        table(
            record_batch(
                [_dec(["1.5", "2.25", None], decimal256(40, 2))], names=["d"]
            )
        )
        .aggregate([col("d").sum()], List[DynValue]())
        .execute()
    )
    assert_true(out.columns[0].dtype() == DynType(decimal256(76, 2)))
    _assert_text(out.columns[0], ["3.75"])
