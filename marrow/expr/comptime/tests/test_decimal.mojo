# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The fused decimal nodes: arithmetic, comparison, casts and aggregates.

Results are asserted as text, through the fused `DecimalToString` node, so the
scale is part of every answer. The arithmetic values and result types are
`pyarrow.compute` 23.0's (2026-09-27); the one deliberate divergence is the zero
divisor, which is NULL here as it is for `//`.
"""

from std.testing import assert_equal, assert_raises, assert_true

from ...builders import col, lit, table
from ...logical import DynRelation, DynValue
from ....arrays import DynArray
from ....builders import array
from ....dtypes import (
    DynType,
    decimal128,
    decimal256,
    float64,
    int64,
    string,
)
from ....kernels.cast import cast
from ....tabular import record_batch


def _dec(values: List[Optional[String]], dt: DynType) raises -> DynArray:
    return cast(array(values), dt)


def _sales() raises -> DynRelation:
    """`a` is `decimal128(10, 2)`, `b` a `decimal128(5, 1)` holding a zero,
    `f` a float column and `s` its text, `g` a grouping key."""
    return table(
        record_batch(
            [
                _dec(["1.50", None, "-1.25", "0.01"], decimal128(10, 2)),
                _dec(["3.0", "1.0", "0.0", "2.0"], decimal128(5, 1)),
                array([1.5, 2.25, None, -0.5], float64).to_dyn(),
                array(["1.5", "x", None, "-0.5"]).to_dyn(),
                array([1, 1, 2, 2], int64).to_dyn(),
            ],
            names=["a", "b", "f", "s", "g"],
        )
    )


def _text(column: DynArray) raises -> List[Optional[String]]:
    var s = column.as_string().copy()
    var out = List[Optional[String]]()
    for i in range(len(s)):
        if s.is_valid(i):
            out.append(String(s.unsafe_get(UInt(i))))
        else:
            out.append(None)
    return out^


def _assert_text(column: DynArray, expected: List[Optional[String]]) raises:
    var got = _text(column)
    assert_equal(len(got), len(expected))
    for i in range(len(got)):
        if expected[i]:
            assert_true(Bool(got[i]), String(t"row {i}: got null"))
            assert_equal(got[i].value(), expected[i].value())
        else:
            assert_true(not got[i], String(t"row {i}: expected null"))


comptime A = decimal128(10, 2)
comptime B = decimal128(5, 1)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_fused_decimal_add_aligns_scales() raises:
    var e = col("a", A) + col("b", B)
    var plan = _sales().project(["x", "t"], [e, e.cast(string)])
    var out = plan.execute()
    assert_true(out.columns[0].dtype() == DynType(decimal128(11, 2)))
    assert_true(plan.schema() == out.schema)
    _assert_text(out.columns[1], ["4.50", None, "-1.25", "2.01"])


def test_fused_decimal_subtract_and_multiply() raises:
    var out = (
        _sales()
        .project(
            ["d", "p"],
            [
                (col("a", A) - col("b", B)).cast(string),
                (col("a", A) * col("b", B)).cast(string),
            ],
        )
        .execute()
    )
    _assert_text(out.columns[0], ["-1.50", None, "-1.25", "-1.99"])
    _assert_text(out.columns[1], ["4.500", None, "0.000", "0.020"])


def test_fused_decimal_multiply_type() raises:
    var plan = _sales().project(["p"], [col("a", A) * col("b", B)])
    assert_true(plan.schema().fields[0].dtype == DynType(decimal128(16, 3)))


def test_fused_decimal_divide_truncates_and_nulls_zero() raises:
    var plan = _sales().project(["q"], [col("a", A) / col("b", B)])
    assert_true(plan.schema().fields[0].dtype == DynType(decimal128(16, 7)))
    var out = (
        _sales()
        .project(["q"], [(col("a", A) / col("b", B)).cast(string)])
        .execute()
    )
    _assert_text(out.columns[0], ["0.5000000", None, None, "0.0050000"])


def test_fused_decimal_literal_operand() raises:
    var out = (
        _sales()
        .project(["x"], [(col("a", A) * lit(3, decimal128(1, 0))).cast(string)])
        .execute()
    )
    _assert_text(out.columns[0], ["4.50", None, "-3.75", "0.03"])


def test_fused_decimal_scalar_shape() raises:
    var out = (
        _sales()
        .project(
            ["x"],
            [(lit(150, A) + lit(25, B)).cast(string)],
        )
        .execute()
    )
    _assert_text(out.columns[0], ["4.00", "4.00", "4.00", "4.00"])


def test_fused_decimal256_add() raises:
    var w = lit(15, decimal256(40, 1))
    var plan = _sales().project(["x"], [(col("a", A) + w).cast(string)])
    _assert_text(plan.execute().columns[0], ["3.00", None, "0.25", "1.51"])


def test_fused_decimal_precision_overflow_raises() raises:
    """Raised when the plan is built — the projection's schema needs the
    result type — not on the first batch."""
    with assert_raises(contains="out of range [1, 38]: 39"):
        _ = _sales().project(["x"], [col("a", A) + lit(1, decimal128(38, 2))])


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_fused_decimal_compare_across_scales() raises:
    var out = (
        _sales()
        .project(
            ["eq", "lt"],
            [
                col("a", A) == lit(15, B),
                col("a", A) < col("b", B),
            ],
        )
        .execute()
    )
    assert_true(out.columns[0].as_bool() == array([True, None, False, False]))
    assert_true(out.columns[1].as_bool() == array([True, None, True, True]))


def test_fused_decimal_filter() raises:
    var out = (
        _sales()
        .filter(col("a", A) > lit(0, decimal128(1, 0)))
        .project(["a"], [col("a", A).cast(string)])
        .execute()
    )
    _assert_text(out.columns[0], ["1.50", "0.01"])


def test_fused_decimal_compare_beyond_the_register_raises() raises:
    """Raised when the plan is built, as the arithmetic nodes' precision
    errors are."""
    with assert_raises(contains="runtime lane"):
        _ = _sales().project(
            ["x"], [lit(1, decimal128(38, 0)) < lit(1, decimal128(38, 10))]
        )


# ---------------------------------------------------------------------------
# Casts
# ---------------------------------------------------------------------------


def test_fused_decimal_rescale() raises:
    var out = (
        _sales()
        .project(
            ["up", "down"],
            [
                col("a", A).cast(decimal128(12, 4)).cast(string),
                col("a", A).cast(decimal128(10, 1)).cast(string),
            ],
        )
        .execute()
    )
    _assert_text(out.columns[0], ["1.5000", None, "-1.2500", "0.0100"])
    _assert_text(out.columns[1], ["1.5", None, "-1.2", "0.0"])


def test_fused_decimal_to_numeric() raises:
    var out = (
        _sales()
        .project(
            ["f", "i"],
            [col("a", A).cast(float64), col("a", A).cast(int64)],
        )
        .execute()
    )
    assert_true(
        out.columns[0].as_float64() == array([1.5, None, -1.25, 0.01], float64)
    )
    assert_true(out.columns[1].as_int64() == array([1, None, -1, 0], int64))


def test_fused_numeric_to_decimal() raises:
    var out = (
        _sales()
        .project(
            ["f", "g"],
            [
                col("f", float64).cast(A).cast(string),
                col("g", int64).cast(A).cast(string),
            ],
        )
        .execute()
    )
    _assert_text(out.columns[0], ["1.50", "2.25", None, "-0.50"])
    _assert_text(out.columns[1], ["1.00", "1.00", "2.00", "2.00"])


def test_fused_string_to_decimal_nulls_what_does_not_parse() raises:
    var out = (
        _sales()
        .project(["d"], [col("s", string).cast(A).cast(string)])
        .execute()
    )
    _assert_text(out.columns[0], ["1.50", None, None, "-0.50"])


def test_fused_decimal_checked_cast_is_refused() raises:
    with assert_raises(contains="no checked cast"):
        _ = col("a", A).cast(decimal128(12, 4), safe=True)


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_fused_decimal_sum_and_mean() raises:
    var plan = _sales().aggregate(
        [
            col("a", A).sum().alias("s"),
            col("a", A).mean().alias("m"),
            col("a", A).min().alias("lo"),
            col("a", A).max().alias("hi"),
            col("a", A).count().alias("n"),
        ],
        List[DynValue](),
    )
    var out = plan.execute()
    assert_true(plan.schema() == out.schema)
    assert_true(out.columns[0].dtype() == DynType(decimal128(38, 2)))
    assert_true(out.columns[2].dtype() == DynType(A))
    var text = (
        table(out^)
        .project(
            ["s", "m", "lo", "hi"],
            [
                col("s", decimal128(38, 2)).cast(string),
                col("m", decimal128(38, 2)).cast(string),
                col("lo", A).cast(string),
                col("hi", A).cast(string),
            ],
        )
        .execute()
    )
    _assert_text(text.columns[0], ["0.26"])
    _assert_text(text.columns[1], ["0.09"])
    _assert_text(text.columns[2], ["-1.25"])
    _assert_text(text.columns[3], ["1.50"])


def test_fused_decimal_sum_grouped_and_of_an_expression() raises:
    var plan = _sales().aggregate(
        [(col("a", A) * col("b", B)).sum().alias("s")],
        [col("g", int64)],
    )
    var out = plan.execute()
    assert_true(out.columns[1].dtype() == DynType(decimal128(38, 3)))
    var text = (
        table(out^)
        .project(["s"], [col("s", decimal128(38, 3)).cast(string)])
        .execute()
    )
    _assert_text(text.columns[0], ["4.500", "0.020"])
