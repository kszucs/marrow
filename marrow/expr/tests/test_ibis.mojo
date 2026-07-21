"""Type-architecture tests for marrow.expr.ibis.

No execution: these verify that expression trees CONSTRUCT with the correct
family conformance (a node statically is a NumericValue / BoolValue based on its
kernel's result family), COMPOSE across families, and carry the right output
dtype (from the kernel's promotion rule). Compilation is the assertion; `write_to`
and `dtype_of` check structure and dtype.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite
from marrow.dtypes import (
    int32,
    int64,
    float64,
    string,
    Int32Type,
    Int64Type,
    Float64Type,
    DataType,
)
from marrow.expr.ibis import (
    Add,
    Div,
    Less,
    Greater,
    Equal,
    And,
    Or,
    Neg,
    Not,
    col,
    lit,
    Value,
    NumericValue,
    BoolValue,
    StringValue,
)


# generic acceptors — compile-time proof of family membership
def _takes_numeric[N: NumericValue](x: N) -> String:
    return String(x)


def _takes_bool[B: BoolValue](x: B) -> String:
    return String(x)


# check a node's output dtype (its comptime OutType) by type identity
def out_type_is[Want: DataType, V: Value](x: V) -> Bool:
    return V.OutType == Want


def test_arithmetic_is_numeric() raises:
    var a = col("a", int64)
    var b = col("b", int64)
    assert_equal(_takes_numeric(Add(a, b)), "add(Col[a], Col[b])")
    assert_equal(_takes_numeric(a + b), "add(Col[a], Col[b])")
    assert_equal(_takes_numeric(-a), "negate(Col[a])")


def test_comparison_is_bool() raises:
    var a = col("a", int64)
    var b = col("b", int64)
    assert_equal(_takes_bool(Less(a, b)), "less(Col[a], Col[b])")
    assert_equal(_takes_bool(a < b), "less(Col[a], Col[b])")
    assert_equal(_takes_bool(a == b), "equal(Col[a], Col[b])")


def test_cross_family_composition() raises:
    var a = col("a", int64)
    var b = col("b", int64)
    assert_equal(
        _takes_bool((a + b) < a), "less(add(Col[a], Col[b]), Col[a])"
    )


def test_logical_over_predicates() raises:
    var a = col("a", int64)
    var b = col("b", int64)
    var c = col("c", int64)
    assert_equal(
        _takes_bool((a < b) & (b < c)),
        "and(less(Col[a], Col[b]), less(Col[b], Col[c]))",
    )
    assert_equal(_takes_bool(~(a < b)), "not(less(Col[a], Col[b]))")


# --- promotion rules: output dtype from the kernel's rule -------------------


def test_integer_widening() raises:
    # highest_precedence: Add(int32, int64) -> int64
    assert_true(out_type_is[Int64Type](Add(col("a", int32), col("b", int64))))
    assert_true(out_type_is[Int32Type](col("a", int32) + col("b", int32)))


def test_divide_is_float() raises:
    # float_result: Divide -> float64 regardless of operand types
    assert_true(out_type_is[Float64Type](Div(col("a", int64), col("b", int64))))


# --- cross-family kernels: string operand, non-string result ---------------


def test_string_length_is_numeric() raises:
    var s = col("s", string)  # a StringValue
    # length() : StringValue -> NumericValue (int32)
    assert_equal(_takes_numeric(s.length()), "length(StrCol[s])")
    assert_true(out_type_is[Int32Type](s.length()))


def test_startswith_is_bool() raises:
    var s = col("s", string)
    var t = col("t", string)
    # startswith() : StringValue x StringValue -> BoolValue
    assert_equal(
        _takes_bool(s.startswith(t)), "startswith(StrCol[s], StrCol[t])"
    )


def main() raises:
    TestSuite.run[__functions_in_module()]()
