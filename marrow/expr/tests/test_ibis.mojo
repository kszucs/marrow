"""Type-architecture tests for marrow.expr.ibis.

No execution: these verify that expression trees CONSTRUCT with the correct
family conformance (a node statically is a NumericValue / BoolValue / StringValue
by its own declared family), COMPOSE across families, and carry the right output
dtype (each node's `comptime OutType`). Compilation of the generic acceptors is
the family assertion; `out_type_is` checks the dtype by type identity.
"""

from std.testing import assert_true

from marrow.testing import TestSuite
from marrow.dtypes import (
    int32,
    int64,
    float64,
    string,
    Int32Type,
    Int64Type,
    Float64Type,
    StringType,
    DataType,
)
from marrow.expr.ibis import (
    Add,
    Div,
    Less,
    Mod,
    col,
    lit,
    Value,
    NumericValue,
    BoolValue,
    StringValue,
)


# generic acceptors — instantiation is a COMPILE-TIME proof of family membership
def _takes_numeric[N: NumericValue](x: N) -> Bool:
    return True


def _takes_bool[B: BoolValue](x: B) -> Bool:
    return True


def _takes_string[S: StringValue](x: S) -> Bool:
    return True


# check a node's output dtype (its comptime OutType) by type identity
def out_type_is[Want: DataType, V: Value](x: V) -> Bool:
    return V.OutType == Want


# --- family membership: result family follows the op, not the operand --------


def test_arithmetic_is_numeric() raises:
    var a = col("a", int64)
    var b = col("b", int64)
    # explicit node form takes ownership — pass fresh temporaries
    assert_true(_takes_numeric(Add(col("a", int64), col("b", int64))))
    # operator forms borrow, so a reused operand is fine
    assert_true(_takes_numeric(a + b))
    assert_true(_takes_numeric(a - b))
    assert_true(_takes_numeric(a * b))
    assert_true(_takes_numeric(a % b))
    assert_true(_takes_numeric(a**b))
    assert_true(_takes_numeric(-a))
    assert_true(_takes_numeric(a.abs()))
    assert_true(_takes_numeric(a.ceil()))
    assert_true(_takes_numeric(a.floor()))
    assert_true(_takes_numeric(a.round()))
    assert_true(_takes_numeric(a.sign()))


def test_preserving_unary_dtype() raises:
    # ceil/floor/round/sign preserve the operand dtype (like neg/abs)
    assert_true(out_type_is[Int32Type](col("a", int32).ceil()))
    assert_true(out_type_is[Int32Type](col("a", int32).floor()))
    assert_true(out_type_is[Int32Type](col("a", int32).sign()))
    # power widens like the other arithmetic binaries
    assert_true(out_type_is[Int64Type](col("a", int32) ** col("b", int64)))


def test_float_unary_ops() raises:
    var a = col("a", int64)
    assert_true(_takes_numeric(a.exp()))
    assert_true(_takes_numeric(a.ln()))
    assert_true(out_type_is[Float64Type](a.exp()))
    assert_true(out_type_is[Float64Type](a.ln()))


def test_xor_is_bool() raises:
    var a = col("a", int64)
    var b = col("b", int64)
    assert_true(_takes_bool((a < b) ^ (a > b)))


def test_string_predicates() raises:
    var s = col("s", string)
    var t = col("t", string)
    assert_true(_takes_bool(s.endswith(t)))
    assert_true(_takes_bool(s.contains(t)))
    assert_true(_takes_bool(s != t))


def test_reverse_stays_string() raises:
    var s = col("s", string)
    assert_true(_takes_string(s.reverse()))
    assert_true(out_type_is[StringType](s.reverse()))


def test_comparison_is_bool() raises:
    var a = col("a", int64)
    var b = col("b", int64)
    assert_true(_takes_bool(Less(col("a", int64), col("b", int64))))
    assert_true(_takes_bool(a < b))
    assert_true(_takes_bool(a == b))
    assert_true(_takes_bool(a >= b))


def test_cross_family_composition() raises:
    var a = col("a", int64)
    var b = col("b", int64)
    # (a + b) is NumericValue, comparing it yields BoolValue
    assert_true(_takes_bool((a + b) < a))


def test_logical_over_predicates() raises:
    var a = col("a", int64)
    var b = col("b", int64)
    var c = col("c", int64)
    assert_true(_takes_bool((a < b) & (b < c)))
    assert_true(_takes_bool((a < b) | (b < c)))
    assert_true(_takes_bool(~(a < b)))


def test_isnull_is_bool() raises:
    # isnull() is on the base Value — works for any family, always -> BoolValue
    var a = col("a", int64)
    var s = col("s", string)
    assert_true(_takes_bool(a.isnull()))
    assert_true(_takes_bool(s.isnull()))


# --- output dtype from each node's declared OutType --------------------------


def test_integer_widening() raises:
    # highest_precedence: Add(int32, int64) -> int64
    assert_true(out_type_is[Int64Type](Add(col("a", int32), col("b", int64))))
    assert_true(out_type_is[Int32Type](col("a", int32) + col("b", int32)))


def test_modulo_widening() raises:
    assert_true(out_type_is[Int64Type](Mod(col("a", int32), col("b", int64))))


def test_negate_and_abs_preserve_dtype() raises:
    assert_true(out_type_is[Int32Type](-col("a", int32)))
    assert_true(out_type_is[Int32Type](col("a", int32).abs()))


def test_divide_is_float() raises:
    # Divide -> float64 regardless of operand types
    assert_true(out_type_is[Float64Type](Div(col("a", int64), col("b", int64))))


def test_sqrt_is_float() raises:
    var a = col("a", int64)
    assert_true(_takes_numeric(a.sqrt()))
    assert_true(out_type_is[Float64Type](a.sqrt()))


# --- cross-family + string result families -----------------------------------


def test_string_length_is_numeric() raises:
    var s = col("s", string)  # a StringValue
    # length() : StringValue -> NumericValue (int32)
    assert_true(_takes_numeric(s.length()))
    assert_true(out_type_is[Int32Type](s.length()))


def test_startswith_and_equal_are_bool() raises:
    var s = col("s", string)
    var t = col("t", string)
    assert_true(_takes_bool(s.startswith(t)))
    assert_true(_takes_bool(s == t))


def test_upper_lower_stay_string() raises:
    var s = col("s", string)
    # upper()/lower() : StringValue -> StringValue (dtype preserved)
    assert_true(_takes_string(s.upper()))
    assert_true(_takes_string(s.lower()))
    assert_true(out_type_is[StringType](s.upper()))
    # and a string result re-enters the string surface: upper().length() -> int32
    assert_true(_takes_numeric(s.upper().length()))
    assert_true(out_type_is[Int32Type](s.upper().length()))


# --- leaves: literal stores a typed scalar dependent on the dtype family ------


def test_literal_family_and_dtype() raises:
    assert_true(_takes_numeric(lit(2, int64)))
    assert_true(out_type_is[Int64Type](lit(2, int64)))
    assert_true(_takes_string(lit("x", string)))
    assert_true(out_type_is[StringType](lit("x", string)))


def main() raises:
    TestSuite.run[__functions_in_module()]()
