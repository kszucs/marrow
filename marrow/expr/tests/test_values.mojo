"""Tests for marrow.expr.values — the comptime-typed expression system.

Two halves:
  * Type architecture (no execution): expression trees CONSTRUCT with the right
    family conformance (a node statically is a NumericValue / BoolValue /
    StringValue / ListValue) and carry the right output dtype (`comptime
    OutType`). Compilation of the generic acceptors is the family assertion;
    `out_type_is` checks the dtype by type identity.
  * Execution: the numeric family fuses to a single vectorized pass, the string
    family materializes through the real kernels, and `AnyValue` erases + runs
    any node.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite
from marrow.builders import array
from marrow.arrays import AnyArray
from marrow.dtypes import (
    int32,
    int64,
    float64,
    string,
    list_,
    Int64Type,
    Int32Type,
    Float64Type,
    StringType,
    BoolType,
    DataType,
)
from marrow.tabular import record_batch, RecordBatch
from marrow.expr.values import (
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
    ListValue,
    AnyValue,
    Table,
)


struct Orders:
    var a: Int64Type
    var b: Int64Type
    var c: Int32Type


# generic acceptors — instantiation is a COMPILE-TIME proof of family membership
def _takes_numeric[N: NumericValue](x: N) -> Bool:
    return True


def _takes_bool[B: BoolValue](x: B) -> Bool:
    return True


def _takes_string[S: StringValue](x: S) -> Bool:
    return True


def _takes_list[L: ListValue](x: L) -> Bool:
    return True


# check a node's output dtype (its comptime OutType) by type identity
def out_type_is[Want: DataType, V: Value](x: V) -> Bool:
    return V.OutType == Want


def _batch() raises -> RecordBatch:
    return record_batch(
        [
            array([1, 2, 3, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
            array([2, 2, 2, 2], int32).copy(),
        ],
        names=["a", "b", "c"],
    )


# execute() returns PrimitiveArray[symbolic OutType]; compare value-wise via AnyArray
def _eq[V: NumericValue](x: V, expected: AnyArray) raises -> Bool:
    return x.execute(_batch()).to_any() == expected


# ===========================================================================
# Type architecture — family membership + output dtype (no execution)
# ===========================================================================


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
    # power always yields float64 (Power / numpy semantics)
    assert_true(out_type_is[Float64Type](col("a", int32) ** col("b", int64)))


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


def test_list_column_is_list() raises:
    var l = col("l", list_(int64))
    assert_true(_takes_list(l))


def test_list_length_is_int32() raises:
    var l = col("l", list_(int64))
    assert_true(out_type_is[Int32Type](l.length()))


def test_list_contains_is_bool() raises:
    var l = col("l", list_(int64))
    assert_true(_takes_bool(l.contains(col("x", int64))))


def test_sum_widens_to_64bit() raises:
    assert_true(out_type_is[Int64Type](col("a", int32).sum()))
    assert_true(out_type_is[Int64Type](col("a", int64).sum()))
    assert_true(out_type_is[Float64Type](col("a", float64).sum()))


def test_mean_is_float() raises:
    assert_true(out_type_is[Float64Type](col("a", int64).mean()))


def test_min_max_preserve_dtype() raises:
    assert_true(out_type_is[Int32Type](col("a", int32).min()))
    assert_true(out_type_is[Int32Type](col("a", int32).max()))
    assert_true(out_type_is[Int64Type](col("a", int64).max()))


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
    assert_true(_takes_bool((a + b) < a))


def test_logical_over_predicates() raises:
    var a = col("a", int64)
    var b = col("b", int64)
    var c = col("c", int64)
    assert_true(_takes_bool((a < b) & (b < c)))
    assert_true(_takes_bool((a < b) | (b < c)))
    assert_true(_takes_bool(~(a < b)))


def test_isnull_is_bool() raises:
    var a = col("a", int64)
    var s = col("s", string)
    assert_true(_takes_bool(a.isnull()))
    assert_true(_takes_bool(s.isnull()))


def test_integer_widening() raises:
    assert_true(out_type_is[Int64Type](Add(col("a", int32), col("b", int64))))
    assert_true(out_type_is[Int32Type](col("a", int32) + col("b", int32)))


def test_modulo_widening() raises:
    assert_true(out_type_is[Int64Type](Mod(col("a", int32), col("b", int64))))


def test_negate_and_abs_preserve_dtype() raises:
    assert_true(out_type_is[Int32Type](-col("a", int32)))
    assert_true(out_type_is[Int32Type](col("a", int32).abs()))


def test_divide_is_float_type() raises:
    assert_true(out_type_is[Float64Type](Div(col("a", int64), col("b", int64))))


def test_sqrt_is_float() raises:
    var a = col("a", int64)
    assert_true(_takes_numeric(a.sqrt()))
    assert_true(out_type_is[Float64Type](a.sqrt()))


def test_string_length_is_int32() raises:
    var s = col("s", string)
    assert_true(out_type_is[Int32Type](s.length()))


def test_startswith_and_equal_are_bool() raises:
    var s = col("s", string)
    var t = col("t", string)
    assert_true(_takes_bool(s.startswith(t)))
    assert_true(_takes_bool(s == t))


def test_upper_lower_stay_string() raises:
    var s = col("s", string)
    assert_true(_takes_string(s.upper()))
    assert_true(_takes_string(s.lower()))
    assert_true(out_type_is[StringType](s.upper()))
    assert_true(out_type_is[Int32Type](s.upper().length()))


def test_literal_family_and_dtype() raises:
    assert_true(_takes_numeric(lit(2, int64)))
    assert_true(out_type_is[Int64Type](lit(2, int64)))
    assert_true(_takes_string(lit("x", string)))
    assert_true(out_type_is[StringType](lit("x", string)))


def test_numeric_predicates_are_bool() raises:
    var a = col("a", float64)
    assert_true(_takes_bool(a.isnan()))
    assert_true(_takes_bool(a.isinf()))
    assert_true(_takes_bool(a.notnull()))


def test_math_unary_is_float() raises:
    var a = col("a", int64)
    assert_true(_takes_numeric(a.sin()))
    assert_true(out_type_is[Float64Type](a.cos()))
    assert_true(out_type_is[Float64Type](a.log10()))
    assert_true(out_type_is[Int32Type](col("a", int32).trunc()))


def test_string_transforms_stay_string() raises:
    var s = col("s", string)
    assert_true(_takes_string(s.strip()))
    assert_true(_takes_string(s.lstrip()))
    assert_true(_takes_string(s.rstrip()))
    assert_true(_takes_string(s.capitalize()))
    assert_true(out_type_is[StringType](s.capitalize()))


# ===========================================================================
# Numeric execution — fused single-pass
# ===========================================================================


def test_add_columns() raises:
    assert_true(
        _eq(col("a", int64) + col("b", int64), array([11, 22, 33, 44], int64))
    )


def test_fused_chain_single_pass() raises:
    # (a + b) * c  — one fused vectorize loop, no intermediate arrays
    var expr = (col("a", int64) + col("b", int64)) * col("c", int32)
    assert_true(_eq(expr, array([22, 44, 66, 88], int64)))


def test_literal_broadcast() raises:
    assert_true(
        _eq(col("a", int64) * lit(10, int64), array([10, 20, 30, 40], int64))
    )


def test_widening_out_dtype() raises:
    var expr = col("a", int64) + col("c", int32)
    assert_true(out_type_is[Int64Type](expr))
    assert_true(_eq(expr, array([3, 4, 5, 6], int64)))


def test_divide_is_float() raises:
    var expr = col("b", int64) / col("a", int64)
    assert_true(out_type_is[Float64Type](expr))
    assert_true(_eq(expr, array([10.0, 10.0, 10.0, 10.0], float64)))


def test_unary_neg() raises:
    assert_true(_eq(-col("a", int64), array([-1, -2, -3, -4], int64)))


def test_trunc_preserves_and_executes() raises:
    var expr = col("a", int64).trunc()
    assert_true(out_type_is[Int64Type](expr))
    assert_true(_eq(expr, array([1, 2, 3, 4], int64)))


def test_log2_executes_to_float() raises:
    var expr = col("c", int32).log2()
    assert_true(out_type_is[Float64Type](expr))
    assert_true(_eq(expr, array([1.0, 1.0, 1.0, 1.0], float64)))


def test_fused_math_chain() raises:
    assert_true(
        _eq(col("a", int64).exp2(), array([2.0, 4.0, 8.0, 16.0], float64))
    )


def test_anyvalue_erases_and_executes() raises:
    var boxed = AnyValue((col("a", int64) + col("b", int64)) * col("c", int32))
    assert_true(boxed.execute(_batch()) == array([22, 44, 66, 88], int64))


def test_anyvalue_heterogeneous_list() raises:
    var batch = _batch()
    var exprs = List[AnyValue]()
    exprs.append(col("a", int64) + col("b", int64))
    exprs.append(-col("a", int64))
    exprs.append(col("b", int64) / col("a", int64))
    assert_true(exprs[0].execute(batch) == array([11, 22, 33, 44], int64))
    assert_true(exprs[1].execute(batch) == array([-1, -2, -3, -4], int64))
    assert_true(
        exprs[2].execute(batch) == array([10.0, 10.0, 10.0, 10.0], float64)
    )


def test_table_reflects_and_executes() raises:
    var t = Table[Orders]()
    var expr = (t.a + t.b) * t.c
    assert_true(out_type_is[Int64Type](expr))
    assert_true(_eq(expr, array([22, 44, 66, 88], int64)))


# ===========================================================================
# String execution — materialized through the real kernels
# ===========================================================================


def _sbatch() raises -> RecordBatch:
    return record_batch(
        [array(["Hello", "WORLD", " pad ", "abc"]).copy()],
        names=["s"],
    )


def test_string_column_execute() raises:
    var r = col("s", string).execute(_sbatch())
    assert_true(r == array(["Hello", "WORLD", " pad ", "abc"]))


def test_string_const_broadcast() raises:
    var r = lit("x", string).execute(_sbatch())
    assert_true(r == array(["x", "x", "x", "x"]))


def test_string_upper() raises:
    var r = col("s", string).upper().execute(_sbatch())
    assert_true(r == array(["HELLO", "WORLD", " PAD ", "ABC"]))


def test_string_lower() raises:
    var r = col("s", string).lower().execute(_sbatch())
    assert_true(r == array(["hello", "world", " pad ", "abc"]))


def test_string_strip() raises:
    var r = col("s", string).strip().execute(_sbatch())
    assert_true(r == array(["Hello", "WORLD", "pad", "abc"]))


def test_string_reverse() raises:
    var r = col("s", string).reverse().execute(_sbatch())
    assert_true(r == array(["olleH", "DLROW", " dap ", "cba"]))


def test_string_capitalize() raises:
    var r = col("s", string).capitalize().execute(_sbatch())
    assert_true(r == array(["Hello", "World", " pad ", "Abc"]))


def test_string_chained_unary() raises:
    var r = col("s", string).strip().upper().execute(_sbatch())
    assert_true(r == array(["HELLO", "WORLD", "PAD", "ABC"]))


def test_string_length() raises:
    var r = col("s", string).length().execute(_sbatch())
    assert_true(r == array([5, 5, 5, 3], int32))


def test_string_startswith() raises:
    var r = col("s", string).startswith(lit("W", string)).execute(_sbatch())
    assert_true(r == array([False, True, False, False]))


def test_string_endswith() raises:
    var r = col("s", string).endswith(lit("D", string)).execute(_sbatch())
    assert_true(r == array([False, True, False, False]))


def test_string_contains() raises:
    var r = col("s", string).contains(lit("o", string)).execute(_sbatch())
    assert_true(r == array([True, False, False, False]))


def test_anyvalue_erases_string_unary() raises:
    var boxed = AnyValue(col("s", string).upper())
    assert_true(
        boxed.execute(_sbatch()) == array(["HELLO", "WORLD", " PAD ", "ABC"])
    )


def test_anyvalue_erases_string_predicate() raises:
    var boxed = AnyValue(col("s", string).contains(lit("a", string)))
    assert_true(boxed.execute(_sbatch()) == array([False, False, True, True]))


def main() raises:
    TestSuite.run[__functions_in_module()]()
