"""Execution tests for marrow.expr.fused — proves the nodes hook to real kernels
and fuse (one vectorized pass) with correct results and dtypes.
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
    Int64Type,
    Int32Type,
    Float64Type,
    BoolType,
    DataType,
)
from marrow.tabular import record_batch, RecordBatch
from marrow.expr.ibis import col, lit, Value, NumericValue, AnyValue, Table


struct Orders:
    var a: Int64Type
    var b: Int64Type
    var c: Int32Type


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


def out_type_is[Want: DataType, V: Value](x: V) -> Bool:
    return V.OutType == Want


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
    # a: int64, c: int32 -> result int64 (highest precedence)
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
    # trunc on ints is identity; preserves dtype and fuses
    var expr = col("a", int64).trunc()
    assert_true(out_type_is[Int64Type](expr))
    assert_true(_eq(expr, array([1, 2, 3, 4], int64)))


def test_log2_executes_to_float() raises:
    # c is all 2s -> log2 == 1.0; a float64 result via the real Log2Kernel core
    var expr = col("c", int32).log2()
    assert_true(out_type_is[Float64Type](expr))
    assert_true(_eq(expr, array([1.0, 1.0, 1.0, 1.0], float64)))


def test_fused_math_chain() raises:
    # exp2(a) fused, then a fixed check on a=[1,2,3,4] -> [2,4,8,16]
    assert_true(
        _eq(col("a", int64).exp2(), array([2.0, 4.0, 8.0, 16.0], float64))
    )


def test_anyvalue_erases_and_executes() raises:
    # box a fused numeric expression, execute it through the erased handle
    var boxed = AnyValue((col("a", int64) + col("b", int64)) * col("c", int32))
    assert_true(boxed.execute(_batch()) == array([22, 44, 66, 88], int64))


def test_anyvalue_heterogeneous_list() raises:
    # a heterogeneous list of erased expressions, each executed
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
    # Table[Orders]().a reflects field `a`'s dtype -> NumericColumn[Int64Type]
    var t = Table[Orders]()
    var expr = (t.a + t.b) * t.c
    assert_true(out_type_is[Int64Type](expr))
    assert_true(_eq(expr, array([22, 44, 66, 88], int64)))


# ---------------------------------------------------------------------------
# String execution
# ---------------------------------------------------------------------------


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
    # strip then upper — two materializing passes
    var r = col("s", string).strip().upper().execute(_sbatch())
    assert_true(r == array(["HELLO", "WORLD", "PAD", "ABC"]))


def test_string_length() raises:
    var r = col("s", string).length().execute(_sbatch())
    assert_true(r == array([5, 5, 5, 3], int32))


def test_string_length_dtype() raises:
    assert_true(out_type_is[Int32Type](col("s", string).length()))


def test_string_startswith() raises:
    var r = col("s", string).startswith(lit("W", string)).execute(_sbatch())
    assert_true(r == array([False, True, False, False]))


def test_string_endswith() raises:
    var r = col("s", string).endswith(lit("D", string)).execute(_sbatch())
    assert_true(r == array([False, True, False, False]))


def test_string_contains() raises:
    var r = col("s", string).contains(lit("o", string)).execute(_sbatch())
    # "Hello"->o yes; "WORLD"->lowercase o no; " pad "->no; "abc"->no
    assert_true(r == array([True, False, False, False]))


def test_string_predicate_dtype() raises:
    assert_true(
        out_type_is[BoolType](col("s", string).startswith(lit("W", string)))
    )


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
