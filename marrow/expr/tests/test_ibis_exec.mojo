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
    Int64Type,
    Float64Type,
    DataType,
)
from marrow.tabular import record_batch, RecordBatch
from marrow.expr.ibis import col, lit, Value, NumericValue, AnyValue


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


def main() raises:
    TestSuite.run[__functions_in_module()]()
