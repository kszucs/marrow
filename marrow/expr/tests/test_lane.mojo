"""Tests for marrow.expr.lane — the strategy-pluggable, staged execution model."""

from std.testing import assert_true, assert_equal

from marrow.testing import TestSuite
from marrow.builders import array
from marrow.arrays import AnyArray
from marrow.dtypes import int64, int32, Int64Type
from marrow.tabular import record_batch, RecordBatch
from marrow.expr.lane import (
    col,
    lit,
    run,
    Add,
    Mul,
    Neg,
    NumericCast,
    Sum,
    Max,
    Lt,
    RowNumber,
    WindowSpec,
    FrameBound,
    Datum,
    NumericValue,
    AnyValue,
    Context,
    into_array,
)
from marrow.scalars import AnyScalar


# instantiation is a COMPILE-TIME proof the operand is a fused `NumericValue` node
def _takes_fusable[F: NumericValue](x: F) -> Bool:
    return True


def _batch() raises -> RecordBatch:
    return record_batch(
        [
            array([1, 2, 3, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def test_column_add_fuses() raises:
    var cv = run(Add(col(0, int64), col(1, int64)), _batch())
    assert_true(not cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([11, 22, 33, 44], int64).to_any())


def test_literal_broadcast() raises:
    var cv = run(Mul(col(0, int64), lit(10, int64)), _batch())
    assert_true(into_array(cv, 4) == array([10, 20, 30, 40], int64).to_any())


def test_scalar_literal_evaluates_once() raises:
    var cv = run(lit(7, int64), _batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 3) == array([7, 7, 7], int64).to_any())


def test_fused_chain() raises:
    # (a + b) * a  over a=[1,2,3,4], b=[10,20,30,40]
    var cv = run(Mul(Add(col(0, int64), col(1, int64)), col(0, int64)), _batch())
    assert_true(into_array(cv, 4) == array([11, 44, 99, 176], int64).to_any())


def test_reduction_is_scalar() raises:
    # sum(a) over [1,2,3,4] = 10, a scalar
    var cv = run(Sum(col(0, int64)), _batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([10, 10, 10, 10], int64).to_any())


def test_reduction_broadcasts_into_columnar() raises:
    # a + sum(a) = [1,2,3,4] + 10 = [11,12,13,14] — the SINGLE Add, sum(a) is a
    # fused leaf reading its stage result from the context and splatting.
    var cv = run(Add(col(0, int64), Sum(col(0, int64))), _batch())
    assert_true(not cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([11, 12, 13, 14], int64).to_any())


def test_scalar_plus_scalar_stays_scalar() raises:
    # sum(a) + max(a) = 10 + 4 = 14, still scalar
    var cv = run(Add(Sum(col(0, int64)), Max(col(0, int64))), _batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 2) == array([14, 14], int64).to_any())


def test_arithmetic_above_reduction() raises:
    # (a + b) fuses, then * sum(a) broadcasts:  [11,22,33,44] * 10
    var cv = run(
        Mul(Add(col(0, int64), col(1, int64)), Sum(col(0, int64))), _batch()
    )
    assert_true(into_array(cv, 4) == array([110, 220, 330, 440], int64).to_any())


def test_fused_node_is_fusable() raises:
    # `Add` over fusable operands is itself `NumericValue`; `_takes_fusable`
    # compiling is the compile-time proof.
    assert_true(_takes_fusable(Add(col(0, int64), col(1, int64))))


def test_unary_neg_fuses() raises:
    var cv = run(Neg(col(0, int64)), _batch())
    assert_true(into_array(cv, 4) == array([-1, -2, -3, -4], int64).to_any())


def test_cast_fuses_in_chain() raises:
    # a fused cast composes with arithmetic in the same pass (identity cast here)
    var cv = run(
        Add(NumericCast[Int64Type](col(0, int64)), col(1, int64)), _batch()
    )
    assert_true(into_array(cv, 4) == array([11, 22, 33, 44], int64).to_any())


def _spec() -> WindowSpec:
    return WindowSpec(FrameBound(0, 0), FrameBound(2, 0))


def test_window_row_number() raises:
    var cv = run(RowNumber(col(0, int64), _spec()), _batch())
    assert_true(into_array(cv, 4) == array([1, 2, 3, 4], int64).to_any())


def test_arithmetic_above_window_materializes() raises:
    # row_number() + 1 → [2,3,4,5]  (Add above a columnar window breaker)
    var cv = run(Add(RowNumber(col(0, int64), _spec()), lit(1, int64)), _batch())
    assert_true(into_array(cv, 4) == array([2, 3, 4, 5], int64).to_any())


def test_comparison_fuses_to_bool() raises:
    # a < 3 over [1,2,3,4] → bit-packed [T,T,F,F] (the bool fused strategy)
    var cv = run(Lt(col(0, int64), lit(3, int64)), _batch())
    assert_true(into_array(cv, 4) == array([True, True, False, False]).to_any())


def test_anyvalue_erases_to_datum() raises:
    # box a comptime node; its erased execute still yields the same Datum
    var boxed: AnyValue = Add(col(0, int64), lit(10, int64))
    var ctx = Context()
    var cv = boxed.execute(_batch(), ctx)
    assert_true(into_array(cv, 4) == array([11, 12, 13, 14], int64).to_any())


def main() raises:
    TestSuite.run[__functions_in_module()]()
