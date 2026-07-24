"""Tests for marrow.expr.lane — the strategy-pluggable, staged execution model."""

from std.testing import assert_true, assert_equal

from marrow.testing import TestSuite
from marrow.builders import array
from marrow.arrays import AnyArray
from marrow.dtypes import int64, int32, float64, Int64Type
from marrow.tabular import record_batch, RecordBatch
from marrow.expr.lane import (
    col,
    lit,
    Add,
    Mul,
    Neg,
    Div,
    NumericCast,
    Sum,
    Max,
    Lt,
    Gt,
    And,
    Or,
    Not,
    Any,
    All,
    Count,
    IsNull,
    NotNull,
    IsNan,
    NumToBool,
    BoolToNum,
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
from marrow.expr.dynamic import col as dyn_col


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
    var cv = (Add(col("a", int64), col("b", int64))).execute(_batch())
    assert_true(not cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([11, 22, 33, 44], int64).to_any())


def test_literal_broadcast() raises:
    var cv = (Mul(col("a", int64), lit(10, int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([10, 20, 30, 40], int64).to_any())


def test_scalar_literal_evaluates_once() raises:
    var cv = (lit(7, int64)).execute(_batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 3) == array([7, 7, 7], int64).to_any())


def test_fused_chain() raises:
    # (a + b) * a  over a=[1,2,3,4], b=[10,20,30,40]
    var cv = (Mul(Add(col("a", int64), col("b", int64)), col("a", int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([11, 44, 99, 176], int64).to_any())


def test_reduction_is_scalar() raises:
    # sum(a) over [1,2,3,4] = 10, a scalar
    var cv = (Sum(col("a", int64))).execute(_batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([10, 10, 10, 10], int64).to_any())


def test_reduction_broadcasts_into_columnar() raises:
    # a + sum(a) = [1,2,3,4] + 10 = [11,12,13,14] — the SINGLE Add, sum(a) is a
    # fused leaf reading its stage result from the context and splatting.
    var cv = (Add(col("a", int64), Sum(col("a", int64)))).execute(_batch())
    assert_true(not cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([11, 12, 13, 14], int64).to_any())


def test_scalar_plus_scalar_stays_scalar() raises:
    # sum(a) + max(a) = 10 + 4 = 14, still scalar
    var cv = (Add(Sum(col("a", int64)), Max(col("a", int64)))).execute(_batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 2) == array([14, 14], int64).to_any())


def test_arithmetic_above_reduction() raises:
    # (a + b) fuses, then * sum(a) broadcasts:  [11,22,33,44] * 10
    var cv = (Mul(Add(col("a", int64), col("b", int64)), Sum(col("a", int64)))).execute(_batch())
    assert_true(into_array(cv, 4) == array([110, 220, 330, 440], int64).to_any())


def test_fused_node_is_fusable() raises:
    # `Add` over fusable operands is itself `NumericValue`; `_takes_fusable`
    # compiling is the compile-time proof.
    assert_true(_takes_fusable(Add(col("a", int64), col("b", int64))))


def test_div_is_true_division() raises:
    # 1/2,2/2,3/2,4/2 = [0.5,1.0,1.5,2.0] float64 — true division, not integer
    var cv = (Div(col("a", int64), lit(2, int64))).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([0.5, 1.0, 1.5, 2.0], float64).to_any()
    )


def test_unary_neg_fuses() raises:
    var cv = (Neg(col("a", int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([-1, -2, -3, -4], int64).to_any())


def test_cast_fuses_in_chain() raises:
    # a fused cast composes with arithmetic in the same pass (identity cast here)
    var cv = (Add(NumericCast[Int64Type](col("a", int64)), col("b", int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([11, 22, 33, 44], int64).to_any())


def _spec() -> WindowSpec:
    return WindowSpec(FrameBound(0, 0), FrameBound(2, 0))


def test_window_row_number() raises:
    var cv = (RowNumber(col("a", int64), _spec())).execute(_batch())
    assert_true(into_array(cv, 4) == array([1, 2, 3, 4], int64).to_any())


def test_arithmetic_above_window_materializes() raises:
    # row_number() + 1 → [2,3,4,5]  (Add above a columnar window breaker)
    var cv = (Add(RowNumber(col("a", int64), _spec()), lit(1, int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([2, 3, 4, 5], int64).to_any())


def test_comparison_fuses_to_bool() raises:
    # a < 3 over [1,2,3,4] → bit-packed [T,T,F,F] (the bool fused strategy)
    var cv = (Lt(col("a", int64), lit(3, int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([True, True, False, False]).to_any())


def test_bool_and_fuses() raises:
    # (a < 3) & (b > 15) → [T,T,F,F] & [F,T,T,T] = [F,T,F,F], one fused bitwise pass
    var cv = (And(Lt(col("a", int64), lit(3, int64)), Gt(col("b", int64), lit(15, int64)))).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([False, True, False, False]).to_any()
    )


def test_bool_not_fuses() raises:
    # not (a < 3) → not [T,T,F,F] = [F,F,T,T]
    var cv = (Not(Lt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([False, False, True, True]).to_any()
    )


def test_bool_or_fuses() raises:
    # (a < 2) | (a > 3) → [T,F,F,F] | [F,F,F,T] = [T,F,F,T]
    var cv = (Or(Lt(col("a", int64), lit(2, int64)), Gt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([True, False, False, True]).to_any()
    )


def test_any_all_reductions() raises:
    # any(a < 3) = True, all(a < 3) = False over [1,2,3,4]
    var an = (Any(Lt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(an.isa[AnyScalar]() and an[AnyScalar].as_bool().value())
    var al = (All(Lt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(al.isa[AnyScalar]() and not al[AnyScalar].as_bool().value())


def test_count_reduction() raises:
    # count(a) over [1,2,3,4] = 4 (int64 scalar)
    var cv = (Count(col("a", int64))).execute(_batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([4, 4, 4, 4], int64).to_any())


def test_notnull_and_isnull() raises:
    # no nulls in a=[1,2,3,4] → not_null all true, is_null all false
    var nn = (NotNull(col("a", int64))).execute(_batch())
    assert_true(into_array(nn, 4) == array([True, True, True, True]).to_any())
    var isn = (IsNull(col("a", int64))).execute(_batch())
    assert_true(
        into_array(isn, 4) == array([False, False, False, False]).to_any()
    )


def test_isnan_fuses_over_float() raises:
    # is_nan over finite floats → all false, computed in a fused SIMD pass
    var b = record_batch(
        [array([1.0, 2.0, 3.0, 4.0], float64).copy()], names=["f"]
    )
    var cv = (IsNan(col("f", float64))).execute(b)
    assert_true(
        into_array(cv, 4) == array([False, False, False, False]).to_any()
    )


def test_num_to_bool_fuses() raises:
    # a*0 = 0 → false ; a (nonzero) → true — fused per-lane num->bool
    var z = (NumToBool(Mul(col("a", int64), lit(0, int64)))).execute(_batch())
    assert_true(
        into_array(z, 4) == array([False, False, False, False]).to_any()
    )
    var nz = (NumToBool(col("a", int64))).execute(_batch())
    assert_true(into_array(nz, 4) == array([True, True, True, True]).to_any())


def test_bool_to_num_fuses() raises:
    # (a < 3) -> int64 = [1,1,0,0] — fused bool->num, composes in the numeric lane
    var cv = (BoolToNum[Int64Type](Lt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(into_array(cv, 4) == array([1, 1, 0, 0], int64).to_any())


def test_fluent_numeric_and_bool() raises:
    # operators/methods build the same nodes as the explicit builders
    var s = (col("a", int64) + col("b", int64)).execute(_batch())
    assert_true(into_array(s, 4) == array([11, 22, 33, 44], int64).to_any())
    # mean-centering via `x - x.mean()`
    var mc = (col("a", int64) - col("a", int64).mean()).execute(_batch())
    assert_true(
        into_array(mc, 4) == array([-1.5, -0.5, 0.5, 1.5], float64).to_any()
    )
    # (a < 3) & (b > 15) via `<`, `>`, `&`
    var mask = (
        (col("a", int64) < lit(3, int64)) & (col("b", int64) > lit(15, int64))
    ).execute(_batch())
    assert_true(
        into_array(mask, 4) == array([False, True, False, False]).to_any()
    )


def test_anyvalue_wraps_dynvalue() raises:
    # the untyped runtime interpreter (DynValue), boxed in lane.AnyValue, runs via
    # the tag dispatch — this is what the relational engine builds plans from
    var boxed: AnyValue = dyn_col(0) + dyn_col(1)
    var cv = boxed.execute(_batch())
    assert_true(cv == array([11, 22, 33, 44], int64).to_any())


def test_anyvalue_erases_to_array() raises:
    # box a comptime node; its erased execute yields a column (AnyArray), the
    # interface the relational engine consumes
    var boxed: AnyValue = Add(col("a", int64), lit(10, int64))
    var cv = boxed.execute(_batch())
    assert_true(cv == array([11, 12, 13, 14], int64).to_any())


def main() raises:
    TestSuite.run[__functions_in_module()]()
