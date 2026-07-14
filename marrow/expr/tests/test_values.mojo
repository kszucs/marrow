"""Tests for the comptime-typed expression nodes in ``marrow.expr.values``.

Two families, both exercising the single-fused-vectorize-pass execution model:

- ``NumericValue`` nodes — ``Add``/``Sub``/``Length`` (over the named column
  leaves) evaluate an expression tree via ``execute(batch)`` with zero
  intermediate arrays.
- ``BoolValue`` nodes — ``Less``/``Greater``/``Equal`` bit-pack the comparison mask
  directly into a ``BoolArray``, and compose with ``NumericValue`` children in
  the same fused pass.

The leaf columns come from ``col(name, dtype)`` (the named ``NumericColumn`` /
``StringColumn`` in ``marrow.expr.values``); positions resolve by name.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import PrimitiveArray
from marrow.builders import array
from marrow.dtypes import (
    Int32Type,
    Int64Type,
    Float64Type,
    bool_,
    int8,
    int32,
    int64,
    float64,
    uint32,
    string,
)
from marrow.tabular import RecordBatch, record_batch
from marrow.expr.values import (
    Add,
    Sub,
    Less,
    Greater,
    Equal,
    Length,
    Cast,
    NumToBoolValue,
    BoolToNumValue,
    col,
)
from marrow.kernels.cast import cast as eager_cast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fused_add_exec(
    name_a: String, name_b: String, batch: RecordBatch
) raises -> PrimitiveArray[Int64Type]:
    """Build an Add expression and evaluate against the batch."""
    var expr = Add(col(name_a, int64), col(name_b, int64))
    return expr.execute(batch)


def _fused_sub_exec(
    name_a: String, name_b: String, batch: RecordBatch
) raises -> PrimitiveArray[Int64Type]:
    """Build a Sub expression and evaluate against the batch."""
    var expr = Sub(col(name_a, int64), col(name_b, int64))
    return expr.execute(batch)


# ---------------------------------------------------------------------------
# Basic fused add
# ---------------------------------------------------------------------------


def test_fused_add_basic() raises:
    """Add(col(a), col(b)) produces element-wise addition."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec("c0", "c1", batch)
    assert_true(result == array([11, 22, 33, 44, 55], int64))


def test_fused_add_single_element() raises:
    """Add works with a single-element array."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec("c0", "c1", batch)
    assert_equal(result[0].value(), 50)


def test_fused_add_non_aligned() raises:
    """Add works with non-SIMD-aligned lengths."""
    var a = array([1, 2, 3, 4, 5, 6, 7], int64)
    var b = array([10, 20, 30, 40, 50, 60, 70], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec("c0", "c1", batch)
    assert_true(result == array([11, 22, 33, 44, 55, 66, 77], int64))


def test_fused_add_negative_values() raises:
    """Add handles negative values correctly."""
    var a = array([-1, 2, -3, 4, -5], int64)
    var b = array([1, -2, 3, -4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec("c0", "c1", batch)
    assert_true(result == array([0, 0, 0, 0, 0], int64))


def test_fused_add_large_values() raises:
    """Add handles large int64 values."""
    var a = array([9_223_372_036_854_775_806], int64)  # near max
    var b = array([1], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec("c0", "c1", batch)
    assert_equal(result[0].value(), 9_223_372_036_854_775_807)  # max int64


def test_fused_add_same_column() raises:
    """Add(col(c0), col(c0)) doubles the values."""
    var a = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var result = _fused_add_exec("c0", "c0", batch)
    assert_true(result == array([2, 4, 6, 8, 10], int64))


def test_fused_add_zero() raises:
    """Add with zeros produces all zeros."""
    var a = array([0, 0, 0], int64)
    var b = array([0, 0, 0], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec("c0", "c1", batch)
    assert_true(result == array([0, 0, 0], int64))


# ---------------------------------------------------------------------------
# Sub
# ---------------------------------------------------------------------------


def test_fused_sub_basic() raises:
    """Sub(col(c0), col(c1)) produces element-wise subtraction."""
    var a = array([10, 20, 30, 40, 50], int64)
    var b = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_sub_exec("c0", "c1", batch)
    assert_true(result == array([9, 18, 27, 36, 45], int64))


def test_fused_sub_single_element() raises:
    """Sub works with a single-element array."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_sub_exec("c0", "c1", batch)
    assert_equal(result[0].value(), 34)


def test_fused_sub_negative_result() raises:
    """Sub handles negative results."""
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_sub_exec("c0", "c1", batch)
    assert_true(result == array([-9, -18, -27], int64))


# ---------------------------------------------------------------------------
# Length (string)
# ---------------------------------------------------------------------------


def test_fused_length_basic() raises:
    """Length(col(c0, string)) produces per-element byte lengths."""
    var a = array(["ab", "cde", "", "f"])
    var batch = record_batch([a^], names=["c0"])
    var expr = Length(col("c0", string))
    var result = expr.execute(batch)
    assert_true(result == array([2, 3, 0, 1], uint32))


def test_fused_length_non_aligned() raises:
    """Length works with non-SIMD-aligned lengths."""
    var a = array(["a", "bb", "ccc", "dddd", "e", "ff", "ggg"])
    var batch = record_batch([a^], names=["c0"])
    var expr = Length(col("c0", string))
    var result = expr.execute(batch)
    assert_true(result == array([1, 2, 3, 4, 1, 2, 3], uint32))


def test_fused_length_sliced() raises:
    """Length matches kernels.string.string_lengths on a sliced array."""
    var full = array(["aa", "b", "ccc", "dddd", "e"])
    var a = full.slice(1, 3)
    var batch = record_batch([a^], names=["c0"])
    var expr = Length(col("c0", string))
    var result = expr.execute(batch)
    assert_true(result == array([1, 3, 4], uint32))


def test_fused_length_write_to() raises:
    """Length.write_to produces nested readable output."""
    var expr = Length(col("s", string))
    assert_equal(String(expr), "Length(StrCol[s])")


# ---------------------------------------------------------------------------
# Trait conformance
# ---------------------------------------------------------------------------


def test_fused_column_implements_numeric_typed_value() raises:
    """col(name, int64) satisfies NumericValue."""
    # The fact that .execute() compiles on this type proves trait conformance.
    var expr = col("c0", int64)
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var result = expr.execute(batch)
    assert_true(result == array([1, 2, 3], int64))


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def test_fused_column_write_to() raises:
    """NumericColumn.write_to produces readable output."""
    var c = col("x", int64)
    assert_equal(String(c), "Col[x]")


def test_fused_add_write_to() raises:
    """Add.write_to produces nested readable output."""
    var expr = Add(col("a", int64), col("b", int64))
    assert_equal(String(expr), "Add(Col[a], Col[b])")


# ---------------------------------------------------------------------------
# Copy semantics
# ---------------------------------------------------------------------------


def test_fused_column_copy() raises:
    """A named column can be copied."""
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var c = col("c0", int64)
    var copy = c.copy()
    var result = copy.execute(batch)
    assert_true(result == array([1, 2, 3], int64))


def test_fused_add_copy() raises:
    """Add can be copied and re-executed."""
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var expr = Add(col("c0", int64), col("c1", int64))
    var copy = expr.copy()
    var result = copy.execute(batch)
    assert_true(result == array([11, 22, 33], int64))


# ---------------------------------------------------------------------------
# BoolValue nodes — Less / Greater / Equal
# ---------------------------------------------------------------------------


def test_lt_fuses() raises:
    """Less(col, col) produces the correct bit-packed BoolArray."""
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var lt = Less(col("a", int64), col("b", int64))
    var result = lt.execute(batch)
    assert_true(result[0].value())
    assert_true(not result[1].value())
    assert_true(result[2].value())
    assert_true(not result[3].value())
    assert_true(result[4].value())


def test_gt_fuses() raises:
    """Greater(col, col) produces the correct bit-packed BoolArray."""
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var gt = Greater(col("a", int64), col("b", int64))
    var result = gt.execute(batch)
    assert_true(not result[0].value())
    assert_true(result[1].value())
    assert_true(not result[2].value())
    assert_true(result[3].value())
    assert_true(not result[4].value())


def test_eq_fuses() raises:
    """Equal(col, col) produces the correct bit-packed BoolArray."""
    var a = array([1, 4, 3, 4, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var eq = Equal(col("a", int64), col("b", int64))
    var result = eq.execute(batch)
    assert_true(not result[0].value())
    assert_true(result[1].value())
    assert_true(not result[2].value())
    assert_true(result[3].value())
    assert_true(not result[4].value())


def test_comparison_composes_with_add() raises:
    """Greater((a + b), b) composes NumericValue and BoolValue nodes in one fused
    pass -- no intermediate arrays for either the add or the comparison.
    """
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var added = Add(col("a", int64), col("b", int64))
    var pred = Greater(added, col("b", int64))
    var result = pred.execute(batch)
    # a + b = [5, 9, 7, 12, 6]; compared to b = [4, 4, 4, 4, 4] -> all True
    for i in range(5):
        assert_true(result[i].value())


def test_comparison_write_to() raises:
    """Less/Greater/Equal.write_to() display the expected nested structure."""
    assert_equal(
        String(Less(col("a", int64), col("b", int64))), "Less(Col[a], Col[b])"
    )
    assert_equal(
        String(Greater(col("a", int64), col("b", int64))),
        "Greater(Col[a], Col[b])",
    )
    assert_equal(
        String(Equal(col("a", int64), col("b", int64))), "Equal(Col[a], Col[b])"
    )


def test_comparison_dtype_is_bool() raises:
    """Less/Greater/Equal.dtype() reports bool_, matching BoolValue's default impl.
    """
    var lt = Less(col("a", int64), col("b", int64))
    assert_true(lt.dtype().is_bool())


# ---------------------------------------------------------------------------
# Fused Cast
# ---------------------------------------------------------------------------


def test_fused_cast_basic() raises:
    """Cast(col(a), int64) widens an int32 column to int64 in one pass."""
    var a = array([1, 2, 3], int32)
    var batch = record_batch([a^], names=["c0"])
    var expr = Cast(col("c0", int32), int64)
    var result = expr.execute(batch)
    assert_true(result == array([1, 2, 3], int64))


def test_fused_cast_to_float() raises:
    var a = array([1, 2, 3], int32)
    var batch = record_batch([a^], names=["c0"])
    var result = Cast(col("c0", int32), float64).execute(batch)
    assert_true(result == array([1.0, 2.0, 3.0], float64))


def test_fused_cast_over_add() raises:
    """Cast(Add(a, b), int64) fuses the add and the cast into one pass and
    matches the eager kernel applied to the eager add."""
    var a = array([1, 2, 3], int32)
    var b = array([10, 20, 30], int32)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var expr = Cast(Add(col("c0", int32), col("c1", int32)), int64)
    var result = expr.execute(batch)
    assert_true(result == array([11, 22, 33], int64))


def test_fused_cast_matches_eager() raises:
    var a = array([5, 6, 7, 8], int32)
    var batch = record_batch([a.copy()], names=["c0"])
    var fused = Cast(col("c0", int32), float64).execute(batch)
    var eager = eager_cast(a.copy(), float64)
    assert_true(fused == eager.as_float64())


def test_fused_cast_composes_in_predicate() raises:
    """Less(Cast(a, int64), col(b, int64)) — a cast child inside a predicate."""
    var a = array([1, 2, 3], int32)
    var b = array([5, 1, 9], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var mask = Less(Cast(col("c0", int32), int64), col("c1", int64)).execute(
        batch
    )
    assert_true(mask == array([True, False, True]))


def test_fused_cast_dtype_and_write() raises:
    var expr = Cast(col("a", int32), int64)
    assert_true(expr.dtype().is_int64())
    assert_equal(String(expr), "Cast(Col[a], int64)")


def test_cast_method_on_node() raises:
    """The ``.cast(dtype)`` method on a NumericValue builds the same fused node
    and composes with other nodes."""
    var a = array([1, 2, 3], int32)
    var b = array([10, 20, 30], int32)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    # (a + b).cast(int64)
    var expr = Add(col("c0", int32), col("c1", int32)).cast(int64)
    assert_true(expr.execute(batch) == array([11, 22, 33], int64))
    # a.cast(float64)
    var f = col("c0", int32).cast(float64).execute(batch)
    assert_true(f == array([1.0, 2.0, 3.0], float64))


# ---------------------------------------------------------------------------
# Fused numeric → bool cast (NumToBoolValue)
# ---------------------------------------------------------------------------


def test_fused_num_to_bool() raises:
    """Cast(col(a, int32), bool_) → x != 0, bit-packed in one pass."""
    var a = array([0, 5, 0, -3], int32)
    var batch = record_batch([a^], names=["c0"])
    var mask = col("c0", int32).cast(bool_).execute(batch)
    assert_true(mask == array([False, True, False, True]))


def test_fused_num_to_bool_matches_eager() raises:
    var a = array([0, 1, 2, 0, 7], int32)
    var batch = record_batch([a.copy()], names=["c0"])
    var fused = col("c0", int32).cast(bool_).execute(batch)
    var eager = eager_cast(a.copy(), bool_)
    assert_true(fused == eager.as_bool())


def test_fused_num_to_bool_over_add() raises:
    """(a + b).cast(bool_) fuses the add and the != 0 into one pass."""
    var a = array([1, 0, 2], int32)
    var b = array([-1, 0, 3], int32)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var mask = (
        Add(col("c0", int32), col("c1", int32)).cast(bool_).execute(batch)
    )
    assert_true(mask == array([False, False, True]))  # 0, 0, 5


def test_fused_num_to_bool_write() raises:
    var expr = NumToBoolValue(col("a", int32))
    assert_true(expr.dtype().is_bool())
    assert_equal(String(expr), "Cast(Col[a], bool)")


# ---------------------------------------------------------------------------
# Fused bool → numeric cast (BoolToNumValue)
# ---------------------------------------------------------------------------


def test_fused_bool_to_num() raises:
    """(a < b).cast(int8) → 1/0 written straight into the fused buffer."""
    var a = array([1, 2, 3], int32)
    var b = array([5, 1, 9], int32)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var counts = (
        Less(col("c0", int32), col("c1", int32)).cast(int8).execute(batch)
    )
    assert_true(counts == array([1, 0, 1], int8))


def test_fused_bool_to_num_composes() raises:
    """A bool→num cast is a NumericValue, so it composes back into arithmetic:
    (a < b).cast(int32) + (b < a).cast(int32) is 1 wherever they differ."""
    var a = array([1, 2, 3], int32)
    var b = array([5, 2, 1], int32)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var lt = Less(col("c0", int32), col("c1", int32)).cast(int32)
    var gt = Greater(col("c0", int32), col("c1", int32)).cast(int32)
    var result = Add(lt, gt).execute(batch)
    assert_true(result == array([1, 0, 1], int32))  # <, ==, >


def test_fused_bool_to_num_write() raises:
    var expr = BoolToNumValue(Less(col("a", int32), col("b", int32)), int8)
    assert_true(expr.dtype().is_int8())
    assert_equal(String(expr), "Cast(Less(Col[a], Col[b]), int8)")


def main() raises:
    TestSuite.run[__functions_in_module()]()
