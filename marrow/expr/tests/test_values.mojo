"""Tests for the comptime-typed expression nodes in ``marrow.expr.values``.

Two families, both exercising the single-fused-vectorize-pass execution model:

- ``NumericValue`` nodes — ``NumericColumn``/``Add``/``Sub``/``Length`` evaluate
  an expression tree via ``execute(batch)`` with zero intermediate arrays.
- ``BoolValue`` nodes — ``Lt``/``Gt``/``Eq`` bit-pack the comparison mask
  directly into a ``BoolArray``, and compose with ``NumericValue`` children in
  the same fused pass.

See ``dyn/tests/test_aot_injection.mojo`` for these nodes boxed into the runtime
``DynValue`` via the ``FUSED`` tag.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import PrimitiveArray
from marrow.builders import array
from marrow.dtypes import Int64Type, int64, uint32
from marrow.tabular import RecordBatch, record_batch
from marrow.expr.values import (
    NumericColumn,
    Add,
    Sub,
    Lt,
    Gt,
    Eq,
    StringColumn,
    Length,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fused_add_exec(
    col_a: Int, col_b: Int, batch: RecordBatch
) raises -> PrimitiveArray[Int64Type]:
    """Build an Add expression and evaluate against the batch."""
    var a = NumericColumn[Int64Type](col_a)
    var b = NumericColumn[Int64Type](col_b)
    var expr = Add(a, b)
    return expr.execute(batch)


def _fused_sub_exec(
    col_a: Int, col_b: Int, batch: RecordBatch
) raises -> PrimitiveArray[Int64Type]:
    """Build a Sub expression and evaluate against the batch."""
    var a = NumericColumn[Int64Type](col_a)
    var b = NumericColumn[Int64Type](col_b)
    var expr = Sub(a, b)
    return expr.execute(batch)


# ---------------------------------------------------------------------------
# Basic fused add
# ---------------------------------------------------------------------------


def test_fused_add_basic() raises:
    """Add(col(0), col(1)) produces element-wise addition."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([11, 22, 33, 44, 55], int64))


def test_fused_add_single_element() raises:
    """Add works with a single-element array."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_equal(result[0].value(), 50)


def test_fused_add_non_aligned() raises:
    """Add works with non-SIMD-aligned lengths."""
    var a = array([1, 2, 3, 4, 5, 6, 7], int64)
    var b = array([10, 20, 30, 40, 50, 60, 70], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([11, 22, 33, 44, 55, 66, 77], int64))


def test_fused_add_negative_values() raises:
    """Add handles negative values correctly."""
    var a = array([-1, 2, -3, 4, -5], int64)
    var b = array([1, -2, 3, -4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([0, 0, 0, 0, 0], int64))


def test_fused_add_large_values() raises:
    """Add handles large int64 values."""
    var a = array([9_223_372_036_854_775_806], int64)  # near max
    var b = array([1], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_equal(result[0].value(), 9_223_372_036_854_775_807)  # max int64


def test_fused_add_same_column() raises:
    """Add(col(0), col(0)) doubles the values."""
    var a = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var result = _fused_add_exec(0, 0, batch)
    assert_true(result == array([2, 4, 6, 8, 10], int64))


def test_fused_add_zero() raises:
    """Add with zeros produces all zeros."""
    var a = array([0, 0, 0], int64)
    var b = array([0, 0, 0], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([0, 0, 0], int64))


# ---------------------------------------------------------------------------
# Sub
# ---------------------------------------------------------------------------


def test_fused_sub_basic() raises:
    """Sub(col(0), col(1)) produces element-wise subtraction."""
    var a = array([10, 20, 30, 40, 50], int64)
    var b = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_sub_exec(0, 1, batch)
    assert_true(result == array([9, 18, 27, 36, 45], int64))


def test_fused_sub_single_element() raises:
    """Sub works with a single-element array."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_sub_exec(0, 1, batch)
    assert_equal(result[0].value(), 34)


def test_fused_sub_negative_result() raises:
    """Sub handles negative results."""
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_sub_exec(0, 1, batch)
    assert_true(result == array([-9, -18, -27], int64))


# ---------------------------------------------------------------------------
# Length (string)
# ---------------------------------------------------------------------------


def test_fused_length_basic() raises:
    """Length(StringColumn(0)) produces per-element byte lengths."""
    var a = array(["ab", "cde", "", "f"])
    var batch = record_batch([a^], names=["c0"])
    var expr = Length(StringColumn(0))
    var result = expr.execute(batch)
    assert_true(result == array([2, 3, 0, 1], uint32))


def test_fused_length_non_aligned() raises:
    """Length works with non-SIMD-aligned lengths."""
    var a = array(["a", "bb", "ccc", "dddd", "e", "ff", "ggg"])
    var batch = record_batch([a^], names=["c0"])
    var expr = Length(StringColumn(0))
    var result = expr.execute(batch)
    assert_true(result == array([1, 2, 3, 4, 1, 2, 3], uint32))


def test_fused_length_sliced() raises:
    """Length matches kernels.string.string_lengths on a sliced array."""
    var full = array(["aa", "b", "ccc", "dddd", "e"])
    var a = full.slice(1, 3)
    var batch = record_batch([a^], names=["c0"])
    var expr = Length(StringColumn(0))
    var result = expr.execute(batch)
    assert_true(result == array([1, 3, 4], uint32))


def test_fused_length_write_to() raises:
    """Length.write_to produces nested readable output."""
    var expr = Length(StringColumn(2))
    assert_equal(String(expr), "Length(StrCol[2])")


# ---------------------------------------------------------------------------
# Trait conformance
# ---------------------------------------------------------------------------


def test_fused_column_implements_numeric_typed_value() raises:
    """NumericColumn[Int64Type] satisfies NumericValue."""
    # The fact that .execute() compiles on this type proves trait conformance.
    var expr = NumericColumn[Int64Type](0)
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var result = expr.execute(batch)
    assert_true(result == array([1, 2, 3], int64))


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def test_fused_column_write_to() raises:
    """NumericColumn.write_to produces readable output."""
    var col = NumericColumn[Int64Type](3)
    assert_equal(String(col), "Col[3]")


def test_fused_add_write_to() raises:
    """Add.write_to produces nested readable output."""
    var a = NumericColumn[Int64Type](0)
    var b = NumericColumn[Int64Type](1)
    var expr = Add(a, b)
    assert_equal(String(expr), "Add(Col[0], Col[1])")


# ---------------------------------------------------------------------------
# Copy semantics
# ---------------------------------------------------------------------------


def test_fused_column_copy() raises:
    """NumericColumn can be copied."""
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var col = NumericColumn[Int64Type](0)
    var copy = col.copy()
    var result = copy.execute(batch)
    assert_true(result == array([1, 2, 3], int64))


def test_fused_add_copy() raises:
    """Add can be copied and re-executed."""
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var col_a = NumericColumn[Int64Type](0)
    var col_b = NumericColumn[Int64Type](1)
    var expr = Add(col_a, col_b)
    var copy = expr.copy()
    var result = copy.execute(batch)
    assert_true(result == array([11, 22, 33], int64))


# ---------------------------------------------------------------------------
# BoolValue nodes — Lt / Gt / Eq
# ---------------------------------------------------------------------------


def test_lt_fuses() raises:
    """Lt(NumericColumn, NumericColumn) produces the correct bit-packed
    BoolArray."""
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var lt = Lt(NumericColumn[Int64Type](0), NumericColumn[Int64Type](1))
    var result = lt.execute(batch)
    assert_true(result[0].value())
    assert_true(not result[1].value())
    assert_true(result[2].value())
    assert_true(not result[3].value())
    assert_true(result[4].value())


def test_gt_fuses() raises:
    """Gt(NumericColumn, NumericColumn) produces the correct bit-packed
    BoolArray."""
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var gt = Gt(NumericColumn[Int64Type](0), NumericColumn[Int64Type](1))
    var result = gt.execute(batch)
    assert_true(not result[0].value())
    assert_true(result[1].value())
    assert_true(not result[2].value())
    assert_true(result[3].value())
    assert_true(not result[4].value())


def test_eq_fuses() raises:
    """Eq(NumericColumn, NumericColumn) produces the correct bit-packed
    BoolArray."""
    var a = array([1, 4, 3, 4, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var eq = Eq(NumericColumn[Int64Type](0), NumericColumn[Int64Type](1))
    var result = eq.execute(batch)
    assert_true(not result[0].value())
    assert_true(result[1].value())
    assert_true(not result[2].value())
    assert_true(result[3].value())
    assert_true(not result[4].value())


def test_comparison_composes_with_add() raises:
    """Gt((a + b), b) composes NumericValue and BoolValue nodes in one fused
    pass -- no intermediate arrays for either the add or the comparison.
    """
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var added = Add(NumericColumn[Int64Type](0), NumericColumn[Int64Type](1))
    var pred = Gt(added, NumericColumn[Int64Type](1))
    var result = pred.execute(batch)
    # a + b = [5, 9, 7, 12, 6]; compared to b = [4, 4, 4, 4, 4] -> all True
    for i in range(5):
        assert_true(result[i].value())


def test_comparison_write_to() raises:
    """Lt/Gt/Eq.write_to() display the expected nested structure."""
    var col_a = NumericColumn[Int64Type](0)
    var col_b = NumericColumn[Int64Type](1)
    assert_equal(String(Lt(col_a, col_b)), "Lt(Col[0], Col[1])")
    assert_equal(String(Gt(col_a, col_b)), "Gt(Col[0], Col[1])")
    assert_equal(String(Eq(col_a, col_b)), "Eq(Col[0], Col[1])")


def test_comparison_dtype_is_bool() raises:
    """Lt/Gt/Eq.dtype() reports bool_, matching BoolValue's default impl."""
    var col_a = NumericColumn[Int64Type](0)
    var col_b = NumericColumn[Int64Type](1)
    var lt = Lt(col_a, col_b)
    assert_true(lt.dtype().value().is_bool())


def main() raises:
    TestSuite.run[__functions_in_module()]()
