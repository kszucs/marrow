"""Tests for the comptime-typed expression nodes in ``marrow.expr.values``.

Two families, both exercising the single-fused-vectorize-pass execution model:

- ``NumericValue`` nodes — ``Add``/``Sub``/``Length`` (over the named column
  leaves) evaluate an expression tree via ``execute(batch)`` with zero
  intermediate arrays.
- ``BoolValue`` nodes — ``Lt``/``Gt``/``Eq`` bit-pack the comparison mask
  directly into a ``BoolArray``, and compose with ``NumericValue`` children in
  the same fused pass.

The leaf columns come from ``col(name, dtype)`` (the named ``NumericColumn`` /
``StringColumn`` in ``marrow.expr.relations``); positions resolve by name.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import PrimitiveArray
from marrow.builders import array
from marrow.dtypes import Int64Type, int64, uint32, string
from marrow.tabular import RecordBatch, record_batch
from marrow.expr.values import Add, Sub, Lt, Gt, Eq, Length, col


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
# BoolValue nodes — Lt / Gt / Eq
# ---------------------------------------------------------------------------


def test_lt_fuses() raises:
    """Lt(col, col) produces the correct bit-packed BoolArray."""
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var lt = Lt(col("a", int64), col("b", int64))
    var result = lt.execute(batch)
    assert_true(result[0].value())
    assert_true(not result[1].value())
    assert_true(result[2].value())
    assert_true(not result[3].value())
    assert_true(result[4].value())


def test_gt_fuses() raises:
    """Gt(col, col) produces the correct bit-packed BoolArray."""
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var gt = Gt(col("a", int64), col("b", int64))
    var result = gt.execute(batch)
    assert_true(not result[0].value())
    assert_true(result[1].value())
    assert_true(not result[2].value())
    assert_true(result[3].value())
    assert_true(not result[4].value())


def test_eq_fuses() raises:
    """Eq(col, col) produces the correct bit-packed BoolArray."""
    var a = array([1, 4, 3, 4, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var eq = Eq(col("a", int64), col("b", int64))
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

    var added = Add(col("a", int64), col("b", int64))
    var pred = Gt(added, col("b", int64))
    var result = pred.execute(batch)
    # a + b = [5, 9, 7, 12, 6]; compared to b = [4, 4, 4, 4, 4] -> all True
    for i in range(5):
        assert_true(result[i].value())


def test_comparison_write_to() raises:
    """Lt/Gt/Eq.write_to() display the expected nested structure."""
    assert_equal(
        String(Lt(col("a", int64), col("b", int64))), "Lt(Col[a], Col[b])"
    )
    assert_equal(
        String(Gt(col("a", int64), col("b", int64))), "Gt(Col[a], Col[b])"
    )
    assert_equal(
        String(Eq(col("a", int64), col("b", int64))), "Eq(Col[a], Col[b])"
    )


def test_comparison_dtype_is_bool() raises:
    """Lt/Gt/Eq.dtype() reports bool_, matching BoolValue's default impl."""
    var lt = Lt(col("a", int64), col("b", int64))
    assert_true(lt.dtype().value().is_bool())


def main() raises:
    TestSuite.run[__functions_in_module()]()
