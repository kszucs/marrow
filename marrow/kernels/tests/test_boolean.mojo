"""Unit tests for the boolean compute kernels (marrow.kernels.boolean).

The expected null/value patterns match PyArrow's Kleene kernels
(`pc.and_kleene`, `pc.or_kleene`, `pc.xor`, `pc.invert`).
"""

from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)

from ...arrays import BoolArray
from ...builders import array

from ...kernels.boolean import (
    AndKernel,
    OrKernel,
    NotKernel,
    XorKernel,
)


def _a() raises -> BoolArray:
    """Left operand covering the full 3-valued cross product."""
    return array([True, True, True, False, False, False, None, None, None])


def _b() raises -> BoolArray:
    """Right operand covering the full 3-valued cross product."""
    return array([True, False, None, True, False, None, True, False, None])


# --- Kleene AND ------------------------------------------------------------


def test_and_no_nulls() raises:
    var a = array([True, True, False, False])
    var b = array([True, False, True, False])
    assert_true(AndKernel.apply(a, b) == array([True, False, False, False]))


def test_and_kleene() raises:
    # pc.and_kleene: TRUE&NULL=NULL, FALSE&NULL=FALSE, NULL&NULL=NULL
    var r = AndKernel.apply(_a(), _b())
    assert_true(
        r == array([True, False, None, False, False, False, None, False, None])
    )
    assert_equal(r.null_count(), 3)


def test_and_recovers_with_valid_operand() raises:
    # FALSE dominates: NULL AND FALSE = FALSE, so a null operand can still yield
    # a fully-valid result.
    var a = array([True, False, None])
    var b = array([False, False, False])
    var r = AndKernel.apply(a, b)
    assert_equal(r.null_count(), 0)
    assert_true(r == array([False, False, False]))


# --- Kleene OR -------------------------------------------------------------


def test_or_no_nulls() raises:
    var a = array([True, True, False, False])
    var b = array([True, False, True, False])
    assert_true(OrKernel.apply(a, b) == array([True, True, True, False]))


def test_or_kleene() raises:
    # pc.or_kleene: TRUE|NULL=TRUE, FALSE|NULL=NULL, NULL|NULL=NULL
    var r = OrKernel.apply(_a(), _b())
    assert_true(
        r == array([True, True, True, True, False, None, True, None, None])
    )
    assert_equal(r.null_count(), 3)


def test_or_recovers_with_valid_operand() raises:
    # TRUE dominates: NULL OR TRUE = TRUE.
    var a = array([True, False, None])
    var b = array([True, True, True])
    var r = OrKernel.apply(a, b)
    assert_equal(r.null_count(), 0)
    assert_true(r == array([True, True, True]))


# --- XOR (two-valued null) -------------------------------------------------


def test_xor_no_nulls() raises:
    var a = array([True, True, False, False])
    var b = array([True, False, True, False])
    assert_true(XorKernel.apply(a, b) == array([False, True, True, False]))


def test_xor_nulls() raises:
    # pc.xor: valid only where BOTH operands are valid.
    var r = XorKernel.apply(_a(), _b())
    assert_true(
        r == array([False, True, None, True, False, None, None, None, None])
    )
    assert_equal(r.null_count(), 5)


# --- NOT (null propagates) -------------------------------------------------


def test_not_no_nulls() raises:
    assert_true(
        NotKernel.apply(array([True, False, True]))
        == array([False, True, False])
    )


def test_not_propagates_nulls() raises:
    # pc.invert: NOT NULL = NULL; validity is unchanged.
    var r = NotKernel.apply(_a())
    assert_true(
        r == array([False, False, False, True, True, True, None, None, None])
    )
    assert_equal(r.null_count(), 3)


# --- errors ----------------------------------------------------------------


def test_length_mismatch_raises() raises:
    var a = array([True, False])
    var b = array([True])
    with assert_raises():
        _ = AndKernel.apply(a, b)
