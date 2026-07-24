"""Cross-driver parity harness for the expression system.

Every stable op is expressible two ways: as a fused comptime ``Value`` tree
(``values.mojo``) and as a runtime tag-based ``DynValue`` tree (``dynamic.mojo``).
Both box into the shared ``AnyValue`` and expose ``execute(batch) -> AnyArray``.
This suite asserts the two drivers agree element-for-element on the same input,
so the runtime interpreter can never silently diverge from the fused algebra it
mirrors.

``assert_parity`` is the reusable primitive: hand it a fused ``Value`` and an
equivalent ``DynValue`` (each implicitly boxed into ``AnyValue``) plus a
``RecordBatch``; it runs both and asserts the resulting arrays are equal. The
fused column leaves resolve by name (``col("a", int64)``) and the ``DynValue``
leaves by position (``col(0)``), so a batch whose columns are named ``a``/``b``
lets the two trees reference the same data.

Seeded with STABLE ops that exist on both drivers: arithmetic
(``+ - * mod floordiv``), numeric comparisons, ``cast``, and ``if_else``. There
is no fused ``if_else`` node, so its fused reference is the equivalent branch-free
mask identity ``a*c + b*(1-c)`` (with ``c = (a > b)`` promoted to int via
``BoolToNum``) — a genuinely different fused path that must still match the
runtime ``select``.
"""

from std.testing import assert_true
from marrow.testing import TestSuite

from marrow.arrays import AnyArray, Int64Array
from marrow.builders import array
from marrow.dtypes import int64, float64, Int64Type, Float64Type
from marrow.tabular import RecordBatch, record_batch

# Fused comptime algebra (values.mojo)
from marrow.expr.values import (
    AnyValue,
    col as fcol,
    lit as flit,
    NumericCast,
    BoolToNum,
    IsNull,
)

# Runtime tag interpreter (dynamic.mojo)
from marrow.expr.dynamic import (
    DynValue,
    col as dcol,
    lit as dlit,
    if_else,
)


def assert_parity(
    var fused: AnyValue, var dyn: AnyValue, batch: RecordBatch
) raises:
    """Execute both drivers against *batch* and assert the arrays are equal."""
    var expected = fused.execute(batch)
    var actual = dyn.execute(batch)
    assert_true(expected == actual)


def _ab_batch() raises -> RecordBatch:
    """A two-column int64 batch (columns ``a`` and ``b``) shared by the cases.
    """
    var a = array([1, 5, 3, 10, 7, 2], int64)
    var b = array([9, 2, 3, 1, 4, 8], int64)
    return record_batch([a^, b^], names=["a", "b"])


# ---------------------------------------------------------------------------
# Arithmetic: + - * mod floordiv
# ---------------------------------------------------------------------------


def test_parity_add() raises:
    assert_parity(
        fcol("a", int64) + fcol("b", int64), dcol(0) + dcol(1), _ab_batch()
    )


def test_parity_sub() raises:
    assert_parity(
        fcol("a", int64) - fcol("b", int64), dcol(0) - dcol(1), _ab_batch()
    )


def test_parity_mul() raises:
    assert_parity(
        fcol("a", int64) * fcol("b", int64), dcol(0) * dcol(1), _ab_batch()
    )


def test_parity_mod() raises:
    assert_parity(
        fcol("a", int64) % fcol("b", int64), dcol(0) % dcol(1), _ab_batch()
    )


def test_parity_floordiv() raises:
    assert_parity(
        fcol("a", int64) // fcol("b", int64), dcol(0) // dcol(1), _ab_batch()
    )


# ---------------------------------------------------------------------------
# Numeric comparisons
# ---------------------------------------------------------------------------


def test_parity_gt() raises:
    assert_parity(
        fcol("a", int64) > fcol("b", int64), dcol(0) > dcol(1), _ab_batch()
    )


def test_parity_lt() raises:
    assert_parity(
        fcol("a", int64) < fcol("b", int64), dcol(0) < dcol(1), _ab_batch()
    )


def test_parity_eq() raises:
    assert_parity(
        fcol("a", int64) == fcol("b", int64), dcol(0) == dcol(1), _ab_batch()
    )


# ---------------------------------------------------------------------------
# Cast (there is a fused NumericCast node)
# ---------------------------------------------------------------------------


def test_parity_cast() raises:
    assert_parity(
        NumericCast[Float64Type](fcol("a", int64)),
        dcol(0).cast(float64),
        _ab_batch(),
    )


# ---------------------------------------------------------------------------
# if_else — no fused node; reference is the equivalent mask identity
#   if_else(a > b, a, b) == a*c + b*(1 - c),  c = (a > b) as int64
# ---------------------------------------------------------------------------


def test_parity_if_else() raises:
    var cnum = BoolToNum[Int64Type](fcol("a", int64) > fcol("b", int64))
    var one = flit(1, int64)
    var fused_ref = fcol("a", int64) * cnum.copy() + fcol("b", int64) * (
        one - cnum.copy()
    )
    var dyn_expr = if_else(dcol(0) > dcol(1), dcol(0), dcol(1))
    assert_parity(fused_ref, dyn_expr, _ab_batch())


# ---------------------------------------------------------------------------
# Null propagation through the fused lane (T0.7). A shared nullable batch: `a`
# and `b` each carry nulls in different rows, so an AND-combine of their
# validities is a non-trivial pattern the fused lane must reproduce.
# ---------------------------------------------------------------------------


def _nullable_ab_batch() raises -> RecordBatch:
    var a = array([1, None, 3, None, 7, 2], int64)
    var b = array([9, 2, None, 1, None, 8], int64)
    return record_batch([a^, b^], names=["a", "b"])


def test_parity_add_nulls() raises:
    # a + b nulls where either operand is null — fused AND-combine == dynamic.
    assert_parity(
        fcol("a", int64) + fcol("b", int64),
        dcol(0) + dcol(1),
        _nullable_ab_batch(),
    )


def test_parity_mul_nulls() raises:
    assert_parity(
        fcol("a", int64) * fcol("b", int64),
        dcol(0) * dcol(1),
        _nullable_ab_batch(),
    )


def test_parity_gt_nulls() raises:
    # (a > b) is valid only where both operands are valid.
    assert_parity(
        fcol("a", int64) > fcol("b", int64),
        dcol(0) > dcol(1),
        _nullable_ab_batch(),
    )


def test_parity_cast_nulls() raises:
    # cast preserves the operand's validity.
    assert_parity(
        NumericCast[Float64Type](fcol("a", int64)),
        dcol(0).cast(float64),
        _nullable_ab_batch(),
    )


def test_parity_isnull_never_null() raises:
    # an IS NULL result is itself always valid (no null bit set).
    assert_parity(
        IsNull(fcol("a", int64)), dcol(0).is_null(), _nullable_ab_batch()
    )


# ---------------------------------------------------------------------------
# Kleene 3-valued and_/or_ over nullable masks (T0.7). The fused `BoolValue` lane
# now tracks validity, reusing the null-correct `AndKernel`/`OrKernel` (Kleene,
# fixed in T0.1) that the runtime `DynValue` path already routes through, so the
# two drivers agree element-for-element — including where a known-false operand
# forces a valid AND result and a known-true operand forces a valid OR result.
# ---------------------------------------------------------------------------


def test_parity_and_kleene() raises:
    var fused = (fcol("a", int64) > flit(0, int64)) & (
        fcol("b", int64) > flit(0, int64)
    )
    var dyn = (dcol(0) > dlit[Int64Type](0)) & (dcol(1) > dlit[Int64Type](0))
    assert_parity(fused, dyn, _nullable_ab_batch())


def test_parity_or_kleene() raises:
    var fused = (fcol("a", int64) > flit(0, int64)) | (
        fcol("b", int64) > flit(0, int64)
    )
    var dyn = (dcol(0) > dlit[Int64Type](0)) | (dcol(1) > dlit[Int64Type](0))
    assert_parity(fused, dyn, _nullable_ab_batch())


def main() raises:
    TestSuite.run[__functions_in_module()]()
