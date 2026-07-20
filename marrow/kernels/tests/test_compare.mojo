from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite

from marrow.arrays import PrimitiveArray, AnyArray
from marrow.builders import (
    array,
    PrimitiveBuilder,
    Int64Builder,
    Float64Builder,
)
from marrow.dtypes import int64, float64, Int64Type, Float64Type

from marrow.kernels.compare import (
    equal,
    EqKernel,
    NeKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
)


# ---------------------------------------------------------------------------
# Typed overloads — int64
# ---------------------------------------------------------------------------


def test_equal_true_and_false() raises:
    """Equal: True where values match, False elsewhere."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([1, 0, 3, 0, 5], int64)
    var result = EqKernel.apply[Int64Type](a, b)

    assert_true(result[0].value())  # 1 == 1
    assert_false(result[1].value())  # 2 != 0
    assert_true(result[2].value())  # 3 == 3
    assert_false(result[3].value())  # 4 != 0
    assert_true(result[4].value())  # 5 == 5


def test_not_equal() raises:
    """``not_equal`` is the inverse of equal."""
    var a = array([1, 2, 3], int64)
    var b = array([1, 9, 3], int64)
    var result = NeKernel.apply[Int64Type](a, b)

    assert_false(result[0].value())  # 1 == 1
    assert_true(result[1].value())  # 2 != 9
    assert_false(result[2].value())  # 3 == 3


def test_less() raises:
    """``less``: True where a < b."""
    var a = array([1, 5, 3, 10], int64)
    var b = array([5, 1, 3, 20], int64)
    var result = LtKernel.apply[Int64Type](a, b)

    assert_true(result[0].value())  # 1 < 5
    assert_false(result[1].value())  # 5 > 1
    assert_false(result[2].value())  # 3 == 3, not strictly less
    assert_true(result[3].value())  # 10 < 20


def test_less_equal() raises:
    """``less_equal``: True where a <= b."""
    var a = array([1, 5, 3, 10], int64)
    var b = array([5, 1, 3, 20], int64)
    var result = LeKernel.apply[Int64Type](a, b)

    assert_true(result[0].value())  # 1 <= 5
    assert_false(result[1].value())  # 5 > 1
    assert_true(result[2].value())  # 3 <= 3
    assert_true(result[3].value())  # 10 <= 20


def test_greater() raises:
    """``greater``: True where a > b."""
    var a = array([5, 1, 3, 20], int64)
    var b = array([1, 5, 3, 10], int64)
    var result = GtKernel.apply[Int64Type](a, b)

    assert_true(result[0].value())  # 5 > 1
    assert_false(result[1].value())  # 1 < 5
    assert_false(result[2].value())  # 3 == 3
    assert_true(result[3].value())  # 20 > 10


def test_greater_equal() raises:
    """``greater_equal``: True where a >= b."""
    var a = array([5, 1, 3, 20], int64)
    var b = array([1, 5, 3, 10], int64)
    var result = GeKernel.apply[Int64Type](a, b)

    assert_true(result[0].value())  # 5 >= 1
    assert_false(result[1].value())  # 1 < 5
    assert_true(result[2].value())  # 3 >= 3
    assert_true(result[3].value())  # 20 >= 10


# ---------------------------------------------------------------------------
# Float64
# ---------------------------------------------------------------------------


def test_less_float64() raises:
    """``less`` works for float64."""
    var ab = Float64Builder(3)
    ab.unsafe_append(1.0)
    ab.unsafe_append(2.5)
    ab.unsafe_append(3.0)
    var bb = Float64Builder(3)
    bb.unsafe_append(1.0)
    bb.unsafe_append(2.0)
    bb.unsafe_append(5.0)
    var a = ab.finish()
    var b = bb.finish()
    var result = LtKernel.apply[Float64Type](a, b)

    assert_false(result[0].value())  # 1.0 == 1.0
    assert_false(result[1].value())  # 2.5 > 2.0
    assert_true(result[2].value())  # 3.0 < 5.0


# ---------------------------------------------------------------------------
# Length validation
# ---------------------------------------------------------------------------


def test_length_mismatch_raises() raises:
    """Comparison of arrays with different lengths raises an error."""
    var a = array([1, 2, 3], int64)
    var b = array([1, 2], int64)
    var raised = False
    try:
        _ = EqKernel.apply[Int64Type](a, b)
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# Single element
# ---------------------------------------------------------------------------


def test_single_element() raises:
    """Comparisons work on length-1 arrays."""
    var a = array([7], int64)
    var b = array([7], int64)
    assert_true(EqKernel.apply[Int64Type](a, b)[0].value())
    assert_false(LtKernel.apply[Int64Type](a, b)[0].value())


# ---------------------------------------------------------------------------
# Non-SIMD-aligned length
# ---------------------------------------------------------------------------


def test_non_aligned_length() raises:
    """Comparisons work on lengths that are not multiples of SIMD width."""
    var n = 7
    var a = array([1, 2, 3, 4, 5, 6, 7], int64)
    var b = array([7, 6, 5, 4, 3, 2, 1], int64)
    var result = LtKernel.apply[Int64Type](a, b)

    for i in range(n):
        var expected = a[i].value() < b[i].value()
        assert_equal(result[i], expected)


# ---------------------------------------------------------------------------
# Output type is bool_
# ---------------------------------------------------------------------------


def test_output_length() raises:
    """Output array has the same length as inputs."""
    var a = array([10, 20, 30, 40, 50], int64)
    var b = array([10, 10, 40, 40, 40], int64)
    var result = GeKernel.apply[Int64Type](a, b)
    assert_equal(len(result), 5)


# ---------------------------------------------------------------------------
# Runtime-typed AnyArray overloads
# ---------------------------------------------------------------------------


def test_equal_array_overload() raises:
    """Type-erased equal(AnyArray, AnyArray) dispatches correctly."""
    var a: AnyArray = array([1, 2, 3], int64)
    var b: AnyArray = array([1, 0, 3], int64)
    var result = equal(a, b)
    assert_equal(result.length(), 3)


def test_dtype_mismatch_raises() raises:
    """Type-erased kernels raise on dtype mismatch."""
    var a: AnyArray = array([1, 2, 3], int64)
    var fb = Float64Builder(3)
    fb.unsafe_append(1.0)
    fb.unsafe_append(2.0)
    fb.unsafe_append(3.0)
    var b: AnyArray = fb.finish()
    var raised = False
    try:
        _ = equal(a, b)
    except:
        raised = True
    assert_true(raised)


def test_equal_large_array() raises:
    """Regression: equal must write all bitmap bytes, not just the first
    of each SIMD batch (previously only byte 0 of every 16 was written)."""
    var n = 200
    var ab = Int64Builder(n)
    var bb = Int64Builder(n)
    for i in range(n):
        ab.unsafe_append(Scalar[int64.native](i))
        bb.unsafe_append(Scalar[int64.native](i))
    var a = ab.finish()
    var b = bb.finish()
    var result = EqKernel.apply[Int64Type](a, b)
    assert_equal(len(result), n)
    for i in range(n):
        assert_true(result[i].value())


def main() raises:
    TestSuite.run[__functions_in_module()]()
