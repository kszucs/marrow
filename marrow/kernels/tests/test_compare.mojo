from std.testing import assert_equal, assert_true, assert_false, TestSuite

from marrow.arrays import PrimitiveArray, AnyArray
from marrow.builders import array, PrimitiveBuilder
from marrow.dtypes import int64, float64

from marrow.kernels.compare import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)


# ---------------------------------------------------------------------------
# Typed overloads — int64
# ---------------------------------------------------------------------------


def test_equal_true_and_false() raises:
    """Equal: True where values match, False elsewhere."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([1, 0, 3, 0, 5])
    var result = equal[int64](a, b)

    assert_true(result[0])  # 1 == 1
    assert_false(result[1])  # 2 != 0
    assert_true(result[2])  # 3 == 3
    assert_false(result[3])  # 4 != 0
    assert_true(result[4])  # 5 == 5


def test_not_equal() raises:
    """``not_equal`` is the inverse of equal."""
    var a = array[int64]([1, 2, 3])
    var b = array[int64]([1, 9, 3])
    var result = not_equal[int64](a, b)

    assert_false(result[0])  # 1 == 1
    assert_true(result[1])  # 2 != 9
    assert_false(result[2])  # 3 == 3


def test_less() raises:
    """``less``: True where a < b."""
    var a = array[int64]([1, 5, 3, 10])
    var b = array[int64]([5, 1, 3, 20])
    var result = less[int64](a, b)

    assert_true(result[0])  # 1 < 5
    assert_false(result[1])  # 5 > 1
    assert_false(result[2])  # 3 == 3, not strictly less
    assert_true(result[3])  # 10 < 20


def test_less_equal() raises:
    """``less_equal``: True where a <= b."""
    var a = array[int64]([1, 5, 3, 10])
    var b = array[int64]([5, 1, 3, 20])
    var result = less_equal[int64](a, b)

    assert_true(result[0])  # 1 <= 5
    assert_false(result[1])  # 5 > 1
    assert_true(result[2])  # 3 <= 3
    assert_true(result[3])  # 10 <= 20


def test_greater() raises:
    """``greater``: True where a > b."""
    var a = array[int64]([5, 1, 3, 20])
    var b = array[int64]([1, 5, 3, 10])
    var result = greater[int64](a, b)

    assert_true(result[0])  # 5 > 1
    assert_false(result[1])  # 1 < 5
    assert_false(result[2])  # 3 == 3
    assert_true(result[3])  # 20 > 10


def test_greater_equal() raises:
    """``greater_equal``: True where a >= b."""
    var a = array[int64]([5, 1, 3, 20])
    var b = array[int64]([1, 5, 3, 10])
    var result = greater_equal[int64](a, b)

    assert_true(result[0])  # 5 >= 1
    assert_false(result[1])  # 1 < 5
    assert_true(result[2])  # 3 >= 3
    assert_true(result[3])  # 20 >= 10


# ---------------------------------------------------------------------------
# Float64
# ---------------------------------------------------------------------------


def test_less_float64() raises:
    """``less`` works for float64."""
    var ab = PrimitiveBuilder[float64](3)
    ab.unsafe_append(1.0)
    ab.unsafe_append(2.5)
    ab.unsafe_append(3.0)
    var bb = PrimitiveBuilder[float64](3)
    bb.unsafe_append(1.0)
    bb.unsafe_append(2.0)
    bb.unsafe_append(5.0)
    var a = ab.finish()
    var b = bb.finish()
    var result = less[float64](a, b)

    assert_false(result[0])  # 1.0 == 1.0
    assert_false(result[1])  # 2.5 > 2.0
    assert_true(result[2])  # 3.0 < 5.0


# ---------------------------------------------------------------------------
# Length validation
# ---------------------------------------------------------------------------


def test_length_mismatch_raises() raises:
    """Comparison of arrays with different lengths raises an error."""
    var a = array[int64]([1, 2, 3])
    var b = array[int64]([1, 2])
    var raised = False
    try:
        _ = equal[int64](a, b)
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# Single element
# ---------------------------------------------------------------------------


def test_single_element() raises:
    """Comparisons work on length-1 arrays."""
    var a = array[int64]([7])
    var b = array[int64]([7])
    assert_true(equal[int64](a, b)[0])
    assert_false(less[int64](a, b)[0])


# ---------------------------------------------------------------------------
# Non-SIMD-aligned length
# ---------------------------------------------------------------------------


def test_non_aligned_length() raises:
    """Comparisons work on lengths that are not multiples of SIMD width."""
    var n = 7
    var a = array[int64]([1, 2, 3, 4, 5, 6, 7])
    var b = array[int64]([7, 6, 5, 4, 3, 2, 1])
    var result = less[int64](a, b)

    for i in range(n):
        var expected = a[i].value() < b[i].value()
        assert_equal(result[i], expected)


# ---------------------------------------------------------------------------
# Output type is bool_
# ---------------------------------------------------------------------------


def test_output_length() raises:
    """Output array has the same length as inputs."""
    var a = array[int64]([10, 20, 30, 40, 50])
    var b = array[int64]([10, 10, 40, 40, 40])
    var result = greater_equal[int64](a, b)
    assert_equal(len(result), 5)


# ---------------------------------------------------------------------------
# Runtime-typed AnyArray overloads
# ---------------------------------------------------------------------------


def test_equal_array_overload() raises:
    """Type-erased equal(AnyArray, AnyArray) dispatches correctly."""
    var a = AnyArray(array[int64]([1, 2, 3]))
    var b = AnyArray(array[int64]([1, 0, 3]))
    var result = equal(a, b)
    assert_equal(result.length(), 3)


def test_dtype_mismatch_raises() raises:
    """Type-erased kernels raise on dtype mismatch."""
    var a = AnyArray(array[int64]([1, 2, 3]))
    var fb = PrimitiveBuilder[float64](3)
    fb.unsafe_append(1.0)
    fb.unsafe_append(2.0)
    fb.unsafe_append(3.0)
    var b = AnyArray(fb.finish())
    var raised = False
    try:
        _ = equal(a, b)
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
