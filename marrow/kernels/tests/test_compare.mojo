from std.testing import assert_equal, assert_true, assert_false

from ...arrays import PrimitiveArray, DynArray
from ...builders import (
    array,
    PrimitiveBuilder,
    Int64Builder,
    Float64Builder,
    StringBuilder,
)
from ...dtypes import (
    int64,
    float64,
    Int64Type,
    Float64Type,
    Int32Type,
    large_string,
    date32,
    duration,
    second,
    decimal128,
    Date32Type,
    Decimal128Type,
)
from ...builders import Date32Builder, Decimal128Builder
from ...kernels.cast import cast

from ...kernels.string import (
    StringEqKernel,
    StringNeKernel,
    StringLtKernel,
    StringLeKernel,
    StringGtKernel,
    StringGeKernel,
)
from ...kernels.numeric import (
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


def test_compare_length_mismatch_raises() raises:
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


def test_compare_single_element() raises:
    """Comparisons work on length-1 arrays."""
    var a = array([7], int64)
    var b = array([7], int64)
    assert_true(EqKernel.apply[Int64Type](a, b)[0].value())
    assert_false(LtKernel.apply[Int64Type](a, b)[0].value())


# ---------------------------------------------------------------------------
# Non-SIMD-aligned length
# ---------------------------------------------------------------------------


def test_compare_non_aligned_length() raises:
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
# Runtime-typed DynArray overloads
# ---------------------------------------------------------------------------


def test_equal_array_overload() raises:
    """Type-erased EqKernel.dispatch(DynArray, DynArray) resolves the dtype."""
    var a: DynArray = array([1, 2, 3], int64)
    var b: DynArray = array([1, 0, 3], int64)
    var result = EqKernel.dispatch(a, b)
    assert_equal(result.length(), 3)


def test_dtype_mismatch_raises() raises:
    """Type-erased kernels raise on dtype mismatch."""
    var a: DynArray = array([1, 2, 3], int64)
    var fb = Float64Builder(3)
    fb.unsafe_append(1.0)
    fb.unsafe_append(2.0)
    fb.unsafe_append(3.0)
    var b: DynArray = fb.finish()
    var raised = False
    try:
        _ = EqKernel.dispatch(a, b)
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


# ---------------------------------------------------------------------------
# String ordering comparisons (lexicographic byte order, matches pyarrow)
# ---------------------------------------------------------------------------


def test_string_less() raises:
    var a = array(["apple", "banana", "cherry", "apple", ""])
    var b = array(["apricot", "banana", "cherry", "ab", "a"])
    assert_true(
        StringLtKernel.apply(a, b) == array([True, False, False, False, True])
    )


def test_string_less_equal() raises:
    var a = array(["apple", "banana", "cherry", "apple", ""])
    var b = array(["apricot", "banana", "cherry", "ab", "a"])
    assert_true(
        StringLeKernel.apply(a, b) == array([True, True, True, False, True])
    )


def test_string_greater() raises:
    var a = array(["apple", "banana", "cherry", "apple", ""])
    var b = array(["apricot", "banana", "cherry", "ab", "a"])
    assert_true(
        StringGtKernel.apply(a, b) == array([False, False, False, True, False])
    )


def test_string_greater_equal() raises:
    var a = array(["apple", "banana", "cherry", "apple", ""])
    var b = array(["apricot", "banana", "cherry", "ab", "a"])
    assert_true(
        StringGeKernel.apply(a, b) == array([False, True, True, True, False])
    )


def test_string_equal_via_kernel() raises:
    var a = array(["x", "yy", "z"])
    var b = array(["x", "yz", "z"])
    assert_true(StringEqKernel.apply(a, b) == array([True, False, True]))
    assert_true(StringNeKernel.apply(a, b) == array([False, True, False]))


def test_string_prefix_ordering() raises:
    # a shorter string that is a prefix compares less than the longer one
    var a = array(["ab", "abc", "abc"])
    var b = array(["abc", "ab", "abc"])
    assert_true(StringLtKernel.apply(a, b) == array([True, False, False]))
    assert_true(StringGtKernel.apply(a, b) == array([False, True, False]))


def test_string_compare_nulls() raises:
    # validity = AND of operands; null positions are invalid in the output
    var lb = StringBuilder(capacity=3)
    lb.append("x")
    lb.append_null()
    lb.append("y")
    var rb = StringBuilder(capacity=3)
    rb.append("x")
    rb.append("z")
    rb.append_null()
    var left = lb.finish()
    var right = rb.finish()
    var r = StringLtKernel.apply(left, right)
    assert_equal(r.null_count(), 2)
    assert_true(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_false(r.is_valid(2))
    assert_false(r[0].value())  # 'x' < 'x' is False


def test_string_dispatch_anyarray() raises:
    """String ordering goes through the string kernel family: `LtKernel` is
    numeric-only and would not resolve a string dtype."""
    var a: DynArray = array(["a", "bb", "c"])
    var b: DynArray = array(["b", "bb", "a"])
    var r = StringLtKernel.dispatch(a, b)
    assert_equal(r.length(), 3)
    ref rb = r.as_bool()
    assert_true(rb[0].value())  # 'a' < 'b'
    assert_false(rb[1].value())  # 'bb' == 'bb'
    assert_false(rb[2].value())  # 'c' > 'a'


def test_large_string_ordering() raises:
    var a = cast(array(["apple", "banana", "cherry"]), large_string)
    var b = cast(array(["apricot", "banana", "berry"]), large_string)
    var r = StringLtKernel.dispatch(a, b)
    ref rb = r.as_bool()
    assert_true(rb[0].value())  # apple < apricot
    assert_false(rb[1].value())  # banana == banana
    assert_false(rb[2].value())  # cherry > berry


# ---------------------------------------------------------------------------
# M1.0 — the erased comparison must accept every dtype its typed leaf accepts.
#
# `apply` is bound on `PrimitiveType`; `dispatch` narrowed to `NumericType`, so
# runtime-typed comparison raised on temporal, interval and decimal columns even
# though the leaf handles them. Exactly the defect CLAUDE.md's "dispatch on the
# widest family the typed leaf accepts" rule names, and already fixed in
# `filter`/`take` and `sort`.
#
# The consequence reached well past comparison: `pruning.mojo` mirrors this
# bound, so no row group or page was ever pruned on a date or decimal predicate.
# ---------------------------------------------------------------------------


def _date32_arr(vals: List[Int]) raises -> DynArray:
    var b = Date32Builder(date32(), len(vals))
    for v in vals:
        b.append(Scalar[Date32Type.native](v))
    return b.finish()


def test_erased_compare_accepts_date32() raises:
    """A date column compares through the erased dispatch."""
    var a = _date32_arr([19000, 18500, 19100])
    var b = _date32_arr([19000, 19000, 18000])
    ref r = LtKernel.dispatch(a, b).as_bool()
    assert_false(r[0].value())
    assert_true(r[1].value())
    assert_false(r[2].value())


def test_erased_compare_accepts_decimal128() raises:
    """A decimal column compares through the erased dispatch."""
    var d = decimal128(10, 2)
    var ab = Decimal128Builder(d, 2)
    ab.append(Scalar[Decimal128Type.native](150))
    ab.append(Scalar[Decimal128Type.native](250))
    var bb = Decimal128Builder(d, 2)
    bb.append(Scalar[Decimal128Type.native](200))
    bb.append(Scalar[Decimal128Type.native](200))
    ref r = LtKernel.dispatch(ab.finish(), bb.finish()).as_bool()
    assert_true(r[0].value())
    assert_false(r[1].value())
