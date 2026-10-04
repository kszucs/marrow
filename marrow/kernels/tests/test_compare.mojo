# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from std.math import inf, nan
from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)

from ...arrays import (
    DictionaryArray,
    DynArray,
    FixedSizeListArray,
    Int32Array,
    PrimitiveArray,
    StringViewArray,
)
from ...buffers import Bitmap
from ...builders import Int32Builder, ListBuilder, StructBuilder
from ...dtypes import field, int32, string
from ...execution import ExecContext
from ...kernels.hashing import KeyCompare
from ...builders import (
    array,
    PrimitiveBuilder,
    BinaryLikeBuilder,
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
    BinaryLikeType,
    BinaryType,
    LargeBinaryType,
    StringType,
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


def test_equal_is_ieee_on_nan_and_nan_safe_is_not() raises:
    """The two answers marrow has to have, side by side — see `EqKernel` for
    which consumer needs which, and why `=` stays IEEE.

    `-0.0 == 0.0` is asserted true under *both* so that a later change cannot
    quietly make `EqKernel[nan_safe=True]` a bit comparison, which would answer
    false here and still pass every NaN assertion.
    """
    var q = nan[DType.float64]()
    var a = array([q, q, 0.0, 1.0], float64)
    var b = array([q, 1.0, -0.0, 1.0], float64)

    var ieee = EqKernel.apply[Float64Type](a, b)
    assert_false(ieee[0].value())
    assert_false(ieee[1].value())
    assert_true(ieee[2].value())
    assert_true(ieee[3].value())

    var total = EqKernel[nan_safe=True].apply[Float64Type](a, b)
    assert_true(total[0].value())
    assert_false(total[1].value())
    assert_true(total[2].value())
    assert_true(total[3].value())


def test_nan_safe_comparisons_take_sqls_total_order() raises:
    """`nan_safe=True` is SQL's comparison, measured against three engines.

    DuckDB 1.5.5, DataFusion 54.0.0 and Polars 1.43.2 all answer `nan = nan`
    true, `nan <> nan` false and `nan > 1.0` true — NaN is greater than every
    number and than `inf`. This is what `marrow.expr` binds; `pc.*` keeps
    pyarrow's IEEE answers, which `test_equal_is_ieee_on_nan_and_nan_safe_is_not`
    pins on the other side.

    `<=` and `>=` are the two that surprise: a NaN is `<=` and `>=` a NaN under
    the total order, where IEEE denies both.
    """
    var q = nan[DType.float64]()
    var i = inf[DType.float64]()
    #              nan~nan  nan~1.0  nan~inf  1.0~nan
    var a = array([q, q, q, 1.0], float64)
    var b = array([q, 1.0, i, q], float64)

    var eq = EqKernel[nan_safe=True].apply[Float64Type](a, b)
    assert_true(eq == array([True, False, False, False]))

    var ne = NeKernel[nan_safe=True].apply[Float64Type](a, b)
    assert_true(ne == array([False, True, True, True]))

    var lt = LtKernel[nan_safe=True].apply[Float64Type](a, b)
    assert_true(lt == array([False, False, False, True]))
    var le = LeKernel[nan_safe=True].apply[Float64Type](a, b)
    assert_true(le == array([True, False, False, True]))
    var gt = GtKernel[nan_safe=True].apply[Float64Type](a, b)
    assert_true(gt == array([False, True, True, False]))
    var ge = GeKernel[nan_safe=True].apply[Float64Type](a, b)
    assert_true(ge == array([True, True, True, False]))


def test_not_equal_is_unordered_on_nan() raises:
    """`nan <> 1.0` is true, and `a.ne(b)` does not say so.

    Mojo's `SIMD.ne` lowers to an *ordered* compare, so it answered False
    whenever either operand was a NaN — `pc.not_equal(nan, 1.0)` came out False
    where `pyarrow.compute.not_equal` says True, which is a parity bug
    independent of the total-order question. `NeKernel` negates `EqKernel`
    instead, which is right under both rules by construction.
    """
    var q = nan[DType.float64]()
    var a = array([q, q, 2.0, 3.0], float64)
    var b = array([q, 1.0, 9.0, 3.0], float64)
    var ne = NeKernel.apply[Float64Type](a, b)
    assert_true(ne == array([True, True, True, False]))


def test_nan_safe_equality_leaves_nulls_alone() raises:
    """Null in, null out — the same rule `equal` follows, and deliberately.

    Whether two nulls are the same key is the *caller's* question and the
    answers differ — `KeyCompare` says they are, a filter's `=` says unknown —
    so `EqKernel[nan_safe=True]` corrects only NaN and leaves the validity for
    the caller to read. Pinned because the correction is inside the SIMD
    `core`, where a `select` over the validity instead of a lane op would be
    an easy and invisible way to lose it.
    """
    var a = Float64Builder(3)
    a.append(nan[DType.float64]())
    a.append_null()
    a.append(inf[DType.float64]())
    var b = Float64Builder(3)
    b.append_null()
    b.append_null()
    b.append(inf[DType.float64]())

    var total = EqKernel[nan_safe=True].apply[Float64Type](
        a.finish(), b.finish()
    )
    assert_equal(total.null_count(), 2)
    assert_true(total.is_null(0))
    assert_true(total.is_null(1))
    assert_true(total[2].value())


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
# The consequence reached well past comparison: statistics pruning is these very
# kernels run over per-chunk extremes, so no row group was ever pruned on a date
# or decimal predicate.
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


# ---------------------------------------------------------------------------
# equal — the "equality over an arbitrary dtype" primitive
#
# `nullif` is built on this. It used to pick its kernel family with
# `is_string() or is_large_string()`, so `binary` fell into the numeric arm and
# `dispatch_primitive` raised. The family test is
# `is_binary_like()` now: what selects the kernel is whether the payload is
# variable-width, not whether it is text.
# ---------------------------------------------------------------------------


def _bytes_pair[
    T: BinaryLikeType
](left: List[String], right: List[String]) raises -> Tuple[DynArray, DynArray]:
    var lb = BinaryLikeBuilder[T](len(left))
    for v in left:
        lb.append(v)
    var rb = BinaryLikeBuilder[T](len(right))
    for v in right:
        rb.append(v)
    var la: DynArray = lb.finish()
    var ra: DynArray = rb.finish()
    return (la^, ra^)


def _assert_equal_bytes[T: BinaryLikeType]() raises:
    var pair = _bytes_pair[T](["a", "bb", "c"], ["a", "xx", "c"])
    var r = equal(pair[0], pair[1])
    assert_true(r[0].value())
    assert_false(r[1].value())
    assert_true(r[2].value())


def test_equal_binary() raises:
    _assert_equal_bytes[BinaryType]()


def test_equal_large_binary() raises:
    _assert_equal_bytes[LargeBinaryType]()


def test_equal_string_unchanged() raises:
    _assert_equal_bytes[StringType]()


def test_equal_binary_nulls_propagate() raises:
    """Null on either side yields null out — same rule the string path used."""
    var lb = BinaryLikeBuilder[BinaryType](3)
    lb.append("a")
    lb.append_null()
    lb.append("c")
    var rb = BinaryLikeBuilder[BinaryType](3)
    rb.append("a")
    rb.append("b")
    rb.append_null()
    var r = equal(lb.finish(), rb.finish())
    assert_equal(r.null_count(), 2)
    assert_true(r.is_valid(0))
    assert_true(r[0].value())
    assert_false(r.is_valid(1))
    assert_false(r.is_valid(2))


def test_equal_mismatched_dtypes_raise() raises:
    """`equal` resolves the comptime type from the *left* dtype and reads
    the right operand at that same type, so mismatched dtypes must be rejected
    before the downcast rather than reaching `as_type` with the wrong one."""
    var sb = StringBuilder(1)
    sb.append("a")
    var bb = BinaryLikeBuilder[BinaryType](1)
    bb.append("a")
    with assert_raises():
        _ = equal(sb.finish(), bb.finish())


def test_equal_numeric_still_dispatches() raises:
    """The non-binarylike arm is unchanged."""
    var r = equal(array([1, 2, 3], int64), array([1, 9, 3], int64))
    assert_true(r[0].value())
    assert_false(r[1].value())
    assert_true(r[2].value())


# ---------------------------------------------------------------------------
# KeyCompare — key equality (IS NOT DISTINCT FROM), indexed on both sides
# ---------------------------------------------------------------------------


def _i64(values: List[Optional[Int]]) raises -> DynArray:
    return array[Int64Type](values, int64)


def _i32(values: List[Optional[Int]]) raises -> DynArray:
    return array[Int32Type](values, int32)


def _idx(values: List[Int]) raises -> Int32Array:
    var b = Int32Builder(capacity=len(values))
    for v in values:
        b.append(Int32(v))
    return b.finish()


def _same_keys(
    left: DynArray,
    li: List[Int],
    right: DynArray,
    ri: List[Int],
    ctx: ExecContext = ExecContext.serial(),
) raises -> List[Bool]:
    var n = len(li)
    var same = Bitmap.alloc_zeroed(n)
    same.set_range(0, n, True)
    KeyCompare.apply(left, _idx(li), right, _idx(ri), same, ctx)
    var out = List[Bool]()
    for i in range(n):
        out.append(same.test(i))
    return out^


def test_key_compare_nulls_are_one_value() raises:
    """NULL matches NULL and nothing else — unlike `=`, which answers NULL."""
    var a = _i64([1, None, 3, None])
    var got = _same_keys(a, [1, 1, 0, 2], a, [3, 0, 0, 2])
    assert_true(got == [True, False, True, True])


def test_key_compare_nan_and_signed_zero() raises:
    """One NaN, and `-0.0` is `0.0` — the identity the key hash hashes by."""
    var b = Float64Builder()
    for v in [nan[DType.float64](), nan[DType.float64](), -0.0, 0.0, 1.0]:
        b.append(v)
    var a: DynArray = b.finish()
    var got = _same_keys(a, [0, 2, 0, 4], a, [1, 3, 4, 4])
    assert_true(got == [True, True, False, True])


def test_key_compare_strings() raises:
    """Length first, then bytes; the empty string is not NULL."""
    var a: DynArray = array(["ab", "", None, "abc", "ab", ""])
    var got = _same_keys(a, [0, 1, 2, 0, 1], a, [4, 5, 2, 3, 2])
    assert_true(got == [True, True, True, False, False])


def test_key_compare_sliced_inputs() raises:
    """Indices are relative to each array's own offset."""
    var a = _i64([9, 9, 1, 2, 3])
    var b = _i64([1, 2, 3])
    var got = _same_keys(a.slice(2, 3), [0, 1, 2], b, [0, 2, 2])
    assert_true(got == [True, False, True])


def test_key_compare_lists_compare_elements() raises:
    """`[1, 2]` vs `[1, 2]`, `[2, 1]`, `[1]`, `[]` and NULL."""
    var lb = ListBuilder(Int32Builder(), capacity=5)
    var child_any = lb.values()
    ref child = child_any.as_int32()
    child.append(1)
    child.append(2)
    lb.append_valid()  # [1, 2]
    child.append(2)
    child.append(1)
    lb.append_valid()  # [2, 1]
    child.append(1)
    lb.append_valid()  # [1]
    lb.append_valid()  # []
    lb.append_null()  # null
    child.append(1)
    child.append(2)
    lb.append_valid()  # [1, 2]
    var a: DynArray = lb.finish().to_dyn()
    var got = _same_keys(a, [0, 0, 0, 0, 0, 3, 4], a, [5, 1, 2, 3, 4, 3, 4])
    assert_true(got == [True, False, False, False, False, True, True])


def test_key_compare_list_of_strings_is_not_concatenation() raises:
    """`["ab", "c"]` and `["a", "bc"]` are different keys."""
    var lb = ListBuilder(StringBuilder(), capacity=2)
    var child_any = lb.values()
    ref child = child_any.as_string()
    child.append("ab")
    child.append("c")
    lb.append_valid()
    child.append("a")
    child.append("bc")
    lb.append_valid()
    var a: DynArray = lb.finish().to_dyn()
    assert_true(_same_keys(a, [0], a, [1]) == [False])


def test_key_compare_fixed_size_lists() raises:
    """`[1, 2]` against `[1, 2]`, `[2, 1]` and NULL, read through a slice so
    the array's offset reaches the element positions: the first row, `[9, 9]`,
    is sliced off and would match nothing if the offset were dropped."""
    var values: DynArray = array([9, 9, 1, 2, 2, 1, 0, 0, 1, 2], int32)
    # PyArrow's mask convention: True marks a NULL row.
    var fsl = FixedSizeListArray.from_arrays(
        values^, 2, array([False, False, False, True, False])
    )
    var a: DynArray = fsl.slice(1)  # [1, 2], [2, 1], NULL, [1, 2]
    var got = _same_keys(a, [0, 0, 2, 2, 0], a, [3, 1, 2, 0, 0])
    assert_true(got == [True, False, True, False, True])


def test_key_compare_string_views() raises:
    """Inline (at most twelve bytes) and out-of-line views, two long keys of
    one length and prefix that differ only in their last byte, and NULL, read
    through a slice so the array's offset reaches every view."""
    var values = List[Optional[String]]()
    values.append(String("zzzz"))  # sliced off
    values.append(String("abcd"))
    values.append(String("abcd-a key longer than twelve"))
    values.append(String("abcd-a key longer than twelvf"))
    values.append(None)
    values.append(String("abcd-a key longer than twelve"))
    values.append(String("abcd"))
    var a: DynArray = StringViewArray.from_values(values).slice(1)
    var got = _same_keys(a, [0, 1, 1, 3, 3, 0], a, [5, 4, 2, 3, 0, 1])
    assert_true(got == [True, True, False, True, False, False])


def test_key_compare_structs() raises:
    """Field by field; a NULL struct is one value whatever its fields hold."""
    var sb = StructBuilder([field("a", int32), field("b", string)])
    var a_vals = [1, 1, 2, 1]
    var b_vals = ["x", "y", "x", "x"]
    for i in range(4):
        sb.field_builder(0).as_int32().append(Int32(a_vals[i]))
        sb.field_builder(1).as_string().append(b_vals[i])
        sb.append_valid()
    sb.field_builder(0).as_int32().append(7)
    sb.field_builder(1).as_string().append("z")
    sb.append_null()
    sb.field_builder(0).as_int32().append(8)
    sb.field_builder(1).as_string().append("w")
    sb.append_null()
    var s: DynArray = sb.finish().to_dyn()
    var got = _same_keys(s, [0, 0, 0, 4, 4], s, [3, 1, 2, 5, 0])
    assert_true(got == [True, False, False, True, False])


def test_key_compare_dictionaries_by_value() raises:
    """Two dictionaries, one value: equal by what the indices decode to."""
    var left: DynArray = DictionaryArray.from_arrays(
        _i32([0, 1]), array(["a", "b"])
    )
    var right: DynArray = DictionaryArray.from_arrays(
        _i32([1, 0]), array(["b", "a"])
    )
    var got = _same_keys(left, [0, 1, 0], right, [0, 1, 1])
    assert_true(got == [True, True, False])


def test_key_compare_skips_rows_already_distinct() raises:
    """A cleared bit stays cleared: columns AND into one bitmap."""
    var a = _i64([1, 1])
    var same = Bitmap.alloc_zeroed(1)
    KeyCompare.apply(a, _idx([0]), a, _idx([1]), same)
    assert_false(same.test(0))


def test_key_compare_striped_matches_serial() raises:
    """Stripes are 64-row aligned, so parallel workers never share a byte of
    the result; the answer is the serial one."""
    var n = 10_000
    var b = Int64Builder(capacity=n)
    for i in range(n):
        b.append(Int64(i % 97))
    var a: DynArray = b.finish()
    var li = List[Int]()
    var ri = List[Int]()
    for i in range(n):
        li.append(i)
        ri.append((i * 7) % n)
    var serial = _same_keys(a, li, a, ri)
    var striped = _same_keys(a, li, a, ri, ExecContext.parallel(4))
    assert_true(serial == striped)
