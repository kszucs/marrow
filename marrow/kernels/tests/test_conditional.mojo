"""Unit tests for the conditional / null-handling kernels
(marrow.kernels.conditional).

Expected value/null patterns match PyArrow's ``pc.case_when``, ``pc.coalesce``,
``pc.if_else`` (SQL ``NULLIF``) and ``pc.fill_null``:

- ``case_when``: a null condition counts as false; falls through to the next
  case, then the ``else`` value, then null. A selected value that is itself
  null yields null.
- ``coalesce``: first non-null across inputs; null only if all inputs are null.
- ``nullif``: null exactly where both operands are valid and equal.
- ``fill_null``: nulls replaced by the scalar/array replacement (a null
  replacement element stays null).
"""

from std.testing import assert_equal, assert_true, assert_false, assert_raises

from ...arrays import AnyArray, BoolArray, Int64Array, StringArray
from ...builders import array, StringBuilder
from ...dtypes import Int64Type, int64, int32, float64
from ...scalars import Int64Scalar

from ...kernels.conditional import case_when, coalesce, nullif, fill_null


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _any(a: Int64Array) raises -> AnyArray:
    return a.copy()


def _strs(values: List[String], valid: List[Bool]) raises -> StringArray:
    """Build a string array; ``valid[i] == False`` produces a null element."""
    var b = StringBuilder(len(values))
    for i in range(len(values)):
        if valid[i]:
            b.append(values[i])
        else:
            b.append_null()
    return b.finish()


# ---------------------------------------------------------------------------
# case_when
# ---------------------------------------------------------------------------


def test_case_when_two_branches_with_else() raises:
    var c0 = array([True, False, None, False, True])
    var c1 = array([False, True, True, False, None])
    var v0 = array([10, 11, 12, 13, 14], int64)
    var v1 = array([20, 21, 22, 23, 24], int64)
    var e = array([30, 31, 32, 33, 34], int64)

    var conds = List[BoolArray]()
    conds.append(c0.copy())
    conds.append(c1.copy())
    var vals = List[AnyArray]()
    vals.append(v0.copy())
    vals.append(v1.copy())
    var else_: AnyArray = e.copy()

    # row2: c0 null -> treated as false -> c1 true -> v1[2]
    # row3: no cond true -> else
    var r = case_when(conds, vals, else_^)
    assert_true(r.as_int64() == array([10, 21, 22, 33, 14], int64))
    assert_equal(r.null_count(), 0)


def test_case_when_selected_value_null_propagates() raises:
    var c0 = array([True, True])
    var v0 = array([None, 5], int64)
    var e = array([9, 9], int64)

    var conds = List[BoolArray]()
    conds.append(c0.copy())
    var vals = List[AnyArray]()
    vals.append(v0.copy())
    var else_: AnyArray = e.copy()

    var r = case_when(conds, vals, else_^)
    # row0: c0 true but v0[0] is null -> null
    assert_true(r.as_int64() == array([None, 5], int64))
    assert_equal(r.null_count(), 1)


def test_case_when_no_else_is_null() raises:
    var c0 = array([True, False])
    var v0 = array([7, 8], int64)

    var conds = List[BoolArray]()
    conds.append(c0.copy())
    var vals = List[AnyArray]()
    vals.append(v0.copy())

    var r = case_when(conds, vals)
    # row1: no match, no else -> null
    assert_true(r.as_int64() == array([7, None], int64))
    assert_equal(r.null_count(), 1)


def test_case_when_string() raises:
    var c0 = array([True, False, None])
    var v0 = _strs(["a", "b", "c"], [True, True, True])
    var e = _strs(["x", "y", "z"], [True, True, True])

    var conds = List[BoolArray]()
    conds.append(c0.copy())
    var vals = List[AnyArray]()
    vals.append(v0.copy())
    var else_: AnyArray = e.copy()

    var r = case_when(conds, vals, else_^)
    # ["a", else "y", c0 null -> else "z"]
    var expected = _strs(["a", "y", "z"], [True, True, True])
    assert_true(r.as_string() == expected)


def test_case_when_typed_overload() raises:
    var c0 = array([True, False, True])
    var v0 = array([1, 2, 3], int64)
    var e = array([7, 8, 9], int64)

    var conds = List[BoolArray]()
    conds.append(c0.copy())
    var vals = List[Int64Array]()
    vals.append(v0.copy())

    # typed overload -> returns Int64Array directly
    var r = case_when[Int64Type](conds, vals, e.copy())
    assert_true(r == array([1, 8, 3], int64))


# ---------------------------------------------------------------------------
# coalesce
# ---------------------------------------------------------------------------


def test_coalesce_three_arrays() raises:
    var a = array([1, None, None, None], int64)
    var b = array([None, 2, None, None], int64)
    var c = array([None, None, 3, None], int64)

    var arrays = List[AnyArray]()
    arrays.append(a.copy())
    arrays.append(b.copy())
    arrays.append(c.copy())

    var r = coalesce(arrays)
    # last row: all null -> null
    assert_true(r.as_int64() == array([1, 2, 3, None], int64))
    assert_equal(r.null_count(), 1)


def test_coalesce_prefers_first_non_null() raises:
    var a = array([None, 10, None], int64)
    var b = array([1, 20, None], int64)
    var c = array([2, 30, 3], int64)

    var arrays = List[AnyArray]()
    arrays.append(a.copy())
    arrays.append(b.copy())
    arrays.append(c.copy())

    var r = coalesce(arrays)
    assert_true(r.as_int64() == array([1, 10, 3], int64))
    assert_equal(r.null_count(), 0)


def test_coalesce_string() raises:
    var a = _strs(["x", "", ""], [True, False, False])
    var b = _strs(["", "y", ""], [False, True, False])

    var arrays = List[AnyArray]()
    arrays.append(a.copy())
    arrays.append(b.copy())

    var r = coalesce(arrays)
    var expected = _strs(["x", "y", ""], [True, True, False])
    assert_true(r.as_string() == expected)
    assert_equal(r.null_count(), 1)


# ---------------------------------------------------------------------------
# nullif
# ---------------------------------------------------------------------------


def test_nullif_basic() raises:
    var a = array([1, 2, 3, 4], int64)
    var b = array([1, 0, 3, 0], int64)
    var aa: AnyArray = a.copy()
    var bb: AnyArray = b.copy()

    var r = nullif(aa, bb)
    # null where a == b (rows 0 and 2)
    assert_true(r.as_int64() == array([None, 2, None, 4], int64))
    assert_equal(r.null_count(), 2)


def test_nullif_with_nulls() raises:
    var a = array([1, None, 3], int64)
    var b = array([1, 2, None], int64)
    var aa: AnyArray = a.copy()
    var bb: AnyArray = b.copy()

    var r = nullif(aa, bb)
    # row0: 1==1 -> null; row1: a null -> keep (null); row2: b null -> keep 3
    assert_true(r.as_int64() == array([None, None, 3], int64))
    assert_equal(r.null_count(), 2)


def test_nullif_string() raises:
    var a = _strs(["a", "b", "c"], [True, True, True])
    var b = _strs(["a", "x", "c"], [True, True, True])
    var aa: AnyArray = a.copy()
    var bb: AnyArray = b.copy()

    var r = nullif(aa, bb)
    var expected = _strs(["", "b", ""], [False, True, False])
    assert_true(r.as_string() == expected)
    assert_equal(r.null_count(), 2)


def test_nullif_typed_overload() raises:
    var a = array([5, 6, 7], int64)
    var b = array([5, 0, 7], int64)
    var r = nullif(a.copy(), b.copy())  # typed -> Int64Array
    assert_true(r == array([None, 6, None], int64))


# ---------------------------------------------------------------------------
# fill_null
# ---------------------------------------------------------------------------


def test_fill_null_array() raises:
    var a = array([1, None, 3, None], int64)
    var fill = array([10, 20, 30, 40], int64)
    var aa: AnyArray = a.copy()
    var ff: AnyArray = fill.copy()

    var r = fill_null(aa, ff)
    assert_true(r.as_int64() == array([1, 20, 3, 40], int64))
    assert_equal(r.null_count(), 0)


def test_fill_null_array_with_null_fill() raises:
    var a = array([1, None, 3, None], int64)
    var fill = array([10, None, 30, 40], int64)
    var aa: AnyArray = a.copy()
    var ff: AnyArray = fill.copy()

    var r = fill_null(aa, ff)
    # row1: a null and fill null -> stays null
    assert_true(r.as_int64() == array([1, None, 3, 40], int64))
    assert_equal(r.null_count(), 1)


def test_fill_null_scalar() raises:
    var a = array([1, None, 3, None], int64)
    var aa: AnyArray = a.copy()

    var r = fill_null(aa, Int64Scalar(99).to_any())
    assert_true(r.as_int64() == array([1, 99, 3, 99], int64))
    assert_equal(r.null_count(), 0)


def test_fill_null_string_array() raises:
    var a = _strs(["a", "", "c"], [True, False, True])
    var fill = _strs(["X", "Y", "Z"], [True, True, True])
    var aa: AnyArray = a.copy()
    var ff: AnyArray = fill.copy()

    var r = fill_null(aa, ff)
    var expected = _strs(["a", "Y", "c"], [True, True, True])
    assert_true(r.as_string() == expected)
    assert_equal(r.null_count(), 0)


def test_fill_null_typed_scalar() raises:
    var a = array([1, None, 3], int64)
    var r = fill_null(a.copy(), Int64(0))  # typed -> Int64Array
    assert_true(r == array([1, 0, 3], int64))


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def test_case_when_condition_count_mismatch_raises() raises:
    var c0 = array([True, False])
    var v0 = array([1, 2], int64)
    var v1 = array([3, 4], int64)

    var conds = List[BoolArray]()
    conds.append(c0.copy())
    var vals = List[AnyArray]()
    vals.append(v0.copy())
    vals.append(v1.copy())
    with assert_raises():
        _ = case_when(conds, vals)


def test_coalesce_empty_raises() raises:
    var arrays = List[AnyArray]()
    with assert_raises():
        _ = coalesce(arrays)


def test_nullif_dtype_mismatch_raises() raises:
    var a = array([1, 2], int64)
    var b = array([1, 2], int32)
    var aa: AnyArray = a.copy()
    var bb: AnyArray = b.copy()
    with assert_raises():
        _ = nullif(aa, bb)


def test_fill_null_length_mismatch_raises() raises:
    var a = array([1, None, 3], int64)
    var fill = array([9, 9], int64)
    var aa: AnyArray = a.copy()
    var ff: AnyArray = fill.copy()
    with assert_raises():
        _ = fill_null(aa, ff)
