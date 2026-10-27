# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Compute functions for marrow arrays.

This module follows the same naming and signature conventions as
``pyarrow.compute`` so that code can be ported between libraries with
minimal changes.

All functions accept :class:`~marrow.Array` (or raw C binding objects) and
return :class:`~marrow.Array` or :class:`~marrow.Scalar` as appropriate.
"""

from . import libmarrow as _ma
from ._wrapper import unwrap
from .arrays import Array, _nulls_first, _serial


# ── Arithmetic ────────────────────────────────────────────────────────────────


def add(left, right, memory_pool=None, ctx=None):
    """Add two arrays element-wise.

    Equivalent to ``pyarrow.compute.add``.
    """
    return Array.wrap(_ma.add(left.unwrap(), right.unwrap(), (ctx or _serial())))


def subtract(left, right, memory_pool=None, ctx=None):
    """Subtract *right* from *left* element-wise.

    Equivalent to ``pyarrow.compute.subtract``.
    """
    return Array.wrap(_ma.subtract(left.unwrap(), right.unwrap(), (ctx or _serial())))


def multiply(left, right, memory_pool=None, ctx=None):
    """Multiply two arrays element-wise.

    Equivalent to ``pyarrow.compute.multiply``.
    """
    return Array.wrap(_ma.multiply(left.unwrap(), right.unwrap(), (ctx or _serial())))


def divide(left, right, memory_pool=None, ctx=None):
    """Divide *left* by *right* element-wise.

    Equivalent to ``pyarrow.compute.divide``: two integers divide to an
    integer, so ``-1 / 3`` is 0, ``INT_MIN / -1`` is 0, and a zero divisor
    raises rather than answering.

    ``/`` on an :class:`~marrow.expr.Expr` is the other division, Python's:
    both operands widen to ``float64`` first, so ``-1 / 3`` is -0.333 and a
    zero divisor answers an infinity or a NaN. Floats behave the same either
    way; only the integers and the zero divisor differ.
    """
    return Array.wrap(_ma.divide(left.unwrap(), right.unwrap(), (ctx or _serial())))


# ── Aggregations ────────────────────────────────────────────────────────────────────────


def any(array, *, skip_nulls=True, min_count=1, memory_pool=None, ctx=None):
    """Return whether any element in the array is true.

    Equivalent to ``pyarrow.compute.any``: ``None`` when fewer than
    *min_count* elements are valid, and, with ``skip_nulls=False``, when a null
    could still make the answer true.
    """
    return _ma.any(array.unwrap(), skip_nulls, min_count, (ctx or _serial()))


def all(array, *, skip_nulls=True, min_count=1, memory_pool=None, ctx=None):
    """Return whether all elements in the array are true.

    Equivalent to ``pyarrow.compute.all``: ``None`` when fewer than
    *min_count* elements are valid, and, with ``skip_nulls=False``, when a null
    could still make the answer false.
    """
    return _ma.all(array.unwrap(), skip_nulls, min_count, (ctx or _serial()))


# ── List aggregates ─────────────────────────────────────────────────────────


def list_sum(lists, *, memory_pool=None, ctx=None):
    """The sum of each list's elements, skipping nulls.

    Equivalent to DuckDB's ``list_sum``: an empty list, a list of nulls and a
    null list each answer null. Signed integers sum to ``int64``, unsigned
    ones to ``uint64`` and floats to ``float64``.
    """
    return Array.wrap(_ma.list_sum(lists.unwrap(), (ctx or _serial())))


def list_min(lists, *, memory_pool=None, ctx=None):
    """The least of each list's elements, skipping nulls.

    Equivalent to DuckDB's ``list_min``: an empty list, a list of nulls and a
    null list each answer null. The answer keeps the element type, and is NaN
    only when every value is NaN, as DuckDB, which orders NaN above every
    value, answers.
    """
    return Array.wrap(_ma.list_min(lists.unwrap(), (ctx or _serial())))


def list_max(lists, *, memory_pool=None, ctx=None):
    """The greatest of each list's elements, skipping nulls and NaN.

    Equivalent to DuckDB's ``list_max``: an empty list, a list of nulls and a
    null list each answer null. The answer keeps the element type. DuckDB
    orders NaN above every value, so it answers NaN where this skips it.
    """
    return Array.wrap(_ma.list_max(lists.unwrap(), (ctx or _serial())))


def list_mean(lists, *, memory_pool=None, ctx=None):
    """The mean of each list's elements, skipping nulls, as ``float64``.

    Equivalent to DuckDB's ``list_avg``: an empty list, a list of nulls and a
    null list each answer null.
    """
    return Array.wrap(_ma.list_mean(lists.unwrap(), (ctx or _serial())))


def list_count(lists, *, memory_pool=None, ctx=None):
    """The number of non-null elements in each list, as ``int64``.

    Equivalent to DuckDB's ``list_count``: an empty list and a list of nulls
    answer 0, and a null list answers null.
    """
    return Array.wrap(_ma.list_count(lists.unwrap(), (ctx or _serial())))


# ── Comparisons ───────────────────────────────────────────────────────────────


def equal(left, right, memory_pool=None, ctx=None):
    """Return a boolean array indicating where *left* == *right*.

    Equivalent to ``pyarrow.compute.equal``.
    """
    return Array.wrap(_ma.equal(left.unwrap(), right.unwrap(), (ctx or _serial())))


def not_equal(left, right, memory_pool=None, ctx=None):
    """Return a boolean array indicating where *left* != *right*.

    Equivalent to ``pyarrow.compute.not_equal``.
    """
    return Array.wrap(_ma.not_equal(left.unwrap(), right.unwrap(), (ctx or _serial())))


def less(left, right, memory_pool=None, ctx=None):
    """Return a boolean array indicating where *left* < *right*.

    Equivalent to ``pyarrow.compute.less``.
    """
    return Array.wrap(_ma.less(left.unwrap(), right.unwrap(), (ctx or _serial())))


def less_equal(left, right, memory_pool=None, ctx=None):
    """Return a boolean array indicating where *left* <= *right*.

    Equivalent to ``pyarrow.compute.less_equal``.
    """
    return Array.wrap(_ma.less_equal(left.unwrap(), right.unwrap(), (ctx or _serial())))


def greater(left, right, memory_pool=None, ctx=None):
    """Return a boolean array indicating where *left* > *right*.

    Equivalent to ``pyarrow.compute.greater``.
    """
    return Array.wrap(_ma.greater(left.unwrap(), right.unwrap(), (ctx or _serial())))


def greater_equal(left, right, memory_pool=None, ctx=None):
    """Return a boolean array indicating where *left* >= *right*.

    Equivalent to ``pyarrow.compute.greater_equal``.
    """
    return Array.wrap(
        _ma.greater_equal(left.unwrap(), right.unwrap(), (ctx or _serial()))
    )


# ── Selection ─────────────────────────────────────────────────────────────────


def filter(input, selection_filter, memory_pool=None, ctx=None):
    """Filter an array with a boolean selection mask.

    Null positions in the mask are dropped from the output.

    Equivalent to ``pyarrow.compute.filter``.
    """
    return Array.wrap(
        _ma.filter(input.unwrap(), selection_filter.unwrap(), (ctx or _serial()))
    )


def is_null(input, *, nan_is_null=False, memory_pool=None, ctx=None):
    """Return a boolean array marking null positions.

    Equivalent to ``pyarrow.compute.is_null``.
    """
    if nan_is_null:
        raise NotImplementedError("nan_is_null=True is not implemented")
    return Array.wrap(_ma.is_null(input.unwrap(), (ctx or _serial())))


def is_valid(input, memory_pool=None, ctx=None):
    """Return a boolean array marking non-null positions.

    Equivalent to ``pyarrow.compute.is_valid``.
    """
    return Array.wrap(_ma.is_valid(input.unwrap(), (ctx or _serial())))


def drop_null(input, memory_pool=None, ctx=None):
    """Remove null values from an array.

    Equivalent to ``pyarrow.compute.drop_null``.
    """
    return Array.wrap(_ma.drop_null(input.unwrap(), (ctx or _serial())))


def take(data, indices, *, boundscheck=True, memory_pool=None, ctx=None):
    """Select values from *data* at the positions given by *indices*.

    Equivalent to ``pyarrow.compute.take``.
    """
    if not boundscheck:
        raise NotImplementedError("boundscheck=False is not implemented")
    return Array.wrap(_ma.take(data.unwrap(), indices.unwrap(), (ctx or _serial())))


# ── Conversion ────────────────────────────────────────────────────────────────


def cast(arr, target_type, *, safe=True, memory_pool=None, ctx=None):
    """Cast *arr* to *target_type* (numeric, bool, or temporal).

    With ``safe=True`` (the default) a lossy conversion raises; ``safe=False``
    uses the raw truncating/wrapping conversion.

    Equivalent to ``pyarrow.compute.cast``.
    """
    return Array.wrap(
        _ma.cast(arr.unwrap(), unwrap(target_type), safe, (ctx or _serial()))
    )


# ── Sorting ───────────────────────────────────────────────────────────────────


def sort_indices(
    input, sort_keys=(), *, null_placement="at_end", memory_pool=None, ctx=None
):
    """Return the indices that would sort an array.

    *sort_keys* is ignored for plain arrays (only the *null_placement* kwarg
    matters).  Pass ``null_placement="at_start"`` to place nulls first.

    Equivalent to ``pyarrow.compute.sort_indices``.
    """
    if isinstance(sort_keys, (list, tuple)) and len(sort_keys) > 1:
        raise NotImplementedError("multi-key sort_keys is not implemented")
    if isinstance(sort_keys, (list, tuple)) and len(sort_keys) == 1:
        _, order = (
            sort_keys[0]
            if isinstance(sort_keys[0], tuple)
            else (sort_keys[0], "ascending")
        )
        asc = order != "descending"
    else:
        asc = True
    nulls_first = _nulls_first(null_placement)
    return Array.wrap(
        _ma.sort_indices(input.unwrap(), asc, nulls_first, (ctx or _serial()))
    )


def sort(input, sort_keys=(), *, null_placement="at_end", memory_pool=None, ctx=None):
    """Return a sorted copy of an array.

    PyArrow has no ``pc.sort``; it spells this ``Array.sort()``. Kept as a
    functional form for symmetry with `sort_indices`, which does exist there.
    """
    if isinstance(sort_keys, (list, tuple)) and len(sort_keys) > 1:
        raise NotImplementedError("multi-key sort_keys is not implemented")
    _, order = (
        sort_keys[0]
        if isinstance(sort_keys, (list, tuple)) and sort_keys
        else (None, "ascending")
    )
    ascending = order != "descending"
    nulls_first = _nulls_first(null_placement)
    return Array.wrap(
        _ma.sort(input.unwrap(), ascending, nulls_first, (ctx or _serial()))
    )
