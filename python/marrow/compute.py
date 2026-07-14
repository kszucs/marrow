"""Compute functions for marrow arrays.

This module follows the same naming and signature conventions as
``pyarrow.compute`` so that code can be ported between libraries with
minimal changes.

All functions accept :class:`~marrow.Array` (or raw C binding objects) and
return :class:`~marrow.Array` or :class:`~marrow.Scalar` as appropriate.
"""

from . import libmarrow as _ma
from . import Array, Scalar, RecordBatch, _serial


# ── Arithmetic ────────────────────────────────────────────────────────────────


def add(left, right, memory_pool=None):
    """Add two arrays element-wise.

    Equivalent to ``pyarrow.compute.add``.
    """
    return Array.wrap(_ma.add(left.unwrap(), right.unwrap(), _serial()))


def subtract(left, right, memory_pool=None):
    """Subtract *right* from *left* element-wise.

    Equivalent to ``pyarrow.compute.subtract``.
    """
    return Array.wrap(_ma.subtract(left.unwrap(), right.unwrap(), _serial()))


def multiply(left, right, memory_pool=None):
    """Multiply two arrays element-wise.

    Equivalent to ``pyarrow.compute.multiply``.
    """
    return Array.wrap(_ma.multiply(left.unwrap(), right.unwrap(), _serial()))


def divide(left, right, memory_pool=None):
    """Divide *left* by *right* element-wise.

    Equivalent to ``pyarrow.compute.divide``.
    """
    return Array.wrap(_ma.divide(left.unwrap(), right.unwrap(), _serial()))


# ── Aggregations ──────────────────────────────────────────────────────────────


def sum(array, *, skip_nulls=True, memory_pool=None):
    """Sum the values of an array.

    Equivalent to ``pyarrow.compute.sum``.
    """
    return Scalar.wrap(_ma.sum(array.unwrap(), _serial()))


def product(array, *, skip_nulls=True, memory_pool=None):
    """Compute the product of all values in an array.

    Equivalent to ``pyarrow.compute.product``.
    """
    return Scalar.wrap(_ma.product(array.unwrap(), _serial()))


def min(array, *, skip_nulls=True, memory_pool=None):
    """Compute the minimum value of an array.

    Equivalent to ``pyarrow.compute.min``.
    """
    return Scalar.wrap(_ma.min(array.unwrap(), _serial()))


def max(array, *, skip_nulls=True, memory_pool=None):
    """Compute the maximum value of an array.

    Equivalent to ``pyarrow.compute.max``.
    """
    return Scalar.wrap(_ma.max(array.unwrap(), _serial()))


def mean(array, *, skip_nulls=True, memory_pool=None):
    """Compute the arithmetic mean of an array, as a float64 scalar.

    Nulls are excluded. Returns a null scalar for an empty or all-null array.
    Equivalent to ``pyarrow.compute.mean``.
    """
    return Scalar.wrap(_ma.mean(array.unwrap(), _serial()))


def any(array, *, skip_nulls=True, memory_pool=None):
    """Return whether any element in the array is true.

    Equivalent to ``pyarrow.compute.any``.
    """
    return _ma.any(array.unwrap(), _serial())


def all(array, *, skip_nulls=True, memory_pool=None):
    """Return whether all elements in the array are true.

    Equivalent to ``pyarrow.compute.all``.
    """
    return _ma.all(array.unwrap(), _serial())


# ── Comparisons ───────────────────────────────────────────────────────────────


def equal(left, right, memory_pool=None):
    """Return a boolean array indicating where *left* == *right*.

    Equivalent to ``pyarrow.compute.equal``.
    """
    return Array.wrap(_ma.equal(left.unwrap(), right.unwrap(), _serial()))


def not_equal(left, right, memory_pool=None):
    """Return a boolean array indicating where *left* != *right*.

    Equivalent to ``pyarrow.compute.not_equal``.
    """
    return Array.wrap(_ma.not_equal(left.unwrap(), right.unwrap(), _serial()))


def less(left, right, memory_pool=None):
    """Return a boolean array indicating where *left* < *right*.

    Equivalent to ``pyarrow.compute.less``.
    """
    return Array.wrap(_ma.less(left.unwrap(), right.unwrap(), _serial()))


def less_equal(left, right, memory_pool=None):
    """Return a boolean array indicating where *left* <= *right*.

    Equivalent to ``pyarrow.compute.less_equal``.
    """
    return Array.wrap(_ma.less_equal(left.unwrap(), right.unwrap(), _serial()))


def greater(left, right, memory_pool=None):
    """Return a boolean array indicating where *left* > *right*.

    Equivalent to ``pyarrow.compute.greater``.
    """
    return Array.wrap(_ma.greater(left.unwrap(), right.unwrap(), _serial()))


def greater_equal(left, right, memory_pool=None):
    """Return a boolean array indicating where *left* >= *right*.

    Equivalent to ``pyarrow.compute.greater_equal``.
    """
    return Array.wrap(_ma.greater_equal(left.unwrap(), right.unwrap(), _serial()))


# ── Selection ─────────────────────────────────────────────────────────────────


def filter(input, selection_filter, memory_pool=None):
    """Filter an array with a boolean selection mask.

    Null positions in the mask are dropped from the output.

    Equivalent to ``pyarrow.compute.filter``.
    """
    return Array.wrap(_ma.filter(input.unwrap(), selection_filter.unwrap(), _serial()))


def drop_null(input, memory_pool=None):
    """Remove null values from an array.

    Equivalent to ``pyarrow.compute.drop_null``.
    """
    return Array.wrap(_ma.drop_null(input.unwrap(), _serial()))


def take(data, indices, *, boundscheck=True, memory_pool=None):
    """Select values from *data* at the positions given by *indices*.

    Equivalent to ``pyarrow.compute.take``.
    """
    return Array.wrap(_ma.take(data.unwrap(), indices.unwrap(), _serial()))


# ── Conversion ────────────────────────────────────────────────────────────────


def cast(arr, target_type, *, safe=True, memory_pool=None):
    """Cast *arr* to *target_type* (numeric, bool, or temporal).

    With ``safe=True`` (the default) a lossy conversion raises; ``safe=False``
    uses the raw truncating/wrapping conversion.

    Equivalent to ``pyarrow.compute.cast``.
    """
    return Array.wrap(_ma.cast(arr.unwrap(), target_type, safe, _serial()))


# ── Sorting ───────────────────────────────────────────────────────────────────


def sort_indices(input, sort_keys=(), *, null_placement="at_end", memory_pool=None):
    """Return the indices that would sort an array.

    *sort_keys* is ignored for plain arrays (only the *null_placement* kwarg
    matters).  Pass ``null_placement="at_start"`` to place nulls first.

    Equivalent to ``pyarrow.compute.sort_indices``.
    """
    if isinstance(sort_keys, (list, tuple)) and len(sort_keys) == 1:
        _, order = (
            sort_keys[0]
            if isinstance(sort_keys[0], tuple)
            else (sort_keys[0], "ascending")
        )
        asc = order != "descending"
    else:
        asc = True
    nulls_first = null_placement != "at_end"
    return Array.wrap(_ma.sort_indices(input.unwrap(), asc, nulls_first, _serial()))
