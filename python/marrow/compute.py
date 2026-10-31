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
from .arrays import Array, Scalar, _nulls_first, _serial


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


# ── Bitwise ──────────────────────────────────────────────────────────────────


def bit_wise_and(x, y, *, memory_pool=None, ctx=None):
    """The bitwise AND of two integer arrays.

    Equivalent to ``pyarrow.compute.bit_wise_and``, but both operands must
    share one integer type, where PyArrow promotes ``int8 & int16`` to
    ``int16``.
    """
    return Array.wrap(_ma.bit_wise_and(x.unwrap(), y.unwrap(), (ctx or _serial())))


def bit_wise_or(x, y, *, memory_pool=None, ctx=None):
    """The bitwise OR of two integer arrays.

    Equivalent to ``pyarrow.compute.bit_wise_or``, but both operands must
    share one integer type, where PyArrow promotes mixed integer types.
    """
    return Array.wrap(_ma.bit_wise_or(x.unwrap(), y.unwrap(), (ctx or _serial())))


def bit_wise_xor(x, y, *, memory_pool=None, ctx=None):
    """The bitwise XOR of two integer arrays.

    Equivalent to ``pyarrow.compute.bit_wise_xor``, but both operands must
    share one integer type, where PyArrow promotes mixed integer types.
    """
    return Array.wrap(_ma.bit_wise_xor(x.unwrap(), y.unwrap(), (ctx or _serial())))


def bit_wise_not(x, *, memory_pool=None, ctx=None):
    """The bitwise NOT of an integer array.

    Equivalent to ``pyarrow.compute.bit_wise_not``.
    """
    return Array.wrap(_ma.bit_wise_not(x.unwrap(), (ctx or _serial())))


def shift_left(x, y, *, memory_pool=None, ctx=None):
    """*x* shifted left by *y* bits, wrapping.

    Equivalent to ``pyarrow.compute.shift_left``: an amount below zero, or not
    below the type's precision (its bit width, less the sign bit for a signed
    type), leaves the value as it is. Both operands must share one integer
    type, where PyArrow promotes mixed integer types.
    """
    return Array.wrap(_ma.shift_left(x.unwrap(), y.unwrap(), (ctx or _serial())))


def shift_right(x, y, *, memory_pool=None, ctx=None):
    """*x* shifted right by *y* bits, arithmetically for a signed type.

    Equivalent to ``pyarrow.compute.shift_right``: an amount below zero, or
    not below the type's precision (its bit width, less the sign bit for a
    signed type), leaves the value as it is. Both operands must share one
    integer type, where PyArrow promotes mixed integer types.
    """
    return Array.wrap(_ma.shift_right(x.unwrap(), y.unwrap(), (ctx or _serial())))


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


def mode(array, n=1, *, skip_nulls=True, min_count=0, memory_pool=None, ctx=None):
    """The *n* most common values of *array*, as ``struct<mode, count>`` rows.

    Equivalent to ``pyarrow.compute.mode``: the most common first, ties
    smallest value first, NaN above every number; no rows when *skip_nulls* is
    false and a value is null, or when fewer than *min_count* values are valid.
    ``-0.0`` and ``0.0`` are one value, and so is every NaN. Strings and
    temporal columns are accepted too, as DuckDB accepts them, where PyArrow
    refuses them.
    """
    return Array.wrap(
        _ma.mode(array.unwrap(), n, skip_nulls, min_count, (ctx or _serial()))
    )


# ── Cumulative ───────────────────────────────────────────────────────────────


def cumulative_sum(values, start=None, *, skip_nulls=False, memory_pool=None, ctx=None):
    """The running sum of *values*.

    Equivalent to ``pyarrow.compute.cumulative_sum``: the input's numeric type,
    integers wrapping. A null stays null; with ``skip_nulls=False`` every value
    after the first null is null too.
    """
    if start is not None:
        raise NotImplementedError("start is not implemented")
    return Array.wrap(
        _ma.cumulative_sum(values.unwrap(), skip_nulls, (ctx or _serial()))
    )


def cumulative_prod(
    values, start=None, *, skip_nulls=False, memory_pool=None, ctx=None
):
    """The running product of *values*.

    Equivalent to ``pyarrow.compute.cumulative_prod``: the input's numeric
    type, integers wrapping. A null stays null; with ``skip_nulls=False`` every
    value after the first null is null too.
    """
    if start is not None:
        raise NotImplementedError("start is not implemented")
    return Array.wrap(
        _ma.cumulative_prod(values.unwrap(), skip_nulls, (ctx or _serial()))
    )


def cumulative_min(values, start=None, *, skip_nulls=False, memory_pool=None, ctx=None):
    """The running minimum of *values*.

    Equivalent to ``pyarrow.compute.cumulative_min``: the input's numeric type.
    A null stays null; with ``skip_nulls=False`` every value after the first
    null is null too. NaN is skipped, as ``min`` skips it, so the answer is NaN
    only while every value so far is NaN, where PyArrow answers the type's
    greatest value.
    """
    if start is not None:
        raise NotImplementedError("start is not implemented")
    return Array.wrap(
        _ma.cumulative_min(values.unwrap(), skip_nulls, (ctx or _serial()))
    )


def cumulative_max(values, start=None, *, skip_nulls=False, memory_pool=None, ctx=None):
    """The running maximum of *values*.

    Equivalent to ``pyarrow.compute.cumulative_max``: the input's numeric type.
    A null stays null; with ``skip_nulls=False`` every value after the first
    null is null too. NaN is skipped, as ``max`` skips it, so the answer is NaN
    only while every value so far is NaN, where PyArrow answers the type's
    least value.
    """
    if start is not None:
        raise NotImplementedError("start is not implemented")
    return Array.wrap(
        _ma.cumulative_max(values.unwrap(), skip_nulls, (ctx or _serial()))
    )


# ── Order ────────────────────────────────────────────────────────────────────


def pairwise_diff(input, period=1, *, memory_pool=None, ctx=None):
    """Each value minus the value *period* rows before it.

    Equivalent to ``pyarrow.compute.pairwise_diff``: numeric input only, in its
    own type, integers wrapping; the first *period* rows are null, and a
    negative *period* reads ahead. PyArrow also takes temporal input, answering
    a duration; this does not.
    """
    return Array.wrap(_ma.pairwise_diff(input.unwrap(), period, (ctx or _serial())))


def shift(values, n=1, *, fill_value=None, ctx=None):
    """*values* moved *n* rows down, nulls filling the rows left empty.

    Equivalent to Polars' ``Series.shift``: a negative *n* moves the values up,
    and every type is accepted.
    """
    if fill_value is not None:
        raise NotImplementedError("fill_value is not implemented")
    return Array.wrap(_ma.shift(values.unwrap(), n, (ctx or _serial())))


# ── Rolling windows ──────────────────────────────────────────────────────────


def _rolling(kernel, values, window_size, weights, min_samples, center, ctx):
    """*kernel* over Polars' window, as the frame ``ROWS BETWEEN lo AND hi``:
    the *window_size* rows ending at each row, or centred on it."""
    if weights is not None:
        raise NotImplementedError("weights is not implemented")
    if center:
        lo, hi = -(window_size // 2), (window_size - 1) // 2
    else:
        lo, hi = -(window_size - 1), 0
    if min_samples is None:
        min_samples = window_size
    return Array.wrap(kernel(values.unwrap(), lo, hi, min_samples, (ctx or _serial())))


def rolling_sum(
    values, window_size, weights=None, *, min_samples=None, center=False, ctx=None
):
    """The sum of each row's window of *window_size* rows, nulls skipped.

    Equivalent to Polars' ``Series.rolling_sum``: the window ends at the row,
    or is centred on it with *center*, and a row is null when its window holds
    fewer than *min_samples* valid values (*window_size* by default). Numeric
    input only, summed in its own type, where Polars widens 8- and 16-bit
    integers to ``int64``.
    """
    return _rolling(
        _ma.rolling_sum, values, window_size, weights, min_samples, center, ctx
    )


def rolling_min(
    values, window_size, weights=None, *, min_samples=None, center=False, ctx=None
):
    """The least value of each row's window of *window_size* rows.

    Equivalent to Polars' ``Series.rolling_min``, with its window and
    *min_samples* rules. NaN is skipped unless the whole window is NaN, as
    ``min`` skips it; Polars propagates it.
    """
    return _rolling(
        _ma.rolling_min, values, window_size, weights, min_samples, center, ctx
    )


def rolling_max(
    values, window_size, weights=None, *, min_samples=None, center=False, ctx=None
):
    """The greatest value of each row's window of *window_size* rows.

    Equivalent to Polars' ``Series.rolling_max``, with its window and
    *min_samples* rules. NaN is skipped unless the whole window is NaN, as
    ``max`` skips it; Polars propagates it.
    """
    return _rolling(
        _ma.rolling_max, values, window_size, weights, min_samples, center, ctx
    )


def rolling_mean(
    values, window_size, weights=None, *, min_samples=None, center=False, ctx=None
):
    """The mean of each row's window of *window_size* rows, as ``float64``.

    Equivalent to Polars' ``Series.rolling_mean``, with its window and
    *min_samples* rules; Polars keeps ``float32`` for a ``float32`` input.
    """
    return _rolling(
        _ma.rolling_mean, values, window_size, weights, min_samples, center, ctx
    )


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


def list_bit_and(lists, *, memory_pool=None, ctx=None):
    """The bitwise AND of each list's elements, skipping nulls.

    Equivalent to DuckDB's ``list_bit_and``: integer elements only, the
    answer in the element type, and an empty list, a list of nulls and a null
    list each null.
    """
    return Array.wrap(_ma.list_bit_and(lists.unwrap(), (ctx or _serial())))


def list_bit_or(lists, *, memory_pool=None, ctx=None):
    """The bitwise OR of each list's elements, skipping nulls.

    Equivalent to DuckDB's ``list_bit_or``: integer elements only, the answer
    in the element type, and an empty list, a list of nulls and a null list
    each null.
    """
    return Array.wrap(_ma.list_bit_or(lists.unwrap(), (ctx or _serial())))


def list_bit_xor(lists, *, memory_pool=None, ctx=None):
    """The bitwise XOR of each list's elements, skipping nulls.

    Equivalent to DuckDB's ``list_bit_xor``: integer elements only, the
    answer in the element type, and an empty list, a list of nulls and a null
    list each null.
    """
    return Array.wrap(_ma.list_bit_xor(lists.unwrap(), (ctx or _serial())))


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


def indices_nonzero(values, *, memory_pool=None, ctx=None):
    """The positions of the true, or non-zero, values, as ``uint64``.

    Equivalent to ``pyarrow.compute.indices_nonzero``: bool, numeric and
    decimal input. A null is skipped, NaN is non-zero and ``-0.0`` is zero.
    """
    return Array.wrap(_ma.indices_nonzero(values.unwrap(), (ctx or _serial())))


def replace_with_mask(values, mask, replacements, *, memory_pool=None, ctx=None):
    """*values*, with each row *mask* selects replaced by the next unused
    value of *replacements*.

    Equivalent to ``pyarrow.compute.replace_with_mask``: a null in *mask*
    makes its row null and uses no replacement, and spare replacements are
    ignored. Every layout ``take`` accepts is accepted, where PyArrow refuses
    string views, lists, structs and dictionaries. A scalar *mask* or
    *replacements* is not implemented.
    """
    if isinstance(mask, Scalar) or isinstance(replacements, Scalar):
        raise NotImplementedError("a scalar mask or replacements is not implemented")
    return Array.wrap(
        _ma.replace_with_mask(
            values.unwrap(),
            mask.unwrap(),
            replacements.unwrap(),
            (ctx or _serial()),
        )
    )


# ── Associative transforms ───────────────────────────────────────────────────


def dictionary_encode(array, null_encoding="mask", *, memory_pool=None, ctx=None):
    """*array* as a dictionary of its distinct values, with ``int32`` indices.

    Equivalent to ``pyarrow.compute.dictionary_encode``: the dictionary is in
    first-seen order, and a dictionary array is returned as it is. A null has
    a null index with ``null_encoding="mask"`` and an entry of its own with
    ``"encode"``. Values are distinct as a group-by keys them, so ``-0.0`` and
    ``0.0`` are one entry where PyArrow keeps two. Lists and structs are
    accepted, where PyArrow refuses them; ``fixed_size_binary`` and ``null``
    columns are not.
    """
    if null_encoding not in ("mask", "encode"):
        raise ValueError(f'"{null_encoding}" is not a valid null encoding')
    return Array.wrap(
        _ma.dictionary_encode(
            array.unwrap(), null_encoding == "encode", (ctx or _serial())
        )
    )


def value_counts(array, *, memory_pool=None, ctx=None):
    """How often each distinct value of *array* occurs, as
    ``struct<values, counts>`` rows.

    Equivalent to ``pyarrow.compute.value_counts``: one row per distinct
    value in first-seen order, null counted as a value. Lists and structs
    are accepted, where PyArrow refuses them; ``fixed_size_binary`` and
    ``null`` columns are not.

    Values are distinct as a group-by keys them: ``-0.0`` and ``0.0`` are
    one value, where PyArrow keeps two. A dictionary array is counted by
    value, so two dictionary entries holding the same value make one row;
    PyArrow makes one row per index.
    """
    return Array.wrap(_ma.value_counts(array.unwrap(), (ctx or _serial())))


def run_end_encode(array, run_end_type=None, *, memory_pool=None, ctx=None):
    """*array* as runs of equal values, each value once with the row its run
    ends at, the run ends typed *run_end_type* (``int32`` by default).

    Equivalent to ``pyarrow.compute.run_end_encode``. Rows are equal when
    their bits are, so NaNs with different payloads and ``-0.0`` and ``0.0``
    stay apart, and consecutive nulls are one run. Lists, structs,
    dictionaries and run-end encoded arrays are refused; string and binary
    views are accepted, where PyArrow refuses them.
    """
    run_ends = _ma.int32() if run_end_type is None else unwrap(run_end_type)
    return Array.wrap(_ma.run_end_encode(array.unwrap(), run_ends, (ctx or _serial())))


def run_end_decode(array, *, memory_pool=None, ctx=None):
    """The rows of a run-end encoded *array*, each run's value repeated over
    its run.

    Equivalent to ``pyarrow.compute.run_end_decode``.
    """
    return Array.wrap(_ma.run_end_decode(array.unwrap(), (ctx or _serial())))


def run_index(values, *, ctx=None):
    """Each row's run of equal values, numbered from 0, as ``uint32``.

    Rows are equal as a group-by keys them: a null equals a null, every NaN
    is one value and ``-0.0`` equals ``0.0``. ``[1, 1, None, None, 2, 1]``
    gives ``[0, 0, 1, 1, 2, 3]``.
    """
    return Array.wrap(_ma.run_index(values.unwrap(), (ctx or _serial())))


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
