# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Tests for `dictionary_encode`, `value_counts` and `mode`, the kernels built
on `DictionaryEncoder`."""

from std.math import nan
from std.testing import assert_equal, assert_raises, assert_true

from ...arrays import DictionaryArray, DynArray, RunEndEncodedArray, StructArray
from ...builders import Int64Builder, array, nulls
from ...dtypes import float64, int8, int32, int64, string, uint8
from ...execution import ExecContext
from ...kernels.dictionary import (
    DictionaryEncodeKernel,
    ModeKernel,
    ValueCountsKernel,
)
from ...kernels.runs import RunEndDecodeKernel


def _strings() raises -> DynArray:
    return array(["b", "a", None, "b", "c", None, "a"])


def _check_encoded(
    encoded: DynArray, indices: DynArray, entries: DynArray
) raises:
    """`encoded` is a dictionary with `indices` over `entries`."""
    ref d = encoded.as_dictionary()
    assert_true(d.indices().dtype() == int32)
    assert_equal(String(d.indices()), String(indices))
    assert_equal(String(d.dictionary()), String(entries))


# ---------------------------------------------------------------------------
# dictionary_encode
# ---------------------------------------------------------------------------


def test_dictionary_encode_masks_nulls() raises:
    """Entries in first-seen order; a null row has a null index, or with
    `encode_nulls` an entry of its own."""
    _check_encoded(
        DictionaryEncodeKernel.dispatch(_strings()),
        array([0, 1, None, 0, 2, None, 1], int32),
        array(["b", "a", "c"]),
    )
    _check_encoded(
        DictionaryEncodeKernel.dispatch(_strings(), encode_nulls=True),
        array([0, 1, 2, 0, 3, 2, 1], int32),
        array(["b", "a", None, "c"]),
    )


def test_dictionary_encode_shifts_codes_past_the_null_key() raises:
    var values: DynArray = array([None, 3, None, 3, 1], uint8)
    _check_encoded(
        DictionaryEncodeKernel.dispatch(values),
        array([None, 0, None, 0, 1], int32),
        array([3, 1], uint8),
    )
    _check_encoded(
        DictionaryEncodeKernel.dispatch(values, encode_nulls=True),
        array([0, 1, 0, 1, 2], int32),
        array([None, 3, 1], uint8),
    )


def test_dictionary_encode_float_keys_follow_grouping() raises:
    """Keyed as a group-by keys: every NaN is one entry, and so are `-0.0`
    and `0.0`."""
    var values: DynArray = array(
        [1.0, nan[DType.float64](), -0.0, 0.0, nan[DType.float64](), None],
        float64,
    )
    var d = DictionaryEncodeKernel.dispatch(values)
    assert_equal(
        String(d.as_dictionary().indices()),
        String(array([0, 1, 2, 2, 1, None], int32)),
    )
    ref entries = d.as_dictionary().dictionary().as_float64()
    assert_equal(len(entries), 3)
    assert_equal(entries.unsafe_get(0), 1.0)
    assert_true(entries.unsafe_get(1) != entries.unsafe_get(1))
    assert_equal(entries.unsafe_get(2), 0.0)


def test_dictionary_encode_edges() raises:
    _check_encoded(
        DictionaryEncodeKernel.dispatch(array(int32)),
        array(int32),
        array(int32),
    )
    var all_null: DynArray = nulls(2, int64)
    _check_encoded(
        DictionaryEncodeKernel.dispatch(all_null),
        nulls(2, int32),
        array(int64),
    )
    _check_encoded(
        DictionaryEncodeKernel.dispatch(all_null, encode_nulls=True),
        array([0, 0], int32),
        nulls(1, int64),
    )
    var sliced: DynArray = array([5, 6, 5, 7, 6], int64)
    _check_encoded(
        DictionaryEncodeKernel.dispatch(sliced.slice(1)),
        array([0, 1, 2, 0], int32),
        array([6, 5, 7], int64),
    )
    var dictionary: DynArray = DictionaryArray.from_arrays(
        array([1, 0, 1], int8), array(["x", "y"])
    )
    var same = DictionaryEncodeKernel.dispatch(dictionary)
    assert_true(same.dtype() == dictionary.dtype())
    assert_equal(String(same), String(dictionary))


def test_dictionary_encode_is_first_seen_on_a_parallel_context() raises:
    var b = Int64Builder(100_000)
    for i in range(100_000):
        b.append(Int64((i * 7919) % 40_000))
    var values: DynArray = b.finish()
    var d = DictionaryEncodeKernel.dispatch(values, ctx=ExecContext.parallel())
    ref indices = d.as_dictionary().indices().as_int32()
    ref entries = d.as_dictionary().dictionary().as_int64()
    assert_equal(len(entries), 40_000)
    for i in range(40_000):
        assert_equal(Int(entries.unsafe_get(i)), (i * 7919) % 40_000)
    for i in range(100_000):
        assert_equal(Int(indices.unsafe_get(i)), i % 40_000)


# ---------------------------------------------------------------------------
# value_counts
# ---------------------------------------------------------------------------


def _check_counts(counted: DynArray, values: DynArray, counts: DynArray) raises:
    ref s = counted.as_struct()
    assert_equal(String(s.field("values")), String(values))
    assert_true(s.field("counts").as_int64() == counts.as_int64())


def test_value_counts_counts_null_as_a_value() raises:
    var ints: DynArray = array([3, 1, 3, None, 2, None, 3], int64)
    _check_counts(
        ValueCountsKernel.dispatch(ints),
        array([3, 1, None, 2], int64),
        array([3, 1, 2, 1], int64),
    )
    _check_counts(
        ValueCountsKernel.dispatch(_strings()),
        array(["b", "a", None, "c"]),
        array([2, 2, 2, 1], int64),
    )
    var empty: DynArray = array(int64)
    assert_equal(len(ValueCountsKernel.dispatch(empty)), 0)


def test_value_counts_of_a_dictionary() raises:
    """A dictionary column is counted by value, in first-seen order, and the
    values stay a dictionary."""
    var dictionary: DynArray = DictionaryArray.from_arrays(
        array([1, None, 1, 0], int8), array(["x", "y", "z"])
    )
    var counted = ValueCountsKernel.dispatch(dictionary)
    ref s = counted.as_struct()
    var values = s.field("values")
    assert_true(values.dtype() == dictionary.dtype())
    ref d = values.as_dictionary()
    assert_equal(String(d[0]), "y")
    assert_true(not d.is_valid(1))
    assert_equal(String(d[2]), "x")
    assert_true(s.field("counts").as_int64() == array([2, 1, 1], int64))


# ---------------------------------------------------------------------------
# mode
# ---------------------------------------------------------------------------


def _check_mode(found: DynArray, modes: DynArray, counts: DynArray) raises:
    ref s = found.as_struct()
    assert_equal(String(s.field("mode")), String(modes))
    assert_true(s.field("count").as_int64() == counts.as_int64())


def _common() raises -> DynArray:
    return array([3, 1, 3, None, 2, 1, 2, None, None], int64)


def test_mode_orders_by_count_then_value() raises:
    _check_mode(
        ModeKernel.dispatch(_common()), array([1], int64), array([2], int64)
    )
    _check_mode(
        ModeKernel.dispatch(_common(), 2),
        array([1, 2], int64),
        array([2, 2], int64),
    )
    _check_mode(
        ModeKernel.dispatch(_common(), 5),
        array([1, 2, 3], int64),
        array([2, 2, 2], int64),
    )
    _check_mode(
        ModeKernel.dispatch(array([5, -3, 5, -3, 0], int8), 3),
        array([-3, 5, 0], int8),
        array([2, 2, 1], int64),
    )


def test_mode_nan_is_the_greatest_value() raises:
    var values: DynArray = array(
        [1.0, nan[DType.float64](), 2.0, 2.0, nan[DType.float64]()], float64
    )
    var found = ModeKernel.dispatch(values, 3)
    ref modes = found.as_struct().field("mode").as_float64()
    assert_equal(len(modes), 3)
    assert_equal(modes.unsafe_get(0), 2.0)
    assert_true(modes.unsafe_get(1) != modes.unsafe_get(1))
    assert_equal(modes.unsafe_get(2), 1.0)
    assert_true(
        found.as_struct().field("count").as_int64() == array([2, 2, 1], int64)
    )


def test_mode_empty_answers() raises:
    """No rows when a null is not skipped, when too few values are valid, and
    over no values at all; `n` below 1 is refused."""
    assert_equal(len(ModeKernel.dispatch(_common(), skip_nulls=False)), 0)
    assert_equal(len(ModeKernel.dispatch(_common(), min_count=8)), 0)
    assert_equal(len(ModeKernel.dispatch(_common(), min_count=6)), 1)
    assert_equal(len(ModeKernel.dispatch(array(int64))), 0)
    assert_equal(len(ModeKernel.dispatch(nulls(2, int64))), 0)
    with assert_raises(contains="InvalidError"):
        _ = ModeKernel.dispatch(_common(), 0)


def test_mode_of_strings() raises:
    _check_mode(
        ModeKernel.dispatch(array(["a", "b", "a", None])),
        array(["a"]),
        array([2], int64),
    )


# ---------------------------------------------------------------------------
# Run-end encoded input
# ---------------------------------------------------------------------------


def test_encoder_kernels_decode_run_end_encoded() raises:
    """Each kernel answers over a run-end encoded column what it answers
    over the decoded rows: b, null, a, a, b, read through a window inside the
    runs."""
    var ree: DynArray = RunEndEncodedArray.from_arrays(
        array([2, 3, 5, 7], int32), array(["b", None, "a", "b"])
    ).slice(1, 5)
    var decoded = RunEndDecodeKernel.dispatch(ree)
    for encode_nulls in [False, True]:
        assert_equal(
            String(DictionaryEncodeKernel.dispatch(ree, encode_nulls)),
            String(DictionaryEncodeKernel.dispatch(decoded, encode_nulls)),
        )
    assert_equal(
        String(ValueCountsKernel.dispatch(ree)),
        String(ValueCountsKernel.dispatch(decoded)),
    )
    assert_equal(
        String(ModeKernel.dispatch(ree, 2)),
        String(ModeKernel.dispatch(decoded, 2)),
    )
    assert_equal(len(ModeKernel.dispatch(ree, 1, skip_nulls=False)), 0)
