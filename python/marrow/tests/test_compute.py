# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Test compute functions exposed to Python.

Covers:
  - add  (element-wise arithmetic)
  - sum, product, min, max  (numeric aggregates → float64)
  - any, all  (boolean aggregates → bool)
"""

import math
import struct
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import marrow as ma


# ── add ──────────────────────────────────────────────────────────────────────


def test_add_int64():
    a = ma.array([1, 2, 3])
    b = ma.array([10, 20, 30])
    result = ma.compute.add(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 0


def test_add_float64():
    a = ma.array([1.0, 2.0, 3.0])
    b = ma.array([0.5, 1.5, 2.5])
    result = ma.compute.add(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 0


def test_add_propagates_nulls():
    a = ma.array([1, None, 3])
    b = ma.array([10, 20, 30])
    result = ma.compute.add(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 1


def test_add_both_null():
    a = ma.array([None, 2, None])
    b = ma.array([10, None, 30])
    result = ma.compute.add(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 3


# ── sum ──────────────────────────────────────────────────────────────────────


# ── product ──────────────────────────────────────────────────────────────────


# ── min ──────────────────────────────────────────────────────────────────────


# ── max ──────────────────────────────────────────────────────────────────────


# ── any ──────────────────────────────────────────────────────────────────────


def test_any_with_true():
    assert ma.compute.any(ma.array([False, True, False])) == True


def test_any_all_false():
    assert ma.compute.any(ma.array([False, False, False])) == False


def test_any_skips_nulls_finds_true():
    assert ma.compute.any(ma.array([False, None, True])) == True


def test_any_skips_nulls_all_false():
    assert ma.compute.any(ma.array([False, None, False])) == False


def test_any_empty_or_all_null_returns_null():
    assert ma.compute.any(ma.array([], type=ma.bool_())) is None
    assert ma.compute.any(ma.array([None, None], type=ma.bool_())) is None
    assert ma.compute.any(ma.array([], type=ma.bool_()), min_count=0) == False


def test_any_without_skip_nulls():
    assert ma.compute.any(ma.array([False, None]), skip_nulls=False) is None
    assert ma.compute.any(ma.array([True, None]), skip_nulls=False) == True


# ── all ──────────────────────────────────────────────────────────────────────


def test_all_all_true():
    assert ma.compute.all(ma.array([True, True, True])) == True


def test_all_with_false():
    assert ma.compute.all(ma.array([True, False, True])) == False


def test_all_skips_nulls_all_valid_true():
    assert ma.compute.all(ma.array([True, None, True])) == True


def test_all_skips_nulls_finds_false():
    assert ma.compute.all(ma.array([True, None, False])) == False


def test_all_empty_or_all_null_returns_null():
    assert ma.compute.all(ma.array([], type=ma.bool_())) is None
    assert ma.compute.all(ma.array([None, None], type=ma.bool_())) is None
    assert ma.compute.all(ma.array([], type=ma.bool_()), min_count=0) == True


def test_all_without_skip_nulls():
    assert ma.compute.all(ma.array([True, None]), skip_nulls=False) is None
    assert ma.compute.all(ma.array([False, None]), skip_nulls=False) == False


# ── count_distinct / approx_count_distinct ────────────────────────────────────


# ── subtract ─────────────────────────────────────────────────────────────────


def test_sub_int64():
    a = ma.array([10, 20, 30])
    b = ma.array([1, 2, 3])
    result = ma.compute.subtract(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 0


def test_sub_float64():
    a = ma.array([5.0, 3.0, 1.0])
    b = ma.array([1.0, 1.0, 1.0])
    result = ma.compute.subtract(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 0


def test_sub_propagates_nulls():
    a = ma.array([10, None, 30])
    b = ma.array([1, 2, 3])
    result = ma.compute.subtract(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 1


# ── multiply ─────────────────────────────────────────────────────────────────


def test_mul_int64():
    a = ma.array([2, 3, 4])
    b = ma.array([5, 6, 7])
    result = ma.compute.multiply(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 0


def test_mul_float64():
    a = ma.array([1.5, 2.0, 3.0])
    b = ma.array([2.0, 2.0, 2.0])
    result = ma.compute.multiply(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 0


def test_mul_propagates_nulls():
    a = ma.array([2, None, 4])
    b = ma.array([5, 6, None])
    result = ma.compute.multiply(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 2


# ── divide ───────────────────────────────────────────────────────────────────


def test_div_int64():
    a = ma.array([10, 20, 30])
    b = ma.array([2, 4, 5])
    result = ma.compute.divide(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 0


def test_div_float64():
    a = ma.array([9.0, 6.0, 3.0])
    b = ma.array([3.0, 2.0, 1.0])
    result = ma.compute.divide(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 0


def test_div_propagates_nulls():
    a = ma.array([10, None, 30])
    b = ma.array([2, 4, None])
    result = ma.compute.divide(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count == 2


def test_div_by_zero_on_integers_raises():
    """`divide` mirrors `pyarrow.compute.divide`, which is the *checked*
    kernel: an integer lane has no `inf` to answer with, so a zero divisor is
    an error rather than the dividend the kernel substitutes its way to."""
    a = ma.array([10, -10, 0])
    b = ma.array([0, 0, 0])
    with pytest.raises(ma.ArrowInvalid, match="divide by zero"):
        ma.compute.divide(a, b, ctx=ma.ExecContext.serial())


def test_div_by_zero_on_floats_is_an_infinity():
    """Floats are not checked, because there the zero divisor has an answer —
    and pyarrow returns it too rather than raising."""
    a = ma.array([10.0, -10.0, 0.0])
    b = ma.array([0.0, 0.0, 0.0])
    result = ma.compute.divide(a, b, ctx=ma.ExecContext.serial())
    assert result.null_count == 0
    values = result.to_pylist()
    assert values[0] == math.inf
    assert values[1] == -math.inf
    assert math.isnan(values[2])


def test_div_by_zero_under_a_null_dividend_is_not_an_error():
    """A row the output nulls anyway is not an error: pyarrow raises only for
    a row that would have produced a value, so this answers NULL."""
    a = ma.array([None, 10])
    b = ma.array([0, 2])
    result = ma.compute.divide(a, b, ctx=ma.ExecContext.serial())
    assert result.null_count == 1
    assert result.to_pylist()[1] == 5


def test_div_by_a_null_divisor_is_not_an_error():
    """A *missing* divisor answers null where a zero one is an error.

    The second row is the one under test, and it is the shape that needs
    Kleene `or` and `all`'s valid-only reading to agree: the cast leaves a
    null divisor null, `null or false` is null, and a null is not a false. A
    null slot's data byte is unspecified, so it is the validity and not the
    value that has to carry this.

    Both non-raising shapes are in the one call — a zero divisor under a null
    dividend and a null divisor under a valid one — since a divisor holding
    neither never reaches the check at all.
    """
    a = ma.array([None, 10, 20])
    b = ma.array([0, None, 2])
    result = ma.compute.divide(a, b, ctx=ma.ExecContext.serial())
    assert result.null_count == 2
    assert result.to_pylist() == [None, None, 10]


# ── filter ───────────────────────────────────────────────────────────────────


def test_filter_int64_keeps_selected():
    a = ma.array([1, 2, 3, 4, 5])
    mask = ma.array([True, False, True, False, True])
    result = ma.compute.filter(a, mask)
    assert result.__len__() == 3
    assert result.null_count == 0


def test_filter_float64():
    a = ma.array([1.0, 2.0, 3.0])
    mask = ma.array([False, True, True])
    result = ma.compute.filter(a, mask)
    assert result.__len__() == 2
    assert result.null_count == 0


def test_filter_preserves_nulls():
    a = ma.array([1, None, 3, None, 5])
    mask = ma.array([True, True, True, False, True])
    result = ma.compute.filter(a, mask)
    assert result.__len__() == 4
    assert result.null_count == 1


def test_filter_all_false_returns_empty():
    a = ma.array([1, 2, 3])
    mask = ma.array([False, False, False])
    result = ma.compute.filter(a, mask)
    assert result.__len__() == 0


def test_filter_string_array():
    a = ma.array(["hello", "world", "foo"])
    mask = ma.array([True, False, True])
    result = ma.compute.filter(a, mask)
    assert result.__len__() == 2
    assert result.null_count == 0


# ── drop_null ────────────────────────────────────────────────────────────────


def test_drop_nulls_int64():
    a = ma.array([1, None, 3, None, 5])
    result = ma.compute.drop_null(a)
    assert result.__len__() == 3
    assert result.null_count == 0


def test_drop_nulls_no_nulls():
    a = ma.array([1, 2, 3])
    result = ma.compute.drop_null(a)
    assert result.__len__() == 3
    assert result.null_count == 0


def test_drop_nulls_all_null():
    a = ma.array([None, None, None], type=ma.int64())
    result = ma.compute.drop_null(a)
    assert result.__len__() == 0
    assert result.null_count == 0


def test_drop_nulls_float64():
    a = ma.array([1.0, None, 3.0])
    result = ma.compute.drop_null(a)
    assert result.__len__() == 2
    assert result.null_count == 0


# ── Threads ──────────────────────────────────────────────────────────────────


def test_cpu_count_follows_set_cpu_count():
    original = ma.cpu_count()
    assert original >= 1
    try:
        ma.set_cpu_count(3)
        assert ma.cpu_count() == 3
        # Work runs on the new pool.
        a = ma.array(list(range(100_000)), type=ma.int64())
        doubled = ma.compute.add(a, a, ctx=ma.ExecContext.parallel())
        assert doubled.to_pylist()[-1] == 2 * 99_999
    finally:
        ma.set_cpu_count(original)
    assert ma.cpu_count() == original


def test_io_thread_count_follows_set_io_thread_count():
    original = ma.io_thread_count()
    assert original >= 1
    try:
        ma.set_io_thread_count(2)
        assert ma.io_thread_count() == 2
    finally:
        ma.set_io_thread_count(original)
    assert ma.io_thread_count() == original


@pytest.mark.parametrize("setter", [ma.set_cpu_count, ma.set_io_thread_count])
def test_thread_counts_refuse_fewer_than_one_thread(setter):
    for count in (0, -1):
        with pytest.raises(ValueError, match="concurrency"):
            setter(count)
    with pytest.raises(TypeError):
        setter(2.5)


# ── GPU ──────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
def test_device_construct():
    device = ma.ExecContext.parallel()
    assert device is not None


@pytest.mark.gpu
def test_array_to_device_and_back():
    device = ma.ExecContext.parallel()
    a = ma.array([1, 2, 3], type=ma.int32())
    a_gpu = a.to_device(device)
    a_cpu = a_gpu.to_cpu(device)
    assert a_cpu.__len__() == 3
    assert a_cpu.null_count == 0


@pytest.mark.gpu
def test_add_gpu():
    device = ma.ExecContext.parallel()
    a = ma.array([1, 2, 3], type=ma.int32())
    b = ma.array([10, 20, 30], type=ma.int32())
    result = ma.compute.add(
        a.to_device(device), b.to_device(device), ctx=device
    ).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count == 0


@pytest.mark.gpu
def test_sub_gpu():
    device = ma.ExecContext.parallel()
    a = ma.array([10, 20, 30], type=ma.int32())
    b = ma.array([1, 2, 3], type=ma.int32())
    result = ma.compute.subtract(
        a.to_device(device), b.to_device(device), ctx=device
    ).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count == 0


@pytest.mark.gpu
def test_mul_gpu():
    device = ma.ExecContext.parallel()
    a = ma.array([2, 3, 4], type=ma.int32())
    b = ma.array([5, 6, 7], type=ma.int32())
    result = ma.compute.multiply(
        a.to_device(device), b.to_device(device), ctx=device
    ).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count == 0


@pytest.mark.gpu
def test_sum_gpu():
    device = ma.ExecContext.parallel()
    a = ma.array([1.0, 2.0, 3.0], type=ma.float32())
    result = ma.compute.sum(a.to_device(device), ctx=device)
    assert float(result) == 6.0


@pytest.mark.gpu
def test_min_gpu():
    device = ma.ExecContext.parallel()
    a = ma.array([3, 1, 4, 1, 5], type=ma.int32())
    result = ma.compute.min(a.to_device(device), ctx=device)
    assert float(result) == 1.0


@pytest.mark.gpu
def test_max_gpu():
    device = ma.ExecContext.parallel()
    a = ma.array([3, 1, 4, 1, 5], type=ma.int32())
    result = ma.compute.max(a.to_device(device), ctx=device)
    assert float(result) == 5.0


@pytest.mark.gpu
def test_equal_gpu():
    device = ma.ExecContext.parallel()
    a = ma.array([1, 2, 3], type=ma.int32())
    b = ma.array([1, 0, 3], type=ma.int32())
    result = ma.compute.equal(
        a.to_device(device), b.to_device(device), ctx=device
    ).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count == 0


def test_filter_null_mask_entry_is_not_selected():
    """A null in the mask must not select its row.

    Arrow drops rows whose mask entry is null (pyarrow's default
    ``null_selection_behavior="drop"``), and SQL agrees: ``WHERE v < 4``
    omits the row where ``v`` is NULL.

    The mask comes from a comparison rather than being built by hand, because
    the defect needs the *data* bit under the null to be set and only a real
    comparison produces that: the kernel evaluates every SIMD lane whatever
    the validity says, so a null input compares its raw payload (0), writes
    ``0 < 4`` = True, and only then marks the lane invalid. ``filter`` read
    that bit and selected the row.

    ``v > 3`` is asserted alongside because it was *accidentally* correct —
    ``0 > 3`` is False, so the stray bit happened to be clear — and a
    regression would reintroduce the asymmetry rather than break both.
    """
    values = ma.array([1, 2, 3, 4, None, 6, 7], ma.int64())
    bound = ma.array([4] * 7, ma.int64())

    mask = ma.compute.less(values, bound)
    assert mask.to_pylist() == [True, True, True, False, None, False, False]
    assert mask.null_count == 1

    assert ma.compute.filter(values, mask).to_pylist() == [1, 2, 3]

    above = ma.compute.greater(values, ma.array([3] * 7, ma.int64()))
    assert ma.compute.filter(values, above).to_pylist() == [4, 6, 7]


# ── list aggregates ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name, expected",
    [
        ("list_sum", [6, None, None, None, 4]),
        ("list_min", [1, None, None, None, 1]),
        ("list_max", [3, None, None, None, 3]),
        ("list_mean", [2.0, None, None, None, 2.0]),
        ("list_count", [3, 0, 0, None, 2]),
    ],
)
def test_list_aggregates_follow_duckdb(name, expected):
    """DuckDB's answers: an empty list, a list of nulls and a null list are
    null, except that ``list_count`` counts the first two as 0."""
    lists = pa.array([[1, 2, 3], [], [None], None, [1, None, 3]], pa.list_(pa.int64()))
    result = getattr(ma.compute, name)(ma.array(lists))
    assert pa.array(result).to_pylist() == expected


def test_list_aggregates_on_a_sliced_large_list():
    lists = pa.array([[9], [1, 2], None, [3, None]], pa.large_list(pa.int32()))
    result = pa.array(ma.compute.list_sum(ma.array(lists.slice(1))))
    assert result.type == pa.int64()
    assert result.to_pylist() == [3, None, 3]


# ── cumulative ───────────────────────────────────────────────────────────────


def _same_values(got, expected):
    """Equal lists, NaN equal to NaN."""
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        if isinstance(e, float) and math.isnan(e):
            assert isinstance(g, float) and math.isnan(g)
        else:
            assert g == e


@pytest.mark.parametrize("name", ["sum", "prod", "min", "max"])
@pytest.mark.parametrize(
    "type_",
    [pa.int8(), pa.int32(), pa.int64(), pa.uint8(), pa.float32(), pa.float64()],
    ids=str,
)
@pytest.mark.parametrize("skip_nulls", [False, True])
def test_cumulative_matches_pyarrow(name, type_, skip_nulls):
    data = [3, 1, None, 4, 1, 5, None, 2]
    if pa.types.is_floating(type_):
        data = [3.0, 1.0, None, 4.0, math.nan, 1.0, 5.0, None, 2.0]
    values = pa.array(data, type_)
    expected = getattr(pc, f"cumulative_{name}")(values, skip_nulls=skip_nulls)
    result = pa.array(
        getattr(ma.compute, f"cumulative_{name}")(
            ma.array(values), skip_nulls=skip_nulls
        )
    )
    assert result.type == expected.type
    _same_values(result.to_pylist(), expected.to_pylist())


def test_cumulative_start_is_not_implemented():
    with pytest.raises(NotImplementedError):
        ma.compute.cumulative_sum(ma.array([1, 2, 3]), start=1)


# ── pairwise_diff, shift ─────────────────────────────────────────────────────


@pytest.mark.parametrize("period", [-3, -1, 0, 1, 2, 3, 9])
@pytest.mark.parametrize(
    "type_", [pa.int8(), pa.uint8(), pa.int64(), pa.float64()], ids=str
)
def test_pairwise_diff_matches_pyarrow(period, type_):
    values = pa.array([5, 1, None, 7, 2, 9, 4], type_)
    expected = pc.pairwise_diff(values, period=period)
    result = pa.array(ma.compute.pairwise_diff(ma.array(values), period=period))
    assert result.type == expected.type
    assert result.to_pylist() == expected.to_pylist()


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, [None, 1, 2, None, 4]),
        (-2, [None, 4, 5, None, None]),
        (0, [1, 2, None, 4, 5]),
        (9, [None, None, None, None, None]),
    ],
)
def test_shift_matches_polars(n, expected):
    """Polars 1.43.2's answers."""
    values = ma.array([1, 2, None, 4, 5])
    assert pa.array(ma.compute.shift(values, n)).to_pylist() == expected


def test_shift_strings_match_polars():
    values = ma.array(["a", "b", None])
    assert pa.array(ma.compute.shift(values, 1)).to_pylist() == [None, "a", "b"]


def test_shift_fill_value_is_not_implemented():
    with pytest.raises(NotImplementedError):
        ma.compute.shift(ma.array([1, 2, 3]), 1, fill_value=0)


# ── rolling windows ──────────────────────────────────────────────────────────


_GAPPED = [1.0, 2.0, None, 4.0, 5.0, 6.0]


@pytest.mark.parametrize(
    "name, values, kwargs, expected",
    [
        (
            "rolling_sum",
            _GAPPED,
            {"window_size": 2},
            [None, 3.0, None, None, 9.0, 11.0],
        ),
        (
            "rolling_sum",
            _GAPPED,
            {"window_size": 2, "min_samples": 1},
            [1.0, 3.0, 2.0, 4.0, 9.0, 11.0],
        ),
        (
            "rolling_sum",
            _GAPPED,
            {"window_size": 3, "center": True},
            [None, None, None, None, 15.0, None],
        ),
        (
            "rolling_sum",
            [1, 2, 3, 4, 5, 6],
            {"window_size": 4, "center": True, "min_samples": 1},
            [3, 6, 10, 14, 18, 15],
        ),
        (
            "rolling_sum",
            [1, 2, 3, 4, 5, 6],
            {"window_size": 3, "center": True, "min_samples": 1},
            [3, 6, 9, 12, 15, 11],
        ),
        ("rolling_mean", [1, 2, 3, 4], {"window_size": 2}, [None, 1.5, 2.5, 3.5]),
        (
            "rolling_sum",
            [1.0, math.nan, 3.0, 4.0],
            {"window_size": 2},
            [None, math.nan, math.nan, 7.0],
        ),
    ],
)
def test_rolling_matches_polars(name, values, kwargs, expected):
    """Polars 1.43.2's answers."""
    result = getattr(ma.compute, name)(ma.array(values), **kwargs)
    _same_values(pa.array(result).to_pylist(), expected)


def test_rolling_window_wider_than_sys_maxsize():
    """A window of more than ``sys.maxsize`` rows reaches the column's first
    row, and both its ends when centred."""
    values = ma.array([1, 2, 3])
    trailing = ma.compute.rolling_sum(values, sys.maxsize + 1, min_samples=1)
    assert pa.array(trailing).to_pylist() == [1, 3, 6]
    centred = ma.compute.rolling_sum(
        values, 2 * sys.maxsize, center=True, min_samples=1
    )
    assert pa.array(centred).to_pylist() == [6, 6, 6]


def test_rolling_min_samples_past_32_bits():
    """No row's window holds ``2**32 + 1`` valid values, so every row is null."""
    values = ma.array([1, 2, 3])
    for name in ("rolling_sum", "rolling_mean"):
        result = getattr(ma.compute, name)(values, 2**32 + 1, min_samples=2**32 + 1)
        assert pa.array(result).to_pylist() == [None, None, None]


def test_rolling_rejects_bad_window():
    values = ma.array([1, 2, 3])
    with pytest.raises(ma.ArrowInvalid):
        ma.compute.rolling_sum(values, 0)
    with pytest.raises(ma.ArrowInvalid):
        ma.compute.rolling_sum(values, 2, min_samples=3)
    with pytest.raises(NotImplementedError):
        ma.compute.rolling_sum(values, 2, weights=[1, 2])


# ── indices_nonzero, replace_with_mask ───────────────────────────────────────


@pytest.mark.parametrize(
    "values",
    [
        pa.array([True, False, None, True]),
        pa.array([0, 3, None, -1, 0], pa.int8()),
        pa.array([0.0, -0.0, math.nan, None, 2.5]),
        pa.array([0, 1], pa.decimal128(5, 2)),
        pa.array([True, None, True, False, True]).slice(1),
        pa.array([i % 3 == 0 for i in range(130)]),
        pa.array([], pa.bool_()),
        pa.array([0, 5, 0, 1], pa.uint64()),
        pa.array([4, 0, None, 2, 0, 9], pa.int32()).slice(2),
    ],
    ids=str,
)
def test_indices_nonzero_matches_pyarrow(values):
    expected = pc.indices_nonzero(values)
    result = pa.array(ma.compute.indices_nonzero(ma.array(values)))
    assert result.type == expected.type
    assert result.to_pylist() == expected.to_pylist()


_REPLACE_VALUES = pa.array([1, 2, None, 4, 5], pa.int64())
_REPLACE_MASK = pa.array([True, False, True, None, True])


@pytest.mark.parametrize(
    "values, mask, replacements",
    [
        (_REPLACE_VALUES, _REPLACE_MASK, pa.array([10, 30, 50], pa.int64())),
        (_REPLACE_VALUES, _REPLACE_MASK, pa.array([None, 30, None], pa.int64())),
        (_REPLACE_VALUES, _REPLACE_MASK, pa.array([10, 30, 50, 70], pa.int64())),
        (
            pa.array(["a", "b", None]),
            pa.array([False, True, True]),
            pa.array(["X", "Y"]),
        ),
        (
            pa.array([1, 2, 3, 4, 5, 6], pa.int8()),
            pa.array([False, True, None, True, False, True]),
            pa.array([9, None, 8, 7], pa.int8()),
        ),
        (
            pa.array([1, 2], pa.timestamp("s")),
            pa.array([True, False]),
            pa.array([5], pa.timestamp("s")),
        ),
        (
            pa.array([1, 2], pa.decimal128(5, 2)),
            pa.array([True, False]),
            pa.array([5], pa.decimal128(5, 2)),
        ),
        (
            pa.array([b"ab", b"cd"], pa.binary(2)),
            pa.array([True, False]),
            pa.array([b"zz"], pa.binary(2)),
        ),
        (
            pa.array(["a", None, "c"], pa.large_string()),
            pa.array([True, True, False]),
            pa.array(["x", "y"], pa.large_string()),
        ),
        (_REPLACE_VALUES, pa.array([False] * 5), pa.array([], pa.int64())),
        (pa.array([], pa.int64()), pa.array([], pa.bool_()), pa.array([], pa.int64())),
        (
            _REPLACE_VALUES.slice(1),
            _REPLACE_MASK.slice(1),
            pa.array([0, 0, 0], pa.int64()).slice(1),
        ),
    ],
)
def test_replace_with_mask_matches_pyarrow(values, mask, replacements):
    expected = pc.replace_with_mask(values, mask, replacements)
    result = pa.array(
        ma.compute.replace_with_mask(
            ma.array(values), ma.array(mask), ma.array(replacements)
        )
    )
    assert result.type == expected.type
    assert result.to_pylist() == expected.to_pylist()


def test_replace_with_mask_rejects():
    values = ma.array(_REPLACE_VALUES)
    mask = ma.array(_REPLACE_MASK)
    with pytest.raises(ma.ArrowInvalid, match="appropriate length|expected"):
        ma.compute.replace_with_mask(values, mask, ma.array([10, 30], ma.int64()))
    with pytest.raises(ma.ArrowTypeError):
        ma.compute.replace_with_mask(values, mask, ma.array([10, 30, 50], ma.int32()))
    with pytest.raises(NotImplementedError):
        ma.compute.replace_with_mask(
            values, ma.array([True])[0], ma.array([1, 2, 3, 4, 5], ma.int64())
        )


def test_replace_with_mask_on_lists():
    """PyArrow refuses lists; marrow takes every layout ``take`` does."""
    lists = pa.array([[1], [2], None], pa.list_(pa.int64()))
    result = ma.compute.replace_with_mask(
        ma.array(lists),
        ma.array([False, True, True]),
        ma.array(pa.array([[7, 8], []], pa.list_(pa.int64()))),
    )
    assert pa.array(result).to_pylist() == [[1], [7, 8], []]


# ── dictionary_encode, value_counts, mode ────────────────────────────────────


_ENCODED = [
    pa.array(["b", "a", None, "b", "c", None, "a"]),
    pa.array([3, 1, 3, None, 2], pa.int64()),
    pa.array([True, None, False, True]),
    pa.array([1.0, math.nan, 2.0, math.nan, None]),
    pa.array([1, 1, 2], pa.decimal128(5, 2)),
    pa.array([1, 1, 2], pa.timestamp("ms")),
    pa.array(["a", None, "a"], pa.large_string()),
    # No null: PyArrow 23 answers a string view's null entry as "", not null.
    pa.array(["a", "b", "a"], pa.string_view()),
    pa.array([], pa.int32()),
    pa.array([None, None], pa.int64()),
    pa.array([5, 6, 5, 7, 6], pa.int64()).slice(1),
]


@pytest.mark.parametrize("values", _ENCODED, ids=str)
@pytest.mark.parametrize("null_encoding", ["mask", "encode"])
def test_dictionary_encode_matches_pyarrow(values, null_encoding):
    expected = pc.dictionary_encode(values, null_encoding=null_encoding)
    result = pa.array(
        ma.compute.dictionary_encode(ma.array(values), null_encoding=null_encoding)
    )
    assert result.type == expected.type
    assert result.indices.to_pylist() == expected.indices.to_pylist()
    _same_values(result.dictionary.to_pylist(), expected.dictionary.to_pylist())


def test_dictionary_encode_passes_a_dictionary_through():
    values = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 0], pa.int8()), pa.array(["x", "y"])
    )
    result = pa.array(ma.compute.dictionary_encode(ma.array(values)))
    assert result.type == values.type
    assert result.to_pylist() == values.to_pylist()


def test_dictionary_encode_rejects_an_unknown_null_encoding():
    with pytest.raises(ValueError, match="not a valid null encoding"):
        ma.compute.dictionary_encode(ma.array([1, 2]), null_encoding="drop")


@pytest.mark.parametrize(
    "values",
    _ENCODED
    + [
        pa.DictionaryArray.from_arrays(
            pa.array([1, None, 1, 0], pa.int8()), pa.array(["x", "y", "z"])
        )
    ],
    ids=str,
)
def test_value_counts_matches_pyarrow(values):
    expected = pc.value_counts(values)
    result = pa.array(ma.compute.value_counts(ma.array(values)))
    assert result.type == expected.type
    _same_values(
        result.field("values").to_pylist(), expected.field("values").to_pylist()
    )
    assert result.field("counts").to_pylist() == expected.field("counts").to_pylist()


_MODE_INTS = pa.array([3, 1, 3, None, 2, 1, 2, None, None], pa.int64())


@pytest.mark.parametrize(
    "values, kwargs",
    [
        (_MODE_INTS, {}),
        (_MODE_INTS, {"n": 2}),
        (_MODE_INTS, {"n": 5}),
        (_MODE_INTS, {"skip_nulls": False}),
        (_MODE_INTS, {"min_count": 6}),
        (_MODE_INTS, {"min_count": 8}),
        (pa.array([], pa.int64()), {}),
        (pa.array([None, None], pa.int64()), {}),
        (pa.array([1.0, math.nan, math.nan, 2.0, 2.0, math.nan]), {"n": 3}),
        (pa.array([True, None, False, True]), {"n": 2}),
        (pa.array([200, 200, 1], pa.uint8()), {"n": 2}),
        (pa.array([5, -3, 5, -3, 0], pa.int8()), {"n": 3}),
        (pa.array([1, 2, 2, 1, 3], pa.decimal128(5, 2)), {"n": 3}),
        (pa.array([9, 9, 9, 1, 1, 2], pa.int64()).slice(2), {"n": 2}),
    ],
)
def test_mode_matches_pyarrow(values, kwargs):
    expected = pc.mode(values, **kwargs)
    result = pa.array(ma.compute.mode(ma.array(values), **kwargs))
    assert result.type == expected.type
    _same_values(result.field("mode").to_pylist(), expected.field("mode").to_pylist())
    assert result.field("count").to_pylist() == expected.field("count").to_pylist()


def test_mode_of_strings_follows_duckdb():
    result = ma.compute.mode(ma.array(["a", "b", "a", None]))
    assert pa.array(result).to_pylist() == [{"mode": "a", "count": 2}]


def test_mode_rejects_n_below_one():
    with pytest.raises(ma.ArrowInvalid):
        ma.compute.mode(ma.array([1, 2]), n=0)


# ── run-end encoding ─────────────────────────────────────────────────────────


_NAN_PAYLOAD = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000001))[0]

_RUNS = [
    pa.array([5, 5, 5, 1, None, None, 2, 2], pa.int64()),
    # Two NaN payloads and both zeros: every one a run of its own bits.
    pa.array([math.nan, _NAN_PAYLOAD, _NAN_PAYLOAD, -0.0, 0.0, 0.0, None]),
    pa.array([True, True, None, False, False]),
    pa.array(["a", "a", None, "b", "b"]),
    pa.array([b"x", b"x", b"y"], pa.large_binary()),
    pa.array([1, 1, 2], pa.decimal128(5, 2)),
    pa.array([1, 1, None, 3], pa.date32()),
    pa.nulls(3),
    pa.array([], pa.int64()),
    pa.array([9, 5, 5, 1, 1, 7], pa.int64()).slice(1, 4),
]


def _bits(values):
    """A float array's values as their bits, so NaN payloads and signed
    zeros compare; any other array's values as they are."""
    if pa.types.is_floating(values.type):
        return values.view(pa.uint64()).to_pylist()
    return values.to_pylist()


@pytest.mark.parametrize("run_end_type", [pa.int16(), pa.int32(), pa.int64()], ids=str)
@pytest.mark.parametrize("values", _RUNS, ids=str)
def test_run_end_encode_matches_pyarrow(values, run_end_type):
    expected = pc.run_end_encode(values, run_end_type=run_end_type)
    result = pa.array(
        ma.compute.run_end_encode(ma.array(values), run_end_type=run_end_type)
    )
    assert result.type == expected.type
    assert result.run_ends.to_pylist() == expected.run_ends.to_pylist()
    assert _bits(result.values) == _bits(expected.values)


def test_run_end_encode_defaults_to_int32_run_ends():
    result = pa.array(ma.compute.run_end_encode(ma.array([1, 1, 2])))
    assert result.type == pa.run_end_encoded(pa.int32(), pa.int64())


@pytest.mark.parametrize(
    "values",
    [
        pa.array([[1], [1]]),
        pa.array([{"a": 1}]),
        pa.array(["a", "a"]).dictionary_encode(),
        pc.run_end_encode(pa.array([1, 1])),
    ],
    ids=str,
)
def test_run_end_encode_refuses_what_pyarrow_refuses(values):
    with pytest.raises(pa.ArrowNotImplementedError):
        pc.run_end_encode(values)
    with pytest.raises(ma.ArrowTypeError):
        ma.compute.run_end_encode(ma.array(values))


def test_run_end_encode_refuses_an_int8_run_end_type():
    with pytest.raises(pa.ArrowInvalid):
        pc.run_end_encode(pa.array([1]), run_end_type=pa.int8())
    with pytest.raises(ma.ArrowTypeError):
        ma.compute.run_end_encode(ma.array([1]), run_end_type=pa.int8())


def test_run_end_encode_overflows_int16():
    fits = ma.array(pa.array(range(32767), pa.int64()))
    assert len(ma.compute.run_end_encode(fits, run_end_type=pa.int16())) == 32767
    over = ma.array(pa.array(range(32768), pa.int64()))
    with pytest.raises(ma.ArrowInvalid, match="32767"):
        ma.compute.run_end_encode(over, run_end_type=pa.int16())


@pytest.mark.parametrize("sliced", [False, True], ids=["whole", "sliced"])
@pytest.mark.parametrize("values", _RUNS, ids=str)
def test_run_end_decode_matches_pyarrow(values, sliced):
    encoded = pc.run_end_encode(values)
    if sliced:
        # The window starts and ends inside a run where there are any.
        encoded = encoded.slice(1, max(len(encoded) - 2, 0))
    expected = pc.run_end_decode(encoded)
    result = pa.array(ma.compute.run_end_decode(ma.array(encoded)))
    assert result.type == expected.type
    assert _bits(result) == _bits(expected)


def test_run_end_decode_refuses_a_plain_array():
    with pytest.raises(ma.ArrowTypeError):
        ma.compute.run_end_decode(ma.array([1, 2]))


@pytest.mark.parametrize(
    "values, expected",
    [
        (pa.array([1, 1, None, None, 2, 1]), [0, 0, 1, 1, 2, 3]),
        (pa.array([math.nan, math.nan, 1.0]), [0, 0, 1]),
        (pa.array([math.nan, _NAN_PAYLOAD]), [0, 0]),
        (pa.array([-0.0, 0.0, 1.0]), [0, 0, 1]),
        (pa.array(["a", "a", None, "b"]), [0, 0, 1, 2]),
        (pa.array([True, True, None, False]), [0, 0, 1, 2]),
        (pa.array([], pa.int64()), []),
        (pa.array([[1], [1], [2]]), [0, 0, 1]),
        (pa.array([{"a": 1}, {"a": 1}, {"a": 2}]), [0, 0, 1]),
        (pa.array([None, None], pa.int64()), [0, 0]),
    ],
    ids=str,
)
def test_run_index_numbers_each_run(values, expected):
    """Each row's run, numbered from 0 under the group-by identity."""
    result = pa.array(ma.compute.run_index(ma.array(values)))
    assert result.type == pa.uint32()
    assert result.to_pylist() == expected


@pytest.mark.parametrize("name", ["is_null", "is_valid"])
def test_is_null_run_end_encoded_matches_the_decoded(name):
    """An encoded array's nulls are its values'; PyArrow 23 answers from the
    encoded array's own (absent) bitmap, so the oracle is the decoded array."""
    encoded = pc.run_end_encode(pa.array([1, 1, None, 2, 2, None])).slice(1, 4)
    expected = getattr(pc, name)(pc.run_end_decode(encoded))
    result = pa.array(getattr(ma.compute, name)(ma.array(encoded)))
    assert result.to_pylist() == expected.to_pylist()


# Rows 5, 5, null, null, 2, 2, 2, 7: a window starting and ending inside runs.
# PyArrow 23 has no filter, take, sort, cast or aggregate over run-end encoded
# values, so each kernel is compared with PyArrow's over the decoded rows.
_WINDOW = pc.run_end_encode(
    pa.array([5, 5, 5, None, None, 2, 2, 2, 7, 7], pa.int64())
).slice(1, 8)
_ROWS = pc.run_end_decode(_WINDOW)


def _rows(result):
    """A marrow result as PyArrow, decoded when it is run-end encoded."""
    out = pa.array(result)
    if pa.types.is_run_end_encoded(out.type):
        out = pc.run_end_decode(out)
    return out


def test_filter_run_end_encoded_matches_the_decoded():
    mask = pa.array([False, False, True, None, True, False, True, True])
    result = pa.array(ma.compute.filter(ma.array(_WINDOW), ma.array(mask)))
    assert result.type == _WINDOW.type
    assert _rows(result).to_pylist() == pc.filter(_ROWS, mask).to_pylist()


def test_take_run_end_encoded_matches_the_decoded():
    indices = pa.array([0, 1, 4, 6, None, 2, 7, 0], pa.int32())
    result = pa.array(ma.compute.take(ma.array(_WINDOW), ma.array(indices)))
    assert result.type == _WINDOW.type
    assert _rows(result).to_pylist() == pc.take(_ROWS, indices).to_pylist()
    with pytest.raises(ma.ArrowIndexError):
        ma.compute.take(ma.array(_WINDOW), ma.array(pa.array([8], pa.int32())))


@pytest.mark.parametrize("null_placement", ["at_start", "at_end"])
@pytest.mark.parametrize("order", ["ascending", "descending"])
def test_sort_run_end_encoded_matches_the_decoded(order, null_placement):
    keys = [("", order)]
    expected = pc.sort_indices(_ROWS, sort_keys=keys, null_placement=null_placement)
    indices = ma.compute.sort_indices(
        ma.array(_WINDOW), keys, null_placement=null_placement
    )
    assert pa.array(indices).to_pylist() == expected.to_pylist()
    result = pa.array(
        ma.compute.sort(ma.array(_WINDOW), keys, null_placement=null_placement)
    )
    assert result.type == _WINDOW.type
    assert _rows(result).to_pylist() == pc.take(_ROWS, expected).to_pylist()


def test_cast_run_end_encoded_matches_the_decoded():
    """To another run-end encoded type the values and run ends are cast; to
    a plain type the rows are decoded, then cast. A plain array is not cast
    to a run-end encoded type, as PyArrow does not."""
    expected = pc.cast(_ROWS, pa.float64()).to_pylist()
    target = pa.run_end_encoded(pa.int16(), pa.float64())
    result = pa.array(ma.compute.cast(ma.array(_WINDOW), target))
    assert result.type == target
    assert _rows(result).to_pylist() == expected
    result = pa.array(ma.compute.cast(ma.array(_WINDOW), pa.float64()))
    assert result.type == pa.float64()
    assert result.to_pylist() == expected
    with pytest.raises(pa.ArrowNotImplementedError):
        pc.cast(_ROWS, _WINDOW.type)
    with pytest.raises(ma.ArrowNotImplementedError):
        ma.compute.cast(ma.array(_ROWS), _WINDOW.type)


def test_bit_wise_not_run_end_encoded_matches_the_decoded():
    result = pa.array(ma.compute.bit_wise_not(ma.array(_WINDOW)))
    assert result.type == _WINDOW.type
    assert _rows(result).to_pylist() == pc.bit_wise_not(_ROWS).to_pylist()


@pytest.mark.parametrize("name", ["add", "equal"])
def test_binary_kernels_decode_run_end_encoded_operands(name):
    """An encoded operand is decoded: beside a plain one, and beside an
    encoding of the same rows in other runs."""
    plain = pa.array([10, 5, 20, None, 2, 30, 2, 7], pa.int64())
    oracle, kernel = getattr(pc, name), getattr(ma.compute, name)
    result = pa.array(kernel(ma.array(_WINDOW), ma.array(plain)))
    assert result.to_pylist() == oracle(_ROWS, plain).to_pylist()
    result = pa.array(kernel(ma.array(plain), ma.array(_WINDOW)))
    assert result.to_pylist() == oracle(plain, _ROWS).to_pylist()
    other = pc.run_end_encode(_ROWS)
    result = pa.array(kernel(ma.array(_WINDOW), ma.array(other)))
    assert result.to_pylist() == oracle(_ROWS, _ROWS).to_pylist()


def test_drop_null_run_end_encoded_matches_the_decoded():
    result = pa.array(ma.compute.drop_null(ma.array(_WINDOW)))
    assert result.type == _WINDOW.type
    assert _rows(result).to_pylist() == pc.drop_null(_ROWS).to_pylist()


@pytest.mark.parametrize("min_count", [0, 1, 3, 4])
@pytest.mark.parametrize("skip_nulls", [True, False])
@pytest.mark.parametrize("name", ["any", "all"])
def test_any_all_run_end_encoded_matches_the_decoded(name, skip_nulls, min_count):
    """`min_count` counts rows, not runs; these no longer abort the process."""
    values = pc.run_end_encode(
        pa.array([False, False, False, None, None, None, True, True])
    ).slice(1, 5)
    expected = getattr(pc, name)(
        pc.run_end_decode(values), skip_nulls=skip_nulls, min_count=min_count
    ).as_py()
    got = getattr(ma.compute, name)(
        ma.array(values), skip_nulls=skip_nulls, min_count=min_count
    )
    assert got == expected


def test_any_refuses_a_non_boolean_array():
    with pytest.raises(ma.ArrowTypeError):
        ma.compute.any(ma.array([1, 2]))


def test_encoder_kernels_run_end_encoded_match_the_decoded():
    encoded = pc.run_end_encode(pa.array([2, 2, None, 1, 1, 2, 2], pa.int64())).slice(
        1, 5
    )
    rows = pc.run_end_decode(encoded)
    for null_encoding in ["mask", "encode"]:
        expected = pc.dictionary_encode(rows, null_encoding=null_encoding)
        result = pa.array(
            ma.compute.dictionary_encode(ma.array(encoded), null_encoding=null_encoding)
        )
        assert result.type == expected.type
        assert result.to_pylist() == expected.to_pylist()
    expected = pc.value_counts(rows)
    result = pa.array(ma.compute.value_counts(ma.array(encoded)))
    assert result.type == expected.type
    assert result.to_pylist() == expected.to_pylist()
    expected = pc.mode(rows, n=2)
    result = pa.array(ma.compute.mode(ma.array(encoded), n=2))
    assert result.to_pylist() == expected.to_pylist()


def test_run_index_and_run_end_decode_read_run_end_encoded_values():
    """``run_index`` numbers an encoded array's rows as it numbers the decoded
    ones, and decoding an encoding of encoded values answers their
    encoding."""
    encoded = pc.run_end_encode(pa.array([1, 1, None, None, 2, 1, 1])).slice(1, 5)
    result = pa.array(ma.compute.run_index(ma.array(encoded)))
    assert result.to_pylist() == [0, 1, 1, 2, 3]
    inner = pc.run_end_encode(pa.array([5, 5, 6]))
    outer = pa.RunEndEncodedArray.from_arrays(pa.array([2, 5, 6], pa.int32()), inner)
    result = pa.array(ma.compute.run_end_decode(ma.array(outer)))
    assert result.type == inner.type
    assert pc.run_end_decode(result).to_pylist() == [5, 5, 5, 5, 5, 6]


# ── bitwise ──────────────────────────────────────────────────────────────────


_INTEGERS = [pa.int8(), pa.uint8(), pa.int32(), pa.int64(), pa.uint64()]


@pytest.mark.parametrize("name", ["bit_wise_and", "bit_wise_or", "bit_wise_xor"])
@pytest.mark.parametrize("type_", _INTEGERS, ids=str)
def test_bitwise_matches_pyarrow(name, type_):
    x = [15, 5, None, 7, 0, 127]
    y = [60, 3, 1, None, 0, 1]
    if pa.types.is_signed_integer(type_):
        x += [-1, -128]
        y += [7, 1]
    x, y = pa.array(x, type_), pa.array(y, type_)
    expected = getattr(pc, name)(x, y)
    result = pa.array(getattr(ma.compute, name)(ma.array(x), ma.array(y)))
    assert result.type == expected.type
    assert result.to_pylist() == expected.to_pylist()

    expected = pc.bit_wise_not(x)
    result = pa.array(ma.compute.bit_wise_not(ma.array(x)))
    assert result.type == expected.type
    assert result.to_pylist() == expected.to_pylist()


@pytest.mark.parametrize("name", ["shift_left", "shift_right"])
@pytest.mark.parametrize("type_", _INTEGERS, ids=str)
def test_shift_matches_pyarrow(name, type_):
    w = type_.bit_width
    signed = pa.types.is_signed_integer(type_)
    x = pa.array([1, 3, -5 if signed else 5, 1, 1, 7, None, 2], type_)
    amounts = pa.array([0, 1, w - 2, w - 1, w, -1 if signed else w + 1, 1, None], type_)
    expected = getattr(pc, name)(x, amounts)
    result = pa.array(getattr(ma.compute, name)(ma.array(x), ma.array(amounts)))
    assert result.type == expected.type
    assert result.to_pylist() == expected.to_pylist()


def test_bitwise_rejects_floats_and_mixed_types():
    floats = ma.array([1.0, 2.0])
    with pytest.raises(ma.ArrowTypeError):
        ma.compute.bit_wise_and(floats, floats)
    with pytest.raises(ma.ArrowTypeError):
        ma.compute.bit_wise_and(
            ma.array(pa.array([1], pa.int8())), ma.array(pa.array([1], pa.int16()))
        )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("list_bit_and", [8, None, None, None, 1, 6]),
        ("list_bit_or", [14, None, None, None, 7, -1]),
        ("list_bit_xor", [8, None, None, None, 6, -7]),
    ],
)
def test_list_bit_aggregates_follow_duckdb(name, expected):
    """DuckDB 1.5.5's answers: an empty list, a list of nulls and a null list
    are null, and the element type is kept."""
    lists = pa.array(
        [[12, 10, 14], [], [None], None, [5, None, 3], [-1, 6]],
        pa.list_(pa.int64()),
    )
    result = pa.array(getattr(ma.compute, name)(ma.array(lists)))
    assert result.type == pa.int64()
    assert result.to_pylist() == expected


def test_list_bit_and_keeps_the_element_type_and_takes_integers_only():
    narrow = pa.array([[1, 3]], pa.list_(pa.int8()))
    result = pa.array(ma.compute.list_bit_and(ma.array(narrow)))
    assert result.type == pa.int8()
    assert result.to_pylist() == [1]
    with pytest.raises(ma.ArrowTypeError):
        ma.compute.list_bit_and(ma.array(pa.array([[1.5]], pa.list_(pa.float64()))))
