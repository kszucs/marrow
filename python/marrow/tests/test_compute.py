# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Test compute functions exposed to Python.

Covers:
  - add  (element-wise arithmetic)
  - sum, product, min, max  (numeric aggregates → float64)
  - any, all  (boolean aggregates → bool)
"""

import math

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


def test_rolling_rejects_bad_window():
    values = ma.array([1, 2, 3])
    with pytest.raises(ma.ArrowInvalid):
        ma.compute.rolling_sum(values, 0)
    with pytest.raises(ma.ArrowInvalid):
        ma.compute.rolling_sum(values, 2, min_samples=3)
    with pytest.raises(NotImplementedError):
        ma.compute.rolling_sum(values, 2, weights=[1, 2])
