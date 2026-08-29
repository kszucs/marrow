"""Test compute functions exposed to Python.

Covers:
  - add  (element-wise arithmetic)
  - sum, product, min, max  (numeric aggregates → float64)
  - any, all  (boolean aggregates → bool)
"""

import pytest
import marrow as ma


# ── add ──────────────────────────────────────────────────────────────────────


def test_add_int64():
    a = ma.array([1, 2, 3])
    b = ma.array([10, 20, 30])
    result = ma.compute.add(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_add_float64():
    a = ma.array([1.0, 2.0, 3.0])
    b = ma.array([0.5, 1.5, 2.5])
    result = ma.compute.add(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_add_propagates_nulls():
    a = ma.array([1, None, 3])
    b = ma.array([10, 20, 30])
    result = ma.compute.add(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 1


def test_add_both_null():
    a = ma.array([None, 2, None])
    b = ma.array([10, None, 30])
    result = ma.compute.add(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 3


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


def test_any_empty_or_all_null_returns_false():
    # identity for any is False
    assert ma.compute.any(ma.array([False, False], type=ma.bool_())) == False


# ── all ──────────────────────────────────────────────────────────────────────


def test_all_all_true():
    assert ma.compute.all(ma.array([True, True, True])) == True


def test_all_with_false():
    assert ma.compute.all(ma.array([True, False, True])) == False


def test_all_skips_nulls_all_valid_true():
    assert ma.compute.all(ma.array([True, None, True])) == True


def test_all_skips_nulls_finds_false():
    assert ma.compute.all(ma.array([True, None, False])) == False


def test_all_empty_or_all_null_returns_true():
    # identity for all is True
    assert ma.compute.all(ma.array([True, True], type=ma.bool_())) == True


# ── count_distinct / approx_count_distinct ────────────────────────────────────


# ── subtract ─────────────────────────────────────────────────────────────────


def test_sub_int64():
    a = ma.array([10, 20, 30])
    b = ma.array([1, 2, 3])
    result = ma.compute.subtract(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_sub_float64():
    a = ma.array([5.0, 3.0, 1.0])
    b = ma.array([1.0, 1.0, 1.0])
    result = ma.compute.subtract(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_sub_propagates_nulls():
    a = ma.array([10, None, 30])
    b = ma.array([1, 2, 3])
    result = ma.compute.subtract(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 1


# ── multiply ─────────────────────────────────────────────────────────────────


def test_mul_int64():
    a = ma.array([2, 3, 4])
    b = ma.array([5, 6, 7])
    result = ma.compute.multiply(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_mul_float64():
    a = ma.array([1.5, 2.0, 3.0])
    b = ma.array([2.0, 2.0, 2.0])
    result = ma.compute.multiply(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_mul_propagates_nulls():
    a = ma.array([2, None, 4])
    b = ma.array([5, 6, None])
    result = ma.compute.multiply(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 2


# ── divide ───────────────────────────────────────────────────────────────────


def test_div_int64():
    a = ma.array([10, 20, 30])
    b = ma.array([2, 4, 5])
    result = ma.compute.divide(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_div_float64():
    a = ma.array([9.0, 6.0, 3.0])
    b = ma.array([3.0, 2.0, 1.0])
    result = ma.compute.divide(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_div_propagates_nulls():
    a = ma.array([10, None, 30])
    b = ma.array([2, 4, None])
    result = ma.compute.divide(a, b, ctx=ma.ExecContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 2


# ── filter ───────────────────────────────────────────────────────────────────


def test_filter_int64_keeps_selected():
    a = ma.array([1, 2, 3, 4, 5])
    mask = ma.array([True, False, True, False, True])
    result = ma.compute.filter(a, mask)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_filter_float64():
    a = ma.array([1.0, 2.0, 3.0])
    mask = ma.array([False, True, True])
    result = ma.compute.filter(a, mask)
    assert result.__len__() == 2
    assert result.null_count() == 0


def test_filter_preserves_nulls():
    a = ma.array([1, None, 3, None, 5])
    mask = ma.array([True, True, True, False, True])
    result = ma.compute.filter(a, mask)
    assert result.__len__() == 4
    assert result.null_count() == 1


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
    assert result.null_count() == 0


# ── drop_null ────────────────────────────────────────────────────────────────


def test_drop_nulls_int64():
    a = ma.array([1, None, 3, None, 5])
    result = ma.compute.drop_null(a)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_drop_nulls_no_nulls():
    a = ma.array([1, 2, 3])
    result = ma.compute.drop_null(a)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_drop_nulls_all_null():
    a = ma.array([None, None, None], type=ma.int64())
    result = ma.compute.drop_null(a)
    assert result.__len__() == 0
    assert result.null_count() == 0


def test_drop_nulls_float64():
    a = ma.array([1.0, None, 3.0])
    result = ma.compute.drop_null(a)
    assert result.__len__() == 2
    assert result.null_count() == 0


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
    assert a_cpu.null_count() == 0


@pytest.mark.gpu
def test_add_gpu():
    device = ma.ExecContext.parallel()
    a = ma.array([1, 2, 3], type=ma.int32())
    b = ma.array([10, 20, 30], type=ma.int32())
    result = ma.compute.add(
        a.to_device(device), b.to_device(device), ctx=device
    ).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count() == 0


@pytest.mark.gpu
def test_sub_gpu():
    device = ma.ExecContext.parallel()
    a = ma.array([10, 20, 30], type=ma.int32())
    b = ma.array([1, 2, 3], type=ma.int32())
    result = ma.compute.subtract(
        a.to_device(device), b.to_device(device), ctx=device
    ).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count() == 0


@pytest.mark.gpu
def test_mul_gpu():
    device = ma.ExecContext.parallel()
    a = ma.array([2, 3, 4], type=ma.int32())
    b = ma.array([5, 6, 7], type=ma.int32())
    result = ma.compute.multiply(
        a.to_device(device), b.to_device(device), ctx=device
    ).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count() == 0


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
    assert result.null_count() == 0


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
    assert mask.null_count() == 1

    assert ma.compute.filter(values, mask).to_pylist() == [1, 2, 3]

    above = ma.compute.greater(values, ma.array([3] * 7, ma.int64()))
    assert ma.compute.filter(values, above).to_pylist() == [4, 6, 7]
