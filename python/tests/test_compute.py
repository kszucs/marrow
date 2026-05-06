"""Test compute functions exposed to Python.

Covers:
  - add  (element-wise arithmetic)
  - sum_, product, min_, max_  (numeric aggregates → float64)
  - any_, all_  (boolean aggregates → bool)
"""

import pytest
import marrow as ma


# ── add ──────────────────────────────────────────────────────────────────────


def test_add_int64():
    a = ma.array([1, 2, 3])
    b = ma.array([10, 20, 30])
    result = ma.add(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_add_float64():
    a = ma.array([1.0, 2.0, 3.0])
    b = ma.array([0.5, 1.5, 2.5])
    result = ma.add(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_add_propagates_nulls():
    a = ma.array([1, None, 3])
    b = ma.array([10, 20, 30])
    result = ma.add(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 1


def test_add_both_null():
    a = ma.array([None, 2, None])
    b = ma.array([10, None, 30])
    result = ma.add(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 3


# ── sum_ ─────────────────────────────────────────────────────────────────────


def test_sum_int64():
    assert ma.sum_(ma.array([1, 2, 3, 4]), None) == 10.0


def test_sum_float64():
    assert ma.sum_(ma.array([1.5, 2.5, 3.0]), None) == 7.0


def test_sum_skips_nulls():
    assert ma.sum_(ma.array([1, None, 3, None]), None) == 4.0


def test_sum_all_nulls_returns_zero():
    assert ma.sum_(ma.array([1, 2, 3], type=ma.int64()), None) == 6.0


# ── product ──────────────────────────────────────────────────────────────────


def test_product_int64():
    assert ma.product(ma.array([1, 2, 3, 4]), None) == 24.0


def test_product_float64():
    assert ma.product(ma.array([1.5, 2.0, 2.0]), None) == 6.0


def test_product_skips_nulls():
    assert ma.product(ma.array([2, None, 3, None]), None) == 6.0


# ── min_ ─────────────────────────────────────────────────────────────────────


def test_min_int64():
    assert ma.min_(ma.array([3, 1, 4, 1, 5]), None) == 1.0


def test_min_float64():
    assert ma.min_(ma.array([3.5, 1.5, 2.0]), None) == 1.5


def test_min_skips_nulls():
    assert ma.min_(ma.array([3, None, 1, None]), None) == 1.0


# ── max_ ─────────────────────────────────────────────────────────────────────


def test_max_int64():
    assert ma.max_(ma.array([3, 1, 4, 1, 5]), None) == 5.0


def test_max_float64():
    assert ma.max_(ma.array([3.5, 1.5, 4.0]), None) == 4.0


def test_max_skips_nulls():
    assert ma.max_(ma.array([3, None, 5, None]), None) == 5.0


# ── any_ ─────────────────────────────────────────────────────────────────────


def test_any_with_true():
    assert ma.any_(ma.array([False, True, False])) == True


def test_any_all_false():
    assert ma.any_(ma.array([False, False, False])) == False


def test_any_skips_nulls_finds_true():
    assert ma.any_(ma.array([False, None, True])) == True


def test_any_skips_nulls_all_false():
    assert ma.any_(ma.array([False, None, False])) == False


def test_any_empty_or_all_null_returns_false():
    # identity for any_ is False
    assert ma.any_(ma.array([False, False], type=ma.bool_())) == False


# ── all_ ─────────────────────────────────────────────────────────────────────


def test_all_all_true():
    assert ma.all_(ma.array([True, True, True])) == True


def test_all_with_false():
    assert ma.all_(ma.array([True, False, True])) == False


def test_all_skips_nulls_all_valid_true():
    assert ma.all_(ma.array([True, None, True])) == True


def test_all_skips_nulls_finds_false():
    assert ma.all_(ma.array([True, None, False])) == False


def test_all_empty_or_all_null_returns_true():
    # identity for all_ is True
    assert ma.all_(ma.array([True, True], type=ma.bool_())) == True


# ── sub ──────────────────────────────────────────────────────────────────────


def test_sub_int64():
    a = ma.array([10, 20, 30])
    b = ma.array([1, 2, 3])
    result = ma.sub(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_sub_float64():
    a = ma.array([5.0, 3.0, 1.0])
    b = ma.array([1.0, 1.0, 1.0])
    result = ma.sub(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_sub_propagates_nulls():
    a = ma.array([10, None, 30])
    b = ma.array([1, 2, 3])
    result = ma.sub(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 1


# ── mul ──────────────────────────────────────────────────────────────────────


def test_mul_int64():
    a = ma.array([2, 3, 4])
    b = ma.array([5, 6, 7])
    result = ma.mul(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_mul_float64():
    a = ma.array([1.5, 2.0, 3.0])
    b = ma.array([2.0, 2.0, 2.0])
    result = ma.mul(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_mul_propagates_nulls():
    a = ma.array([2, None, 4])
    b = ma.array([5, 6, None])
    result = ma.mul(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 2


# ── div ──────────────────────────────────────────────────────────────────────


def test_div_int64():
    a = ma.array([10, 20, 30])
    b = ma.array([2, 4, 5])
    result = ma.div(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_div_float64():
    a = ma.array([9.0, 6.0, 3.0])
    b = ma.array([3.0, 2.0, 1.0])
    result = ma.div(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_div_propagates_nulls():
    a = ma.array([10, None, 30])
    b = ma.array([2, 4, None])
    result = ma.div(a, b, None)
    assert result.__len__() == 3
    assert result.null_count() == 2


# ── filter_ ──────────────────────────────────────────────────────────────────


def test_filter_int64_keeps_selected():
    a = ma.array([1, 2, 3, 4, 5])
    mask = ma.array([True, False, True, False, True])
    result = ma.filter_(a, mask)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_filter_float64():
    a = ma.array([1.0, 2.0, 3.0])
    mask = ma.array([False, True, True])
    result = ma.filter_(a, mask)
    assert result.__len__() == 2
    assert result.null_count() == 0


def test_filter_preserves_nulls():
    a = ma.array([1, None, 3, None, 5])
    mask = ma.array([True, True, True, False, True])
    result = ma.filter_(a, mask)
    assert result.__len__() == 4
    assert result.null_count() == 1


def test_filter_all_false_returns_empty():
    a = ma.array([1, 2, 3])
    mask = ma.array([False, False, False])
    result = ma.filter_(a, mask)
    assert result.__len__() == 0


def test_filter_string_array():
    a = ma.array(["hello", "world", "foo"])
    mask = ma.array([True, False, True])
    result = ma.filter_(a, mask)
    assert result.__len__() == 2
    assert result.null_count() == 0


# ── drop_nulls ────────────────────────────────────────────────────────────────


def test_drop_nulls_int64():
    a = ma.array([1, None, 3, None, 5])
    result = ma.drop_nulls(a)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_drop_nulls_no_nulls():
    a = ma.array([1, 2, 3])
    result = ma.drop_nulls(a)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_drop_nulls_all_null():
    a = ma.array([None, None, None], type=ma.int64())
    result = ma.drop_nulls(a)
    assert result.__len__() == 0
    assert result.null_count() == 0


def test_drop_nulls_float64():
    a = ma.array([1.0, None, 3.0])
    result = ma.drop_nulls(a)
    assert result.__len__() == 2
    assert result.null_count() == 0


# ── GPU ──────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
def test_device_construct():
    device = ma.Device()
    assert device is not None


@pytest.mark.gpu
def test_array_to_device_and_back():
    device = ma.Device()
    a = ma.array([1, 2, 3], type=ma.int32())
    a_gpu = a.to_device(device)
    a_cpu = a_gpu.to_cpu(device)
    assert a_cpu.__len__() == 3
    assert a_cpu.null_count() == 0


@pytest.mark.gpu
def test_add_gpu():
    device = ma.Device()
    a = ma.array([1, 2, 3], type=ma.int32())
    b = ma.array([10, 20, 30], type=ma.int32())
    result = ma.add(a.to_device(device), b.to_device(device), device).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count() == 0


@pytest.mark.gpu
def test_sub_gpu():
    device = ma.Device()
    a = ma.array([10, 20, 30], type=ma.int32())
    b = ma.array([1, 2, 3], type=ma.int32())
    result = ma.sub(a.to_device(device), b.to_device(device), device).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count() == 0


@pytest.mark.gpu
def test_mul_gpu():
    device = ma.Device()
    a = ma.array([2, 3, 4], type=ma.int32())
    b = ma.array([5, 6, 7], type=ma.int32())
    result = ma.mul(a.to_device(device), b.to_device(device), device).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count() == 0


@pytest.mark.gpu
def test_sum_gpu():
    device = ma.Device()
    a = ma.array([1.0, 2.0, 3.0], type=ma.float32())
    result = ma.sum_(a.to_device(device), device)
    assert float(result) == 6.0


@pytest.mark.gpu
def test_min_gpu():
    device = ma.Device()
    a = ma.array([3, 1, 4, 1, 5], type=ma.int32())
    result = ma.min_(a.to_device(device), device)
    assert float(result) == 1.0


@pytest.mark.gpu
def test_max_gpu():
    device = ma.Device()
    a = ma.array([3, 1, 4, 1, 5], type=ma.int32())
    result = ma.max_(a.to_device(device), device)
    assert float(result) == 5.0


@pytest.mark.gpu
def test_equal_gpu():
    device = ma.Device()
    a = ma.array([1, 2, 3], type=ma.int32())
    b = ma.array([1, 0, 3], type=ma.int32())
    result = ma.equal(a.to_device(device), b.to_device(device), device).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count() == 0
