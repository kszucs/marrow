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
    result = ma.add(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_add_float64():
    a = ma.array([1.0, 2.0, 3.0])
    b = ma.array([0.5, 1.5, 2.5])
    result = ma.add(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_add_propagates_nulls():
    a = ma.array([1, None, 3])
    b = ma.array([10, 20, 30])
    result = ma.add(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 1


def test_add_both_null():
    a = ma.array([None, 2, None])
    b = ma.array([10, None, 30])
    result = ma.add(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 3


# ── sum ──────────────────────────────────────────────────────────────────────


def test_sum_int64():
    assert ma.sum(ma.array([1, 2, 3, 4]), ma.ExecutionContext.serial()) == 10.0


def test_sum_float64():
    assert ma.sum(ma.array([1.5, 2.5, 3.0]), ma.ExecutionContext.serial()) == 7.0


def test_sum_skips_nulls():
    assert ma.sum(ma.array([1, None, 3, None]), ma.ExecutionContext.serial()) == 4.0


def test_sum_all_nulls_returns_zero():
    assert (
        ma.sum(ma.array([1, 2, 3], type=ma.int64()), ma.ExecutionContext.serial())
        == 6.0
    )


# ── product ──────────────────────────────────────────────────────────────────


def test_product_int64():
    assert ma.product(ma.array([1, 2, 3, 4]), ma.ExecutionContext.serial()) == 24.0


def test_product_float64():
    assert ma.product(ma.array([1.5, 2.0, 2.0]), ma.ExecutionContext.serial()) == 6.0


def test_product_skips_nulls():
    assert ma.product(ma.array([2, None, 3, None]), ma.ExecutionContext.serial()) == 6.0


# ── min ──────────────────────────────────────────────────────────────────────


def test_min_int64():
    assert ma.min(ma.array([3, 1, 4, 1, 5]), ma.ExecutionContext.serial()) == 1.0


def test_min_float64():
    assert ma.min(ma.array([3.5, 1.5, 2.0]), ma.ExecutionContext.serial()) == 1.5


def test_min_skips_nulls():
    assert ma.min(ma.array([3, None, 1, None]), ma.ExecutionContext.serial()) == 1.0


# ── max ──────────────────────────────────────────────────────────────────────


def test_max_int64():
    assert ma.max(ma.array([3, 1, 4, 1, 5]), ma.ExecutionContext.serial()) == 5.0


def test_max_float64():
    assert ma.max(ma.array([3.5, 1.5, 4.0]), ma.ExecutionContext.serial()) == 4.0


def test_max_skips_nulls():
    assert ma.max(ma.array([3, None, 5, None]), ma.ExecutionContext.serial()) == 5.0


# ── any ──────────────────────────────────────────────────────────────────────


def test_any_with_true():
    assert ma.any(ma.array([False, True, False])) == True


def test_any_all_false():
    assert ma.any(ma.array([False, False, False])) == False


def test_any_skips_nulls_finds_true():
    assert ma.any(ma.array([False, None, True])) == True


def test_any_skips_nulls_all_false():
    assert ma.any(ma.array([False, None, False])) == False


def test_any_empty_or_all_null_returns_false():
    # identity for any is False
    assert ma.any(ma.array([False, False], type=ma.bool_())) == False


# ── all ──────────────────────────────────────────────────────────────────────


def test_all_all_true():
    assert ma.all(ma.array([True, True, True])) == True


def test_all_with_false():
    assert ma.all(ma.array([True, False, True])) == False


def test_all_skips_nulls_all_valid_true():
    assert ma.all(ma.array([True, None, True])) == True


def test_all_skips_nulls_finds_false():
    assert ma.all(ma.array([True, None, False])) == False


def test_all_empty_or_all_null_returns_true():
    # identity for all is True
    assert ma.all(ma.array([True, True], type=ma.bool_())) == True


# ── count_distinct / approx_count_distinct ────────────────────────────────────


def test_count_distinct_basic():
    assert ma.count_distinct(ma.array([1, 2, 2, 3, 3, 3])) == 3


def test_count_distinct_excludes_nulls():
    assert ma.count_distinct(ma.array([1, 2, None, 2, None, 3])) == 3


def test_count_distinct_empty_and_all_null():
    assert ma.count_distinct(ma.array([], type=ma.int64())) == 0
    assert ma.count_distinct(ma.array([None, None], type=ma.int64())) == 0


def test_count_distinct_matches_pyarrow():
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    a = pa.array(np.random.default_rng(0).integers(0, 4000, 200_000))
    assert ma.count_distinct(ma.array(a)) == pc.count_distinct(a).as_py()


def test_approx_count_distinct_small_is_near_exact():
    # linear-counting regime keeps small cardinalities within ~1 of exact
    a = ma.array([i % 100 for i in range(5000)])
    assert abs(int(ma.approx_count_distinct(a).as_py()) - 100) <= 2


def test_approx_count_distinct_within_tolerance():
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    a = pa.array(np.arange(1_000_000))
    true = pc.count_distinct(a).as_py()
    est = int(ma.approx_count_distinct(ma.array(a)).as_py())
    assert abs(est - true) / true < 0.02


# ── subtract ─────────────────────────────────────────────────────────────────


def test_sub_int64():
    a = ma.array([10, 20, 30])
    b = ma.array([1, 2, 3])
    result = ma.subtract(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_sub_float64():
    a = ma.array([5.0, 3.0, 1.0])
    b = ma.array([1.0, 1.0, 1.0])
    result = ma.subtract(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_sub_propagates_nulls():
    a = ma.array([10, None, 30])
    b = ma.array([1, 2, 3])
    result = ma.subtract(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 1


# ── multiply ─────────────────────────────────────────────────────────────────


def test_mul_int64():
    a = ma.array([2, 3, 4])
    b = ma.array([5, 6, 7])
    result = ma.multiply(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_mul_float64():
    a = ma.array([1.5, 2.0, 3.0])
    b = ma.array([2.0, 2.0, 2.0])
    result = ma.multiply(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_mul_propagates_nulls():
    a = ma.array([2, None, 4])
    b = ma.array([5, 6, None])
    result = ma.multiply(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 2


# ── divide ───────────────────────────────────────────────────────────────────


def test_div_int64():
    a = ma.array([10, 20, 30])
    b = ma.array([2, 4, 5])
    result = ma.divide(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_div_float64():
    a = ma.array([9.0, 6.0, 3.0])
    b = ma.array([3.0, 2.0, 1.0])
    result = ma.divide(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_div_propagates_nulls():
    a = ma.array([10, None, 30])
    b = ma.array([2, 4, None])
    result = ma.divide(a, b, ma.ExecutionContext.serial())
    assert result.__len__() == 3
    assert result.null_count() == 2


# ── filter ───────────────────────────────────────────────────────────────────


def test_filter_int64_keeps_selected():
    a = ma.array([1, 2, 3, 4, 5])
    mask = ma.array([True, False, True, False, True])
    result = ma.filter(a, mask)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_filter_float64():
    a = ma.array([1.0, 2.0, 3.0])
    mask = ma.array([False, True, True])
    result = ma.filter(a, mask)
    assert result.__len__() == 2
    assert result.null_count() == 0


def test_filter_preserves_nulls():
    a = ma.array([1, None, 3, None, 5])
    mask = ma.array([True, True, True, False, True])
    result = ma.filter(a, mask)
    assert result.__len__() == 4
    assert result.null_count() == 1


def test_filter_all_false_returns_empty():
    a = ma.array([1, 2, 3])
    mask = ma.array([False, False, False])
    result = ma.filter(a, mask)
    assert result.__len__() == 0


def test_filter_string_array():
    a = ma.array(["hello", "world", "foo"])
    mask = ma.array([True, False, True])
    result = ma.filter(a, mask)
    assert result.__len__() == 2
    assert result.null_count() == 0


# ── drop_null ────────────────────────────────────────────────────────────────


def test_drop_nulls_int64():
    a = ma.array([1, None, 3, None, 5])
    result = ma.drop_null(a)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_drop_nulls_no_nulls():
    a = ma.array([1, 2, 3])
    result = ma.drop_null(a)
    assert result.__len__() == 3
    assert result.null_count() == 0


def test_drop_nulls_all_null():
    a = ma.array([None, None, None], type=ma.int64())
    result = ma.drop_null(a)
    assert result.__len__() == 0
    assert result.null_count() == 0


def test_drop_nulls_float64():
    a = ma.array([1.0, None, 3.0])
    result = ma.drop_null(a)
    assert result.__len__() == 2
    assert result.null_count() == 0


# ── GPU ──────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
def test_device_construct():
    device = ma.ExecutionContext.parallel()
    assert device is not None


@pytest.mark.gpu
def test_array_to_device_and_back():
    device = ma.ExecutionContext.parallel()
    a = ma.array([1, 2, 3], type=ma.int32())
    a_gpu = a.to_device(device)
    a_cpu = a_gpu.to_cpu(device)
    assert a_cpu.__len__() == 3
    assert a_cpu.null_count() == 0


@pytest.mark.gpu
def test_add_gpu():
    device = ma.ExecutionContext.parallel()
    a = ma.array([1, 2, 3], type=ma.int32())
    b = ma.array([10, 20, 30], type=ma.int32())
    result = ma.add(a.to_device(device), b.to_device(device), device).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count() == 0


@pytest.mark.gpu
def test_sub_gpu():
    device = ma.ExecutionContext.parallel()
    a = ma.array([10, 20, 30], type=ma.int32())
    b = ma.array([1, 2, 3], type=ma.int32())
    result = ma.subtract(a.to_device(device), b.to_device(device), device).to_cpu(
        device
    )
    assert result.__len__() == 3
    assert result.null_count() == 0


@pytest.mark.gpu
def test_mul_gpu():
    device = ma.ExecutionContext.parallel()
    a = ma.array([2, 3, 4], type=ma.int32())
    b = ma.array([5, 6, 7], type=ma.int32())
    result = ma.multiply(a.to_device(device), b.to_device(device), device).to_cpu(
        device
    )
    assert result.__len__() == 3
    assert result.null_count() == 0


@pytest.mark.gpu
def test_sum_gpu():
    device = ma.ExecutionContext.parallel()
    a = ma.array([1.0, 2.0, 3.0], type=ma.float32())
    result = ma.sum(a.to_device(device), device)
    assert float(result) == 6.0


@pytest.mark.gpu
def test_min_gpu():
    device = ma.ExecutionContext.parallel()
    a = ma.array([3, 1, 4, 1, 5], type=ma.int32())
    result = ma.min(a.to_device(device), device)
    assert float(result) == 1.0


@pytest.mark.gpu
def test_max_gpu():
    device = ma.ExecutionContext.parallel()
    a = ma.array([3, 1, 4, 1, 5], type=ma.int32())
    result = ma.max(a.to_device(device), device)
    assert float(result) == 5.0


@pytest.mark.gpu
def test_equal_gpu():
    device = ma.ExecutionContext.parallel()
    a = ma.array([1, 2, 3], type=ma.int32())
    b = ma.array([1, 0, 3], type=ma.int32())
    result = ma.equal(a.to_device(device), b.to_device(device), device).to_cpu(device)
    assert result.__len__() == 3
    assert result.null_count() == 0
