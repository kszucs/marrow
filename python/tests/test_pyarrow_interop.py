"""Test that pyarrow generated array can be processed with Mojo compute."""

import pyarrow as pa
import marrow as ma
import pytest


INT_TYPES = (pa.int8, pa.int16, pa.int32, pa.int64)
FLOAT_TYPES = (pa.float32, pa.float64)

# ── add ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_add_pyarrow_arrays(pa_type: pa.DType) -> None:
    """Test that we can call add on arrays created with pyarrow."""
    pa_a = pa.array([1, 2, 3], type=pa_type())
    pa_b = pa.array([10, 20, 30], type=pa_type())
    a = ma.array(pa_a)
    b = ma.array(pa_b)
    result = ma.add(a, b)
    assert len(result) == 3
    assert result.null_count() == 0
    out = pa.array(result)
    assert out[0].as_py() == 11
    assert out[1].as_py() == 22
    assert out[2].as_py() == 33


@pytest.mark.parametrize("pa_type", FLOAT_TYPES)
def test_mojo_add_pyarrow_float(pa_type: pa.DType) -> None:
    pa_a = pa.array([1.0, 2.0, 3.0], type=pa_type())
    pa_b = pa.array([0.5, 1.5, 2.5], type=pa_type())
    a = ma.array(pa_a)
    b = ma.array(pa_b)
    result = ma.add(a, b)
    assert len(result) == 3
    assert result.null_count() == 0
    out = pa.array(result)
    assert out[0].as_py() == pytest.approx(1.5)
    assert out[1].as_py() == pytest.approx(3.5)
    assert out[2].as_py() == pytest.approx(5.5)


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_add_pyarrow_nulls_propagate(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, None, 3], type=pa_type())
    pa_b = pa.array([10, 20, 30], type=pa_type())
    a = ma.array(pa_a)
    b = ma.array(pa_b)
    result = ma.add(a, b)
    assert len(result) == 3
    assert result.null_count() == 1
    out = pa.array(result)
    assert out[0].as_py() == 11
    assert out[1].as_py() is None
    assert out[2].as_py() == 33


# ── sub ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_sub_pyarrow_arrays(pa_type: pa.DType) -> None:
    pa_a = pa.array([10, 20, 30], type=pa_type())
    pa_b = pa.array([1, 2, 3], type=pa_type())
    a = ma.array(pa_a)
    b = ma.array(pa_b)
    result = ma.sub(a, b)
    assert len(result) == 3
    assert result.null_count() == 0
    out = pa.array(result)
    assert out[0].as_py() == 9
    assert out[1].as_py() == 18
    assert out[2].as_py() == 27


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_sub_pyarrow_nulls_propagate(pa_type: pa.DType) -> None:
    pa_a = pa.array([10, None, 30], type=pa_type())
    pa_b = pa.array([1, 2, None], type=pa_type())
    a = ma.array(pa_a)
    b = ma.array(pa_b)
    result = ma.sub(a, b)
    assert result.__len__() == 3
    assert len(result) == 3
    assert result.null_count() == 2


# ── mul ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_mul_pyarrow_arrays(pa_type: pa.DType) -> None:
    pa_a = pa.array([2, 3, 4], type=pa_type())
    pa_b = pa.array([5, 6, 7], type=pa_type())
    a = ma.array(pa_a)
    b = ma.array(pa_b)
    result = ma.mul(a, b)
    assert len(result) == 3
    assert result.null_count() == 0
    out = pa.array(result)
    assert out[0].as_py() == 10
    assert out[1].as_py() == 18
    assert out[2].as_py() == 28


# ── div ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_div_pyarrow_arrays(pa_type: pa.DType) -> None:
    pa_a = pa.array([10, 20, 30], type=pa_type())
    pa_b = pa.array([2, 4, 5], type=pa_type())
    a = ma.array(pa_a)
    b = ma.array(pa_b)
    result = ma.div(a, b)
    assert len(result) == 3
    assert result.null_count() == 0
    out = pa.array(result)
    assert out[0].as_py() == 5
    assert out[1].as_py() == 5
    assert out[2].as_py() == 6


# ── aggregates ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_sum_pyarrow_int64(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, 2, 3, 4], type=pa_type())
    assert ma.sum_(ma.array(pa_a)) == 10.0


@pytest.mark.parametrize("pa_type", FLOAT_TYPES)
def test_mojo_sum_pyarrow_float(pa_type: pa.DType) -> None:
    pa_a = pa.array([1.5, 2.5, 3.0], type=pa_type())
    assert ma.sum_(ma.array(pa_a)) == pytest.approx(7.0)


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_sum_pyarrow_skips_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, None, 3, None], type=pa_type())
    assert ma.sum_(ma.array(pa_a)) == 4.0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_min_pyarrow(pa_type: pa.DType) -> None:
    pa_a = pa.array([3, 1, 4, 1, 5], type=pa_type())
    assert ma.min_(ma.array(pa_a)) == 1.0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_max_pyarrow(pa_type: pa.DType) -> None:
    pa_a = pa.array([3, 1, 4, 1, 5], type=pa_type())
    assert ma.max_(ma.array(pa_a)) == 5.0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_min_pyarrow_skips_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([3, None, 1, None], type=pa_type())
    assert ma.min_(ma.array(pa_a)) == 1.0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_max_pyarrow_skips_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([3, None, 5, None], type=pa_type())
    assert ma.max_(ma.array(pa_a)) == 5.0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_product_pyarrow(pa_type: pa.DType) -> None:
    pa_a = pa.array([2, 3, 4], type=pa_type())
    assert ma.product(ma.array(pa_a)) == 24.0


# ── filter ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_filter_pyarrow_arrays(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, 2, 3, 4, 5], type=pa_type())
    pa_mask = pa.array([True, False, True, False, True])
    result = ma.filter_(ma.array(pa_a), ma.array(pa_mask))
    assert result.__len__() == 3
    assert result.null_count() == 0
    out = pa.array(result)
    assert out[0].as_py() == 1
    assert out[1].as_py() == 3
    assert out[2].as_py() == 5


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_filter_pyarrow_preserves_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, None, 3, None, 5], type=pa_type())
    pa_mask = pa.array([True, True, True, False, True])
    result = ma.filter_(ma.array(pa_a), ma.array(pa_mask))
    assert result.__len__() == 4
    assert result.null_count() == 1


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_mojo_filter_pyarrow_all_false(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, 2, 3], type=pa_type())
    pa_mask = pa.array([False, False, False])
    result = ma.filter_(ma.array(pa_a), ma.array(pa_mask))
    assert result.__len__() == 0


# ── pyarrow → mojo compute → pyarrow roundtrip ───────────────────────────────


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_pyarrow_to_mojo_compute_to_pyarrow_int32(pa_type: pa.DType) -> None:
    pa_a = pa.array([7, 42, -1], type=pa_type())
    pa_b = pa.array([3, 8, 1], type=pa_type())
    result = ma.add(ma.array(pa_a), ma.array(pa_b))
    out = pa.array(result)
    assert out.type == pa_type()
    assert out[0].as_py() == 10
    assert out[1].as_py() == 50
    assert out[2].as_py() == 0


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_pyarrow_to_mojo_compute_to_pyarrow_with_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, None, 3, None], type=pa_type())
    pa_b = pa.array([10, 20, None, 40], type=pa_type())
    result = ma.add(ma.array(pa_a), ma.array(pa_b))
    out = pa.array(result)
    assert out.type == pa_type()
    assert out[0].as_py() == 11
    assert out[1].as_py() is None
    assert out[2].as_py() is None
    assert out[3].as_py() is None


@pytest.mark.parametrize("pa_type", INT_TYPES)
def test_pyarrow_to_mojo_drop_nulls(pa_type: pa.DType) -> None:
    pa_a = pa.array([1, None, 3, None, 5], type=pa_type())
    result = ma.drop_nulls(ma.array(pa_a))
    assert len(result) == 3
    assert result.null_count() == 0
    out = pa.array(result)
    assert out[0].as_py() == 1
    assert out[1].as_py() == 3
    assert out[2].as_py() == 5
