"""Test the pyarrow-style ``marrow.compute.cast`` wrapper against PyArrow."""

import pytest
import pyarrow as pa
import pyarrow.compute as papc

import marrow as ma
from marrow import compute as pc


def _pylist(marrow_array):
    return pa.array(marrow_array).to_pylist()


def test_cast_int_to_float():
    a = ma.array([1, 2, 3], type=ma.int32())
    r = pc.cast(a, ma.float64())
    assert (
        _pylist(r)
        == papc.cast(pa.array([1, 2, 3], pa.int32()), pa.float64()).to_pylist()
    )


def test_cast_float_to_int_unsafe_truncates():
    a = ma.array([1.9, -1.9, 2.5], type=ma.float64())
    r = pc.cast(a, ma.int32(), safe=False)
    assert _pylist(r) == [1, -1, 2]


def test_cast_safe_overflow_raises():
    a = ma.array([300], type=ma.int32())
    with pytest.raises(Exception):
        pc.cast(a, ma.int8(), safe=True)


def test_cast_bool_to_int():
    a = ma.array([True, False, True])
    r = pc.cast(a, ma.int8())
    assert _pylist(r) == [1, 0, 1]


def test_cast_int_to_bool():
    a = ma.array([0, 5, 0], type=ma.int32())
    r = pc.cast(a, ma.bool_())
    assert _pylist(r) == [False, True, False]


def test_cast_temporal_unit():
    # NOTE: marrow's `timestamp` binding needs the tz argument passed explicitly
    # (default args aren't applied through the current FFI calling convention).
    a = ma.array([1, 2, 3], type=ma.int64())
    ts_s = pc.cast(a, ma.timestamp("s", ""))
    ts_ms = pc.cast(ts_s, ma.timestamp("ms", ""))
    back = pc.cast(ts_ms, ma.int64())
    assert _pylist(back) == [1000, 2000, 3000]


def test_cast_nulls_preserved():
    a = ma.array([1, None, 3], type=ma.int32())
    r = pc.cast(a, ma.float64())
    assert _pylist(r) == [1.0, None, 3.0]
