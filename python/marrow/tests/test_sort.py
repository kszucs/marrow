"""Tests for sort Python bindings.

Covers:
  - array.argsort(order, null_placement) — PyArrow-compatible string order
  - array.sort(order, null_placement)
  - array.take(indices)
  - record_batch.sort_by(by, null_placement)
  - ma.sort_indices(array, *, ascending, null_placement)
  - ma.sort(array, *, ascending, null_placement)
"""

import pytest
import pyarrow as pa
import marrow as ma


# ── Helpers ───────────────────────────────────────────────────────────────────


def arr_to_pylist(marrow_arr) -> list:
    """Convert a marrow Array to a Python list via PyArrow zero-copy."""
    return pa.chunked_array([pa.array(marrow_arr)]).to_pylist()


# ── array.argsort ─────────────────────────────────────────────────────────────


def test_argsort_int64_ascending():
    arr = ma.array([3, 1, 2], type=ma.int64())
    idx = arr.argsort("ascending", "at_start")
    assert arr_to_pylist(idx) == [1, 2, 0]


def test_argsort_int64_descending():
    arr = ma.array([3, 1, 2], type=ma.int64())
    idx = arr.argsort("descending", "at_start")
    assert arr_to_pylist(idx) == [0, 2, 1]


def test_argsort_nulls_first():
    arr = ma.array([3, None, 1], type=ma.int64())
    idx = arr.argsort("ascending", "at_start")
    result = arr_to_pylist(idx)
    assert result[0] == 1  # null at index 1 sorted first
    assert result[1:] == [2, 0]


def test_argsort_nulls_last():
    arr = ma.array([3, None, 1], type=ma.int64())
    idx = arr.argsort("ascending", "at_end")
    result = arr_to_pylist(idx)
    assert result[-1] == 1  # null at index 1 sorted last
    assert result[:2] == [2, 0]


def test_argsort_float64():
    arr = ma.array([2.5, 1.0, 3.0], type=ma.float64())
    idx = arr.argsort("ascending", "at_start")
    assert arr_to_pylist(idx) == [1, 0, 2]


def test_argsort_string():
    arr = ma.array(["banana", "apple", "cherry"])
    idx = arr.argsort("ascending", "at_start")
    assert arr_to_pylist(idx) == [1, 0, 2]


def test_argsort_default_args():
    arr = ma.array([3, 1, 2], type=ma.int64())
    idx = arr.argsort()
    assert arr_to_pylist(idx) == [1, 2, 0]


# ── ma.sort_indices (module-level, supports kwargs) ───────────────────────────


# ── array.sort ────────────────────────────────────────────────────────────────


def test_sort_int64_ascending():
    arr = ma.array([3, 1, 2], type=ma.int64())
    result = arr.sort("ascending", "at_start")
    assert arr_to_pylist(result) == [1, 2, 3]


def test_sort_int64_descending():
    arr = ma.array([3, 1, 2], type=ma.int64())
    result = arr.sort("descending", "at_start")
    assert arr_to_pylist(result) == [3, 2, 1]


def test_sort_with_nulls_first():
    arr = ma.array([3, None, 1], type=ma.int64())
    result = arr.sort("ascending", "at_start")
    vals = arr_to_pylist(result)
    assert vals[0] is None
    assert vals[1:] == [1, 3]


def test_sort_with_nulls_last():
    arr = ma.array([3, None, 1], type=ma.int64())
    result = arr.sort("ascending", "at_end")
    vals = arr_to_pylist(result)
    assert vals[-1] is None
    assert vals[:2] == [1, 3]


def test_sort_default_args():
    arr = ma.array([3, 1, 2], type=ma.int64())
    result = arr.sort()
    assert arr_to_pylist(result) == [1, 2, 3]


# ── array.take ────────────────────────────────────────────────────────────────


def test_take_int64():
    arr = ma.array([10, 20, 30], type=ma.int64())
    idx = ma.array([2, 0, 1], type=ma.int32())
    result = arr.take(idx)
    assert arr_to_pylist(result) == [30, 10, 20]


def test_take_strings():
    arr = ma.array(["a", "b", "c"])
    idx = ma.array([1, 2, 0], type=ma.int32())
    result = arr.take(idx)
    assert arr_to_pylist(result) == ["b", "c", "a"]


def test_module_sort_indices_ascending():
    arr = ma.array([3, 1, 2], type=ma.int64())
    idx = ma.sort_indices(arr, ascending=True, null_placement="at_start")
    assert arr_to_pylist(idx) == [1, 2, 0]


def test_module_sort_indices_descending():
    arr = ma.array([3, 1, 2], type=ma.int64())
    idx = ma.sort_indices(arr, ascending=False, null_placement="at_start")
    assert arr_to_pylist(idx) == [0, 2, 1]


def test_module_sort_indices_defaults():
    arr = ma.array([3, 1, 2], type=ma.int64())
    idx = ma.sort_indices(arr)
    assert arr_to_pylist(idx) == [1, 2, 0]


# ── ma.sort (module-level, supports kwargs) ───────────────────────────────────


def test_module_sort_ascending():
    arr = ma.array([3, 1, 2], type=ma.int64())
    result = ma.sort(arr, ascending=True, null_placement="at_start")
    assert arr_to_pylist(result) == [1, 2, 3]


def test_module_sort_descending():
    arr = ma.array([3, 1, 2], type=ma.int64())
    result = ma.sort(arr, ascending=False, null_placement="at_start")
    assert arr_to_pylist(result) == [3, 2, 1]


def test_module_sort_defaults():
    arr = ma.array([3, 1, 2], type=ma.int64())
    result = ma.sort(arr)
    assert arr_to_pylist(result) == [1, 2, 3]


# ── record_batch.sort_by ──────────────────────────────────────────────────────


def test_sort_by_bare_string():
    rb = ma.record_batch(
        {
            "k": ma.array([3, 1, 2], type=ma.int64()),
            "v": ma.array([10, 20, 30], type=ma.int64()),
        }
    )
    result = rb.sort_by("k", None)
    assert arr_to_pylist(result.column("k")) == [1, 2, 3]
    assert arr_to_pylist(result.column("v")) == [20, 30, 10]


def test_sort_by_list_of_strings():
    rb = ma.record_batch(
        {
            "k": ma.array([3, 1, 2], type=ma.int64()),
            "v": ma.array([10, 20, 30], type=ma.int64()),
        }
    )
    result = rb.sort_by(["k"], None)
    assert arr_to_pylist(result.column("k")) == [1, 2, 3]


def test_sort_by_list_of_tuples_ascending():
    rb = ma.record_batch(
        {
            "k": ma.array([3, 1, 2], type=ma.int64()),
            "v": ma.array([10, 20, 30], type=ma.int64()),
        }
    )
    result = rb.sort_by([("k", "ascending")], None)
    assert arr_to_pylist(result.column("k")) == [1, 2, 3]


def test_sort_by_list_of_tuples_descending():
    rb = ma.record_batch(
        {
            "k": ma.array([3, 1, 2], type=ma.int64()),
            "v": ma.array([10, 20, 30], type=ma.int64()),
        }
    )
    result = rb.sort_by([("k", "descending")], None)
    assert arr_to_pylist(result.column("k")) == [3, 2, 1]
    assert arr_to_pylist(result.column("v")) == [10, 30, 20]


def test_sort_by_null_placement_at_end():
    rb = ma.record_batch(
        {
            "k": ma.array([3, None, 1], type=ma.int64()),
        }
    )
    result = rb.sort_by("k", "at_end")
    vals = arr_to_pylist(result.column("k"))
    assert vals[-1] is None
    assert vals[:2] == [1, 3]


def test_sort_by_preserves_row_correspondence():
    rb = ma.record_batch(
        {
            "a": ma.array([30, 10, 20], type=ma.int64()),
            "b": ma.array(["x", "y", "z"]),
        }
    )
    result = rb.sort_by([("a", "ascending")], None)
    assert arr_to_pylist(result.column("a")) == [10, 20, 30]
    assert arr_to_pylist(result.column("b")) == ["y", "z", "x"]
