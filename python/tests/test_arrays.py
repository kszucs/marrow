"""Test array() and infer_type() with Python built-in types.

Ported from PyArrow's test_convert_builtin.py, excluding pandas/numpy,
Decimal, Date/Time/Timestamp, Dictionary, and Map cases.
"""

import pytest
import marrow as ma


# ── infer_type ──────────────────────────────────────────────────────────────


def test_infer_empty():
    assert str(ma.infer_type([])) == "null"


def test_infer_all_none():
    assert str(ma.infer_type([None, None])) == "null"


def test_infer_bool():
    assert str(ma.infer_type([True, False, None])) == "bool"


def test_infer_int():
    assert str(ma.infer_type([1, 2, None])) == "int64"


def test_infer_float():
    assert str(ma.infer_type([1.5, 2.5, None])) == "float64"


def test_infer_string():
    assert str(ma.infer_type(["foo", "bar", None])) == "string"


def test_infer_bytes():
    assert str(ma.infer_type([b"foo", b"bar"])) == "binary"


def test_infer_bytes_with_none():
    assert str(ma.infer_type([b"foo", None, b"bar"])) == "binary"


def test_infer_mixed_int_float():
    # integers coerce up to float64 when floats are present
    assert str(ma.infer_type([1, 2.5])) == "float64"


def test_infer_mixed_bool_int():
    # bool + int → int64 (bool is a numeric subtype in Python)
    assert str(ma.infer_type([True, 1])) == "int64"


def test_infer_tuple_as_list():
    # tuples are treated the same as lists
    assert str(ma.infer_type([(1, 2), (3, 4)])) == "list<int64>"


def test_infer_nested_list():
    assert str(ma.infer_type([[1, 2], [3, 4]])) == "list<int64>"


def test_infer_nested_list_with_none():
    assert str(ma.infer_type([[1, 2], None, [3]])) == "list<int64>"


def test_infer_struct():
    t = ma.infer_type([{"a": 5, "b": "foo", "c": True}])
    assert str(t) == "struct<a: int64, b: string, c: bool>"


def test_infer_struct_empty_dict():
    assert str(ma.infer_type([{}])) == "struct<>"


def test_infer_struct_nested():
    t = ma.infer_type([{"a": [1, 2], "b": True}])
    assert str(t) == "struct<a: list<int64>, b: bool>"


def test_infer_mixed_types_error():
    with pytest.raises(Exception):
        ma.infer_type([1, "foo"])


def test_infer_mixed_bytes_string_error():
    with pytest.raises(Exception):
        ma.array([b"foo", "bar"])


def test_infer_mixed_list_scalar_error():
    with pytest.raises(Exception):
        ma.array([[1, 2], 3])


def test_infer_mixed_struct_scalar_error():
    with pytest.raises(Exception):
        ma.array([{"a": 1}, 2])


# ── array() ─────────────────────────────────────────────────────────────────


def test_array_bool():
    arr = ma.array([True, None, False, None])
    assert type(arr).__name__ == "BoolArray"
    assert len(arr) == 4
    assert arr.null_count() == 2
    assert arr[0]
    assert not arr[2]


def test_array_int64():
    arr = ma.array([1, None, 3, None])
    assert type(arr).__name__ == "Int64Array"
    assert len(arr) == 4
    assert arr.null_count() == 2


def test_array_float64():
    arr = ma.array([1.5, None, 2.5, None, None])
    assert type(arr).__name__ == "Float64Array"
    assert len(arr) == 5
    assert arr.null_count() == 3


def test_array_string():
    arr = ma.array(["foo", "bar", None, "mañana"])
    assert type(arr).__name__ == "StringArray"
    assert len(arr) == 4
    assert arr.null_count() == 1


def test_array_explicit_type_int32():
    arr = ma.array([1, 2, 3], type=ma.int32())
    assert type(arr).__name__ == "Int32Array"


def test_array_explicit_type_float32():
    arr = ma.array([1.0, 2.0], type=ma.float32())
    assert type(arr).__name__ == "Float32Array"


def test_array_explicit_type_bool():
    arr = ma.array([True, False, None], type=ma.bool_())
    assert type(arr).__name__ == "BoolArray"
    assert arr.null_count() == 1


def test_array_mixed_int_float():
    # int coerces up to float64
    arr = ma.array([1, 2.5])
    assert type(arr).__name__ == "Float64Array"


def test_array_bool_int_coercion():
    # bool + int infers int64
    arr = ma.array([True, 1])
    assert type(arr).__name__ == "Int64Array"


def test_array_empty_raises():
    with pytest.raises(Exception):
        ma.array([])


def test_array_all_none_raises():
    with pytest.raises(Exception):
        ma.array([None, None])


def test_array_mixed_types_error():
    with pytest.raises(Exception):
        ma.array([1, "foo"])


def test_array_bytes_raises():
    # binary array construction not yet supported
    with pytest.raises(Exception):
        ma.array([b"foo", b"bar"])


def test_array_nested_list_int():
    arr = ma.array([[1, 2], [3, 4]])
    assert type(arr).__name__ == "ListArray"
    assert len(arr) == 2
    assert arr.null_count() == 0


def test_array_nested_list_with_null_outer():
    arr = ma.array([[1, 2], None, [3]])
    assert type(arr).__name__ == "ListArray"
    assert len(arr) == 3
    assert arr.null_count() == 1


def test_array_nested_list_string():
    arr = ma.array([["foo", "bar"], None, ["baz"]])
    assert type(arr).__name__ == "ListArray"
    assert len(arr) == 3
    assert arr.null_count() == 1


def test_array_struct_basic():
    arr = ma.array([{"a": 5, "b": "foo", "c": True}, {"a": 6, "b": "bar", "c": False}])
    assert type(arr).__name__ == "StructArray"
    assert len(arr) == 2
    assert arr.null_count() == 0


def test_array_struct_null_row():
    arr = ma.array([{"a": 1}, None, {"a": 3}])
    assert type(arr).__name__ == "StructArray"
    assert len(arr) == 3
    assert arr.null_count() == 1


def test_array_struct_missing_key():
    # Missing dict key → null for that field; struct row is valid
    arr = ma.array([{"a": 5, "b": "foo"}, {"a": 6}])
    assert type(arr).__name__ == "StructArray"
    assert len(arr) == 2
    assert arr.null_count() == 0


def test_array_struct_explicit_type():
    ty = ma.struct([ma.field("x", ma.int32()), ma.field("y", ma.float64())])
    arr = ma.array([{"x": 1, "y": 2.5}, {"x": 3, "y": 4.5}], type=ty)
    assert type(arr).__name__ == "StructArray"
    assert len(arr) == 2


# ── indexing ─────────────────────────────────────────────────────────────────


def test_index_int64():
    arr = ma.array([10, 20, 30])
    assert arr[0] == 10
    assert arr[1] == 20
    assert arr[2] == 30


def test_index_float64():
    arr = ma.array([1.5, 2.5, 3.5])
    assert arr[0] == 1.5
    assert arr[2] == 3.5


def test_index_string():
    arr = ma.array(["hello", "world", "mañana"])
    assert arr[0] == "hello"
    assert arr[1] == "world"
    assert arr[2] == "mañana"


def test_index_list():
    arr = ma.array([[1, 2], [3, 4, 5]])
    child0 = arr[0]
    child1 = arr[1]
    assert len(child0) == 2
    assert len(child1) == 3
    assert child0[0] == 1
    assert child0[1] == 2
    assert child1[0] == 3
    assert child1[1] == 4
    assert child1[2] == 5
    assert arr.__len__() == 2


# ── error handling ──────────────────────────────────────────────────────────


def test_array_int_overflow():
    # int8 can only hold -128..127; 200 overflows
    with pytest.raises(Exception):
        ma.array([200], type=ma.int8())


def test_array_wrong_type_in_int_array():
    with pytest.raises(Exception):
        ma.array(["hello"], type=ma.int64())


def test_array_wrong_type_in_float_array():
    with pytest.raises(Exception):
        ma.array(["hello"], type=ma.float64())


def test_array_wrong_type_in_string_array():
    with pytest.raises(Exception):
        ma.array([1, 2, 3], type=ma.string())


def test_array_float_nan():
    # NaN is a valid float value, not an error
    arr = ma.array([float("nan"), 1.0])
    assert type(arr).__name__ == "Float64Array"
    assert arr.__len__() == 2


def test_array_float_inf():
    arr = ma.array([float("inf"), -float("inf"), 1.0])
    assert type(arr).__name__ == "Float64Array"
    assert arr.__len__() == 3


def test_array_struct_non_dict_raises():
    with pytest.raises(Exception):
        ma.array([{"a": 1}, "not_a_dict", {"a": 3}])


def test_array_nested_list_null_inner():
    # None inside inner list — inner list has a null element
    arr = ma.array([[1, None, 2], [3]])
    assert type(arr).__name__ == "ListArray"
    assert arr.__len__() == 2


def test_array_struct_wrong_field_type():
    ty = ma.struct([ma.field("x", ma.int64())])
    with pytest.raises(Exception):
        ma.array([{"x": "not_an_int"}], type=ty)


# ── integer boundary values ──────────────────────────────────────────────────


def test_array_int8_boundaries():
    arr = ma.array([-128, 127, None], type=ma.int8())
    assert arr.__len__() == 3
    assert arr.null_count() == 1


def test_array_int16_boundaries():
    arr = ma.array([-32768, 32767, None], type=ma.int16())
    assert arr.__len__() == 3


def test_array_int32_boundaries():
    arr = ma.array([-2147483648, 2147483647], type=ma.int32())
    assert arr.__len__() == 2


def test_array_uint8_boundaries():
    arr = ma.array([0, 255, None], type=ma.uint8())
    assert arr.__len__() == 3
    assert arr.null_count() == 1


def test_array_uint16_boundaries():
    arr = ma.array([0, 65535], type=ma.uint16())
    assert arr.__len__() == 2


def test_array_uint32_boundaries():
    arr = ma.array([0, 4294967295], type=ma.uint32())
    assert arr.__len__() == 2


def test_array_uint64_valid():
    arr = ma.array([0, 1, 1000], type=ma.uint64())
    assert arr.__len__() == 3


# ── integer overflow / underflow ─────────────────────────────────────────────


def test_array_int8_high_overflow():
    with pytest.raises(Exception):
        ma.array([128], type=ma.int8())


def test_array_int8_low_overflow():
    with pytest.raises(Exception):
        ma.array([-129], type=ma.int8())


def test_array_int16_overflow():
    with pytest.raises(Exception):
        ma.array([32768], type=ma.int16())


def test_array_int32_overflow():
    with pytest.raises(Exception):
        ma.array([2147483648], type=ma.int32())


def test_array_uint8_overflow():
    with pytest.raises(Exception):
        ma.array([256], type=ma.uint8())


def test_array_uint8_underflow():
    with pytest.raises(Exception):
        ma.array([-1], type=ma.uint8())


def test_array_uint16_overflow():
    with pytest.raises(Exception):
        ma.array([65536], type=ma.uint16())


def test_array_uint16_underflow():
    with pytest.raises(Exception):
        ma.array([-1], type=ma.uint16())


def test_array_uint32_underflow():
    with pytest.raises(Exception):
        ma.array([-1], type=ma.uint32())


def test_array_uint64_underflow():
    with pytest.raises(Exception):
        ma.array([-1], type=ma.uint64())


# ── type coercion ────────────────────────────────────────────────────────────


def test_array_bool_as_int():
    # Python bools are ints: True=1, False=0
    arr = ma.array([True, False, None], type=ma.int64())
    assert type(arr).__name__ == "Int64Array"
    assert arr.__len__() == 3
    assert arr.null_count() == 1


def test_array_int_in_float64_explicit():
    arr = ma.array([1, 2, 3], type=ma.float64())
    assert type(arr).__name__ == "Float64Array"


def test_array_int_in_float32_explicit():
    arr = ma.array([1, 2, 3], type=ma.float32())
    assert type(arr).__name__ == "Float32Array"


# ── list type errors ─────────────────────────────────────────────────────────


def test_array_list_scalar_raises():
    # Scalar where a list element is expected should raise
    with pytest.raises(Exception):
        ma.array([1, 2, 3], type=ma.list_(ma.int64()))


