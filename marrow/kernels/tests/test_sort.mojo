"""Correctness tests for the sort kernel."""

from std.testing import assert_true, assert_equal

from ...arrays import (
    DynArray,
    BoolArray,
    DictionaryArray,
    Int32Array,
    PrimitiveArray,
    StringArray,
    StructArray,
)
from ...builders import (
    array,
    BoolBuilder,
    Date32Builder,
    Decimal128Builder,
    Int8Builder,
    Int16Builder,
    Int32Builder,
    Int64Builder,
    LargeStringBuilder,
    UInt8Builder,
    UInt16Builder,
    UInt32Builder,
    UInt64Builder,
    Float16Builder,
    Float32Builder,
    Float64Builder,
    StringBuilder,
    TimestampBuilder,
)
from ...dtypes import (
    PrimitiveType,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    bool_ as _bool_dtype,
    string as _string_dtype,
    date32,
    decimal128,
    microsecond,
    timestamp,
    Field,
)
from ...tabular import record_batch
from ...kernels.sort import sort_indices, sort
from ...kernels.filter import take as _take
from ...kernels.execution import ExecutionContext
from std.utils.numerics import nan, inf, neg_inf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _idx(indices: Int32Array, i: Int) -> Int:
    return Int(indices.unsafe_get(i))


def _make_struct(col0: List[Int], col1: List[Int]) raises -> StructArray:
    var a = Int32Builder(capacity=len(col0))
    var b = Int32Builder(capacity=len(col1))
    for v in col0:
        a.append(Scalar[int32.native](v))
    for v in col1:
        b.append(Scalar[int32.native](v))
    var cols = List[DynArray]()
    cols.append(a.finish().to_dyn())
    cols.append(b.finish().to_dyn())
    return record_batch(cols^, names=["k", "v"]).to_struct_array()


# Verify sort order for every consecutive pair among valid elements, plus null
# placement.  Works for all numeric dtypes, bool, and string.
def _check_order[
    T: PrimitiveType
](arr: PrimitiveArray[T], start: Int, end: Int, ascending: Bool) raises:
    for i in range(start, end - 1):
        var vi = arr.unsafe_get(i)
        var vj = arr.unsafe_get(i + 1)
        if ascending:
            assert_true(
                vi <= vj, "ascending order violated at position " + String(i)
            )
        else:
            assert_true(
                vi >= vj, "descending order violated at position " + String(i)
            )


def _check_order_float[
    T: PrimitiveType
](arr: PrimitiveArray[T], start: Int, end: Int, ascending: Bool) raises:
    for i in range(start, end - 1):
        var vi = arr.unsafe_get(i)
        var vj = arr.unsafe_get(i + 1)
        var vi_nan = vi != vi  # IEEE 754: NaN != NaN is True
        var vj_nan = vj != vj
        if ascending:
            # NaN sorts last — NaN followed by a finite value is wrong.
            assert_true(
                not vi_nan or vj_nan,
                "NaN before finite at position " + String(i),
            )
            if not vi_nan and not vj_nan:
                assert_true(
                    vi <= vj,
                    "ascending order violated at position " + String(i),
                )
        else:
            # NaN sorts first — a finite value followed by NaN is wrong.
            assert_true(
                not vj_nan or vi_nan,
                "finite before NaN at position " + String(i),
            )
            if not vi_nan and not vj_nan:
                assert_true(
                    vi >= vj,
                    "descending order violated at position " + String(i),
                )


def _check_order_bool(
    arr: BoolArray, start: Int, end: Int, ascending: Bool
) raises:
    for i in range(start, end - 1):
        var vi = arr[i].value()
        var vj = arr[i + 1].value()
        if ascending:
            # False < True — True followed by False is wrong.
            assert_true(
                not vi or vj, "bool ascending violated at position " + String(i)
            )
        else:
            assert_true(
                vi or not vj,
                "bool descending violated at position " + String(i),
            )


def _check_order_string(
    arr: StringArray, start: Int, end: Int, ascending: Bool
) raises:
    for i in range(start, end - 1):
        var vi = arr[i].to_string()
        var vj = arr[i + 1].to_string()
        if ascending:
            assert_true(
                vi <= vj, "string ascending violated at position " + String(i)
            )
        else:
            assert_true(
                vi >= vj, "string descending violated at position " + String(i)
            )


def _assert_sorted(
    a: DynArray,
    idx: Int32Array,
    ascending: Bool = True,
    nulls_first: Bool = True,
) raises:
    """Assert idx is a correct sort permutation for a (full order check)."""
    var n = len(a)
    assert_equal(len(idx), n)
    if n <= 1:
        return
    var null_count = a.null_count()
    for i in range(null_count):
        var pos = i if nulls_first else n - null_count + i
        assert_true(
            not a.is_valid(_idx(idx, pos)),
            "expected null at sorted position " + String(pos),
        )
    var vs = null_count if nulls_first else 0
    var ve = n if nulls_first else n - null_count
    if ve - vs <= 1:
        return
    var s = _take(a, idx, ExecutionContext.serial())
    var dt = a.dtype()
    if dt == int8:
        _check_order(s.as_int8(), vs, ve, ascending)
    elif dt == int16:
        _check_order(s.as_int16(), vs, ve, ascending)
    elif dt == int32:
        _check_order(s.as_int32(), vs, ve, ascending)
    elif dt == int64:
        _check_order(s.as_int64(), vs, ve, ascending)
    elif dt == uint8:
        _check_order(s.as_uint8(), vs, ve, ascending)
    elif dt == uint16:
        _check_order(s.as_uint16(), vs, ve, ascending)
    elif dt == uint32:
        _check_order(s.as_uint32(), vs, ve, ascending)
    elif dt == uint64:
        _check_order(s.as_uint64(), vs, ve, ascending)
    elif dt == float16:
        _check_order_float(s.as_float16(), vs, ve, ascending)
    elif dt == float32:
        _check_order_float(s.as_float32(), vs, ve, ascending)
    elif dt == float64:
        _check_order_float(s.as_float64(), vs, ve, ascending)
    elif dt == _bool_dtype:
        _check_order_bool(s.as_bool(), vs, ve, ascending)
    elif dt == _string_dtype:
        _check_order_string(s.as_string(), vs, ve, ascending)
    else:
        raise Error("_assert_sorted: unsupported dtype")


def _assert_values_sorted(
    a: DynArray, ascending: Bool = True, nulls_first: Bool = True
) raises:
    """Assert that the array's values are already in sorted order."""
    var n = len(a)
    var b = Int32Builder(capacity=n)
    for i in range(n):
        b.append(Int32(i))
    _assert_sorted(a, b.finish(), ascending, nulls_first)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_sort_indices_empty() raises:
    var a: DynArray = array(int32)
    var idx = sort_indices(a)
    assert_equal(len(idx), 0)
    _assert_sorted(a, idx)


def test_sort_indices_single_element() raises:
    var a: DynArray = array([42], int32)
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 0)
    _assert_sorted(a, idx)


def test_sort_indices_two_elements_asc() raises:
    var a: DynArray = array([5, 3], int32)
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 0)
    _assert_sorted(a, idx)


def test_sort_indices_all_equal() raises:
    var a: DynArray = array([7, 7, 7], int32)
    var idx = sort_indices(a)
    assert_equal(len(idx), 3)
    var seen: List[Bool] = [False, False, False]
    for i in range(3):
        seen[_idx(idx, i)] = True
    assert_true(seen[0] and seen[1] and seen[2])
    _assert_sorted(a, idx)


# ---------------------------------------------------------------------------
# int32
# ---------------------------------------------------------------------------


def test_sort_indices_int32_ascending() raises:
    var a: DynArray = array([3, 1, 4, 1, 5], int32)
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 2), 0)  # value 3
    assert_equal(_idx(idx, 3), 2)  # value 4
    assert_equal(_idx(idx, 4), 4)  # value 5
    var p0 = _idx(idx, 0)
    var p1 = _idx(idx, 1)
    assert_true((p0 == 1 or p0 == 3) and (p1 == 1 or p1 == 3) and p0 != p1)
    _assert_sorted(a, idx)


def test_sort_indices_int32_descending() raises:
    var a: DynArray = array([3, 1, 4, 1, 5], int32)
    var idx = sort_indices(a, ascending=False)
    assert_equal(_idx(idx, 0), 4)  # 5
    assert_equal(_idx(idx, 1), 2)  # 4
    assert_equal(_idx(idx, 2), 0)  # 3
    _assert_sorted(a, idx, ascending=False)


def test_sort_indices_int32_already_sorted() raises:
    var a: DynArray = array([1, 2, 3, 4, 5], int32)
    var idx = sort_indices(a)
    for i in range(5):
        assert_equal(_idx(idx, i), i)
    _assert_sorted(a, idx)


def test_sort_indices_int32_reverse_sorted() raises:
    var a: DynArray = array([5, 4, 3, 2, 1], int32)
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 4)
    assert_equal(_idx(idx, 1), 3)
    assert_equal(_idx(idx, 2), 2)
    assert_equal(_idx(idx, 3), 1)
    assert_equal(_idx(idx, 4), 0)
    _assert_sorted(a, idx)


# ---------------------------------------------------------------------------
# int64 — sign-bit encoding, PDQsort path, and radix path
# ---------------------------------------------------------------------------


def test_sort_indices_int64_ascending() raises:
    var a: DynArray = array([300, 100, 200], int64)
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 0)
    _assert_sorted(a, idx)


def test_sort_indices_int64_descending() raises:
    var a: DynArray = array([300, 100, 200], int64)
    var idx = sort_indices(a, ascending=False)
    assert_equal(_idx(idx, 0), 0)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 1)
    _assert_sorted(a, idx, ascending=False)


def test_sort_indices_int64_negative() raises:
    # INT64_MIN and INT64_MAX must be encoded correctly (sign-bit XOR).
    var b = Int64Builder(capacity=4)
    b.append(Int64(0))
    b.append(Int64(-9223372036854775808))  # INT64_MIN
    b.append(Int64(9223372036854775807))  # INT64_MAX
    b.append(Int64(-1))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 1)  # INT64_MIN
    assert_equal(_idx(idx, 1), 3)  # -1
    assert_equal(_idx(idx, 2), 0)  # 0
    assert_equal(_idx(idx, 3), 2)  # INT64_MAX
    _assert_sorted(a, idx)


def test_sort_indices_int64_radix() raises:
    # N > _RADIX_THRESHOLD = 32768 — exercises the LSD radix path.
    comptime N = 40_000
    var b = Int64Builder(capacity=N)
    for i in range(N):
        b.append(Int64(N - 1 - i))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(len(idx), N)
    _assert_sorted(a, idx)


# ---------------------------------------------------------------------------
# Signed integer edge cases
# ---------------------------------------------------------------------------


def test_sort_indices_int8_negative() raises:
    var b = Int8Builder(capacity=4)
    b.append(Int8(-128))
    b.append(Int8(-1))
    b.append(Int8(0))
    b.append(Int8(127))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 0)
    assert_equal(_idx(idx, 1), 1)
    assert_equal(_idx(idx, 2), 2)
    assert_equal(_idx(idx, 3), 3)
    _assert_sorted(a, idx)


def test_sort_indices_int16_mixed() raises:
    var a: DynArray = array([-100, 0, -1, 100], int16)
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 0)  # -100
    assert_equal(_idx(idx, 1), 2)  # -1
    assert_equal(_idx(idx, 2), 1)  # 0
    assert_equal(_idx(idx, 3), 3)  # 100
    _assert_sorted(a, idx)


# ---------------------------------------------------------------------------
# Unsigned integer types
# ---------------------------------------------------------------------------


def test_sort_indices_uint8() raises:
    var a: DynArray = array([200, 100, 50, 150], uint8)
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 2)
    assert_equal(_idx(idx, 1), 1)
    assert_equal(_idx(idx, 2), 3)
    assert_equal(_idx(idx, 3), 0)
    _assert_sorted(a, idx)


def test_sort_indices_uint16() raises:
    var a: DynArray = array([65535, 0, 1000, 255], uint16)
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 1)  # 0
    assert_equal(_idx(idx, 1), 3)  # 255
    assert_equal(_idx(idx, 2), 2)  # 1000
    assert_equal(_idx(idx, 3), 0)  # 65535
    _assert_sorted(a, idx)


def test_sort_indices_uint32() raises:
    var a: DynArray = array([3, 1, 2], uint32)
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 0)
    _assert_sorted(a, idx)


def test_sort_indices_uint64() raises:
    var b = UInt64Builder(capacity=3)
    b.append(UInt64(18446744073709551615))  # UINT64_MAX
    b.append(UInt64(0))
    b.append(UInt64(1000))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 1)  # 0
    assert_equal(_idx(idx, 1), 2)  # 1000
    assert_equal(_idx(idx, 2), 0)  # UINT64_MAX
    _assert_sorted(a, idx)


# ---------------------------------------------------------------------------
# Float types — NaN, infinity, sign-bit encoding
# ---------------------------------------------------------------------------


def test_sort_indices_float16() raises:
    var b = Float16Builder(capacity=3)
    b.append(Float16(3.0))
    b.append(Float16(1.0))
    b.append(Float16(-2.0))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 2)  # -2.0
    assert_equal(_idx(idx, 1), 1)  # 1.0
    assert_equal(_idx(idx, 2), 0)  # 3.0
    _assert_sorted(a, idx)


def test_sort_indices_float32_ascending() raises:
    var a: DynArray = array([3.0, 1.0, 2.0], float32)
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 0)
    _assert_sorted(a, idx)


def test_sort_indices_float32_nan_ascending() raises:
    # NaN sorts last in ascending order.
    var b = Float32Builder(capacity=4)
    b.append(Float32(1.0))
    b.append(Float32(3.0e38))
    b.append(nan[float32.native]())
    b.append(Float32(-1.0))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 3)  # -1.0
    assert_equal(_idx(idx, 1), 0)  # 1.0
    assert_equal(_idx(idx, 2), 1)  # 3e38
    assert_equal(_idx(idx, 3), 2)  # NaN last
    _assert_sorted(a, idx)


def test_sort_indices_float32_nan_descending() raises:
    # NaN sorts first in descending order (complement of uint_max = 0).
    var b = Float32Builder(capacity=4)
    b.append(Float32(1.0))
    b.append(Float32(3.0e38))
    b.append(nan[float32.native]())
    b.append(Float32(-1.0))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, ascending=False)
    assert_equal(_idx(idx, 0), 2)  # NaN first
    assert_equal(_idx(idx, 1), 1)  # 3e38
    assert_equal(_idx(idx, 2), 0)  # 1.0
    assert_equal(_idx(idx, 3), 3)  # -1.0
    _assert_sorted(a, idx, ascending=False)


def test_sort_indices_float64_inf() raises:
    var b = Float64Builder(capacity=4)
    b.append(Float64(0.0))
    b.append(neg_inf[float64.native]())
    b.append(Float64(1.0))
    b.append(inf[float64.native]())
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 1)  # -inf
    assert_equal(_idx(idx, 1), 0)  # 0.0
    assert_equal(_idx(idx, 2), 2)  # 1.0
    assert_equal(_idx(idx, 3), 3)  # +inf
    _assert_sorted(a, idx)


def test_sort_indices_float64_nan() raises:
    var b = Float64Builder(capacity=3)
    b.append(Float64(1.0))
    b.append(nan[float64.native]())
    b.append(Float64(-1.0))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 2)  # -1.0
    assert_equal(_idx(idx, 1), 0)  # 1.0
    assert_equal(_idx(idx, 2), 1)  # NaN last
    _assert_sorted(a, idx)


def test_sort_indices_float64_negative() raises:
    # Negative floats must sort correctly (sign-bit encoding for all-bits XOR).
    var b = Float64Builder(capacity=4)
    b.append(Float64(-1.5))
    b.append(Float64(-0.5))
    b.append(Float64(0.5))
    b.append(Float64(1.5))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 0)
    assert_equal(_idx(idx, 1), 1)
    assert_equal(_idx(idx, 2), 2)
    assert_equal(_idx(idx, 3), 3)
    _assert_sorted(a, idx)


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------


def test_sort_indices_nulls_first() raises:
    var b = Int32Builder(capacity=5)
    b.append(Int32(3))
    b.append_null()
    b.append(Int32(1))
    b.append_null()
    b.append(Int32(5))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, nulls_first=True)
    var i0 = _idx(idx, 0)
    var i1 = _idx(idx, 1)
    assert_true((i0 == 1 or i0 == 3) and (i1 == 1 or i1 == 3) and i0 != i1)
    assert_equal(_idx(idx, 2), 2)  # 1
    assert_equal(_idx(idx, 3), 0)  # 3
    assert_equal(_idx(idx, 4), 4)  # 5
    _assert_sorted(a, idx, nulls_first=True)


def test_sort_indices_nulls_last() raises:
    var b = Int32Builder(capacity=5)
    b.append(Int32(3))
    b.append_null()
    b.append(Int32(1))
    b.append_null()
    b.append(Int32(5))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, nulls_first=False)
    assert_equal(_idx(idx, 0), 2)  # 1
    assert_equal(_idx(idx, 1), 0)  # 3
    assert_equal(_idx(idx, 2), 4)  # 5
    var i3 = _idx(idx, 3)
    var i4 = _idx(idx, 4)
    assert_true((i3 == 1 or i3 == 3) and (i4 == 1 or i4 == 3) and i3 != i4)
    _assert_sorted(a, idx, nulls_first=False)


def test_sort_indices_all_null() raises:
    var b = Int32Builder(capacity=3)
    b.append_null()
    b.append_null()
    b.append_null()
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(len(idx), 3)
    _assert_sorted(a, idx)


def test_sort_indices_float64_null() raises:
    var b = Float64Builder(capacity=4)
    b.append(Float64(2.0))
    b.append_null()
    b.append(Float64(1.0))
    b.append_null()
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, nulls_first=False)
    assert_equal(_idx(idx, 0), 2)  # 1.0
    assert_equal(_idx(idx, 1), 0)  # 2.0
    var i2 = _idx(idx, 2)
    var i3 = _idx(idx, 3)
    assert_true((i2 == 1 or i2 == 3) and (i3 == 1 or i3 == 3) and i2 != i3)
    _assert_sorted(a, idx, nulls_first=False)


# ---------------------------------------------------------------------------
# BoolArray — counting sort
# ---------------------------------------------------------------------------


def test_sort_indices_bool_ascending() raises:
    var b = BoolBuilder(capacity=4)
    b.append(True)
    b.append(False)
    b.append(True)
    b.append(False)
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 3)
    assert_equal(_idx(idx, 2), 0)
    assert_equal(_idx(idx, 3), 2)
    _assert_sorted(a, idx)


def test_sort_indices_bool_descending() raises:
    var b = BoolBuilder(capacity=4)
    b.append(True)
    b.append(False)
    b.append(True)
    b.append(False)
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, ascending=False)
    assert_equal(_idx(idx, 0), 0)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 1)
    assert_equal(_idx(idx, 3), 3)
    _assert_sorted(a, idx, ascending=False)


def test_sort_indices_bool_nulls_first() raises:
    var b = BoolBuilder(capacity=4)
    b.append(True)
    b.append_null()
    b.append(False)
    b.append(True)
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, nulls_first=True)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 0)
    assert_equal(_idx(idx, 3), 3)
    _assert_sorted(a, idx, nulls_first=True)


def test_sort_indices_bool_nulls_last() raises:
    var b = BoolBuilder(capacity=4)
    b.append(True)
    b.append_null()
    b.append(False)
    b.append(True)
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, nulls_first=False)
    assert_equal(_idx(idx, 0), 2)  # False
    assert_equal(_idx(idx, 1), 0)  # True
    assert_equal(_idx(idx, 2), 3)  # True
    assert_equal(_idx(idx, 3), 1)  # null
    _assert_sorted(a, idx, nulls_first=False)


# ---------------------------------------------------------------------------
# StringArray
# ---------------------------------------------------------------------------


def test_sort_indices_string_ascending() raises:
    var b = StringBuilder(capacity=4)
    b.append("banana")
    b.append("apple")
    b.append("cherry")
    b.append("apricot")
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 3)
    assert_equal(_idx(idx, 2), 0)
    assert_equal(_idx(idx, 3), 2)
    _assert_sorted(a, idx)


def test_sort_indices_string_descending() raises:
    var b = StringBuilder(capacity=3)
    b.append("a")
    b.append("c")
    b.append("b")
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, ascending=False)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 0)
    _assert_sorted(a, idx, ascending=False)


def test_sort_indices_string_nulls_first() raises:
    var b = StringBuilder(capacity=4)
    b.append("b")
    b.append_null()
    b.append("a")
    b.append_null()
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, nulls_first=True)
    var i0 = _idx(idx, 0)
    var i1 = _idx(idx, 1)
    assert_true((i0 == 1 or i0 == 3) and (i1 == 1 or i1 == 3) and i0 != i1)
    assert_equal(_idx(idx, 2), 2)  # "a"
    assert_equal(_idx(idx, 3), 0)  # "b"
    _assert_sorted(a, idx, nulls_first=True)


def test_sort_indices_string_nulls_last() raises:
    var b = StringBuilder(capacity=4)
    b.append("b")
    b.append_null()
    b.append("a")
    b.append_null()
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, nulls_first=False)
    assert_equal(_idx(idx, 0), 2)  # "a"
    assert_equal(_idx(idx, 1), 0)  # "b"
    var i2 = _idx(idx, 2)
    var i3 = _idx(idx, 3)
    assert_true((i2 == 1 or i2 == 3) and (i3 == 1 or i3 == 3) and i2 != i3)
    _assert_sorted(a, idx, nulls_first=False)


# ---------------------------------------------------------------------------
# Stable sort — equal elements preserve original relative order
# ---------------------------------------------------------------------------


def test_sort_indices_stable_int32() raises:
    # [2, 1, 2, 1, 3] — stable asc: 1@1, 1@3, 2@0, 2@2, 3@4
    var b = Int32Builder(capacity=5)
    b.append(Int32(2))
    b.append(Int32(1))
    b.append(Int32(2))
    b.append(Int32(1))
    b.append(Int32(3))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, stable=True)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 3)
    assert_equal(_idx(idx, 2), 0)
    assert_equal(_idx(idx, 3), 2)
    assert_equal(_idx(idx, 4), 4)
    _assert_sorted(a, idx)


def test_sort_indices_stable_string() raises:
    # ["b", "a", "b", "a"] — stable asc: a@1, a@3, b@0, b@2
    var b = StringBuilder(capacity=4)
    b.append("b")
    b.append("a")
    b.append("b")
    b.append("a")
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, stable=True)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 3)
    assert_equal(_idx(idx, 2), 0)
    assert_equal(_idx(idx, 3), 2)
    _assert_sorted(a, idx)


# ---------------------------------------------------------------------------
# limit (top-K)
# ---------------------------------------------------------------------------


def test_sort_indices_limit() raises:
    var a: DynArray = array([5, 3, 1, 4, 2], int32)
    var idx = sort_indices(a, limit=3)
    assert_equal(len(idx), 3)
    assert_equal(_idx(idx, 0), 2)  # 1
    assert_equal(_idx(idx, 1), 4)  # 2
    assert_equal(_idx(idx, 2), 1)  # 3
    # Verify the 3 selected values are in sorted order (indices refer into full a).
    var taken = _take(a, idx, ExecutionContext.serial())
    _assert_values_sorted(taken)


def test_sort_indices_limit_exceeds_length() raises:
    var a: DynArray = array([2, 1], int32)
    var idx = sort_indices(a, limit=100)
    assert_equal(len(idx), 2)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 0)
    _assert_sorted(a, idx)


# ---------------------------------------------------------------------------
# Large arrays — PDQsort path (N < 32768) and radix path (N >= 32768)
# ---------------------------------------------------------------------------


def test_sort_indices_int32_large_pdqsort() raises:
    comptime N = 10_000
    var b = Int32Builder(capacity=N)
    for i in range(N // 2):
        b.append(Int32(N - 1 - i))
        b.append(Int32(i))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(len(idx), N)
    _assert_sorted(a, idx)


def test_sort_indices_int64_large_radix() raises:
    comptime N = 40_000
    var b = Int64Builder(capacity=N)
    for i in range(N):
        b.append(Int64(N - 1 - i))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a)
    assert_equal(len(idx), N)
    _assert_sorted(a, idx)


# ---------------------------------------------------------------------------
# sort(StructArray) — key ordering and value-column integrity
# ---------------------------------------------------------------------------


def test_sort_struct_ascending() raises:
    var sa = _make_struct([3, 1, 4, 1, 5], [30, 10, 40, 10, 50])
    var result = sort(sa, [0], [True])
    var k = result.field(0)
    assert_equal(k[0].as_int32().value(), Int32(1))
    assert_equal(k[1].as_int32().value(), Int32(1))
    assert_equal(k[2].as_int32().value(), Int32(3))
    assert_equal(k[3].as_int32().value(), Int32(4))
    assert_equal(k[4].as_int32().value(), Int32(5))
    _assert_values_sorted(k)


def test_sort_struct_descending() raises:
    var sa = _make_struct([3, 1, 4], [30, 10, 40])
    var result = sort(sa, [0], [False])
    var k = result.field(0)
    assert_equal(k[0].as_int32().value(), Int32(4))
    assert_equal(k[1].as_int32().value(), Int32(3))
    assert_equal(k[2].as_int32().value(), Int32(1))
    _assert_values_sorted(k, ascending=False)


def test_sort_struct_value_integrity() raises:
    # The non-key column must be permuted consistently with the key column.
    var sa = _make_struct([30, 10, 20], [300, 100, 200])
    var result = sort(sa, [0], [True])
    var k = result.field(0)
    var v = result.field(1)
    assert_equal(k[0].as_int32().value(), Int32(10))
    assert_equal(v[0].as_int32().value(), Int32(100))
    assert_equal(k[1].as_int32().value(), Int32(20))
    assert_equal(v[1].as_int32().value(), Int32(200))
    assert_equal(k[2].as_int32().value(), Int32(30))
    assert_equal(v[2].as_int32().value(), Int32(300))
    _assert_values_sorted(k)
    _assert_values_sorted(v)


def test_sort_struct_multi_key_asc_asc() raises:
    # ORDER BY k ASC, v ASC — the secondary key breaks ties within equal k
    # (column-oriented LSD sort). Expect (1,20),(1,40),(2,10),(2,20),(2,30).
    var sa = _make_struct([2, 1, 2, 1, 2], [30, 20, 10, 40, 20])
    var result = sort(sa, [0, 1], [True, True])
    var k = result.field(0)
    var v = result.field(1)
    assert_equal(k[0].as_int32().value(), Int32(1))
    assert_equal(v[0].as_int32().value(), Int32(20))
    assert_equal(k[1].as_int32().value(), Int32(1))
    assert_equal(v[1].as_int32().value(), Int32(40))
    assert_equal(k[2].as_int32().value(), Int32(2))
    assert_equal(v[2].as_int32().value(), Int32(10))
    assert_equal(k[3].as_int32().value(), Int32(2))
    assert_equal(v[3].as_int32().value(), Int32(20))
    assert_equal(k[4].as_int32().value(), Int32(2))
    assert_equal(v[4].as_int32().value(), Int32(30))


def test_sort_struct_multi_key_asc_desc() raises:
    # ORDER BY k ASC, v DESC — mixed per-key direction.
    # Expect (1,40),(1,20),(2,30),(2,20),(2,10).
    var sa = _make_struct([2, 1, 2, 1, 2], [30, 20, 10, 40, 20])
    var result = sort(sa, [0, 1], [True, False])
    var k = result.field(0)
    var v = result.field(1)
    assert_equal(k[0].as_int32().value(), Int32(1))
    assert_equal(v[0].as_int32().value(), Int32(40))
    assert_equal(k[1].as_int32().value(), Int32(1))
    assert_equal(v[1].as_int32().value(), Int32(20))
    assert_equal(k[2].as_int32().value(), Int32(2))
    assert_equal(v[2].as_int32().value(), Int32(30))
    assert_equal(k[3].as_int32().value(), Int32(2))
    assert_equal(v[3].as_int32().value(), Int32(20))
    assert_equal(k[4].as_int32().value(), Int32(2))
    assert_equal(v[4].as_int32().value(), Int32(10))


# ---------------------------------------------------------------------------
# ExecutionContext — use N > _PARALLEL_THRESHOLD (524_288) to exercise the
# parallel radix path; verify endpoints and midpoint of a reverse-sorted array.
# ---------------------------------------------------------------------------


def test_sort_indices_serial_context() raises:
    # N > _RADIX_THRESHOLD but < _PARALLEL_THRESHOLD — serial radix path.
    comptime N = 100_000
    var b = Int64Builder(capacity=N)
    for i in range(N):
        b.append(Int64(N - 1 - i))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, ctx=ExecutionContext.serial())
    assert_equal(len(idx), N)
    _assert_sorted(a, idx)


def test_sort_indices_parallel_context() raises:
    # N > _PARALLEL_THRESHOLD (524_288) — exercises the parallel radix path.
    comptime N = 600_000
    var b = Int64Builder(capacity=N)
    for i in range(N):
        b.append(Int64(N - 1 - i))
    var a: DynArray = b.finish().to_dyn()
    var idx = sort_indices(a, ctx=ExecutionContext.parallel())
    assert_equal(len(idx), N)
    _assert_sorted(a, idx)


# ---------------------------------------------------------------------------
# Temporal / large_string / decimal / dictionary
#
# These dtypes used to fall off the end of the dispatch ladder and raise, which
# broke `ORDER BY` on any timestamp column (docs/code-quality-review.md D5).
# The permutation is asserted directly — `_assert_sorted` only knows the
# numeric/bool/string dtypes.
# ---------------------------------------------------------------------------


def _assert_perm(idx: Int32Array, expected: List[Int]) raises:
    assert_equal(len(idx), len(expected))
    for i in range(len(expected)):
        assert_equal(_idx(idx, i), expected[i])


def test_sort_indices_date32() raises:
    var b = Date32Builder(date32(), 4)
    for d in [19000, 18500, 19100, 18800]:
        b.append(Scalar[int32.native](d))
    var a: DynArray = b.finish()
    _assert_perm(sort_indices(a), [1, 3, 0, 2])
    _assert_perm(sort_indices(a, ascending=False), [2, 0, 3, 1])


def test_sort_indices_timestamp_negative() raises:
    """Pre-epoch timestamps are negative int64 — the signed sign-flip in the key
    encoding must keep them below the positive ones."""
    var b = TimestampBuilder(timestamp(microsecond, "UTC"), 4)
    for t in [1_000, -5_000, 0, -1]:
        b.append(Scalar[int64.native](t))
    var a: DynArray = b.finish()
    _assert_perm(sort_indices(a), [1, 3, 2, 0])


def test_sort_indices_timestamp_nulls() raises:
    var b = TimestampBuilder(timestamp(microsecond), 4)
    b.append(Scalar[int64.native](30))
    b.append_null()
    b.append(Scalar[int64.native](10))
    b.append(Scalar[int64.native](20))
    var a: DynArray = b.finish()
    _assert_perm(sort_indices(a, nulls_first=False), [2, 3, 0, 1])
    _assert_perm(sort_indices(a, nulls_first=True), [1, 2, 3, 0])


def test_sort_indices_large_string() raises:
    var b = LargeStringBuilder(4)
    for s in ["pear", "apple", "fig", "banana"]:
        b.append(s)
    var a: DynArray = b.finish()
    _assert_perm(sort_indices(a), [1, 3, 2, 0])
    _assert_perm(sort_indices(a, ascending=False), [0, 2, 3, 1])


def test_sort_indices_decimal128() raises:
    """`decimal128` has no UInt64 radix key, so it takes the comparison path —
    including values that differ only above bit 63."""
    var b = Decimal128Builder(decimal128(38, 0), 3)
    b.append(Scalar[DType.int128](1) << Scalar[DType.int128](70))
    b.append(Scalar[DType.int128](-1))
    b.append(Scalar[DType.int128](5))
    var a: DynArray = b.finish()
    _assert_perm(sort_indices(a), [1, 2, 0])


def test_sort_indices_dictionary() raises:
    """Ordering follows the decoded values, not the dictionary index order."""
    var values = StringBuilder(3)
    values.append("pear")  # index 0
    values.append("apple")  # index 1
    values.append("fig")  # index 2
    var ib = Int32Builder(3)
    for i in [0, 1, 2]:
        ib.append(Int32(i))
    var a: DynArray = DictionaryArray.from_arrays(ib.finish(), values.finish())
    _assert_perm(sort_indices(a), [1, 2, 0])


def test_sort_struct_timestamp_key() raises:
    """ORDER BY <timestamp> through the StructArray path — the sorted column
    keeps its logical dtype (unit + timezone)."""
    var ts = TimestampBuilder(timestamp(microsecond, "UTC"), 3)
    for t in [300, 100, 200]:
        ts.append(Scalar[int64.native](t))
    var v = Int32Builder(3)
    for x in [3, 1, 2]:
        v.append(Int32(x))
    var cols = List[DynArray]()
    cols.append(ts.finish().to_dyn())
    cols.append(v.finish().to_dyn())
    var sa = record_batch(cols^, names=["t", "v"]).to_struct_array()

    var result = sort(sa, [0], [True])
    ref k = result.field(0).as_timestamp()
    assert_equal(k[0].value(), 100)
    assert_equal(k[1].value(), 200)
    assert_equal(k[2].value(), 300)
    assert_true(
        result.field(0).dtype() == timestamp(microsecond, "UTC").to_dyn()
    )
    ref vals = result.field(1).as_int32()
    assert_equal(vals[0].value(), 1)
    assert_equal(vals[2].value(), 3)
