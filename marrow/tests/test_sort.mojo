"""Correctness tests for the sort kernel."""

from std.testing import assert_true, assert_equal
from marrow.testing import TestSuite

from marrow.arrays import (
    AnyArray,
    BoolArray,
    PrimitiveArray,
    StringArray,
    StructArray,
    Int32Array,
)
from marrow.builders import (
    array,
    BoolBuilder,
    Int8Builder,
    Int16Builder,
    Int32Builder,
    Int64Builder,
    UInt8Builder,
    Float32Builder,
    Float64Builder,
    StringBuilder,
)
from marrow.dtypes import (
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
    string,
    Field,
)
from marrow.tabular import record_batch
from marrow.kernels.sort import argsort, sort
from marrow.kernels.execution import ExecutionContext
from std.utils.numerics import nan, inf, neg_inf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _idx(indices: Int32Array, i: Int) -> Int:
    return Int(indices.unsafe_get(i))


def _make_struct(
    col0: List[Int], col1: List[Int]
) raises -> StructArray:
    var a = Int32Builder(capacity=len(col0))
    var b = Int32Builder(capacity=len(col1))
    for v in col0:
        a.append(Scalar[int32.native](v))
    for v in col1:
        b.append(Scalar[int32.native](v))
    var cols = List[AnyArray]()
    cols.append(a.finish().to_any())
    cols.append(b.finish().to_any())
    return record_batch(cols^, names=["k", "v"]).to_struct_array()


# ---------------------------------------------------------------------------
# Empty and single-element edge cases
# ---------------------------------------------------------------------------


def test_argsort_empty() raises:
    var a: AnyArray = array(int32)
    var idx = argsort(a)
    assert_equal(len(idx), 0)


def test_argsort_single_element() raises:
    var a: AnyArray = array([42], int32)
    var idx = argsort(a)
    assert_equal(len(idx), 1)
    assert_equal(_idx(idx, 0), 0)


# ---------------------------------------------------------------------------
# int32 — ascending, descending
# ---------------------------------------------------------------------------


def test_argsort_int32_ascending() raises:
    var a: AnyArray = array([3, 1, 4, 1, 5], int32)
    var idx = argsort(a)
    assert_equal(len(idx), 5)
    # sorted: 1,1,3,4,5 → indices 1,3,0,2,4 (ties in original order for radix)
    assert_equal(_idx(idx, 2), 0)  # value 3
    assert_equal(_idx(idx, 3), 2)  # value 4
    assert_equal(_idx(idx, 4), 4)  # value 5
    # Both 1s must appear in positions 0 and 1
    var p0 = _idx(idx, 0)
    var p1 = _idx(idx, 1)
    assert_true(p0 == 1 or p0 == 3)
    assert_true(p1 == 1 or p1 == 3)
    assert_true(p0 != p1)


def test_argsort_int32_descending() raises:
    var a: AnyArray = array([3, 1, 4, 1, 5], int32)
    var idx = argsort(a, ascending=False)
    assert_equal(len(idx), 5)
    assert_equal(_idx(idx, 0), 4)  # value 5
    assert_equal(_idx(idx, 1), 2)  # value 4
    assert_equal(_idx(idx, 2), 0)  # value 3


def test_argsort_int32_already_sorted() raises:
    var a: AnyArray = array([1, 2, 3, 4, 5], int32)
    var idx = argsort(a)
    assert_equal(_idx(idx, 0), 0)
    assert_equal(_idx(idx, 1), 1)
    assert_equal(_idx(idx, 2), 2)
    assert_equal(_idx(idx, 3), 3)
    assert_equal(_idx(idx, 4), 4)


def test_argsort_int32_reverse_sorted() raises:
    var a: AnyArray = array([5, 4, 3, 2, 1], int32)
    var idx = argsort(a)
    assert_equal(_idx(idx, 0), 4)
    assert_equal(_idx(idx, 1), 3)
    assert_equal(_idx(idx, 2), 2)
    assert_equal(_idx(idx, 3), 1)
    assert_equal(_idx(idx, 4), 0)


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------


def test_argsort_nulls_first() raises:
    var b = Int32Builder(capacity=5)
    b.append(Scalar[int32.native](3))
    b.append_null()
    b.append(Scalar[int32.native](1))
    b.append_null()
    b.append(Scalar[int32.native](5))
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a, nulls_first=True)
    assert_equal(len(idx), 5)
    # First two: null indices (1 and 3)
    var i0 = _idx(idx, 0)
    var i1 = _idx(idx, 1)
    assert_true(i0 == 1 or i0 == 3)
    assert_true(i1 == 1 or i1 == 3)
    assert_true(i0 != i1)
    # Then sorted values
    assert_equal(_idx(idx, 2), 2)  # value 1
    assert_equal(_idx(idx, 3), 0)  # value 3
    assert_equal(_idx(idx, 4), 4)  # value 5


def test_argsort_nulls_last() raises:
    var b = Int32Builder(capacity=5)
    b.append(Scalar[int32.native](3))
    b.append_null()
    b.append(Scalar[int32.native](1))
    b.append_null()
    b.append(Scalar[int32.native](5))
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a, nulls_first=False)
    assert_equal(_idx(idx, 0), 2)  # value 1
    assert_equal(_idx(idx, 1), 0)  # value 3
    assert_equal(_idx(idx, 2), 4)  # value 5
    var i3 = _idx(idx, 3)
    var i4 = _idx(idx, 4)
    assert_true(i3 == 1 or i3 == 3)
    assert_true(i4 == 1 or i4 == 3)
    assert_true(i3 != i4)


def test_argsort_all_null() raises:
    var b = Int32Builder(capacity=3)
    b.append_null()
    b.append_null()
    b.append_null()
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a)
    assert_equal(len(idx), 3)


def test_argsort_no_nulls() raises:
    var a: AnyArray = array([10, 20, 30], int32)
    var idx = argsort(a)
    assert_equal(len(idx), 3)
    assert_equal(_idx(idx, 0), 0)
    assert_equal(_idx(idx, 1), 1)
    assert_equal(_idx(idx, 2), 2)


# ---------------------------------------------------------------------------
# float32 — NaN and infinity
# ---------------------------------------------------------------------------


def test_argsort_float32_nan_ascending() raises:
    var b = Float32Builder(capacity=4)
    b.append(Scalar[float32.native](1.0))
    b.append(Scalar[float32.native](3.0e38))
    b.append(nan[float32.native]())
    b.append(Scalar[float32.native](-1.0))
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a)
    assert_equal(len(idx), 4)
    assert_equal(_idx(idx, 0), 3)  # -1.0
    assert_equal(_idx(idx, 1), 0)  # 1.0
    assert_equal(_idx(idx, 2), 1)  # MAX_FINITE
    assert_equal(_idx(idx, 3), 2)  # NaN last


def test_argsort_float64_inf() raises:
    var b = Float64Builder(capacity=4)
    b.append(Scalar[float64.native](0.0))
    b.append(neg_inf[float64.native]())
    b.append(Scalar[float64.native](1.0))
    b.append(inf[float64.native]())
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a)
    assert_equal(_idx(idx, 0), 1)   # -inf
    assert_equal(_idx(idx, 1), 0)   # 0.0
    assert_equal(_idx(idx, 2), 2)   # 1.0
    assert_equal(_idx(idx, 3), 3)   # +inf


# ---------------------------------------------------------------------------
# Signed integer edge cases (sign bit encoding)
# ---------------------------------------------------------------------------


def test_argsort_int8_negative() raises:
    var b = Int8Builder(capacity=4)
    b.append(Scalar[int8.native](-128))
    b.append(Scalar[int8.native](-1))
    b.append(Scalar[int8.native](0))
    b.append(Scalar[int8.native](127))
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a)
    assert_equal(_idx(idx, 0), 0)
    assert_equal(_idx(idx, 1), 1)
    assert_equal(_idx(idx, 2), 2)
    assert_equal(_idx(idx, 3), 3)


def test_argsort_int16_mixed() raises:
    var b = Int16Builder(capacity=4)
    b.append(Scalar[int16.native](-100))
    b.append(Scalar[int16.native](0))
    b.append(Scalar[int16.native](-1))
    b.append(Scalar[int16.native](100))
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a)
    assert_equal(_idx(idx, 0), 0)  # -100
    assert_equal(_idx(idx, 1), 2)  # -1
    assert_equal(_idx(idx, 2), 1)  # 0
    assert_equal(_idx(idx, 3), 3)  # 100


# ---------------------------------------------------------------------------
# BoolArray — counting sort
# ---------------------------------------------------------------------------


def test_argsort_bool_ascending() raises:
    var b = BoolBuilder(capacity=4)
    b.append(True)
    b.append(False)
    b.append(True)
    b.append(False)
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a)
    # false first (1,3), then true (0,2)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 3)
    assert_equal(_idx(idx, 2), 0)
    assert_equal(_idx(idx, 3), 2)


def test_argsort_bool_descending() raises:
    var b = BoolBuilder(capacity=4)
    b.append(True)
    b.append(False)
    b.append(True)
    b.append(False)
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a, ascending=False)
    # true first (0,2), then false (1,3)
    assert_equal(_idx(idx, 0), 0)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 1)
    assert_equal(_idx(idx, 3), 3)


def test_argsort_bool_with_nulls() raises:
    var b = BoolBuilder(capacity=4)
    b.append(True)
    b.append_null()
    b.append(False)
    b.append(True)
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a, nulls_first=True)
    assert_equal(_idx(idx, 0), 1)  # null
    assert_equal(_idx(idx, 1), 2)  # false
    assert_equal(_idx(idx, 2), 0)  # true
    assert_equal(_idx(idx, 3), 3)  # true


# ---------------------------------------------------------------------------
# StringArray
# ---------------------------------------------------------------------------


def test_argsort_string_ascending() raises:
    var b = StringBuilder(capacity=4)
    b.append("banana")
    b.append("apple")
    b.append("cherry")
    b.append("apricot")
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a)
    # apple(1), apricot(3), banana(0), cherry(2)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 3)
    assert_equal(_idx(idx, 2), 0)
    assert_equal(_idx(idx, 3), 2)


def test_argsort_string_descending() raises:
    var b = StringBuilder(capacity=3)
    b.append("a")
    b.append("c")
    b.append("b")
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a, ascending=False)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 0)


def test_argsort_string_with_nulls() raises:
    var b = StringBuilder(capacity=4)
    b.append("b")
    b.append_null()
    b.append("a")
    b.append_null()
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a, nulls_first=True)
    var i0 = _idx(idx, 0)
    var i1 = _idx(idx, 1)
    assert_true(i0 == 1 or i0 == 3)
    assert_true(i1 == 1 or i1 == 3)
    assert_true(i0 != i1)
    assert_equal(_idx(idx, 2), 2)  # "a"
    assert_equal(_idx(idx, 3), 0)  # "b"


# ---------------------------------------------------------------------------
# limit (top-K)
# ---------------------------------------------------------------------------


def test_argsort_limit() raises:
    var a: AnyArray = array([5, 3, 1, 4, 2], int32)
    var idx = argsort(a, limit=3)
    assert_equal(len(idx), 3)
    assert_equal(_idx(idx, 0), 2)  # value 1
    assert_equal(_idx(idx, 1), 4)  # value 2
    assert_equal(_idx(idx, 2), 1)  # value 3


# ---------------------------------------------------------------------------
# sort(StructArray) — single key column
# ---------------------------------------------------------------------------


def test_sort_struct_single_key() raises:
    var keys = List[Int]()
    keys.append(0)
    var asc = List[Bool]()
    asc.append(True)
    var col0 = List[Int]()
    col0.append(3)
    col0.append(1)
    col0.append(4)
    col0.append(1)
    col0.append(5)
    var col1 = List[Int]()
    col1.append(10)
    col1.append(20)
    col1.append(30)
    col1.append(40)
    col1.append(50)
    var sa = _make_struct(col0, col1)
    var result = sort(sa, keys, asc)
    var k = result.field(0)
    assert_equal(k[0].as_int32().value(), Scalar[int32.native](1))
    assert_equal(k[1].as_int32().value(), Scalar[int32.native](1))
    assert_equal(k[2].as_int32().value(), Scalar[int32.native](3))
    assert_equal(k[3].as_int32().value(), Scalar[int32.native](4))
    assert_equal(k[4].as_int32().value(), Scalar[int32.native](5))


def test_sort_struct_descending() raises:
    var keys = List[Int]()
    keys.append(0)
    var asc = List[Bool]()
    asc.append(False)
    var col0 = List[Int]()
    col0.append(3)
    col0.append(1)
    col0.append(4)
    var col1 = List[Int]()
    col1.append(10)
    col1.append(20)
    col1.append(30)
    var sa = _make_struct(col0, col1)
    var result = sort(sa, keys, asc)
    var k = result.field(0)
    assert_equal(k[0].as_int32().value(), Scalar[int32.native](4))
    assert_equal(k[1].as_int32().value(), Scalar[int32.native](3))
    assert_equal(k[2].as_int32().value(), Scalar[int32.native](1))


# ---------------------------------------------------------------------------
# Large array — basic correctness
# ---------------------------------------------------------------------------


def test_argsort_int32_large() raises:
    comptime N = 10_000
    var b = Int32Builder(capacity=N)
    for i in range(N // 2):
        b.append(Scalar[int32.native](N - 1 - i))
        b.append(Scalar[int32.native](i))
    var a: AnyArray = b.finish().to_any()
    var idx = argsort(a)
    assert_equal(len(idx), N)
    var first_val = a.as_int32().unsafe_get(Int(idx.unsafe_get(0)))
    var last_val = a.as_int32().unsafe_get(Int(idx.unsafe_get(N - 1)))
    assert_equal(first_val, Scalar[int32.native](0))
    assert_equal(last_val, Scalar[int32.native](N - 1))


# ---------------------------------------------------------------------------
# All numeric dtypes — ascending sort smoke test
# ---------------------------------------------------------------------------


def test_argsort_uint8() raises:
    var a: AnyArray = array([200, 100, 50, 150], uint8)
    var idx = argsort(a)
    assert_equal(_idx(idx, 0), 2)
    assert_equal(_idx(idx, 1), 1)
    assert_equal(_idx(idx, 2), 3)
    assert_equal(_idx(idx, 3), 0)


def test_argsort_uint32() raises:
    var a: AnyArray = array([3, 1, 2], uint32)
    var idx = argsort(a)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 0)


def test_argsort_float32_ascending() raises:
    var a: AnyArray = array([3.0, 1.0, 2.0], float32)
    var idx = argsort(a)
    assert_equal(_idx(idx, 0), 1)
    assert_equal(_idx(idx, 1), 2)
    assert_equal(_idx(idx, 2), 0)


def main() raises:
    TestSuite.run[__functions_in_module()]()
