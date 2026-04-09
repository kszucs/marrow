from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from marrow.testing import TestSuite

from marrow.arrays import (
    AnyArray,
    PrimitiveArray,
    BoolArray,
)
from marrow.builders import (
    array,
    arange,
    nulls,
    PrimitiveBuilder,
    StringBuilder,
)
from marrow.dtypes import (
    int32,
    int64,
    uint8,
    float32,
    bool_,
    Int32Type,
    Int64Type,
    UInt8Type,
    Float32Type,
)
from marrow.kernels.filter import filter_, drop_nulls


# ---------------------------------------------------------------------------
# filter — primitive arrays
# ---------------------------------------------------------------------------


def test_filter_keep_all() raises:
    var a = array[Int32Type]([1, 2, 3, 4])
    var result = filter_(a, array([True, True, True, True]))
    assert_equal(len(result), 4)
    assert_equal(result[0], 1)
    assert_equal(result[3], 4)


def test_filter_keep_none() raises:
    var a = array[Int32Type]([1, 2, 3])
    var result = filter_(a, array([False, False, False]))
    assert_equal(len(result), 0)


def test_filter_alternating() raises:
    var a = array[Int32Type]([10, 20, 30, 40, 50])
    var result = filter_(a, array([True, False, True, False, True]))
    assert_equal(len(result), 3)
    assert_equal(result[0], 10)
    assert_equal(result[1], 30)
    assert_equal(result[2], 50)


def test_filter_first_and_last() raises:
    var a = array[Int32Type]([1, 2, 3, 4, 5])
    var result = filter_(a, array([True, False, False, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0], 1)
    assert_equal(result[1], 5)


def test_filter_empty_array() raises:
    var a = array[Int32Type]()
    var result = filter_(a, array(List[Optional[Bool]]()))
    assert_equal(len(result), 0)


def test_filter_single_true() raises:
    var a = array[Int64Type]([42])
    var result = filter_(a, array([True]))
    assert_equal(len(result), 1)
    assert_equal(result[0], 42)


def test_filter_single_false() raises:
    var a = array[Int64Type]([42])
    var result = filter_(a, array([False]))
    assert_equal(len(result), 0)


def test_filter_exactly_8_elements() raises:
    """Tests that a single full byte of selection is processed correctly."""
    var a = array[Int32Type]([1, 2, 3, 4, 5, 6, 7, 8])
    var result = filter_(
        a, array([True, False, True, False, True, False, True, False])
    )
    assert_equal(len(result), 4)
    assert_equal(result[0], 1)
    assert_equal(result[1], 3)
    assert_equal(result[2], 5)
    assert_equal(result[3], 7)


def test_filter_cross_byte_boundary() raises:
    """Tests selection spanning multiple bytes (> 8 elements)."""
    var a = array[Int32Type]([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Keep last 2 of first byte and first 2 of second byte
    var result = filter_(
        a,
        array(
            [False, False, False, False, False, False, True, True, True, True]
        ),
    )
    assert_equal(len(result), 4)
    assert_equal(result[0], 7)
    assert_equal(result[1], 8)
    assert_equal(result[2], 9)
    assert_equal(result[3], 10)


def test_filter_sparse_zero_byte() raises:
    """Zero bytes in selection bitmap are skipped without inspecting elements.
    """
    var a = arange[Int32Type](0, 20)
    # Only keep element 16 (first element of the third selection byte)
    var sel = List[Optional[Bool]]()
    for i in range(20):
        sel.append(i == 16)
    var result = filter_(a, array(sel))
    assert_equal(len(result), 1)
    assert_equal(result[0], 16)


def test_filter_preserves_null_count() raises:
    """Nulls in the source are preserved at filtered positions."""
    var b = PrimitiveBuilder[Int32Type](4)
    b.append(1)
    b.append_null()
    b.append(3)
    b.append_null()
    var a = b.finish()
    # Select elements 0 (valid), 1 (null), 3 (null)
    var result = filter_(a, array([True, True, False, True]))
    assert_equal(len(result), 3)
    assert_equal(result.nulls, 2)
    assert_true(result.is_valid(0))
    assert_equal(result[0], 1)
    assert_true(not result.is_valid(1))
    assert_true(not result.is_valid(2))


def test_filter_all_null_source() raises:
    var a = nulls[Int32Type](4)
    var result = filter_(a, array([True, False, True, False]))
    assert_equal(len(result), 2)
    assert_equal(result.nulls, 2)


def test_filter_length_mismatch_raises() raises:
    var a = array[Int32Type]([1, 2, 3])
    with assert_raises():
        _ = filter_(a, array([True, False]))


def test_filter_float32() raises:
    var a = array[Float32Type]([1, 2, 3, 4])
    var result = filter_(a, array([False, True, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0], 2.0)
    assert_equal(result[1], 4.0)


def test_filter_bool_array() raises:
    """Filter of a bool array produces correct bit-packed output."""
    var a = array([True, False, True, True, False, True])
    var result = filter_(a, array([True, True, False, True, False, False]))
    assert_equal(len(result), 3)
    assert_equal(result[0].value(), True)
    assert_equal(result[1].value(), False)
    assert_equal(result[2].value(), True)


# ---------------------------------------------------------------------------
# filter — string arrays
# ---------------------------------------------------------------------------


def test_filter_strings_basic() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    s.append("foo")
    var a = s.finish()
    var result = filter_(a, array([True, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0], "hello")
    assert_equal(result[1], "foo")


def test_filter_strings_keep_all() raises:
    var s = StringBuilder()
    s.append("a")
    s.append("bb")
    s.append("ccc")
    var a = s.finish()
    var result = filter_(a, array([True, True, True]))
    assert_equal(len(result), 3)
    assert_equal(result[0], "a")
    assert_equal(result[1], "bb")
    assert_equal(result[2], "ccc")


def test_filter_strings_keep_none() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    var a = s.finish()
    var result = filter_(a, array([False, False]))
    assert_equal(len(result), 0)


def test_filter_strings_single() raises:
    var s = StringBuilder()
    s.append("only")
    var a = s.finish()
    var result = filter_(a, array([True]))
    assert_equal(len(result), 1)
    assert_equal(result[0], "only")


def test_filter_strings_with_nulls() raises:
    """Null strings in source are preserved at selected positions."""
    var s = StringBuilder()
    s.append("valid")
    s.append_null()
    s.append("also_valid")
    s.append_null()
    var a = s.finish()
    # Keep: "valid" (pos 0), null (pos 1), null (pos 3)
    var result = filter_(a, array([True, True, False, True]))
    assert_equal(len(result), 3)
    assert_equal(result.nulls, 2)
    assert_true(result.is_valid(0))
    assert_equal(result[0], "valid")
    assert_false(result.is_valid(1))
    assert_false(result.is_valid(2))


def test_filter_strings_run_merging() raises:
    """Consecutive selected elements are merged into a single copy."""
    var s = StringBuilder()
    s.append("aaa")
    s.append("bbb")
    s.append("ccc")
    s.append("ddd")
    var a = s.finish()
    # Select 0,1,2 — consecutive, single memcpy internally
    var result = filter_(a, array([True, True, True, False]))
    assert_equal(len(result), 3)
    assert_equal(result[0], "aaa")
    assert_equal(result[1], "bbb")
    assert_equal(result[2], "ccc")


def test_filter_strings_non_consecutive() raises:
    """Non-consecutive selection forces separate memcpy calls per run."""
    var s = StringBuilder()
    s.append("first")
    s.append("skip")
    s.append("third")
    s.append("skip2")
    s.append("fifth")
    var a = s.finish()
    var result = filter_(a, array([True, False, True, False, True]))
    assert_equal(len(result), 3)
    assert_equal(result[0], "first")
    assert_equal(result[1], "third")
    assert_equal(result[2], "fifth")


def test_filter_strings_empty_strings() raises:
    """Empty strings have zero bytes and don't corrupt offsets."""
    var s = StringBuilder()
    s.append("")
    s.append("x")
    s.append("")
    var a = s.finish()
    var result = filter_(a, array([True, True, True]))
    assert_equal(len(result), 3)
    assert_equal(result[0], "")
    assert_equal(result[1], "x")
    assert_equal(result[2], "")


def test_filter_strings_offsets_correct() raises:
    """Verify offsets buffer is a valid prefix sum after filtering."""
    var s = StringBuilder()
    s.append("ab")
    s.append("cde")
    s.append("f")
    var a = s.finish()
    # Keep "ab" and "f" → offsets [0, 2, 3]
    var result = filter_(a, array([True, False, True]))
    assert_equal(result.offsets.unsafe_get[DType.uint32](0), 0)
    assert_equal(result.offsets.unsafe_get[DType.uint32](1), 2)
    assert_equal(result.offsets.unsafe_get[DType.uint32](2), 3)


def test_filter_strings_length_mismatch_raises() raises:
    var s = StringBuilder()
    s.append("a")
    s.append("b")
    var a = s.finish()
    with assert_raises():
        _ = filter_(a, array([True]))


# ---------------------------------------------------------------------------
# filter — runtime-typed AnyArray dispatch
# ---------------------------------------------------------------------------


def test_filter_array_dispatch_int32() raises:
    var a = AnyArray(array[Int32Type]([10, 20, 30]))
    var result = filter_(a, array([False, True, True]))
    assert_equal(result.length(), 2)


def test_filter_array_dispatch_float32() raises:
    var a = AnyArray(array[Float32Type]([1, 2, 3]))
    var result = filter_(a, array([True, False, True]))
    assert_equal(result.length(), 2)


def test_filter_array_dispatch_string() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    var a = AnyArray(s.finish())
    var result = filter_(a, array([True, False]))
    assert_equal(result.length(), 1)


def test_filter_array_dispatch_length_mismatch_raises() raises:
    var a = AnyArray(array[Int32Type]([1, 2, 3]))
    with assert_raises():
        _ = filter_(a, array([True, False]))


# ---------------------------------------------------------------------------
# drop_nulls
# ---------------------------------------------------------------------------


def test_drop_nulls_typed() raises:
    var b = PrimitiveBuilder[Int32Type](4)
    b.append(10)
    b.append_null()
    b.append(30)
    b.append_null()
    var result = drop_nulls(b.finish())
    assert_equal(len(result), 2)
    assert_equal(result[0], 10)
    assert_equal(result[1], 30)


def test_drop_nulls_no_nulls() raises:
    var a = array[Int64Type]([1, 2, 3])
    var result = drop_nulls(a)
    assert_equal(len(result), 3)
    assert_equal(result[0], 1)
    assert_equal(result[1], 2)
    assert_equal(result[2], 3)


def test_drop_nulls_all_nulls() raises:
    var result = drop_nulls(nulls[Int64Type](5))
    assert_equal(len(result), 0)


def test_drop_nulls_empty() raises:
    var result = drop_nulls(array[Int32Type]())
    assert_equal(len(result), 0)


def test_drop_nulls_untyped() raises:
    var result = drop_nulls(
        array[UInt8Type]([None, 1, None, 3, None, 5, None, 7, None, 9])
    )
    assert_equal(result.length, 5)


def test_drop_nulls_values_correct() raises:
    var result = drop_nulls(
        array[UInt8Type]([None, 1, None, 3, None, 5, None, 7, None, 9])
    )
    assert_equal(len(result), 5)
    assert_equal(result[0], 1)
    assert_equal(result[1], 3)
    assert_equal(result[2], 5)
    assert_equal(result[3], 7)
    assert_equal(result[4], 9)


# ---------------------------------------------------------------------------
# filter — sliced (offset) arrays
# ---------------------------------------------------------------------------


def test_filter_sliced_array() raises:
    """Filter a sliced int32 array with alternating selection."""
    var a = array[Int32Type]([10, 20, 30, 40, 50])
    var sliced = a.slice(1, 3)  # [20, 30, 40] with offset=1
    assert_equal(sliced.offset, 1)
    var result = filter_(sliced, array([True, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0], 20)
    assert_equal(result[1], 40)


def test_filter_sliced_keep_all() raises:
    """All-selected path with offset array."""
    var a = array[Int32Type]([1, 2, 3, 4, 5])
    var sliced = a.slice(2, 3)  # [3, 4, 5] with offset=2
    var result = filter_(sliced, array([True, True, True]))
    assert_equal(len(result), 3)
    assert_equal(result[0], 3)
    assert_equal(result[1], 4)
    assert_equal(result[2], 5)


def test_filter_sliced_with_nulls() raises:
    """Sliced array with nulls preserves validity."""
    var b = PrimitiveBuilder[Int32Type](6)
    b.append(1)
    b.append_null()
    b.append(3)
    b.append_null()
    b.append(5)
    b.append(6)
    var a = b.finish()
    var sliced = a.slice(1, 4)  # [null, 3, null, 5] with offset=1
    var result = filter_(sliced, array([True, True, True, False]))
    assert_equal(len(result), 3)
    assert_equal(result.nulls, 2)
    assert_false(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_equal(result[1], 3)
    assert_false(result.is_valid(2))


def test_filter_sliced_bool() raises:
    """Filter a sliced bool array."""
    var a = array([True, False, True, True, False])
    var sliced = a.slice(1, 3)  # [False, True, True] with offset=1
    var result = filter_(sliced, array([True, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), False)
    assert_equal(result[1].value(), True)


def test_filter_sliced_strings() raises:
    """Filter a sliced StringArray."""
    var s = StringBuilder()
    s.append("aa")
    s.append("bb")
    s.append("cc")
    s.append("dd")
    s.append("ee")
    var a = s.finish()
    var sliced = a.slice(1, 3)  # ["bb", "cc", "dd"] with offset=1
    var result = filter_(sliced, array([True, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0], "bb")
    assert_equal(result[1], "dd")


def test_drop_nulls_sliced() raises:
    """``drop_nulls`` on a sliced array with nulls."""
    var b = PrimitiveBuilder[Int32Type](6)
    b.append(10)
    b.append_null()
    b.append(30)
    b.append_null()
    b.append(50)
    b.append(60)
    var a = b.finish()
    var sliced = a.slice(1, 4)  # [null, 30, null, 50] with offset=1
    var result = drop_nulls(sliced)
    assert_equal(len(result), 2)
    assert_equal(result[0], 30)
    assert_equal(result[1], 50)


def main() raises:
    TestSuite.run[__functions_in_module()]()
