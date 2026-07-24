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
    Int32Builder,
    Date32Builder,
    TimestampBuilder,
    DurationBuilder,
)
from marrow.dtypes import (
    int32,
    int64,
    uint8,
    float32,
    bool_,
    date32,
    timestamp,
    duration,
    second,
    Int32Type,
    Int64Type,
    UInt8Type,
    Float32Type,
)
from marrow.kernels.filter import filter, drop_null, take


# ---------------------------------------------------------------------------
# filter — primitive arrays
# ---------------------------------------------------------------------------


def test_filterkeep_all() raises:
    var a = array([1, 2, 3, 4], int32)
    var result = filter(a, array([True, True, True, True]))
    assert_equal(len(result), 4)
    assert_equal(result[0].value(), 1)
    assert_equal(result[3].value(), 4)


def test_filterkeep_none() raises:
    var a = array([1, 2, 3], int32)
    var result = filter(a, array([False, False, False]))
    assert_equal(len(result), 0)


def test_filteralternating() raises:
    var a = array([10, 20, 30, 40, 50], int32)
    var result = filter(a, array([True, False, True, False, True]))
    assert_equal(len(result), 3)
    assert_equal(result[0].value(), 10)
    assert_equal(result[1].value(), 30)
    assert_equal(result[2].value(), 50)


def test_filterfirst_and_last() raises:
    var a = array([1, 2, 3, 4, 5], int32)
    var result = filter(a, array([True, False, False, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 5)


def test_filterempty_array() raises:
    var a = array(int32)
    var result = filter(a, array(List[Optional[Bool]]()))
    assert_equal(len(result), 0)


def test_filtersingle_true() raises:
    var a = array([42], int64)
    var result = filter(a, array([True]))
    assert_equal(len(result), 1)
    assert_equal(result[0].value(), 42)


def test_filtersingle_false() raises:
    var a = array([42], int64)
    var result = filter(a, array([False]))
    assert_equal(len(result), 0)


def test_filterexactly_8_elements() raises:
    """Tests that a single full byte of selection is processed correctly."""
    var a = array([1, 2, 3, 4, 5, 6, 7, 8], int32)
    var result = filter(
        a, array([True, False, True, False, True, False, True, False])
    )
    assert_equal(len(result), 4)
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 3)
    assert_equal(result[2].value(), 5)
    assert_equal(result[3].value(), 7)


def test_filtercross_byte_boundary() raises:
    """Tests selection spanning multiple bytes (> 8 elements)."""
    var a = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], int32)
    # Keep last 2 of first byte and first 2 of second byte
    var result = filter(
        a,
        array(
            [False, False, False, False, False, False, True, True, True, True]
        ),
    )
    assert_equal(len(result), 4)
    assert_equal(result[0].value(), 7)
    assert_equal(result[1].value(), 8)
    assert_equal(result[2].value(), 9)
    assert_equal(result[3].value(), 10)


def test_filtersparse_zero_byte() raises:
    """Zero bytes in selection bitmap are skipped without inspecting elements.
    """
    var a = arange[Int32Type](0, 20)
    # Only keep element 16 (first element of the third selection byte)
    var sel = List[Optional[Bool]]()
    for i in range(20):
        sel.append(i == 16)
    var result = filter(a, array(sel))
    assert_equal(len(result), 1)
    assert_equal(result[0].value(), 16)


def test_filterpreserves_null_count() raises:
    """Nulls in the source are preserved at filtered positions."""
    var b = Int32Builder(4)
    b.append(1)
    b.append_null()
    b.append(3)
    b.append_null()
    var a = b.finish()
    # Select elements 0 (valid), 1 (null), 3 (null)
    var result = filter(a, array([True, True, False, True]))
    assert_equal(len(result), 3)
    assert_equal(result.nulls, 2)
    assert_true(result.is_valid(0))
    assert_equal(result[0].value(), 1)
    assert_true(not result.is_valid(1))
    assert_true(not result.is_valid(2))


def test_filterall_null_source() raises:
    var a = nulls(4, int32)
    var result = filter(a, array([True, False, True, False]))
    assert_equal(len(result), 2)
    assert_equal(result.nulls, 2)


def test_filterlength_mismatch_raises() raises:
    var a = array([1, 2, 3], int32)
    with assert_raises():
        _ = filter(a, array([True, False]))


def test_filterfloat32() raises:
    var a = array([1, 2, 3, 4], float32)
    var result = filter(a, array([False, True, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), 2.0)
    assert_equal(result[1].value(), 4.0)


def test_filterbool_array() raises:
    """Filter of a bool array produces correct bit-packed output."""
    var a = array([True, False, True, True, False, True])
    var result = filter(a, array([True, True, False, True, False, False]))
    assert_equal(len(result), 3)
    assert_equal(result[0].value(), True)
    assert_equal(result[1].value(), False)
    assert_equal(result[2].value(), True)


# ---------------------------------------------------------------------------
# filter — string arrays
# ---------------------------------------------------------------------------


def test_filterstrings_basic() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    s.append("foo")
    var a = s.finish()
    var result = filter(a, array([True, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0].to_string(), "hello")
    assert_equal(result[1].to_string(), "foo")


def test_filterstrings_keep_all() raises:
    var s = StringBuilder()
    s.append("a")
    s.append("bb")
    s.append("ccc")
    var a = s.finish()
    var result = filter(a, array([True, True, True]))
    assert_equal(len(result), 3)
    assert_equal(result[0].to_string(), "a")
    assert_equal(result[1].to_string(), "bb")
    assert_equal(result[2].to_string(), "ccc")


def test_filterstrings_keep_none() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    var a = s.finish()
    var result = filter(a, array([False, False]))
    assert_equal(len(result), 0)


def test_filterstrings_single() raises:
    var s = StringBuilder()
    s.append("only")
    var a = s.finish()
    var result = filter(a, array([True]))
    assert_equal(len(result), 1)
    assert_equal(result[0].to_string(), "only")


def test_filterstrings_with_nulls() raises:
    """Null strings in source are preserved at selected positions."""
    var s = StringBuilder()
    s.append("valid")
    s.append_null()
    s.append("also_valid")
    s.append_null()
    var a = s.finish()
    # Keep: "valid" (pos 0), null (pos 1), null (pos 3)
    var result = filter(a, array([True, True, False, True]))
    assert_equal(len(result), 3)
    assert_equal(result.nulls, 2)
    assert_true(result.is_valid(0))
    assert_equal(result[0].to_string(), "valid")
    assert_false(result.is_valid(1))
    assert_false(result.is_valid(2))


def test_filterstrings_run_merging() raises:
    """Consecutive selected elements are merged into a single copy."""
    var s = StringBuilder()
    s.append("aaa")
    s.append("bbb")
    s.append("ccc")
    s.append("ddd")
    var a = s.finish()
    # Select 0,1,2 — consecutive, single memcpy internally
    var result = filter(a, array([True, True, True, False]))
    assert_equal(len(result), 3)
    assert_equal(result[0].to_string(), "aaa")
    assert_equal(result[1].to_string(), "bbb")
    assert_equal(result[2].to_string(), "ccc")


def test_filterstrings_non_consecutive() raises:
    """Non-consecutive selection forces separate memcpy calls per run."""
    var s = StringBuilder()
    s.append("first")
    s.append("skip")
    s.append("third")
    s.append("skip2")
    s.append("fifth")
    var a = s.finish()
    var result = filter(a, array([True, False, True, False, True]))
    assert_equal(len(result), 3)
    assert_equal(result[0].to_string(), "first")
    assert_equal(result[1].to_string(), "third")
    assert_equal(result[2].to_string(), "fifth")


def test_filterstrings_empty_strings() raises:
    """Empty strings have zero bytes and don't corrupt offsets."""
    var s = StringBuilder()
    s.append("")
    s.append("x")
    s.append("")
    var a = s.finish()
    var result = filter(a, array([True, True, True]))
    assert_equal(len(result), 3)
    assert_equal(result[0], "")
    assert_equal(result[1].to_string(), "x")
    assert_equal(result[2], "")


def test_filterstrings_offsets_correct() raises:
    """Verify offsets buffer is a valid prefix sum after filtering."""
    var s = StringBuilder()
    s.append("ab")
    s.append("cde")
    s.append("f")
    var a = s.finish()
    # Keep "ab" and "f" → offsets [0, 2, 3]
    var result = filter(a, array([True, False, True]))
    assert_equal(result.offsets.unsafe_get[DType.uint32](0), 0)
    assert_equal(result.offsets.unsafe_get[DType.uint32](1), 2)
    assert_equal(result.offsets.unsafe_get[DType.uint32](2), 3)


def test_filterstrings_length_mismatch_raises() raises:
    var s = StringBuilder()
    s.append("a")
    s.append("b")
    var a = s.finish()
    with assert_raises():
        _ = filter(a, array([True]))


# ---------------------------------------------------------------------------
# filter — runtime-typed AnyArray dispatch
# ---------------------------------------------------------------------------


def test_filterarray_dispatch_int32() raises:
    var a: AnyArray = array([10, 20, 30], int32)
    var result = filter(a, array([False, True, True]))
    assert_equal(result.length(), 2)


def test_filterarray_dispatch_float32() raises:
    var a: AnyArray = array([1, 2, 3], float32)
    var result = filter(a, array([True, False, True]))
    assert_equal(result.length(), 2)


def test_filterarray_dispatch_string() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    var a: AnyArray = s.finish()
    var result = filter(a, array([True, False]))
    assert_equal(result.length(), 1)


def test_filterarray_dispatch_length_mismatch_raises() raises:
    var a: AnyArray = array([1, 2, 3], int32)
    with assert_raises():
        _ = filter(a, array([True, False]))


# ---------------------------------------------------------------------------
# drop_null
# ---------------------------------------------------------------------------


def test_drop_null_typed() raises:
    var b = Int32Builder(4)
    b.append(10)
    b.append_null()
    b.append(30)
    b.append_null()
    var result = drop_null(b.finish())
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), 10)
    assert_equal(result[1].value(), 30)


def test_drop_null_no_nulls() raises:
    var a = array([1, 2, 3], int64)
    var result = drop_null(a)
    assert_equal(len(result), 3)
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 2)
    assert_equal(result[2].value(), 3)


def test_drop_null_all_nulls() raises:
    var result = drop_null(nulls(5, int64))
    assert_equal(len(result), 0)


def test_drop_null_empty() raises:
    var result = drop_null(array(int32))
    assert_equal(len(result), 0)


def test_drop_null_untyped() raises:
    var result = drop_null(
        array([None, 1, None, 3, None, 5, None, 7, None, 9], uint8)
    )
    assert_equal(result.length, 5)


def test_drop_null_values_correct() raises:
    var result = drop_null(
        array([None, 1, None, 3, None, 5, None, 7, None, 9], uint8)
    )
    assert_equal(len(result), 5)
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 3)
    assert_equal(result[2].value(), 5)
    assert_equal(result[3].value(), 7)
    assert_equal(result[4].value(), 9)


# ---------------------------------------------------------------------------
# filter — sliced (offset) arrays
# ---------------------------------------------------------------------------


def test_filtersliced_array() raises:
    """Filter a sliced int32 array with alternating selection."""
    var a = array([10, 20, 30, 40, 50], int32)
    var sliced = a.slice(1, 3)  # [20, 30, 40] with offset=1
    assert_equal(sliced.offset, 1)
    var result = filter(sliced, array([True, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), 20)
    assert_equal(result[1].value(), 40)


def test_filtersliced_keep_all() raises:
    """All-selected path with offset array."""
    var a = array([1, 2, 3, 4, 5], int32)
    var sliced = a.slice(2, 3)  # [3, 4, 5] with offset=2
    var result = filter(sliced, array([True, True, True]))
    assert_equal(len(result), 3)
    assert_equal(result[0].value(), 3)
    assert_equal(result[1].value(), 4)
    assert_equal(result[2].value(), 5)


def test_filtersliced_with_nulls() raises:
    """Sliced array with nulls preserves validity."""
    var b = Int32Builder(6)
    b.append(1)
    b.append_null()
    b.append(3)
    b.append_null()
    b.append(5)
    b.append(6)
    var a = b.finish()
    var sliced = a.slice(1, 4)  # [null, 3, null, 5] with offset=1
    var result = filter(sliced, array([True, True, True, False]))
    assert_equal(len(result), 3)
    assert_equal(result.nulls, 2)
    assert_false(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_equal(result[1].value(), 3)
    assert_false(result.is_valid(2))


def test_filtersliced_bool() raises:
    """Filter a sliced bool array."""
    var a = array([True, False, True, True, False])
    var sliced = a.slice(1, 3)  # [False, True, True] with offset=1
    var result = filter(sliced, array([True, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), False)
    assert_equal(result[1].value(), True)


def test_filtersliced_strings() raises:
    """Filter a sliced StringArray."""
    var s = StringBuilder()
    s.append("aa")
    s.append("bb")
    s.append("cc")
    s.append("dd")
    s.append("ee")
    var a = s.finish()
    var sliced = a.slice(1, 3)  # ["bb", "cc", "dd"] with offset=1
    var result = filter(sliced, array([True, False, True]))
    assert_equal(len(result), 2)
    assert_equal(result[0].to_string(), "bb")
    assert_equal(result[1].to_string(), "dd")


def test_drop_null_sliced() raises:
    """``drop_null`` on a sliced array with nulls."""
    var b = Int32Builder(6)
    b.append(10)
    b.append_null()
    b.append(30)
    b.append_null()
    b.append(50)
    b.append(60)
    var a = b.finish()
    var sliced = a.slice(1, 4)  # [null, 30, null, 50] with offset=1
    var result = drop_null(sliced)
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), 30)
    assert_equal(result[1].value(), 50)


# ---------------------------------------------------------------------------
# filter / take / drop_null — temporal columns (routed through int backing)
# ---------------------------------------------------------------------------


def _date32(var days: List[Int]) raises -> AnyArray:
    var b = Date32Builder(date32(), len(days))
    for d in days:
        b.append(Scalar[int32.native](d))
    return b.finish()


def _timestamp(var vals: List[Int]) raises -> AnyArray:
    var b = TimestampBuilder(timestamp(second, "UTC"), len(vals))
    for v in vals:
        b.append(Scalar[int64.native](v))
    return b.finish()


def _duration(var vals: List[Int]) raises -> AnyArray:
    var b = DurationBuilder(duration(second), len(vals))
    for v in vals:
        b.append(Scalar[int64.native](v))
    return b.finish()


def test_filter_date32() raises:
    """Filter a date32 column — dtype preserved, values selected."""
    var a = _date32([19000, 18500, 19100, 18800])
    var result = filter(a, array([True, False, True, True]))
    assert_true(result.dtype() == date32().to_any())  # dtype preserved
    assert_equal(len(result), 3)
    ref r = result.as_date32()
    assert_equal(r[0].value(), Scalar[int32.native](19000))
    assert_equal(r[1].value(), Scalar[int32.native](19100))
    assert_equal(r[2].value(), Scalar[int32.native](18800))


def test_filter_timestamp_preserves_unit_tz() raises:
    """Filter a timestamp column — unit/tz preserved through the reinterpret."""
    var a = _timestamp([1000, 2000, 3000, 4000, 5000])
    var result = filter(a, array([False, True, False, True, True]))
    assert_true(result.dtype() == timestamp(second, "UTC").to_any())
    assert_equal(len(result), 3)
    ref r = result.as_timestamp()
    assert_equal(r[0].value(), Scalar[int64.native](2000))
    assert_equal(r[1].value(), Scalar[int64.native](4000))
    assert_equal(r[2].value(), Scalar[int64.native](5000))


def test_filter_temporal_with_nulls() raises:
    """Filter a timestamp column with nulls — validity rides through unchanged.
    """
    var b = TimestampBuilder(timestamp(second, "UTC"), 4)
    b.append(Scalar[int64.native](1000))
    b.append_null()
    b.append(Scalar[int64.native](3000))
    b.append_null()
    var a: AnyArray = b.finish()
    var result = filter(a, array([True, True, False, True]))
    assert_equal(len(result), 3)
    assert_equal(result.null_count(), 2)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result.as_timestamp()[0].value(), Scalar[int64.native](1000))


def test_take_date32() raises:
    """Gather rows from a date32 column at arbitrary indices."""
    var a = _date32([19000, 18500, 19100, 18800])
    var result = take(a, array([2, 0, 3, 1], int32))
    assert_true(result.dtype() == date32().to_any())
    assert_equal(len(result), 4)
    ref r = result.as_date32()
    assert_equal(r[0].value(), Scalar[int32.native](19100))
    assert_equal(r[1].value(), Scalar[int32.native](19000))
    assert_equal(r[2].value(), Scalar[int32.native](18800))
    assert_equal(r[3].value(), Scalar[int32.native](18500))


def test_take_duration_null_index() raises:
    """Take on a duration column — a null index produces a null output row."""
    var a = _duration([10, 20, 30])
    var idx = Int32Builder(capacity=3)
    idx.append(Scalar[int32.native](2))
    idx.append_null()
    idx.append(Scalar[int32.native](0))
    var result = take(a, idx.finish())
    assert_true(result.dtype() == duration(second).to_any())
    assert_equal(result.null_count(), 1)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    ref r = result.as_duration()
    assert_equal(r[0].value(), Scalar[int64.native](30))
    assert_equal(r[2].value(), Scalar[int64.native](10))


def test_drop_null_temporal() raises:
    """``drop_null`` on a timestamp column removes the null rows."""
    var b = TimestampBuilder(timestamp(second, "UTC"), 5)
    b.append(Scalar[int64.native](1000))
    b.append_null()
    b.append(Scalar[int64.native](3000))
    b.append_null()
    b.append(Scalar[int64.native](5000))
    var a: AnyArray = b.finish()
    var result = drop_null(a)
    assert_true(result.dtype() == timestamp(second, "UTC").to_any())
    assert_equal(len(result), 3)
    assert_equal(result.null_count(), 0)
    ref r = result.as_timestamp()
    assert_equal(r[0].value(), Scalar[int64.native](1000))
    assert_equal(r[1].value(), Scalar[int64.native](3000))
    assert_equal(r[2].value(), Scalar[int64.native](5000))


def test_cross_check_temporal_pyarrow() raises:
    """Cross-check temporal filter/take against pyarrow ``pc.filter``/``pc.take``.
    """
    from std.python import Python

    var pa = Python.import_module("pyarrow")
    var pc = Python.import_module("pyarrow.compute")

    var raw = [0, 1_560_601_845, 1_582_934_400, 1_609_459_200, -1, 915_148_800]
    var pylist = Python.list()
    for v in raw:
        pylist.append(v)
    var pa_arr = pa.array(pylist, type=pa.timestamp("s", "UTC"))
    var a = _timestamp(raw^)

    # filter
    var mask: List[Optional[Bool]] = [True, False, True, True, False, True]
    var pa_mask = Python.list()
    for m in mask:
        pa_mask.append(m.value())
    var got_f = filter(a, array(mask^))
    var pa_f = pc.filter(pa_arr, pa.array(pa_mask)).cast(pa.int64())
    ref rf = got_f.as_timestamp()
    assert_equal(len(got_f), Int(py=pa_f.__len__()))
    for i in range(len(got_f)):
        assert_equal(Int(rf[i].value()), Int(py=pa_f[i].as_py()))

    # take
    var idx: List[Optional[Int]] = [4, 0, 5, 2, 1]
    var pa_idx = Python.list()
    for k in idx:
        pa_idx.append(k.value())
    var got_t = take(a, array(idx^, int32))
    var pa_t = pc.take(pa_arr, pa.array(pa_idx)).cast(pa.int64())
    ref rt = got_t.as_timestamp()
    for i in range(len(got_t)):
        assert_equal(Int(rt[i].value()), Int(py=pa_t[i].as_py()))


def main() raises:
    TestSuite.run[__functions_in_module()]()
