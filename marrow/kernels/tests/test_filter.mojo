from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)

from ...arrays import (
    DynArray,
    PrimitiveArray,
    BoolArray,
)
from ...builders import (
    array,
    arange,
    nulls,
    PrimitiveBuilder,
    StringBuilder,
    Int32Builder,
    Int64Builder,
    Date32Builder,
    TimestampBuilder,
    DurationBuilder,
    BoolBuilder,
)
from ...dtypes import (
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
from ...buffers import Bitmap
from ...kernels.filter import Filter, Take, filter, take, drop_null


# ---------------------------------------------------------------------------
# filter — primitive arrays
# ---------------------------------------------------------------------------


def test_filterkeep_all() raises:
    var a = array([1, 2, 3, 4], int32)
    var result = Filter.apply(a, (array([True, True, True, True])).values())
    assert_equal(len(result), 4)
    assert_equal(result[0].value(), 1)
    assert_equal(result[3].value(), 4)


def test_filterkeep_none() raises:
    var a = array([1, 2, 3], int32)
    var result = Filter.apply(a, (array([False, False, False])).values())
    assert_equal(len(result), 0)


def test_filteralternating() raises:
    var a = array([10, 20, 30, 40, 50], int32)
    var result = Filter.apply(
        a, (array([True, False, True, False, True])).values()
    )
    assert_equal(len(result), 3)
    assert_equal(result[0].value(), 10)
    assert_equal(result[1].value(), 30)
    assert_equal(result[2].value(), 50)


def test_filterfirst_and_last() raises:
    var a = array([1, 2, 3, 4, 5], int32)
    var result = Filter.apply(
        a, (array([True, False, False, False, True])).values()
    )
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 5)


def test_filterempty_array() raises:
    var a = array(int32)
    var result = Filter.apply(a, (array(List[Optional[Bool]]())).values())
    assert_equal(len(result), 0)


def test_filtersingle_true() raises:
    var a = array([42], int64)
    var result = Filter.apply(a, (array([True])).values())
    assert_equal(len(result), 1)
    assert_equal(result[0].value(), 42)


def test_filtersingle_false() raises:
    var a = array([42], int64)
    var result = Filter.apply(a, (array([False])).values())
    assert_equal(len(result), 0)


def test_filterexactly_8_elements() raises:
    """Tests that a single full byte of selection is processed correctly."""
    var a = array([1, 2, 3, 4, 5, 6, 7, 8], int32)
    var result = Filter.apply(
        a, array([True, False, True, False, True, False, True, False]).values()
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
    var result = Filter.apply(
        a,
        array(
            [False, False, False, False, False, False, True, True, True, True]
        ).values(),
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
    var result = Filter.apply(a, (array(sel)).values())
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
    var result = Filter.apply(a, (array([True, True, False, True])).values())
    assert_equal(len(result), 3)
    assert_equal(result.nulls, 2)
    assert_true(result.is_valid(0))
    assert_equal(result[0].value(), 1)
    assert_true(not result.is_valid(1))
    assert_true(not result.is_valid(2))


def test_filterall_null_source() raises:
    var a = nulls(4, int32)
    var result = Filter.apply(a, (array([True, False, True, False])).values())
    assert_equal(len(result), 2)
    assert_equal(result.nulls, 2)


def test_filterlength_mismatch_raises() raises:
    var a = array([1, 2, 3], int32)
    with assert_raises():
        _ = Filter.apply(a, (array([True, False])).values())


def test_filterfloat32() raises:
    var a = array([1, 2, 3, 4], float32)
    var result = Filter.apply(a, (array([False, True, False, True])).values())
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), 2.0)
    assert_equal(result[1].value(), 4.0)


def test_filterbool_array() raises:
    """Filter of a bool array produces correct bit-packed output."""
    var a = array([True, False, True, True, False, True])
    var result = Filter.apply(
        a, (array([True, True, False, True, False, False])).values()
    )
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
    var result = Filter.apply(a, (array([True, False, True])).values())
    assert_equal(len(result), 2)
    assert_equal(result[0].to_string(), "hello")
    assert_equal(result[1].to_string(), "foo")


def test_filterstrings_keep_all() raises:
    var s = StringBuilder()
    s.append("a")
    s.append("bb")
    s.append("ccc")
    var a = s.finish()
    var result = Filter.apply(a, (array([True, True, True])).values())
    assert_equal(len(result), 3)
    assert_equal(result[0].to_string(), "a")
    assert_equal(result[1].to_string(), "bb")
    assert_equal(result[2].to_string(), "ccc")


def test_filterstrings_keep_none() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    var a = s.finish()
    var result = Filter.apply(a, (array([False, False])).values())
    assert_equal(len(result), 0)


def test_filterstrings_single() raises:
    var s = StringBuilder()
    s.append("only")
    var a = s.finish()
    var result = Filter.apply(a, (array([True])).values())
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
    var result = Filter.apply(a, (array([True, True, False, True])).values())
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
    var result = Filter.apply(a, (array([True, True, True, False])).values())
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
    var result = Filter.apply(
        a, (array([True, False, True, False, True])).values()
    )
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
    var result = Filter.apply(a, (array([True, True, True])).values())
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
    var result = Filter.apply(a, (array([True, False, True])).values())
    assert_equal(result.offsets.unsafe_get[DType.uint32](0), 0)
    assert_equal(result.offsets.unsafe_get[DType.uint32](1), 2)
    assert_equal(result.offsets.unsafe_get[DType.uint32](2), 3)


def test_filterstrings_length_mismatch_raises() raises:
    var s = StringBuilder()
    s.append("a")
    s.append("b")
    var a = s.finish()
    with assert_raises():
        _ = Filter.apply(a, (array([True])).values())


# ---------------------------------------------------------------------------
# filter — runtime-typed DynArray dispatch
# ---------------------------------------------------------------------------


def test_filterarray_dispatch_int32() raises:
    var a: DynArray = array([10, 20, 30], int32)
    var result = filter(a, (array([False, True, True])))
    assert_equal(result.length(), 2)


def test_filterarray_dispatch_float32() raises:
    var a: DynArray = array([1, 2, 3], float32)
    var result = filter(a, (array([True, False, True])))
    assert_equal(result.length(), 2)


def test_filterarray_dispatch_string() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    var a: DynArray = s.finish()
    var result = filter(a, (array([True, False])))
    assert_equal(result.length(), 1)


def test_filterarray_dispatch_length_mismatch_raises() raises:
    var a: DynArray = array([1, 2, 3], int32)
    with assert_raises():
        _ = filter(a, (array([True, False])))


# ---------------------------------------------------------------------------
# drop_null
# ---------------------------------------------------------------------------


def test_drop_null_typed() raises:
    var b = Int32Builder(4)
    b.append(10)
    b.append_null()
    b.append(30)
    b.append_null()
    var result = Filter.drop_null(b.finish())
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), 10)
    assert_equal(result[1].value(), 30)


def test_drop_null_no_nulls() raises:
    var a = array([1, 2, 3], int64)
    var result = Filter.drop_null(a)
    assert_equal(len(result), 3)
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 2)
    assert_equal(result[2].value(), 3)


def test_drop_null_all_nulls() raises:
    var result = Filter.drop_null(nulls(5, int64))
    assert_equal(len(result), 0)


def test_drop_null_empty() raises:
    var result = Filter.drop_null(array(int32))
    assert_equal(len(result), 0)


def test_drop_null_untyped() raises:
    var result = drop_null(
        array([None, 1, None, 3, None, 5, None, 7, None, 9], uint8).to_dyn()
    )
    assert_equal(result.length(), 5)


def test_drop_null_values_correct() raises:
    var result = Filter.drop_null(
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
    var result = Filter.apply(sliced, (array([True, False, True])).values())
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), 20)
    assert_equal(result[1].value(), 40)


def test_filtersliced_keep_all() raises:
    """All-selected path with offset array."""
    var a = array([1, 2, 3, 4, 5], int32)
    var sliced = a.slice(2, 3)  # [3, 4, 5] with offset=2
    var result = Filter.apply(sliced, (array([True, True, True])).values())
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
    var result = Filter.apply(
        sliced, (array([True, True, True, False])).values()
    )
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
    var result = Filter.apply(sliced, (array([True, False, True])).values())
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
    var result = Filter.apply(sliced, (array([True, False, True])).values())
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
    var result = Filter.drop_null(sliced)
    assert_equal(len(result), 2)
    assert_equal(result[0].value(), 30)
    assert_equal(result[1].value(), 50)


# ---------------------------------------------------------------------------
# filter / take / drop_null — temporal columns (routed through int backing)
# ---------------------------------------------------------------------------


def _date32(var days: List[Int]) raises -> DynArray:
    var b = Date32Builder(date32(), len(days))
    for d in days:
        b.append(Scalar[int32.native](d))
    return b.finish()


def _timestamp(var vals: List[Int]) raises -> DynArray:
    var b = TimestampBuilder(timestamp(second, "UTC"), len(vals))
    for v in vals:
        b.append(Scalar[int64.native](v))
    return b.finish()


def _duration(var vals: List[Int]) raises -> DynArray:
    var b = DurationBuilder(duration(second), len(vals))
    for v in vals:
        b.append(Scalar[int64.native](v))
    return b.finish()


def test_filter_date32() raises:
    """Filter a date32 column — dtype preserved, values selected."""
    var a = _date32([19000, 18500, 19100, 18800])
    var result = filter(a, (array([True, False, True, True])))
    assert_true(result.dtype() == date32().to_dyn())  # dtype preserved
    assert_equal(len(result), 3)
    ref r = result.as_date32()
    assert_equal(r[0].value(), Scalar[int32.native](19000))
    assert_equal(r[1].value(), Scalar[int32.native](19100))
    assert_equal(r[2].value(), Scalar[int32.native](18800))


def test_filter_timestamp_preserves_unit_tz() raises:
    """Filter a timestamp column — unit/tz preserved through the reinterpret."""
    var a = _timestamp([1000, 2000, 3000, 4000, 5000])
    var result = filter(a, (array([False, True, False, True, True])))
    assert_true(result.dtype() == timestamp(second, "UTC").to_dyn())
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
    var a: DynArray = b.finish()
    var result = filter(a, (array([True, True, False, True])))
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
    assert_true(result.dtype() == date32().to_dyn())
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
    assert_true(result.dtype() == duration(second).to_dyn())
    assert_equal(result.null_count(), 1)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    ref r = result.as_duration()
    assert_equal(r[0].value(), Scalar[int64.native](30))
    assert_equal(r[2].value(), Scalar[int64.native](10))


def test_filter_bool_array_after_primitive_narrowing() raises:
    """`Filter.dispatch` peels bool off before `is_primitive()`. Narrowing that
    predicate must not disturb the bool arm."""
    var bb = BoolBuilder(capacity=4)
    bb.append(True)
    bb.append(False)
    bb.append_null()
    bb.append(True)
    var arr = bb.finish()

    var mb = BoolBuilder(capacity=4)
    mb.append(True)
    mb.append(True)
    mb.append(True)
    mb.append(False)
    var mask = mb.finish()

    var out = filter(arr^.to_dyn(), mask^).as_bool().copy()
    assert_equal(len(out), 3)
    assert_true(out[0].value())
    assert_true(not out[1].value())
    assert_true(out.is_null(2))


def test_take_bool_array_after_primitive_narrowing() raises:
    """`Take.dispatch` peels bool off before `is_primitive()`. Narrowing that
    predicate must not disturb the bool arm, and a null index still produces a
    null output row."""
    var a = array([True, False, True, False])
    var idx = Int32Builder(capacity=4)
    idx.append(Scalar[int32.native](2))
    idx.append_null()
    idx.append(Scalar[int32.native](0))
    idx.append(Scalar[int32.native](3))
    var result = take(a^, idx.finish())
    assert_equal(len(result), 4)
    ref r = result.as_bool()
    assert_true(r[0].value())
    assert_false(r.is_valid(1))
    assert_true(r[2].value())
    assert_false(r[3].value())


def test_drop_null_temporal() raises:
    """``drop_null`` on a timestamp column removes the null rows."""
    var b = TimestampBuilder(timestamp(second, "UTC"), 5)
    b.append(Scalar[int64.native](1000))
    b.append_null()
    b.append(Scalar[int64.native](3000))
    b.append_null()
    b.append(Scalar[int64.native](5000))
    var a: DynArray = b.finish()
    var result = Filter.drop_null(a)
    assert_true(result.dtype() == timestamp(second, "UTC").to_dyn())
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

    var raw: List[Int] = [
        0,
        1_560_601_845,
        1_582_934_400,
        1_609_459_200,
        -1,
        915_148_800,
    ]
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
    var got_f = filter(a, (array(mask^)))
    # `pc.filter` / `pc.take` are PyArrow's own API — this is the reference side
    # of the cross-check, so it must NOT be renamed to marrow's `Filter.apply`.
    # The result is cast to int64 so `as_py()` yields the epoch value that
    # `rf[i].value()` holds, rather than a datetime.
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


# ---------------------------------------------------------------------------
# Moved here from `test_join.mojo` (B23).
#
# These are `take` tests and belong beside the other `take` tests, but the move
# was forced rather than tidy: calling the *erased* `take(DynArray, Int32Array)`
# from `test_join.mojo` deadlocked the Mojo compiler whenever that file was
# selected on its own — 7 hours of wall clock against 10 s of CPU, with every
# worker thread parked. The identical call compiles fine here. See B23 for the
# minimal reproduction.
# ---------------------------------------------------------------------------


def test_take_primitive_basic_int32() raises:
    """Gather elements from a primitive array at given indices."""
    var a: DynArray = array([10, 20, 30, 40], int32)
    var result = take(a.copy(), array([2, 0, 3], int32))
    ref r = result.as_int32()
    assert_equal(r[0].value(), Scalar[int32.native](30))
    assert_equal(r[1].value(), Scalar[int32.native](10))
    assert_equal(r[2].value(), Scalar[int32.native](40))


def test_take_null_index_produces_null_int32() raises:
    """Null index in take produces a null output element."""
    var a: DynArray = array([10, 20, 30], int32)
    var idx = Int32Builder(capacity=2)
    idx.append_null()
    idx.append(Scalar[int32.native](1))
    var result = take(a.copy(), idx.finish())
    assert_equal(result.null_count(), 1)
    assert_false(result.is_valid(0))
    assert_true(result.is_valid(1))


def test_filtersliced_multiword_offset() raises:
    """Filter a sliced array long enough to span several selection words.

    Every other sliced test here fits inside a single 64-bit tail block, where
    the tail mask hides the top of the word. Past 64 elements the bulk loop
    runs, and `BitmapView.load_bits` on a view carrying a sub-byte bit offset
    used to return the run's top ``offset`` bits as zeros — silently dropping
    rows from the answer rather than raising.
    """
    var n = 400
    var vb = Int32Builder()
    for i in range(n):
        vb.append(Int32(i))
    var a = vb.finish()

    var start = 3  # a sub-byte bit offset on the sliced validity/data views
    var length = n - start
    var sliced = a.slice(start, length)
    assert_equal(sliced.offset, start)

    var mb = BoolBuilder()
    var expected = List[Int32]()
    for i in range(length):
        var keep = (i * 7) % 5 < 2
        mb.append(keep)
        if keep:
            expected.append(Int32(start + i))
    var mask = mb.finish()

    var result = Filter.apply(sliced, mask.values())
    assert_equal(len(result), len(expected))
    for i in range(len(expected)):
        assert_equal(result[i].value(), expected[i])


def test_filtersliced_multiword_offset_with_nulls() raises:
    """The same shape with a validity bitmap, which is filtered through the
    same `load_bits` path as the selection."""
    var n = 400
    var start = 5
    var length = n - start

    var b = Int32Builder()
    for i in range(n):
        if i % 3 == 0:
            b.append_null()
        else:
            b.append(Int32(i))
    var a = b.finish()
    var sliced = a.slice(start, length)

    var mb = BoolBuilder()
    var expect_values = List[Int32]()
    var expect_null = List[Bool]()
    for i in range(length):
        var keep = (i * 11) % 7 < 3
        mb.append(keep)
        if keep:
            var src = start + i
            expect_null.append(src % 3 == 0)
            expect_values.append(Int32(src))
    var mask = mb.finish()

    var result = Filter.apply(sliced, mask.values())
    assert_equal(len(result), len(expect_values))
    for i in range(len(expect_values)):
        if expect_null[i]:
            assert_true(result.is_null(i))
        else:
            assert_equal(result[i].value(), expect_values[i])


def test_filter_null_mask_entry_is_not_selected() raises:
    """A null in the mask must not select its row.

    Arrow drops rows whose mask entry is null — pyarrow's default
    `null_selection_behavior="drop"` — and SQL agrees: `WHERE v < 4` omits the
    row where `v` is NULL.

    The mask is built by hand because the defect needs the *data* bit under
    the null to be **set**, and no builder produces that — `append_null()`
    clears both bits. A comparison kernel does: it evaluates every SIMD lane
    whatever the validity says, so a null input compares its raw payload (0),
    writes the result, and only then marks the lane invalid. `filter` read
    that stray bit and selected the row.

    Building the mask with `LtKernel` would be the faithful spelling, but the
    hand-built form is equivalent and avoids one more instantiation.

    **This case fails by wedging the compiler, not by asserting.** With the
    `filter` fix reverted the whole compilation unit parks at 0% CPU with no
    diagnostic — verified four ways, and it is not the mask construction, the
    `LtKernel` import, the `as_int64()` narrowing or the argument spelling:
    the unit builds in 36s with the fix and hangs without it. So it does
    catch a regression, just noisily. `test_filter_null_mask_entry_is_not_selected`
    in `python/marrow/tests/test_compute.py` is the clean guard — that one
    fails by assertion in under a second.
    """
    var values = array([1, 5, 6, 2], int64)

    var bits = Bitmap.alloc_zeroed(4)
    for i in range(4):
        bits.set(i)  # every lane's data bit says True
    var valid = Bitmap.alloc_zeroed(4)
    valid.set(0)
    valid.set(2)
    valid.set(3)  # lane 1 is null, but its data bit stays set

    var mask = BoolArray(
        4,
        1,
        0,
        Optional(valid^.to_immutable(length=4)),
        bits^.to_immutable(length=4),
    )
    assert_equal(mask.null_count(), 1)

    # The row *count* is the whole assertion: the defect selected the null
    # lane, giving 4. Narrowing the erased result with `as_int64()` to check
    # the values as well wedges the elaborator — that narrowing inside
    # filter's dispatch ladder parks the compiler at 0% CPU with no
    # diagnostic — and the count already distinguishes the two behaviours.
    var erased: DynArray = values.copy()
    var kept = filter(erased, mask.copy())
    assert_equal(kept.length(), 3)  # 1, 6, 2 — never the null lane
