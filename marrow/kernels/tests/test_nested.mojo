# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the nested (list) compute kernels (marrow.kernels.nested).

Expected patterns match PyArrow's `pc.list_value_length`.
"""

from std.testing import assert_equal, assert_raises, assert_true, assert_false

from ...arrays import DynArray, ListArray, ListLikeArray, Int32Array, MapArray
from ...builders import (
    Float64Builder,
    Int8Builder,
    Int32Builder,
    ListBuilder,
    ListLikeBuilder,
    Int64Builder,
    MapBuilder,
    PrimitiveBuilder,
    StringBuilder,
    array,
)
from ...dtypes import (
    Float64Type,
    Int32Type,
    Int64Type,
    LargeListType,
    ListLikeType,
    ListType,
    NumericType,
    int8,
    int32,
    int64,
    string,
    map_,
    DynType,
)

from ...kernels.aggregate import (
    BitAndFold,
    BitOrFold,
    BitXorFold,
    CountFold,
    FoldKernel,
    MaxFold,
    MeanFold,
    MinFold,
    SumFold,
)
from ...kernels.nested import (
    ArrayContainsKernel,
    ArrayLengthKernel,
    ListReduceKernel,
)


def _lists_with_null() raises -> ListArray:
    """[[1, 2], [3], null, [4, 5, 6]]."""
    var ints_b = Int64Builder()
    var list_b = ListBuilder(ints_b^)
    var child_any = list_b.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    list_b.append_valid()  # [1, 2]
    child.append(3)
    list_b.append_valid()  # [3]
    list_b.append_null()  # null
    child.append(4)
    child.append(5)
    child.append(6)
    list_b.append_valid()  # [4, 5, 6]
    return list_b.finish()


def _lists_no_null() raises -> ListArray:
    """[[10], [], [20, 30]]."""
    var ints_b = Int64Builder()
    var list_b = ListBuilder(ints_b^)
    var child_any = list_b.values()
    ref child = child_any.as_int64()
    child.append(10)
    list_b.append_valid()  # [10]
    list_b.append_valid()  # []
    child.append(20)
    child.append(30)
    list_b.append_valid()  # [20, 30]
    return list_b.finish()


def test_array_length_no_nulls() raises:
    var r = ArrayLengthKernel.apply(_lists_no_null())
    assert_equal(r.null_count(), 0)
    assert_true(r == array([1, 0, 2], int32))


def test_array_length_propagates_nulls() raises:
    # matches pc.list_value_length: null list -> null length (not 0)
    var r = ArrayLengthKernel.apply(_lists_with_null())
    assert_equal(r.null_count(), 1)
    assert_true(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_false(r.is_valid(2))
    assert_true(r.is_valid(3))
    assert_equal(r[0].value(), 2)
    assert_equal(r[1].value(), 1)
    assert_equal(r[3].value(), 3)


def test_array_length_sliced_with_nulls() raises:
    var full = _lists_with_null()  # [[1,2],[3],null,[4,5,6]]
    var a = full.slice(2, 2)  # [null, [4,5,6]]
    var r = ArrayLengthKernel.apply(a)
    assert_equal(r.null_count(), 1)
    assert_false(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_equal(r[1].value(), 3)


def _maps() raises -> MapArray:
    """[{"a": 1}, {}, {"b": 2, "c": 3}]."""
    var b = MapBuilder(map_(DynType(string), DynType(int64)))
    var entries_any = b.entries()
    ref entries = entries_any.as_struct()
    var keys_any = entries.field_builder(0)
    var values_any = entries.field_builder(1)
    ref keys = keys_any.as_string()
    ref values = values_any.as_int64()

    keys.append("a")
    values.append(1)
    entries.append_valid()
    b.append_valid()  # {"a": 1}

    b.append_valid()  # {}

    keys.append("b")
    values.append(2)
    entries.append_valid()
    keys.append("c")
    values.append(3)
    entries.append_valid()
    b.append_valid()  # {"b": 2, "c": 3}

    return b.finish()


def test_array_length_of_map() raises:
    # A map is physically a list of (key, value) entries, so its length is its
    # entry count. This dispatches only because `is_list_like()` admits map,
    # matching `ListLikeType`, which `MapType` conforms to.
    var r = ArrayLengthKernel.apply(_maps())
    assert_equal(r.null_count(), 0)
    assert_equal(r[0].value(), 1)
    assert_equal(r[1].value(), 0)
    assert_equal(r[2].value(), 2)


def test_array_contains_finds_the_row_value() raises:
    # [[10], [], [20, 30]] searched for 10 / 99 / 30.
    var r = ArrayContainsKernel.apply(
        _lists_no_null(), array([10, 99, 30], int64)
    )
    assert_equal(r.null_count(), 0)
    assert_true(r[0].value())
    assert_false(r[1].value())
    assert_true(r[2].value())


def test_array_contains_rejects_a_mismatched_element_type() raises:
    """A search value of a different type than the list's child **raises**.

    `dispatch` reads `V` off the element array and `T` off the list's offset
    width; neither looks at the child. Without the guard in `apply` this is
    `as_primitive[Int32Type]()` on an `int64` child -- a bad variant access,
    which aborts the process instead of raising, taking the whole test binary
    with it rather than failing one case.
    """
    var raised = False
    try:
        _ = ArrayContainsKernel.apply(
            _lists_no_null(), array([10, 99, 30], int32)
        )
    except e:
        raised = True
        assert_true("does not match" in String(e), String(e))
    assert_true(raised, "a mismatched element type must raise")


# ---------------------------------------------------------------------------
# ListReduceKernel
# ---------------------------------------------------------------------------


def _step(state: UInt64) -> UInt64:
    return state * 6364136223846793005 + 1442695040888963407


def _random_lists[
    T: ListLikeType, V: NumericType
](n: Int) raises -> ListLikeArray[T]:
    """`n` lists of 0 to 64 elements, about one element in five null and one
    list in ten null; a null list still spans its elements."""
    comptime N = V.native
    var b = ListLikeBuilder[T](PrimitiveBuilder[V]())
    var child_any = b.values()
    ref child = child_any.as_primitive[V]()
    var state = UInt64(42)
    for _ in range(n):
        state = _step(state)
        var size = Int(state >> 33) % 65
        for _ in range(size):
            state = _step(state)
            var r = Int(state >> 33)
            if r % 5 == 0:
                child.append_null()
            else:
                child.append(Scalar[N](r % 201 - 100))
        state = _step(state)
        if Int(state >> 33) % 10 == 0:
            b.append_null()
        else:
            b.append_valid()
    return b.finish()


def _check_list_reduce[F: FoldKernel, V: NumericType](lists: ListArray) raises:
    """Each row against a scalar fold of its list's valid elements."""
    comptime A = F.AccType[V].native
    var got = ListReduceKernel[F].apply[V](lists)
    assert_equal(len(got), len(lists))
    ref child = lists.values().as_primitive[V]()
    var offsets = lists.offsets.view[DType.int32](lists.offset)
    for i in range(len(lists)):
        var at = String(F.name, " ", V(), " row ", i)
        var acc = F.identity[A]()
        var count = 0
        for j in range(Int(offsets[i]), Int(offsets[i + 1])):
            if child.is_valid(j):
                acc = F.combine[A, 1](acc, child.unsafe_get(j).cast[A]())
                count += 1
        if not lists.is_valid(i) or (count == 0 and F.empty_is_null):
            assert_false(got.is_valid(i), at)
            assert_equal(got.unsafe_get(i), Scalar[A](0), at + " payload")
        else:
            assert_true(got.is_valid(i), at)
            assert_equal(got.unsafe_get(i), F.finalize[A](acc, count), at)


def _check_list_folds[V: NumericType]() raises:
    var lists = _random_lists[ListType, V](400)
    _check_list_reduce[SumFold, V](lists)
    _check_list_reduce[MinFold, V](lists)
    _check_list_reduce[MaxFold, V](lists)
    _check_list_reduce[MeanFold, V](lists)
    _check_list_reduce[CountFold, V](lists)


def test_list_reduce_matches_reference() raises:
    _check_list_folds[Int32Type]()
    _check_list_folds[Int64Type]()
    _check_list_folds[Float64Type]()


def test_list_reduce_sliced_list() raises:
    var sliced = _lists_with_null().slice(1, 3)  # [[3], null, [4, 5, 6]]
    assert_equal(sliced.offsets.view[DType.int32](sliced.offset)[0], 2)
    var sums = ListReduceKernel[SumFold].apply[Int64Type](sliced)
    assert_true(sums == array([3, None, 15], int64))


def test_list_reduce_large_list() raises:
    var lists = _random_lists[ListType, Int64Type](100)
    var large = _random_lists[LargeListType, Int64Type](100)
    assert_true(
        ListReduceKernel[SumFold].apply[Int64Type](large)
        == ListReduceKernel[SumFold].apply[Int64Type](lists)
    )
    assert_true(
        ListReduceKernel[MeanFold].apply[Int64Type](large)
        == ListReduceKernel[MeanFold].apply[Int64Type](lists)
    )


def test_list_reduce_dispatch() raises:
    var b = ListBuilder(Int32Builder())
    var child_any = b.values()
    ref child = child_any.as_int32()
    child.append(1)
    child.append(2)
    b.append_valid()
    var lists: DynArray = b.finish()
    var sums = ListReduceKernel[SumFold].dispatch(lists)
    assert_true(sums.as_int64() == array([3], int64))

    var s = ListBuilder(StringBuilder())
    var strings_any = s.values()
    ref strings = strings_any.as_string()
    strings.append("a")
    s.append_valid()
    with assert_raises(contains="not numeric"):
        _ = ListReduceKernel[SumFold].dispatch(s.finish())

    with assert_raises(contains="expected a list array"):
        _ = ListReduceKernel[SumFold].dispatch(array([1, 2], int64))


def test_list_reduce_bitwise_folds() raises:
    """DuckDB's `list_bit_and`, `list_bit_or` and `list_bit_xor`: the element
    type kept, and an empty, all-null or null list null."""
    var b = ListBuilder(Int64Builder())
    var child_any = b.values()
    ref child = child_any.as_int64()
    child.append(12)
    child.append(10)
    child.append(14)
    b.append_valid()  # [12, 10, 14]
    b.append_valid()  # []
    child.append_null()
    b.append_valid()  # [null]
    b.append_null()  # null
    child.append(5)
    child.append_null()
    child.append(3)
    b.append_valid()  # [5, null, 3]
    child.append(-1)
    child.append(6)
    b.append_valid()  # [-1, 6]
    var lists: DynArray = b.finish()
    assert_true(
        ListReduceKernel[BitAndFold].dispatch(lists).as_int64()
        == array([8, None, None, None, 1, 6], int64)
    )
    assert_true(
        ListReduceKernel[BitOrFold].dispatch(lists).as_int64()
        == array([14, None, None, None, 7, -1], int64)
    )
    assert_true(
        ListReduceKernel[BitXorFold].dispatch(lists).as_int64()
        == array([8, None, None, None, 6, -7], int64)
    )

    var narrow = ListBuilder(Int8Builder())
    var narrow_any = narrow.values()
    ref bytes = narrow_any.as_int8()
    bytes.append(1)
    bytes.append(3)
    narrow.append_valid()
    var anded = ListReduceKernel[BitAndFold].dispatch(narrow.finish())
    assert_true(anded.as_int8() == array([1], int8))

    var floats = ListBuilder(Float64Builder())
    var floats_any = floats.values()
    ref values = floats_any.as_float64()
    values.append(1.0)
    floats.append_valid()
    with assert_raises(contains="not an integer type"):
        _ = ListReduceKernel[BitAndFold].dispatch(floats.finish())
