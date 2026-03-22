from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
    TestSuite,
)

from marrow.arrays import (
    array,
    arange,
    AnyArray,
    PrimitiveArray,
    BoolArray,
    StringArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
    ChunkedArray,
)
from marrow.builders import (
    AnyBuilder,
    PrimitiveBuilder,
    StringBuilder,
    ListBuilder,
    FixedSizeListBuilder,
    StructBuilder,
)
from marrow.dtypes import *
from marrow.kernels.concat import concat


# ---------------------------------------------------------------------------
# concat — primitive arrays
# ---------------------------------------------------------------------------


def test_concat_primitive() raises:
    var arrs: List[AnyArray] = [array[int32]([1, 2]), array[int32]([3, 4, 5])]
    var result = concat(arrs).as_primitive[int32]()
    assert_equal(result.length, 5)
    assert_equal(result[0], 1)
    assert_equal(result[1], 2)
    assert_equal(result[2], 3)
    assert_equal(result[3], 4)
    assert_equal(result[4], 5)


def test_concat_single() raises:
    var arrs: List[AnyArray] = [array[int32]([10, 20, 30])]
    var result = concat(arrs).as_primitive[int32]()
    assert_equal(result.length, 3)
    assert_equal(result[0], 10)
    assert_equal(result[2], 30)


def test_concat_empty_list_raises() raises:
    var arrs = List[AnyArray]()
    with assert_raises():
        _ = concat(arrs)


def test_concat_with_nulls() raises:
    var b1 = PrimitiveBuilder[int32]()
    b1.append(1)
    b1.append_null()
    b1.append(3)
    var b2 = PrimitiveBuilder[int32]()
    b2.append(4)
    b2.append_null()
    var arrs: List[AnyArray] = [
        AnyArray(b1.finish_typed()),
        AnyArray(b2.finish_typed()),
    ]
    var result = concat(arrs).as_primitive[int32]()
    assert_equal(result.length, 5)
    assert_equal(result.null_count(), 2)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_true(result.is_valid(3))
    assert_false(result.is_valid(4))
    assert_equal(result[0], 1)
    assert_equal(result[2], 3)
    assert_equal(result[3], 4)


def test_concat_with_offset() raises:
    # arange [0..4], take slice [1..3] (offset=1, length=3) and [4] (offset=4)
    var a = arange[int32](0, 5)
    var s1 = a.slice(1, 3)  # [1, 2, 3], offset=1
    var s2 = a.slice(4, 1)  # [4], offset=4
    var arrs: List[AnyArray] = [AnyArray(s1), AnyArray(s2)]
    var result = concat(arrs).as_primitive[int32]()
    assert_equal(result.length, 4)
    assert_equal(result[0], 1)
    assert_equal(result[1], 2)
    assert_equal(result[2], 3)
    assert_equal(result[3], 4)


def test_concat_with_offset_and_nulls() raises:
    # Build [1, null, 3], take slice [null, 3] (offset=1)
    var b = PrimitiveBuilder[int32]()
    b.append(1)
    b.append_null()
    b.append(3)
    var sliced = b.finish_typed().slice(1, 2)  # [null, 3], offset=1
    var arrs: List[AnyArray] = [AnyArray(sliced), AnyArray(array[int32]([4]))]
    var result = concat(arrs).as_primitive[int32]()
    assert_equal(result.length, 3)
    assert_equal(result.null_count(), 1)
    assert_false(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_equal(result[1], 3)
    assert_equal(result[2], 4)


# ---------------------------------------------------------------------------
# concat — bool arrays
# ---------------------------------------------------------------------------


def test_concat_bool() raises:
    var arrs: List[AnyArray] = [
        AnyArray(array([True, False, True])),
        AnyArray(array([False, True])),
    ]
    var result = concat(arrs).as_primitive[bool_]()
    assert_equal(result.length, 5)
    assert_true(result[0].__bool__())
    assert_false(result[1].__bool__())
    assert_true(result[2].__bool__())
    assert_false(result[3].__bool__())
    assert_true(result[4].__bool__())


def test_concat_bool_with_offset() raises:
    # [True, False, True, False] slice at offset=1 → [False, True, False]
    var a = array([True, False, True, False])
    var sliced = a.slice(1, 3)
    var arrs: List[AnyArray] = [AnyArray(sliced), AnyArray(array([True]))]
    var result = concat(arrs).as_primitive[bool_]()
    assert_equal(result.length, 4)
    assert_false(result[0].__bool__())
    assert_true(result[1].__bool__())
    assert_false(result[2].__bool__())
    assert_true(result[3].__bool__())


# ---------------------------------------------------------------------------
# concat — string arrays
# ---------------------------------------------------------------------------


def test_concat_string() raises:
    var s1 = StringBuilder()
    s1.append("hello")
    s1.append("world")
    var s2 = StringBuilder()
    s2.append("foo")
    var arrs: List[AnyArray] = [
        AnyArray(s1.finish_typed()),
        AnyArray(s2.finish_typed()),
    ]
    var result = StringArray(data=concat(arrs))
    assert_equal(result.length, 3)
    assert_equal(result[0], "hello")
    assert_equal(result[1], "world")
    assert_equal(result[2], "foo")


def test_concat_string_with_nulls() raises:
    var s1 = StringBuilder()
    s1.append("a")
    s1.append_null()
    s1.append("b")
    var s2 = StringBuilder()
    s2.append("c")
    var arrs: List[AnyArray] = [
        AnyArray(s1.finish_typed()),
        AnyArray(s2.finish_typed()),
    ]
    var result = StringArray(data=concat(arrs))
    assert_equal(result.length, 4)
    assert_equal(result.null_count(), 1)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_true(result.is_valid(3))
    assert_equal(result[0], "a")
    assert_equal(result[2], "b")
    assert_equal(result[3], "c")


# ---------------------------------------------------------------------------
# concat — list arrays
# ---------------------------------------------------------------------------


def test_concat_list() raises:
    # Chunk 1: [[1, 2], [3]]
    var lb1 = ListBuilder(AnyBuilder(PrimitiveBuilder[int32]()), capacity=2)
    var c1 = lb1.values().as_primitive[int32]()
    c1[].append(1)
    c1[].append(2)
    lb1.append_valid()  # [1, 2]
    c1[].append(3)
    lb1.append_valid()  # [3]
    # Chunk 2: [[4, 5, 6]]
    var lb2 = ListBuilder(AnyBuilder(PrimitiveBuilder[int32]()), capacity=1)
    var c2 = lb2.values().as_primitive[int32]()
    c2[].append(4)
    c2[].append(5)
    c2[].append(6)
    lb2.append_valid()  # [4, 5, 6]
    var arrs: List[AnyArray] = [
        AnyArray(lb1.finish_typed()),
        AnyArray(lb2.finish_typed()),
    ]
    var result = ListArray(data=concat(arrs))
    assert_equal(result.length, 3)
    var elem0 = result[0].as_primitive[int32]()
    assert_equal(elem0.length, 2)
    assert_equal(elem0[0], 1)
    assert_equal(elem0[1], 2)
    var elem1 = result[1].as_primitive[int32]()
    assert_equal(elem1.length, 1)
    assert_equal(elem1[0], 3)
    var elem2 = result[2].as_primitive[int32]()
    assert_equal(elem2.length, 3)
    assert_equal(elem2[0], 4)
    assert_equal(elem2[2], 6)


def test_concat_list_with_nulls() raises:
    var lb1 = ListBuilder(AnyBuilder(PrimitiveBuilder[int32]()), capacity=2)
    var c1 = lb1.values().as_primitive[int32]()
    c1[].append(1)
    lb1.append_valid()  # [1]
    lb1.append_null()  # null
    var lb2 = ListBuilder(AnyBuilder(PrimitiveBuilder[int32]()), capacity=1)
    var c2 = lb2.values().as_primitive[int32]()
    c2[].append(2)
    c2[].append(3)
    lb2.append_valid()  # [2, 3]
    var arrs: List[AnyArray] = [
        AnyArray(lb1.finish_typed()),
        AnyArray(lb2.finish_typed()),
    ]
    var result = ListArray(data=concat(arrs))
    assert_equal(result.length, 3)
    assert_equal(result.null_count(), 1)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    var elem0 = result[0].as_primitive[int32]()
    assert_equal(elem0[0], 1)
    var elem2 = result[2].as_primitive[int32]()
    assert_equal(elem2.length, 2)


# ---------------------------------------------------------------------------
# concat — fixed-size list arrays
# ---------------------------------------------------------------------------


def test_concat_fixed_size_list() raises:
    # Chunk 1: [[1.0, 2.0], [3.0, 4.0]]
    var child1 = PrimitiveBuilder[float32]()
    child1.append(1.0)
    child1.append(2.0)
    child1.append(3.0)
    child1.append(4.0)
    var fsl1 = FixedSizeListBuilder(child1^, list_size=2)
    fsl1.unsafe_append_valid()
    fsl1.unsafe_append_valid()
    # Chunk 2: [[5.0, 6.0]]
    var child2 = PrimitiveBuilder[float32]()
    child2.append(5.0)
    child2.append(6.0)
    var fsl2 = FixedSizeListBuilder(child2^, list_size=2)
    fsl2.unsafe_append_valid()
    var arrs: List[AnyArray] = [
        AnyArray(fsl1.finish_typed()),
        AnyArray(fsl2.finish_typed()),
    ]
    var result = FixedSizeListArray(data=concat(arrs))
    assert_equal(result.length, 3)
    var elem0 = result[0].as_primitive[float32]()
    assert_equal(elem0[0], 1.0)
    assert_equal(elem0[1], 2.0)
    var elem1 = result[1].as_primitive[float32]()
    assert_equal(elem1[0], 3.0)
    var elem2 = result[2].as_primitive[float32]()
    assert_equal(elem2[0], 5.0)
    assert_equal(elem2[1], 6.0)


def test_concat_fixed_size_list_with_offset() raises:
    # Build [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], then slice at offset=1
    var child = PrimitiveBuilder[float32]()
    child.append(1.0)
    child.append(2.0)
    child.append(3.0)
    child.append(4.0)
    child.append(5.0)
    child.append(6.0)
    var fsl = FixedSizeListBuilder(child^, list_size=2)
    fsl.unsafe_append_valid()
    fsl.unsafe_append_valid()
    fsl.unsafe_append_valid()
    var sliced = fsl.finish_typed().slice(1, 2)  # [[3.0, 4.0], [5.0, 6.0]]
    var child2 = PrimitiveBuilder[float32]()
    child2.append(7.0)
    child2.append(8.0)
    var fsl2 = FixedSizeListBuilder(child2^, list_size=2)
    fsl2.unsafe_append_valid()
    var arrs: List[AnyArray] = [AnyArray(sliced), AnyArray(fsl2.finish_typed())]
    var result = FixedSizeListArray(data=concat(arrs))
    assert_equal(result.length, 3)
    var elem0 = result[0].as_primitive[float32]()
    assert_equal(elem0[0], 3.0)
    assert_equal(elem0[1], 4.0)
    var elem1 = result[1].as_primitive[float32]()
    assert_equal(elem1[0], 5.0)
    var elem2 = result[2].as_primitive[float32]()
    assert_equal(elem2[0], 7.0)
    assert_equal(elem2[1], 8.0)


# ---------------------------------------------------------------------------
# concat — struct arrays
# ---------------------------------------------------------------------------


def test_concat_struct() raises:
    # Chunk 1: [{id:1, score:0.5}, {id:2, score:0.6}]
    var id1 = PrimitiveBuilder[int32]()
    id1.append(1)
    id1.append(2)
    var score1 = PrimitiveBuilder[float32]()
    score1.append(0.5)
    score1.append(0.6)
    var fields: List[Field] = [Field("id", int32), Field("score", float32)]
    var children1: List[AnyBuilder] = [id1^, score1^]
    var sb1 = StructBuilder(fields.copy(), children1^)
    sb1.append_valid()
    sb1.append_valid()
    # Chunk 2: [{id:3, score:0.7}]
    var id2 = PrimitiveBuilder[int32]()
    id2.append(3)
    var score2 = PrimitiveBuilder[float32]()
    score2.append(0.7)
    var children2: List[AnyBuilder] = [id2^, score2^]
    var sb2 = StructBuilder(fields^, children2^)
    sb2.append_valid()
    var arrs: List[AnyArray] = [
        AnyArray(sb1.finish_typed()),
        AnyArray(sb2.finish_typed()),
    ]
    var result = StructArray(data=concat(arrs))
    assert_equal(result.length, 3)
    ref id_data = result.unsafe_get("id")
    var id_arr = id_data.as_primitive[int32]()
    assert_equal(id_arr[0], 1)
    assert_equal(id_arr[1], 2)
    assert_equal(id_arr[2], 3)
    ref score_data = result.unsafe_get("score")
    var score_arr = score_data.as_primitive[float32]()
    assert_equal(score_arr[0], 0.5)
    assert_equal(score_arr[2], 0.7)


# ---------------------------------------------------------------------------
# combine_chunks delegates to concat
# ---------------------------------------------------------------------------


def test_combine_chunks_delegates() raises:
    var chunks: List[AnyArray] = [
        array[int32]([10, 20]),
        array[int32]([30]),
        array[int32]([40, 50]),
    ]
    var ca = ChunkedArray(int32, chunks^)
    var combined = ca^.combine_chunks()
    var result = combined.as_primitive[int32]()
    assert_equal(result.length, 5)
    assert_equal(result[0], 10)
    assert_equal(result[1], 20)
    assert_equal(result[2], 30)
    assert_equal(result[3], 40)
    assert_equal(result[4], 50)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
