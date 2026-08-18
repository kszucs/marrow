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
    StringArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
    ChunkedArray,
)
from ...builders import (
    array,
    arange,
    PrimitiveBuilder,
    BinaryLikeBuilder,
    StringBuilder,
    ListBuilder,
    FixedSizeListBuilder,
    StructBuilder,
    Int32Builder,
    Float32Builder,
)
from ...dtypes import *
from ...kernels.concat import concat


# ---------------------------------------------------------------------------
# concat — primitive arrays
# ---------------------------------------------------------------------------


def test_concat_primitive() raises:
    var arrs: List[DynArray] = [
        array([1, 2], int32),
        array([3, 4, 5], int32),
    ]
    var tmp = concat(arrs)
    ref result = tmp.as_int32()
    assert_equal(result.length, 5)
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 2)
    assert_equal(result[2].value(), 3)
    assert_equal(result[3].value(), 4)
    assert_equal(result[4].value(), 5)


def test_concat_single() raises:
    var arrs: List[DynArray] = [array([10, 20, 30], int32)]
    var tmp = concat(arrs)
    ref result = tmp.as_int32()
    assert_equal(result.length, 3)
    assert_equal(result[0].value(), 10)
    assert_equal(result[2].value(), 30)


def test_concat_empty_list_raises() raises:
    var arrs = List[DynArray]()
    with assert_raises():
        _ = concat(arrs)


def test_concat_with_nulls() raises:
    var b1 = Int32Builder()
    b1.append(1)
    b1.append_null()
    b1.append(3)
    var b2 = Int32Builder()
    b2.append(4)
    b2.append_null()
    var arrs: List[DynArray] = [
        b1.finish().to_dyn(),
        b2.finish().to_dyn(),
    ]
    var tmp_with_nulls = concat(arrs)
    ref result = tmp_with_nulls.as_int32()
    assert_equal(result.length, 5)
    assert_equal(result.null_count(), 2)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_true(result.is_valid(3))
    assert_false(result.is_valid(4))
    assert_equal(result[0].value(), 1)
    assert_equal(result[2].value(), 3)
    assert_equal(result[3].value(), 4)


def test_concat_with_offset() raises:
    # arange [0..4], take slice [1..3] (offset=1, length=3) and [4] (offset=4)
    var a = arange[Int32Type](0, 5)
    var s1 = a.slice(1, 3)  # [1, 2, 3], offset=1
    var s2 = a.slice(4, 1)  # [4], offset=4
    var arrs: List[DynArray] = [s1^, s2^]
    var tmp_offset = concat(arrs)
    ref result = tmp_offset.as_int32()
    assert_equal(result.length, 4)
    assert_equal(result[0].value(), 1)
    assert_equal(result[1].value(), 2)
    assert_equal(result[2].value(), 3)
    assert_equal(result[3].value(), 4)


def test_concat_with_offset_and_nulls() raises:
    # Build [1, null, 3], take slice [null, 3] (offset=1)
    var b = Int32Builder()
    b.append(1)
    b.append_null()
    b.append(3)
    var sliced = b.finish().slice(1, 2)  # [null, 3], offset=1
    var arrs: List[DynArray] = [
        (sliced^).to_dyn(),
        array([4], int32),
    ]
    var tmp_offset_nulls = concat(arrs)
    ref result = tmp_offset_nulls.as_int32()
    assert_equal(result.length, 3)
    assert_equal(result.null_count(), 1)
    assert_false(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_equal(result[1].value(), 3)
    assert_equal(result[2].value(), 4)


# ---------------------------------------------------------------------------
# concat — bool arrays
# ---------------------------------------------------------------------------


def test_concat_bool() raises:
    var arrs: List[DynArray] = [
        array([True, False, True]).to_dyn(),
        array([False, True]).to_dyn(),
    ]
    var tmp_bool = concat(arrs)
    ref result = tmp_bool.as_bool()
    assert_equal(result.length, 5)
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_true(result[2].value())
    assert_false(result[3].value())
    assert_true(result[4].value())


def test_concat_bool_with_offset() raises:
    # [True, False, True, False] slice at offset=1 → [False, True, False]
    var a = array([True, False, True, False])
    var sliced = a.slice(1, 3)
    var arrs: List[DynArray] = [(sliced^).to_dyn(), array([True]).to_dyn()]
    var tmp_bool_offset = concat(arrs)
    ref result = tmp_bool_offset.as_bool()
    assert_equal(result.length, 4)
    assert_false(result[0].value())
    assert_true(result[1].value())
    assert_false(result[2].value())
    assert_true(result[3].value())


# ---------------------------------------------------------------------------
# concat — string arrays
# ---------------------------------------------------------------------------


def test_concat_string() raises:
    var s1 = StringBuilder()
    s1.append("hello")
    s1.append("world")
    var s2 = StringBuilder()
    s2.append("foo")
    var arrs: List[DynArray] = [
        s1.finish().to_dyn(),
        s2.finish().to_dyn(),
    ]
    var tmp_str = concat(arrs)
    ref result = tmp_str.as_string()
    assert_equal(result.length, 3)
    assert_equal(result[0].to_string(), "hello")
    assert_equal(result[1].to_string(), "world")
    assert_equal(result[2].to_string(), "foo")


def test_concat_string_with_nulls() raises:
    var s1 = StringBuilder()
    s1.append("a")
    s1.append_null()
    s1.append("b")
    var s2 = StringBuilder()
    s2.append("c")
    var arrs: List[DynArray] = [
        s1.finish().to_dyn(),
        s2.finish().to_dyn(),
    ]
    var tmp_str_nulls = concat(arrs)
    ref result = tmp_str_nulls.as_string()
    assert_equal(result.length, 4)
    assert_equal(result.null_count(), 1)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_true(result.is_valid(3))
    assert_equal(result[0].to_string(), "a")
    assert_equal(result[2].to_string(), "b")
    assert_equal(result[3].to_string(), "c")


# ---------------------------------------------------------------------------
# concat — list arrays
# ---------------------------------------------------------------------------


def test_concat_list() raises:
    # Chunk 1: [[1, 2], [3]]
    var lb1 = ListBuilder(Int32Builder(), capacity=2)
    var c1_any = lb1.values()
    ref c1 = c1_any.as_int32()
    c1.append(1)
    c1.append(2)
    lb1.append_valid()  # [1, 2]
    c1.append(3)
    lb1.append_valid()  # [3]
    # Chunk 2: [[4, 5, 6]]
    var lb2 = ListBuilder(Int32Builder(), capacity=1)
    var c2_any = lb2.values()
    ref c2 = c2_any.as_int32()
    c2.append(4)
    c2.append(5)
    c2.append(6)
    lb2.append_valid()  # [4, 5, 6]
    var arrs: List[DynArray] = [
        lb1.finish().to_dyn(),
        lb2.finish().to_dyn(),
    ]
    var tmp_list = concat(arrs)
    ref result = tmp_list.as_list()
    assert_equal(result.length, 3)
    var raw_elem0 = result[0].value()
    ref elem0 = raw_elem0.as_int32()
    assert_equal(elem0.length, 2)
    assert_equal(elem0[0].value(), 1)
    assert_equal(elem0[1].value(), 2)
    var raw_elem1 = result[1].value()
    ref elem1 = raw_elem1.as_int32()
    assert_equal(elem1.length, 1)
    assert_equal(elem1[0].value(), 3)
    var raw_elem2 = result[2].value()
    ref elem2 = raw_elem2.as_int32()
    assert_equal(elem2.length, 3)
    assert_equal(elem2[0].value(), 4)
    assert_equal(elem2[2].value(), 6)


def test_concat_list_with_nulls() raises:
    var lb1 = ListBuilder(Int32Builder(), capacity=2)
    var c1_any = lb1.values()
    ref c1 = c1_any.as_int32()
    c1.append(1)
    lb1.append_valid()  # [1]
    lb1.append_null()  # null
    var lb2 = ListBuilder(Int32Builder(), capacity=1)
    var c2_any = lb2.values()
    ref c2 = c2_any.as_int32()
    c2.append(2)
    c2.append(3)
    lb2.append_valid()  # [2, 3]
    var arrs: List[DynArray] = [
        lb1.finish().to_dyn(),
        lb2.finish().to_dyn(),
    ]
    var tmp_list_nulls = concat(arrs)
    ref result = tmp_list_nulls.as_list()
    assert_equal(result.length, 3)
    assert_equal(result.null_count(), 1)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    var raw_elem0 = result[0].value()
    ref elem0 = raw_elem0.as_int32()
    assert_equal(elem0[0].value(), 1)
    var raw_elem2 = result[2].value()
    ref elem2 = raw_elem2.as_int32()
    assert_equal(elem2.length, 2)


# ---------------------------------------------------------------------------
# concat — fixed-size list arrays
# ---------------------------------------------------------------------------


def test_concat_fixed_size_list() raises:
    # Chunk 1: [[1.0, 2.0], [3.0, 4.0]]
    var child1 = Float32Builder()
    child1.append(1.0)
    child1.append(2.0)
    child1.append(3.0)
    child1.append(4.0)
    var fsl1 = FixedSizeListBuilder(child1^, list_size=2)
    fsl1.append_valid()
    fsl1.append_valid()
    # Chunk 2: [[5.0, 6.0]]
    var child2 = Float32Builder()
    child2.append(5.0)
    child2.append(6.0)
    var fsl2 = FixedSizeListBuilder(child2^, list_size=2)
    fsl2.append_valid()
    var arrs: List[DynArray] = [
        fsl1.finish().to_dyn(),
        fsl2.finish().to_dyn(),
    ]
    var tmp_fsl = concat(arrs)
    ref result = tmp_fsl.as_fixed_size_list()
    assert_equal(result.length, 3)
    var raw_fsl_elem0 = result[0].value()
    ref elem0 = raw_fsl_elem0.as_float32()
    assert_equal(elem0[0].value(), 1.0)
    assert_equal(elem0[1].value(), 2.0)
    var raw_fsl_elem1 = result[1].value()
    ref elem1 = raw_fsl_elem1.as_float32()
    assert_equal(elem1[0].value(), 3.0)
    var raw_fsl_elem2 = result[2].value()
    ref elem2 = raw_fsl_elem2.as_float32()
    assert_equal(elem2[0].value(), 5.0)
    assert_equal(elem2[1].value(), 6.0)


def test_concat_fixed_size_list_with_offset() raises:
    # Build [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], then slice at offset=1
    var child = Float32Builder()
    child.append(1.0)
    child.append(2.0)
    child.append(3.0)
    child.append(4.0)
    child.append(5.0)
    child.append(6.0)
    var fsl = FixedSizeListBuilder(child^, list_size=2)
    fsl.append_valid()
    fsl.append_valid()
    fsl.append_valid()
    var sliced = fsl.finish().slice(1, 2)  # [[3.0, 4.0], [5.0, 6.0]]
    var child2 = Float32Builder()
    child2.append(7.0)
    child2.append(8.0)
    var fsl2 = FixedSizeListBuilder(child2^, list_size=2)
    fsl2.append_valid()
    var arrs: List[DynArray] = [
        (sliced^).to_dyn(),
        fsl2.finish().to_dyn(),
    ]
    var tmp_fsl_offset = concat(arrs)
    ref result = tmp_fsl_offset.as_fixed_size_list()
    assert_equal(result.length, 3)
    var raw_fsl_off_elem0 = result[0].value()
    ref elem0 = raw_fsl_off_elem0.as_float32()
    assert_equal(elem0[0].value(), 3.0)
    assert_equal(elem0[1].value(), 4.0)
    var raw_fsl_off_elem1 = result[1].value()
    ref elem1 = raw_fsl_off_elem1.as_float32()
    assert_equal(elem1[0].value(), 5.0)
    var raw_fsl_off_elem2 = result[2].value()
    ref elem2 = raw_fsl_off_elem2.as_float32()
    assert_equal(elem2[0].value(), 7.0)
    assert_equal(elem2[1].value(), 8.0)


# ---------------------------------------------------------------------------
# concat — struct arrays
# ---------------------------------------------------------------------------


def test_concat_struct() raises:
    # Chunk 1: [{id:1, score:0.5}, {id:2, score:0.6}]
    var sb1 = StructBuilder([field("id", int32), field("score", float32)])
    sb1.field_builder(0).as_int32().append(1)
    sb1.field_builder(0).as_int32().append(2)
    sb1.field_builder(1).as_float32().append(0.5)
    sb1.field_builder(1).as_float32().append(0.6)
    sb1.append_valid()
    sb1.append_valid()
    # Chunk 2: [{id:3, score:0.7}]
    var sb2 = StructBuilder([field("id", int32), field("score", float32)])
    sb2.field_builder(0).as_int32().append(3)
    sb2.field_builder(1).as_float32().append(0.7)
    sb2.append_valid()
    var arrs: List[DynArray] = [
        sb1.finish().to_dyn(),
        sb2.finish().to_dyn(),
    ]
    var tmp_struct = concat(arrs)
    ref result = tmp_struct.as_struct()
    assert_equal(result.length, 3)
    ref id_data = result.unsafe_get("id")
    ref id_arr = id_data.as_int32()
    assert_equal(id_arr[0].value(), 1)
    assert_equal(id_arr[1].value(), 2)
    assert_equal(id_arr[2].value(), 3)
    ref score_data = result.unsafe_get("score")
    ref score_arr = score_data.as_float32()
    assert_equal(score_arr[0].value(), 0.5)
    assert_equal(score_arr[2].value(), 0.7)


# ---------------------------------------------------------------------------
# combine_chunks delegates to concat
# ---------------------------------------------------------------------------


def test_combine_chunks_delegates() raises:
    var chunks: List[DynArray] = [
        array([10, 20], int32),
        array([30], int32),
        array([40, 50], int32),
    ]
    var ca = ChunkedArray(int32, chunks^)
    var combined = ca^.combine_chunks()
    ref result = combined.as_int32()
    assert_equal(result.length, 5)
    assert_equal(result[0].value(), 10)
    assert_equal(result[1].value(), 20)
    assert_equal(result[2].value(), 30)
    assert_equal(result[3].value(), 40)
    assert_equal(result[4].value(), 50)


# ---------------------------------------------------------------------------
# concat — binary / large_binary
#
# `concat` is entirely `DynBuilder.extend`-driven, so it shared the abort that
# surfaced through the thread-local group-by: `BinaryLikeBuilder`'s erased
# `extend` resolved the *source* array type from the builder's offset width and
# named a `stringlike` type for every 32-bit-offset builder. Nothing here had
# binary coverage, which is how it shipped.
# ---------------------------------------------------------------------------


def _bytes_array[T: BinaryLikeType](values: List[String]) raises -> DynArray:
    var b = BinaryLikeBuilder[T](len(values))
    for v in values:
        b.append(v)
    var out: DynArray = b.finish()
    return out^


def _assert_bytes_concat[T: BinaryLikeType]() raises:
    var arrs: List[DynArray] = [
        _bytes_array[T](["a", "bb"]),
        _bytes_array[T](["ccc"]),
        _bytes_array[T](["", "dddd"]),
    ]
    var result = concat(arrs)
    ref r = result.as_binary_like[T]()
    assert_equal(r.length, 5)
    assert_equal(r[0].to_string(), "a")
    assert_equal(r[1].to_string(), "bb")
    assert_equal(r[2].to_string(), "ccc")
    assert_equal(r[3].to_string(), "")
    assert_equal(r[4].to_string(), "dddd")
    assert_true(result.dtype() == T().to_dyn())


def test_concat_binary() raises:
    _assert_bytes_concat[BinaryType]()


def test_concat_large_binary() raises:
    _assert_bytes_concat[LargeBinaryType]()


def test_concat_binary_with_nulls() raises:
    var a = BinaryLikeBuilder[BinaryType](2)
    a.append("x")
    a.append_null()
    var b = BinaryLikeBuilder[BinaryType](2)
    b.append_null()
    b.append("y")
    var arrs: List[DynArray] = [a.finish(), b.finish()]
    var result = concat(arrs)
    ref r = result.as_binary_like[BinaryType]()
    assert_equal(r.length, 4)
    assert_equal(r.null_count(), 2)
    assert_equal(r[0].to_string(), "x")
    assert_false(r.is_valid(1))
    assert_false(r.is_valid(2))
    assert_equal(r[3].to_string(), "y")


def test_combine_chunks_binary() raises:
    """`ChunkedArray.combine_chunks` routes through `concat`, so a binary
    column in a `Table` aborted on combine too."""
    var chunks: List[DynArray] = [
        _bytes_array[BinaryType](["p", "q"]),
        _bytes_array[BinaryType](["r"]),
    ]
    var ca = ChunkedArray(binary, chunks^)
    var combined = ca^.combine_chunks()
    ref r = combined.as_binary_like[BinaryType]()
    assert_equal(r.length, 3)
    assert_equal(r[0].to_string(), "p")
    assert_equal(r[1].to_string(), "q")
    assert_equal(r[2].to_string(), "r")
