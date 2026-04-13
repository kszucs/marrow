from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite

from marrow.buffers import Buffer, Bitmap
from marrow.views import BufferView, BitmapView


@always_inline
def _inc_int32[W: Int](v: SIMD[DType.int32, W]) -> SIMD[DType.int32, W]:
    return v + Int32(1)


@always_inline
def _nonzero_int32[W: Int](v: SIMD[DType.int32, W]) -> SIMD[DType.bool, W]:
    return v.cast[DType.bool]()


# ---------------------------------------------------------------------------
# BufferView — construction and element access
# ---------------------------------------------------------------------------


def test_bufferview_len() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    var view = buf.view[DType.int32]()
    assert_equal(len(view), len(buf) // 4)


def test_bufferview_getitem() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    buf.unsafe_set[DType.int32](0, Int32(10))
    buf.unsafe_set[DType.int32](1, Int32(20))
    buf.unsafe_set[DType.int32](2, Int32(30))
    buf.unsafe_set[DType.int32](3, Int32(40))
    var view = buf.view[DType.int32](0)
    assert_equal(view[0], 10)
    assert_equal(view[1], 20)
    assert_equal(view[2], 30)
    assert_equal(view[3], 40)


def test_bufferview_bool_nonempty() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    var view = buf.view[DType.int32]()
    assert_true(view.__bool__())


def test_bufferview_bool_empty() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](0)
    var view = buf.view[DType.int32]()
    assert_false(view.__bool__())


def test_bufferview_contains() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    buf.unsafe_set[DType.int32](0, Int32(7))
    buf.unsafe_set[DType.int32](1, Int32(42))
    var view = buf.view[DType.int32]()
    assert_true(42 in view)
    assert_false(99 in view)


def test_bufferview_slice() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](8)
    for i in range(8):
        buf.unsafe_set[DType.int32](i, Int32(i * 10))
    var view = buf.view[DType.int32]()
    var sub = view.slice(2, 3)
    assert_equal(len(sub), 3)
    assert_equal(sub[0], 20)
    assert_equal(sub[1], 30)
    assert_equal(sub[2], 40)


def test_bufferview_load() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](8)
    for i in range(8):
        buf.unsafe_set[DType.int32](i, Int32(i + 1))
    var view = buf.view[DType.int32]()
    var v = view.load[4](0)
    assert_equal(v[0], 1)
    assert_equal(v[1], 2)
    assert_equal(v[2], 3)
    assert_equal(v[3], 4)


def test_bufferview_store() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](8)
    var view = buf.view[DType.int32](0)
    view.store[4](0, SIMD[DType.int32, 4](5, 6, 7, 8))
    assert_equal(view[0], 5)
    assert_equal(view[1], 6)
    assert_equal(view[2], 7)
    assert_equal(view[3], 8)


def test_bufferview_element_access() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    buf.unsafe_set[DType.int32](0, 99)
    var view = buf.view[DType.int32]()
    assert_equal(view[0], 99)


def test_bufferview_offset_baked_in() raises:
    """View with offset baked into the pointer starts at the right element."""
    var buf = Buffer.alloc_zeroed[DType.int32](8)
    for i in range(8):
        buf.unsafe_set[DType.int32](i, Int32(i * 2))
    var view = buf.view[DType.int32](3)
    assert_equal(view[0], 6)
    assert_equal(view[1], 8)


# ---------------------------------------------------------------------------
# BufferView — TrivialRegisterPassable (implicit copy)
# ---------------------------------------------------------------------------


def test_bufferview_implicit_copy() raises:
    """TrivialRegisterPassable: copy is a memcpy of the two fields."""
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    buf.unsafe_set[DType.int32](0, Int32(11))
    buf.unsafe_set[DType.int32](1, Int32(22))
    var original = buf.view[DType.int32]()
    var copy = original  # implicit copy via TrivialRegisterPassable
    assert_equal(copy[0], 11)
    assert_equal(copy[1], 22)
    assert_equal(len(copy), len(original))


# ---------------------------------------------------------------------------
# BufferView — DevicePassable
# ---------------------------------------------------------------------------


def test_bufferview_get_type_name() raises:
    assert_equal(
        BufferView[DType.int32, ImmutAnyOrigin].get_type_name(),
        "BufferView[int32]",
    )


def test_bufferview_get_type_name_float() raises:
    assert_equal(
        BufferView[DType.float64, ImmutAnyOrigin].get_type_name(),
        "BufferView[float64]",
    )


# ---------------------------------------------------------------------------
# BitmapView — construction and bit access
# ---------------------------------------------------------------------------


def test_bitmapview_len() raises:
    var bm = Bitmap.alloc_zeroed(10)
    var view = bm.view(0, 10)
    assert_equal(len(view), 10)


def test_bitmapview_test() raises:
    var bm = Bitmap.alloc_zeroed(8)
    bm.set(0)
    bm.set(3)
    bm.set(7)
    var view = bm.view(0, 8)
    assert_true(view.test(0))
    assert_false(view.test(1))
    assert_false(view.test(2))
    assert_true(view.test(3))
    assert_true(view.test(7))


def test_bitmapview_getitem() raises:
    var bm = Bitmap.alloc_zeroed(8)
    bm.set(2)
    bm.set(5)
    var view = bm.view(0, 8)
    assert_false(view[0])
    assert_false(view[1])
    assert_true(view[2])
    assert_false(view[3])
    assert_true(view[5])


def test_bitmapview_bool_any_set() raises:
    var bm = Bitmap.alloc_zeroed(8)
    var view = bm.view(0, 8)
    assert_false(Bool(view))
    bm.set(4)
    assert_true(Bool(bm.view(0, 8)))


def test_bitmapview_bit_offset() raises:
    var bm = Bitmap.alloc_zeroed(16)
    var view = bm.view(5, 8)
    assert_equal(view.bit_offset(), 5)


def test_bitmapview_slice() raises:
    """`slice()` creates a sub-view with the offset summed."""
    var bm = Bitmap.alloc_zeroed(16)
    bm.set(4)
    bm.set(5)
    bm.set(6)
    var view = bm.view(0, 16)
    var sub = view.slice(4, 3)
    assert_equal(len(sub), 3)
    assert_true(sub.test(0))
    assert_true(sub.test(1))
    assert_true(sub.test(2))


def test_bitmapview_getitem_slice() raises:
    """BitmapView[slice] returns a sub-view."""
    var bm = Bitmap.alloc_zeroed(16)
    bm.set(2)
    bm.set(3)
    var view = bm.view(0, 16)
    var sub = view[2:4]
    assert_equal(len(sub), 2)
    assert_true(sub[0])
    assert_true(sub[1])


def test_bitmapview_with_offset() raises:
    """`view()` with a non-zero offset reads bits correctly."""
    var bm = Bitmap.alloc_zeroed(16)
    bm.set(8)
    bm.set(9)
    var view = bm.view(8, 4)
    assert_true(view[0])
    assert_true(view[1])
    assert_false(view[2])
    assert_false(view[3])


def test_bitmapview_count_set_bits() raises:
    var bm = Bitmap.alloc_zeroed(16)
    bm.set(1)
    bm.set(5)
    bm.set(10)
    var view = bm.view(0, 16)
    assert_equal(view.count_set_bits(), 3)


def test_bitmapview_all_set_true() raises:
    var bm = Bitmap.alloc_zeroed(4)
    bm.set_range(0, 4, True)
    assert_true(bm.view(0, 4).all_set())


def test_bitmapview_all_set_false() raises:
    var bm = Bitmap.alloc_zeroed(4)
    bm.set(0)
    bm.set(1)
    assert_false(bm.view(0, 4).all_set())


# ---------------------------------------------------------------------------
# BitmapView — TrivialRegisterPassable (implicit copy)
# ---------------------------------------------------------------------------


def test_bitmapview_implicit_copy() raises:
    """TrivialRegisterPassable: copy carries pointer, offset, and length."""
    var bm = Bitmap.alloc_zeroed(8)
    bm.set(2)
    bm.set(6)
    var original = bm.view(0, 8)
    var copy = original  # implicit copy via TrivialRegisterPassable
    assert_equal(len(copy), 8)
    assert_equal(copy.bit_offset(), 0)
    assert_true(copy[2])
    assert_true(copy[6])
    assert_false(copy[0])


# ---------------------------------------------------------------------------
# BitmapView — DevicePassable
# ---------------------------------------------------------------------------


def test_bitmapview_get_type_name() raises:
    assert_equal(BitmapView[ImmutAnyOrigin].get_type_name(), "BitmapView")


# ---------------------------------------------------------------------------
# BufferView — additional coverage
# ---------------------------------------------------------------------------


def test_bufferview_getitem_slice() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](6)
    for i in range(6):
        buf.unsafe_set[DType.int32](i, Int32(i * 3))
    var view = buf.view[DType.int32]()
    var sub = view[2:5]
    assert_equal(len(sub), 3)
    assert_equal(sub[0], 6)
    assert_equal(sub[1], 9)
    assert_equal(sub[2], 12)


def test_bufferview_unsafe_get() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    buf.unsafe_set[DType.int32](2, Int32(77))
    var view = buf.view[DType.int32]()
    assert_equal(view.unsafe_get(2), 77)


def test_bufferview_unsafe_set() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    var view = buf.view[DType.int32](0)
    view.unsafe_set(1, Int32(55))
    assert_equal(view[1], 55)


def test_bufferview_gather() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](8)
    for i in range(8):
        buf.unsafe_set[DType.int32](i, Int32(i * 10))
    var view = buf.view[DType.int32]()
    var offsets = SIMD[DType.int64, 4](0, 3, 5, 7)
    var result = view.gather[4](offsets)
    assert_equal(result[0], 0)
    assert_equal(result[1], 30)
    assert_equal(result[2], 50)
    assert_equal(result[3], 70)


def test_bufferview_compressed_store_llvm() raises:
    """Writes only masked lanes via compressed_store[W](value, mask)."""
    var buf = Buffer.alloc_zeroed[DType.int32](8)
    var view = buf.view[DType.int32](0)
    var values = SIMD[DType.int32, 4](10, 20, 30, 40)
    var mask = SIMD[DType.bool, 4](True, False, True, False)
    view.compressed_store[4](values, mask)
    assert_equal(view[0], 10)
    assert_equal(view[1], 30)


def test_bufferview_compressed_store_adaptive() raises:
    """Adaptive compressed_store(src, sel_bits) dispatch returns popcount."""
    var src_buf = Buffer.alloc_zeroed[DType.int32](8)
    for i in range(8):
        src_buf.unsafe_set[DType.int32](i, Int32(i + 1))
    var dst_buf = Buffer.alloc_zeroed[DType.int32](8)
    var src = src_buf.view[DType.int32]()
    var dst = dst_buf.view[DType.int32](0)
    # select bits 0, 2, 4 → elements 1, 3, 5
    var sel_bits = UInt64(0b00010101)
    var n = dst.compressed_store(src, sel_bits)
    assert_equal(n, 3)
    assert_equal(dst[0], 1)
    assert_equal(dst[1], 3)
    assert_equal(dst[2], 5)


def test_bufferview_copy_from() raises:
    var src_buf = Buffer.alloc_zeroed[DType.int32](4)
    for i in range(4):
        src_buf.unsafe_set[DType.int32](i, Int32(i + 100))
    var dst_buf = Buffer.alloc_zeroed[DType.int32](4)
    var src = src_buf.view[DType.int32]()
    var dst = dst_buf.view[DType.int32](0)
    dst.copy_from(src, 4)
    assert_equal(dst[0], 100)
    assert_equal(dst[1], 101)
    assert_equal(dst[2], 102)
    assert_equal(dst[3], 103)


def test_bufferview_slice_default_length() raises:
    """Slice with no length argument extends to the end of the view."""
    var buf = Buffer.alloc_zeroed[DType.int32](6)
    for i in range(6):
        buf.unsafe_set[DType.int32](i, Int32(i))
    var view = buf.view[DType.int32](0, 6)
    var sub = view.slice(3)
    assert_equal(len(sub), 3)
    assert_equal(sub[0], 3)
    assert_equal(sub[1], 4)
    assert_equal(sub[2], 5)


def test_bufferview_apply() raises:
    """Modifies all elements in-place via apply[func] using SIMD."""
    var buf = Buffer.alloc_zeroed[DType.int32](8)
    for i in range(8):
        buf.unsafe_set[DType.int32](i, Int32(i + 1))
    var view = buf.view[DType.int32](0)
    view.apply[_inc_int32]()
    for i in range(8):
        assert_equal(view[i], Int32(i + 2))


def test_bufferview_count() raises:
    """Returns number of elements matching count[pred]."""
    var buf = Buffer.alloc_zeroed[DType.int32](8)
    for i in range(8):
        buf.unsafe_set[DType.int32](i, Int32(i))  # 0..7
    var view = buf.view[DType.int32]()
    # values 0..7: seven non-zero elements
    assert_equal(view.count[_nonzero_int32](), 7)


def test_bufferview_write_to() raises:
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    var view = buf.view[DType.int32](0, 4)
    var s = String(view)
    assert_true("BufferView" in s)
    assert_true("4" in s)


# ---------------------------------------------------------------------------
# BitmapView — mutable write operations
# ---------------------------------------------------------------------------


def test_bitmapview_set_via_view() raises:
    var bm = Bitmap.alloc_zeroed(8)
    var view = bm.view()
    view.set(3)
    view.set(7)
    assert_false(view[0])
    assert_true(view[3])
    assert_true(view[7])


def test_bitmapview_clear_via_view() raises:
    var bm = Bitmap.alloc_zeroed(8)
    bm.set(2)
    bm.set(5)
    var view = bm.view()
    view.clear(2)
    assert_false(view[2])
    assert_true(view[5])


def test_bitmapview_toggle_via_view() raises:
    var bm = Bitmap.alloc_zeroed(8)
    bm.set(4)
    var view = bm.view()
    view.toggle(4)
    assert_false(view[4])
    view.toggle(4)
    assert_true(view[4])
    view.toggle(1)
    assert_true(view[1])


# ---------------------------------------------------------------------------
# BitmapView — mask / load_bits / pext
# ---------------------------------------------------------------------------


def test_bitmapview_mask() raises:
    """Expands W consecutive bits into a SIMD[bool, W] via mask[W]."""
    var bm = Bitmap.alloc_zeroed(16)
    bm.set(0)
    bm.set(2)
    bm.set(3)
    var view = bm.view(0, 16)
    var m = view.mask[8](0)
    assert_true(m[0])
    assert_false(m[1])
    assert_true(m[2])
    assert_true(m[3])
    assert_false(m[4])


def test_bitmapview_load_bits() raises:
    """Reads raw bits at a given logical position via load_bits."""
    var bm = Bitmap.alloc_zeroed(16)
    bm.set(0)
    bm.set(1)
    var view = bm.view(0, 16)
    var bits = view.load_bits[DType.uint8](0)
    assert_equal(Int(bits) & 0b11, 0b11)


def test_bitmapview_pext() raises:
    """Extracts and packs bits at mask=1 positions via pext."""
    var bm = Bitmap.alloc_zeroed(64)
    bm.set(0)
    bm.set(2)
    bm.set(4)
    var view = bm.view(0, 64)
    # mask selects bit positions 0, 2, 4 from the first 8 bits
    var result = view.pext(0, UInt64(0b00010101))
    assert_equal(result, UInt64(0b111))


# ---------------------------------------------------------------------------
# BitmapView — equality
# ---------------------------------------------------------------------------


def test_bitmapview_eq_equal() raises:
    var bm1 = Bitmap.alloc_zeroed(16)
    var bm2 = Bitmap.alloc_zeroed(16)
    bm1.set(3)
    bm1.set(10)
    bm2.set(3)
    bm2.set(10)
    assert_true(bm1.view(0, 16) == bm2.view(0, 16))


def test_bitmapview_eq_not_equal() raises:
    var bm1 = Bitmap.alloc_zeroed(16)
    var bm2 = Bitmap.alloc_zeroed(16)
    bm1.set(3)
    bm2.set(4)
    assert_false(bm1.view(0, 16) == bm2.view(0, 16))


def test_bitmapview_eq_different_lengths() raises:
    var bm1 = Bitmap.alloc_zeroed(8)
    var bm2 = Bitmap.alloc_zeroed(16)
    assert_false(bm1.view(0, 8) == bm2.view(0, 16))


# ---------------------------------------------------------------------------
# BitmapView — set operations
# ---------------------------------------------------------------------------


def test_bitmapview_intersection() raises:
    var bm1 = Bitmap.alloc_zeroed(8)
    var bm2 = Bitmap.alloc_zeroed(8)
    bm1.set(1)
    bm1.set(3)
    bm2.set(3)
    bm2.set(5)
    var result = bm1.view(0, 8).intersection(bm2.view(0, 8))
    var v = result.view(0, 8)
    assert_false(v[1])
    assert_true(v[3])
    assert_false(v[5])


def test_bitmapview_union() raises:
    var bm1 = Bitmap.alloc_zeroed(8)
    var bm2 = Bitmap.alloc_zeroed(8)
    bm1.set(1)
    bm2.set(5)
    var result = bm1.view(0, 8).union(bm2.view(0, 8))
    var v = result.view(0, 8)
    assert_true(v[1])
    assert_true(v[5])
    assert_false(v[0])


def test_bitmapview_symmetric_difference() raises:
    var bm1 = Bitmap.alloc_zeroed(8)
    var bm2 = Bitmap.alloc_zeroed(8)
    bm1.set(1)
    bm1.set(3)
    bm2.set(3)
    bm2.set(5)
    var result = bm1.view(0, 8).symmetric_difference(bm2.view(0, 8))
    var v = result.view(0, 8)
    assert_true(v[1])
    assert_false(v[3])
    assert_true(v[5])


def test_bitmapview_difference() raises:
    var bm1 = Bitmap.alloc_zeroed(8)
    var bm2 = Bitmap.alloc_zeroed(8)
    bm1.set(1)
    bm1.set(3)
    bm2.set(3)
    var result = bm1.view(0, 8).difference(bm2.view(0, 8))
    var v = result.view(0, 8)
    assert_true(v[1])
    assert_false(v[3])


def test_bitmapview_invert() raises:
    var bm = Bitmap.alloc_zeroed(8)
    bm.set(0)
    bm.set(2)
    var result = bm.view(0, 8).__invert__()
    var v = result.view(0, 8)
    assert_false(v[0])
    assert_true(v[1])
    assert_false(v[2])
    assert_true(v[3])


def test_bitmapview_operator_and() raises:
    var bm1 = Bitmap.alloc_zeroed(8)
    var bm2 = Bitmap.alloc_zeroed(8)
    bm1.set(2)
    bm2.set(2)
    bm2.set(4)
    var result = bm1.view(0, 8) & bm2.view(0, 8)
    var v = result.view(0, 8)
    assert_true(v[2])
    assert_false(v[4])


def test_bitmapview_operator_or() raises:
    var bm1 = Bitmap.alloc_zeroed(8)
    var bm2 = Bitmap.alloc_zeroed(8)
    bm1.set(2)
    bm2.set(4)
    var result = bm1.view(0, 8) | bm2.view(0, 8)
    var v = result.view(0, 8)
    assert_true(v[2])
    assert_true(v[4])


def test_bitmapview_operator_xor() raises:
    var bm1 = Bitmap.alloc_zeroed(8)
    var bm2 = Bitmap.alloc_zeroed(8)
    bm1.set(2)
    bm1.set(4)
    bm2.set(4)
    var result = bm1.view(0, 8) ^ bm2.view(0, 8)
    var v = result.view(0, 8)
    assert_true(v[2])
    assert_false(v[4])


def test_bitmapview_operator_sub() raises:
    var bm1 = Bitmap.alloc_zeroed(8)
    var bm2 = Bitmap.alloc_zeroed(8)
    bm1.set(2)
    bm1.set(4)
    bm2.set(4)
    var result = bm1.view(0, 8) - bm2.view(0, 8)
    var v = result.view(0, 8)
    assert_true(v[2])
    assert_false(v[4])


# ---------------------------------------------------------------------------
# BitmapView — count_set_bits_with_range
# ---------------------------------------------------------------------------


def test_bitmapview_count_set_bits_with_range_nonzero() raises:
    var bm = Bitmap.alloc_zeroed(64)
    bm.set(5)
    bm.set(10)
    bm.set(60)
    count, start, end = bm.view(0, 64).count_set_bits_with_range()
    assert_equal(count, 3)
    assert_true(start <= 5)
    assert_true(end >= 61)


def test_bitmapview_count_set_bits_with_range_zero() raises:
    var bm = Bitmap.alloc_zeroed(64)
    count, start, end = bm.view(0, 64).count_set_bits_with_range()
    assert_equal(count, 0)
    assert_equal(start, 0)
    assert_equal(end, 0)


def test_bitmapview_count_set_bits_with_range_empty() raises:
    var bm = Bitmap.alloc_zeroed(8)
    count, start, end = bm.view(0, 0).count_set_bits_with_range()
    assert_equal(count, 0)
    assert_equal(start, 0)
    assert_equal(end, 0)


# ---------------------------------------------------------------------------
# BitmapView — all_set edge cases
# ---------------------------------------------------------------------------


def test_bitmapview_all_set_empty() raises:
    """Empty view is vacuously all-set."""
    var bm = Bitmap.alloc_zeroed(8)
    assert_true(bm.view(0, 0).all_set())


# ---------------------------------------------------------------------------
# BitmapView — __bool__ edge cases
# ---------------------------------------------------------------------------


def test_bitmapview_bool_empty_length() raises:
    var bm = Bitmap.alloc_zeroed(8)
    bm.set(0)
    assert_false(Bool(bm.view(0, 0)))


def test_bitmapview_bool_single_byte() raises:
    """Single-byte range with non-zero bit."""
    var bm = Bitmap.alloc_zeroed(8)
    bm.set(3)
    assert_true(Bool(bm.view(0, 8)))


# ---------------------------------------------------------------------------
# BitmapView — write_to
# ---------------------------------------------------------------------------


def test_bitmapview_write_to() raises:
    var bm = Bitmap.alloc_zeroed(8)
    var view = bm.view(2, 4)
    var s = String(view)
    assert_true("BitmapView" in s)
    assert_true("2" in s)
    assert_true("4" in s)


def main() raises:
    TestSuite.run[__functions_in_module()]()
