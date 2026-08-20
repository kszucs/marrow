import std.math as math

from std.bit import count_trailing_zeros, pop_count
from std.testing import assert_equal, assert_true, assert_false

from ..buffers import Buffer, Bitmap
from ..views import BufferView, BitmapView, reduce


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


def test_bufferview_compressed_store_dense_stays_in_bounds() raises:
    """The dense (branchless) path must not write past ``popcount`` elements.

    ``compressed_store_dense`` stores *before* it knows whether the lane is
    kept, so every lane above the highest set bit lands on element
    ``popcount(sel_bits)`` — one past the packed output. When the destination
    view is exactly ``popcount`` elements long that is an out-of-bounds write,
    and marrow's buffers carry no slack whenever the byte size is already a
    multiple of 64 (``Buffer._aligned_size`` aligns, it does not pad). This
    guards the case with a sentinel element the store must never reach.
    """
    var src_buf = Buffer.alloc_zeroed[DType.int64](64)
    var src = src_buf.view[DType.int64]()
    for i in range(64):
        src_buf.unsafe_set[DType.int64](i, Int64(i + 1))

    # 32 of 64 lanes set, all in the low half: over the sparse threshold (so
    # the dense path runs) and with every high lane clear (so the trailing
    # stores pile onto element 32).
    var sel_bits = UInt64(0x0000_0000_FFFF_FFFF)
    var cnt = Int(pop_count(sel_bits))

    var dst_buf = Buffer.alloc_zeroed[DType.int64](cnt + 1)
    dst_buf.unsafe_set[DType.int64](cnt, Int64(-99))  # sentinel
    # A view exactly as long as the packed output — what `BufferView.filter`
    # hands to the last selection word of a filter.
    var dst = dst_buf.view[DType.int64](0, cnt)

    var n = dst.compressed_store(src, sel_bits)
    assert_equal(n, cnt)
    for i in range(cnt):
        assert_equal(dst[i], Int64(i + 1))
    assert_equal(dst_buf.view[DType.int64]().unsafe_get(cnt), Int64(-99))


def test_bufferview_filter_last_word_stays_in_bounds() raises:
    """``BufferView.filter`` packs correctly when the final word is dense.

    The end-to-end shape of the overflow above — a selection whose *final*
    64-bit word has popcount over the sparse threshold with its top lanes
    clear — which is the case that now falls back to the sparse path. Values,
    not memory safety: the one-past store wrote *correct* data one element too
    far, so only the sentinel test above can catch the regression.
    """
    var n = 128
    var src_buf = Buffer.alloc_zeroed[DType.int64](n)
    for i in range(n):
        src_buf.unsafe_set[DType.int64](i, Int64(i))
    var src = src_buf.view[DType.int64]()

    # Word 0: 32 low lanes set. Word 1: 32 low lanes set. 64 selected in all,
    # so the output is 64 * 8 == 512 bytes — a multiple of 64, hence no slack.
    var sel = Bitmap.alloc_zeroed(n)
    for i in range(32):
        sel.set(i)
        sel.set(64 + i)
    var sel_view = sel.view()
    var out_len, sel_start, sel_end = sel_view.count_set_bits_with_range()
    assert_equal(out_len, 64)

    var out = src.filter(sel_view, sel_start, sel_end, out_len)
    var out_view = out.view[DType.int64](0, out_len)
    for i in range(32):
        assert_equal(out_view[i], Int64(i))
        assert_equal(out_view[32 + i], Int64(64 + i))


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
    var m = view.load[8](0)
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


def test_bitmapview_load_at_the_last_byte_stays_in_bounds() raises:
    """`load[W]` must not read past the allocation at a bitmap's tail.

    512 bits is exactly 64 bytes, and `Buffer._aligned_size` rounds *up* to 64
    — so this allocation ends at the last live byte and has no padding at all.
    `load[W]` takes an unconditional 4-byte `UInt32`, so a load addressed at
    the final byte wants bytes 63..66 of a 64-byte allocation.

    The same hole opens whenever the byte extent is 62, 63 or 0 mod 64; a
    nullable 150,000-element column (18,750 bytes, 62 mod 64) is the shape that
    found it, through the masked `apply` lane.
    """
    var bm = Bitmap.alloc_zeroed(512)
    bm.set(504)
    bm.set(511)
    var view = bm.view(0, 512)
    var bits = view.load[8](504)
    assert_true(bits[0])
    assert_false(bits[1])
    assert_true(bits[7])


def test_bitmapview_load_at_a_62_mod_64_extent() raises:
    """The other failing residue class, and the one seen in the wild.

    A bitmap's byte extent has only ``(-extent) mod 64`` bytes of padding, and
    `load[W]` wants 3. So 62, 63 and 0 mod 64 all fall short. 150,000 bits is
    18,750 bytes, 62 mod 64, leaving 2 — one byte short.
    """
    var bits = 150_000
    var bm = Bitmap.alloc_zeroed(bits)
    bm.set(bits - 1)
    var view = bm.view(0, bits)
    var tail = view.load[8](bits - 8)
    assert_true(tail[7])
    assert_false(tail[0])


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


def test_views_bitmapview_eq_equal() raises:
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


def test_views_bitmapview_invert() raises:
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
    var count, start, end = bm.view(0, 64).count_set_bits_with_range()
    assert_equal(count, 3)
    assert_true(start <= 5)
    assert_true(end >= 61)


def test_bitmapview_count_set_bits_with_range_zero() raises:
    var bm = Bitmap.alloc_zeroed(64)
    var count, start, end = bm.view(0, 64).count_set_bits_with_range()
    assert_equal(count, 0)
    assert_equal(start, 0)
    assert_equal(end, 0)


def test_bitmapview_count_set_bits_with_range_empty() raises:
    var bm = Bitmap.alloc_zeroed(8)
    var count, start, end = bm.view(0, 0).count_set_bits_with_range()
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


# ---------------------------------------------------------------------------
# reduce — combine helpers
# ---------------------------------------------------------------------------


@always_inline
def _add_i32[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a + b


@always_inline
def _max_i32[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return math.max(a, b)


# ---------------------------------------------------------------------------
# reduce — plain BufferView (no bitmap)
# ---------------------------------------------------------------------------


def test_reduce_sum_plain() raises:
    """`reduce` sums all elements when there is no validity bitmap."""
    var buf = Buffer.alloc_zeroed[DType.int32](5)
    for i in range(5):
        buf.unsafe_set[DType.int32](i, Int32(i + 1))  # [1,2,3,4,5]
    var view = buf.view[DType.int32]()
    var result = reduce[DType.int32, _add_i32](view, Int32(0))
    assert_equal(result, Int32(15))


def test_reduce_max_plain() raises:
    """`reduce` finds the maximum element."""
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    buf.unsafe_set[DType.int32](0, Int32(3))
    buf.unsafe_set[DType.int32](1, Int32(7))
    buf.unsafe_set[DType.int32](2, Int32(1))
    buf.unsafe_set[DType.int32](3, Int32(5))
    var view = buf.view[DType.int32]()
    var result = reduce[DType.int32, _max_i32](view, Int32.MIN_FINITE)
    assert_equal(result, Int32(7))


def test_reduce_empty_returns_identity() raises:
    """`reduce` on an empty view returns the identity value."""
    var buf = Buffer.alloc_zeroed[DType.int32](0)
    var view = buf.view[DType.int32](0, 0)
    var result = reduce[DType.int32, _add_i32](view, Int32(42))
    assert_equal(result, Int32(42))


# ---------------------------------------------------------------------------
# reduce — bitmap-masked BufferView
# ---------------------------------------------------------------------------


def test_reduce_sum_with_bitmap() raises:
    """`reduce` with a bitmap skips null (False) positions."""
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    buf.unsafe_set[DType.int32](0, Int32(10))
    buf.unsafe_set[DType.int32](1, Int32(20))  # null
    buf.unsafe_set[DType.int32](2, Int32(30))
    buf.unsafe_set[DType.int32](3, Int32(40))  # null
    var bm = Bitmap.alloc_zeroed(4)
    bm.set(0)
    bm.set(2)
    var result = reduce[DType.int32, _add_i32](
        buf.view[DType.int32](), bm.view(), Int32(0)
    )
    assert_equal(result, Int32(40))


def test_reduce_all_null_returns_identity() raises:
    """`reduce` with all-null bitmap returns the identity value."""
    var buf = Buffer.alloc_zeroed[DType.int32](3)
    buf.unsafe_set[DType.int32](0, Int32(99))
    buf.unsafe_set[DType.int32](1, Int32(99))
    buf.unsafe_set[DType.int32](2, Int32(99))
    var bm = Bitmap.alloc_zeroed(3)  # all bits clear = all null
    var result = reduce[DType.int32, _add_i32](
        buf.view[DType.int32](), bm.view(), Int32(0)
    )
    assert_equal(result, Int32(0))


# ---------------------------------------------------------------------------
# View bounds matrix
#
# A computed destination size meeting an unchecked write is what produced the
# heap overflow in `compressed_store_dense` (docs/alpha-findings/f1-*.md), and
# it cost ~4 hours to trace because the crash named an innocent allocation.
# Four tests is a regression guard, not coverage; these walk the axes that
# decide whether the overflow is reachable — selection shape, the sparse/dense
# threshold, element width, destination slack, word count — and check a
# sentinel past the destination on every cell.
# ---------------------------------------------------------------------------


def _probe_compressed_store[
    T: DType
](sel_bits: UInt64, slack: Int, label: String) raises:
    """One (dtype, selection, slack) cell of the compressed-store matrix.

    Packs a 64-element source holding ``1..64`` through the adaptive
    `BufferView.compressed_store` into a destination of ``popcount + slack``
    elements, with a guard element immediately past it. Checks the packed
    values *and* the guard: the dense path stores a lane before consulting its
    selection bit, so every lane above the highest set bit lands on element
    ``popcount``, which is out of bounds when ``slack`` is 0.
    """
    var cnt = Int(pop_count(sel_bits))
    var dst_len = cnt + slack

    var src_buf = Buffer.alloc_zeroed[T](64)
    for i in range(64):
        src_buf.unsafe_set[T](i, Scalar[T](i + 1))
    var src = src_buf.view[T](0, 64)

    var dst_buf = Buffer.alloc_zeroed[T](dst_len + 1)
    dst_buf.unsafe_set[T](dst_len, Scalar[T](99))
    var dst = dst_buf.view[T](0, dst_len)

    var n = dst.compressed_store(src, sel_bits)
    assert_true(n == cnt, String(label, ": returned ", n, ", want ", cnt))

    var w = sel_bits
    var k = 0
    while w != 0:
        var lane = Int(count_trailing_zeros(w))
        var got = dst_buf.unsafe_get[T](k)
        assert_true(
            got == Scalar[T](lane + 1),
            String(label, ": element ", k, " is ", got, ", want ", lane + 1),
        )
        w &= w - 1
        k += 1

    var guard = dst_buf.unsafe_get[T](dst_len)
    assert_true(
        guard == Scalar[T](99),
        String(label, ": wrote past the destination — guard is ", guard),
    )


def _selection_words() -> List[UInt64]:
    """Selection shapes that decide whether the dense path oversteps.

    The two degenerate ends never could: an all-ones word writes its last lane
    at index 63 *before* consuming bit 63, and a word whose highest bit is set
    likewise has a real home for the trailing store. It is the middle — dense
    enough to take the branchless path, with the top lanes clear — that lands
    on element ``popcount``.
    """
    var words: List[UInt64] = [
        UInt64(0),  # nothing selected
        ~UInt64(0),  # everything selected
        UInt64(1),  # only the lowest lane
        UInt64(1) << 63,  # only the highest lane
        UInt64(1) << 37,  # a single interior lane
        UInt64(0x0000_0000_FFFF_FFFF),  # low half — F1's shape
        UInt64(0x5555_5555_5555_5555),  # alternating, top lane clear
        UInt64(0xAAAA_AAAA_AAAA_AAAA),  # alternating, top lane set
        UInt64(0x0000_FFFF_FFFF_FFFF),  # 48 set, top 16 clear
        UInt64(0x00FF_FFFF_0000_0000),  # a dense interior run
    ]
    return words^


def test_bounds_compressed_store_selection_patterns() raises:
    """Every selection shape, at zero and non-zero destination slack."""
    var words = _selection_words()
    for i in range(len(words)):
        _probe_compressed_store[DType.int64](
            words[i], 0, String("int64 word#", i, " slack=0")
        )
        _probe_compressed_store[DType.int64](
            words[i], 1, String("int64 word#", i, " slack=1")
        )


def test_bounds_compressed_store_sparse_dense_threshold() raises:
    """The 24-bit sparse/dense boundary is behaviour-critical after the fix.

    At or below it the CTZ path runs and never oversteps; above it the
    branchless path runs and needs a slack element, so `compressed_store` has
    to demote to sparse when the destination has none. Pin 23 / 24 / 25 with
    the top lanes clear, which is the shape the trailing store escapes on.
    """
    for bits in range(23, 26):
        var sel = (UInt64(1) << UInt64(bits)) - 1
        _probe_compressed_store[DType.int64](
            sel, 0, String("threshold ", bits, " slack=0")
        )
        _probe_compressed_store[DType.int64](
            sel, 1, String("threshold ", bits, " slack=1")
        )


def test_bounds_compressed_store_element_widths() raises:
    """Element width decides how many elements fit a 64-byte block, and so
    whether a destination sized to the popcount has any slack at all — the
    original defect needed an ``int64`` length divisible by 8."""
    var breaking = UInt64(0x0000_0000_FFFF_FFFF)
    var alternating = UInt64(0x5555_5555_5555_5555)
    var full = ~UInt64(0)
    _probe_compressed_store[DType.int8](breaking, 0, "int8 low-half")
    _probe_compressed_store[DType.int16](breaking, 0, "int16 low-half")
    _probe_compressed_store[DType.int32](breaking, 0, "int32 low-half")
    _probe_compressed_store[DType.int64](breaking, 0, "int64 low-half")
    _probe_compressed_store[DType.float64](breaking, 0, "float64 low-half")
    _probe_compressed_store[DType.int8](alternating, 0, "int8 alternating")
    _probe_compressed_store[DType.int16](alternating, 0, "int16 alternating")
    _probe_compressed_store[DType.int32](alternating, 0, "int32 alternating")
    _probe_compressed_store[DType.int64](alternating, 0, "int64 alternating")
    _probe_compressed_store[DType.float64](
        alternating, 0, "float64 alternating"
    )
    _probe_compressed_store[DType.int8](full, 0, "int8 all-ones")
    _probe_compressed_store[DType.int32](full, 0, "int32 all-ones")
    _probe_compressed_store[DType.float64](full, 0, "float64 all-ones")


def _probe_filter_zero_slack[T: DType](label: String) raises:
    """`BufferView.filter` over two selection words, each with its low 32
    lanes set: 64 elements out, so the destination is 64 / 128 / 256 / 512
    bytes — always a 64-byte multiple, hence zero slack, which is the
    condition that turned the dense path's one-past store into a heap
    overflow rather than a harmless scribble on padding."""
    var n = 128
    var src_buf = Buffer.alloc_zeroed[T](n)
    for i in range(n):
        src_buf.unsafe_set[T](i, Scalar[T](i % 100))
    var src = src_buf.view[T](0, n)

    var sel = Bitmap.alloc_zeroed(n)
    for i in range(32):
        sel.set(i)
        sel.set(64 + i)
    var sel_view = sel.view()
    var out_len, sel_start, sel_end = sel_view.count_set_bits_with_range()
    assert_true(out_len == 64, String(label, ": out_len is ", out_len))

    var out = src.filter(sel_view, sel_start, sel_end, out_len)
    var out_view = out.view[T](0, out_len)
    for i in range(32):
        assert_true(
            out_view[i] == Scalar[T](i % 100),
            String(label, ": low word element ", i),
        )
        assert_true(
            out_view[32 + i] == Scalar[T]((64 + i) % 100),
            String(label, ": high word element ", i),
        )


def test_bounds_filter_zero_slack_all_widths() raises:
    """The multi-word end-to-end shape, at every element width."""
    _probe_filter_zero_slack[DType.int8]("filter int8")
    _probe_filter_zero_slack[DType.int16]("filter int16")
    _probe_filter_zero_slack[DType.int32]("filter int32")
    _probe_filter_zero_slack[DType.int64]("filter int64")
    _probe_filter_zero_slack[DType.float64]("filter float64")


# --- BitmapView ------------------------------------------------------------


def test_bounds_bitmapview_compressed_store_zero_slack() raises:
    """Deposit into the last byte of a bitmap that has no slack behind it.

    512 bits is exactly 64 bytes, and `Buffer._aligned_size` rounds up to a
    64-byte multiple — which 64 already is — so the allocation ends at the
    bitmap's final byte. `BitmapView.compressed_store` used to store 8 bytes
    unconditionally and justify it with "Arrow buffers are 64-byte padded",
    so this shape read-modify-wrote 7 bytes past the allocation.
    """
    var bm = Bitmap.alloc_zeroed(512)
    var view = bm.view()
    view.compressed_store(504, UInt64(0b1011_0011), 8)
    for i in range(8):
        assert_true(
            view.test(504 + i) == Bool((0b1011_0011 >> i) & 1),
            String("bit ", 504 + i),
        )
    for i in range(504):
        assert_false(view.test(i), String("spill at bit ", i))


def test_bounds_bitmapview_compressed_store_guard_bytes() raises:
    """A deposit into a view's final byte must leave the bytes after it alone.

    The view spans bits 0..511 (bytes 0..63) of a 1024-bit bitmap whose bits
    512..575 are all set. Those bits are the guard; the wide store used to
    read-modify-write straight through them.
    """
    var bm = Bitmap.alloc_zeroed(1024)
    for i in range(512, 576):
        bm.set(i)
    var view = bm.view(0, 512)
    view.compressed_store(505, UInt64(0b101), 3)
    assert_true(view.test(505), "bit 505")
    assert_false(view.test(506), "bit 506")
    assert_true(view.test(507), "bit 507")
    for i in range(512, 576):
        assert_true(bm.test(i), String("guard bit ", i, " was cleared"))


def test_bounds_bitmapview_compressed_store_nine_byte_straddle() raises:
    """``count == 64`` at a non-zero bit offset genuinely needs a ninth byte —
    the one case where the wide path is the right answer."""
    var bm = Bitmap.alloc_zeroed(128)
    var view = bm.view()
    var bits = UInt64(0xF0F0_F0F0_0F0F_0F0F)
    view.compressed_store(3, bits, 64)
    for i in range(64):
        assert_true(
            view.test(3 + i) == Bool((bits >> UInt64(i)) & 1),
            String("straddle bit ", i),
        )
    assert_false(view.test(0), "spill below the deposit")
    assert_false(view.test(67), "spill above the deposit")


def test_bounds_bitmapview_compressed_store_all_bit_offsets() raises:
    """Every bit offset 0..7 crossed with counts 1..16.

    This is the tapered narrow path (``nbytes < 8``), which is new code and is
    exactly what the final block of a filter takes. Walk it exhaustively
    rather than sampling, and assert nothing outside the deposited run is set.
    """
    for off in range(8):
        for count in range(1, 17):
            var bm = Bitmap.alloc_zeroed(128)
            var view = bm.view()
            var bits = UInt64(0xA53C_96D2_4B7E_1F08) & (
                (UInt64(1) << UInt64(count)) - 1
            )
            view.compressed_store(off, bits, count)
            for i in range(128):
                var want = False
                if off <= i and i < off + count:
                    want = Bool((bits >> UInt64(i - off)) & 1)
                assert_true(
                    view.test(i) == want,
                    String("off=", off, " count=", count, " bit ", i),
                )


def test_bounds_bitmapview_compressed_store_ragged_length() raises:
    """A bitmap whose bit length is not a whole number of bytes.

    517 bits occupy 65 bytes with the last holding 5 live bits, so the view's
    byte extent rounds up to 65 and byte 65 is off-limits — the boundary the
    tapered store has to respect.
    """
    var bm = Bitmap.alloc_zeroed(517)
    var view = bm.view()
    view.compressed_store(512, UInt64(0b10101), 5)
    for i in range(5):
        assert_true(
            view.test(512 + i) == Bool((0b10101 >> i) & 1),
            String("ragged bit ", 512 + i),
        )
    for i in range(512):
        assert_false(view.test(i), String("ragged spill at ", i))


def test_bounds_bitmapview_filter_zero_slack_output() raises:
    """`BitmapView.filter` producing exactly 512 bits — 64 bytes, no slack.

    Every 64-bit selection word keeps its low half, so the run-merge paths are
    skipped and every block goes through `pext` + `compressed_store`, ending
    on the destination's very last byte.
    """
    var n = 1024
    var data = Bitmap.alloc_zeroed(n)
    for i in range(0, n, 3):
        data.set(i)
    var sel = Bitmap.alloc_zeroed(n)
    for w in range(16):
        for i in range(32):
            sel.set(w * 64 + i)

    var sel_view = sel.view()
    var out_len, sel_start, sel_end = sel_view.count_set_bits_with_range()
    assert_true(out_len == 512, String("out_len is ", out_len))

    var filtered, zeros = data.view().filter(
        sel_view, sel_start, sel_end, out_len
    )
    var fv = filtered.view()
    var k = 0
    var expect_zeros = 0
    for i in range(n):
        if sel.test(i):
            assert_true(fv.test(k) == data.test(i), String("filtered bit ", k))
            if not data.test(i):
                expect_zeros += 1
            k += 1
    assert_true(k == out_len, String("wrote ", k, " bits, want ", out_len))
    assert_true(zeros == expect_zeros, String("zero_count is ", zeros))


def test_bounds_bitmapview_load_bits_with_bit_offset() raises:
    """`load_bits[uint64]` must answer for all 64 bits of a shifted view.

    A view over a sliced array carries a sub-byte ``_offset``; the read is a
    single unaligned 8-byte load shifted down by that offset, so the top
    ``_offset`` bits of the result come from the ninth byte — the one the load
    did not fetch.
    """
    var bm = Bitmap.alloc_zeroed(256)
    for i in range(256):
        if (i * 7) % 5 < 2:
            bm.set(i)
    var view = bm.view(4, 128)
    var w = view.load_bits[DType.uint64](0)
    for i in range(64):
        assert_true(
            Bool((w >> UInt64(i)) & 1) == view.test(i),
            String("offset load_bits bit ", i),
        )


def test_bounds_bitmapview_filter_with_bit_offset() raises:
    """Filter through views that carry a sub-byte bit offset — the shape a
    sliced array hands to `Filter.apply`.

    Both the selection and the data are read with `load_bits`, so an offset
    that truncates the top of each 64-bit word drops rows from the answer.
    """
    var n = 300
    var data = Bitmap.alloc_zeroed(n + 8)
    var sel = Bitmap.alloc_zeroed(n + 8)
    for i in range(n + 8):
        if (i * 11) % 7 < 3:
            data.set(i)
        if (i * 5) % 4 < 2:
            sel.set(i)

    var data_view = data.view(5, n)
    var sel_view = sel.view(5, n)
    var out_len, sel_start, sel_end = sel_view.count_set_bits_with_range()

    var expected = 0
    for i in range(n):
        if sel_view.test(i):
            expected += 1
    assert_true(
        out_len == expected,
        String("out_len is ", out_len, ", want ", expected),
    )

    var filtered, zeros = data_view.filter(
        sel_view, sel_start, sel_end, out_len
    )
    var fv = filtered.view()
    var k = 0
    var expect_zeros = 0
    for i in range(n):
        if sel_view.test(i):
            assert_true(
                fv.test(k) == data_view.test(i),
                String("offset filter bit ", k),
            )
            if not data_view.test(i):
                expect_zeros += 1
            k += 1
    assert_true(k == out_len, String("wrote ", k, " bits, want ", out_len))
    assert_true(zeros == expect_zeros, String("zero_count is ", zeros))


def test_bounds_bufferview_filter_with_bit_offset() raises:
    """The same offset shape through `BufferView.filter`, which reads the
    selection with the same `load_bits`."""
    var n = 300
    var src_buf = Buffer.alloc_zeroed[DType.int32](n)
    for i in range(n):
        src_buf.unsafe_set[DType.int32](i, Int32(i))
    var src = src_buf.view[DType.int32](0, n)

    var sel = Bitmap.alloc_zeroed(n + 8)
    for i in range(n + 8):
        if (i * 5) % 4 < 2:
            sel.set(i)
    var sel_view = sel.view(5, n)
    var out_len, sel_start, sel_end = sel_view.count_set_bits_with_range()

    var out = src.filter(sel_view, sel_start, sel_end, out_len)
    var out_view = out.view[DType.int32](0, out_len)
    var k = 0
    for i in range(n):
        if sel_view.test(i):
            assert_true(
                out_view[k] == Int32(i),
                String("offset filter element ", k, " is ", out_view[k]),
            )
            k += 1
    assert_true(k == out_len, String("kept ", k, ", want ", out_len))
