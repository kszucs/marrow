from std.testing import assert_equal, assert_true, assert_false, TestSuite

from marrow.buffers import Buffer, Bitmap
from marrow.views import BufferView, BitmapView


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
    """slice() creates a sub-view with the offset summed."""
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
    """view() with a non-zero offset reads bits correctly."""
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
