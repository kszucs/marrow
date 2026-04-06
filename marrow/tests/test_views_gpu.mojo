"""GPU tests: BufferView and BitmapView as DevicePassable in GPU kernels.

Each helper kernel captures a view as a *function parameter* (not a closure
capture) so that Mojo's DevicePassable mechanism transfers the struct fields
— pointer + metadata — to GPU device memory correctly.
"""

from std.algorithm.functional import elementwise
from std.gpu.host import DeviceContext, get_gpu_target
from std.sys import has_accelerator
from std.sys.info import simd_width_of
from std.testing import assert_equal, assert_true, TestSuite
from std.utils.index import IndexList

from marrow.buffers import Bitmap, Buffer
from marrow.views import BitmapView, BufferView


# ---------------------------------------------------------------------------
# BufferView GPU kernel
# ---------------------------------------------------------------------------


def _scale_by_two[
    T: DType,
    dst_o: Origin[mut=True],
](
    src: BufferView[T, _],
    dst: BufferView[T, dst_o],
    length: Int,
    ctx: DeviceContext,
) raises:
    """Read each element via BufferView, multiply by 2, store via dst (GPU).

    Both src and dst are function parameters so DevicePassable transfers
    them — pointer + length — to the GPU before elementwise launches the kernel.
    """

    @parameter
    @always_inline
    def process[
        W: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank],) -> None:
        var i = idx[0]
        dst.store[W](i, src.load[W](i) * 2)

    comptime if has_accelerator():
        comptime width = simd_width_of[T, target=get_gpu_target()]()
        elementwise[process, width, target="gpu"](length, ctx)
    else:
        raise Error("_scale_by_two: no GPU accelerator available")


# ---------------------------------------------------------------------------
# BitmapView GPU kernel
# ---------------------------------------------------------------------------


def _bits_to_bytes[
    dst_o: Origin[mut=True],
](
    bv: BitmapView[_],
    dst: BufferView[DType.uint8, dst_o],
    length: Int,
    ctx: DeviceContext,
) raises:
    """Expand each bit in bv to a UInt8 (0 or 1) in dst (GPU, scalar kernel).

    bv is a function parameter so its pointer + offset + length are
    transferred to the GPU via DevicePassable.
    """

    @parameter
    @always_inline
    def process[
        W: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank],) -> None:
        var i = idx[0]
        dst.unsafe_set(i, UInt8(1) if bv.test(i) else UInt8(0))

    comptime if has_accelerator():
        elementwise[process, 1, target="gpu"](length, ctx)
    else:
        raise Error("_bits_to_bytes: no GPU accelerator available")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bufferview_get_type_name_gpu() raises:
    """DevicePassable.get_type_name() is accessible and correct."""
    assert_equal(
        BufferView[DType.int32, ImmutAnyOrigin].get_type_name(),
        "BufferView[int32]",
    )
    assert_equal(
        BufferView[DType.float64, ImmutAnyOrigin].get_type_name(),
        "BufferView[float64]",
    )


def test_bitmapview_get_type_name_gpu() raises:
    """DevicePassable.get_type_name() is accessible and correct."""
    assert_equal(BitmapView[ImmutAnyOrigin].get_type_name(), "BitmapView")


def test_bufferview_gpu_scale() raises:
    """BufferView transferred to GPU doubles each int32 element correctly."""
    var ctx = DeviceContext()

    # Upload [1, 2, 3, 4] to the GPU
    var cpu_buf = Buffer.alloc_zeroed[DType.int32](4)
    cpu_buf.unsafe_set[DType.int32](0, Int32(1))
    cpu_buf.unsafe_set[DType.int32](1, Int32(2))
    cpu_buf.unsafe_set[DType.int32](2, Int32(3))
    cpu_buf.unsafe_set[DType.int32](3, Int32(4))
    var dev_src = cpu_buf^.to_immutable().to_device(ctx)

    # Build a BufferView backed by device memory and run the GPU kernel
    var src = dev_src.device_view[DType.int32]()

    var dev_dst = Buffer.alloc_device[DType.int32](ctx, 4)
    _scale_by_two(src, dev_dst.view[DType.int32](), 4, ctx)

    var frozen_dst = dev_dst^.to_immutable()
    assert_true(dev_src.is_device())
    assert_true(frozen_dst.is_device())

    # Download and verify [2, 4, 6, 8]
    var result = frozen_dst.to_cpu(ctx)
    assert_equal(result.unsafe_get[DType.int32](0), Int32(2))
    assert_equal(result.unsafe_get[DType.int32](1), Int32(4))
    assert_equal(result.unsafe_get[DType.int32](2), Int32(6))
    assert_equal(result.unsafe_get[DType.int32](3), Int32(8))


def test_bufferview_gpu_scale_float32() raises:
    """BufferView[Float32Type] transferred to GPU doubles each element."""
    var ctx = DeviceContext()

    var cpu_buf = Buffer.alloc_zeroed[DType.float32](4)
    cpu_buf.unsafe_set[DType.float32](0, Float32(1.0))
    cpu_buf.unsafe_set[DType.float32](1, Float32(2.5))
    cpu_buf.unsafe_set[DType.float32](2, Float32(3.0))
    cpu_buf.unsafe_set[DType.float32](3, Float32(4.5))
    var dev_src = cpu_buf^.to_immutable().to_device(ctx)

    var src = dev_src.device_view[DType.float32]()

    var dev_dst = Buffer.alloc_device[DType.float32](ctx, 4)
    _scale_by_two(src, dev_dst.view[DType.float32](), 4, ctx)

    var frozen_dst = dev_dst^.to_immutable()
    assert_true(dev_src.is_device())
    assert_true(frozen_dst.is_device())

    var result = frozen_dst.to_cpu(ctx)
    assert_true(result.unsafe_get[DType.float32](0) == Float32(2.0))
    assert_true(result.unsafe_get[DType.float32](1) == Float32(5.0))
    assert_true(result.unsafe_get[DType.float32](2) == Float32(6.0))
    assert_true(result.unsafe_get[DType.float32](3) == Float32(9.0))


def test_bitmapview_gpu_bits_to_bytes() raises:
    """BitmapView transferred to GPU expands bits to byte values correctly."""
    var ctx = DeviceContext()

    # Bits 0, 2, 4 set → expected pattern: [1, 0, 1, 0, 1, 0, 0, 0]
    var bm = Bitmap.alloc_zeroed(8)
    bm.set(0)
    bm.set(2)
    bm.set(4)

    # Upload bitmap data to GPU
    var dev_bm = bm^.to_immutable().to_device(ctx)

    # Build a BitmapView backed by device memory and run the GPU kernel
    var bv = dev_bm.view()

    var dev_dst = Buffer.alloc_device[DType.uint8](ctx, 8)
    _bits_to_bytes(bv, dev_dst.view[DType.uint8](), 8, ctx)

    var frozen_dst = dev_dst^.to_immutable()
    assert_true(dev_bm.is_device())
    assert_true(frozen_dst.is_device())

    # Download and verify
    var result = frozen_dst.to_cpu(ctx)
    assert_equal(result.unsafe_get(0), UInt8(1))
    assert_equal(result.unsafe_get(1), UInt8(0))
    assert_equal(result.unsafe_get(2), UInt8(1))
    assert_equal(result.unsafe_get(3), UInt8(0))
    assert_equal(result.unsafe_get(4), UInt8(1))
    assert_equal(result.unsafe_get(5), UInt8(0))
    assert_equal(result.unsafe_get(6), UInt8(0))
    assert_equal(result.unsafe_get(7), UInt8(0))


def test_bitmapview_gpu_with_offset() raises:
    """BitmapView with non-zero bit offset transfers correctly to GPU."""
    var ctx = DeviceContext()

    # Set bits 8 and 10 (offset=8 means the view starts at bit 8)
    var bm = Bitmap.alloc_zeroed(16)
    bm.set(8)
    bm.set(10)

    var dev_bm = bm^.to_immutable().to_device(ctx)

    # View of 4 bits starting at bit offset 8
    var bv = dev_bm.view(8, 4)

    var dev_dst = Buffer.alloc_device[DType.uint8](ctx, 4)
    _bits_to_bytes(bv, dev_dst.view[DType.uint8](), 4, ctx)

    var frozen_dst = dev_dst^.to_immutable()
    assert_true(dev_bm.is_device())
    assert_true(frozen_dst.is_device())

    # Expected: bit 8 → 1, bit 9 → 0, bit 10 → 1, bit 11 → 0
    var result = frozen_dst.to_cpu(ctx)
    assert_equal(result.unsafe_get(0), UInt8(1))
    assert_equal(result.unsafe_get(1), UInt8(0))
    assert_equal(result.unsafe_get(2), UInt8(1))
    assert_equal(result.unsafe_get(3), UInt8(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
