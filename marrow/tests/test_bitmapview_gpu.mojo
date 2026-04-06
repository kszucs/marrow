"""Diagnostic GPU tests for BitmapView methods via enqueue_function.

Each test targets exactly one method to isolate which operations work on
Metal/GPU and where bitpacking or atomic-write patterns break.

Expected outcomes (noted per test):
- test(), mask[W](), load_bits[T]()      — read-only, PASS expected
- store[T, W=1]()                        — raw byte write (one thread/byte), PASS expected
- store[W=8](SIMD[bool, 8])              — calls pack_bits, FAIL expected (Metal internal error)
- set/clear/toggle with stride 8         — one thread per byte, PASS expected
- set with stride 1 (8 threads→byte 0)  — non-atomic RMW race, FAIL/wrong expected
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from marrow.buffers import Bitmap, Buffer
from marrow.views import BitmapView, BufferView


# ---------------------------------------------------------------------------
# Kernels — one per BitmapView method under test
# ---------------------------------------------------------------------------


def _k_test(
    src: BitmapView[MutAnyOrigin],
    dst: BufferView[DType.uint8, MutAnyOrigin],
    n: Int,
):
    """BitmapView.test(i) → UInt8(1|0) per bit."""
    var i = Int(global_idx.x)
    if i < n:
        dst.unsafe_set(i, UInt8(1) if src.test(i) else UInt8(0))


def _k_mask(
    src: BitmapView[MutAnyOrigin],
    dst: BufferView[DType.uint8, MutAnyOrigin],
    n_chunks: Int,
):
    """BitmapView.mask[8](i*8) — expand 8 bits per thread into dst."""
    var chunk = Int(global_idx.x)
    if chunk < n_chunks:
        var bits = src.mask[8](chunk * 8)
        for j in range(8):
            dst.unsafe_set(chunk * 8 + j, UInt8(1) if bits[j] else UInt8(0))


def _k_load_bits(
    src: BitmapView[MutAnyOrigin],
    dst: BufferView[DType.uint8, MutAnyOrigin],
    n_bytes: Int,
):
    """BitmapView.load_bits[uint8](i*8) — read one raw byte per thread."""
    var i = Int(global_idx.x)
    if i < n_bytes:
        dst.unsafe_set(i, src.load_bits[DType.uint8](i * 8))


def _k_store_bytes(
    src: BufferView[DType.uint8, MutAnyOrigin],
    dst: BitmapView[MutAnyOrigin],
    n_bytes: Int,
):
    """BitmapView.store[uint8, 1](i, val) — write one raw byte per thread."""
    var i = Int(global_idx.x)
    if i < n_bytes:
        dst.store[DType.uint8, 1](i, src.unsafe_get(i))


def _k_store_bools(
    src: BitmapView[MutAnyOrigin],
    dst: BitmapView[MutAnyOrigin],
    n_chunks: Int,
):
    """BitmapView.store[8](i*8, SIMD[bool,8]) — pack bits via pack_bits.

    Expected: FAIL — Metal shader compiler crashes on pack_bits intrinsic.
    """
    var chunk = Int(global_idx.x)
    if chunk < n_chunks:
        dst.store[8](chunk * 8, src.mask[8](chunk * 8))


def _k_set_stride8(dst: BitmapView[MutAnyOrigin], n_bytes: Int):
    """BitmapView.set(i*8) — set LSB of each byte; no byte-level race."""
    var i = Int(global_idx.x)
    if i < n_bytes:
        dst.set(i * 8)


def _k_clear_stride8(dst: BitmapView[MutAnyOrigin], n_bytes: Int):
    """BitmapView.clear(i*8) — clear LSB of each byte; no byte-level race."""
    var i = Int(global_idx.x)
    if i < n_bytes:
        dst.clear(i * 8)


def _k_toggle_stride8(dst: BitmapView[MutAnyOrigin], n_bytes: Int):
    """BitmapView.toggle(i*8) — toggle LSB of each byte; no byte-level race."""
    var i = Int(global_idx.x)
    if i < n_bytes:
        dst.toggle(i * 8)


def _k_set_stride1(dst: BitmapView[MutAnyOrigin], n: Int):
    """BitmapView.set(i) — all threads target byte 0; intentional race."""
    var i = Int(global_idx.x)
    if i < n:
        dst.set(i)


# ---------------------------------------------------------------------------
# Helper: upload byte values to a mutable device BitmapView
# ---------------------------------------------------------------------------


def _enqueue_store_bytes(
    ctx: DeviceContext,
    src: Buffer[mut=False],
    dst: Bitmap[mut=True],
    n: Int,
) raises:
    var compiled = ctx.compile_function_experimental[_k_store_bytes]()
    ctx.enqueue_function(
        compiled, src.device_view[DType.uint8](), dst.view(), n, grid_dim=1, block_dim=n
    )


# ---------------------------------------------------------------------------
# Test: BitmapView.test()
# ---------------------------------------------------------------------------


def test_bitmapview_test_gpu() raises:
    """BitmapView.test(i) reads individual bits from GPU (expected: PASS)."""
    var ctx = DeviceContext()

    # bits 0, 2, 4, 6 set → [1,0,1,0,1,0,1,0]
    var bm = Bitmap.alloc_zeroed(8)
    bm.set(0)
    bm.set(2)
    bm.set(4)
    bm.set(6)
    var dev_src = bm^.to_immutable().to_device(ctx)
    var dev_dst = Buffer.alloc_device[DType.uint8](ctx, 8)

    var compiled = ctx.compile_function_experimental[_k_test]()
    ctx.enqueue_function(
        compiled, dev_src.view(), dev_dst.device_view[DType.uint8](), 8, grid_dim=1, block_dim=8
    )

    var result = dev_dst^.to_immutable().to_cpu(ctx)
    assert_equal(result.unsafe_get[DType.uint8](0), UInt8(1))
    assert_equal(result.unsafe_get[DType.uint8](1), UInt8(0))
    assert_equal(result.unsafe_get[DType.uint8](2), UInt8(1))
    assert_equal(result.unsafe_get[DType.uint8](3), UInt8(0))
    assert_equal(result.unsafe_get[DType.uint8](4), UInt8(1))
    assert_equal(result.unsafe_get[DType.uint8](5), UInt8(0))
    assert_equal(result.unsafe_get[DType.uint8](6), UInt8(1))
    assert_equal(result.unsafe_get[DType.uint8](7), UInt8(0))


# ---------------------------------------------------------------------------
# Test: BitmapView.mask[8]()
# ---------------------------------------------------------------------------


def test_bitmapview_mask_gpu() raises:
    """BitmapView.mask[8](i*8) expands 8 bits per call on GPU (expected: PASS)."""
    var ctx = DeviceContext()

    # 16 bits — odd positions set
    var bm = Bitmap.alloc_zeroed(16)
    for i in range(16):
        if i % 2 == 1:
            bm.set(i)
    var dev_src = bm^.to_immutable().to_device(ctx)
    var dev_dst = Buffer.alloc_device[DType.uint8](ctx, 16)

    var compiled = ctx.compile_function_experimental[_k_mask]()
    ctx.enqueue_function(
        compiled, dev_src.view(), dev_dst.device_view[DType.uint8](), 2, grid_dim=1, block_dim=2
    )

    var result = dev_dst^.to_immutable().to_cpu(ctx)
    for i in range(16):
        if i % 2 == 1:
            assert_equal(result.unsafe_get[DType.uint8](i), UInt8(1))
        else:
            assert_equal(result.unsafe_get[DType.uint8](i), UInt8(0))


# ---------------------------------------------------------------------------
# Test: BitmapView.load_bits[uint8]()
# ---------------------------------------------------------------------------


def test_bitmapview_load_bits_gpu() raises:
    """BitmapView.load_bits[uint8](i*8) reads raw bytes on GPU (expected: PASS)."""
    var ctx = DeviceContext()

    # byte 0 = 0b10110001 = 0xB1 (bits 0,4,5,7 set)
    # byte 1 = 0b01000111 = 0x47 (bits 8,9,10,14 set, i.e. byte1 bits 0,1,2,6)
    var bm = Bitmap.alloc_zeroed(16)
    bm.set(0)
    bm.set(4)
    bm.set(5)
    bm.set(7)
    bm.set(8)
    bm.set(9)
    bm.set(10)
    bm.set(14)
    var dev_src = bm^.to_immutable().to_device(ctx)
    var dev_dst = Buffer.alloc_device[DType.uint8](ctx, 2)

    var compiled = ctx.compile_function_experimental[_k_load_bits]()
    ctx.enqueue_function(
        compiled, dev_src.view(), dev_dst.device_view[DType.uint8](), 2, grid_dim=1, block_dim=2
    )

    var result = dev_dst^.to_immutable().to_cpu(ctx)
    assert_equal(result.unsafe_get[DType.uint8](0), UInt8(0xB1))
    assert_equal(result.unsafe_get[DType.uint8](1), UInt8(0x47))


# ---------------------------------------------------------------------------
# Test: BitmapView.store[uint8, 1]()  — raw byte write
# ---------------------------------------------------------------------------


def test_bitmapview_store_bytes_gpu() raises:
    """BitmapView.store[uint8,1](i, val) writes raw bytes on GPU (expected: PASS)."""
    var ctx = DeviceContext()

    var cpu_vals = Buffer.alloc_zeroed[DType.uint8](4)
    cpu_vals.unsafe_set[DType.uint8](0, UInt8(0xAA))
    cpu_vals.unsafe_set[DType.uint8](1, UInt8(0x55))
    cpu_vals.unsafe_set[DType.uint8](2, UInt8(0xFF))
    cpu_vals.unsafe_set[DType.uint8](3, UInt8(0x00))
    var dev_vals = cpu_vals^.to_immutable().to_device(ctx)

    var dev_bm = Bitmap.alloc_device(ctx, 32)
    _enqueue_store_bytes(ctx, dev_vals, dev_bm, 4)

    var result = dev_bm^.to_immutable().to_cpu(ctx)
    var v = result.view(0, 32)
    # byte 0 = 0xAA = 0b10101010: odd bits set
    for i in range(8):
        if i % 2 == 0:
            assert_false(v.test(i))
        else:
            assert_true(v.test(i))
    # byte 1 = 0x55 = 0b01010101: even bits set
    for i in range(8):
        if i % 2 == 0:
            assert_true(v.test(8 + i))
        else:
            assert_false(v.test(8 + i))
    # byte 2 = 0xFF: all set
    for i in range(8):
        assert_true(v.test(16 + i))
    # byte 3 = 0x00: all clear
    for i in range(8):
        assert_false(v.test(24 + i))


# ---------------------------------------------------------------------------
# Test: BitmapView.store[8](SIMD[bool, 8]) — bool bitpack write
# ---------------------------------------------------------------------------


def test_bitmapview_store_bools_gpu() raises:
    """BitmapView.store[8](i, SIMD[bool,8]) packs bits on GPU.

    Expected: FAIL — Metal shader compiler crashes on the pack_bits
    intrinsic used inside store[bool].
    """
    var ctx = DeviceContext()

    var bm = Bitmap.alloc_zeroed(16)
    bm.set(0)
    bm.set(3)
    bm.set(7)
    bm.set(9)
    var dev_src = bm^.to_immutable().to_device(ctx)
    var dev_dst = Bitmap.alloc_device(ctx, 16)

    var compiled = ctx.compile_function_experimental[_k_store_bools]()
    ctx.enqueue_function(
        compiled, dev_src.view(), dev_dst.view(), 2, grid_dim=1, block_dim=2
    )

    var result = dev_dst^.to_immutable().to_cpu(ctx)
    var v = result.view(0, 16)
    assert_true(v.test(0))
    assert_false(v.test(1))
    assert_false(v.test(2))
    assert_true(v.test(3))
    assert_false(v.test(4))
    assert_false(v.test(5))
    assert_false(v.test(6))
    assert_true(v.test(7))
    assert_false(v.test(8))
    assert_true(v.test(9))


# ---------------------------------------------------------------------------
# Test: BitmapView.set() — no byte-level race (stride 8)
# ---------------------------------------------------------------------------


def test_bitmapview_set_norace_gpu() raises:
    """BitmapView.set(i*8): one thread per byte sets bit 0 (expected: PASS)."""
    var ctx = DeviceContext()

    var dev_bm = Bitmap.alloc_device(ctx, 64)
    var compiled = ctx.compile_function_experimental[_k_set_stride8]()
    ctx.enqueue_function(compiled, dev_bm.view(), 8, grid_dim=1, block_dim=8)

    var result = dev_bm^.to_immutable().to_cpu(ctx)
    var v = result.view(0, 64)
    for byte_i in range(8):
        assert_true(v.test(byte_i * 8))
        for bit_j in range(1, 8):
            assert_false(v.test(byte_i * 8 + bit_j))


# ---------------------------------------------------------------------------
# Test: BitmapView.clear() — no byte-level race (stride 8)
# ---------------------------------------------------------------------------


def test_bitmapview_clear_norace_gpu() raises:
    """BitmapView.clear(i*8): one thread per byte clears bit 0 (expected: PASS).

    Bitmap is pre-filled with 0xFF bytes before clearing.
    """
    var ctx = DeviceContext()

    # Pre-fill all bytes to 0xFF via store[uint8,1]
    var dev_bm = Bitmap.alloc_device(ctx, 64)
    var cpu_ff = Buffer.alloc_zeroed[DType.uint8](8)
    for i in range(8):
        cpu_ff.unsafe_set[DType.uint8](i, UInt8(0xFF))
    var dev_ff = cpu_ff^.to_immutable().to_device(ctx)
    _enqueue_store_bytes(ctx, dev_ff, dev_bm, 8)

    var compiled = ctx.compile_function_experimental[_k_clear_stride8]()
    ctx.enqueue_function(compiled, dev_bm.view(), 8, grid_dim=1, block_dim=8)

    var result = dev_bm^.to_immutable().to_cpu(ctx)
    var v = result.view(0, 64)
    for byte_i in range(8):
        assert_false(v.test(byte_i * 8))
        for bit_j in range(1, 8):
            assert_true(v.test(byte_i * 8 + bit_j))


# ---------------------------------------------------------------------------
# Test: BitmapView.toggle() — no byte-level race (stride 8)
# ---------------------------------------------------------------------------


def test_bitmapview_toggle_norace_gpu() raises:
    """BitmapView.toggle(i*8): toggles LSB of each byte (expected: PASS).

    Pre-set LSBs to 1 via store, then toggle them to 0.
    """
    var ctx = DeviceContext()

    # Pre-fill each byte to 0x01 (only LSB set)
    var dev_bm = Bitmap.alloc_device(ctx, 64)
    var cpu_01 = Buffer.alloc_zeroed[DType.uint8](8)
    for i in range(8):
        cpu_01.unsafe_set[DType.uint8](i, UInt8(0x01))
    var dev_01 = cpu_01^.to_immutable().to_device(ctx)
    _enqueue_store_bytes(ctx, dev_01, dev_bm, 8)

    var compiled = ctx.compile_function_experimental[_k_toggle_stride8]()
    ctx.enqueue_function(compiled, dev_bm.view(), 8, grid_dim=1, block_dim=8)

    var result = dev_bm^.to_immutable().to_cpu(ctx)
    var v = result.view(0, 64)
    # All bits should be 0 now (LSBs toggled off; rest were already 0)
    for i in range(64):
        assert_false(v.test(i))


# ---------------------------------------------------------------------------
# Test: BitmapView.set() — WITH byte-level race (stride 1)
# ---------------------------------------------------------------------------


def test_bitmapview_set_race_gpu() raises:
    """BitmapView.set(i): 8 threads all write to byte 0 (expected: FAIL/wrong).

    Non-atomic RMW on a shared byte produces undefined results on GPU;
    the assertion that all 8 bits are set is expected to fail.
    """
    var ctx = DeviceContext()

    var dev_bm = Bitmap.alloc_device(ctx, 8)
    var compiled = ctx.compile_function_experimental[_k_set_stride1]()
    ctx.enqueue_function(compiled, dev_bm.view(), 8, grid_dim=1, block_dim=8)

    var result = dev_bm^.to_immutable().to_cpu(ctx)
    var v = result.view(0, 8)
    for i in range(8):
        assert_true(v.test(i))


def main() raises:
    var suite = TestSuite()
    suite.test[test_bitmapview_test_gpu]()
    suite.test[test_bitmapview_mask_gpu]()
    suite.test[test_bitmapview_load_bits_gpu]()
    suite.test[test_bitmapview_store_bytes_gpu]()
    suite.test[test_bitmapview_store_bools_gpu]()
    suite.test[test_bitmapview_set_norace_gpu]()
    suite.test[test_bitmapview_clear_norace_gpu]()
    suite.test[test_bitmapview_toggle_norace_gpu]()
    suite.test[test_bitmapview_set_race_gpu]()
    suite^.run()
