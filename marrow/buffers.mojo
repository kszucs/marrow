"""Arrow-compatible memory buffers: immutable types and mutable builders.

Buffer Kinds
------------
A `Buffer` has one of four memory kinds, encoded in the `Allocation` it owns:

  CPU
      Owned Mojo heap allocation.  Created exclusively by `BufferBuilder.finish()`.
      The `Allocation` holds the raw pointer; `Allocation.__del__` calls `ptr.free()`.

  FOREIGN
      External CPU memory provided by a producer (Arrow C Data Interface or Arrow C
      Device Data Interface with device_type=CPU).  A custom release callback stored
      in the `Allocation` is invoked when the last `Buffer` view is dropped.
      Multiple `Buffer` views share the *same* `ArcPointer[Allocation]` (the "keeper")
      so the release fires exactly once when the last view is destroyed.

  HOST
      Pinned CPU memory managed by Mojo's `HostBuffer` (AsyncRT reference-counted).
      CPU-accessible (the `ptr` field is valid) and fast for DMA to/from the GPU.
      Created via `Buffer.from_host()`.  The `Allocation._host` Optional field owns
      the `HostBuffer`; its destructor cascades to `AsyncRT_DeviceBuffer_release`.

  DEVICE
      GPU device memory managed by Mojo's `DeviceBuffer` (AsyncRT reference-counted).
      NOT CPU-accessible (`ptr` is null).  Created via `Buffer.from_device()` or by
      calling `buffer.to_device(ctx)` on any CPU-accessible buffer.
      The `Allocation._device` Optional field owns the `DeviceBuffer`; its destructor
      cascades to `AsyncRT_DeviceBuffer_release`.

CPU accessibility is encoded directly in `Buffer.ptr`: a non-null `ptr` means
CPU-accessible (CPU, FOREIGN, HOST); a null `ptr` means device-only (DEVICE).
`is_cpu()` and `is_device()` use this O(1) check.  `is_host()` checks `_owner._host`.

Ownership Model
---------------
`Buffer` is `ImplicitlyCopyable`: copying a Buffer is O(1) and bumps the
`ArcPointer[Allocation]` reference count.  The backing memory is freed / released
only when the *last* copy is dropped.

`BufferBuilder` is the mutable counterpart — it exclusively owns a writable pointer.
`BufferBuilder.finish()` transfers that pointer into an owned CPU `Buffer`.

Allocation Invariant
--------------------
Each `Allocation` has exactly one active release mechanism (checked in `__del__`):
  - `release is Some` → FOREIGN: invoke the producer's C release callback.
  - `ptr is non-null` → CPU: call `ptr.free()` directly (no callback).
  - `_host is Some`   → HOST: `HostBuffer.__del__` cascades to AsyncRT release.
  - `_device is Some` → DEVICE: `DeviceBuffer.__del__` cascades to AsyncRT release.

device_type / device_id
------------------------
`Buffer.device_type() raises -> Int32` returns the Arrow C Device Data Interface
`DeviceType` value (1=CPU, 2=CUDA, 3=CUDA_HOST, 8=Metal, etc.) for interoperability.
The value is inferred from the GPU runtime context's API name via `context().api()`:
  HOST:   cuda→CUDA_HOST(3), hip→ROCM_HOST(11), otherwise raises
  DEVICE: cuda→CUDA(2), hip→ROCM(10), metal→METAL(8), otherwise raises
CPU and FOREIGN buffers always return `DeviceType.CPU` (1); `device_id()` returns -1.

Transfer methods
----------------
  `to_device(ctx) -> Buffer`  — uploads any CPU-accessible buffer (CPU / FOREIGN / HOST)
                                to the GPU; returns a new DEVICE buffer.
  `to_cpu(ctx) -> Buffer`     — downloads a DEVICE buffer to an owned CPU heap buffer;
                                returns a new CPU buffer.  HOST buffers are already
                                CPU-accessible via `ptr` and do not need downloading.

BufferBuilder lifecycle
------------------------
CPU heap allocation (kind=CPU):
  1. `var b = BufferBuilder.alloc[T](n)` — 64-byte-aligned heap allocation.
  2. `b.unsafe_set(i, v)` / `b.simd_store(...)` — write through the mutable pointer.
  3. `var buf = b.finish()` — zero-cost transfer into an immutable CPU Buffer.

Pinned host allocation (kind=HOST):
  1. `var b = BufferBuilder.alloc_host[T](ctx, n)` — page-locked allocation via DeviceContext.
  2. `b.unsafe_set(i, v)` / `b.simd_store(...)` — write through the mutable pointer.
  3. `var buf = b.finish()` — transfer into an immutable HOST Buffer.

Bitmap operations
-----------------
Validity bitmaps use the dedicated `Bitmap` / `BitmapBuilder` types from
`marrow.bitmap`, which wrap `Buffer` / `BufferBuilder` with bit-level and
SIMD bulk operations.
"""

from std.memory import (
    memset_zero,
    memcpy,
    memset,
    ArcPointer,
)
from std.sys.info import simd_byte_width
from std.sys import size_of
import std.math as math
from std.gpu.host import DeviceBuffer, DeviceContext, HostBuffer


struct DeviceType:
    """Device type constants from the Arrow C Device Data Interface / DLPack.

    Use these when constructing HOST or DEVICE buffers (`from_host`,
    `from_device`, `BufferBuilder.finish`) and when exporting via
    `CArrowDeviceArray`.  CPU and FOREIGN buffers always have `device_type()
    == DeviceType.CPU`.
    """

    comptime CPU: Int32 = 1
    """Standard CPU (host) memory."""

    comptime CUDA: Int32 = 2
    """NVIDIA GPU memory allocated via the CUDA runtime or driver API."""

    comptime CUDA_HOST: Int32 = 3
    """Pinned CPU memory allocated via `cudaMallocHost` / `cudaHostAlloc`."""

    comptime OPENCL: Int32 = 4
    """OpenCL device memory."""

    comptime VULKAN: Int32 = 7
    """Vulkan device memory."""

    comptime METAL: Int32 = 8
    """Apple Metal GPU memory."""

    comptime ROCM: Int32 = 10
    """AMD ROCm GPU memory."""

    comptime ROCM_HOST: Int32 = 11
    """Pinned CPU memory allocated via `hipMallocHost`."""

    comptime CUDA_MANAGED: Int32 = 13
    """CUDA unified (managed) memory."""

    comptime ONEAPI: Int32 = 14
    """Intel oneAPI USM memory."""

    comptime WEBGPU: Int32 = 15
    """WebGPU device memory."""

    comptime HEXAGON: Int32 = 16
    """Qualcomm Hexagon DSP memory."""


comptime simd_width = simd_byte_width()
comptime simd_widths = (simd_width, simd_width // 2, 1)


# ---------------------------------------------------------------------------
# Allocation — owns a memory region, one release mechanism active at a time
# ---------------------------------------------------------------------------


struct Allocation(Movable):
    """Owns a buffer's backing memory with exactly one active release mechanism.

    Release rules (in `__del__`):
      - `release is Some`  → FOREIGN: invoke the producer's C callback.
      - `ptr is non-null`  → CPU: call `ptr.free()`.
      - `_host is Some`    → HOST: HostBuffer.__del__ cascades to AsyncRT release.
      - `_device is Some`  → DEVICE: DeviceBuffer.__del__ cascades to AsyncRT release.

    Always accessed through `ArcPointer[Allocation]` so that multiple `Buffer`
    views can share ownership.  Lifetime: the last ArcPointer to drop triggers
    `__del__`, which fires the appropriate release.

    Use the static factory methods (`cpu`, `foreign`, `host`, `device`) rather
    than the raw `__init__`.
    """

    var ptr: UnsafePointer[UInt8, MutAnyOrigin]
    """Raw CPU pointer.  Non-null for CPU and FOREIGN; null (default) for HOST/DEVICE."""

    var release: Optional[def(UnsafePointer[UInt8, MutAnyOrigin]) -> None]
    """Release callback.  Set for CPU (_cpu_release) and FOREIGN (producer callback);
    None for HOST and DEVICE (their Optional field destructors handle release)."""

    var _host: Optional[HostBuffer[DType.uint8]]
    """Pinned host buffer.  Set only for HOST kind; None otherwise."""

    var _device: Optional[DeviceBuffer[DType.uint8]]
    """GPU device buffer.  Set only for DEVICE kind; None otherwise."""

    def __init__(
        out self,
        ptr: UnsafePointer[UInt8, MutAnyOrigin],
        release: Optional[def(UnsafePointer[UInt8, MutAnyOrigin]) -> None],
        host: Optional[HostBuffer[DType.uint8]],
        device: Optional[DeviceBuffer[DType.uint8]],
    ):
        self.ptr = ptr
        self.release = release
        self._host = host
        self._device = device

    @staticmethod
    def cpu(ptr: UnsafePointer[UInt8, MutAnyOrigin]) -> Allocation:
        """Create an owned CPU allocation.  `__del__` calls `ptr.free()`."""
        return Allocation(ptr, None, None, None)

    @staticmethod
    def foreign(
        ptr: UnsafePointer[UInt8, MutAnyOrigin],
        release: def(UnsafePointer[UInt8, MutAnyOrigin]) -> None,
    ) -> Allocation:
        """Create a foreign CPU allocation with a custom release callback."""
        return Allocation(ptr, release, None, None)

    @staticmethod
    def host(host_buf: HostBuffer[DType.uint8]) -> Allocation:
        """Create a HOST (pinned) allocation.  HostBuffer.__del__ handles release.
        """
        return Allocation(
            UnsafePointer[UInt8, MutAnyOrigin](), None, host_buf, None
        )

    @staticmethod
    def device(dev_buf: DeviceBuffer[DType.uint8]) -> Allocation:
        """Create a DEVICE (GPU) allocation.  DeviceBuffer.__del__ handles release.
        """
        return Allocation(
            UnsafePointer[UInt8, MutAnyOrigin](), None, None, dev_buf
        )

    def device_type(self) raises -> Int32:
        """Return the Arrow C Device Data Interface DeviceType value.

        Inferred from the GPU runtime context's API name:
          - HOST + "cuda"    → DeviceType.CUDA_HOST (3)
          - HOST + "hip"     → DeviceType.ROCM_HOST (11)
          - DEVICE + "cuda"  → DeviceType.CUDA (2)
          - DEVICE + "hip"   → DeviceType.ROCM (10)
          - DEVICE + "metal" → DeviceType.METAL (8)
          - CPU / FOREIGN    → DeviceType.CPU (1)
          - HOST/DEVICE with unrecognised API → raises Error
        """
        if self._host:
            var api = self._host.value().context().api()
            if api == "cuda":
                return DeviceType.CUDA_HOST
            elif api == "hip":
                return DeviceType.ROCM_HOST
            else:
                raise Error("device_type: unsupported host API: ", api)
        elif self._device:
            var api = self._device.value().context().api()
            if api == "cuda":
                return DeviceType.CUDA
            elif api == "hip":
                return DeviceType.ROCM
            elif api == "metal":
                return DeviceType.METAL
            else:
                raise Error("device_type: unsupported device API: ", api)
        else:
            return DeviceType.CPU

    def device_id(self) raises -> Int64:
        """Return the physical device index.  -1 for CPU and FOREIGN allocations.

        For HOST allocations reads from `HostBuffer.context().id()`.
        For DEVICE allocations reads from `DeviceBuffer.context().id()`.
        """
        if self._host:
            var hb = self._host.value()
            return hb.context().id()
        elif self._device:
            var db = self._device.value()
            return db.context().id()
        else:
            return -1

    def __del__(deinit self):
        if self.release:
            # FOREIGN: invoke the producer's C release callback.
            self.release.value()(self.ptr)
        elif self.ptr:
            # CPU: free the Mojo heap allocation directly.
            # HOST and DEVICE have a null ptr, so this branch is CPU-only.
            self.ptr.free()
        # HOST/DEVICE: null ptr; Optional field destructors cascade to AsyncRT release.


# ---------------------------------------------------------------------------
# BufferBuilder — mutable buffer for building arrays
# ---------------------------------------------------------------------------


struct BufferBuilder(Movable):
    """Mutable contiguous memory region with 64-byte alignment.

    Two allocation modes:
      - `BufferBuilder.alloc[T](n)` — Mojo heap (CPU); freeze with `finish()`.
      - `BufferBuilder.alloc_host[T](ctx, n)` — pinned host memory; freeze with
        `finish(device_type)`.

    In both cases write via `unsafe_set()` / `simd_store()`, then call the
    appropriate `finish*` method to obtain an immutable `Buffer`.
    """

    var ptr: UnsafePointer[UInt8, MutExternalOrigin]
    var size: Int
    var _host: Optional[HostBuffer[DType.uint8]]
    """Set when allocated via `alloc_host`; None for CPU heap allocations."""
    var _device: Optional[DeviceBuffer[DType.uint8]]
    """Set when allocated via `alloc_device`; None otherwise."""

    def __init__(
        out self,
        ptr: UnsafePointer[UInt8, MutExternalOrigin],
        size: Int,
        host: Optional[HostBuffer[DType.uint8]] = None,
        device: Optional[DeviceBuffer[DType.uint8]] = None,
    ):
        self.ptr = ptr
        self.size = size
        self._host = host
        self._device = device

    @staticmethod
    def _aligned_size[T: DType](length: Int) -> Int:
        """Compute the 64-byte-aligned allocation size for `length` elements of type T.

        For DType.bool, `length` is the number of bits.
        """
        comptime if T == DType.bool:
            return math.align_up(math.ceildiv(length, 8), 64)
        else:
            return math.align_up(length * size_of[T](), 64)

    @staticmethod
    def alloc[
        I: Intable, //, T: DType = DType.uint8
    ](length: I) -> BufferBuilder:
        """Allocate a 64-byte-aligned buffer for `length` elements of type T.

        For DType.bool, `length` is the number of bits; the buffer will hold
        ceildiv(length, 8) bytes, zero-padded to a 64-byte boundary.
        """
        var byte_size = Self._aligned_size[T](Int(length))
        var ptr = alloc[UInt8](byte_size, alignment=64)
        memset_zero(ptr, byte_size)
        return BufferBuilder(ptr, byte_size)

    @staticmethod
    def alloc_uninit(byte_size: Int) -> BufferBuilder:
        """Allocate a 64-byte-aligned buffer without zero-filling.

        Use only when the caller guarantees every byte will be written
        before the buffer is read (e.g. SIMD loops that cover the full range).
        """
        var size = math.align_up(byte_size, 64)
        var ptr = alloc[UInt8](size, alignment=64)
        return BufferBuilder(ptr, size)

    @staticmethod
    def alloc_host[
        I: Intable, //, T: DType = DType.uint8
    ](ctx: DeviceContext, length: I) raises -> BufferBuilder:
        """Allocate page-locked (pinned) host memory for `length` elements of type T.

        Pinned memory is CPU-accessible and enables fast DMA transfers to/from
        the GPU.  Use `unsafe_set()` / `simd_store()` to write, then call
        `finish()` to obtain an immutable HOST Buffer.

        For DType.bool, `length` is the number of bits; the buffer will hold
        ceildiv(length, 8) bytes, zero-padded to a 64-byte boundary.

        Args:
            ctx:    DeviceContext used to allocate the HostBuffer.
            length: Number of elements (bits for DType.bool).

        Returns:
            A BufferBuilder backed by pinned host memory.
        """
        var byte_size = Self._aligned_size[T](Int(length))
        var host = ctx.enqueue_create_host_buffer[DType.uint8](byte_size)
        var ptr = rebind[UnsafePointer[UInt8, MutExternalOrigin]](
            host.unsafe_ptr()
        )
        memset_zero(ptr, byte_size)
        return BufferBuilder(ptr, byte_size, host)

    @staticmethod
    def alloc_device[
        I: Intable, //, T: DType = DType.uint8
    ](ctx: DeviceContext, length: I) raises -> BufferBuilder:
        """Allocate a device (GPU) buffer for `length` elements of type T.

        The returned builder exposes `ptr` as a `MutAnyOrigin` device pointer
        suitable for GPU kernel writes. Call `finish()` to obtain an immutable
        device-resident `Buffer`.
        """
        var byte_size = Self._aligned_size[T](Int(length))
        var dev = ctx.enqueue_create_buffer[DType.uint8](byte_size)
        var ptr = rebind[UnsafePointer[UInt8, MutExternalOrigin]](
            dev.unsafe_ptr()
        )
        return BufferBuilder(ptr, byte_size, device=dev)

    def finish(mut self) -> Buffer:
        """Snapshot the mutable builder into an immutable Buffer and reset state.

        For CPU builders (`alloc`): returns kind=CPU.
        For HOST builders (`alloc_host`): returns kind=HOST.
        For DEVICE builders (`alloc_device`): returns kind=DEVICE.

        A fresh zero-capacity CPU allocation is installed on this builder so
        it can continue to be used after the call.
        """
        var new = Self.alloc(0)
        swap(self, new)
        # After swap: self has the fresh empty allocation; new has the old state.
        # We null new.ptr before returning so new.__del__ skips ptr.free() —
        # the returned Buffer's Allocation owns the memory from this point on.
        if new._device:
            var result = Buffer.from_device(new._device.take(), new.size)
            new.ptr = UnsafePointer[UInt8, MutExternalOrigin]()
            return result
        if new._host:
            var result = Buffer.from_host(new._host.take())
            new.ptr = UnsafePointer[UInt8, MutExternalOrigin]()
            return result
        var result = Buffer(
            new.ptr.as_immutable(),
            new.size,
            ArcPointer(
                Allocation.cpu(
                    rebind[UnsafePointer[UInt8, MutAnyOrigin]](new.ptr)
                )
            ),
        )
        new.ptr = UnsafePointer[UInt8, MutExternalOrigin]()
        return result

    def resize[
        I: Intable, //, T: DType = DType.uint8
    ](mut self, length: I) raises:
        """Resize the buffer to hold `length` elements of type T.

        For HOST builders the new allocation is also pinned host memory using
        the same `DeviceContext`; for CPU builders a plain heap allocation is used.

        No-op if the new size maps to the same byte allocation as the current one.
        """
        var byte_size = Self._aligned_size[T](Int(length))
        if byte_size == self.size:
            return
        var new: BufferBuilder
        if self._host:
            new = BufferBuilder.alloc_host[T](
                self._host.value().context(), length
            )
        else:
            new = BufferBuilder.alloc[T](length)
        memcpy(dest=new.ptr, src=self.ptr, count=min(new.size, self.size))
        swap(self, new)

    @always_inline
    def length[T: DType = DType.uint8](self) -> Int:
        comptime if T == DType.bool:
            return self.size * 8
        else:
            return self.size // size_of[T]()

    @always_inline
    def unsafe_ptr[
        T: DType = DType.uint8
    ](self) -> UnsafePointer[Scalar[T], MutExternalOrigin]:
        return self.ptr.bitcast[Scalar[T]]()

    # TODO: use Indexable for index?
    @always_inline
    def unsafe_get[T: DType = DType.uint8](self, index: Int) -> Scalar[T]:
        comptime output = Scalar[T]
        comptime if T == DType.bool:
            var byte = self.ptr[index // 8]
            return output((byte >> UInt8(index % 8)) & 1)
        else:
            return self.ptr.bitcast[output]()[index]

    @always_inline
    def unsafe_set[
        T: DType = DType.uint8
    ](mut self, index: Int, value: Scalar[T]):
        comptime if T == DType.bool:
            var byte_index = index // 8
            var bit_mask = UInt8(1 << (index % 8))
            if value:
                self.ptr[byte_index] = self.ptr[byte_index] | bit_mask
            else:
                self.ptr[byte_index] = self.ptr[byte_index] & ~bit_mask
        else:
            comptime output = Scalar[T]
            self.ptr.bitcast[output]()[index] = value

    @always_inline
    def simd_load[T: DType, W: Int](self, index: Int) -> SIMD[T, W]:
        """Load W elements of type T at element index `index`."""
        return (self.ptr.bitcast[Scalar[T]]() + index).load[width=W]()

    @always_inline
    def simd_store[T: DType, W: Int](mut self, index: Int, value: SIMD[T, W]):
        """Store W elements of type T at element index `index`."""
        (self.ptr.bitcast[Scalar[T]]() + index).store(value)

    def __del__(deinit self):
        if self.ptr and not self._host:
            # CPU heap allocation: free the pointer directly.
            # HOST allocation: the _host HostBuffer field destructor cascades to
            # AsyncRT release after this body; do NOT call ptr.free() on it.
            # Null ptr: ownership was transferred by finish(); skip free.
            self.ptr.free()


# ---------------------------------------------------------------------------
# Buffer — immutable buffer for read-only array data
# ---------------------------------------------------------------------------


# TODO: add assertions to ensure alignment and padding invariants hold
struct Buffer(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Immutable contiguous memory region.

    CPU accessibility is encoded in `ptr`: a non-null ptr means the buffer is
    CPU-accessible (CPU, FOREIGN, or HOST kinds); a null ptr means device-only
    (DEVICE kind).  Use `is_cpu()` / `is_device()` for the primary check.

    `is_cpu()` returns True for CPU, FOREIGN, and HOST (all have a valid ptr).
    Calling `unsafe_get` or `unsafe_ptr` on a DEVICE buffer raises a debug assert.
    Call `to_cpu(ctx)` to download device data before reading on the CPU.

    `device_type()` and `device_id()` delegate to `_owner` (see `Allocation`).

    Buffers are `ImplicitlyCopyable` with O(1) shared semantics: copying bumps the
    internal `ArcPointer[Allocation]` ref-count without copying any data.
    When the last copy is dropped the appropriate release fires (see Allocation).
    """

    var ptr: UnsafePointer[UInt8, ImmutExternalOrigin]
    """CPU data pointer.  Non-null for CPU / FOREIGN / HOST; null for DEVICE."""

    var size: Int
    """Buffer size in bytes (always 64-byte aligned)."""

    var _owner: ArcPointer[Allocation]
    """Shared ownership handle."""

    def __init__(
        out self,
        ptr: UnsafePointer[UInt8, ImmutExternalOrigin],
        size: Int,
        owner: ArcPointer[Allocation],
    ):
        self.ptr = ptr
        self.size = size
        self._owner = owner

    def __init__(out self, *, copy: Self):
        self.ptr = copy.ptr
        self.size = copy.size
        self._owner = copy._owner

    @staticmethod
    def from_foreign[
        I: Intable, //
    ](
        ptr: OpaquePointer[MutAnyOrigin],
        size: I,
        owner: ArcPointer[Allocation],
    ) -> Buffer:
        """Create an immutable view into foreign CPU memory.

        The caller passes an `ArcPointer[Allocation]` (the "keeper") that holds
        the producer's release callback.  All `Buffer` views sharing the same
        keeper bump its ref-count on copy; when the last view drops, the keeper
        releases and the C callback fires automatically.

        Precondition: `owner` must have been created with `Allocation.foreign(...)`.
        """
        return Buffer(
            rebind[UnsafePointer[UInt8, ImmutExternalOrigin]](
                ptr.bitcast[UInt8]()
            ),
            math.align_up(Int(size), 64),
            owner,
        )

    @staticmethod
    def from_host(host: HostBuffer[DType.uint8]) -> Buffer:
        """Create a HOST (pinned) buffer from a Mojo HostBuffer.

        The HostBuffer is moved into an `Allocation` behind `ArcPointer`;
        its destructor cascades to `AsyncRT_DeviceBuffer_release` when the
        last Buffer copy is dropped.

        The CPU pointer is taken from `host.unsafe_ptr()` — it remains valid
        for the lifetime of the Allocation.  `device_type()` is inferred from
        the context API (cuda→CUDA_HOST, hip→ROCM_HOST, otherwise CPU).
        """
        return Buffer(
            rebind[UnsafePointer[UInt8, ImmutExternalOrigin]](
                host.unsafe_ptr()
            ),
            len(host),
            ArcPointer(Allocation.host(host)),
        )

    @staticmethod
    def from_device(dev: DeviceBuffer[DType.uint8], size: Int) -> Buffer:
        """Create a DEVICE (GPU) buffer from a Mojo DeviceBuffer.

        The DeviceBuffer is moved into an `Allocation` behind `ArcPointer`;
        its destructor cascades to `AsyncRT_DeviceBuffer_release` when the
        last Buffer copy is dropped.

        `ptr` is set to null — call `to_cpu(ctx)` to read data on the CPU.
        `device_type()` is inferred from the context API (cuda→CUDA, hip→ROCM,
        metal→METAL).
        """
        return Buffer(
            UnsafePointer[UInt8, ImmutExternalOrigin](),
            size,
            ArcPointer(Allocation.device(dev)),
        )

    @always_inline
    def is_cpu(self) -> Bool:
        """Return True if the buffer is CPU-accessible (ptr is non-null).

        True for CPU, FOREIGN, and HOST kinds; False for DEVICE.
        """
        return self.ptr.__bool__()

    @always_inline
    def is_device(self) -> Bool:
        """Return True if the buffer lives on a GPU device (ptr is null)."""
        return not self.ptr.__bool__()

    @always_inline
    def is_host(self) -> Bool:
        """Return True if the buffer is pinned host memory (HOST kind)."""
        return self._owner[]._host.__bool__()

    def __eq__(self, other: Buffer) -> Bool:
        """Return True if both buffers have identical CPU-accessible contents.

        Compares full backing bytes using SIMD 64-byte blocks.
        Returns False if either buffer is device-resident (no CPU access).
        """
        if not self.is_cpu() or not other.is_cpu():
            return False
        if self.size != other.size:
            return False
        comptime width = simd_byte_width()
        comptime unroll = 64 // width
        var pa = self.ptr
        var pb = other.ptr
        for i in range(0, self.size, 64):
            comptime for j in range(unroll):
                comptime k = j * width
                if (pa + i + k).load[width=width]() != (pb + i + k).load[
                    width=width
                ]():
                    return False
        return True

    def device_type(self) raises -> Int32:
        """Return the Arrow C Device Data Interface DeviceType value.

        Delegates to `Allocation.device_type()`.
        """
        return self._owner[].device_type()

    def device_id(self) raises -> Int64:
        """Return the physical device index.  -1 for CPU and FOREIGN buffers.

        Delegates to `Allocation.device_id()`, which reads from
        `HostBuffer.context().id()` or `DeviceBuffer.context().id()` as needed.
        """
        return self._owner[].device_id()

    def device_buffer(self) -> DeviceBuffer[DType.uint8]:
        """Return the DeviceBuffer handle.

        Precondition: `is_device()` must be True.  The returned DeviceBuffer is
        an `ImplicitlyCopyable` copy that bumps the AsyncRT ref-count; it is safe
        to hold briefly for kernel calls.
        """
        debug_assert(self.is_device(), "not a device buffer")
        return self._owner[]._device.value()

    @always_inline
    def device_ptr[
        T: DType
    ](self, offset: Int = 0) -> UnsafePointer[Scalar[T], MutAnyOrigin]:
        """Return a typed device pointer into GPU memory at element offset.

        Precondition: `is_device()` must be True.
        """
        return self.device_buffer().unsafe_ptr().bitcast[Scalar[T]]() + offset

    # TODO: maybe should check for host buffer to copy that as well over the device
    def to_device(self, ctx: DeviceContext) raises -> Buffer:
        """Upload this CPU-accessible buffer to the GPU.

        Returns a new DEVICE buffer with the same `device_id` as the context
        device and `device_type` derived from the context (currently hardcoded
        to 2 for CUDA; update when Mojo exposes per-context device-type queries).

        Precondition: `is_cpu()` must be True (CPU, FOREIGN, or HOST).

        Returns:
            A new Buffer with kind=DEVICE containing the uploaded data.
        """
        if self.is_device():
            raise Error("to_device: buffer is already on device")
        var dev = ctx.enqueue_create_buffer[DType.uint8](self.size)
        ctx.enqueue_copy(dev, self.ptr)
        return Buffer.from_device(dev, self.size)

    def to_cpu(self, ctx: DeviceContext) raises -> Buffer:
        """Download this DEVICE buffer to an owned CPU heap buffer.

        HOST (pinned) buffers are already CPU-accessible via `ptr`; this method
        is only needed for DEVICE buffers.

        Precondition: `is_device()` must be True.

        Returns:
            A new Buffer with kind=CPU containing the downloaded data.
        """
        if not self.is_device():
            raise Error("to_cpu: buffer is not on device")
        var builder = BufferBuilder.alloc(self.size)
        ctx.enqueue_copy(builder.ptr, self._owner[]._device.value())
        ctx.synchronize()
        return builder.finish()

    @always_inline
    def length[T: DType = DType.uint8](self) -> Int:
        comptime if T == DType.bool:
            return self.size * 8
        else:
            return self.size // size_of[T]()

    @always_inline
    def unsafe_ptr[
        T: DType = DType.uint8
    ](self, offset: Int = 0) -> UnsafePointer[Scalar[T], ImmutExternalOrigin]:
        """Return a typed pointer to the element at offset.

        Precondition: `is_cpu()` must be True.
        """
        debug_assert(
            self.is_cpu(),
            "cannot read device buffer, call to_cpu() first",
        )
        return self.ptr.bitcast[Scalar[T]]() + offset

    @always_inline
    def unsafe_get[T: DType = DType.uint8](self, index: Int) -> Scalar[T]:
        debug_assert(
            self.is_cpu(),
            "cannot read device buffer, call to_cpu() first",
        )
        comptime output = Scalar[T]
        comptime if T == DType.bool:
            var byte = self.ptr[index // 8]
            return output((byte >> UInt8(index % 8)) & 1)
        else:
            return self.ptr.bitcast[output]()[index]

    @always_inline
    def simd_load[T: DType, W: Int](self, index: Int) -> SIMD[T, W]:
        """Load W elements of type T at element index `index`.

        Precondition: `is_cpu()` must be True.
        """
        debug_assert(
            self.is_cpu(),
            "cannot read device buffer, call to_cpu() first",
        )
        return (self.ptr.bitcast[Scalar[T]]() + index).load[width=W]()

    def write_to[W: Writer](self, mut writer: W):
        """Write the buffer's bytes to a Writer."""
        writer.write(t"Buffer(ptr={self.ptr}, size={self.size})")
