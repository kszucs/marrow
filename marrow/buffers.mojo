"""Arrow-compatible memory buffers with parametric mutability.

Buffer Kinds
------------
A `Buffer` has one of five memory kinds, encoded in the `Allocation` it owns:

  CPU
      Owned Mojo heap allocation.  Created by `Buffer.alloc_zeroed()` and
      similar factory methods.  `Allocation.__del__` calls `ptr.free()`.

  FOREIGN
      External CPU memory provided by a producer (Arrow C Data Interface or Arrow C
      Device Data Interface with device_type=CPU).  A custom release callback stored
      in the `Allocation` is invoked when the last `Buffer` view is dropped.
      Multiple `Buffer` views share the *same* `ArcPointer[Allocation]` (the "keeper")
      so the release fires exactly once when the last view is destroyed.

  MAPPED
      A memory-mapped file region.  Created by `Buffer.mmap_file(path)`;
      `Allocation.__del__` unmaps it when the last `Buffer` view is dropped.
      This is what lets a mapped file be *owned* by a `Buffer` rather than
      borrowed from something that must be kept alive alongside it — the
      distinction that makes a zero-copy file read safe rather than merely
      fast.

  HOST
      Pinned CPU memory managed by Mojo's `HostBuffer` (AsyncRT reference-counted).
      CPU-accessible (the `_ptr` field is valid) and fast for DMA to/from the GPU.
      Created via `Buffer.from_host()`.  The `Allocation._host` Optional field owns
      the `HostBuffer`; its destructor cascades to `AsyncRT_DeviceBuffer_release`.

  DEVICE
      GPU device memory managed by Mojo's `DeviceBuffer` (AsyncRT reference-counted).
      NOT CPU-accessible (`_ptr` is null for `Buffer[mut=False]`).  Created via
      `Buffer.from_device()` or `Buffer.alloc_device()`.

CPU accessibility
-----------------
For `Buffer[mut=False]`: `_ptr` is non-null for CPU/FOREIGN/HOST; null for DEVICE.
`is_cpu()`, `is_device()` and `is_host()` forward to `Allocation`, which is the
only thing that knows which kind it holds.

For `Buffer[mut=True]`: `_ptr` holds the mutable allocation pointer (CPU heap, pinned
host, or GPU device pointer).  Use `is_cpu()` / `is_device()` only on `Buffer[mut=False]`.

Ownership Model
---------------
`Buffer[mut=False]` is `ImplicitlyCopyable`: copying a Buffer is O(1) and bumps the
`ArcPointer[Allocation]` reference count.  The backing memory is freed / released
only when the *last* copy is dropped.

`Buffer[mut=True]` is the mutable counterpart — it exclusively owns a writable
pointer.  `Buffer[mut=True].to_immutable()` transfers that pointer into an owned CPU/HOST/DEVICE
`Buffer[mut=False]`.  Copying a `Buffer[mut=True]` is a compile-time error.

Both modes share the same struct layout: `(size, ptr, _owner)`.  The `ArcPointer[Allocation]`
is created eagerly at allocation time so `to_immutable()` is a zero-cost type conversion.

Allocation Invariant
--------------------
Each `Allocation` has exactly one active release mechanism (checked in `__del__`,
in this order):
  - `release is Some`      → FOREIGN: invoke the producer's C release callback.
  - `_mapped_size is Some` → MAPPED: `munmap(ptr, size)`.
  - `ptr is non-null`      → CPU: call `ptr.free()` directly (no callback).
  - `_host is Some`        → HOST: `HostBuffer.__del__` cascades to AsyncRT release.
  - `_device is Some`      → DEVICE: `DeviceBuffer.__del__` cascades to AsyncRT release.

MAPPED is checked before CPU because both have a non-null `ptr`.

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

Buffer lifecycle
-----------------
CPU heap allocation (kind=CPU):
  1. `var b = Buffer.alloc_zeroed[T](n)` — 64-byte-aligned heap allocation.
  2. `b.unsafe_set(i, v)` / `b.simd_store(...)` — write through the mutable pointer.
  3. `var buf = b.to_immutable()` — zero-cost transfer into an immutable CPU Buffer.

Pinned host allocation (kind=HOST):
  1. `var b = Buffer.alloc_host[T](ctx, n)` — page-locked allocation via DeviceContext.
  2. `b.unsafe_set(i, v)` / `b.simd_store(...)` — write through the mutable pointer.
  3. `var buf = b.to_immutable()` — transfer into an immutable HOST Buffer.

Bitmap operations
-----------------
Validity bitmaps use the dedicated `Bitmap` / `BitmapBuilder` types from
this module, which wrap `Buffer[mut=False]` / `Buffer[mut=True]` with bit-level
and SIMD bulk operations.
"""

from std.builtin.builtin_slice import ContiguousSlice
from std.ffi import external_call
from std.io.file import FileHandle
from std.memory import (
    unsafe_memset_zero,
    unsafe_memcpy,
    unsafe_memset,
    ArcPointer,
)
from std.memory.alloc import unsafe_alloc
from std.sys.info import simd_byte_width
from std.sys import size_of
import std.math as math
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from .views import (
    BufferView,
    BitmapView,
)


struct DeviceType:
    """Device type constants from the Arrow C Device Data Interface / DLPack.

    Use these when constructing HOST or DEVICE buffers (`from_host`,
    `from_device`, `Buffer.finish`) and when exporting via
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

    Release rules (in `__del__`, checked in this order):
      - `release is Some`      → FOREIGN: invoke the producer's C callback.
      - `_mapped_size is Some` → MAPPED: `munmap(ptr, size)`.
      - `ptr is non-null`      → CPU: call `ptr.free()`.
      - `_host is Some`        → HOST: HostBuffer.__del__ cascades to AsyncRT release.
      - `_device is Some`      → DEVICE: DeviceBuffer.__del__ cascades to AsyncRT release.

    Order matters between MAPPED and CPU: both carry a non-null `ptr`, and
    `free()`-ing mapped memory is undefined.

    Always accessed through `ArcPointer[Allocation]` so that multiple `Buffer`
    views can share ownership.  Lifetime: the last ArcPointer to drop triggers
    `__del__`, which fires the appropriate release.

    Use the static factory methods (`cpu`, `foreign`, `mapped`, `host`,
    `device`) rather than the raw `__init__`.
    """

    var ptr: Optional[Pointer[UInt8, MutUntrackedOrigin]]
    """Raw CPU pointer.  Some for CPU and FOREIGN; None for HOST/DEVICE."""

    var release: Optional[def(Pointer[UInt8, MutUntrackedOrigin]) thin -> None]
    """Release callback.  Set for CPU (_cpu_release) and FOREIGN (producer callback);
    None for HOST and DEVICE (their Optional field destructors handle release)."""

    var _host: Optional[HostBuffer[DType.uint8]]
    """Pinned host buffer.  Set only for HOST kind; None otherwise."""

    var _device: Optional[DeviceBuffer[DType.uint8]]
    """GPU device buffer.  Set only for DEVICE kind; None otherwise."""

    var _mapped_size: Optional[Int]
    """Mapped byte length.  Set only for MAPPED kind; None otherwise.

    `munmap` needs the extent and cannot recover it from the pointer, so a
    mapping's length is part of what identifies the kind — the same way `_host`
    and `_device` identify theirs.  It is not a general "size of this
    allocation": CPU and FOREIGN do not track one because neither release needs
    one."""

    def __init__(
        out self,
        ptr: Optional[Pointer[UInt8, MutUntrackedOrigin]],
        release: Optional[def(Pointer[UInt8, MutUntrackedOrigin]) thin -> None],
        host: Optional[HostBuffer[DType.uint8]],
        device: Optional[DeviceBuffer[DType.uint8]],
        mapped_size: Optional[Int] = None,
    ):
        self.ptr = ptr
        self.release = release
        self._host = host
        self._device = device
        self._mapped_size = mapped_size

    @always_inline
    def is_host(self) -> Bool:
        """Pinned host memory (HOST kind) — CPU-addressable, page-locked."""
        return Bool(self._host)

    @always_inline
    def is_device(self) -> Bool:
        """GPU device memory (DEVICE kind) — *not* CPU-addressable."""
        return Bool(self._device)

    def host_context(self) raises -> DeviceContext:
        """The context this allocation was pinned with (HOST kind only)."""
        if not self._host:
            raise Error("Allocation.host_context: not a pinned host allocation")
        return self._host.value().context()

    def mapped_size(self) raises -> Int:
        """The mapping's true extent in bytes (MAPPED kind only).

        Distinct from the owning `Buffer`'s logical size, which is padded up to
        Arrow's 64 bytes. A caller addressing *file* offsets — the Parquet
        footer does — wants this one."""
        if not self._mapped_size:
            raise Error("Allocation.mapped_size: not a mapped allocation")
        return self._mapped_size.value()

    @staticmethod
    def cpu(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> Allocation:
        """Create an owned CPU allocation.  `__del__` calls `ptr.free()`."""
        return Allocation(Optional(ptr), None, None, None)

    @staticmethod
    def foreign(
        ptr: Pointer[UInt8, MutUntrackedOrigin],
        release: def(Pointer[UInt8, MutUntrackedOrigin]) thin -> None,
    ) -> Allocation:
        """Create a foreign CPU allocation with a custom release callback."""
        return Allocation(Optional(ptr), release, None, None)

    @staticmethod
    def mapped(
        ptr: Pointer[UInt8, MutUntrackedOrigin], size: Int
    ) -> Allocation:
        """Create a MAPPED allocation over an existing memory mapping.

        `__del__` unmaps it. The caller does the `mmap` — which file, which
        flags, which offset is the mapper's business — and hands the result
        here; this owns the release, exactly as HOST/DEVICE allocations are
        created through a `DeviceContext` elsewhere and released here.

        `size` must be the length the mapping was created with, not the extent
        any one `Buffer` view addresses: a mapping is unmapped whole.
        """
        return Allocation(Optional(ptr), None, None, None, Optional(size))

    @staticmethod
    def host(host_buf: HostBuffer[DType.uint8]) -> Allocation:
        """Create a HOST (pinned) allocation.  HostBuffer.__del__ handles release.
        """
        return Allocation(None, None, host_buf, None)

    @staticmethod
    def device(dev_buf: DeviceBuffer[DType.uint8]) -> Allocation:
        """Create a DEVICE (GPU) allocation.  DeviceBuffer.__del__ handles release.
        """
        return Allocation(None, None, None, dev_buf)

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

    def __deinit__(deinit self):
        if self.release:
            # FOREIGN: invoke the producer's C release callback.
            self.release.value()(self.ptr.value())
        elif self._mapped_size:
            # MAPPED: unmap the region. Must be checked *before* the CPU branch
            # — a mapping has a non-null `ptr` too, and `ptr.free()` on mapped
            # memory is undefined.
            _ = external_call["munmap", Int32](
                self.ptr.value(), self._mapped_size.value()
            )
        elif self.ptr:
            # CPU: free the Mojo heap allocation directly.
            # HOST and DEVICE have ptr=None, so this branch is CPU-only.
            self.ptr.value().unsafe_free()
        # HOST/DEVICE: ptr=None; Optional field destructors cascade to AsyncRT release.


# ---------------------------------------------------------------------------
# Buffer — unified mutable/immutable buffer with parametric mutability
# ---------------------------------------------------------------------------


struct Buffer[*, mut: Bool = False](
    ImplicitlyCopyable, Movable, Sized, Writable
):
    """Contiguous memory region with parametric mutability.

    `Buffer[mut=True]`  — mutable, exclusively owned.  Use `alloc_*` factory
                          methods to allocate; write via `unsafe_set` /
                          `simd_store`; freeze with `to_immutable()`.

    `Buffer[mut=False]` — immutable, shared ownership.  Copying is O(1) via
                          `ArcPointer[Allocation]` ref-counting.

    Both modes share the same three-field layout `(_size, _ptr, _owner)`.
    The `ArcPointer[Allocation]` is created eagerly at allocation time so
    `to_immutable()` is a near-zero-cost type rebind (one ArcPointer copy).

    CPU accessibility (mut=False only):
      `is_cpu()` returns True for CPU, FOREIGN, and HOST kinds (ptr non-null).
      `is_device()` returns True for DEVICE kind (ptr null).
      Call `to_cpu(ctx)` before reading a DEVICE buffer on the CPU.
    """

    var _ptr: Pointer[UInt8, UntrackedOrigin[mut=Self.mut]]
    """Raw allocation pointer.
    For `mut=True` CPU/HOST allocations: the CPU-accessible data pointer.
    For `mut=True` DEVICE allocations: the GPU device pointer (used by kernels).
    For `mut=False` CPU/HOST allocations: the CPU-accessible data pointer.
    For `mut=False` DEVICE allocations: null (no CPU access; use device_ptr()).
    Use `unsafe_ptr()` for raw access; prefer `ptr_at()`, `view()`, or `BufferView` methods.
    """

    var _size: Int
    """Buffer size in bytes — always a multiple of 64 (`_aligned_size`).

    A multiple of 64 is *not* spare room: when the logical byte count is
    already a multiple of 64 the allocation ends at the last live byte.
    """

    var _owner: ArcPointer[Allocation]
    """Shared ownership handle.  Ref-count is 1 for `mut=True` (exclusive)."""

    # --- Lifecycle ---

    def __init__(
        out self,
        size: Int,
        ptr: Pointer[UInt8, UntrackedOrigin[mut=Self.mut]],
        owner: ArcPointer[Allocation],
    ):
        debug_assert(
            Int(ptr) % 64 == 0 or Int(ptr) == 0,
            "Buffer pointer must be 64-byte aligned",
        )
        debug_assert(
            size % 64 == 0 or size == 0,
            "Buffer size must be 64-byte aligned",
        )
        self._size = size
        self._ptr = ptr
        self._owner = owner

    def __init__(out self, *, copy: Self):
        comptime assert (
            not Self.mut
        ), "cannot copy mutable Buffer[mut=True]; call to_immutable() to freeze"
        self._size = copy._size
        self._ptr = copy._ptr
        self._owner = copy._owner

    @staticmethod
    def _aligned_size[T: DType](length: Int) -> Int:
        """Allocation size in bytes for ``length`` elements of ``T``.

        This is Arrow's padding, and it is the same rule Arrow C++ applies —
        `PoolBuffer::RoundCapacity` is `bit_util::RoundUpToMultipleOf64`, so
        `AllocateBuffer(n)` allocates `align_up(n, 64)` bytes and zero-fills
        the difference. The Columnar spec asks implementations to "pad
        (overallocate) to a length that is a multiple of 8 or 64 bytes", and
        that is exactly what rounding up to a multiple does.

        **It is not slack.** When ``length * size_of[T]()`` is already a
        multiple of 64 the allocation ends precisely at the last live byte —
        an `int64` buffer of 8 elements is 64 bytes with nothing behind it. Any
        code that reads or writes past the logical end "because Arrow buffers
        are padded" is wrong, and one such write was a heap overflow that
        corrupted tcmalloc's freelist (see `BufferView.compressed_store_dense`
        and docs/alpha-findings/f1-distinct-segfault.md). The invariant this
        upholds is *alignment of the size*, nothing more.
        """
        return math.align_up(length * size_of[T](), 64)

    # --- Mutable factory methods (return Buffer[mut=True]) ---

    @staticmethod
    def alloc_zeroed[
        I: Intable, //, T: DType = DType.uint8
    ](length: I) -> Buffer[mut=True]:
        """Allocate a 64-byte-aligned, zero-filled buffer for `length` elements of type T.
        """
        var result = Buffer.alloc_uninit[T](length)
        unsafe_memset_zero(result._ptr, result._size)
        return result^

    @staticmethod
    def alloc_filled[
        I: Intable, //, T: DType = DType.uint8
    ](length: I, fill: Scalar[T]) -> Buffer[mut=True]:
        """Allocate a 64-byte-aligned buffer filled with ``fill``."""
        var result = Buffer.alloc_uninit[T](length)
        unsafe_memset(result._ptr, UInt8(fill), result._size)
        return result^

    @staticmethod
    def alloc_uninit[
        I: Intable, //, T: DType = DType.uint8
    ](length: I) -> Buffer[mut=True]:
        """Allocate a 64-byte-aligned buffer for ``length`` elements of type T
        without zero-filling.

        Use only when the caller guarantees every element will be written
        before the buffer is read.
        """
        var size = Buffer._aligned_size[T](Int(length))
        var raw = unsafe_alloc[UInt8](size, alignment=64)
        var ptr = rebind[Pointer[UInt8, MutUntrackedOrigin]](raw)
        return Buffer[mut=True](
            size=size,
            ptr=rebind[Pointer[UInt8, MutUntrackedOrigin]](ptr),
            owner=ArcPointer(Allocation.cpu(ptr)),
        )

    @staticmethod
    def alloc_host[
        I: Intable, //, T: DType = DType.uint8
    ](ctx: DeviceContext, length: I) raises -> Buffer[mut=True]:
        """Allocate page-locked (pinned) host memory for `length` elements of type T.

        Pinned memory is CPU-accessible and enables fast DMA transfers to/from
        the GPU.  Use `unsafe_set()` / `simd_store()` to write, then call
        `to_immutable()` to obtain an immutable HOST Buffer.

        Args:
            ctx:    DeviceContext used to allocate the HostBuffer.
            length: Number of elements.

        Returns:
            A mutable Buffer backed by pinned host memory.
        """
        var byte_size = Buffer._aligned_size[T](Int(length))
        var host = ctx.enqueue_create_host_buffer[DType.uint8](byte_size)
        var ptr = rebind[Pointer[UInt8, MutUntrackedOrigin]](host.unsafe_ptr())
        unsafe_memset_zero(ptr, byte_size)
        return Buffer[mut=True](
            size=byte_size,
            ptr=rebind[Pointer[UInt8, MutUntrackedOrigin]](ptr),
            owner=ArcPointer(Allocation.host(host)),
        )

    @staticmethod
    def alloc_device[
        I: Intable, //, T: DType = DType.uint8
    ](ctx: DeviceContext, length: I) raises -> Buffer[mut=True]:
        """Allocate a device (GPU) buffer for `length` elements of type T.

        The returned buffer exposes `ptr` as a `MutUntrackedOrigin` device pointer
        suitable for GPU kernel writes. Call `to_immutable()` to obtain an immutable
        device-resident `Buffer[mut=False]`.
        """
        var byte_size = Buffer._aligned_size[T](Int(length))
        var dev = ctx.enqueue_create_buffer[DType.uint8](byte_size)
        var ptr = rebind[Pointer[UInt8, MutUntrackedOrigin]](dev.unsafe_ptr())
        return Buffer[mut=True](
            size=byte_size,
            ptr=rebind[Pointer[UInt8, MutUntrackedOrigin]](ptr),
            owner=ArcPointer(Allocation.device(dev)),
        )

    # --- Immutable factory methods (return Buffer[mut=False]) ---

    @staticmethod
    def mmap_file(path: String) raises -> Buffer[mut=False]:
        """Memory-map a whole file read-only, owned by the returned Buffer.

        The mapping lives exactly as long as the `Buffer`s that reference it:
        the last one dropped unmaps. Nothing has to be kept alive alongside it,
        which is the difference between this and handing out borrowed spans over
        a mapping some other object owns.

        This sits next to `alloc_host` / `alloc_device` on purpose — creating
        the memory and naming its `Allocation` kind belong together, and every
        other kind is already built here. It is also what makes the mapping's
        extent unforgeable: `Allocation.mapped` needs a size that matches the
        real mapping, and only the code that called `mmap` knows it.

        Zero-copy: no file bytes are read here; the kernel faults pages in on
        access.

        Two sizes are in play and they are deliberately different. The
        `Allocation` records the **true** file length, because that is what
        `munmap` must be given. The `Buffer`'s logical size is that rounded up
        to Arrow's 64-byte padding, as `from_foreign` does — safe because `mmap`
        rounds the mapping up to a whole page and bytes past EOF within it read
        as zero, so the padding is always addressable.
        """
        var f = FileHandle(path, "r")
        var size = Int(
            external_call["lseek", Int64](f.handle, Int64(0), Int(2))
        )  # SEEK_END
        if size <= 0:
            raise Error("Buffer.mmap_file: empty or unreadable file ", path)
        # PROT_READ=1, MAP_PRIVATE=2; the mapping outlives the fd.
        var ptr = external_call["mmap", Pointer[UInt8, MutUntrackedOrigin]](
            UInt(0), size, Int32(1), Int32(2), Int32(f.handle), Int64(0)
        )
        _ = f^  # close the fd; the mapping stays valid
        if Int(ptr) == 0 or Int(ptr) == -1:
            raise Error("Buffer.mmap_file: mmap failed for ", path)
        return Buffer[mut=False](
            size=math.align_up(size, 64),
            ptr=ptr,
            owner=ArcPointer(Allocation.mapped(ptr, size)),
        )

    @staticmethod
    def from_foreign[
        I: Intable, //
    ](
        ptr: OpaquePointer[MutUntrackedOrigin],
        size: I,
        owner: ArcPointer[Allocation],
    ) -> Buffer[mut=False]:
        """Create an immutable view into foreign CPU memory.

        The caller passes an `ArcPointer[Allocation]` (the "keeper") that holds
        the producer's release callback.  All `Buffer` views sharing the same
        keeper bump its ref-count on copy; when the last view drops, the keeper
        releases and the C callback fires automatically.

        The logical size is rounded up to a 64-byte multiple so that `len()`
        reads the same as it does for an owned buffer.

        **That rounding is a convention, not a guarantee about the memory.**
        The C Data Interface spec makes even *alignment* "recommended, but not
        required" and says nothing at all about padding, so an imported buffer
        may end exactly at its logical last byte. Nothing may read or write
        past `size` on a FOREIGN buffer on the strength of this rounding; see
        docs/alpha-findings/g1-buffer-invariants.md.

        Precondition: `owner` must have been created with `Allocation.foreign(...)`.
        """
        return Buffer[mut=False](
            size=math.align_up(Int(size), 64),
            ptr=rebind[Pointer[UInt8, ImmUntrackedOrigin]](ptr),
            owner=owner,
        )

    @staticmethod
    def from_host(host: HostBuffer[DType.uint8]) -> Buffer[mut=False]:
        """Create a HOST (pinned) buffer from a Mojo HostBuffer.

        The HostBuffer is moved into an `Allocation` behind `ArcPointer`;
        its destructor cascades to `AsyncRT_DeviceBuffer_release` when the
        last Buffer copy is dropped.

        The CPU pointer is taken from `host.unsafe_ptr()` — it remains valid
        for the lifetime of the Allocation.  `device_type()` is inferred from
        the context API (cuda→CUDA_HOST, hip→ROCM_HOST, otherwise CPU).
        """
        var ptr = rebind[Pointer[UInt8, ImmUntrackedOrigin]](host.unsafe_ptr())
        return Buffer[mut=False](
            size=len(host),
            ptr=ptr,
            owner=ArcPointer(Allocation.host(host)),
        )

    @staticmethod
    def from_device(
        dev: DeviceBuffer[DType.uint8], size: Int
    ) -> Buffer[mut=False]:
        """Create a DEVICE (GPU) buffer from a Mojo DeviceBuffer.

        The DeviceBuffer is moved into an `Allocation` behind `ArcPointer`;
        its destructor cascades to `AsyncRT_DeviceBuffer_release` when the
        last Buffer copy is dropped.

        `ptr` holds the device pointer — call `to_cpu(ctx)` to read on CPU.
        `device_type()` is inferred from the context API (cuda→CUDA, hip→ROCM,
        metal→METAL).
        """
        var ptr = rebind[Pointer[UInt8, ImmUntrackedOrigin]](dev.unsafe_ptr())
        return Buffer[mut=False](
            size=size,
            ptr=ptr,
            owner=ArcPointer(Allocation.device(dev)),
        )

    # --- Mutability transition ---

    def to_immutable(deinit self) -> Buffer[mut=False] where Self.mut:
        """Consume the mutable buffer and return an immutable Buffer.

        For CPU buffers (`alloc_zeroed`, `alloc_uninit`): returns kind=CPU.
        For HOST buffers (`alloc_host`): returns kind=HOST.
        For DEVICE buffers (`alloc_device`): returns kind=DEVICE; ``ptr`` holds
        the device pointer so ``view()`` works without a separate ``device_view``
        call.
        """
        var imm_ptr = rebind[Pointer[UInt8, ImmUntrackedOrigin]](self._ptr)
        return Buffer[mut=False](
            size=self._size, ptr=imm_ptr, owner=self._owner^
        )

    # --- CPU/device checks (mut=False only) ---

    @always_inline
    def is_cpu(self) -> Bool:
        """Return True if the buffer is CPU-accessible.

        True for CPU, FOREIGN, and HOST kinds; False for DEVICE.
        """
        return not self._owner[].is_device()

    @always_inline
    def is_device(self) -> Bool:
        """Return True if the buffer lives on a GPU device."""
        return self._owner[].is_device()

    @always_inline
    def is_host(self) -> Bool:
        """Return True if the buffer is pinned host memory (HOST kind)."""
        return self._owner[].is_host()

    def mapped_size(self) raises -> Int:
        """The backing mapping's true extent in bytes (MAPPED kind only).

        `len(self)` is the *padded* logical size; this is the file length the
        mapping was made with. Delegates to `Allocation.mapped_size()`."""
        return self._owner[].mapped_size()

    # --- Length helper (both modes) ---

    @always_inline
    def length[T: DType = DType.uint8](self) -> Int:
        comptime if T == DType.bool:
            return self._size * 8
        else:
            return self._size // size_of[T]()

    # --- Write operations (mut=True only) ---

    # TODO(MOCO-4220): `where Self.mut` does not refine `Self` to
    # `Buffer[mut=True]` in the body, so `swap(self, new)` below cannot
    # typecheck.  Drop the decorator once the refinement lands.
    @__allow_legacy_custom_self_type
    def resize[
        I: Intable, //, T: DType = DType.uint8
    ](mut self: Buffer[mut=True], length: I) raises:
        """Resize the buffer to hold `length` elements of type T.

        For HOST buffers the new allocation is also pinned host memory using
        the same `DeviceContext`; for CPU buffers a plain heap allocation is used.

        No-op if the new size maps to the same byte allocation as the current one.
        """
        var byte_size = Buffer._aligned_size[T](Int(length))
        if byte_size == self._size:
            return
        if self._owner[].is_device():
            # The copy below reads through `_ptr`, which a DEVICE allocation
            # does not have — growing one has to go through the device API.
            raise Error(
                "Buffer.resize: device memory cannot be resized; download with"
                " to_host(ctx), resize, and upload again"
            )
        var new: Buffer[mut=True]
        if self._owner[].is_host():
            new = Buffer.alloc_host[T](self._owner[].host_context(), length)
        else:
            new = Buffer.alloc_zeroed[T](length)
        unsafe_memcpy(
            dest=new._ptr, src=self._ptr, count=min(new._size, self._size)
        )
        swap(self, new)

    def extend[
        T: DType,
        src_origin: Origin,
    ](
        mut self,
        src: BufferView[T, src_origin],
        dst_offset: Int,
        count: Int,
    ) where Self.mut:
        """Copy `count` elements of type T from `src` into self at `dst_offset`.
        """
        unsafe_memcpy(
            dest=self._ptr.unsafe_mut_cast[True]()
            .unsafe_bitcast[Scalar[T]]()
            .unsafe_offset(dst_offset),
            src=src._data,
            count=count,
        )

    @always_inline
    def _check_bounds[T: DType](self, index: Int):
        debug_assert(
            0 <= index < self.length[T](),
            "Buffer index ",
            index,
            " out of bounds for length ",
            self.length[T](),
        )

    @always_inline
    def unsafe_set[
        T: DType = DType.uint8
    ](self, index: Int, value: Scalar[T]) where Self.mut:
        """Write `value` at element `index`, striding by `size_of[T]()`.

        **Always type the value.** `T` is inferred from `value`, so a bare
        integer literal does *not* give you the `uint8` default — it widens, and
        the write strides by the wider type. `unsafe_get` has no argument to
        infer from, so it *does* default to `uint8`. The pair then silently
        disagrees:

            buf.unsafe_set(1, 13)          # 8-byte store at byte offset 8
            buf.unsafe_get(1)              # reads byte 1 -> 0, not 13

        Write `unsafe_set(1, UInt8(13))` or `unsafe_set[DType.uint8](1, 13)`.
        This cost two GPU tests, which read back zeros and were filed as a
        device-transfer data-loss bug (B22) until the writes turned out to be
        landing eight bytes away from the reads.

        The bound is checked by `debug_assert`, so it costs nothing in release —
        "unsafe" promises no *runtime* check, not that a debug build should stay
        quiet while the caller overruns. Note the bound is the buffer's
        *allocated* element count, which `_aligned_size` rounds up to a 64-byte
        multiple, so it is looser than a logical row count by up to 63 bytes.
        Size a `view()` explicitly when the destination comes from a computed
        count; see docs/alpha-findings/g1-buffer-invariants.md.
        """
        self._check_bounds[T](index)
        comptime output = Scalar[T]
        self._ptr.unsafe_mut_cast[True]().unsafe_bitcast[output]()[
            unsafe_offset=index
        ] = value

    # --- Read operations (both modes) ---

    @always_inline
    def unsafe_get[T: DType = DType.uint8](self, index: Int) -> Scalar[T]:
        debug_assert(
            self.is_cpu(),
            "cannot read device buffer, call to_cpu() first",
        )
        self._check_bounds[T](index)
        comptime output = Scalar[T]
        return self._ptr.unsafe_bitcast[output]()[unsafe_offset=index]

    # TODO: remove this method in favor of `view()` and `BufferView` for both CPU and DEVICE buffers.
    @always_inline
    def device_view[
        T: DType = DType.uint8
    ](ref self, offset: Int = 0) -> BufferView[T, origin_of(self)]:
        """Typed view backed by the GPU device pointer at element ``offset``.

        The origin is tied to ``self`` (``origin_of(self)``) so the view keeps
        the backing device buffer alive for the duration of any GPU kernel that
        captures it. An ``UntrackedOrigin`` view would let ASAP free the buffer
        before the kernel runs, so the kernel would read/write freed memory.
        The view's mutability follows ``self``. Precondition: ``is_device()``.
        """
        var ptr = rebind[Pointer[Scalar[T], origin_of(self)]](
            self._ptr.unsafe_bitcast[Scalar[T]]().unsafe_offset(offset)
        )
        return BufferView(ptr=ptr, length=(self._size // size_of[T]()) - offset)

    # --- Device type / id ---

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

    # --- Transfer (mut=False only) ---

    def to_device(
        self, ctx: DeviceContext
    ) raises -> Buffer[mut=False] where not Self.mut:
        """Upload this CPU-accessible buffer to the GPU.

        Returns a new DEVICE buffer with the same `device_id` as the context
        device.

        Precondition: `is_cpu()` must be True (CPU, FOREIGN, or HOST).

        Returns:
            A new Buffer with kind=DEVICE containing the uploaded data.
        """
        if self.is_device():
            raise Error("to_device: buffer is already on device")
        var dev = ctx.enqueue_create_buffer[DType.uint8](self._size)
        ctx.enqueue_copy(
            dev, rebind[Pointer[UInt8, ImmUntrackedOrigin]](self._ptr)
        )
        return Buffer.from_device(dev, self._size)

    def to_cpu(
        self, ctx: DeviceContext
    ) raises -> Buffer[mut=False] where not Self.mut:
        """Download this DEVICE buffer to an owned CPU heap buffer.

        HOST (pinned) buffers are already CPU-accessible via `_ptr`; this method
        is only needed for DEVICE buffers.

        Precondition: `is_device()` must be True.

        Returns:
            A new Buffer with kind=CPU containing the downloaded data.
        """
        if not self.is_device():
            raise Error("to_cpu: buffer is not on device")
        var builder = Buffer.alloc_zeroed(self._size)
        ctx.enqueue_copy(
            rebind[Pointer[UInt8, MutUntrackedOrigin]](builder._ptr),
            self._owner[]._device.value(),
        )
        ctx.synchronize()
        return builder^.to_immutable()

    def __eq__[m: Bool](self, other: Buffer[mut=m]) -> Bool:
        """Compare two buffers byte-by-byte (64-bit chunks for speed)."""
        if self._size != other._size:
            return False
        var lhs = self._ptr.unsafe_bitcast[UInt64]()
        var rhs = other._ptr.unsafe_bitcast[UInt64]()
        for i in range(self._size // 8):
            if lhs[unsafe_offset=i] != rhs[unsafe_offset=i]:
                return False
        return True

    def write_to[W: Writer](self, mut writer: W):
        """Write the buffer's bytes to a Writer."""
        writer.write(t"Buffer(ptr={self._ptr}, size={self._size})")

    def view[
        T: DType = DType.uint8
    ](ref self, offset: Int = 0, length: Int = -1) -> BufferView[
        T, origin_of(self)
    ]:
        """Return a non-owning typed view over this buffer.

        ``_ptr`` always holds the right pointer — CPU address for CPU/HOST
        buffers, device address for DEVICE buffers — so no dispatch is needed.
        ``offset`` and ``length`` are in units of ``T`` elements (bytes when
        ``T=uint8``).
        """
        var n = length if length >= 0 else (self._size // size_of[T]()) - offset
        var ptr = rebind[Pointer[Scalar[T], origin_of(self)]](self._ptr)
        return BufferView(ptr=ptr.unsafe_offset(offset), length=n)

    def slice[
        T: DType = DType.uint8
    ](ref self, offset: Int, length: Int) -> BufferView[T, origin_of(self)]:
        """Return a non-owning typed view of `length` T-elements starting at `offset`.
        """
        return self.view[T](offset, length)

    def __len__(self) -> Int:
        """Return the buffer size in bytes."""
        return self._size

    def __getitem__[T: DType = DType.uint8](self, index: Int) -> Scalar[T]:
        """Return the byte at `index`."""
        self._check_bounds[T](index)
        return self.unsafe_get[T](index)

    def __setitem__[
        T: DType = DType.uint8
    ](self, index: Int, value: Scalar[T]) where Self.mut:
        """Set the byte at `index` to `value`."""
        self._check_bounds[T](index)
        self.unsafe_set[T](index, value)

    def __getitem__[
        T: DType = DType.uint8
    ](ref self, slc: ContiguousSlice) -> BufferView[T, origin_of(self)]:
        """Return a view of the buffer for the given slice."""
        var length = self._size // size_of[T]()
        var start, end = slc.indices(length)
        return self.slice[T](start, end - start)


# ---------------------------------------------------------------------------
# Bitmap — bit-packed validity bitmap with parametric mutability
# ---------------------------------------------------------------------------


struct Bitmap[*, mut: Bool = False](
    ImplicitlyCopyable, Movable, Sized, Writable
):
    """Bit-packed validity bitmap with parametric mutability.

    `Bitmap[mut=True]`  — mutable builder. Use `alloc()` factory.
                          Write via `set`, `clear`, `set_range`,
                          `extend`, `resize`.
                          `to_immutable(length)` freezes to `Bitmap[mut=False]`.

    `Bitmap[mut=False]` — immutable, ref-counted shared ownership.
                          Copying is O(1). Use `slice()`.
    """

    var _buffer: Buffer[mut=Self.mut]
    var _length: Int

    def __init__(out self, var buffer: Buffer[mut=Self.mut]):
        """Construct a Bitmap from an existing buffer (length = buffer bytes * 8).
        """
        var n = len(buffer) * 8
        self._buffer = buffer^
        self._length = n

    def __init__(out self, var buffer: Buffer[mut=Self.mut], *, length: Int):
        """Construct a Bitmap with an explicit bit length."""
        self._buffer = buffer^
        self._length = length

    def __init__(out self, *, copy: Self):
        comptime assert not Self.mut, "cannot copy mutable Bitmap[mut=True]"
        self._buffer = copy._buffer
        self._length = copy._length

    def __init__(out self: Bitmap[mut=True], values: List[Bool]) raises:
        """Construct a mutable Bitmap from a list of boolean values."""
        self._buffer = Buffer.alloc_zeroed(math.ceildiv(len(values), 8))
        self._length = len(values)
        for i, ref v in enumerate(values):
            if v:
                self.unsafe_set(i)
            else:
                self.unsafe_clear(i)

    def __init__(
        out self: Bitmap[mut=True], length: Int, indices: List[Int]
    ) raises:
        """Construct a mutable Bitmap from a list of set bit indices."""
        self._buffer = Buffer.alloc_zeroed(math.ceildiv(length, 8))
        self._length = length
        for idx in indices:
            self.set(idx)

    # --- Factory ---

    @staticmethod
    def alloc_zeroed(capacity: Int) -> Bitmap[mut=True]:
        """Allocate a zero-filled mutable bitmap for `capacity` bits."""
        var byte_size = math.ceildiv(capacity, 8)
        var buffer = Buffer.alloc_zeroed(byte_size)
        return Bitmap[mut=True](buffer^, length=capacity)

    @staticmethod
    def alloc_uninit(capacity: Int) -> Bitmap[mut=True]:
        """Allocate an uninitialized mutable bitmap for `capacity` bits.

        Use only when every bit will be written before the bitmap is read.
        """
        var byte_size = math.ceildiv(capacity, 8)
        var buffer = Buffer.alloc_uninit(byte_size)
        return Bitmap[mut=True](buffer^, length=capacity)

    @staticmethod
    def alloc_device(
        ctx: DeviceContext, capacity: Int
    ) raises -> Bitmap[mut=True]:
        """Allocate a device (GPU) bitmap for `capacity` bits."""
        var byte_size = math.ceildiv(capacity, 8)
        var buffer = Buffer.alloc_device[DType.uint8](ctx, byte_size)
        return Bitmap[mut=True](buffer^, length=capacity)

    # --- Read methods (both modes) ---

    @always_inline
    def __len__(self) -> Int:
        return self._length

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Bitmap(length=", self._length, ")")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    @staticmethod
    def intersect_views[
        ao: Origin[mut=False], bo: Origin[mut=False]
    ](
        a: Optional[BitmapView[ao]], b: Optional[BitmapView[bo]]
    ) raises -> Optional[Bitmap[mut=False]]:
        """Bitwise AND of two optional validity **views**, offset applied.

        **This is the one to use in a kernel.** Views index logically from zero,
        owning `Bitmap`s do not, and a kernel's output is always `offset=0` — so
        combining raw `array.bitmap` values and attaching the result to an
        offset-0 array puts every null `offset` positions from where it belongs.
        `array.validity()` hands you the offset-applied view; this combines two
        of them.

        The one-sided cases go through `to_owned()` rather than returning the
        other operand directly, because a lone view still carries its parent's
        offset — `Bitmap.intersect`'s pass-through is exactly the case that
        looked safe and was not.

        `None` means all-valid, so it is the identity: the result is `None` only
        when both inputs are.
        """
        if not a and not b:
            return None
        if not a:
            return b.value().to_owned()
        if not b:
            return a.value().to_owned()
        return a.value().intersection(b.value()).to_immutable()

    # TODO: ensure that properly covered by tests
    @staticmethod
    def intersect(
        a: Optional[Bitmap[mut=False]], b: Optional[Bitmap[mut=False]]
    ) raises -> Optional[Bitmap[mut=False]]:
        """Bitwise AND of two optional validity bitmaps, **offset-unaware**.

        Both operands are used whole. That is correct only when neither carries
        an offset — i.e. neither array is a slice. A kernel building an
        `offset=0` output from sliced inputs wants `intersect_views` instead;
        this overload silently shifts the nulls. That was Q2.3, and it survived
        because the expression layer recomputes validity itself and so masked
        it from every expr-level test.

        `None` means all-valid, so it is the identity: the result is `None` only
        when both inputs are. Output bit i is set iff both inputs have it set.
        """
        if not a and not b:
            return None
        if not a:
            return b
        if not b:
            return a
        return (a.value().view() & b.value().view()).to_immutable()

    def view(
        ref self, offset: Int = 0, length: Int = -1
    ) -> BitmapView[origin_of(self)]:
        """Return a zero-copy view of the bitmap starting at `offset` for `length` bits.

        If `length` is -1 (the default), the view extends to the end of the bitmap.
        """
        var n = length if length >= 0 else self._length - offset
        var ptr = rebind[Pointer[UInt8, origin_of(self)]](self._buffer._ptr)
        # TODO: consider aligning _data down to a 64-byte boundary here and
        # folding the sub-alignment into _offset (matching BitmapView.slice()),
        # so that the SIMD bulk path in apply() always starts on a cache-line.
        return BitmapView(ptr=ptr, offset=offset, length=n)

    def slice(
        ref self, offset: Int, length: Int
    ) -> BitmapView[origin_of(self)]:
        """Return a zero-copy view of `length` bits starting at `offset`."""
        return self.view(offset, length)

    def __eq__[m: Bool](self, other: Bitmap[mut=m]) -> Bool:
        """Compare two bitmaps over their valid ranges.

        Bit comparison has one implementation, `BitmapView.__eq__` (word-level
        XOR); a bit-by-bit loop here measured ~64x slower for the same answer.
        """
        return self.view() == other.view()

    def unset_count(self) -> Int:
        """How many of these bits are 0 — the null count of a validity bitmap.
        """
        return self._length - self.view().count_set_bits()

    def byte_count(self) -> Int:
        """Return the size of the backing buffer in bytes."""
        return len(self._buffer)

    @always_inline
    def _check_bounds(self, index: Int):
        debug_assert(
            0 <= index < self._length,
            "Bitmap index ",
            index,
            " out of bounds for length ",
            self._length,
        )

    def set(mut self, index: Int) where Self.mut:
        """Set the bit at `index` to 1."""
        self._check_bounds(index)
        self.unsafe_set(index)

    @always_inline
    def unsafe_set(mut self, index: Int) where Self.mut:
        """Set the bit at `index` to 1."""
        var byte_index = index // 8
        var bit_mask = UInt8(1 << (index % 8))
        var ptr = self._buffer._ptr.unsafe_mut_cast[True]()
        ptr[unsafe_offset=byte_index] = ptr[unsafe_offset=byte_index] | bit_mask

    def clear(mut self, index: Int) where Self.mut:
        """Clear the bit at `index` to 0."""
        self._check_bounds(index)
        self.unsafe_clear(index)

    @always_inline
    def unsafe_clear(mut self, index: Int) where Self.mut:
        """Clear the bit at `index` to 0."""
        var byte_index = index // 8
        var bit_mask = UInt8(1 << (index % 8))
        var ptr = self._buffer._ptr.unsafe_mut_cast[True]()
        ptr[unsafe_offset=byte_index] = (
            ptr[unsafe_offset=byte_index] & ~bit_mask
        )

    def test(self, raw_index: Int) -> Bool:
        """Return True if the bit at `raw_index` (not offset-adjusted) is set.
        """
        self._check_bounds(raw_index)
        return self.unsafe_test(raw_index)

    @always_inline
    def unsafe_test(self, raw_index: Int) -> Bool:
        """Return True if the bit at `raw_index` (not offset-adjusted) is set.
        """
        var byte_index = raw_index // 8
        var bit_mask = UInt8(1 << (raw_index % 8))
        return (self._buffer._ptr[unsafe_offset=byte_index] & bit_mask) != 0

    @always_inline
    def __getitem__(self, index: Int) -> Bool:
        """Return the bit at logical `index` (0-based within this bitmap's window).
        """
        var i = index if index >= 0 else index + self._length
        self._check_bounds(i)
        return self.unsafe_test(i)

    @always_inline
    def __getitem__(
        self, slc: ContiguousSlice
    ) -> BitmapView[origin_of(self)] where not Self.mut:
        """Return a zero-copy sub-bitmap view for the given slice."""
        var start, end = slc.indices(self._length)
        return self.slice(start, end - start)

    def __setitem__(mut self, index: Int, value: Bool) where Self.mut:
        """Set or clear the bit at `index`."""
        var i = index if index >= 0 else index + self._length
        self._check_bounds(i)
        if value:
            self.unsafe_set(i)
        else:
            self.unsafe_clear(i)

    def set_range(
        mut self, start: Int, length: Int, value: Bool
    ) where Self.mut:
        """Set `length` bits starting at `start` to `value`."""
        if length == 0:
            return
        var end = start + length
        var start_byte = start >> 3
        var start_bit = start & 7
        var end_byte = end >> 3
        var end_bit = end & 7
        var fill = UInt8(255 if value else 0)
        var ptr = self._buffer._ptr.unsafe_mut_cast[True]()

        if start_byte == end_byte:
            var mask = UInt8((1 << end_bit) - 1) & (
                UInt8(0xFF) << UInt8(start_bit)
            )
            if value:
                ptr[unsafe_offset=start_byte] = (
                    ptr[unsafe_offset=start_byte] | mask
                )
            else:
                ptr[unsafe_offset=start_byte] = (
                    ptr[unsafe_offset=start_byte] & ~mask
                )
            return

        if start_bit != 0:
            var mask = UInt8(0xFF) << UInt8(start_bit)
            if value:
                ptr[unsafe_offset=start_byte] = (
                    ptr[unsafe_offset=start_byte] | mask
                )
            else:
                ptr[unsafe_offset=start_byte] = (
                    ptr[unsafe_offset=start_byte] & ~mask
                )
            start_byte += 1

        if end_bit != 0:
            var mask = UInt8((1 << end_bit) - 1)
            if value:
                ptr[unsafe_offset=end_byte] = ptr[unsafe_offset=end_byte] | mask
            else:
                ptr[unsafe_offset=end_byte] = (
                    ptr[unsafe_offset=end_byte] & ~mask
                )

        if end_byte > start_byte:
            unsafe_memset(
                ptr.unsafe_offset(start_byte), fill, end_byte - start_byte
            )

    def extend(
        mut self,
        src: BitmapView[_],
        dst_start: Int,
        length: Int,
    ) where Self.mut:
        """Copy `length` bits from `src` into self at `dst_start`.

        Three code paths:
        1. Same sub-byte alignment → unsafe_memcpy for middle bytes.
        2. Different alignment → shift-and-merge byte-by-byte.
        3. Short runs (< 16 bits) → bit-by-bit fallback.
        """
        if length == 0:
            return
        var dst = self._buffer._ptr.unsafe_mut_cast[True]()
        var dst_offset = dst_start
        var src_ptr = src._data
        var src_offset = src._offset

        if length < 16:
            for i in range(length):
                var s_byte = (src_offset + i) >> 3
                var s_bit = (src_offset + i) & 7
                var val = (src_ptr[unsafe_offset=s_byte] >> UInt8(s_bit)) & 1
                var d_byte = (dst_offset + i) >> 3
                var d_bit = (dst_offset + i) & 7
                var d_mask = UInt8(1 << d_bit)
                if val:
                    dst[unsafe_offset=d_byte] = (
                        dst[unsafe_offset=d_byte] | d_mask
                    )
                else:
                    dst[unsafe_offset=d_byte] = (
                        dst[unsafe_offset=d_byte] & ~d_mask
                    )
            return

        var src_bit = src_offset & 7
        var dst_bit = dst_offset & 7

        if src_bit == dst_bit:
            var src_byte = src_offset >> 3
            var dst_byte = dst_offset >> 3
            var end_bit = dst_offset + length
            var end_byte = end_bit >> 3
            var end_sub = end_bit & 7

            if dst_bit != 0:
                var keep_mask = UInt8((1 << dst_bit) - 1)
                dst[unsafe_offset=dst_byte] = (
                    dst[unsafe_offset=dst_byte] & keep_mask
                ) | (src_ptr[unsafe_offset=src_byte] & ~keep_mask)
                src_byte += 1
                dst_byte += 1

            if end_byte > dst_byte:
                unsafe_memcpy(
                    dest=dst.unsafe_offset(dst_byte),
                    src=src_ptr.unsafe_offset(src_byte),
                    count=end_byte - dst_byte,
                )

            if end_sub != 0:
                var trail_byte_src = src_byte + (end_byte - dst_byte)
                var keep_mask = UInt8(0xFF) << UInt8(end_sub)
                dst[unsafe_offset=end_byte] = (
                    dst[unsafe_offset=end_byte] & keep_mask
                ) | (src_ptr[unsafe_offset=trail_byte_src] & ~keep_mask)
        else:
            var src_byte = src_offset >> 3
            var dst_byte_start = dst_offset >> 3
            var end_bit = dst_offset + length
            var end_byte = end_bit >> 3
            var end_sub = end_bit & 7
            var delta = src_bit - dst_bit

            if dst_bit != 0:
                var keep_mask = UInt8((1 << dst_bit) - 1)
                var shifted: UInt8
                if delta > 0:
                    shifted = (
                        src_ptr[unsafe_offset=src_byte] >> UInt8(delta)
                    ) | (
                        src_ptr[unsafe_offset=src_byte + 1] << UInt8(8 - delta)
                    )
                else:
                    shifted = src_ptr[unsafe_offset=src_byte] << UInt8(-delta)
                    if src_byte > 0:
                        shifted |= src_ptr[unsafe_offset=src_byte - 1] >> UInt8(
                            8 + delta
                        )
                dst[unsafe_offset=dst_byte_start] = (
                    dst[unsafe_offset=dst_byte_start] & keep_mask
                ) | (shifted & ~keep_mask)
                dst_byte_start += 1

            var src_bit_pos = src_offset + ((dst_byte_start << 3) - dst_offset)
            for j in range(dst_byte_start, end_byte):
                var sb = src_bit_pos >> 3
                var so = src_bit_pos & 7
                if so == 0:
                    dst[unsafe_offset=j] = src_ptr[unsafe_offset=sb]
                else:
                    dst[unsafe_offset=j] = (
                        src_ptr[unsafe_offset=sb] >> UInt8(so)
                    ) | (src_ptr[unsafe_offset=sb + 1] << UInt8(8 - so))
                src_bit_pos += 8

            if end_sub != 0:
                var sb = src_bit_pos >> 3
                var so = src_bit_pos & 7
                var shifted: UInt8
                if so == 0:
                    shifted = src_ptr[unsafe_offset=sb]
                else:
                    shifted = (src_ptr[unsafe_offset=sb] >> UInt8(so)) | (
                        src_ptr[unsafe_offset=sb + 1] << UInt8(8 - so)
                    )
                var keep_mask = UInt8(0xFF) << UInt8(end_sub)
                dst[unsafe_offset=end_byte] = (
                    dst[unsafe_offset=end_byte] & keep_mask
                ) | (shifted & ~keep_mask)

    def extend(
        mut self, src: Bitmap[], dst_start: Int, length: Int
    ) where Self.mut:
        """Copy `length` bits from `src` into self at `dst_start`."""
        # TODO: do we need extend on view? if not move it here
        self.extend(src.view(0, length), dst_start, length)

    # TODO(MOCO-4220): mirrors `Buffer.resize` above — it delegates there,
    # which needs a statically-mutable `_buffer`.
    @__allow_legacy_custom_self_type
    def resize(mut self: Bitmap[mut=True], capacity: Int) raises:
        """Resize the underlying buffer to hold `capacity` bits.

        The logical length tracks the capacity: growing extends it,
        shrinking truncates it.
        """
        self._buffer.resize(math.ceildiv(capacity, 8))
        self._length = capacity

    def is_device(self) -> Bool:
        """Return True if the bitmap lives on a GPU device."""
        return self._buffer.is_device()

    def to_device(
        self, ctx: DeviceContext
    ) raises -> Bitmap[mut=False] where not Self.mut:
        """Upload bitmap to the GPU; returns a new device-resident Bitmap."""
        return Bitmap[mut=False](
            self._buffer.to_device(ctx), length=self._length
        )

    def to_cpu(
        self, ctx: DeviceContext
    ) raises -> Bitmap[mut=False] where not Self.mut:
        """Download bitmap from the GPU to owned CPU heap buffers."""
        return Bitmap[mut=False](self._buffer.to_cpu(ctx), length=self._length)

    def to_immutable(
        deinit self, *, length: Int = -1
    ) -> Bitmap[mut=False] where Self.mut:
        """Consume and freeze the builder into an immutable `Bitmap[]`.

        Pass `length` to set the number of meaningful bits explicitly; otherwise
        the builder's current `_length` is used.
        """
        var n = length if length >= 0 else self._length
        return Bitmap[mut=False](self._buffer^.to_immutable(), length=n)
