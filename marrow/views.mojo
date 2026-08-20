"""Non-owning, DevicePassable views over contiguous memory.

BufferView
----------
Typed, non-owning view over contiguous element data. Two machine words
(pointer + length). The array offset is baked into the pointer at
construction time, so all index operations are zero-based relative to
the view's start.

BitmapView
----------
Non-owning view over bit-packed data. Three machine words (pointer +
bit_offset + bit_count). All indexing is logical (relative to view start).
Supports both read and write operations depending on the `mut` parameter.
Method names follow Mojo's ``std.collections.bitset.BitSet`` conventions.
"""

from std.sys.info import simd_byte_width, simd_width_of
from std.sys import size_of
from std.bit import count_trailing_zeros, pop_count
from std.sys import compressed_store as _compressed_store
import std.math as math
from std.math import iota
from std.memory import bitcast, unsafe_memcpy
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.sys.intrinsics import prefetch
from std.algorithm.backend.vectorize import vectorize
from max.algorithm.functional import sync_parallelize
from max.algorithm.functional import elementwise
from max.algorithm.reduction import _reduce_generator_wrapper
from std.math import ceildiv
from std.utils.index import IndexList
from std.utils.coord import Coord
from std.gpu.host import get_gpu_target

from .buffers import Buffer, Bitmap
from .execution import ExecContext


def _packed_uint_dtype[W: Int]() -> DType:
    """Map a bool SIMD width to the unsigned integer DType that fits W bits."""
    comptime assert W >= 8 and W % 8 == 0, "W must be a multiple of 8"
    if W == 8:
        return DType.uint8
    elif W == 16:
        return DType.uint16
    elif W == 32:
        return DType.uint32
    else:
        return DType.uint64


@always_inline
def _pack_bools[
    W: Int
](mask: SIMD[DType.bool, W]) -> SIMD[_packed_uint_dtype[W](), W]:
    """Portable bit-pack: pack W bools into W-bit lanes.

    Each lane ``i`` becomes ``mask[i].cast[UintW]() << i``.  The caller
    uses ``.reduce_or()`` (for a single scalar result) or stores the
    per-lane shifted values directly.

    **Faster than `std.memory.pack_bits` here, and that is measured.**
    `pack_bits` is a single `pop.bitcast` from `<W x i1>`, which looks
    strictly better and is what an x86 `pmovmskb` compiles to. ARM has no
    mask-move instruction, so the bitcast lowers to a worse sequence than
    this one: swapping to it cost **+12-15% on all 18 `bench_pack_bools_*`
    cases** (w8/w32/w64 x 1k..100m, Apple Silicon, 2026-08-16) with no case
    improving. Do not "simplify" this to `pack_bits` without re-measuring on
    the target you care about.

    W must be 8, 16, 32, or 64.
    """
    comptime T = _packed_uint_dtype[W]()
    var bits = mask.cast[T]()
    return bits << iota[T, W]()


# ---------------------------------------------------------------------------
# BufferView — typed element view
# ---------------------------------------------------------------------------


struct BufferView[
    mut: Bool,
    //,
    T: DType,
    origin: Origin[mut=mut],
](
    DevicePassable,
    ImplicitlyCopyable,
    Sized,
    TrivialRegisterPassable,
    Writable,
):
    """Non-owning, typed, DevicePassable view over contiguous element data.

    Two machine words (pointer + length). The offset from the original
    Buffer/Array is baked into the pointer at construction time, so all
    index operations are zero-based relative to the view's start.

    Parameters:
        mut: Whether the view permits writes.
        T: The element DType (int32, float64, etc.).
        origin: The lifetime origin tying this view to its backing buffer.
    """

    var _data: Pointer[Scalar[Self.T], Self.origin]
    var _length: Int

    # --- DevicePassable ---

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(t"BufferView[{Self.T}]")

    # --- lifecycle ---

    @always_inline
    def __init__(
        out self,
        *,
        ptr: Pointer[Scalar[Self.T], Self.origin],
        length: Int,
    ):
        self._data = ptr
        self._length = length

    # --- Sized ---

    @always_inline
    def __len__(self) -> Int:
        return self._length

    # --- Boolable ---

    @always_inline
    def __bool__(self) -> Bool:
        return self._length > 0

    # --- Bounds check ---

    @always_inline
    def _check_bounds(self, index: Int):
        debug_assert(
            0 <= index < self._length,
            "BufferView index ",
            index,
            " out of bounds for length ",
            self._length,
        )

    @always_inline
    def _check_range(self, index: Int, width: Int):
        """Assert a `width`-element access starting at `index` is in bounds."""
        debug_assert(
            0 <= index and index + width <= self._length,
            "BufferView range [",
            index,
            ", ",
            index + width,
            ") out of bounds for length ",
            self._length,
        )

    # --- Element access ---

    @always_inline
    def as_span(self) -> Span[Scalar[Self.T], Self.origin]:
        """This view as a `std` `Span` — same responsibility (a non-owning,
        length-carrying window over contiguous elements), for APIs typed on
        `Span`. Preserves mutability via the origin."""
        return Span[Scalar[Self.T], Self.origin](
            unsafe_ptr=self._data, length=self._length
        )

    @always_inline
    def __getitem__(self, index: Int) -> Scalar[Self.T]:
        self._check_bounds(index)
        return self._data[unsafe_offset=index]

    @always_inline
    def __getitem__(self, slc: Slice) -> Self:
        var start: Int
        var end: Int
        var step: Int
        start, end, step = slc.indices(self._length)
        debug_assert(step == 1, "BufferView slice step must be 1")
        return Self(ptr=self._data.unsafe_offset(start), length=end - start)

    @always_inline
    def __contains__(self, value: Scalar[Self.T]) -> Bool:
        for i in range(self._length):
            if self._data[unsafe_offset=i] == value:
                return True
        return False

    @always_inline
    def unsafe_get(self, index: Int) -> Scalar[Self.T]:
        """Read element ``index`` without the `__getitem__` slice machinery.

        "Unsafe" is about the *release* build: the bound is checked by
        `debug_assert`, which compiles out unless `-D ASSERT` is set.
        """
        self._check_bounds(index)
        return self._data[unsafe_offset=index]

    @always_inline
    def unsafe_set(
        self,
        index: Int,
        value: Scalar[Self.T],
    ) where Self.mut:
        """Write element ``index``; see `unsafe_get` for what "unsafe" means.

        This bound is the one that matters. The heap overflow in
        `compressed_store_dense` (F1) went undetected under `-D ASSERT=all`
        precisely because this method used to be unchecked, so the only
        signal was a corrupted tcmalloc freelist crashing somewhere else
        entirely. Every write into a `BufferView` funnels through here.
        """
        self._check_bounds(index)
        self._data.unsafe_mut_cast[True]().unsafe_store(index, value)

    # --- SIMD ---

    # TODO: could be good idea to use std.sys.intrinsics.masked_load
    @always_inline
    def load[W: Int](self, index: Int) -> SIMD[Self.T, W]:
        self._check_range(index, W)
        return self._data.unsafe_load[width=W](index)

    # TODO: could be good idea to use std.sys.intrinsics.masked_store
    @always_inline
    def store[
        W: Int
    ](self, index: Int, value: SIMD[Self.T, W],) where Self.mut:
        self._check_range(index, W)
        self._data.unsafe_mut_cast[True]().unsafe_store(index, value)

    @always_inline
    def gather[W: Int](self, offsets: SIMD[DType.int64, W]) -> SIMD[Self.T, W]:
        """SIMD gather: load W elements at positions given by `offsets`.

        Every lane is an independent address, so one bad offset is a wild read
        rather than a one-past overrun — worth the two SIMD compares that
        `debug_assert` costs in a test build and nothing in a release one.
        """
        debug_assert(
            Int(offsets.reduce_min()) >= 0
            and Int(offsets.reduce_max()) < self._length,
            "BufferView.gather offset out of bounds for length ",
            self._length,
        )
        return self._data.unsafe_gather[width=W, alignment=1](offsets)

    # --- Compressed store ---

    @always_inline
    def compressed_store[
        W: Int
    ](self, value: SIMD[Self.T, W], mask: SIMD[DType.bool, W],) where Self.mut:
        """Compress-store via LLVM intrinsic: write only mask=True lanes,
        packed sequentially from the start of this view."""
        debug_assert(
            Int(mask.cast[DType.uint8]().reduce_add()) <= self._length,
            "BufferView.compressed_store writes ",
            Int(mask.cast[DType.uint8]().reduce_add()),
            " lanes into a view of length ",
            self._length,
        )
        _compressed_store(value, self._data.unsafe_mut_cast[True](), mask)

    @always_inline
    def compressed_store_sparse(
        self,
        src: BufferView[Self.T, _],
        sel_bits: UInt64,
    ) where Self.mut:
        """CTZ scatter: write only set-bit positions. O(popcount).

        Best when few bits are set (low popcount).
        """
        debug_assert(
            Int(pop_count(sel_bits)) <= self._length,
            "compressed_store_sparse writes ",
            Int(pop_count(sel_bits)),
            " elements into a view of length ",
            self._length,
        )
        var w = sel_bits
        var k = 0
        while w != 0:
            self.unsafe_set(k, src.unsafe_get(Int(count_trailing_zeros(w))))
            w &= w - 1
            k += 1

    @always_inline
    def compressed_store_dense(
        self,
        src: BufferView[Self.T, _],
        sel_bits: UInt64,
    ) where Self.mut:
        """Byte-chunked branchless scatter. O(64).

        Processes the 64-bit mask one byte at a time, breaking the serial
        dependency into 8 independent chains of depth 8 that the OoO engine
        can overlap.

        **Writes `popcount(sel_bits) + 1` elements, not `popcount(sel_bits)`.**
        Being branchless is exactly what costs the extra slot: the store fires
        before the lane's bit is inspected, so every lane above the highest set
        bit lands on element `popcount(sel_bits)` — a dead write that the next
        set lane would overwrite, except there is no next set lane. Callers must
        leave one element of slack; `compressed_store` below checks `len(self)`
        and picks the sparse path when they cannot.
        """
        debug_assert(
            self._length > Int(pop_count(sel_bits)),
            (
                "compressed_store_dense needs one element of slack past the"
                " packed output: length "
            ),
            self._length,
            " popcount ",
            Int(pop_count(sel_bits)),
        )
        debug_assert(
            len(src) >= 64,
            "compressed_store_dense reads all 64 lanes; source has ",
            len(src),
        )
        var offset = 0
        comptime for i in range(8):
            var byte = (sel_bits >> UInt64(i * 8)) & 0xFF
            var b = byte
            var k = 0
            comptime for bit in range(8):
                self.unsafe_set(offset + k, src.unsafe_get(i * 8 + bit))
                k += Int(b & 1)
                b >>= 1
            offset += Int(pop_count(byte))

    @always_inline
    def compressed_store[
        sparse_threshold: Int = 24
    ](
        self,
        src: BufferView[Self.T, _],
        sel_bits: UInt64,
    ) -> Int where Self.mut:
        """Adaptive compressed store: dispatches to sparse or dense based on
        popcount vs threshold. Returns number of elements written.

        The dense path needs one element of slack past the packed output (see
        `compressed_store_dense`), so it is taken only when this view is longer
        than the popcount. The last selection word of a filter is exactly the
        one that has no slack — `BufferView.filter` sizes its destination to the
        pre-counted set-bit total — and it is the word whose extra store would
        land outside the allocation, since `Buffer._aligned_size` aligns to 64
        bytes without padding beyond it."""
        var cnt = Int(pop_count(sel_bits))
        if cnt <= sparse_threshold or self._length <= cnt:
            self.compressed_store_sparse(src, sel_bits)
        else:
            self.compressed_store_dense(src, sel_bits)
        return cnt

    # --- Slicing ---

    @always_inline
    def slice(self, offset: Int, length: Int = -1) -> Self:
        var actual = length if length >= 0 else self._length - offset
        return Self(ptr=self._data.unsafe_offset(offset), length=actual)

    # --- Raw pointer access ---

    # TODO: consider to remove this and let c_data to poke into _data directly
    # but other componenst shouldn't access unsafe_ptr()
    @always_inline
    def unsafe_ptr(self) -> Pointer[Scalar[Self.T], Self.origin]:
        return self._data

    @always_inline
    def prefetch_at(self, offset: Int):
        """Prefetch the cache line at `offset` elements into L1 cache."""
        prefetch(self._data.unsafe_offset(offset))

    def copy_from(
        self,
        src: BufferView[Self.T, _],
        count: Int,
    ) where Self.mut:
        """Copy `count` elements from `src` into `self`."""
        unsafe_memcpy(
            dest=self._data.unsafe_mut_cast[True]().unsafe_bitcast[UInt8](),
            src=src._data.unsafe_bitcast[UInt8](),
            count=count * size_of[Scalar[Self.T]](),
        )

    def filter(
        self,
        sel: BitmapView[_],
        sel_start: Int,
        sel_end: Int,
        out_len: Int,
    ) -> Buffer[]:
        """Filter these fixed-width values, keeping elements where `sel` is set.

        Run-merges all-ones selection words (unsafe_memcpy) and compress-stores mixed
        words; a sparse tail only reads set-bit positions. `sel_start`/`sel_end`
        are the 64-bit block bounds and `out_len` the pre-counted set-bit count.
        """
        comptime ALL_ONES = ~UInt64(0)
        var buf = Buffer.alloc_uninit(out_len * size_of[Scalar[Self.T]]())
        var dst = buf.view[Self.T](0, out_len)
        var out_pos = 0
        var i = sel_start

        while i + 64 <= sel_end:
            var sel_word = sel.load_bits[DType.uint64](i)
            if sel_word == 0:
                i += 64
                while i + 64 <= sel_end and sel.load_bits[DType.uint64](i) == 0:
                    i += 64
                continue
            if sel_word == ALL_ONES:
                var run_start = i
                i += 64
                while (
                    i + 64 <= sel_end
                    and sel.load_bits[DType.uint64](i) == ALL_ONES
                ):
                    i += 64
                dst.slice(out_pos).copy_from(
                    self.slice(run_start), i - run_start
                )
                out_pos += i - run_start
                continue
            out_pos += dst.slice(out_pos).compressed_store(
                self.slice(i), sel_word
            )
            i += 64

        if i < sel_end:
            var tail = sel_end - i
            var mask = (UInt64(1) << UInt64(tail)) - 1
            var sel_word = sel.load_bits[DType.uint64](i) & mask
            if sel_word != 0:
                dst.slice(out_pos).compressed_store_sparse(
                    self.slice(i), sel_word
                )
                out_pos += Int(pop_count(sel_word))

        debug_assert(
            out_pos == out_len,
            "BufferView.filter produced ",
            out_pos,
            " elements but its destination was sized for ",
            out_len,
        )
        return buf.to_immutable()

    def to_string_slice(self) -> StringSlice[Self.origin]:
        """Convert this byte view to a StringSlice with origin `self_o`."""
        return StringSlice(
            unsafe_from_utf8=Span[Byte](
                unsafe_ptr=self._data.unsafe_bitcast[Byte](),
                length=self._length,
            )
        )

    def copy_from(
        self,
        src: StringSlice[_],
    ) where Self.mut and Self.T == DType.uint8:
        """Copy bytes from a StringSlice into this view."""
        unsafe_memcpy(
            dest=self._data.unsafe_mut_cast[True]().unsafe_bitcast[Byte](),
            src=src.unsafe_ptr(),
            count=src.byte_length(),
        )

    # --- Vectorized operations ---

    # TODO: remove this in favor of the free-function apply with explicit SIMD function parameters
    def apply[
        func: def[W: Int](SIMD[Self.T, W]) thin -> SIMD[Self.T, W]
    ](self) where Self.mut:
        """Apply a SIMD function in-place over all elements."""
        comptime width = simd_byte_width() // size_of[Scalar[Self.T]]()
        var i = 0
        while i + width <= self._length:
            var out = self._data.unsafe_mut_cast[True]()
            out.unsafe_store(
                i, func[width](self._data.unsafe_load[width=width](i))
            )
            i += width
        while i < self._length:
            var out = self._data.unsafe_mut_cast[True]()
            out[unsafe_offset=i] = func[1](self._data[unsafe_offset=i])
            i += 1

    def count[
        func: def[W: Int](SIMD[Self.T, W]) thin -> SIMD[DType.bool, W]
    ](self) -> Int:
        """Count elements matching a vectorized predicate."""
        comptime width = simd_byte_width() // size_of[Scalar[Self.T]]()
        var total = 0
        var i = 0
        while i + width <= self._length:
            total += Int(
                func[width](self._data.unsafe_load[width=width](i))
                .cast[DType.uint8]()
                .reduce_add()
            )
            i += width
        while i < self._length:
            if func[1](self._data[unsafe_offset=i]):
                total += 1
            i += 1
        return total

    # --- Writable ---

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"BufferView(length={self._length})")


# ---------------------------------------------------------------------------
# BitmapView — bit-packed view with BitSet-style API
# ---------------------------------------------------------------------------


struct BitmapView[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
](
    Boolable,
    DevicePassable,
    ImplicitlyCopyable,
    Sized,
    TrivialRegisterPassable,
    Writable,
):
    """Non-owning, DevicePassable view over bit-packed data.

    Three machine words (pointer + bit_offset + bit_count). Parametric
    mutability. All indexing is logical (relative to view start). Method
    names follow ``std.collections.bitset.BitSet`` conventions.

    Parameters:
        mut: Whether the view permits writes.
        origin: The lifetime origin tying this view to its backing buffer.
    """

    var _data: Pointer[UInt8, Self.origin]
    var _offset: Int  # bit offset into _data
    var _length: Int  # number of logical bits

    # --- DevicePassable ---

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return String("BitmapView")

    # --- lifecycle ---

    @always_inline
    def __init__(
        out self,
        *,
        ptr: Pointer[UInt8, Self.origin],
        offset: Int,
        length: Int,
    ):
        self._data = ptr
        self._offset = offset
        self._length = length

    # --- Sized ---

    @always_inline
    def __len__(self) -> Int:
        return self._length

    @always_inline
    def offset(self) -> Int:
        """Bit offset of this view into its backing buffer.

        Byte-level readers need it to derive a starting byte and a sub-byte
        shift; every bit-addressed accessor here applies it for them.
        """
        return self._offset

    # --- Boolable (any bit set) ---

    def __bool__(self) -> Bool:
        """Return True if any bit in the view is set."""
        if self._length == 0:
            return False

        comptime width = simd_width_of[DType.uint8]()
        var ptr = self._data
        var bit_start = self._offset
        var bit_end = bit_start + self._length
        var byte_start = bit_start >> 3
        var byte_end = (bit_end + 7) >> 3
        var nbytes = byte_end - byte_start

        var first_mask = UInt8(0xFF) << UInt8(bit_start & 7)
        var last_mask = UInt8(
            (1 << ((bit_end - 1) & 7) + 1) - 1
        ) if bit_end & 7 != 0 else UInt8(0xFF)

        if nbytes == 1:
            return (ptr[unsafe_offset=byte_start] & first_mask & last_mask) != 0

        if (ptr[unsafe_offset=byte_start] & first_mask) != 0:
            return True
        if (ptr[unsafe_offset=byte_end - 1] & last_mask) != 0:
            return True

        var i = byte_start + 1
        var end = byte_end - 1
        while i + width <= end:
            if (ptr.unsafe_offset(i)).unsafe_load[
                width=width
            ]().reduce_or() != 0:
                return True
            i += width
        while i < end:
            if ptr[unsafe_offset=i] != 0:
                return True
            i += 1

        return False

    # --- Bounds check ---

    @always_inline
    def _check_bounds(self, index: Int):
        debug_assert(
            0 <= index < self._length,
            "BitmapView index ",
            index,
            " out of bounds for length ",
            self._length,
        )

    @always_inline
    def _byte_extent(self) -> Int:
        """Bytes this view spans from ``_data``, ``_offset`` included.

        The last bit of the view lives in byte ``(_offset + _length - 1) >> 3``,
        so the view owns ``[0, (_offset + _length + 7) >> 3)``. This is the only
        bound a `BitmapView` can enforce: it holds a pointer, a bit offset and a
        bit count, never the size of the allocation behind them.
        """
        return (self._offset + self._length + 7) >> 3

    @always_inline
    def _check_byte_range(self, byte_index: Int, byte_count: Int):
        """Assert a raw byte **write** stays inside `_byte_extent`.

        Guards the byte-addressed accessors, which used to justify a wide
        unconditional load/store with "Arrow buffers are 64-byte padded". That
        is false as an overshoot licence: `Buffer._aligned_size` rounds the
        allocation up to a 64-byte multiple exactly as Arrow C++'s
        `RoundUpToMultipleOf64` does, so a bitmap whose byte length is already
        a multiple of 64 has **zero** slack past its last bit.
        """
        debug_assert(
            0 <= byte_index and byte_index + byte_count <= self._byte_extent(),
            "BitmapView byte range [",
            byte_index,
            ", ",
            byte_index + byte_count,
            ") out of bounds for a view spanning ",
            self._byte_extent(),
            " bytes",
        )

    @always_inline
    def _check_byte_read_range(self, byte_index: Int, byte_count: Int):
        """Assert a raw byte **read** stays inside the backing allocation.

        Looser than `_check_byte_range` on purpose. `load[W]` and `load_bits`
        take a deliberately wide unconditional load — 4 and ``size_of[T]()``
        bytes — so a read can legitimately run past the view's last live byte
        into the allocation's padding, and every lane the caller consumes is
        still inside the view. The bound that actually holds is the *allocation*
        size, and for a marrow-owned buffer that is `align_up(bytes, 64)`.

        Two things this deliberately does not paper over:

        - A view whose byte extent is already a 64-byte multiple has **no**
          padding, so a 512-bit bitmap still trips this — which is the case
          worth catching.
        - A FOREIGN buffer imported over the C Data Interface carries no
          padding guarantee at all: the spec makes alignment "recommended, but
          not required" and says nothing about padding. This bound is therefore
          optimistic for imported data; see
          docs/alpha-findings/g1-buffer-invariants.md.
        """
        debug_assert(
            0 <= byte_index
            and byte_index + byte_count
            <= math.align_up(self._byte_extent(), 64),
            "BitmapView byte read [",
            byte_index,
            ", ",
            byte_index + byte_count,
            ") runs past the padded extent of a view spanning ",
            self._byte_extent(),
            " bytes",
        )

    # --- Element access (BitSet-style) ---

    @always_inline
    def __getitem__(self, index: Int) -> Bool:
        return self.test(index)

    @always_inline
    def __getitem__(self, slc: Slice) -> Self:
        var start: Int
        var end: Int
        var step: Int
        start, end, step = slc.indices(self._length)
        debug_assert(step == 1, "BitmapView slice step must be 1")
        return Self(
            ptr=self._data, offset=self._offset + start, length=end - start
        )

    @always_inline
    def bit_offset(self) -> Int:
        """Return the bit offset into the backing buffer."""
        return self._offset

    def unsafe_ptr(self) -> Pointer[UInt8, Self.origin]:
        """Raw byte pointer to the first byte of this view's backing storage.

        Only for use at C FFI boundaries (c_data.mojo). Prefer load/store.
        """
        return self._data

    # --- Compressed store / pext ---

    @always_inline
    def pext(self, index: Int, mask: UInt64) -> UInt64:
        """Parallel bit extract: keep bits at ``index`` where ``mask``=1,
        packed to LSB. O(popcount(mask))."""
        var val = self.load_bits[DType.uint64](index)
        var result = UInt64(0)
        var m = mask
        var k = UInt64(0)
        while m != 0:
            var bit_pos = UInt64(count_trailing_zeros(m))
            result |= ((val >> bit_pos) & 1) << k
            k += 1
            m &= m - 1
        return result

    @always_inline
    def compressed_store(
        self,
        bit_offset: Int,
        bits: UInt64,
        count: Int,
    ) where Self.mut:
        """Deposit ``count`` LSBs from ``bits`` at ``bit_offset``.

        Uses OR — the bitmap must be zero-initialized. Handles arbitrary bit
        alignment, writing between 1 and 9 bytes.

        ``bit_offset`` is measured from ``_data``, not from the view's
        ``_offset``, because the deposit is expressed with the byte-addressed
        accessors; the assertion below pins that to the offset-0 destinations
        this is built for (`BitmapView.filter`'s freshly allocated output).

        **Writes exactly the bytes the run occupies** —
        ``ceildiv(bit_offset % 8 + count, 8)`` — rather than rounding up to a
        full 8-byte word. It used to always store 8 bytes and justify the
        overshoot with "Arrow buffers are 64-byte padded". They are padded to a
        64-byte *multiple* (`Buffer._aligned_size`, matching Arrow C++'s
        `RoundUpToMultipleOf64`), which is not the same as having slack: a
        bitmap of 512 bits is exactly 64 bytes, so depositing into its last byte
        read-modify-wrote 7 bytes of the neighbouring allocation. The written-back
        value was unchanged (``x | 0``), so it never corrupted a heap today, but
        it was a lost-update race between two threads filtering into adjacent
        allocations and an out-of-bounds write to any sanitizer.
        """
        if count == 0:
            return
        debug_assert(
            self._offset == 0,
            (
                "BitmapView.compressed_store is byte-addressed and ignores"
                " _offset; it needs an offset-0 destination, got "
            ),
            self._offset,
        )
        debug_assert(
            0 <= bit_offset and bit_offset + count <= self._length,
            "BitmapView.compressed_store range [",
            bit_offset,
            ", ",
            bit_offset + count,
            ") out of bounds for length ",
            self._length,
        )
        debug_assert(
            count >= 64 or (bits >> UInt64(count)) == 0,
            "BitmapView.compressed_store: bits set above count ",
            count,
        )
        var byte_idx = bit_offset >> 3
        var bit_off = bit_offset & 7
        var nbytes = (bit_off + count + 7) >> 3
        var shifted = bits << UInt64(bit_off)
        if nbytes >= 8:
            self.store_bytes[DType.uint8, 8](
                byte_idx,
                self.load_bytes[DType.uint8, 8](byte_idx)
                | bitcast[DType.uint8, 8](shifted),
            )
            if nbytes > 8:
                self.store_bytes[DType.uint8](
                    byte_idx + 8,
                    self.load_bytes[DType.uint8](byte_idx + 8)
                    | UInt8(bits >> UInt64(64 - bit_off)),
                )
        else:
            # Short tail: at most 7 single-byte deposits, and only ever on the
            # final block of a filter, so the hot path above is untouched.
            var packed = bitcast[DType.uint8, 8](shifted)
            for k in range(nbytes):
                self.store_bytes[DType.uint8](
                    byte_idx + k,
                    self.load_bytes[DType.uint8](byte_idx + k) | packed[k],
                )

    # --- Bit access ---

    @always_inline
    def test(self, index: Int) -> Bool:
        """Test if the bit at ``index`` is set."""
        self._check_bounds(index)
        var bit_index = self._offset + index
        return Bool(
            (self._data[unsafe_offset=bit_index >> 3] >> UInt8(bit_index & 7))
            & 1
        )

    @always_inline
    def load[W: Int](self, index: Int) -> SIMD[DType.bool, W]:
        """Expand W consecutive bits starting at logical ``index`` into a
        SIMD bool vector — the bit-addressed counterpart of `store[W]`.

        Each lane j is True iff bit (index + j) is set. ``_offset`` is applied.

        The 4-byte `UInt32` is loaded unconditionally, so near the end of a
        bitmap the window is **slid backwards** to finish on the view's last
        live byte, and the distance it slid is added to the shift. Without
        that, a load addressed at the final byte reads 3 bytes past it — and a
        buffer whose byte extent is already a multiple of 64 has no padding at
        all (`Buffer._aligned_size` rounds *up* to 64), so those bytes are
        outside the allocation. 512 bits is the smallest case; a nullable
        150,000-element column, 18,750 bytes and 62 mod 64, is the shape that
        found it through the masked `apply` lane.

        Sliding is correct rather than merely in-bounds. The caller only ever
        consumes lanes inside the view, so the last bit needed satisfies
        ``abs_pos + W <= 8 * _byte_extent()``; when the window is clamped to
        end at ``_byte_extent()`` the shifted word holds exactly
        ``8 * _byte_extent() - abs_pos >= W`` live bits. It costs two ALU ops
        and no branch, which is why it is done here rather than by giving the
        caller a special tail path.

        This is the default reader, and it mirrors `BufferView.load[W]`: both
        take a logical element index and return W elements. Reach for
        `load_bytes` only for whole-byte bitmap arithmetic.
        """
        var abs_pos = self._offset + index
        # `max(..., 0)` covers a view spanning fewer than 4 bytes, where there
        # is nothing to slide back to. A marrow-owned allocation is at least 64
        # bytes, so the load stays inside it; an imported FOREIGN buffer that
        # small carries no such guarantee, and never did — see
        # docs/alpha-findings/g1-buffer-invariants.md.
        var base = min(
            abs_pos >> 3, max(self._byte_extent() - size_of[UInt32](), 0)
        )
        var shift = abs_pos - (base << 3)
        self._check_byte_read_range(base, size_of[UInt32]())

        var bits = (
            (self._data.unsafe_offset(base))
            .unsafe_bitcast[UInt32]()
            .unsafe_load[alignment=1]()
        )
        bits >>= UInt32(shift)

        return (
            (SIMD[DType.uint32, W](bits) >> iota[DType.uint32, W]()) & 1
        ).cast[DType.bool]()

    @always_inline
    def load_bits[T: DType](self, index: Int) -> Scalar[T]:
        """Load ``size_of[T]() * 8`` bits starting at logical position
        ``index``, still **packed** into a scalar.

        Bit-addressed and ``_offset``-applying, like `load[W]` — the difference
        is the result shape: `load[W]` expands each bit into its own SIMD lane,
        this returns the run as packed bits.

        A sub-byte ``_offset`` (what any sliced array carries) pushes the run's
        last ``_offset & 7`` bits into the byte *after* the ``size_of[T]()``
        this loads, so a second byte is folded in from the top. Without it the
        high bits came back as zeros: an offset-4 view read bits 60..63 as
        unset regardless of the bitmap, which silently dropped rows from a
        filter over a sliced column. The extra byte is fetched only when it is
        inside the view, so a caller reading a short tail is not made worse.
        """
        comptime NBITS = size_of[T]() * 8
        var abs_pos = self._offset + index
        var byte_idx = abs_pos >> 3
        var bit_off = abs_pos & 7
        self._check_byte_read_range(byte_idx, size_of[T]())
        var raw = (
            (self._data.unsafe_offset(byte_idx))
            .unsafe_bitcast[Scalar[T]]()
            .unsafe_load[alignment=1]()
        )
        var result = raw >> Scalar[T](bit_off)
        if bit_off > 0 and byte_idx + size_of[T]() < self._byte_extent():
            var hi = Scalar[T](
                self._data[unsafe_offset=byte_idx + size_of[T]()]
            )
            result = result | (hi << Scalar[T](NBITS - bit_off))
        return result

    # TODO: could be good idea to use std.sys.intrinsics.masked_load
    # --- Raw byte access ---
    #
    # Everything above is bit-addressed and applies `_offset`. The two below are
    # byte-addressed and do NOT. Keeping that split visible is the point: the
    # names used to be `load`/`store`, and `load[DType.bool, W]` silently read a
    # bit-packed mask one byte per element (B29).

    @always_inline
    def load_bytes[T: DType, W: Int = 1](self, index: Int) -> SIMD[T, W]:
        """Load W elements of type T from the raw bitmap bytes at ``index``.

        **Byte-addressed, and it ignores ``_offset``** — both unlike every other
        accessor here. ``index`` is in units of T (index=2 with T=uint32 reads
        bytes 8–11) and the caller owns computing that address. Bounded by
        `_check_byte_read_range`, not by any padding guarantee.

        For bit-addressed reads use `load[W]` (W bits as SIMD bool), `test` (one
        bit) or `load_bits[T]` (a packed run of bits). This exists for whole-byte
        bitmap arithmetic — the bitwise and/or/xor kernels below — and has no
        caller outside this module. It used to be spelled `load`, which made
        `load[DType.bool, W]` an easy way to read a bit-packed mask a *byte* at
        a time: `Scalar[DType.bool]` is a byte, so `[F, F, T]` is `0b100`, which
        answers `True` at element 0. Five fused bool lanes did exactly that and
        returned wrong rows (B29, fixed 2026-08-06)."""
        self._check_byte_read_range(index * size_of[T](), W * size_of[T]())
        return self._data.unsafe_bitcast[Scalar[T]]().unsafe_load[
            width=W, alignment=1
        ](index)

    # TODO: probably should be removed
    # TODO: could be good idea to use std.sys.intrinsics.masked_store
    @always_inline
    def store_bytes[
        T: DType, W: Int = 1
    ](self, index: Int, val: SIMD[T, W]) where Self.mut:
        """Store W elements of type T into the raw bitmap bytes at ``index``.

        **Byte-addressed, and it ignores ``_offset``** — the mirror of
        `load_bytes`, and the same caveats apply. For bit-addressed writes use
        `store[W]`.
        """
        self._check_byte_range(index * size_of[T](), W * size_of[T]())
        self._data.unsafe_mut_cast[True]().unsafe_bitcast[
            Scalar[T]
        ]().unsafe_store[width=W](index, val)

    @always_inline
    def store[
        W: Int
    ](self, bit_index: Int, val: SIMD[DType.bool, W],) where Self.mut:
        """Bit-pack W bools and store into the bitmap at ``bit_index``.

        - W divisible by 8: single _pack_bools + bitcast store.
        - W < 8: set/clear individual bits.
        """
        comptime assert (
            W % 8 == 0 or W < 8
        ), "W must be divisible by 8 or less than 8"

        comptime if W % 8 == 0:
            # The whole-byte path is byte-addressed and, unlike the sub-byte
            # path below, does *not* apply `_offset` — see the audit note in
            # docs/alpha-findings/g1-buffer-invariants.md.
            debug_assert(
                self._offset == 0,
                (
                    "BitmapView.store[W] with W % 8 == 0 ignores _offset; it"
                    " needs an offset-0 destination, got "
                ),
                self._offset,
            )
            self._check_byte_range(bit_index >> 3, W // 8)
            var packed = _pack_bools(val).reduce_or()
            var dst = self._data.unsafe_mut_cast[True]().unsafe_offset(
                (bit_index >> 3)
            )
            dst.unsafe_store(bitcast[DType.uint8, W // 8](packed))
        else:
            var abs_pos = self._offset + bit_index
            var out = self._data.unsafe_mut_cast[True]()
            comptime for i in range(W):
                var p = abs_pos + i
                var byte_idx = p >> 3
                var bit_off = UInt8(p & 7)
                if val[i]:
                    out[unsafe_offset=byte_idx] = out[
                        unsafe_offset=byte_idx
                    ] | (UInt8(1) << bit_off)
                else:
                    out[unsafe_offset=byte_idx] = out[
                        unsafe_offset=byte_idx
                    ] & ~(UInt8(1) << bit_off)

    # --- Slicing ---

    @always_inline
    def slice(self, offset: Int, length: Int) -> Self:
        """Return a zero-copy sub-view of ``length`` bits starting at
        ``offset``."""
        return Self(ptr=self._data, offset=self._offset + offset, length=length)

    # --- Bulk read operations ---

    # TODO: optimize this
    def all_set(self) -> Bool:
        """Return True if all bits in the view are set."""
        if self._length == 0:
            return True

        comptime width = simd_width_of[DType.uint8]()
        var ptr = self._data
        var bit_start = self._offset
        var bit_end = bit_start + self._length
        var byte_start = bit_start >> 3
        var byte_end = (bit_end + 7) >> 3
        var nbytes = byte_end - byte_start

        var first_fill = ~(UInt8(0xFF) << UInt8(bit_start & 7))
        var last_fill = ~(
            UInt8((1 << ((bit_end - 1) & 7) + 1) - 1)
        ) if bit_end & 7 != 0 else UInt8(0)

        if nbytes == 1:
            return (
                ptr[unsafe_offset=byte_start] | first_fill | last_fill
            ) == 0xFF

        if (ptr[unsafe_offset=byte_start] | first_fill) != 0xFF:
            return False
        if (ptr[unsafe_offset=byte_end - 1] | last_fill) != 0xFF:
            return False

        var i = byte_start + 1
        var end = byte_end - 1
        while i + width <= end:
            if (ptr.unsafe_offset(i)).unsafe_load[
                width=width
            ]().reduce_and() != 0xFF:
                return False
            i += width
        while i < end:
            if ptr[unsafe_offset=i] != 0xFF:
                return False
            i += 1

        return True

    def _aligned_byte_range(
        self,
    ) -> Tuple[Pointer[UInt8, Self.origin], Int, Int, Int]:
        """Return 64-byte-aligned pointer and byte range with boundary bits.

        Returns (ptr, total_bytes, lead_bits, trail_bits).

        Widening to whole 64-byte blocks is safe *for a marrow-owned buffer*
        and only there: `Buffer._aligned_size` rounds the allocation up to a
        64-byte multiple, so `align_up(byte_end, 64)` can never exceed it, and
        `align_down(byte_start, 64)` stays at or above the allocation's base
        because that base is itself 64-byte aligned. The widened bytes are
        *uninitialised*, not zero — `alloc_uninit` does not clear them — so the
        caller must mask them off with `lead_bits`/`trail_bits` rather than
        assume they read as 0.
        """
        var byte_start = self._offset >> 3
        var bit_end = self._offset + self._length
        var byte_end = (bit_end + 7) >> 3
        var aligned_start = math.align_down(byte_start, 64)
        var aligned_end = math.align_up(byte_end, 64)
        var lead_bits = self._offset - (aligned_start << 3)
        var trail_bits = (aligned_end - byte_end) * 8 + (bit_end & 7)
        return Tuple(
            self._data.unsafe_offset(aligned_start),
            aligned_end - aligned_start,
            lead_bits,
            trail_bits,
        )

    def count_set_bits_with_range(self) -> Tuple[Int, Int, Int]:
        """Count set bits and return the logical bit range covering them.

        Returns (count, start, end):
          count: total set bits
          start: logical bit offset of first block with set bits (64-aligned)
          end:   logical bit offset past last block with set bits (64-aligned)
        If count == 0, returns (0, 0, 0).
        """
        comptime width = simd_width_of[DType.uint8]()
        comptime t1_iters = 512 // width // 2
        comptime t1_bytes = 512
        comptime t2_iters = 64 // width

        if self._length == 0:
            return (0, 0, 0)

        var ptr, total_bytes, lead_bits, trail_bits = self._aligned_byte_range()

        var first_byte = total_bytes
        var last_byte = 0

        # Tier 1: 512-byte blocks, 2 interleaved uint8 accumulators.
        var t1_end = (total_bytes // t1_bytes) * t1_bytes
        var count = 0
        for i in range(0, t1_end, t1_bytes):
            var acc0 = SIMD[DType.uint8, width](0)
            var acc1 = SIMD[DType.uint8, width](0)
            comptime for j in range(t1_iters):
                acc0 += pop_count(
                    (
                        (ptr.unsafe_offset(i)).unsafe_offset((j * 2) * width)
                    ).unsafe_load[width=width]()
                )
                acc1 += pop_count(
                    (
                        (ptr.unsafe_offset(i)).unsafe_offset(
                            (j * 2 + 1) * width
                        )
                    ).unsafe_load[width=width]()
                )
            var block_count = Int(
                (
                    acc0.cast[DType.uint16]() + acc1.cast[DType.uint16]()
                ).reduce_add()
            )
            if block_count > 0:
                if first_byte == total_bytes:
                    first_byte = i
                last_byte = i + t1_bytes
            count += block_count

        # Tier 2: 64-byte blocks for the remainder.
        for i in range(t1_end, total_bytes, 64):
            var acc = SIMD[DType.uint8, width](0)
            comptime for j in range(t2_iters):
                acc += pop_count(
                    (
                        (ptr.unsafe_offset(i)).unsafe_offset(j * width)
                    ).unsafe_load[width=width]()
                )
            var block_count = Int(acc.cast[DType.uint16]().reduce_add())
            if block_count > 0:
                if first_byte == total_bytes:
                    first_byte = i
                last_byte = i + 64
            count += block_count

        # Subtract bits outside [_offset, _offset + _len).
        if lead_bits:
            var lead_bytes = lead_bits >> 3
            var lead_sub_byte = lead_bits & 7
            for i in range(lead_bytes):
                count -= Int(pop_count(ptr[unsafe_offset=i]))
            if lead_sub_byte:
                count -= Int(
                    pop_count(
                        ptr[unsafe_offset=lead_bytes]
                        & UInt8((1 << lead_sub_byte) - 1)
                    )
                )
        if trail_bits:
            var trail_bytes = trail_bits >> 3
            var trail_sub_byte = trail_bits & 7
            var first_trail = total_bytes - trail_bytes
            if trail_sub_byte:
                count -= Int(
                    pop_count(
                        ptr[unsafe_offset=first_trail - 1]
                        >> UInt8(trail_sub_byte)
                    )
                )
            for i in range(first_trail, total_bytes):
                count -= Int(pop_count(ptr[unsafe_offset=i]))

        if count == 0:
            return (0, 0, 0)

        var start = max(0, first_byte * 8 - lead_bits)
        var end = min(self._length, last_byte * 8 - lead_bits)
        start = (start // 64) * 64
        end = min(self._length, ((end + 63) // 64) * 64)
        return (count, start, end)

    def count_set_bits(self) -> Int:
        """Count set bits in the view."""
        var count, _, _ = self.count_set_bits_with_range()
        return count

    def unset_count(self) -> Int:
        """How many of these bits are 0 — the null count of a validity bitmap.

        The offset-aware counterpart of `Bitmap.unset_count`, which counts over
        the whole owning bitmap. Anything deriving a null count for a *slice*
        needs this one.
        """
        return self._length - self.count_set_bits()

    # --- Equality ---

    def __eq__(self, other: BitmapView[_]) -> Bool:
        """Return True if both views have identical logical bit patterns.

        Offsets are applied, so two views agree whenever the bits they each
        span agree, whatever position they start at.
        """
        if self._length != len(other):
            return False
        # Word-level XOR comparison.
        var i = 0
        while i + 64 <= self._length:
            if (
                self.load_bits[DType.uint64](i)
                ^ other.load_bits[DType.uint64](i)
                != 0
            ):
                return False
            i += 64
        if i < self._length:
            var tail = self._length - i
            var mask = (UInt64(1) << UInt64(tail)) - 1
            if (
                self.load_bits[DType.uint64](i)
                ^ other.load_bits[DType.uint64](i)
            ) & mask != 0:
                return False
        return True

    # --- Write operations (mut=True only, BitSet-style) ---

    @always_inline
    def set(self, index: Int) where Self.mut:
        """Set the bit at ``index`` to 1."""
        self._check_bounds(index)
        var abs_index = self._offset + index
        var byte_index = abs_index >> 3
        var bit_mask = UInt8(1 << (abs_index & 7))
        var out = self._data.unsafe_mut_cast[True]()
        out[unsafe_offset=byte_index] = out[unsafe_offset=byte_index] | bit_mask

    @always_inline
    def clear(self, index: Int) where Self.mut:
        """Set the bit at ``index`` to 0."""
        self._check_bounds(index)
        var abs_index = self._offset + index
        var byte_index = abs_index >> 3
        var bit_mask = UInt8(1 << (abs_index & 7))
        var out = self._data.unsafe_mut_cast[True]()
        out[unsafe_offset=byte_index] = (
            out[unsafe_offset=byte_index] & ~bit_mask
        )

    @always_inline
    def toggle(self, index: Int) where Self.mut:
        """Invert the bit at ``index``."""
        self._check_bounds(index)
        var abs_index = self._offset + index
        var byte_index = abs_index >> 3
        var bit_mask = UInt8(1 << (abs_index & 7))
        var out = self._data.unsafe_mut_cast[True]()
        out[unsafe_offset=byte_index] = out[unsafe_offset=byte_index] ^ bit_mask

    # --- Byte-level functors for the set operations above.  `apply` takes a
    # SIMD functor, and these are the five `BitmapView` needs; they live here
    # rather than at module scope because nothing else can use them.

    @always_inline
    @staticmethod
    def _not[W: Int](x: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
        return ~x

    @always_inline
    @staticmethod
    def _and[
        W: Int
    ](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
        return a & b

    @always_inline
    @staticmethod
    def _or[
        W: Int
    ](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
        return a | b

    @always_inline
    @staticmethod
    def _xor[
        W: Int
    ](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
        return a ^ b

    @always_inline
    @staticmethod
    def _and_not[
        W: Int
    ](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
        return a & ~b

    # --- Set operations (return Buffer with offset=0) ---

    def to_owned(self) raises -> Bitmap[mut=False]:
        """These bits as an owned, offset-0 `Bitmap`.

        `Bitmap.extend` copies whole bytes when the source is byte-aligned and
        shift-merges otherwise, so a view that already starts at a byte costs a
        `memcpy` rather than a pass over every bit.
        """
        var out = Bitmap.alloc_zeroed(self._length)
        out.extend(self, 0, self._length)
        return out.to_immutable()

    def intersection(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Return the bitwise AND of self and other."""
        var builder = Bitmap.alloc_uninit(self._length)
        apply[Self._and](self, other, builder.view())
        return builder^

    def union(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Return the bitwise OR of self and other."""
        var builder = Bitmap.alloc_uninit(self._length)
        apply[Self._or](self, other, builder.view())
        return builder^

    def symmetric_difference(
        self, other: BitmapView[_]
    ) raises -> Bitmap[mut=True]:
        """Return the bitwise XOR of self and other."""
        var builder = Bitmap.alloc_uninit(self._length)
        apply[Self._xor](self, other, builder.view())
        return builder^

    def difference(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Return self AND NOT other."""
        var builder = Bitmap.alloc_uninit(self._length)
        apply[Self._and_not](self, other, builder.view())
        return builder^

    def filter(
        self,
        sel: BitmapView[_],
        sel_start: Int,
        sel_end: Int,
        out_len: Int,
    ) -> Tuple[Bitmap[], Int]:
        """Filter these bits, keeping positions where `sel` is set.

        Uses `pext` + `compressed_store` in 64-bit blocks with run-merge for
        all-ones / all-zeros blocks; works for both validity bitmaps and bool
        data. `sel_start`/`sel_end` are the 64-bit block bounds and `out_len`
        the pre-counted set-bit count. Returns `(filtered, zero_count)` where
        `zero_count` is the number of zero bits in the output (the null count
        when filtering a validity bitmap).
        """
        comptime ALL_ONES = ~UInt64(0)
        var builder = Bitmap.alloc_zeroed(out_len)
        var out = builder.view()
        var bm_pos = 0
        var zero_count = 0
        var i = sel_start

        while i + 64 <= sel_end:
            var sel_word = sel.load_bits[DType.uint64](i)
            if sel_word == 0:
                i += 64
                while i + 64 <= sel_end and sel.load_bits[DType.uint64](i) == 0:
                    i += 64
                continue
            if sel_word == ALL_ONES:
                var run_start = i
                i += 64
                while (
                    i + 64 <= sel_end
                    and sel.load_bits[DType.uint64](i) == ALL_ONES
                ):
                    i += 64
                var j = run_start
                while j < i:
                    var src_word = self.load_bits[DType.uint64](j)
                    out.compressed_store(bm_pos, src_word, 64)
                    zero_count += 64 - Int(pop_count(src_word))
                    bm_pos += 64
                    j += 64
                continue

            var count = Int(pop_count(sel_word))
            var compressed = self.pext(i, sel_word)
            out.compressed_store(bm_pos, compressed, count)
            zero_count += count - Int(pop_count(compressed))
            bm_pos += count
            i += 64

        if i < sel_end:
            var tail = sel_end - i
            var mask = (UInt64(1) << UInt64(tail)) - 1
            var sel_word = sel.load_bits[DType.uint64](i) & mask
            if sel_word != 0:
                var count = Int(pop_count(sel_word))
                var compressed = self.pext(i, sel_word)
                out.compressed_store(bm_pos, compressed, count)
                zero_count += count - Int(pop_count(compressed))
                bm_pos += count

        debug_assert(
            bm_pos == out_len,
            "BitmapView.filter produced ",
            bm_pos,
            " bits but its destination was sized for ",
            out_len,
        )
        return builder.to_immutable(length=out_len), zero_count

    def __and__(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        return self.intersection(other)

    def __or__(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        return self.union(other)

    def __xor__(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        return self.symmetric_difference(other)

    def __sub__(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        return self.difference(other)

    def __invert__(self) raises -> Bitmap[mut=True]:
        """Return the bitwise NOT of this view as a new Bitmap (offset=0)."""
        var builder = Bitmap.alloc_uninit(self._length)
        apply[Self._not](self, builder.view())
        return builder^

    # --- Writable ---

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            t"BitmapView(offset={self._offset}, length={self._length})"
        )

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# apply — free-function overloads for BufferView and BitmapView
# ---------------------------------------------------------------------------


comptime UnaryFn[In: DType, Out: DType = In] = def[W: Int](
    SIMD[In, W]
) thin -> SIMD[Out, W]
"""A parameterized unary SIMD function type: maps a vector to a vector."""

comptime BinaryFn[In: DType, Out: DType = In] = def[W: Int](
    SIMD[In, W], SIMD[In, W]
) thin -> SIMD[Out, W]
"""A parameterized binary SIMD function type: combines two vectors into one."""

comptime MaskedFn[In: DType, Out: DType] = def[W: Int](
    SIMD[In, W], SIMD[DType.bool, W]
) thin -> SIMD[Out, W]
"""A parameterized SIMD function that takes a value vector and a validity mask."""


@always_inline
def _cpu_striped[
    Out: DType, Lane: def[W: Int](Int) -> None
](length: Int, lane: Lane, ctx: ExecContext):
    """Run ``lane(i)`` over ``[0, length)`` on the CPU, striped by ``ctx``.

    One ``vectorize`` over a half-open range; ``ctx.stripe`` decides whether it
    runs on the calling thread or across ``ctx.resolved_num_threads()``
    workers. Thread count is owned by ``ctx`` — no Mojo-internal heuristic. No
    ``align``: ``vectorize`` handles its own tail within each stripe.
    """
    comptime cpu_width = simd_byte_width() // size_of[Scalar[Out]]()

    @always_inline
    def span(wid: Int, start: Int, end: Int) {imm}:
        @always_inline
        def strip[
            W: Int
        ](i: Int) {imm start, imm lane,}:
            lane[W](start + i)

        vectorize[cpu_width](end - start, strip)

    ctx.stripe(length, span)


@always_inline
def _cpu_serial[
    In: DType, Lane: def[W: Int](Int) -> None
](length: Int, lane: Lane):
    """Run ``lane(i)`` over ``[0, length)`` on the calling thread only.

    What a **bit-packed** destination requires: a whole-byte-aligned stride is
    the only thing keeping parallel workers off each other's read-modify-write,
    and striping does not preserve it. ``In`` — the *input* dtype — sizes the
    lane, because a bitmap lane is one bit and cannot size anything.
    """
    comptime cpu_width = max(8, simd_byte_width() // size_of[Scalar[In]]())

    @always_inline
    def whole[
        W: Int
    ](i: Int) {imm lane,}:
        lane[W](i)

    vectorize[cpu_width](length, whole)


@always_inline
def _gpu_launch[
    Lane: (def[W: Int](Int) -> None) & RegisterPassable & ImplicitlyCopyable,
    gpu_width: Int,
](shape: Int, lane: Lane, ctx: ExecContext) raises:
    """One grid launch of ``lane`` over ``[0, shape)``.

    The only place in this module that talks to ``elementwise``, and therefore
    the only place that pays for the ``Coord``-shaped, ``alignment``-carrying
    signature it insists on. Everything upstream is written per element index.

    The ``RegisterPassable & ImplicitlyCopyable`` bound is ``elementwise``'s,
    and it is load-bearing: a lane closing over host state (an expression node,
    a `RecordBatch`) satisfies neither, which is precisely the set of lanes
    that cannot run on a device. Such lanes go straight to `_cpu_striped` /
    `_cpu_serial` instead.
    """

    # `lane` is captured **by value**, not `imm`. An `imm` capture is a
    # reference into the host stack frame; the kernel then dereferences a host
    # address on the device and silently computes garbage — it did, and there
    # is no diagnostic: six `test_views_gpu` cases simply returned wrong
    # values. The `ImplicitlyCopyable` half of `_gpu_launch`'s bound is what
    # makes the by-value capture legal.
    var device_lane = lane

    @always_inline
    def process[
        W: Int, alignment: Int = 1
    ](coord: Coord) {var device_lane,} -> None:
        device_lane[W](Int(coord[0].value()))

    elementwise[simd_width=gpu_width, target="gpu"](
        process, Coord(shape), ctx.device.value()
    )


@always_inline
def _apply_dispatch[
    Out: DType,
    gpu_ok: Bool,
    Lane: (def[W: Int](Int) -> None) & RegisterPassable & ImplicitlyCopyable,
](length: Int, lane: Lane, ctx: ExecContext) raises:
    """Run ``lane(i)`` over ``[0, length)`` into a **buffer** destination.

    ``gpu_ok`` is the caller's ``ExecContext.has_accelerator_support[...]`` check, passed
    as a comptime ``Bool`` so the GPU branch is dead-code-eliminated when
    unsupported.

    For a bit-packed destination use `_apply_packed_dispatch`; for a host-only
    lane call `_cpu_striped` directly.
    """
    if ctx.is_gpu():
        comptime if gpu_ok:
            comptime gpu_width = simd_width_of[Out, target=get_gpu_target()]()
            _gpu_launch[gpu_width=gpu_width](length, lane, ctx)
        else:
            raise Error("apply: no GPU accelerator available")
        return

    _cpu_striped[Out](length, lane, ctx)


@always_inline
def _apply_packed_dispatch[
    In: DType,
    gpu_ok: Bool,
    Lane: (def[W: Int](Int) -> None) & RegisterPassable & ImplicitlyCopyable,
](length: Int, lane: Lane, ctx: ExecContext) raises:
    """Run ``lane(i)`` over ``[0, length)`` into a **bit-packed** (bitmap)
    destination.

    The GPU launch is rounded up to a full ``gpu_width`` chunk so every store
    is a complete byte. The clamp keeps that over-write inside the destination
    bitmap's allocation: the launch covers at most `align_up(length, gpu_width)`
    bits, `gpu_width` is a power of two no wider than 64, and a `Bitmap` of
    `length` bits allocates `align_up(ceildiv(length, 8), 64)` bytes — which
    always covers `ceildiv(align_up(length, 64), 8)`. Nothing here rests on
    slack past the logical end. The CPU arm is `_cpu_serial` — see
    there for why it may not stripe.
    """
    if ctx.is_gpu():
        comptime if gpu_ok:
            comptime gpu_width = max(
                8, simd_width_of[In, target=get_gpu_target()]()
            )
            comptime max_pad = 64 // size_of[Scalar[In]]()
            _gpu_launch[gpu_width=gpu_width](
                min(math.align_up(length, gpu_width), length + max_pad),
                lane,
                ctx,
            )
        else:
            raise Error("apply: no GPU accelerator available")
        return

    _cpu_serial[In](length, lane)


def apply[
    Out: DType,
    Op: def[W: Int](Int) -> SIMD[Out, W],
](
    dst: BufferView[mut=True, Out, _],
    op: Op,
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Fill ``dst[i] = op(i)`` element-wise via the shared CPU serial/parallel
    dispatch — a *source-less* producer variant for fused expression lanes that
    compute each value from its index (evaluating a whole sub-tree) rather than
    reading an input buffer. GPU is not offered: the producer typically closes
    over a host `RecordBatch`, so only the CPU paths of ``_apply_dispatch`` apply.
    """
    var length = len(dst)

    @always_inline
    def lane[W: Int](i: Int) {imm} -> None:
        dst.store[W](i, op[W](i))

    if ctx.is_gpu():
        raise Error("apply: a producer lane closes over host state")
    _cpu_striped[Out](length, lane, ctx)


def apply[
    In: DType,
    Op: def[W: Int](Int) -> SIMD[DType.bool, W],
](
    dst: BitmapView[mut=True, _],
    op: Op,
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Bit-pack ``dst[i] = op(i)`` from the index — the bitmap counterpart of the
    source-less producer above, for fused predicate/comparison lanes that produce
    a `SIMD[bool, W]` per index (evaluating a whole sub-tree) with no input
    buffer. ``In`` sizes the SIMD lane to the operand's native type. Serial CPU
    only: bit-packed output needs whole-byte-aligned stride, so parallel workers
    would race on the read-modify-write (matching the other bitmap `apply`s).
    """
    var length = len(dst)

    @always_inline
    def lane[W: Int](i: Int) {imm} -> None:
        dst.store[W](i, op[W](i))

    if ctx.is_gpu():
        raise Error("apply: a producer lane closes over host state")
    _cpu_serial[In](length, lane)


def apply[
    In: DType,
    Out: DType,
    op: UnaryFn[In, Out],
](
    src: BufferView[In, _],
    dst: BufferView[mut=True, Out, _],
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Apply a type-mapping unary SIMD op element-wise over src into dst.

    The ``ctx`` parameter controls both device (CPU vs GPU) and whether CPU
    execution runs in parallel; see ``_apply_dispatch`` for the details.
    """
    var length = len(dst)

    @always_inline
    def lane[W: Int](i: Int) {imm} -> None:
        dst.store[W](i, op[W](src.load[W](i)))

    _apply_dispatch[Out, ExecContext.has_accelerator_support[In, Out]()](
        length, lane, ctx
    )


def apply[
    In: DType,
    Out: DType,
    Op: (def[W: Int](SIMD[In, W]) -> SIMD[Out, W])
    & RegisterPassable
    & ImplicitlyCopyable,
](
    src: BufferView[In, _],
    dst: BufferView[mut=True, Out, _],
    op: Op,
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Like the type-mapping unary ``apply`` above, but ``op`` may *capture*
    runtime state (e.g. a scale factor). Same CPU serial/parallel + GPU dispatch
    via ``ctx``, so a captured-state map still parallelizes and offloads."""
    var length = len(dst)

    @always_inline
    def lane[W: Int](i: Int) {imm} -> None:
        dst.store[W](i, op[W](src.load[W](i)))

    _apply_dispatch[Out, ExecContext.has_accelerator_support[In, Out]()](
        length, lane, ctx
    )


def apply[
    In: DType,
    Out: DType,
    op: BinaryFn[In, Out],
](
    lhs: BufferView[In, _],
    rhs: BufferView[In, _],
    dst: BufferView[mut=True, Out, _],
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Apply a type-mapping binary SIMD op element-wise over lhs,rhs into dst.
    """
    var length = len(dst)

    @always_inline
    def lane[W: Int](i: Int) {imm} -> None:
        dst.store[W](i, op[W](lhs.load[W](i), rhs.load[W](i)))

    _apply_dispatch[Out, ExecContext.has_accelerator_support[In, Out]()](
        length, lane, ctx
    )


def apply[
    In: DType,
    op: BinaryFn[In, DType.bool],
](
    lhs: BufferView[In, _],
    rhs: BufferView[In, _],
    dst: BitmapView[mut=True, _],
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Apply a binary comparison and bit-pack results into a bitmap.

    Compares W elements per call, packs the ``SIMD[bool, W]`` result
    into the output bitmap via ``BitmapView.store``.
    The tail over-read is bounded by `BitmapView._check_byte_read_range`. CPU
    parallelism via ``ctx`` is not used here — bit-packed outputs need
    whole-byte-aligned stride to avoid scalar read-modify-write races
    between workers; threading support is future work.
    """
    var length = len(dst)

    @always_inline
    def lane[W: Int](i: Int) {imm} -> None:
        dst.store[W](i, op[W](lhs.load[W](i), rhs.load[W](i)))

    _apply_packed_dispatch[In, ExecContext.has_accelerator_support[In]()](
        length, lane, ctx
    )


def apply[
    In: DType,
    op: UnaryFn[In, DType.bool],
](
    src: BufferView[In, _],
    dst: BitmapView[mut=True, _],
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Apply a unary predicate and bit-pack results into a bitmap.

    Maps W elements per call, packs the ``SIMD[bool, W]`` result into the
    output bitmap via ``BitmapView.store``. Used e.g. for numeric→bool casts
    (``op = {x => x.ne(0)}``). The tail over-read is bounded by
    `BitmapView._check_byte_read_range`. CPU parallelism via ``ctx`` is not
    used here — bit-packed outputs
    need whole-byte-aligned stride to avoid scalar read-modify-write races
    between workers; threading support is future work.
    """
    var length = len(dst)

    @always_inline
    def lane[W: Int](i: Int) {imm} -> None:
        dst.store[W](i, op[W](src.load[W](i)))

    _apply_packed_dispatch[In, ExecContext.has_accelerator_support[In]()](
        length, lane, ctx
    )


def apply[
    Out: DType,
    op: UnaryFn[DType.bool, Out],
](
    src: BitmapView[_],
    dst: BufferView[mut=True, Out, _],
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Apply a bool-to-Out unary op from a BitmapView into a BufferView."""
    var length = len(dst)

    @always_inline
    def lane[W: Int](i: Int) {imm} -> None:
        dst.store[W](i, op[W](src.load[W](i)))

    _apply_dispatch[Out, ExecContext.has_accelerator_support[Out]()](
        length, lane, ctx
    )


def apply[
    In: DType,
    Out: DType,
    op: MaskedFn[In, Out],
](
    src: BufferView[In, _],
    validity: BitmapView[_],
    dst: BufferView[mut=True, Out, _],
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Apply a masked SIMD op element-wise: op(values, validity) into dst."""
    var length = len(dst)

    @always_inline
    def lane[W: Int](i: Int) {imm} -> None:
        dst.store[W](i, op[W](src.load[W](i), validity.load[W](i)))

    _apply_dispatch[Out, ExecContext.has_accelerator_support[In, Out]()](
        length, lane, ctx
    )


def apply[
    Out: DType,
    op: MaskedFn[DType.bool, Out],
](
    src: BitmapView[_],
    validity: BitmapView[_],
    dst: BufferView[mut=True, Out, _],
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Apply a masked bool-to-Out op: op(bits, validity) into dst."""
    var length = len(dst)

    @always_inline
    def lane[W: Int](i: Int) {imm} -> None:
        dst.store[W](i, op[W](src.load[W](i), validity.load[W](i)))

    _apply_dispatch[Out, ExecContext.has_accelerator_support[Out]()](
        length, lane, ctx
    )


def apply[
    op: UnaryFn[DType.uint8],
](
    src: BitmapView[_],
    dst: BitmapView[mut=True, _],
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Apply a byte-level unary SIMD op from src into dst (pre-allocated, offset-0).

    Reads exactly ceil(length/8) source bytes — no over-read.
    Handles sub-byte bit-offset shifting automatically.
    GPU support is not yet implemented; ctx is reserved for future use.
    """
    # TODO: GPU bitmap op
    var byte_start = src.offset() >> 3
    var bit_shift = src.offset() & 7
    var rshift = UInt8(bit_shift)
    var lshift = UInt8(8 - bit_shift)
    var out_bytes = (len(src) + 7) >> 3
    comptime cpu_width = simd_width_of[DType.uint8]()

    if out_bytes == 0:
        return

    if bit_shift == 0:

        @always_inline
        def process_zero[
            W: Int
        ](i: Int) {imm dst, imm src, imm byte_start,}:
            dst.store_bytes[DType.uint8, W](
                i, op[W](src.load_bytes[DType.uint8, W](byte_start + i))
            )

        vectorize[cpu_width](out_bytes, process_zero)
        return

    # Non-zero bit_shift: shift-combine (lo >> rshift | hi << lshift).
    # Bulk covers indices 0 .. out_bytes-2; hi at i+1 is always in bounds.
    var bulk = out_bytes - 1
    if bulk > 0:

        @always_inline
        def process_shifted[
            W: Int
        ](i: Int) {imm dst, imm src, imm byte_start, imm rshift, imm lshift,}:
            var lo = src.load_bytes[DType.uint8, W](byte_start + i)
            var hi = src.load_bytes[DType.uint8, W](byte_start + i + 1)
            dst.store_bytes[DType.uint8, W](
                i, op[W]((lo >> rshift) | (hi << lshift))
            )

        vectorize[cpu_width](bulk, process_shifted)

    # Last output byte: read hi only when the view's bits span into the next
    # source byte, avoiding a read past the end of source data.
    var last_result = src.load_bytes[DType.uint8](byte_start + bulk) >> rshift
    var remaining_bits = len(src) - bulk * 8
    if remaining_bits > 8 - bit_shift:
        last_result = last_result | (
            src.load_bytes[DType.uint8](byte_start + bulk + 1) << lshift
        )
    dst.store_bytes[DType.uint8, 1](bulk, op[1](last_result))


def apply[
    op: BinaryFn[DType.uint8],
](
    lhs: BitmapView[_],
    rhs: BitmapView[_],
    dst: BitmapView[mut=True, _],
    ctx: ExecContext = ExecContext.serial(),
) raises:
    """Apply a byte-level binary SIMD op from lhs and rhs into dst (pre-allocated, offset-0).

    Reads exactly ceil(length/8) source bytes per operand — no over-read.
    Handles independent sub-byte bit-offset shifting for each operand.
    GPU support is not yet implemented; ctx is reserved for future use.
    """
    if len(lhs) != len(rhs):
        raise Error("BitmapView lengths must match")

    # TODO: GPU bitmap op
    var byte_start_a = lhs.offset() >> 3
    var bit_shift_a = lhs.offset() & 7
    var byte_start_b = rhs.offset() >> 3
    var bit_shift_b = rhs.offset() & 7
    var rs_a = UInt8(bit_shift_a)
    var ls_a = UInt8(8 - bit_shift_a)
    var rs_b = UInt8(bit_shift_b)
    var ls_b = UInt8(8 - bit_shift_b)
    var out_bytes = (len(lhs) + 7) >> 3
    comptime cpu_width = simd_width_of[DType.uint8]()

    if out_bytes == 0:
        return

    if bit_shift_a == 0 and bit_shift_b == 0:

        @always_inline
        def process_zero[
            W: Int
        ](i: Int) {
            imm dst,
            imm lhs,
            imm byte_start_a,
            imm rhs,
            imm byte_start_b,
        }:
            dst.store_bytes[DType.uint8, W](
                i,
                op[W](
                    lhs.load_bytes[DType.uint8, W](byte_start_a + i),
                    rhs.load_bytes[DType.uint8, W](byte_start_b + i),
                ),
            )

        vectorize[cpu_width](out_bytes, process_zero)
        return

    # At least one non-zero shift: shift-combine both operands.
    # When a shift is 0, ls = 8 so hi << 8 == 0, giving lo unchanged.
    # Bulk covers indices 0 .. out_bytes-2; hi at i+1 is always in bounds.
    var bulk = out_bytes - 1
    if bulk > 0:

        @always_inline
        def process_shifted[
            W: Int
        ](i: Int) {
            imm dst,
            imm lhs,
            imm byte_start_a,
            imm rhs,
            imm byte_start_b,
            imm rs_a,
            imm ls_a,
            imm rs_b,
            imm ls_b,
        }:
            var lo_a = lhs.load_bytes[DType.uint8, W](byte_start_a + i)
            var hi_a = lhs.load_bytes[DType.uint8, W](byte_start_a + i + 1)
            var lo_b = rhs.load_bytes[DType.uint8, W](byte_start_b + i)
            var hi_b = rhs.load_bytes[DType.uint8, W](byte_start_b + i + 1)
            dst.store_bytes[DType.uint8, W](
                i,
                op[W](
                    (lo_a >> rs_a) | (hi_a << ls_a),
                    (lo_b >> rs_b) | (hi_b << ls_b),
                ),
            )

        vectorize[cpu_width](bulk, process_shifted)

    # Last output byte: read hi only when bits span into the next source byte.
    var remaining_bits = len(lhs) - bulk * 8
    var result_a = lhs.load_bytes[DType.uint8](byte_start_a + bulk) >> rs_a
    var result_b = rhs.load_bytes[DType.uint8](byte_start_b + bulk) >> rs_b
    if remaining_bits > 8 - bit_shift_a:
        result_a = result_a | (
            lhs.load_bytes[DType.uint8](byte_start_a + bulk + 1) << ls_a
        )
    if remaining_bits > 8 - bit_shift_b:
        result_b = result_b | (
            rhs.load_bytes[DType.uint8](byte_start_b + bulk + 1) << ls_b
        )
    dst.store_bytes[DType.uint8, 1](bulk, op[1](result_a, result_b))


# ---------------------------------------------------------------------------
# reduce — vectorized scalar reduction over a BufferView
# ---------------------------------------------------------------------------


def _reduce_dispatch[
    T: DType,
    input_fn: def[W: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        T, W
    ],
    combine: def[W: Int](SIMD[T, W], SIMD[T, W]) thin -> SIMD[T, W],
](length: Int, identity: Scalar[T], ctx: ExecContext) raises -> Scalar[T]:
    """Dispatch a scalar reduction to GPU or CPU (serial / parallel) based on
    ``ctx``.

    - **GPU** — Mojo's ``_reduce_generator_wrapper[target="gpu"]``, allocating
      a 1-element device buffer and reading back via ``Buffer.to_cpu``.
    - **CPU multi-thread** — each worker computes a partial reduction over its
      slice using ``vectorize`` with a SIMD accumulator; the host thread folds
      the partials with ``combine``.
    - **CPU serial** — a single SIMD accumulator vectorized over the full
      range, then horizontally reduced via ``combine`` lane-by-lane.
    """

    if length == 0:
        return identity

    if ctx.is_gpu():
        # float16 triggers an internal f32→f16 rebind failure in the GPU
        # reduction backend (_reduce_generator_wrapper uses f32 accumulators
        # for f16 and cannot rebind back); bool fails Metal IR verification in
        # the same backend. Exclude both until the stdlib fixes them (they fall
        # through to the CPU path, which is correct — the reducer reads host
        # data anyway).
        comptime if (
            ExecContext.has_accelerator_support[T]()
            and T != DType.float16
            and T != DType.bool
        ):

            @always_inline
            @__parameter
            def combine_capturing[
                W: SIMDLength
            ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
                return combine[Int(W)](a, b)

            var dev_buf = Buffer.alloc_device[T](ctx.device.value(), 1)
            var dev_view = dev_buf.device_view[T]()

            @always_inline
            @__copy_capture(dev_view)
            @__parameter
            def output_fn_gpu[
                W: SIMDLength, rank: Int
            ](idx: IndexList[rank], val: SIMD[T, W]):
                dev_view.store[1](0, val[0])

            _reduce_generator_wrapper[
                T,
                input_fn,
                output_fn_gpu,
                combine_capturing,
                target="gpu",
                reduce_dim=0,
            ](Coord(length), identity, ctx.device.value())
            return (
                dev_buf.to_immutable()
                .to_cpu(ctx.device.value())
                .view[T]()
                .load[1](0)
            )
        else:
            raise Error("reduce: no GPU accelerator available")

    comptime cpu_width = simd_byte_width() // size_of[Scalar[T]]()

    # This stripes by hand, unlike `_apply_dispatch` above, and deliberately so.
    # Routing it through `ctx.stripe` works and removes the duplicated fold body,
    # but it forces the serial arm to allocate a one-slot partials buffer it does
    # not otherwise need. Measured, five interleaved repeats: `sumint64_1k`
    # 0.19-0.20 -> 0.30-0.32 us and `sumfloat64_1k` 0.23-0.24 -> 0.34-0.37 us —
    # ranges fully disjoint, ~55% on small reductions, for one heap allocation.
    # A reduce is a fold *plus* a merge, and the merge's scratch is exactly what
    # a serial fold should not pay for. The parallel arm below allocates it
    # because it genuinely needs it.
    if ctx.wants_parallel(length):
        var workers = ctx.resolved_num_threads()
        var chunk = ceildiv(length, workers)
        var partials = Buffer.alloc_zeroed[T](workers)
        var partials_view = partials.view[T]()
        for w in range(workers):
            partials_view.store[1](w, identity)

        @always_inline
        def task(
            wid: Int,
        ) {imm chunk, imm length, imm identity, imm partials_view,}:
            var start = wid * chunk
            var end = min(start + chunk, length)
            if end <= start:
                return
            var simd_acc = SIMD[T, cpu_width](identity)
            var i = start
            var simd_end = start + ((end - start) // cpu_width) * cpu_width
            while i < simd_end:
                simd_acc = combine[cpu_width](
                    simd_acc, input_fn[cpu_width, 1](IndexList[1](i))
                )
                i += cpu_width
            var acc = identity
            comptime for k in range(cpu_width):
                acc = combine[1](acc, SIMD[T, 1](simd_acc[k]))
            while i < end:
                acc = combine[1](acc, input_fn[1, 1](IndexList[1](i)))
                i += 1
            partials_view.store[1](wid, acc)

        sync_parallelize(task, workers)

        var acc = identity
        for w in range(workers):
            acc = combine[1](acc, SIMD[T, 1](partials_view.load[1](w)))
        return acc

    var simd_acc = SIMD[T, cpu_width](identity)
    var i = 0
    var simd_end = (length // cpu_width) * cpu_width
    while i < simd_end:
        simd_acc = combine[cpu_width](
            simd_acc, input_fn[cpu_width, 1](IndexList[1](i))
        )
        i += cpu_width
    var acc = identity
    comptime for k in range(cpu_width):
        acc = combine[1](acc, SIMD[T, 1](simd_acc[k]))
    while i < length:
        acc = combine[1](acc, input_fn[1, 1](IndexList[1](i)))
        i += 1
    return acc


def reduce[
    In: DType,
    combine: def[T: DType, W: Int](SIMD[T, W], SIMD[T, W]) thin -> SIMD[T, W],
    Acc: DType = In,
](
    src: BufferView[In, _],
    identity: Scalar[Acc],
    ctx: ExecContext = ExecContext.serial(),
) raises -> Scalar[Acc]:
    """Reduce ``src`` to a scalar with a SIMD ``combine``, accumulating in ``Acc``.

    ``In`` is ``src``'s element dtype; ``Acc`` is the accumulator (and result)
    dtype, and defaults to ``In`` — the plain same-type reduce is just
    ``reduce[In, combine]``. When they differ, each SIMD lane is cast from ``In``
    to ``Acc`` as it is loaded, so a narrow input can accumulate in a wider type
    (e.g. summing ``int16`` into ``int64`` without overflow) **without
    materializing a widened copy** of ``src`` — the widening is fused into the
    load. When ``Acc == In`` (the default) the per-lane cast is a compile-time
    no-op, so the same-type reduce carries zero overhead. ``combine`` is generic
    over the dtype and applied at ``Acc``.
    """

    @always_inline
    @__parameter
    def input_fn[W: Int, rank: Int](idx: IndexList[rank]) -> SIMD[Acc, W]:
        return src.load[W](idx[0]).cast[Acc]()

    return _reduce_dispatch[Acc, input_fn, combine[Acc, _]](
        len(src), identity, ctx
    )


def reduce[
    In: DType,
    combine: def[T: DType, W: Int](SIMD[T, W], SIMD[T, W]) thin -> SIMD[T, W],
    Acc: DType = In,
](
    src: BufferView[In, _],
    bitmap: BitmapView[_],
    identity: Scalar[Acc],
    ctx: ExecContext = ExecContext.serial(),
) raises -> Scalar[Acc]:
    """``reduce`` that skips nulls: lanes whose ``bitmap`` bit is False contribute
    ``identity`` (a no-op under ``combine``) rather than their value. Same
    ``In``→``Acc`` per-lane widening as the dense overload (a compile-time no-op
    when ``Acc == In``, the default)."""

    @always_inline
    @__parameter
    def input_fn[W: Int, rank: Int](idx: IndexList[rank]) -> SIMD[Acc, W]:
        var i = idx[0]
        return bitmap.load[W](i).select(
            src.load[W](i).cast[Acc](), SIMD[Acc, W](identity)
        )

    return _reduce_dispatch[Acc, input_fn, combine[Acc, _]](
        len(src), identity, ctx
    )


comptime CheckedFn[In: DType, Out: DType] = def[W: Int](
    SIMD[In, W]
) thin -> Tuple[SIMD[Out, W], SIMD[DType.bool, W]]
"""A unary map that returns ``(mapped value, bad?)`` per lane — the mapped value
and whether that lane failed, computed together so neither is recomputed."""


def apply_checked[
    In: DType,
    Out: DType,
    op: CheckedFn[In, Out],
](src: BufferView[In, _], dst: BufferView[mut=True, Out, _],) raises:
    """Map ``op(src) → dst`` where ``op`` yields ``(value, bad)`` per lane; the
    whole map *fails* the moment any lane is flagged ``bad``, reporting the first
    offending input value.

    Serial by design: a failing block aborts the map, and exceptions can't cross
    parallel-worker or GPU-kernel boundaries. Use the plain ``apply`` for the
    unconditional, parallel / device-dispatched map; layer this on when a value
    must be validated as it is mapped (the mapped result is reused, not
    recomputed)."""
    var length = len(dst)
    comptime w = max(8, simd_byte_width() // size_of[Scalar[In]]())
    var i = 0
    var simd_end = (length // w) * w
    while i < simd_end:
        _checked_block[In, Out, op, w](src, dst, i)
        i += w
    while i < length:
        _checked_block[In, Out, op, 1](src, dst, i)
        i += 1


def apply_checked[
    In: DType,
    Out: DType,
    op: CheckedFn[In, Out],
](
    src: BufferView[In, _],
    validity: BitmapView[_],
    dst: BufferView[mut=True, Out, _],
) raises:
    """``apply_checked`` that skips null lanes (they may hold unrepresentable
    junk): the ``bad`` flag is masked by ``validity`` so only valid lanes can
    fail the map."""
    var length = len(dst)
    comptime w = max(8, simd_byte_width() // size_of[Scalar[In]]())
    var i = 0
    var simd_end = (length // w) * w
    while i < simd_end:
        _checked_block[In, Out, op, w](src, dst, i, validity.load[w](i))
        i += w
    while i < length:
        _checked_block[In, Out, op, 1](src, dst, i, validity.load[1](i))
        i += 1


@always_inline
def _checked_block[
    In: DType,
    Out: DType,
    op: CheckedFn[In, Out],
    W: Int,
](
    src: BufferView[In, _],
    dst: BufferView[mut=True, Out, _],
    i: Int,
    valid: SIMD[DType.bool, W] = SIMD[DType.bool, W](fill=True),
) raises:
    """Map one block (storing the mapped value) and raise on the first valid
    lane the op flagged."""
    var v = src.load[W](i)
    var result = op[W](v)
    dst.store[W](i, result[0])
    var bad = result[1] & valid
    if bad.reduce_or():
        comptime for k in range(W):
            if bad[k]:
                raise Error(
                    t"value {SIMD[In, 1](v[k])} is not representable in the"
                    t" target type"
                )
