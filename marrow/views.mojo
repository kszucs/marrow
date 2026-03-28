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
from std.bit import pop_count
import std.math as math
from std.math import iota
from std.memory import memcpy, memset
from std.builtin.device_passable import DevicePassable

from .buffers import Buffer, Bitmap


# ---------------------------------------------------------------------------
# BufferView — typed element view
# ---------------------------------------------------------------------------


struct BufferView[
    mut: Bool,
    //,
    T: DType,
    origin: Origin[mut=mut],
](
    ImplicitlyCopyable,
    Sized,
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

    var _data: UnsafePointer[Scalar[Self.T], Self.origin]
    var _len: Int

    # --- lifecycle ---

    @always_inline
    def __init__(
        out self,
        *,
        ptr: UnsafePointer[Scalar[Self.T], Self.origin],
        length: Int,
    ):
        self._data = ptr
        self._len = length

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._len = copy._len

    # --- Sized ---

    @always_inline
    def __len__(self) -> Int:
        return self._len

    # --- Boolable ---

    @always_inline
    def __bool__(self) -> Bool:
        return self._len > 0

    # --- Element access ---

    @always_inline
    def __getitem__(self, index: Int) -> Scalar[Self.T]:
        debug_assert(0 <= index < self._len, "BufferView index out of bounds")
        return self._data[index]

    @always_inline
    def __getitem__(self, slc: Slice) -> Self:
        var start: Int
        var end: Int
        var step: Int
        start, end, step = slc.indices(self._len)
        debug_assert(step == 1, "BufferView slice step must be 1")
        return Self(ptr=self._data + start, length=end - start)

    @always_inline
    def __contains__(self, value: Scalar[Self.T]) -> Bool:
        for i in range(self._len):
            if self._data[i] == value:
                return True
        return False

    @always_inline
    def unsafe_get(self, index: Int) -> Scalar[Self.T]:
        return self._data[index]

    @always_inline
    def unsafe_set(self, index: Int, value: Scalar[Self.T]):
        comptime assert Self.mut, "cannot write to immutable BufferView"
        self._data[index] = value

    # --- SIMD ---

    @always_inline
    def simd_load[W: Int](self, index: Int) -> SIMD[Self.T, W]:
        return self._data.load[width=W](index)

    @always_inline
    def simd_store[W: Int](self, index: Int, value: SIMD[Self.T, W]):
        comptime assert Self.mut, "cannot write to immutable BufferView"
        self._data.store(index, value)

    # --- Slicing ---

    @always_inline
    def slice(self, offset: Int, length: Int = -1) -> Self:
        var actual = length if length >= 0 else self._len - offset
        return Self(ptr=self._data + offset, length=actual)

    # --- Raw pointer access ---

    @always_inline
    def unsafe_ptr(self) -> UnsafePointer[Scalar[Self.T], Self.origin]:
        return self._data

    # --- Vectorized operations ---

    def apply[
        func: def[W: Int](SIMD[Self.T, W]) -> SIMD[Self.T, W]
    ](self: BufferView[mut=True, T=Self.T, origin=_]):
        """Apply a SIMD function in-place over all elements."""
        comptime width = simd_byte_width() // size_of[Scalar[Self.T]]()
        var i = 0
        while i + width <= self._len:
            self._data.store(i, func[width](self._data.load[width=width](i)))
            i += width
        while i < self._len:
            self._data[i] = func[1](self._data[i])
            i += 1

    def count[
        F: def[W: Int](SIMD[Self.T, W]) -> SIMD[DType.bool, W]
    ](self, func: F) -> Int:
        """Count elements matching a vectorized predicate."""
        comptime width = simd_byte_width() // size_of[Scalar[Self.T]]()
        var total = 0
        var i = 0
        while i + width <= self._len:
            total += Int(
                func[width](self._data.load[width=width](i))
                .cast[DType.uint8]()
                .reduce_add()
            )
            i += width
        while i < self._len:
            if func[1](self._data[i]):
                total += 1
            i += 1
        return total

    # --- Writable ---

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"BufferView(length={self._len})")


# ---------------------------------------------------------------------------
# BitmapView — bit-packed view with BitSet-style API
# ---------------------------------------------------------------------------


struct BitmapView[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
](
    Boolable,
    ImplicitlyCopyable,
    Sized,
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

    var _data: UnsafePointer[UInt8, Self.origin]
    var _offset: Int  # bit offset into _data
    var _len: Int  # number of logical bits

    # --- lifecycle ---

    @always_inline
    def __init__(
        out self,
        *,
        ptr: UnsafePointer[UInt8, Self.origin],
        offset: Int,
        length: Int,
    ):
        self._data = ptr
        self._offset = offset
        self._len = length

    def __init__(out self: BitmapView[ImmutExternalOrigin], bm: Bitmap[mut=False]):
        """Construct a non-owning view over an immutable Bitmap."""
        self._data = rebind[UnsafePointer[UInt8, ImmutExternalOrigin]](
            bm.buffer.ptr
        )
        self._offset = bm.offset
        self._len = bm.length

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._offset = copy._offset
        self._len = copy._len

    # --- Sized ---

    @always_inline
    def __len__(self) -> Int:
        return self._len

    # --- Boolable (any bit set) ---

    def __bool__(self) -> Bool:
        """Return True if any bit in the view is set."""
        if self._len == 0:
            return False

        comptime width = simd_width_of[DType.uint8]()
        var ptr = self._data
        var bit_start = self._offset
        var bit_end = bit_start + self._len
        var byte_start = bit_start >> 3
        var byte_end = (bit_end + 7) >> 3
        var nbytes = byte_end - byte_start

        var first_mask = UInt8(0xFF) << UInt8(bit_start & 7)
        var last_mask = UInt8(
            (1 << ((bit_end - 1) & 7) + 1) - 1
        ) if bit_end & 7 != 0 else UInt8(0xFF)

        if nbytes == 1:
            return (ptr[byte_start] & first_mask & last_mask) != 0

        if (ptr[byte_start] & first_mask) != 0:
            return True
        if (ptr[byte_end - 1] & last_mask) != 0:
            return True

        var i = byte_start + 1
        var end = byte_end - 1
        while i + width <= end:
            if (ptr + i).load[width=width]().reduce_or() != 0:
                return True
            i += width
        while i < end:
            if ptr[i] != 0:
                return True
            i += 1

        return False

    # --- Element access (BitSet-style) ---

    @always_inline
    def __getitem__(self, index: Int) -> Bool:
        return self.test(index)

    @always_inline
    def __getitem__(self, slc: Slice) -> Self:
        var start: Int
        var end: Int
        var step: Int
        start, end, step = slc.indices(self._len)
        debug_assert(step == 1, "BitmapView slice step must be 1")
        return Self(
            ptr=self._data, offset=self._offset + start, length=end - start
        )

    @always_inline
    def bit_offset(self) -> Int:
        """Return the bit offset into the backing buffer."""
        return self._offset

    @always_inline
    def test(self, index: Int) -> Bool:
        """Test if the bit at ``index`` is set."""
        var bit_index = self._offset + index
        return Bool((self._data[bit_index >> 3] >> UInt8(bit_index & 7)) & 1)

    @always_inline
    def mask[dtype: DType, W: Int](self, index: Int) -> SIMD[DType.bool, W]:
        """Expand W consecutive bits starting at logical ``index`` into a
        SIMD bool vector.

        Each lane j is True iff bit (index + j) is set. Loads a full UInt32
        unconditionally — safe because Arrow buffers are 64-byte padded.
        """
        var abs_pos = self._offset + index
        var byte_idx = abs_pos >> 3
        var bit_off = abs_pos & 7

        var bits = (self._data + byte_idx).bitcast[UInt32]().load[alignment=1]()
        bits >>= UInt32(bit_off)

        return (
            (SIMD[DType.uint32, W](bits) >> iota[DType.uint32, W]()) & 1
        ).cast[DType.bool]()

    @always_inline
    def load_word(self, index: Int) -> UInt64:
        """Load 64 bits starting at logical position ``index``.

        Handles the view's bit offset correctly. Safe because Arrow buffers
        are 64-byte padded.
        """
        var abs_pos = self._offset + index
        var byte_idx = abs_pos >> 3
        var bit_off = abs_pos & 7
        var raw = (self._data + byte_idx).bitcast[UInt64]().load[alignment=1]()
        return raw >> UInt64(bit_off)

    # --- Slicing ---

    @always_inline
    def slice(self, offset: Int, length: Int) -> Self:
        """Return a zero-copy sub-view of ``length`` bits starting at
        ``offset``."""
        return Self(ptr=self._data, offset=self._offset + offset, length=length)

    # --- Bulk read operations ---

    def all_set(self) -> Bool:
        """Return True if all bits in the view are set."""
        if self._len == 0:
            return True

        comptime width = simd_width_of[DType.uint8]()
        var ptr = self._data
        var bit_start = self._offset
        var bit_end = bit_start + self._len
        var byte_start = bit_start >> 3
        var byte_end = (bit_end + 7) >> 3
        var nbytes = byte_end - byte_start

        var first_fill = ~(UInt8(0xFF) << UInt8(bit_start & 7))
        var last_fill = ~(
            UInt8((1 << ((bit_end - 1) & 7) + 1) - 1)
        ) if bit_end & 7 != 0 else UInt8(0)

        if nbytes == 1:
            return (ptr[byte_start] | first_fill | last_fill) == 0xFF

        if (ptr[byte_start] | first_fill) != 0xFF:
            return False
        if (ptr[byte_end - 1] | last_fill) != 0xFF:
            return False

        var i = byte_start + 1
        var end = byte_end - 1
        while i + width <= end:
            if (ptr + i).load[width=width]().reduce_and() != 0xFF:
                return False
            i += width
        while i < end:
            if ptr[i] != 0xFF:
                return False
            i += 1

        return True

    def _aligned_byte_range(
        self,
    ) -> Tuple[UnsafePointer[UInt8, Self.origin], Int, Int, Int]:
        """Return 64-byte-aligned pointer and byte range with boundary bits.

        Returns (ptr, total_bytes, lead_bits, trail_bits).
        Arrow buffers are 64-byte aligned and zero-padded, so reading the
        full range is always safe.
        """
        var byte_start = self._offset >> 3
        var bit_end = self._offset + self._len
        var byte_end = (bit_end + 7) >> 3
        var aligned_start = math.align_down(byte_start, 64)
        var aligned_end = math.align_up(byte_end, 64)
        var lead_bits = self._offset - (aligned_start << 3)
        var trail_bits = (aligned_end - byte_end) * 8 + (bit_end & 7)
        return Tuple(
            self._data + aligned_start,
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

        if self._len == 0:
            return (0, 0, 0)

        ptr, total_bytes, lead_bits, trail_bits = self._aligned_byte_range()

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
                    (ptr + i + (j * 2) * width).load[width=width]()
                )
                acc1 += pop_count(
                    (ptr + i + (j * 2 + 1) * width).load[width=width]()
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
                acc += pop_count((ptr + i + j * width).load[width=width]())
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
                count -= Int(pop_count(ptr[i]))
            if lead_sub_byte:
                count -= Int(
                    pop_count(ptr[lead_bytes] & UInt8((1 << lead_sub_byte) - 1))
                )
        if trail_bits:
            var trail_bytes = trail_bits >> 3
            var trail_sub_byte = trail_bits & 7
            var first_trail = total_bytes - trail_bytes
            if trail_sub_byte:
                count -= Int(
                    pop_count(ptr[first_trail - 1] >> UInt8(trail_sub_byte))
                )
            for i in range(first_trail, total_bytes):
                count -= Int(pop_count(ptr[i]))

        if count == 0:
            return (0, 0, 0)

        var start = max(0, first_byte * 8 - lead_bits)
        var end = min(self._len, last_byte * 8 - lead_bits)
        start = (start // 64) * 64
        end = min(self._len, ((end + 63) // 64) * 64)
        return (count, start, end)

    def count_set_bits(self) -> Int:
        """Count set bits in the view."""
        count, _, _ = self.count_set_bits_with_range()
        return count

    # --- Equality ---

    def __eq__(self, other: BitmapView[_]) -> Bool:
        """Return True if both views have identical logical bit patterns."""
        if self._len != len(other):
            return False
        # Word-level XOR comparison.
        var i = 0
        while i + 64 <= self._len:
            if self.load_word(i) ^ other.load_word(i) != 0:
                return False
            i += 64
        if i < self._len:
            var tail = self._len - i
            var mask = (UInt64(1) << UInt64(tail)) - 1
            if (self.load_word(i) ^ other.load_word(i)) & mask != 0:
                return False
        return True

    # --- Write operations (mut=True only, BitSet-style) ---

    @always_inline
    def set(self: BitmapView[mut=True, origin=_], index: Int):
        """Set the bit at ``index`` to 1."""
        var abs_index = self._offset + index
        var byte_index = abs_index >> 3
        var bit_mask = UInt8(1 << (abs_index & 7))
        self._data[byte_index] = self._data[byte_index] | bit_mask

    @always_inline
    def clear(self: BitmapView[mut=True, origin=_], index: Int):
        """Set the bit at ``index`` to 0."""
        var abs_index = self._offset + index
        var byte_index = abs_index >> 3
        var bit_mask = UInt8(1 << (abs_index & 7))
        self._data[byte_index] = self._data[byte_index] & ~bit_mask

    @always_inline
    def toggle(self: BitmapView[mut=True, origin=_], index: Int):
        """Invert the bit at ``index``."""
        var abs_index = self._offset + index
        var byte_index = abs_index >> 3
        var bit_mask = UInt8(1 << (abs_index & 7))
        self._data[byte_index] = self._data[byte_index] ^ bit_mask

    def set_all(self: BitmapView[mut=True, origin=_]):
        """Set all bits in the view to 1."""
        _set_range(self._data, self._offset, self._len, True)

    def clear_all(self: BitmapView[mut=True, origin=_]):
        """Set all bits in the view to 0."""
        _set_range(self._data, self._offset, self._len, False)

    def toggle_all(self: BitmapView[mut=True, origin=_]):
        """Invert all bits in the view."""
        # Word-level NOT over the logical range.
        var i = 0
        while i + 64 <= self._len:
            var word = self.load_word(i)
            self.deposit_bits(i, ~word, 64)
            i += 64
        if i < self._len:
            var tail = self._len - i
            var mask = (UInt64(1) << UInt64(tail)) - 1
            var word = self.load_word(i) & mask
            self.deposit_bits(i, ~word & mask, tail)

    @always_inline
    def deposit_bits(
        self: BitmapView[mut=True, origin=_],
        bit_offset: Int,
        bits: UInt64,
        count: Int,
    ):
        """Deposit ``count`` LSBs from ``bits`` into the view at
        ``bit_offset``.

        The view must be zero-filled for the target range, as this uses OR
        to set bits.
        """
        if count == 0:
            return
        var abs_offset = self._offset + bit_offset
        var byte_idx = abs_offset >> 3
        var bit_off = abs_offset & 7
        var shifted = bits << UInt64(bit_off)
        var ptr64 = (self._data + byte_idx).bitcast[UInt64]()
        ptr64.store[alignment=1](ptr64.load[alignment=1]() | shifted)
        if bit_off > 0 and bit_off + count > 64:
            self._data[byte_idx + 8] = self._data[byte_idx + 8] | UInt8(
                bits >> UInt64(64 - bit_off)
            )

    def copy_from[
        src_origin: Origin[_]
    ](
        self: BitmapView[mut=True, origin=_],
        src: BitmapView[src_origin],
        dst_start: Int,
        length: Int,
    ):
        """Bulk-copy ``length`` bits from ``src`` (from its start) into self
        at ``dst_start``."""
        _copy_bits(
            self._data,
            self._offset + dst_start,
            src._data,
            src._offset,
            length,
        )

    # --- Set operations (return Buffer with offset=0) ---

    def intersection(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Return the bitwise AND of self and other."""
        return self._binop[_and](other)

    def union(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Return the bitwise OR of self and other."""
        return self._binop[_or](other)

    def symmetric_difference(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Return the bitwise XOR of self and other."""
        return self._binop[_xor](other)

    def difference(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Return self AND NOT other."""
        return self._binop[_and_not](other)

    def __and__(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        return self.intersection(other)

    def __or__(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        return self.union(other)

    def __xor__(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        return self.symmetric_difference(other)

    def __sub__(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        return self.difference(other)

    def __invert__(self) -> Bitmap[mut=True]:
        """Return the bitwise NOT of this view as a new Bitmap (offset=0)."""
        comptime width = simd_width_of[DType.uint8]()
        comptime assert 64 % width == 0
        comptime unroll = 64 // width

        src, total_bytes, lead_bits, _ = self._aligned_byte_range()

        var builder = Buffer.alloc_uninit(total_bytes)
        var dst = builder.ptr
        for i in range(0, total_bytes, 64):
            comptime for j in range(unroll):
                comptime k = j * width
                (dst + i + k).store(~(src + i + k).load[width=width]())

        return _normalize(builder^, lead_bits, self._len)

    def _binop[
        op: def[W: Int](
            SIMD[DType.uint8, W], SIMD[DType.uint8, W]
        ) -> SIMD[DType.uint8, W]
    ](self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Apply a byte-level SIMD binary op. Output always has offset=0.

        Two code paths based on sub-byte alignment:
        - Same sub-byte offset: direct SIMD op, no bit shifting.
        - Different sub-byte offset: ``other`` is bit-shifted to align with
          ``self`` via overlapping loads.
        """
        if self._len != len(other):
            raise Error("BitmapView lengths must match")
        comptime width = simd_width_of[DType.uint8]()
        comptime assert 64 % width == 0
        comptime unroll = 64 // width

        src_a, total_bytes, lead_bits_a, _ = self._aligned_byte_range()
        ptr_b, _, lead_bits_b, _ = other._aligned_byte_range()

        var src_b = ptr_b + ((lead_bits_b >> 3) - (lead_bits_a >> 3))
        var builder = Buffer.alloc_uninit(total_bytes)
        var dst = builder.ptr

        if lead_bits_a & 7 == lead_bits_b & 7:
            for i in range(0, total_bytes, 64):
                comptime for j in range(unroll):
                    comptime k = j * width
                    (dst + i + k).store(
                        op(
                            (src_a + i + k).load[width=width](),
                            (src_b + i + k).load[width=width](),
                        )
                    )
        else:
            var in_byte_shift = (lead_bits_b - lead_bits_a) & 7
            var rs = UInt8(in_byte_shift)
            var ls = UInt8(8 - in_byte_shift)
            for i in range(0, total_bytes, 64):
                comptime for j in range(unroll):
                    comptime k = j * width
                    var a = (src_a + i + k).load[width=width]()
                    var lo = (src_b + i + k).load[width=width]()
                    var hi = (src_b + i + k + 1).load[width=width]()
                    (dst + i + k).store(op(a, (lo >> rs) | (hi << ls)))

        return _normalize(builder^, lead_bits_a, self._len)

    # --- Writable ---

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"BitmapView(offset={self._offset}, length={self._len})")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# SIMD byte-level binary op helpers
# ---------------------------------------------------------------------------


@always_inline
def _and[
    W: Int
](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a & b


@always_inline
def _or[
    W: Int
](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a | b


@always_inline
def _xor[
    W: Int
](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a ^ b


@always_inline
def _and_not[
    W: Int
](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a & ~b


def _normalize(var buffer: Buffer[mut=True], lead_bits: Int, length: Int) -> Bitmap[mut=True]:
    """Shift bits left by ``lead_bits`` so the result starts at offset 0.

    When ``lead_bits == 0`` (the common case for freshly-built arrays),
    wraps the buffer in a Bitmap unchanged.
    """
    if lead_bits == 0:
        return Bitmap[mut=True](buffer^, 0, length)
    var out_bytes = math.align_up(math.ceildiv(length, 8), 64)
    var dst = Buffer.alloc_zeroed(out_bytes)
    _copy_bits(
        dst.ptr,
        0,
        buffer.ptr,
        lead_bits,
        length,
    )
    return Bitmap[mut=True](dst^, 0, length)


# ---------------------------------------------------------------------------
# Bit-level helpers (ported from BitmapBuilder)
# ---------------------------------------------------------------------------


def _set_range[
    origin: MutOrigin
](ptr: UnsafePointer[UInt8, origin], abs_start: Int, length: Int, value: Bool):
    """Set ``length`` bits starting at absolute bit ``abs_start``."""
    if length == 0:
        return
    var end = abs_start + length
    var start_byte = abs_start >> 3
    var start_bit = abs_start & 7
    var end_byte = end >> 3
    var end_bit = end & 7
    var fill = UInt8(255 if value else 0)

    if start_byte == end_byte:
        var mask = UInt8((1 << end_bit) - 1) & (UInt8(0xFF) << UInt8(start_bit))
        if value:
            ptr[start_byte] = ptr[start_byte] | mask
        else:
            ptr[start_byte] = ptr[start_byte] & ~mask
        return

    if start_bit != 0:
        var mask = UInt8(0xFF) << UInt8(start_bit)
        if value:
            ptr[start_byte] = ptr[start_byte] | mask
        else:
            ptr[start_byte] = ptr[start_byte] & ~mask
        start_byte += 1

    if end_bit != 0:
        var mask = UInt8((1 << end_bit) - 1)
        if value:
            ptr[end_byte] = ptr[end_byte] | mask
        else:
            ptr[end_byte] = ptr[end_byte] & ~mask

    if end_byte > start_byte:
        memset(ptr + start_byte, fill, end_byte - start_byte)


def _copy_bits[
    dst_origin: MutOrigin
](
    dst: UnsafePointer[UInt8, dst_origin],
    dst_offset: Int,
    src_ptr: UnsafePointer[UInt8, _],
    src_offset: Int,
    length: Int,
):
    """Bulk-copy ``length`` bits between arbitrary bit offsets.

    Three code paths:
    1. Same sub-byte alignment → memcpy for middle bytes.
    2. Different alignment → shift-and-merge byte-by-byte.
    3. Short runs (< 16 bits) → bit-by-bit fallback.
    """
    if length == 0:
        return

    if length < 16:
        for i in range(length):
            var s_byte = (src_offset + i) >> 3
            var s_bit = (src_offset + i) & 7
            var val = (src_ptr[s_byte] >> UInt8(s_bit)) & 1
            var d_byte = (dst_offset + i) >> 3
            var d_bit = (dst_offset + i) & 7
            var d_mask = UInt8(1 << d_bit)
            if val:
                dst[d_byte] = dst[d_byte] | d_mask
            else:
                dst[d_byte] = dst[d_byte] & ~d_mask
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
            dst[dst_byte] = (dst[dst_byte] & keep_mask) | (
                src_ptr[src_byte] & ~keep_mask
            )
            src_byte += 1
            dst_byte += 1

        if end_byte > dst_byte:
            memcpy(
                dest=dst + dst_byte,
                src=src_ptr + src_byte,
                count=end_byte - dst_byte,
            )

        if end_sub != 0:
            var trail_byte_src = src_byte + (end_byte - dst_byte)
            var keep_mask = UInt8(0xFF) << UInt8(end_sub)
            dst[end_byte] = (dst[end_byte] & keep_mask) | (
                src_ptr[trail_byte_src] & ~keep_mask
            )
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
                shifted = (src_ptr[src_byte] >> UInt8(delta)) | (
                    src_ptr[src_byte + 1] << UInt8(8 - delta)
                )
            else:
                shifted = src_ptr[src_byte] << UInt8(-delta)
                if src_byte > 0:
                    shifted |= src_ptr[src_byte - 1] >> UInt8(8 + delta)
            dst[dst_byte_start] = (dst[dst_byte_start] & keep_mask) | (
                shifted & ~keep_mask
            )
            dst_byte_start += 1

        var src_bit_pos = src_offset + ((dst_byte_start << 3) - dst_offset)
        for j in range(dst_byte_start, end_byte):
            var sb = src_bit_pos >> 3
            var so = src_bit_pos & 7
            if so == 0:
                dst[j] = src_ptr[sb]
            else:
                dst[j] = (src_ptr[sb] >> UInt8(so)) | (
                    src_ptr[sb + 1] << UInt8(8 - so)
                )
            src_bit_pos += 8

        if end_sub != 0:
            var sb = src_bit_pos >> 3
            var so = src_bit_pos & 7
            var shifted: UInt8
            if so == 0:
                shifted = src_ptr[sb]
            else:
                shifted = (src_ptr[sb] >> UInt8(so)) | (
                    src_ptr[sb + 1] << UInt8(8 - so)
                )
            var keep_mask = UInt8(0xFF) << UInt8(end_sub)
            dst[end_byte] = (dst[end_byte] & keep_mask) | (shifted & ~keep_mask)
