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
from std.sys import size_of, has_accelerator
from std.bit import pop_count
import std.math as math
from std.math import iota
from std.memory import memcpy, memset
from std.builtin.device_passable import DevicePassable
from std.sys.intrinsics import prefetch
from std.algorithm.functional import elementwise
from std.utils.index import IndexList
from std.gpu.host import DeviceContext, get_gpu_target

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

    var _data: UnsafePointer[Scalar[Self.T], Self.origin]
    var _length: Int

    # --- DevicePassable ---

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return String(t"BufferView[{Self.T}]")

    # --- lifecycle ---

    @always_inline
    def __init__(
        out self,
        *,
        ptr: UnsafePointer[Scalar[Self.T], Self.origin],
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

    # --- Element access ---

    @always_inline
    def __getitem__(self, index: Int) -> Scalar[Self.T]:
        self._check_bounds(index)
        return self._data[index]

    @always_inline
    def __getitem__(self, slc: Slice) -> Self:
        var start: Int
        var end: Int
        var step: Int
        start, end, step = slc.indices(self._length)
        debug_assert(step == 1, "BufferView slice step must be 1")
        return Self(ptr=self._data + start, length=end - start)

    @always_inline
    def __contains__(self, value: Scalar[Self.T]) -> Bool:
        for i in range(self._length):
            if self._data[i] == value:
                return True
        return False

    @always_inline
    def unsafe_get(self, index: Int) -> Scalar[Self.T]:
        return self._data[index]

    @always_inline
    def unsafe_set(
        self: BufferView[mut=True, T=Self.T, origin=_],
        index: Int,
        value: Scalar[Self.T],
    ):
        self._data.store(index, value)

    # --- SIMD ---

    @always_inline
    def load[W: Int](self, index: Int) -> SIMD[Self.T, W]:
        return self._data.load[width=W](index)

    @always_inline
    def store[
        W: Int
    ](
        self: BufferView[mut=True, T=Self.T, origin=_],
        index: Int,
        value: SIMD[Self.T, W],
    ):
        self._data.store(index, value)

    @always_inline
    def gather[W: Int](self, offsets: SIMD[DType.int64, W]) -> SIMD[Self.T, W]:
        """SIMD gather: load W elements at positions given by `offsets`."""
        return self._data.gather[width=W, alignment=1](offsets)

    # --- Slicing ---

    @always_inline
    def slice(self, offset: Int, length: Int = -1) -> Self:
        var actual = length if length >= 0 else self._length - offset
        return Self(ptr=self._data + offset, length=actual)

    # --- Raw pointer access ---

    @always_inline
    def unsafe_ptr(self) -> UnsafePointer[Scalar[Self.T], Self.origin]:
        return self._data

    @always_inline
    def is_non_null(self) -> Bool:
        """Return True if the backing pointer is non-null."""
        return Bool(self._data)

    @always_inline
    def is_aligned(self, alignment: Int) -> Bool:
        """Return True if the backing pointer is aligned to `alignment` bytes.
        """
        return Int(self._data) % alignment == 0

    @always_inline
    def prefetch_at(self, offset: Int):
        """Prefetch the cache line at `offset` elements into L1 cache."""
        prefetch(self._data + offset)

    def copy_from(
        self: BufferView[mut=True, T=Self.T, origin=_],
        src: BufferView[Self.T, _],
        count: Int,
    ):
        """Copy `count` elements from `src` into `self`."""
        memcpy(
            dest=self._data.bitcast[UInt8](),
            src=src._data.bitcast[UInt8](),
            count=count * size_of[Scalar[Self.T]](),
        )

    def to_string_slice(self) -> StringSlice[Self.origin]:
        """Convert this byte view to a StringSlice with origin `self_o`."""
        return StringSlice(ptr=self._data.bitcast[Byte](), length=self._length)

    def copy_from(
        self: BufferView[mut=True, T=DType.uint8, origin=_],
        src: StringSlice[_],
    ):
        """Copy bytes from a StringSlice into this view."""
        memcpy(
            dest=self._data.bitcast[Byte](),
            src=src.unsafe_ptr(),
            count=len(src),
        )

    # --- Vectorized operations ---

    def apply[
        func: def[W: Int](SIMD[Self.T, W]) -> SIMD[Self.T, W]
    ](self: BufferView[mut=True, T=Self.T, origin=_]):
        """Apply a SIMD function in-place over all elements."""
        comptime width = simd_byte_width() // size_of[Scalar[Self.T]]()
        var i = 0
        while i + width <= self._length:
            self._data.store(i, func[width](self._data.load[width=width](i)))
            i += width
        while i < self._length:
            self._data[i] = func[1](self._data[i])
            i += 1

    def count[
        F: def[W: Int](SIMD[Self.T, W]) -> SIMD[DType.bool, W]
    ](self, func: F) -> Int:
        """Count elements matching a vectorized predicate."""
        comptime width = simd_byte_width() // size_of[Scalar[Self.T]]()
        var total = 0
        var i = 0
        while i + width <= self._length:
            total += Int(
                func[width](self._data.load[width=width](i))
                .cast[DType.uint8]()
                .reduce_add()
            )
            i += width
        while i < self._length:
            if func[1](self._data[i]):
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

    var _data: UnsafePointer[UInt8, Self.origin]
    var _offset: Int  # bit offset into _data
    var _length: Int  # number of logical bits

    # --- DevicePassable ---

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return String("BitmapView")

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
        self._length = length

    # --- Sized ---

    @always_inline
    def __len__(self) -> Int:
        return self._length

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

    def unsafe_ptr(self) -> UnsafePointer[UInt8, Self.origin]:
        """Raw byte pointer to the first byte of this view's backing storage.

        Only for use at C FFI boundaries (c_data.mojo). Prefer load/store.
        """
        return self._data

    @always_inline
    def test(self, index: Int) -> Bool:
        """Test if the bit at ``index`` is set."""
        self._check_bounds(index)
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
    def load_bits[T: DType](self, index: Int) -> Scalar[T]:
        """Load ``sizeof[T]*8`` bits starting at logical position ``index``.

        Handles the view's bit offset correctly. Safe because Arrow buffers
        are 64-byte padded.
        """
        var abs_pos = self._offset + index
        var byte_idx = abs_pos >> 3
        var bit_off = abs_pos & 7
        var raw = (
            (self._data + byte_idx).bitcast[Scalar[T]]().load[alignment=1]()
        )
        return raw >> Scalar[T](bit_off)

    @always_inline
    def load[T: DType, W: Int = 1](self, byte_offset: Int) -> SIMD[T, W]:
        """Load W elements of type T from bitmap data at raw ``byte_offset``.

        No ``_offset`` adjustment — the caller is responsible for computing
        the correct byte address. Safe because Arrow buffers are 64-byte padded.
        """
        return (
            (self._data + byte_offset)
            .bitcast[Scalar[T]]()
            .load[width=W, alignment=1]()
        )

    @always_inline
    def store[
        T: DType, W: Int = 1
    ](self: BitmapView[mut=True, origin=_], byte_offset: Int, val: SIMD[T, W]):
        """Store W elements of type T into bitmap data at raw ``byte_offset``.

        No ``_offset`` adjustment — the caller is responsible for computing
        the correct byte address.
        """
        (self._data + byte_offset).bitcast[Scalar[T]]().store[width=W](val)

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
        var bit_end = self._offset + self._length
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

        if self._length == 0:
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
        var end = min(self._length, last_byte * 8 - lead_bits)
        start = (start // 64) * 64
        end = min(self._length, ((end + 63) // 64) * 64)
        return (count, start, end)

    def count_set_bits(self) -> Int:
        """Count set bits in the view."""
        count, _, _ = self.count_set_bits_with_range()
        return count

    # --- Equality ---

    def __eq__(self, other: BitmapView[_]) -> Bool:
        """Return True if both views have identical logical bit patterns."""
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
    def set(self: BitmapView[mut=True, origin=_], index: Int):
        """Set the bit at ``index`` to 1."""
        self._check_bounds(index)
        var abs_index = self._offset + index
        var byte_index = abs_index >> 3
        var bit_mask = UInt8(1 << (abs_index & 7))
        self._data[byte_index] = self._data[byte_index] | bit_mask

    @always_inline
    def clear(self: BitmapView[mut=True, origin=_], index: Int):
        """Set the bit at ``index`` to 0."""
        self._check_bounds(index)
        var abs_index = self._offset + index
        var byte_index = abs_index >> 3
        var bit_mask = UInt8(1 << (abs_index & 7))
        self._data[byte_index] = self._data[byte_index] & ~bit_mask

    @always_inline
    def toggle(self: BitmapView[mut=True, origin=_], index: Int):
        """Invert the bit at ``index``."""
        self._check_bounds(index)
        var abs_index = self._offset + index
        var byte_index = abs_index >> 3
        var bit_mask = UInt8(1 << (abs_index & 7))
        self._data[byte_index] = self._data[byte_index] ^ bit_mask

    # --- Set operations (return Buffer with offset=0) ---

    def intersection(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Return the bitwise AND of self and other."""
        var builder = Bitmap.alloc_uninit(self._length)
        apply[_and](self, other, builder.view())
        return builder^

    def union(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Return the bitwise OR of self and other."""
        var builder = Bitmap.alloc_uninit(self._length)
        apply[_or](self, other, builder.view())
        return builder^

    def symmetric_difference(
        self, other: BitmapView[_]
    ) raises -> Bitmap[mut=True]:
        """Return the bitwise XOR of self and other."""
        var builder = Bitmap.alloc_uninit(self._length)
        apply[_xor](self, other, builder.view())
        return builder^

    def difference(self, other: BitmapView[_]) raises -> Bitmap[mut=True]:
        """Return self AND NOT other."""
        var builder = Bitmap.alloc_uninit(self._length)
        apply[_and_not](self, other, builder.view())
        return builder^

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
        apply[_invert](self, builder.view())
        return builder^

    # --- Writable ---

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            t"BitmapView(offset={self._offset}, length={self._length})"
        )

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# SIMD byte-level binary op helpers
# ---------------------------------------------------------------------------


@always_inline
def _invert[W: Int](x: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return ~x


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


# ---------------------------------------------------------------------------
# apply — free-function overloads for BufferView and BitmapView
# ---------------------------------------------------------------------------



comptime UnaryFn[T: DType] = def[W: Int](SIMD[T, W]) -> SIMD[T, W]
"""A parameterized unary SIMD function type: maps a vector to a vector of the same type."""

comptime BinaryFn[T: DType] = def[W: Int](SIMD[T, W], SIMD[T, W]) -> SIMD[T, W]
"""A parameterized binary SIMD function type: combines two vectors into one of the same type."""



def apply[
    T: DType,
    op: UnaryFn[T],
](
    src: BufferView[T, _],
    dst: BufferView[mut=True, T, _],
    length: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Apply a unary SIMD op element-wise over src into dst, CPU or GPU.

    Pointers are function parameters (not closure captures) so that
    DevicePassable conversion works correctly on GPU offload.
    """

    @parameter
    @always_inline
    def process[W: Int, rank: Int, alignment: Int = 1](
        idx: IndexList[rank]
    ) -> None:
        var i = idx[0]
        dst.store[W](i, op[W](src.load[W](i)))

    if ctx:
        comptime if has_accelerator():
            comptime gpu_width = simd_width_of[T, target = get_gpu_target()]()
            elementwise[process, gpu_width, target="gpu"](length, ctx.value())
        else:
            raise Error("apply: no GPU accelerator available")
    else:
        comptime cpu_width = simd_byte_width() // size_of[Scalar[T]]()
        elementwise[process, cpu_width, target="cpu", use_blocking_impl=True](
            length
        )


def apply[
    T: DType,
    op: BinaryFn[T],
](
    lhs: BufferView[T, _],
    rhs: BufferView[T, _],
    dst: BufferView[mut=True, T, _],
    length: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Apply a binary SIMD op element-wise over lhs,rhs into dst, CPU or GPU.

    Pointers are function parameters (not closure captures) so that
    DevicePassable conversion works correctly on GPU offload.
    """

    @parameter
    @always_inline
    def process[W: Int, rank: Int, alignment: Int = 1](
        idx: IndexList[rank]
    ) -> None:
        var i = idx[0]
        dst.store[W](i, op[W](lhs.load[W](i), rhs.load[W](i)))

    if ctx:
        comptime if has_accelerator():
            comptime gpu_width = simd_width_of[T, target = get_gpu_target()]()
            elementwise[process, gpu_width, target="gpu"](length, ctx.value())
        else:
            raise Error("apply: no GPU accelerator available")
    else:
        comptime cpu_width = simd_byte_width() // size_of[Scalar[T]]()
        elementwise[process, cpu_width, target="cpu", use_blocking_impl=True](
            length
        )


def apply[
    op: UnaryFn[DType.uint8],
](
    src: BitmapView[_],
    dst: BitmapView[mut=True, _],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Apply a byte-level unary SIMD op from src into dst (pre-allocated, offset-0).

    Reads exactly ceil(length/8) source bytes — no over-read.
    Handles sub-byte bit-offset shifting automatically.
    GPU support is not yet implemented; ctx is reserved for future use.
    """
    # TODO: GPU bitmap op
    var byte_start = src._offset >> 3
    var bit_shift = src._offset & 7
    var rshift = UInt8(bit_shift)
    var lshift = UInt8(8 - bit_shift)
    var out_bytes = (src._length + 7) >> 3
    var data = src._data
    comptime cpu_width = simd_width_of[DType.uint8]()

    if out_bytes == 0:
        return

    if bit_shift == 0:

        @parameter
        @always_inline
        def process_zero[W: Int, rank: Int, alignment: Int = 1](
            idx: IndexList[rank]
        ) -> None:
            var i = idx[0]
            dst.store[DType.uint8, W](
                i, op[W]((data + byte_start + i).load[width=W]())
            )

        elementwise[process_zero, cpu_width, target="cpu", use_blocking_impl=True](
            out_bytes
        )
        return

    # Non-zero bit_shift: shift-combine (lo >> rshift | hi << lshift).
    # Bulk covers indices 0 .. out_bytes-2; hi at i+1 is always in bounds.
    var bulk = out_bytes - 1
    if bulk > 0:

        @parameter
        @always_inline
        def process_shifted[W: Int, rank: Int, alignment: Int = 1](
            idx: IndexList[rank]
        ) -> None:
            var i = idx[0]
            var lo = (data + byte_start + i).load[width=W]()
            var hi = (data + byte_start + i + 1).load[width=W]()
            dst.store[DType.uint8, W](
                i, op[W]((lo >> rshift) | (hi << lshift))
            )

        elementwise[
            process_shifted, cpu_width, target="cpu", use_blocking_impl=True
        ](bulk)

    # Last output byte: read hi only when the view's bits span into the next
    # source byte, avoiding a read past the end of source data.
    var last_lo = (data + byte_start + bulk).load[width=1]()
    var last_result = last_lo >> rshift
    var remaining_bits = src._length - bulk * 8
    if remaining_bits > 8 - bit_shift:
        last_result = last_result | (
            (data + byte_start + bulk + 1).load[width=1]() << lshift
        )
    dst.store[DType.uint8, 1](bulk, op[1](last_result))


def apply[
    op: BinaryFn[DType.uint8],
](
    lhs: BitmapView[_],
    rhs: BitmapView[_],
    dst: BitmapView[mut=True, _],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Apply a byte-level binary SIMD op from lhs and rhs into dst (pre-allocated, offset-0).

    Reads exactly ceil(length/8) source bytes per operand — no over-read.
    Handles independent sub-byte bit-offset shifting for each operand.
    GPU support is not yet implemented; ctx is reserved for future use.
    """
    if len(lhs) != len(rhs):
        raise Error("BitmapView lengths must match")

    # TODO: GPU bitmap op
    var byte_start_a = lhs._offset >> 3
    var bit_shift_a = lhs._offset & 7
    var byte_start_b = rhs._offset >> 3
    var bit_shift_b = rhs._offset & 7
    var rs_a = UInt8(bit_shift_a)
    var ls_a = UInt8(8 - bit_shift_a)
    var rs_b = UInt8(bit_shift_b)
    var ls_b = UInt8(8 - bit_shift_b)
    var out_bytes = (lhs._length + 7) >> 3
    var src_a = lhs._data
    var src_b = rhs._data
    comptime cpu_width = simd_width_of[DType.uint8]()

    if out_bytes == 0:
        return

    if bit_shift_a == 0 and bit_shift_b == 0:

        @parameter
        @always_inline
        def process_zero[W: Int, rank: Int, alignment: Int = 1](
            idx: IndexList[rank]
        ) -> None:
            var i = idx[0]
            dst.store[DType.uint8, W](
                i,
                op[W](
                    (src_a + byte_start_a + i).load[width=W](),
                    (src_b + byte_start_b + i).load[width=W](),
                ),
            )

        elementwise[process_zero, cpu_width, target="cpu", use_blocking_impl=True](
            out_bytes
        )
        return

    # At least one non-zero shift: shift-combine both operands.
    # When a shift is 0, ls = 8 so hi << 8 == 0, giving lo unchanged.
    # Bulk covers indices 0 .. out_bytes-2; hi at i+1 is always in bounds.
    var bulk = out_bytes - 1
    if bulk > 0:

        @parameter
        @always_inline
        def process_shifted[W: Int, rank: Int, alignment: Int = 1](
            idx: IndexList[rank]
        ) -> None:
            var i = idx[0]
            var lo_a = (src_a + byte_start_a + i).load[width=W]()
            var hi_a = (src_a + byte_start_a + i + 1).load[width=W]()
            var lo_b = (src_b + byte_start_b + i).load[width=W]()
            var hi_b = (src_b + byte_start_b + i + 1).load[width=W]()
            dst.store[DType.uint8, W](
                i,
                op[W](
                    (lo_a >> rs_a) | (hi_a << ls_a),
                    (lo_b >> rs_b) | (hi_b << ls_b),
                ),
            )

        elementwise[
            process_shifted, cpu_width, target="cpu", use_blocking_impl=True
        ](bulk)

    # Last output byte: read hi only when bits span into the next source byte.
    var remaining_bits = lhs._length - bulk * 8
    var last_lo_a = (src_a + byte_start_a + bulk).load[width=1]()
    var last_lo_b = (src_b + byte_start_b + bulk).load[width=1]()
    var result_a = last_lo_a >> rs_a
    var result_b = last_lo_b >> rs_b
    if remaining_bits > 8 - bit_shift_a:
        result_a = result_a | (
            (src_a + byte_start_a + bulk + 1).load[width=1]() << ls_a
        )
    if remaining_bits > 8 - bit_shift_b:
        result_b = result_b | (
            (src_b + byte_start_b + bulk + 1).load[width=1]() << ls_b
        )
    dst.store[DType.uint8, 1](bulk, op[1](result_a, result_b))
