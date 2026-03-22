"""Filter and selection kernels.

`filter` selects elements from an array based on a boolean selection array,
producing a new array containing only the selected elements.


All functions support arrays with non-zero offsets (sliced arrays).
"""

import std.math as math
from std.bit import count_trailing_zeros, pop_count
from std.memory import memcpy
from std.sys import size_of

from ..arrays import PrimitiveArray, StringArray, AnyArray
from ..buffers import Buffer, BufferBuilder
from ..bitmap import Bitmap, BitmapBuilder
from ..dtypes import DataType, bool_, uint32, string, numeric_dtypes
from .aggregate import sum_
from .string import string_lengths


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@always_inline
def _filter_sparse[
    T: DType
](
    dst: UnsafePointer[Scalar[T], MutExternalOrigin],
    out_pos: Int,
    src: UnsafePointer[Scalar[T], ImmutExternalOrigin],
    base: Int,
    sel_word: UInt64,
):
    """Sparse filter: CTZ scatter, O(popcount).

    Touches only the set-bit positions via count-trailing-zeros loop.
    Best when few bits are set (low popcount).
    """
    var w = sel_word
    var out = dst + out_pos
    var inp = src + base
    while w != 0:
        out[0] = inp[Int(count_trailing_zeros(w))]
        w &= w - 1
        out = out + 1


@always_inline
def _filter_dense[
    T: DType
](
    dst: UnsafePointer[Scalar[T], MutExternalOrigin],
    out_pos: Int,
    src: UnsafePointer[Scalar[T], ImmutExternalOrigin],
    base: Int,
    sel_word: UInt64,
):
    """Dense filter: byte-chunked branchless scatter.

    Processes the 64-bit mask one byte at a time. Precomputes per-byte
    popcount prefix sums so each of the 8 chunks writes to an independent
    output region. This breaks the 64-deep serial dependency on the output
    pointer into 8 independent chains of depth 8 that the OoO engine can
    overlap (~32 cycles vs ~128 for the naive approach).
    """
    var inp = src + base
    var offset = out_pos

    comptime for i in range(8):
        var byte = (sel_word >> UInt64(i * 8)) & 0xFF
        var out = dst + offset
        var chunk_inp = inp + i * 8
        var b = byte
        var k = 0
        comptime for bit in range(8):
            out[k] = chunk_inp[bit]
            k += Int(b & 1)
            b >>= 1
        offset += Int(pop_count(byte))


# @always_inline
@no_inline
def _filter_block[
    T: DType,
    SPARSE_THRESHOLD: Int = 24,
](
    dst: UnsafePointer[Scalar[T], MutExternalOrigin],
    out_pos: Int,
    src: UnsafePointer[Scalar[T], ImmutExternalOrigin],
    base: Int,
    sel_word: UInt64,
) -> Int:
    """Filter a 64-element block using density-adaptive dispatch.

    Dispatches to one of two strategies based on the selection word:
      - Sparse (≤SPARSE_THRESHOLD bits set): CTZ scatter, O(popcount).
      - Dense (>SPARSE_THRESHOLD bits set): byte-chunked branchless, O(64).

    The caller handles the all-ones and all-zeros fast paths before calling.

    Parameters:
        T: Element dtype.
        SPARSE_THRESHOLD: popcount cutoff; at or below → sparse, above → dense.

    Returns:
        Number of elements written (popcount of sel_word).
    """
    var cnt = Int(pop_count(sel_word))
    if cnt <= SPARSE_THRESHOLD:
        _filter_sparse[T](dst, out_pos, src, base, sel_word)
    else:
        _filter_dense[T](dst, out_pos, src, base, sel_word)
    return cnt


@always_inline
def _pext(val: UInt64, mask: UInt64) -> UInt64:
    """Parallel bit extract: keep bits from `val` where `mask` is set, packed
    to LSB.  Runs in O(popcount(mask)) iterations via CTZ loop.
    """
    var result = UInt64(0)
    var m = mask
    var k = UInt64(0)
    while m != 0:
        var bit_pos = UInt64(count_trailing_zeros(m))
        result |= ((val >> bit_pos) & 1) << k
        k += 1
        m &= m - 1  # clear lowest set bit
    return result


# ---------------------------------------------------------------------------
# filter — bitmap / values helpers
# ---------------------------------------------------------------------------


def _filter_bits(
    src: Bitmap,
    sel: Bitmap,
    sel_start: Int,
    sel_end: Int,
    out_len: Int,
) -> Tuple[Bitmap, Int]:
    """Filter a bitmap, keeping bits where selection is set.

    Uses pext + deposit_bits in 64-bit blocks with run-merge for all-ones
    and all-zeros blocks.  Works for both validity bitmaps and bool data.

    Args:
        src: Source bitmap to filter.
        sel: Selection bitmap (True = keep).
        sel_start: First 64-bit-aligned block with set bits in sel.
        sel_end: Past-the-end 64-bit-aligned block in sel.
        out_len: Pre-counted number of set bits in sel.

    Returns:
        (filtered_bitmap, zero_bit_count) where zero_bit_count is the number
        of zero bits in the filtered output (null count when filtering
        validity bitmaps).
    """
    comptime ALL_ONES = ~UInt64(0)
    var builder = BitmapBuilder.alloc(out_len)
    var bm_pos = 0
    var zero_count = 0
    var i = sel_start

    while i + 64 <= sel_end:
        var sel_word = sel.load_word(i)
        if sel_word == 0:
            i += 64
            while i + 64 <= sel_end and sel.load_word(i) == 0:
                i += 64
            continue
        if sel_word == ALL_ONES:
            var run_start = i
            i += 64
            while i + 64 <= sel_end and sel.load_word(i) == ALL_ONES:
                i += 64
            var j = run_start
            while j < i:
                var src_word = src.load_word(j)
                builder.deposit_bits(bm_pos, src_word, 64)
                zero_count += 64 - Int(pop_count(src_word))
                bm_pos += 64
                j += 64
            continue

        # Mixed block: pext + deposit.
        var src_word = src.load_word(i)
        var count = Int(pop_count(sel_word))
        var compressed = _pext(src_word, sel_word)
        builder.deposit_bits(bm_pos, compressed, count)
        zero_count += count - Int(pop_count(compressed))
        bm_pos += count
        i += 64

    # Tail: masked pext + deposit.
    if i < sel_end:
        var tail = sel_end - i
        var mask = (UInt64(1) << UInt64(tail)) - 1
        var sel_word = sel.load_word(i) & mask
        if sel_word != 0:
            var src_word = src.load_word(i)
            var count = Int(pop_count(sel_word))
            var compressed = _pext(src_word, sel_word)
            builder.deposit_bits(bm_pos, compressed, count)
            zero_count += count - Int(pop_count(compressed))

    return builder.finish(out_len), zero_count


def _filter_values[
    T: DType
](
    src_buf: Buffer,
    src_offset: Int,
    sel: Bitmap,
    sel_start: Int,
    sel_end: Int,
    out_len: Int,
) -> Buffer:
    """Filter fixed-width values, keeping elements where selection is set.

    Uses run-merge for all-ones blocks (memcpy) and density-adaptive
    block dispatch for mixed blocks.

    Args:
        src_buf: Source data buffer.
        src_offset: Element offset into src_buf.
        sel: Selection bitmap (True = keep).
        sel_start: First 64-bit-aligned block with set bits in sel.
        sel_end: Past-the-end 64-bit-aligned block in sel.
        out_len: Pre-counted number of set bits in sel.

    Returns:
        A new Buffer containing only the selected elements.
    """
    comptime ELEM = size_of[Scalar[T]]()
    comptime ALL_ONES = ~UInt64(0)
    var buf = BufferBuilder.alloc_uninit(out_len * ELEM)
    var src = src_buf.unsafe_ptr[T](src_offset)
    var dst = buf.unsafe_ptr[T]()
    var out_pos = 0
    var i = sel_start

    while i + 64 <= sel_end:
        var sel_word = sel.load_word(i)
        if sel_word == 0:
            i += 64
            while i + 64 <= sel_end and sel.load_word(i) == 0:
                i += 64
            continue
        if sel_word == ALL_ONES:
            var run_start = i
            i += 64
            while i + 64 <= sel_end and sel.load_word(i) == ALL_ONES:
                i += 64
            memcpy(
                dest=(dst + out_pos).bitcast[UInt8](),
                src=(src + run_start).bitcast[UInt8](),
                count=(i - run_start) * ELEM,
            )
            out_pos += i - run_start
            continue
        out_pos += _filter_block[T](dst, out_pos, src, i, sel_word)
        i += 64

    # Tail: partial block — force sparse (only reads set-bit positions).
    if i < sel_end:
        var tail = sel_end - i
        var mask = (UInt64(1) << UInt64(tail)) - 1
        var sel_word = sel.load_word(i) & mask
        if sel_word != 0:
            out_pos += _filter_block[T, SPARSE_THRESHOLD=64](
                dst, out_pos, src, i, sel_word
            )

    return buf.finish()


# ---------------------------------------------------------------------------
# filter — primitive arrays
# ---------------------------------------------------------------------------


def filter_[
    T: DataType
](
    array: PrimitiveArray[T], selection: PrimitiveArray[bool_]
) raises -> PrimitiveArray[T]:
    """Filter a primitive array, keeping only elements where selection is True.

    Args:
        array: The input primitive array.
        selection: Boolean selection mask (True = keep).

    Returns:
        A new PrimitiveArray containing only the selected elements.
    """
    var n = len(array)
    if n != len(selection):
        raise Error(
            t"filter: array length {n} != selection length {len(selection)}"
        )

    var sel_bm = Bitmap(selection.buffer, selection.offset, n)
    var out_len, sel_start, sel_end = sel_bm.count_set_bits_with_range()

    if out_len == 0:
        var empty_buf = BufferBuilder.alloc[T.native](0)
        return PrimitiveArray[T](
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=empty_buf.finish(),
        )

    # Filter validity bitmap (shared by both paths).
    var bm: Optional[Bitmap] = None
    var null_count = 0
    if array.bitmap:
        var val_bm = array.bitmap.value().slice(array.offset, n)
        var filtered_bm, nc = _filter_bits(
            val_bm, sel_bm, sel_start, sel_end, out_len
        )
        bm = filtered_bm
        null_count = nc

    # Filter data and return.
    comptime if T == bool_:
        var data_bm = Bitmap(array.buffer, array.offset, n)
        var filtered_data, _ = _filter_bits(
            data_bm, sel_bm, sel_start, sel_end, out_len
        )
        return PrimitiveArray[T](
            length=out_len,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            buffer=filtered_data._buffer,
        )
    else:
        var result_buf = _filter_values[T.native](
            array.buffer,
            array.offset,
            sel_bm,
            sel_start,
            sel_end,
            out_len,
        )
        return PrimitiveArray[T](
            length=out_len,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            buffer=result_buf,
        )


# ---------------------------------------------------------------------------
# filter — string arrays
# ---------------------------------------------------------------------------


def filter_(
    array: StringArray, selection: PrimitiveArray[bool_]
) raises -> StringArray:
    """Filter a string array, keeping only elements where selection is True.

    Uses run merging: consecutive selected elements are copied with a single
    memcpy call to reduce per-element overhead.  When the source has a
    validity bitmap, it is filtered in the same pass (no second traversal).

    Args:
        array: The input string array.
        selection: Boolean selection mask (True = keep).

    Returns:
        A new StringArray containing only the selected strings.
    """
    var n = len(array)
    if n != len(selection):
        raise Error(
            t"filter: array length {n} != selection length {len(selection)}"
        )

    var sel_bm = Bitmap(selection.buffer, selection.offset, n)
    var out_len = sel_bm.count_set_bits()

    if out_len == 0:
        var empty_offsets = BufferBuilder.alloc[DType.uint32](1)
        var empty_values = BufferBuilder.alloc[DType.uint8](0)
        return StringArray(
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            offsets=empty_offsets.finish(),
            values=empty_values.finish(),
        )

    var off = array.offset
    var offsets_ptr = array.offsets.unsafe_ptr[DType.uint32]()
    var values_ptr = array.values.unsafe_ptr()

    # Compute total output bytes.
    var total_bytes = 0
    for i in range(n):
        if sel_bm.is_valid(i):
            total_bytes += Int(offsets_ptr[off + i + 1]) - Int(
                offsets_ptr[off + i]
            )

    # Allocate output buffers.
    # TODO: use alloc_uninit to spare zeroing the output buffers
    var out_offsets = BufferBuilder.alloc[DType.uint32](out_len + 1)
    var out_values = BufferBuilder.alloc[DType.uint8](total_bytes)
    var out_off_ptr = out_offsets.unsafe_ptr[DType.uint32]()
    var out_val_ptr = out_values.unsafe_ptr[DType.uint8]()
    var bm: Optional[Bitmap] = None
    var null_count = 0

    if array.bitmap:
        # --- With bitmap: fused run-merging + bitmap filtering ---
        var src_bm = array.bitmap.value()
        var bm_builder = BitmapBuilder.alloc(out_len)
        var byte_pos = UInt32(0)
        out_off_ptr[0] = 0
        var j = 0
        var i = 0

        while i < n:
            if not sel_bm.is_valid(i):
                i += 1
                continue

            var run_start = i
            while i < n and sel_bm.is_valid(i):
                var elem_start = offsets_ptr[off + i]
                var elem_end = offsets_ptr[off + i + 1]
                byte_pos += elem_end - elem_start
                var valid = src_bm.is_valid(off + i)
                bm_builder.set_bit(j, valid)
                if not valid:
                    null_count += 1
                j += 1
                out_off_ptr[j] = byte_pos
                i += 1

            var src_byte_start = Int(offsets_ptr[off + run_start])
            var src_byte_end = Int(offsets_ptr[off + i - 1 + 1])
            var run_bytes = src_byte_end - src_byte_start
            if run_bytes > 0:
                memcpy(
                    dest=out_val_ptr + Int(byte_pos) - run_bytes,
                    src=values_ptr + src_byte_start,
                    count=run_bytes,
                )

        bm = bm_builder.finish(out_len)

    else:
        # --- No bitmap: run-merging only ---
        var byte_pos = UInt32(0)
        out_off_ptr[0] = 0
        var j = 0
        var i = 0

        while i < n:
            if not sel_bm.is_valid(i):
                i += 1
                continue

            var run_start = i
            while i < n and sel_bm.is_valid(i):
                var elem_start = offsets_ptr[off + i]
                var elem_end = offsets_ptr[off + i + 1]
                byte_pos += elem_end - elem_start
                j += 1
                out_off_ptr[j] = byte_pos
                i += 1

            var src_byte_start = Int(offsets_ptr[off + run_start])
            var src_byte_end = Int(offsets_ptr[off + i - 1 + 1])
            var run_bytes = src_byte_end - src_byte_start
            if run_bytes > 0:
                memcpy(
                    dest=out_val_ptr + Int(byte_pos) - run_bytes,
                    src=values_ptr + src_byte_start,
                    count=run_bytes,
                )

    return StringArray(
        length=out_len,
        nulls=null_count,
        offset=0,
        bitmap=bm,
        offsets=out_offsets.finish(),
        values=out_values.finish(),
    )


# ---------------------------------------------------------------------------
# filter — runtime-typed dispatch
# ---------------------------------------------------------------------------


def filter_(array: AnyArray, selection: AnyArray) raises -> AnyArray:
    """Runtime-typed filter: dispatches to the correct typed overload.

    Args:
        array: The input array (runtime-typed).
        selection: Bit-packed boolean selection (True = keep).

    Returns:
        A new AnyArray with only the selected elements.
    """
    var mask = selection.as_bool()

    if array.dtype == bool_:
        return AnyArray(filter_[bool_](PrimitiveArray[bool_](data=array), mask))

    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return AnyArray(filter_[dtype](array.as_primitive[dtype](), mask))

    if array.dtype.is_string():
        return AnyArray(filter_(array.as_string(), mask))

    raise Error("filter: unsupported dtype ", array.dtype)


# ---------------------------------------------------------------------------
# drop_nulls — reimplemented via filter
# ---------------------------------------------------------------------------


def drop_nulls[
    T: DataType
](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Create a new array containing only the valid (non-null) elements.

    Uses the array's validity bitmap directly as the filter selection.

    Args:
        array: The input array.

    Returns:
        A new PrimitiveArray containing only valid elements.
    """
    if not array.bitmap:
        # All valid: wrap as identity selection
        var all_true = BitmapBuilder.alloc(len(array))
        all_true.set_range(0, len(array), True)
        var selection = PrimitiveArray[bool_](
            length=len(array),
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=all_true._builder.finish(),
        )
        return filter_[T](array, selection)
    var bm = array.bitmap.value()
    var selection = PrimitiveArray[bool_](
        length=len(array),
        nulls=0,
        offset=bm.bit_offset() + array.offset,
        bitmap=None,
        buffer=bm._buffer,
    )
    return filter_[T](array, selection)


def drop_nulls(array: AnyArray) raises -> AnyArray:
    """Runtime-typed drop_nulls: dispatches to the correct typed version.

    Args:
        array: The input array (runtime-typed).

    Returns:
        A new AnyArray with null elements removed.
    """
    if array.dtype == bool_:
        return AnyArray(drop_nulls[bool_](PrimitiveArray[bool_](data=array)))

    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return AnyArray(
                drop_nulls[dtype](PrimitiveArray[dtype](data=array))
            )

    raise Error("drop_nulls: unsupported dtype ", array.dtype)
