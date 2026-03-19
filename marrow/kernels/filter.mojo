"""Filter and selection kernels.

`filter` selects elements from an array based on a boolean selection array,
producing a new array containing only the selected elements.

Design
------
Selection type is `PrimitiveArray[bool_]` (Arrow bit-packed boolean).  The
selection bitmap is read through the offset-aware `Bitmap` abstraction.

Primitive filter
~~~~~~~~~~~~~~~~
Processes the selection bitmap in 64-bit words (8x fewer outer-loop iterations
than byte-by-byte).  Strategy is chosen upfront based on selectivity
(out_len / n), matching Arrow's approach from
https://dl.acm.org/doi/abs/10.1145/3465998.3466009 :

  selectivity > 80% -> SlicesIterator
    Finds contiguous runs of 1-bits using trailing_zeros + trailing_ones on
    each 64-bit word, then bulk-copies each run with `memcpy`.  Avoids
    per-element scatter overhead for dense filters.

  selectivity <= 80% -> IndexIterator
    Iterates only the set bits using trailing_zeros + clear-lowest-bit on each
    64-bit word, then scatters elements one at a time.

  all selected (out_len == n) -> single memcpy of data and validity buffers.

The output validity bitmap is built using `BitmapBuilder` operations
(`set_bit`, `set_range`) and source validity is read via the offset-aware
`PrimitiveArray.is_valid()`.

String filter composition
~~~~~~~~~~~~~~~~~~~~~~~~~
  1. `string_lengths(array)` -> `PrimitiveArray[uint32]` of per-element byte lengths.
  2. `filter_[uint32](lengths, selection)` -> selected lengths.
  3. `sum[uint32](sel_lengths)` -> total output byte size.
  4. Output validity built directly via `BitmapBuilder`.
  5. Prefix-sum of sel_lengths -> output offsets buffer.
  6. Run-merging copy: consecutive selected source ranges are merged into a
     single `memcpy` for efficiency.

All functions support arrays with non-zero offsets (sliced arrays).
"""

import std.math as math
from std.bit import count_trailing_zeros
from std.memory import memcpy
from std.sys import size_of

from ..arrays import PrimitiveArray, StringArray, Array
from ..buffers import Buffer, BufferBuilder
from ..bitmap import Bitmap, BitmapBuilder
from ..dtypes import DataType, bool_, uint32, string, numeric_dtypes
from .aggregate import sum_
from .string import string_lengths


# ---------------------------------------------------------------------------
# filter — primitive arrays
# ---------------------------------------------------------------------------


def filter_[
    T: DataType
](
    array: PrimitiveArray[T], selection: PrimitiveArray[bool_]
) raises -> PrimitiveArray[T]:
    """Filter a primitive array, keeping only elements where selection is True.

    Supports arrays with non-zero offsets (sliced arrays).  When the selection
    offset is byte-aligned (common case), the optimized word-based iteration
    is used.  Otherwise falls back to element-by-element using offset-aware
    accessors.

    Parameters:
        T: Element DataType.

    Args:
        array: The input array.
        selection: Bit-packed boolean selection (True = keep).

    Returns:
        A new PrimitiveArray containing only the selected elements.
    """
    if len(array) != len(selection):
        raise Error(
            t"filter: array length {len(array)} != selection length"
            t" {len(selection)}"
        )
    var n = len(array)
    var sel_bm = Bitmap(selection.buffer, selection.offset, n)
    var out_len = sel_bm.count_set_bits()

    comptime native = T.native
    comptime elem_size = size_of[Scalar[native]]()
    var buf = BufferBuilder.alloc[native](out_len)
    var bm = BitmapBuilder.alloc(out_len)
    var arr_off = array.offset
    var has_nulls = array.nulls > 0

    if out_len == n:
        # All selected: bulk copy of data and validity.
        comptime if native == DType.bool:
            for i in range(n):
                buf.unsafe_set[DType.bool](
                    i,
                    rebind[Scalar[DType.bool]](array.unsafe_get(i)),
                )
        else:
            memcpy(
                dest=buf.ptr,
                src=array.buffer.unsafe_ptr() + arr_off * elem_size,
                count=n * elem_size,
            )
        if has_nulls:
            bm.copy_bits(
                array.bitmap.value()._buffer.unsafe_ptr(),
                array.offset,
                0,
                n,
            )
        else:
            bm.set_range(0, n, True)
    elif out_len > 0 and selection.offset & 7 == 0:
        # Byte-aligned selection: optimized word-based iteration.
        var sel_byte_off = selection.offset >> 3
        var sel_ptr = selection.buffer.unsafe_ptr() + sel_byte_off
        var word_ptr = sel_ptr.bitcast[UInt64]()
        var word_count = math.ceildiv(n, 64)
        var out_idx = 0

        if out_len * 10 > n * 8:
            # Selectivity > 80%: SlicesIterator — find contiguous 1-bit runs
            # and bulk-copy each run.
            for word_i in range(word_count):
                var word = word_ptr[word_i]
                if word == 0:
                    continue
                var base = word_i * 64
                while word != 0:
                    var start_bit = Int(count_trailing_zeros(word))
                    word |= (UInt64(1) << UInt64(start_bit)) - 1
                    var end_bit = Int(count_trailing_zeros(~word))
                    if end_bit < 64:
                        word &= ~((UInt64(1) << UInt64(end_bit)) - 1)
                    else:
                        word = 0
                    var run_start = base + start_bit
                    var run_end = min(base + end_bit, n)
                    var run_len = run_end - run_start
                    # Bulk-copy data for this run.
                    comptime if native == DType.bool:
                        for i in range(run_len):
                            buf.unsafe_set[DType.bool](
                                out_idx + i,
                                rebind[Scalar[DType.bool]](
                                    array.unsafe_get(run_start + i)
                                ),
                            )
                    else:
                        memcpy(
                            dest=buf.ptr + out_idx * elem_size,
                            src=array.buffer.unsafe_ptr()
                            + (arr_off + run_start) * elem_size,
                            count=run_len * elem_size,
                        )
                    # Copy validity for this run.
                    if has_nulls:
                        bm.copy_bits(
                            array.bitmap.value()._buffer.unsafe_ptr(),
                            arr_off + run_start,
                            out_idx,
                            run_len,
                        )
                    else:
                        bm.set_range(out_idx, run_len, True)
                    out_idx += run_len
        else:
            # Selectivity <= 80%: IndexIterator — scatter individual elements.
            if has_nulls:
                for word_i in range(word_count):
                    var word = word_ptr[word_i]
                    if word == 0:
                        continue
                    var base = word_i * 64
                    while word != 0:
                        var bit = Int(count_trailing_zeros(word))
                        var elem_i = base + bit
                        if elem_i >= n:
                            break
                        buf.unsafe_set[native](
                            out_idx, array.unsafe_get(elem_i)
                        )
                        bm.set_bit(out_idx, array.is_valid(elem_i))
                        out_idx += 1
                        word &= word - 1
            else:
                for word_i in range(word_count):
                    var word = word_ptr[word_i]
                    if word == 0:
                        continue
                    var base = word_i * 64
                    while word != 0:
                        var bit = Int(count_trailing_zeros(word))
                        var elem_i = base + bit
                        if elem_i >= n:
                            break
                        buf.unsafe_set[native](
                            out_idx, array.unsafe_get(elem_i)
                        )
                        out_idx += 1
                        word &= word - 1
                bm.set_range(0, out_len, True)
    elif out_len > 0:
        # Non-byte-aligned selection offset: element-by-element fallback.
        var out_idx = 0
        if has_nulls:
            for i in range(n):
                if sel_bm.is_valid(i):
                    buf.unsafe_set[native](out_idx, array.unsafe_get(i))
                    bm.set_bit(out_idx, array.is_valid(i))
                    out_idx += 1
        else:
            for i in range(n):
                if sel_bm.is_valid(i):
                    buf.unsafe_set[native](out_idx, array.unsafe_get(i))
                    out_idx += 1
            bm.set_range(0, out_len, True)

    var bm_bitmap = bm.finish(out_len)
    var ones = bm_bitmap.count_set_bits()
    var null_count = out_len - ones
    return PrimitiveArray[T](
        length=out_len,
        nulls=null_count,
        offset=0,
        bitmap=Optional[Bitmap](bm_bitmap) if null_count > 0 else None,
        buffer=buf.finish(),
    )


# ---------------------------------------------------------------------------
# filter — string arrays
# ---------------------------------------------------------------------------


def filter_(
    array: StringArray, selection: PrimitiveArray[bool_]
) raises -> StringArray:
    """Filter a StringArray, keeping only elements where selection is True.

    Supports arrays with non-zero offsets (sliced arrays).

    Composes:
      1. `string_lengths` to get per-element byte counts.
      2. `filter_[uint32]` to select the relevant lengths.
      3. `sum[uint32]` to compute the total output byte size.
      4. Output validity built directly via `BitmapBuilder`.
      5. Prefix sum of selected lengths to build the output offsets buffer.
      6. Run-merging `memcpy` to copy selected string data.

    Args:
        array: The input string array.
        selection: Bit-packed boolean selection (True = keep).

    Returns:
        A new StringArray containing only the selected elements.
    """
    if len(array) != len(selection):
        raise Error(
            t"filter: array length {len(array)} != selection length"
            t" {len(selection)}"
        )

    var n = len(array)
    var sel_bm = Bitmap(selection.buffer, selection.offset, n)
    var arr_off = array.offset

    # Step 1+2: compute and filter per-element lengths
    var lengths = string_lengths(array)
    var sel_lengths = filter_[uint32](lengths, selection)

    # Step 3: total output byte size
    var total_bytes = Int(sum_[uint32](sel_lengths))

    var out_len = len(sel_lengths)

    # Step 4: build output validity directly via BitmapBuilder
    var out_bm = BitmapBuilder.alloc(out_len)
    if array.nulls > 0:
        var out_idx = 0
        for i in range(n):
            if sel_bm.is_valid(i):
                out_bm.set_bit(out_idx, array.is_valid(i))
                out_idx += 1
    else:
        out_bm.set_range(0, out_len, True)

    # Step 5: build output offsets via prefix sum of sel_lengths
    var offsets_buf = BufferBuilder.alloc[DType.uint32](out_len + 1)
    offsets_buf.unsafe_set[DType.uint32](0, UInt32(0))
    var running = UInt32(0)
    for i in range(out_len):
        running = running + UInt32(sel_lengths.unsafe_get(i))
        offsets_buf.unsafe_set[DType.uint32](i + 1, running)

    # Step 6: copy selected string data with run-merging
    var values_buf = BufferBuilder.alloc[DType.uint8](max(total_bytes, 1))
    var src_ptr = array.values.unsafe_ptr()
    var dst_ptr = values_buf.unsafe_ptr()

    var run_src_start = -1
    var run_src_end = -1
    var dst_offset = 0

    for i in range(n):
        if sel_bm.is_valid(i):
            var src_start = Int(
                array.offsets.unsafe_get[DType.uint32](arr_off + i)
            )
            var src_end = Int(
                array.offsets.unsafe_get[DType.uint32](arr_off + i + 1)
            )
            if run_src_start == -1:
                run_src_start = src_start
                run_src_end = src_end
            elif src_start == run_src_end:
                run_src_end = src_end
            else:
                var run_len = run_src_end - run_src_start
                memcpy(
                    dest=dst_ptr + dst_offset,
                    src=src_ptr + run_src_start,
                    count=run_len,
                )
                dst_offset += run_len
                run_src_start = src_start
                run_src_end = src_end
        else:
            if run_src_start != -1:
                var run_len = run_src_end - run_src_start
                memcpy(
                    dest=dst_ptr + dst_offset,
                    src=src_ptr + run_src_start,
                    count=run_len,
                )
                dst_offset += run_len
                run_src_start = -1
                run_src_end = -1

    # Flush final run
    if run_src_start != -1:
        var run_len = run_src_end - run_src_start
        memcpy(
            dest=dst_ptr + dst_offset,
            src=src_ptr + run_src_start,
            count=run_len,
        )

    var validity_bm = out_bm.finish(out_len)
    var out_valid_count = validity_bm.count_set_bits()
    var null_count = out_len - out_valid_count
    return StringArray(
        length=out_len,
        nulls=null_count,
        offset=0,
        bitmap=Optional[Bitmap](validity_bm) if null_count > 0 else None,
        offsets=offsets_buf.finish(),
        values=values_buf.finish(),
    )


# ---------------------------------------------------------------------------
# filter — runtime-typed dispatch
# ---------------------------------------------------------------------------


def filter_(array: Array, selection: Array) raises -> Array:
    """Runtime-typed filter: dispatches to the correct typed overload.

    Args:
        array: The input array (runtime-typed).
        selection: Bit-packed boolean selection (True = keep).

    Returns:
        A new Array with only the selected elements.
    """
    var mask = selection.as_bool()

    if array.dtype == bool_:
        return Array(filter_[bool_](PrimitiveArray[bool_](data=array), mask))

    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return Array(filter_[dtype](array.as_primitive[dtype](), mask))

    if array.dtype.is_string():
        return Array(filter_(array.as_string(), mask))

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


def drop_nulls(array: Array) raises -> Array:
    """Runtime-typed drop_nulls: dispatches to the correct typed version.

    Args:
        array: The input array (runtime-typed).

    Returns:
        A new Array with null elements removed.
    """
    if array.dtype == bool_:
        return Array(drop_nulls[bool_](PrimitiveArray[bool_](data=array)))

    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return Array(drop_nulls[dtype](PrimitiveArray[dtype](data=array)))

    raise Error("drop_nulls: unsupported dtype ", array.dtype)
