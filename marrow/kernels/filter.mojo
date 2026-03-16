"""Filter and selection kernels.

`filter` selects elements from an array based on a boolean selection array,
producing a new array containing only the selected elements.

Design
------
Selection type is `PrimitiveArray[bool_]` (Arrow bit-packed boolean).  Bit j of
byte i of `selection.buffer` corresponds to element `8*i + j`.

Primitive filter
~~~~~~~~~~~~~~~~
Processes the selection bitmap in 64-bit words (8× fewer outer-loop iterations
than byte-by-byte).  Strategy is chosen upfront based on selectivity
(out_len / n), matching Arrow's approach from
https://dl.acm.org/doi/abs/10.1145/3465998.3466009 :

  selectivity > 80% → SlicesIterator
    Finds contiguous runs of 1-bits using trailing_zeros + trailing_ones on
    each 64-bit word, then bulk-copies each run with `memcpy`.  Avoids
    per-element scatter overhead for dense filters.

  selectivity ≤ 80% → IndexIterator
    Iterates only the set bits using trailing_zeros + clear-lowest-bit on each
    64-bit word, then scatters elements one at a time.

  all selected (out_len == n) → single memcpy of data and validity buffers.

The output validity bitmap is built alongside the data: bulk `bitmap_range_set`
in the slice path (all-valid runs when the source has no nulls), bit-by-bit
copy in the index path.

String filter composition
~~~~~~~~~~~~~~~~~~~~~~~~~
  1. `_string_lengths(array)`  → `PrimitiveArray[uint32]` of per-element byte lengths.
  2. `filter_[uint32](lengths, selection)` → selected lengths.
  3. `sum[uint32](sel_lengths)` → total output byte size.
  4. `filter_[bool_](validity_as_bool, selection)` → output validity array.
  5. Prefix-sum of sel_lengths → output offsets buffer.
  6. Run-merging copy: consecutive selected source ranges are merged into a
     single `memcpy` for efficiency.

`drop_nulls` is reimplemented as `filter(array, _bitmap_as_bool_array(bitmap, n))`.

Assumes `array.offset == 0` throughout (no slice support yet).
"""

import std.math as math
from std.bit import count_trailing_zeros
from std.memory import memcpy

from ..arrays import PrimitiveArray, StringArray, Array
from ..buffers import Buffer, BufferBuilder
from ..bitmap import Bitmap, BitmapBuilder
from ..dtypes import DataType, bool_, uint32, string, numeric_dtypes
from .boolean import count_true
from .aggregate import sum_


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


fn _bitmap_as_bool_array(
    bitmap: Optional[Bitmap], length: Int
) -> PrimitiveArray[bool_]:
    """Wrap a validity bitmap as a BoolArray (for use as a filter selection).

    The bitmap buffer becomes the `buffer` field (bit-packed data), and the
    wrapper's own validity is all-valid (None).

    Args:
        bitmap: A validity bitmap (None = all valid).
        length: Number of elements represented by the bitmap.

    Returns:
        A PrimitiveArray[bool_] where element i is True iff bit i of `bitmap` is set.
    """
    var data: Buffer
    if not bitmap:
        # All valid: build an all-True data buffer so the selection is all-selected.
        var bm = BitmapBuilder.alloc(length)
        bm.set_range(0, length, True)
        data = bm._builder.finish()
    else:
        data = bitmap.value()._buffer
    return PrimitiveArray[bool_](
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,  # wrapper is all-valid
        buffer=data,
    )


fn _string_lengths(array: StringArray) -> PrimitiveArray[uint32]:
    """Compute per-element byte lengths of a StringArray.

    For element i: length = offsets[i+1] - offsets[i].

    Args:
        array: The input string array (offset == 0 assumed).

    Returns:
        A PrimitiveArray[uint32] of byte lengths with a fully-valid bitmap.
    """
    var n = len(array)
    var buf = BufferBuilder.alloc[DType.uint32](n)
    for i in range(n):
        var start = array.offsets.unsafe_get[DType.uint32](i)
        var end = array.offsets.unsafe_get[DType.uint32](i + 1)
        buf.unsafe_set[DType.uint32](i, end - start)
    return PrimitiveArray[uint32](
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,  # all valid
        buffer=buf.finish(),
    )


# ---------------------------------------------------------------------------
# filter — primitive arrays
# ---------------------------------------------------------------------------


# TODO: this is messy, refactor it!
fn filter_[
    T: DataType
](
    array: PrimitiveArray[T], selection: PrimitiveArray[bool_]
) raises -> PrimitiveArray[T]:
    """Filter a primitive array, keeping only elements where selection is True.

    Iterates over the selection bitmap byte-by-byte.  Zero bytes are skipped
    (sparse-filter fast path).  The output validity bitmap is rebuilt bit-by-bit
    from the source array's validity, filtered by the selection.

    Assumes array.offset == 0 and selection.offset == 0.

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
    var out_len = count_true(selection)

    comptime native = T.native
    comptime elem_size = size_of[Scalar[native]]()
    var buf = BufferBuilder.alloc[native](out_len)
    var bm = BitmapBuilder.alloc(out_len)

    if out_len == n:
        # All selected: single bulk copy of data and validity.
        var bm_bytes = math.ceildiv(n, 8)
        if array.bitmap:
            memcpy(
                dest=bm.unsafe_ptr(),
                src=array.bitmap.value()._buffer.unsafe_ptr(),
                count=bm_bytes,
            )
        else:
            bm.set_range(0, n, True)
        comptime if native == DType.bool:
            # bool_ is bit-packed: copy bitmap-sized bytes, not n bytes
            memcpy(dest=buf.ptr, src=array.buffer.unsafe_ptr(), count=bm_bytes)
        else:
            memcpy(
                dest=buf.ptr, src=array.buffer.unsafe_ptr(), count=n * elem_size
            )
    elif out_len > 0:
        var sel_ptr = selection.buffer.unsafe_ptr()
        # Process 64 bits at a time: 8× fewer outer-loop iterations than bytes.
        var word_ptr = sel_ptr.bitcast[UInt64]()
        var word_count = math.ceildiv(n, 64)
        var out_idx = 0

        # Selectivity > 80%: SlicesIterator — find contiguous 1-bit runs and
        # bulk-copy each run.  Threshold: out_len / n > 0.8 ↔ out_len * 10 > n * 8
        if out_len * 10 > n * 8:
            var has_nulls = array.nulls > 0
            var src_bm_ptr = array.bitmap.value()._buffer.unsafe_ptr() if has_nulls else UnsafePointer[
                UInt8, ImmutExternalOrigin
            ]()
            for word_i in range(word_count):
                var word = word_ptr[word_i]
                if word == 0:
                    continue
                var base = word_i * 64
                while word != 0:
                    # Find start of run: position of first set bit.
                    var start_bit = Int(count_trailing_zeros(word))
                    # Set all bits below start_bit so the next CTZ finds the
                    # first *zero* at or after start_bit (= end of run).
                    word |= (UInt64(1) << UInt64(start_bit)) - 1
                    var end_bit = Int(count_trailing_zeros(~word))
                    # Clear the run bits for the next iteration.
                    if end_bit < 64:
                        word &= ~((UInt64(1) << UInt64(end_bit)) - 1)
                    else:
                        word = 0
                    var run_start = base + start_bit
                    var run_end = min(base + end_bit, n)
                    var run_len = run_end - run_start
                    # Bulk-copy data for this run.
                    comptime if native == DType.bool:
                        # bool_ is bit-packed; fall back to bit-by-bit copy.
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
                            + run_start * elem_size,
                            count=run_len * elem_size,
                        )
                    # Copy validity for this run.
                    if has_nulls:
                        for i in range(run_len):
                            var si = run_start + i
                            if Bool((src_bm_ptr[si >> 3] >> UInt8(si & 7)) & 1):
                                var di = out_idx + i
                                bm.unsafe_ptr()[di >> 3] |= UInt8(1) << UInt8(
                                    di & 7
                                )
                    else:
                        bm.set_range(out_idx, run_len, True)
                    out_idx += run_len
        else:
            # Selectivity ≤ 80%: IndexIterator — scatter individual elements.
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
                    buf.unsafe_set[native](out_idx, array.unsafe_get(elem_i))
                    bm.set_bit(out_idx, array.is_valid(elem_i))
                    out_idx += 1
                    word &= word - 1  # clear lowest set bit

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


fn filter_(
    array: StringArray, selection: PrimitiveArray[bool_]
) raises -> StringArray:
    """Filter a StringArray, keeping only elements where selection is True.

    Composes:
      1. `_string_lengths` to get per-element byte counts.
      2. `filter_[uint32]` to select the relevant lengths.
      3. `sum[uint32]` to compute the total output byte size.
      4. `filter_[bool_]` on the wrapped validity bitmap for output validity.
      5. Prefix sum of selected lengths to build the output offsets buffer.
      6. Run-merging `memcpy` to copy selected string data.

    Assumes array.offset == 0.

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

    # Step 1+2: compute and filter per-element lengths
    var lengths = _string_lengths(array)
    var sel_lengths = filter_[uint32](lengths, selection)

    # Step 3: total output byte size
    var total_bytes = Int(sum_[uint32](sel_lengths))

    # Step 4: output validity — filter the array's validity bitmap.
    # _bitmap_as_bool_array wraps the source validity as a bool array's DATA buffer.
    # After filter_[bool_], out_validity.buffer = filtered validity flags (what we want
    # as the output StringArray's bitmap); out_validity.bitmap = all-True (ignored).
    var validity_bool = _bitmap_as_bool_array(array.bitmap, len(array))
    var out_validity = filter_[bool_](validity_bool, selection)

    var out_len = len(sel_lengths)

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

    var sel_ptr = selection.buffer.unsafe_ptr()
    var n = len(array)
    var byte_count = math.ceildiv(n, 8)

    # Run-merge state: accumulate consecutive selected source ranges
    var run_src_start = (
        -1
    )  # byte offset into array.values where current run starts
    var run_src_end = -1  # exclusive end byte offset (not yet written)
    var dst_offset = 0  # next write position in values_buf

    for byte_i in range(byte_count):
        var f = sel_ptr[byte_i]
        var base = byte_i * 8
        if f == 0:
            # No selected elements in this byte; flush any active run.
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
            continue
        # Must visit every bit: unselected elements create run-merge gaps.
        var mask = Int(f)
        var seen: Int = 0
        while seen < 8:
            var elem_i = base + seen
            if elem_i >= n:
                break
            if mask & 1:
                var src_start = Int(
                    array.offsets.unsafe_get[DType.uint32](elem_i)
                )
                var src_end = Int(
                    array.offsets.unsafe_get[DType.uint32](elem_i + 1)
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
            mask >>= 1
            seen += 1

    # Flush final run
    if run_src_start != -1:
        var run_len = run_src_end - run_src_start
        memcpy(
            dest=dst_ptr + dst_offset,
            src=src_ptr + run_src_start,
            count=run_len,
        )

    # out_validity.buffer holds the bit-packed validity for the output elements.
    var validity_bm = Bitmap(out_validity.buffer, 0, out_len)
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


fn filter_(array: Array, selection: Array) raises -> Array:
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


fn drop_nulls[
    T: DataType
](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Create a new array containing only the valid (non-null) elements.

    Args:
        array: The input array.

    Returns:
        A new PrimitiveArray containing only valid elements.
    """
    var selection = _bitmap_as_bool_array(array.bitmap, len(array))
    return filter_[T](array, selection)


fn drop_nulls(array: Array) raises -> Array:
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
