"""Filter, take, and selection kernels.

``filter_``  — select elements where a boolean mask is True.
``take``     — gather elements at arbitrary indices (index -1 → null).
``drop_nulls`` — remove null elements using the validity bitmap.

All functions support arrays with non-zero offsets (sliced arrays).
"""

import std.math as math
from std.bit import pop_count
from std.sys import size_of
from std.sys.info import simd_byte_width

from ..arrays import (
    BoolArray,
    PrimitiveArray,
    StringArray,
    AnyArray,
    StructArray,
)
from ..buffers import Buffer
from ..buffers import Bitmap
from ..builders import BoolBuilder, PrimitiveBuilder, StringBuilder
from ..dtypes import PrimitiveType, bool_, int32, uint32, string, numeric_types
from ..views import BitmapView, BufferView
from .aggregate import sum_
from .string import string_lengths


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# filter — bitmap / values helpers
# ---------------------------------------------------------------------------


def _filter_bits(
    src: BitmapView[_],
    sel: BitmapView[_],
    sel_start: Int,
    sel_end: Int,
    out_len: Int,
) -> Tuple[Bitmap[], Int]:
    """Filter a bitmap, keeping bits where selection is set.

    Uses ``BitmapView.pext`` + ``BitmapView.compressed_store`` in 64-bit
    blocks with run-merge for all-ones and all-zeros blocks.  Works for
    both validity bitmaps and bool data.

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
                i + 64 <= sel_end and sel.load_bits[DType.uint64](i) == ALL_ONES
            ):
                i += 64
            var j = run_start
            while j < i:
                var src_word = src.load_bits[DType.uint64](j)
                out.compressed_store(bm_pos, src_word, 64)
                zero_count += 64 - Int(pop_count(src_word))
                bm_pos += 64
                j += 64
            continue

        # Mixed block: pext + compressed_store.
        var count = Int(pop_count(sel_word))
        var compressed = src.pext(i, sel_word)
        out.compressed_store(bm_pos, compressed, count)
        zero_count += count - Int(pop_count(compressed))
        bm_pos += count
        i += 64

    # Tail: masked pext + compressed_store.
    if i < sel_end:
        var tail = sel_end - i
        var mask = (UInt64(1) << UInt64(tail)) - 1
        var sel_word = sel.load_bits[DType.uint64](i) & mask
        if sel_word != 0:
            var count = Int(pop_count(sel_word))
            var compressed = src.pext(i, sel_word)
            out.compressed_store(bm_pos, compressed, count)
            zero_count += count - Int(pop_count(compressed))

    return builder.to_immutable(length=out_len), zero_count


def _filter_values[
    T: DType
](
    src_buf: Buffer[],
    src_offset: Int,
    sel: BitmapView[_],
    sel_start: Int,
    sel_end: Int,
    out_len: Int,
) -> Buffer[]:
    """Filter fixed-width values, keeping elements where selection is set.

    Uses run-merge for all-ones blocks (memcpy) and density-adaptive
    block dispatch for mixed blocks.

    Args:
        src_buf: Source data buffer.
        src_offset: Element offset into src_buf.
        sel: Selection bitmap view (True = keep).
        sel_start: First 64-bit-aligned block with set bits in sel.
        sel_end: Past-the-end 64-bit-aligned block in sel.
        out_len: Pre-counted number of set bits in sel.

    Returns:
        A new Buffer containing only the selected elements.
    """
    comptime ALL_ONES = ~UInt64(0)
    var buf = Buffer.alloc_uninit(out_len * size_of[Scalar[T]]())
    var src = src_buf.view[T](src_offset)
    var dst = buf.view[T]()
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
                i + 64 <= sel_end and sel.load_bits[DType.uint64](i) == ALL_ONES
            ):
                i += 64
            dst.slice(out_pos).copy_from(src.slice(run_start), i - run_start)
            out_pos += i - run_start
            continue
        out_pos += dst.slice(out_pos).compressed_store(src.slice(i), sel_word)
        i += 64

    # Tail: partial block — force sparse (only reads set-bit positions).
    if i < sel_end:
        var tail = sel_end - i
        var mask = (UInt64(1) << UInt64(tail)) - 1
        var sel_word = sel.load_bits[DType.uint64](i) & mask
        if sel_word != 0:
            dst.slice(out_pos).compressed_store_sparse(src.slice(i), sel_word)
            out_pos += Int(pop_count(sel_word))

    return buf.to_immutable()


# ---------------------------------------------------------------------------
# filter — primitive arrays
# ---------------------------------------------------------------------------


def filter_[
    T: PrimitiveType
](array: PrimitiveArray[T], selection: BoolArray) raises -> PrimitiveArray[T]:
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

    var sel_bm = selection.values()
    var out_len, sel_start, sel_end = sel_bm.count_set_bits_with_range()

    if out_len == 0:
        var empty_buf = Buffer.alloc_zeroed[T.native](0)
        return PrimitiveArray[T](
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=empty_buf.to_immutable(),
        )

    # Filter validity bitmap.
    var bm: Optional[Bitmap[]] = None
    var null_count = 0
    if var val_bm := array.validity():
        var filtered_bm, nc = _filter_bits(
            val_bm.value(), sel_bm, sel_start, sel_end, out_len
        )
        bm = filtered_bm
        null_count = nc

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


def filter_(array: BoolArray, selection: BoolArray) raises -> BoolArray:
    """Filter a bool array, keeping only elements where selection is True.

    Args:
        array: The input bool array.
        selection: Boolean selection mask (True = keep).

    Returns:
        A new BoolArray containing only the selected elements.
    """
    var n = len(array)
    if n != len(selection):
        raise Error(
            t"filter: array length {n} != selection length {len(selection)}"
        )

    var sel_bm = selection.values()
    var out_len, sel_start, sel_end = sel_bm.count_set_bits_with_range()

    if out_len == 0:
        var empty_bm = Bitmap.alloc_zeroed(0)
        return BoolArray(
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=empty_bm.to_immutable(),
        )

    # Filter validity bitmap.
    var bm: Optional[Bitmap[]] = None
    var null_count = 0
    if array.bitmap:
        var val_bm = array.bitmap.value().view(array.offset, n)
        var filtered_bm, nc = _filter_bits(
            val_bm, sel_bm, sel_start, sel_end, out_len
        )
        bm = filtered_bm
        null_count = nc

    # Filter data.
    var data_bm = array.values()
    var filtered_data, _ = _filter_bits(
        data_bm, sel_bm, sel_start, sel_end, out_len
    )
    return BoolArray(
        length=out_len,
        nulls=null_count,
        offset=0,
        bitmap=bm,
        buffer=filtered_data,
    )


# ---------------------------------------------------------------------------
# filter — string arrays
# ---------------------------------------------------------------------------


def filter_(array: StringArray, selection: BoolArray) raises -> StringArray:
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

    var sel_bm = selection.values()
    var out_len = sel_bm.count_set_bits()

    if out_len == 0:
        var empty_offsets = Buffer.alloc_zeroed[DType.uint32](1)
        var empty_values = Buffer.alloc_zeroed[DType.uint8](0)
        return StringArray(
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            offsets=empty_offsets.to_immutable(),
            values=empty_values.to_immutable(),
        )

    var off = array.offset
    var offsets_view = array.offsets.view[DType.uint32]()
    var values_view = array.values.view[DType.uint8]()

    # Phase 1: build output offsets and compute total_bytes in a single pass.
    # This eliminates the separate total_bytes scan over all n elements.
    var out_offsets = Buffer.alloc_uninit[DType.uint32](out_len + 1)
    var out_off_view = out_offsets.view[DType.uint32]()
    var byte_pos = UInt32(0)
    out_off_view.unsafe_set(0, UInt32(0))

    var bm: Optional[Bitmap[]] = None
    var null_count = 0
    var j = 0

    if array.bitmap:
        var src_bm = array.bitmap.value()
        var bm_builder = Bitmap.alloc_zeroed(out_len)
        for i in range(n):
            if sel_bm.test(i):
                byte_pos += offsets_view.unsafe_get(
                    off + i + 1
                ) - offsets_view.unsafe_get(off + i)
                var valid = src_bm.test(off + i)
                if valid:
                    bm_builder.set(j)
                else:
                    null_count += 1
                j += 1
                out_off_view.unsafe_set(j, byte_pos)
        bm = bm_builder.to_immutable(length=out_len)
    else:
        for i in range(n):
            if sel_bm.test(i):
                byte_pos += offsets_view.unsafe_get(
                    off + i + 1
                ) - offsets_view.unsafe_get(off + i)
                j += 1
                out_off_view.unsafe_set(j, byte_pos)

    var total_bytes = Int(byte_pos)

    # Phase 2: copy selected string values using run-merging.
    var out_values = Buffer.alloc_uninit[DType.uint8](total_bytes)
    var out_val_view = out_values.view[DType.uint8]()
    var dst_byte_pos = 0
    var i = 0

    while i < n:
        if not sel_bm.test(i):
            i += 1
            continue

        var run_start = i
        i += 1
        while i < n and sel_bm.test(i):
            i += 1

        var src_byte_start = Int(offsets_view.unsafe_get(off + run_start))
        var src_byte_end = Int(offsets_view.unsafe_get(off + i))
        var run_bytes = src_byte_end - src_byte_start
        if run_bytes > 0:
            out_val_view.slice(dst_byte_pos).copy_from(
                values_view.slice(src_byte_start), run_bytes
            )
            dst_byte_pos += run_bytes

    return StringArray(
        length=out_len,
        nulls=null_count,
        offset=0,
        bitmap=bm,
        offsets=out_offsets.to_immutable(),
        values=out_values.to_immutable(),
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
    var mask = selection.as_bool().copy()

    if array.dtype() == bool_:
        return filter_(array.as_bool().copy(), mask).to_any()

    comptime for dtype in numeric_types:
        if array.dtype() == dtype():
            return filter_[dtype](array.as_primitive[dtype](), mask).to_any()

    if array.dtype().is_string():
        return filter_(array.as_string(), mask).to_any()

    raise Error("filter: unsupported dtype ", array.dtype())


# ---------------------------------------------------------------------------
# drop_nulls — reimplemented via filter
# ---------------------------------------------------------------------------


def drop_nulls[
    T: PrimitiveType
](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Create a new array containing only the valid (non-null) elements.

    Uses the array's validity bitmap directly as the filter selection.

    Args:
        array: The input array.

    Returns:
        A new PrimitiveArray containing only valid elements.
    """
    if not array.bitmap:
        return array.copy()
    var selection = BoolArray(
        length=len(array),
        nulls=0,
        offset=array.offset,
        bitmap=None,
        buffer=array.bitmap.value(),
    )
    return filter_[T](array, selection)


def _drop_nulls_bool(array: BoolArray) raises -> BoolArray:
    """Drop null elements from a bool array."""
    if not array.bitmap:
        return array.copy()
    var selection = BoolArray(
        length=len(array),
        nulls=0,
        offset=array.offset,
        bitmap=None,
        buffer=array.bitmap.value(),
    )
    return filter_(array, selection)


def drop_nulls(array: AnyArray) raises -> AnyArray:
    """Runtime-typed drop_nulls: dispatches to the correct typed version.

    Args:
        array: The input array (runtime-typed).

    Returns:
        A new AnyArray with null elements removed.
    """
    if array.dtype() == bool_:
        return _drop_nulls_bool(array.as_bool().copy()).to_any()

    comptime for dtype in numeric_types:
        if array.dtype() == dtype():
            return drop_nulls[dtype](array.as_primitive[dtype]()).to_any()

    raise Error("drop_nulls: unsupported dtype ", array.dtype())


# ---------------------------------------------------------------------------
# take — gather elements at arbitrary indices (index -1 → null)
# ---------------------------------------------------------------------------


def take[
    T: PrimitiveType
](
    array: PrimitiveArray[T], indices: PrimitiveArray[Int32Type]
) raises -> PrimitiveArray[T]:
    """Gather elements from a primitive array at the given indices.

    Uses SIMD gather for vectorized collection. Null indices produce
    null output elements (used by outer joins for unmatched rows).
    Source nulls are also propagated.

    Args:
        array: Source array.
        indices: Row indices to gather. Null index → null output.

    Returns:
        A new PrimitiveArray with one element per index.
    """
    comptime native = T.native
    var n = len(indices)
    var src = array.values()
    var idx = indices.values()
    var buf = Buffer.alloc_uninit[native](n)
    var out = buf.view[native]()

    var has_null_indices = indices.null_count() > 0
    var has_src_nulls = array.null_count() > 0

    # SIMD gather loop: load W indices, gather W values in parallel.
    # Null indices are masked out (get default value 0).
    comptime W = simd_byte_width() // size_of[Scalar[native]]()
    var i = 0
    var bitmap = Optional[Bitmap[]](None)
    var null_count = 0

    if not has_null_indices and not has_src_nulls:
        # Fast path: no nulls — pure SIMD gather, no bitmap.
        while i + W <= n:
            var offsets = idx.load[W](i).cast[DType.int64]()
            var vals = src.gather[W](offsets)
            out.store[W](i, vals)
            i += W
        while i < n:
            out.unsafe_set(i, src[Int(idx.unsafe_get(i))])
            i += 1
    else:
        # TODO: optimize this, the implementation below could be vectorized
        # Slow path: null indices or source nulls — scalar + bitmap.
        var bm_builder = Bitmap.alloc_zeroed(n)
        while i < n:
            if (has_null_indices and not indices.is_valid(i)) or (
                has_src_nulls and not array.is_valid(Int(idx.unsafe_get(i)))
            ):
                out.unsafe_set(i, Scalar[native](0))
                bm_builder.clear(i)
                null_count += 1
            else:
                out.unsafe_set(i, src[Int(idx.unsafe_get(i))])
                bm_builder.set(i)
            i += 1
        if null_count > 0:
            bitmap = bm_builder.to_immutable(length=n)

    return PrimitiveArray[T](
        length=n,
        nulls=null_count,
        offset=0,
        bitmap=bitmap^,
        buffer=buf.to_immutable(),
    )


def take(
    array: BoolArray, indices: PrimitiveArray[Int32Type]
) raises -> BoolArray:
    """Gather elements from a bool array at the given indices.

    Null indices produce null output elements.

    Args:
        array: Source bool array.
        indices: Row indices to gather. Null index → null output.

    Returns:
        A new BoolArray with one element per index.
    """
    var n = len(indices)
    var has_null_indices = indices.null_count() > 0
    var has_src_nulls = array.null_count() > 0
    var builder = BoolBuilder(capacity=n)
    for i in range(n):
        if has_null_indices and not indices.is_valid(i):
            builder.append_null()
        else:
            var src_idx = Int(indices.unsafe_get(i))
            if has_src_nulls and not array.is_valid(src_idx):
                builder.append_null()
            else:
                builder.append(array[src_idx])
    return builder.finish()


def take(
    array: StringArray, indices: PrimitiveArray[Int32Type]
) raises -> StringArray:
    """Gather elements from a string array at the given indices.

    Null indices produce null output elements.

    Args:
        array: Source string array.
        indices: Row indices to gather. Null index → null output.

    Returns:
        A new StringArray with one element per index.
    """
    var n = len(indices)
    var has_null_indices = indices.null_count() > 0
    var builder = StringBuilder(capacity=n)
    for i in range(n):
        if has_null_indices and not indices.is_valid(i):
            builder.append_null()
        elif array.is_valid(Int(indices.unsafe_get(i))):
            builder.append(array.unsafe_get(UInt(indices.unsafe_get(i))))
        else:
            builder.append_null()
    return builder.finish()


def take(array: AnyArray, indices: PrimitiveArray[Int32Type]) raises -> AnyArray:
    """Gather elements from a type-erased array at the given indices.

    Dispatches to the appropriate typed overload at runtime.

    Args:
        array: Source array (runtime-typed).
        indices: Row indices to gather. -1 produces a null output element.

    Returns:
        A new AnyArray with one element per index.
    """
    if array.dtype() == bool_:
        return take(array.as_bool().copy(), indices).to_any()

    comptime for dt in numeric_types:
        if array.dtype() == dt():
            return take[dt](array.as_primitive[dt](), indices).to_any()

    if array.dtype().is_string():
        return take(array.as_string(), indices).to_any()

    raise Error("take: unsupported dtype ", array.dtype())


def take(
    array: StructArray, indices: PrimitiveArray[Int32Type]
) raises -> StructArray:
    """Gather rows from a StructArray at the given indices.

    Applies ``take`` to each child column independently.
    """
    var children = List[AnyArray]()
    for c in range(len(array.children)):
        children.append(take(array.children[c].copy(), indices))
    var out_length = len(indices)
    return StructArray(
        dtype=array.dtype.copy(),
        length=out_length,
        nulls=0,
        offset=0,
        bitmap=None,
        children=children^,
    )
