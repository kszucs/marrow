"""Filter, take, and selection kernels.

``filter``     — select elements where a boolean mask is True.
``take``       — gather elements at arbitrary indices (index -1 → null).
``drop_null``  — remove null elements using the validity bitmap.

All functions support arrays with non-zero offsets (sliced arrays).
"""

import std.math as math
from std.bit import pop_count, count_trailing_zeros
from std.sys import size_of
from std.sys.info import simd_byte_width

from ..arrays import (
    Array,
    BoolArray,
    PrimitiveArray,
    StringArray,
    BinaryLikeArray,
    AnyArray,
    ArrayData,
    StructArray,
    NullArray,
    FixedSizeBinaryArray,
    ListLikeArray,
    FixedSizeListArray,
    DictionaryArray,
    Int32Array,
)
from ..buffers import Buffer
from ..buffers import Bitmap
from ..builders import (
    BoolBuilder,
    PrimitiveBuilder,
    StringBuilder,
    BinaryLikeBuilder,
)
from ..dtypes import (
    PrimitiveType,
    NumericType,
    BinaryLikeType,
    ListLikeType,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    Float16Type,
    Float32Type,
    Float64Type,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    string,
)
from std.algorithm.functional import sync_parallelize

from ..utils import dispatch_over_numeric
from ..views import BitmapView, BufferView
from .aggregate import sum
from .execution import ExecutionContext
from .helpers import Kernel
from .string import string_lengths


trait SelectionKernel(Kernel):
    """Base for columnar selection kernels (``Filter`` / ``Take``).

    A concrete kernel supplies the ``Selection`` type (the boolean mask for
    filter, the ``Int32`` index array for take) and the per-type ``apply``
    overloads; the shared ``dispatch`` resolves an ``AnyArray``'s runtime dtype
    to the matching ``apply`` — the numeric dtypes fold into a single
    ``dispatch_over_numeric`` arm.
    """

    comptime Selection: Array

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        array: PrimitiveArray[T],
        selection: Self.Selection,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[T]:
        ...

    @staticmethod
    def apply(
        array: BoolArray,
        selection: Self.Selection,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        ...

    @staticmethod
    def apply[
        T: BinaryLikeType
    ](
        array: BinaryLikeArray[T],
        selection: Self.Selection,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BinaryLikeArray[T]:
        ...

    @staticmethod
    def apply(
        array: NullArray,
        selection: Self.Selection,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> NullArray:
        ...

    @staticmethod
    def apply(
        array: FixedSizeBinaryArray,
        selection: Self.Selection,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> FixedSizeBinaryArray:
        ...

    @staticmethod
    def apply[
        T: ListLikeType
    ](
        array: ListLikeArray[T],
        selection: Self.Selection,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> ListLikeArray[T]:
        ...

    @staticmethod
    def apply(
        array: FixedSizeListArray,
        selection: Self.Selection,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> FixedSizeListArray:
        ...

    @staticmethod
    def apply(
        array: DictionaryArray,
        selection: Self.Selection,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> DictionaryArray:
        ...

    @staticmethod
    def apply(
        array: StructArray,
        selection: Self.Selection,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> StructArray:
        ...

    @staticmethod
    def dispatch(
        array: AnyArray,
        selection: Self.Selection,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        """Resolve `array`'s runtime dtype to the matching typed `apply`."""
        var dt = array.dtype()
        if dt == bool_:
            return Self.apply(array.as_bool().copy(), selection, ctx).to_any()
        elif dt.is_numeric():

            @parameter
            def leaf[T: NumericType](d: T) raises -> AnyArray:
                return Self.apply(
                    array.as_primitive[T](), selection, ctx
                ).to_any()

            return dispatch_over_numeric[leaf](dt)
        elif dt.is_string():
            return Self.apply(array.as_string(), selection, ctx).to_any()
        elif dt.is_binary():
            return Self.apply(array.as_binary(), selection, ctx).to_any()
        elif dt.is_large_string():
            return Self.apply(array.as_large_string(), selection, ctx).to_any()
        elif dt.is_large_binary():
            return Self.apply(array.as_large_binary(), selection, ctx).to_any()
        elif dt.is_null():
            return Self.apply(array.as_null(), selection, ctx).to_any()
        elif dt.is_fixed_size_binary():
            return Self.apply(
                array.as_fixed_size_binary(), selection, ctx
            ).to_any()
        elif dt.is_struct():
            return Self.apply(array.as_struct(), selection, ctx).to_any()
        elif dt.is_list():
            return Self.apply(array.as_list(), selection, ctx).to_any()
        elif dt.is_large_list():
            return Self.apply(array.as_large_list(), selection, ctx).to_any()
        elif dt.is_map():
            return Self.apply(array.as_map(), selection, ctx).to_any()
        elif dt.is_fixed_size_list():
            return Self.apply(
                array.as_fixed_size_list(), selection, ctx
            ).to_any()
        elif dt.is_dictionary():
            return Self.apply(array.as_dictionary(), selection, ctx).to_any()
        else:
            raise Error(Self.name, ": unsupported dtype ", dt)


struct Filter(SelectionKernel):
    """Selection kernel — keep elements where a boolean mask is True.

    The typed leaves are the ``apply`` overloads below; ``dispatch`` is inherited
    from ``SelectionKernel``.
    """

    comptime name = "filter"
    comptime Selection = BoolArray

    @staticmethod
    def drop_null(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> AnyArray:
        # A null array is entirely null, so every element is dropped.
        if array.dtype().is_null():
            return NullArray(length=0).to_any()
        var data = array.to_data()
        if not data.bitmap:
            return array.copy()
        # Use the validity bitmap directly as the selection (keep valid).
        var selection = BoolArray(
            length=data.length,
            nulls=0,
            offset=data.offset,
            bitmap=None,
            buffer=data.bitmap.value(),
        )
        return Filter.dispatch(array, selection^, ctx)

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        array: PrimitiveArray[T],
        selection: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """Filter a primitive array, keeping only elements where selection is True.

        Args:
            array: The input primitive array.
            selection: Boolean selection mask (True = keep).
            ctx: Execution context (currently unused — accepted for signature
                uniformity across kernels).

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
                dtype=array.dtype.copy(),
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
            var filtered_bm, nc = val_bm.value().filter(
                sel_bm, sel_start, sel_end, out_len
            )
            bm = filtered_bm
            null_count = nc

        var result_buf = array.values().filter(
            sel_bm, sel_start, sel_end, out_len
        )
        return PrimitiveArray[T](
            dtype=array.dtype.copy(),
            length=out_len,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            buffer=result_buf,
        )

    @staticmethod
    def apply(
        array: BoolArray,
        selection: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        """Filter a bool array, keeping only elements where selection is True.

        Args:
            array: The input bool array.
            selection: Boolean selection mask (True = keep).
            ctx: Execution context (currently unused — accepted for signature
                uniformity across kernels).

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
            var filtered_bm, nc = val_bm.filter(
                sel_bm, sel_start, sel_end, out_len
            )
            bm = filtered_bm
            null_count = nc

        # Filter data.
        var filtered_data, _ = array.values().filter(
            sel_bm, sel_start, sel_end, out_len
        )
        return BoolArray(
            length=out_len,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            buffer=filtered_data,
        )

    @staticmethod
    def apply[
        T: BinaryLikeType
    ](
        array: BinaryLikeArray[T],
        selection: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BinaryLikeArray[T]:
        """Filter a binary-like array (string/binary, 32- or 64-bit offsets),
        keeping only elements where selection is True.

        Uses run merging: consecutive selected elements are copied with a single
        memcpy call to reduce per-element overhead.  When the source has a
        validity bitmap, it is filtered in the same pass (no second traversal).

        Args:
            array: The input binary-like array.
            selection: Boolean selection mask (True = keep).
            ctx: Execution context (currently unused — accepted for signature
                uniformity across kernels).

        Returns:
            A new BinaryLikeArray[T] containing only the selected elements.
        """
        comptime O = T.offset
        var n = len(array)
        if n != len(selection):
            raise Error(
                t"filter: array length {n} != selection length {len(selection)}"
            )

        var sel_bm = selection.values()
        var out_len = sel_bm.count_set_bits()

        if out_len == 0:
            var empty_offsets = Buffer.alloc_zeroed[O](1)
            var empty_values = Buffer.alloc_zeroed[DType.uint8](0)
            return BinaryLikeArray[T](
                length=0,
                nulls=0,
                offset=0,
                bitmap=None,
                offsets=empty_offsets.to_immutable(),
                values=empty_values.to_immutable(),
            )

        comptime ALL_ONES = ~UInt64(0)
        var off = array.offset
        var offsets_view = array.offsets.view[O]()
        var values_view = array.values.view[DType.uint8]()

        # Phase 1: build output offsets (+ filtered validity) and total_bytes in one
        # pass. Word-wise CTZ over the selection bitmap iterates only set bits, so
        # there is no per-element branch to mispredict on high-entropy masks.
        var out_offsets = Buffer.alloc_uninit[O](out_len + 1)
        var out_off_view = out_offsets.view[O]()
        var byte_pos = 0
        out_off_view.unsafe_set(0, Scalar[O](0))

        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        var j = 0

        if array.bitmap:
            var src_bm = array.bitmap.value()
            var bm_builder = Bitmap.alloc_zeroed(out_len)
            var wb = 0
            while wb < n:
                var w = sel_bm.load_bits[DType.uint64](wb)
                var rem = n - wb
                if rem < 64:
                    w &= (UInt64(1) << UInt64(rem)) - 1
                while w != 0:
                    var idx = off + wb + Int(count_trailing_zeros(w))
                    byte_pos += Int(
                        offsets_view.unsafe_get(idx + 1)
                        - offsets_view.unsafe_get(idx)
                    )
                    if src_bm.test(idx):
                        bm_builder.set(j)
                    else:
                        null_count += 1
                    j += 1
                    out_off_view.unsafe_set(j, Scalar[O](byte_pos))
                    w &= w - 1
                wb += 64
            bm = bm_builder.to_immutable(length=out_len)
        else:
            var wb = 0
            while wb < n:
                var w = sel_bm.load_bits[DType.uint64](wb)
                var rem = n - wb
                if rem < 64:
                    w &= (UInt64(1) << UInt64(rem)) - 1
                while w != 0:
                    var idx = off + wb + Int(count_trailing_zeros(w))
                    byte_pos += Int(
                        offsets_view.unsafe_get(idx + 1)
                        - offsets_view.unsafe_get(idx)
                    )
                    j += 1
                    out_off_view.unsafe_set(j, Scalar[O](byte_pos))
                    w &= w - 1
                wb += 64

        var total_bytes = byte_pos

        # Phase 2: copy selected bytes. A fully-selected word (ALL_ONES) copies its
        # entire 64-element span in one memcpy (run-merge for dense/clustered
        # masks); mixed words copy per element via CTZ (branch-free for random).
        var out_values = Buffer.alloc_uninit[DType.uint8](total_bytes)
        var out_val_view = out_values.view[DType.uint8]()
        var dst_byte_pos = 0
        var wb2 = 0
        while wb2 < n:
            var rem = n - wb2
            var w = sel_bm.load_bits[DType.uint64](wb2)
            if rem < 64:
                w &= (UInt64(1) << UInt64(rem)) - 1
            if w == ALL_ONES:
                var s = Int(offsets_view.unsafe_get(off + wb2))
                var e = Int(offsets_view.unsafe_get(off + wb2 + 64))
                if e > s:
                    out_val_view.slice(dst_byte_pos).copy_from(
                        values_view.slice(s), e - s
                    )
                    dst_byte_pos += e - s
            else:
                while w != 0:
                    var idx = off + wb2 + Int(count_trailing_zeros(w))
                    var s = Int(offsets_view.unsafe_get(idx))
                    var run_bytes = Int(offsets_view.unsafe_get(idx + 1)) - s
                    if run_bytes > 0:
                        out_val_view.slice(dst_byte_pos).copy_from(
                            values_view.slice(s), run_bytes
                        )
                        dst_byte_pos += run_bytes
                    w &= w - 1
            wb2 += 64

        return BinaryLikeArray[T](
            length=out_len,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            offsets=out_offsets.to_immutable(),
            values=out_values.to_immutable(),
        )

    @staticmethod
    def apply(
        array: NullArray,
        selection: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> NullArray:
        """Filter a null array — every element is null, so the result is a shorter
        all-null array."""
        return NullArray(length=selection.values().count_set_bits())

    @staticmethod
    def apply(
        array: FixedSizeBinaryArray,
        selection: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> FixedSizeBinaryArray:
        """Filter a fixed-size-binary array by compacting the fixed-width byte
        blocks where selection is set (branch-free word-wise CTZ scan)."""
        var n = len(array)
        var bw = array.byte_width
        var sel = selection.values()
        var out_len, sel_start, sel_end = sel.count_set_bits_with_range()
        var src_view = array.buffer.view[DType.uint8]()
        var out = Buffer.alloc_uninit[DType.uint8](out_len * bw)
        var out_view = out.view[DType.uint8]()

        var dst = 0
        var wb = 0
        while wb < n:
            var w = sel.load_bits[DType.uint64](wb)
            var rem = n - wb
            if rem < 64:
                w &= (UInt64(1) << UInt64(rem)) - 1
            while w != 0:
                var i = wb + Int(count_trailing_zeros(w))
                out_view.slice(dst * bw).copy_from(
                    src_view.slice((array.offset + i) * bw), bw
                )
                dst += 1
                w &= w - 1
            wb += 64

        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        if array.bitmap:
            var filtered_bm, nc = array.bitmap.value().view(
                array.offset, n
            ).filter(sel, sel_start, sel_end, out_len)
            bm = filtered_bm
            null_count = nc
        return FixedSizeBinaryArray(
            length=out_len,
            nulls=null_count,
            offset=0,
            byte_width=bw,
            bitmap=bm,
            buffer=out.to_immutable(),
        )

    @staticmethod
    def apply[
        T: ListLikeType
    ](
        array: ListLikeArray[T],
        selection: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> ListLikeArray[T]:
        """Filter a list/large-list/map array column-wise: mark each selected
        row's contiguous child range in a child mask and filter the child
        recursively, building the output offsets in the same pass — no index
        materialization, no `take`."""
        comptime O = T.offset
        var n = len(array)
        var sel = selection.values()
        var out_len = sel.count_set_bits()
        var offsets = array.offsets.view[O]()
        var child = array.values().copy()

        var new_offsets = Buffer.alloc_uninit[O](out_len + 1)
        var no = new_offsets.view[O]()
        no.unsafe_set(0, Scalar[O](0))
        var child_mask = Bitmap.alloc_zeroed(len(child))
        var need_bm = Bool(array.bitmap)
        var bmb = Bitmap.alloc_zeroed(out_len)
        var null_count = 0
        var total = 0
        var j = 0
        var wb = 0
        while wb < n:
            var w = sel.load_bits[DType.uint64](wb)
            var rem = n - wb
            if rem < 64:
                w &= (UInt64(1) << UInt64(rem)) - 1
            while w != 0:
                var i = wb + Int(count_trailing_zeros(w))
                var s = Int(offsets.unsafe_get(array.offset + i))
                var e = Int(offsets.unsafe_get(array.offset + i + 1))
                if e > s:
                    child_mask.set_range(s, e - s, True)
                total += e - s
                if need_bm:
                    if array.is_valid(i):
                        bmb.set(j)
                    else:
                        null_count += 1
                j += 1
                no.unsafe_set(j, Scalar[O](total))
                w &= w - 1
            wb += 64

        var child_sel = BoolArray(
            length=len(child),
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=child_mask.to_immutable(),
        )
        var new_child = Filter.dispatch(child, child_sel^, ctx)
        var bm: Optional[Bitmap[]] = None
        if need_bm:
            bm = bmb.to_immutable(length=out_len)
        return ListLikeArray[T](
            dtype=array.dtype.copy(),
            length=out_len,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            offsets=new_offsets.to_immutable(),
            values=new_child^,
        )

    @staticmethod
    def apply(
        array: FixedSizeListArray,
        selection: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> FixedSizeListArray:
        """Filter a fixed-size-list array column-wise: mark each selected row's
        `size` contiguous child slots and filter the child recursively."""
        var n = len(array)
        var size = array.dtype.as_fixed_size_list().size
        var sel = selection.values()
        var out_len = sel.count_set_bits()
        var child = array.values().copy()

        var child_mask = Bitmap.alloc_zeroed(len(child))
        var need_bm = Bool(array.bitmap)
        var bmb = Bitmap.alloc_zeroed(out_len)
        var null_count = 0
        var j = 0
        var wb = 0
        while wb < n:
            var w = sel.load_bits[DType.uint64](wb)
            var rem = n - wb
            if rem < 64:
                w &= (UInt64(1) << UInt64(rem)) - 1
            while w != 0:
                var i = wb + Int(count_trailing_zeros(w))
                child_mask.set_range((array.offset + i) * size, size, True)
                if need_bm:
                    if array.is_valid(i):
                        bmb.set(j)
                    else:
                        null_count += 1
                j += 1
                w &= w - 1
            wb += 64

        var child_sel = BoolArray(
            length=len(child),
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=child_mask.to_immutable(),
        )
        var new_child = Filter.dispatch(child, child_sel^, ctx)
        var bm: Optional[Bitmap[]] = None
        if need_bm:
            bm = bmb.to_immutable(length=out_len)
        return FixedSizeListArray(
            dtype=array.dtype.copy(),
            length=out_len,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            values=new_child^,
        )

    @staticmethod
    def apply(
        array: DictionaryArray,
        selection: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> DictionaryArray:
        """Filter a dictionary array by filtering its (logical) index array with the
        fast sequential primitive path and sharing the dictionary values unchanged.

        Filtering the fixed-width codes is a sequential bitmap compaction, far
        cheaper than routing through `take` (index materialization + random gather).
        """
        var data = array.to_data()
        var logical_indices = AnyArray.from_data(
            ArrayData(
                dtype=data.dtype.as_dictionary().index_type().copy(),
                length=data.length,
                nulls=data.nulls,
                offset=data.offset,
                bitmap=data.bitmap,
                buffers=data.buffers.copy(),
                children=List[ArrayData](),
            )
        )
        var new_indices = Filter.dispatch(logical_indices, selection.copy(), ctx)
        var new_nulls = new_indices.null_count()
        return DictionaryArray(
            dtype=array.type(),
            length=len(new_indices),
            nulls=new_nulls,
            offset=0,
            indices=new_indices^,
            values=array.dictionary(),
        )

    @staticmethod
    def apply(
        array: StructArray,
        selection: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> StructArray:
        """Filter a struct array column-wise: filter each child by the same mask and
        the struct-level validity, keeping the layout fully columnar."""
        var n = len(array)
        if n != len(selection):
            raise Error(
                t"filter: array length {n} != selection length {len(selection)}"
            )
        var sel = selection.values()
        var out_len = sel.count_set_bits()

        var children = List[AnyArray]()
        for c in range(len(array.children)):
            children.append(
                Filter.dispatch(
                    array.children[c].slice(array.offset, n),
                    selection.copy(),
                    ctx,
                )
            )

        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        if array.bitmap:
            var bmb = Bitmap.alloc_zeroed(out_len)
            var j = 0
            for i in range(n):
                if sel.test(i):
                    if array.is_valid(i):
                        bmb.set(j)
                    else:
                        null_count += 1
                    j += 1
            bm = bmb.to_immutable(length=out_len)

        return StructArray(
            dtype=array.dtype.copy(),
            length=out_len,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            children=children^,
        )


struct Take(SelectionKernel):
    """Gather kernel — collect elements at arbitrary indices (null index → null).

    The typed leaves are the ``apply`` overloads below; ``dispatch`` is inherited
    from ``SelectionKernel``.
    """

    comptime name = "take"
    comptime Selection = Int32Array

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        array: PrimitiveArray[T],
        indices: Int32Array,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """Gather elements from a primitive array at the given indices.

        Uses SIMD gather for vectorized collection. Null indices produce
        null output elements (used by outer joins for unmatched rows).
        Source nulls are also propagated.

        When ``ctx.num_threads > 1`` and ``indices`` is long enough, the
        no-null fast path stripes the gather loop across workers via
        ``sync_parallelize`` — each worker writes to a disjoint output
        slice.  The slow path (null indices or source nulls) stays serial
        because it builds the validity bitmap in-order.

        Args:
            array: Source array.
            indices: Row indices to gather. Null index → null output.
            ctx: Execution context — controls CPU stripe parallelism.

        Returns:
            A new PrimitiveArray with one element per index.
        """
        comptime native = T.native
        var n = len(indices)
        var src = array.values()
        var idx = indices.values()
        var buf = Buffer.alloc_uninit[native](n)
        var out = buf.view[native](0, n)

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
            if ctx.wants_parallel(n):
                var nt = ctx.resolved_num_threads()
                # Round chunk up to a SIMD width so each worker owns a
                # self-contained gather boundary and the tail scalar loop
                # only runs at the very end of the last worker's stripe.
                var chunk = ((n + nt - 1) // nt + W - 1) // W * W

                @parameter
                def worker(t: Int):
                    var start = t * chunk
                    if start >= n:
                        return
                    var end = min(start + chunk, n)
                    var k = start
                    while k + W <= end:
                        var offsets = idx.load[W](k).cast[DType.int64]()
                        var vals = src.gather[W](offsets)
                        out.store[W](k, vals)
                        k += W
                    while k < end:
                        out.unsafe_set(k, src[Int(idx.unsafe_get(k))])
                        k += 1

                sync_parallelize[worker](nt)
            else:
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
            dtype=array.dtype.copy(),
            length=n,
            nulls=null_count,
            offset=0,
            bitmap=bitmap^,
            buffer=buf.to_immutable(),
        )

    @staticmethod
    def apply(
        array: BoolArray,
        indices: Int32Array,
        ctx: ExecutionContext = ExecutionContext.serial(),
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
                    builder.append(array[src_idx].value())
        return builder.finish()

    @staticmethod
    def apply[
        T: BinaryLikeType
    ](
        array: BinaryLikeArray[T],
        indices: Int32Array,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BinaryLikeArray[T]:
        """Gather elements from a binary-like array at the given indices.

        Null indices produce null output elements.

        Args:
            array: Source binary-like array (string/binary, 32- or 64-bit offsets).
            indices: Row indices to gather. Null index → null output.

        Returns:
            A new BinaryLikeArray[T] with one element per index.
        """
        var n = len(indices)
        var has_null_indices = indices.null_count() > 0
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if has_null_indices and not indices.is_valid(i):
                builder.append_null()
            elif array.is_valid(Int(indices.unsafe_get(i))):
                builder.append(array.unsafe_get(UInt(indices.unsafe_get(i))))
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def apply(
        array: NullArray,
        indices: Int32Array,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> NullArray:
        """Gather from a null array — result is an all-null array of the index count.
        """
        return NullArray(length=len(indices))

    @staticmethod
    def apply(
        array: FixedSizeBinaryArray,
        indices: Int32Array,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> FixedSizeBinaryArray:
        """Gather fixed-width byte blocks from a fixed-size-binary array. Null index
        or null source row → null output block."""
        var n = len(indices)
        var bw = array.byte_width
        var out = Buffer.alloc_zeroed[DType.uint8](n * bw)
        var out_view = out.view[DType.uint8]()
        var src_view = array.buffer.view[DType.uint8]()
        var has_null_idx = indices.null_count() > 0

        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        if array.bitmap or has_null_idx:
            var bmb = Bitmap.alloc_zeroed(n)
            for i in range(n):
                if has_null_idx and not indices.is_valid(i):
                    null_count += 1
                else:
                    var idx = Int(indices.unsafe_get(i))
                    if array.is_valid(idx):
                        out_view.slice(i * bw).copy_from(
                            src_view.slice((array.offset + idx) * bw), bw
                        )
                        bmb.set(i)
                    else:
                        null_count += 1
            bm = bmb.to_immutable(length=n)
        else:
            for i in range(n):
                var idx = Int(indices.unsafe_get(i))
                out_view.slice(i * bw).copy_from(
                    src_view.slice((array.offset + idx) * bw), bw
                )

        return FixedSizeBinaryArray(
            length=n,
            nulls=null_count,
            offset=0,
            byte_width=bw,
            bitmap=bm,
            buffer=out.to_immutable(),
        )

    @staticmethod
    def apply[
        T: ListLikeType
    ](
        array: ListLikeArray[T],
        indices: Int32Array,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> ListLikeArray[T]:
        """Gather rows from a list/large-list/map array: remap offsets and gather the
        referenced child sub-ranges via a single child `take`. Null index or null
        source row → null (empty) output row."""
        comptime O = T.offset
        var n = len(indices)
        var out_offsets = Buffer.alloc_uninit[O](n + 1)
        var oview = out_offsets.view[O]()
        oview.unsafe_set(0, Scalar[O](0))

        var has_null_idx = indices.null_count() > 0
        var need_bm = Bool(array.bitmap) or has_null_idx
        var bmb = Bitmap.alloc_zeroed(n)
        var null_count = 0
        var total = 0

        # Pass 1: output offsets + validity + total child length (upfront so the
        # child-index buffer is sized exactly — indices may repeat rows).
        for k in range(n):
            if has_null_idx and not indices.is_valid(k):
                null_count += 1
            else:
                var idx = Int(indices.unsafe_get(k))
                if array.is_valid(idx):
                    var rng = array.child_range(idx)
                    total += rng[1] - rng[0]
                    bmb.set(k)
                else:
                    null_count += 1
            oview.unsafe_set(k + 1, Scalar[O](total))

        # Pass 2: materialize the (per-row contiguous) child indices into a raw
        # Int32 buffer — no builder append / growth / null-bitmap overhead — then
        # gather the child in a single dispatched `take`.
        var child_idx_buf = Buffer.alloc_uninit[Int32Type.native](total)
        var civ = child_idx_buf.view[Int32Type.native]()
        var pos = 0
        for k in range(n):
            if has_null_idx and not indices.is_valid(k):
                continue
            var idx = Int(indices.unsafe_get(k))
            if array.is_valid(idx):
                var rng = array.child_range(idx)
                for j in range(rng[0], rng[1]):
                    civ.unsafe_set(pos, Int32(j))
                    pos += 1
        var child_indices = Int32Array(
            dtype=int32,
            length=total,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=child_idx_buf.to_immutable(),
        )

        var bm: Optional[Bitmap[]] = None
        if need_bm:
            bm = bmb.to_immutable(length=n)
        var new_child = Take.dispatch(array.values().copy(), child_indices, ctx)
        return ListLikeArray[T](
            dtype=array.dtype.copy(),
            length=n,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            offsets=out_offsets.to_immutable(),
            values=new_child^,
        )

    @staticmethod
    def apply(
        array: FixedSizeListArray,
        indices: Int32Array,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> FixedSizeListArray:
        """Gather rows from a fixed-size-list array: expand each row index into its
        `list_size` contiguous child positions and gather the child in one `take`.
        Null index → a null row of `list_size` null children."""
        var n = len(indices)
        var size = array.dtype.as_fixed_size_list().size
        var total = n * size
        var has_null_idx = indices.null_count() > 0
        var need_bm = Bool(array.bitmap) or has_null_idx
        var bmb = Bitmap.alloc_zeroed(n)
        var null_count = 0

        # Materialize the `size`-strided child indices into a raw (uninitialized)
        # Int32 buffer. Only when some index is null do we need a child validity
        # bitmap (a null index expands to `size` null child slots); the common
        # no-null path skips that zeroed allocation entirely.
        var child_idx_buf = Buffer.alloc_uninit[Int32Type.native](total)
        var civ = child_idx_buf.view[Int32Type.native]()
        var child_bitmap: Optional[Bitmap[]] = None
        var child_nulls = 0
        var pos = 0

        if has_null_idx:
            var child_bm = Bitmap.alloc_zeroed(total)
            for k in range(n):
                if not indices.is_valid(k):
                    null_count += 1
                    child_nulls += size
                    for _ in range(size):
                        civ.unsafe_set(pos, Int32(0))
                        pos += 1
                else:
                    var idx = Int(indices.unsafe_get(k))
                    var base = (array.offset + idx) * size
                    for j in range(size):
                        civ.unsafe_set(pos, Int32(base + j))
                        child_bm.set(pos)
                        pos += 1
                    if array.is_valid(idx):
                        bmb.set(k)
                    else:
                        null_count += 1
            if child_nulls > 0:
                child_bitmap = child_bm.to_immutable(length=total)
        else:
            for k in range(n):
                var idx = Int(indices.unsafe_get(k))
                var base = (array.offset + idx) * size
                for j in range(size):
                    civ.unsafe_set(pos, Int32(base + j))
                    pos += 1
                if array.is_valid(idx):
                    bmb.set(k)
                else:
                    null_count += 1

        var child_indices = Int32Array(
            dtype=int32,
            length=total,
            nulls=child_nulls,
            offset=0,
            bitmap=child_bitmap^,
            buffer=child_idx_buf.to_immutable(),
        )

        var bm: Optional[Bitmap[]] = None
        if need_bm:
            bm = bmb.to_immutable(length=n)
        var new_child = Take.dispatch(array.values().copy(), child_indices, ctx)
        return FixedSizeListArray(
            dtype=array.dtype.copy(),
            length=n,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            values=new_child^,
        )

    @staticmethod
    def apply(
        array: DictionaryArray,
        indices: Int32Array,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> DictionaryArray:
        """Gather rows from a dictionary array: gather its (logical) index array with
        the fast primitive path and share the dictionary values unchanged."""
        var data = array.to_data()
        var logical_indices = AnyArray.from_data(
            ArrayData(
                dtype=data.dtype.as_dictionary().index_type().copy(),
                length=data.length,
                nulls=data.nulls,
                offset=data.offset,
                bitmap=data.bitmap,
                buffers=data.buffers.copy(),
                children=List[ArrayData](),
            )
        )
        var new_indices = Take.dispatch(logical_indices, indices, ctx)
        var new_nulls = new_indices.null_count()
        return DictionaryArray(
            dtype=array.type(),
            length=len(indices),
            nulls=new_nulls,
            offset=0,
            indices=new_indices^,
            values=array.dictionary(),
        )

    @staticmethod
    def apply(
        array: StructArray,
        indices: Int32Array,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> StructArray:
        """Gather rows from a StructArray at the given indices, column-wise.

        Applies ``take`` to each child (sliced to the struct's logical window) and
        gathers the struct-level validity (null index or null source row → null).
        """
        var n = len(array)
        var out_length = len(indices)
        var children = List[AnyArray]()
        for c in range(len(array.children)):
            children.append(
                Take.dispatch(
                    array.children[c].slice(array.offset, n), indices, ctx
                )
            )

        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        var has_null_idx = indices.null_count() > 0
        if array.bitmap or has_null_idx:
            var bmb = Bitmap.alloc_zeroed(out_length)
            for k in range(out_length):
                if has_null_idx and not indices.is_valid(k):
                    null_count += 1
                elif array.is_valid(Int(indices.unsafe_get(k))):
                    bmb.set(k)
                else:
                    null_count += 1
            bm = bmb.to_immutable(length=out_length)

        return StructArray(
            dtype=array.dtype.copy(),
            length=out_length,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            children=children^,
        )


# ---------------------------------------------------------------------------
# Public API — thin free delegators to the Filter / Take kernels
# ---------------------------------------------------------------------------


def filter(
    array: AnyArray,
    selection: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Filter `array`, keeping elements where `selection` is True."""
    return Filter.dispatch(array, selection.as_bool().copy(), ctx)


def filter[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    selection: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    return Filter.apply(array, selection, ctx)


def filter(
    array: BoolArray,
    selection: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    return Filter.apply(array, selection, ctx)


def filter[
    T: BinaryLikeType
](
    array: BinaryLikeArray[T],
    selection: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BinaryLikeArray[T]:
    return Filter.apply(array, selection, ctx)


def filter(
    array: NullArray,
    selection: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> NullArray:
    return Filter.apply(array, selection, ctx)


def filter(
    array: FixedSizeBinaryArray,
    selection: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> FixedSizeBinaryArray:
    return Filter.apply(array, selection, ctx)


def filter[
    T: ListLikeType
](
    array: ListLikeArray[T],
    selection: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> ListLikeArray[T]:
    return Filter.apply(array, selection, ctx)


def filter(
    array: FixedSizeListArray,
    selection: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> FixedSizeListArray:
    return Filter.apply(array, selection, ctx)


def filter(
    array: DictionaryArray,
    selection: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> DictionaryArray:
    return Filter.apply(array, selection, ctx)


def filter(
    array: StructArray,
    selection: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> StructArray:
    return Filter.apply(array, selection, ctx)


def drop_null(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> AnyArray:
    """Remove null elements using the validity bitmap as the selection."""
    return Filter.drop_null(array, ctx)


def take(
    array: AnyArray,
    indices: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Gather elements of `array` at `indices` (null index -> null element)."""
    return Take.dispatch(array, indices, ctx)


def take[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    indices: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    return Take.apply(array, indices, ctx)


def take(
    array: BoolArray,
    indices: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    return Take.apply(array, indices, ctx)


def take[
    T: BinaryLikeType
](
    array: BinaryLikeArray[T],
    indices: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BinaryLikeArray[T]:
    return Take.apply(array, indices, ctx)


def take(
    array: NullArray,
    indices: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> NullArray:
    return Take.apply(array, indices, ctx)


def take(
    array: FixedSizeBinaryArray,
    indices: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> FixedSizeBinaryArray:
    return Take.apply(array, indices, ctx)


def take[
    T: ListLikeType
](
    array: ListLikeArray[T],
    indices: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> ListLikeArray[T]:
    return Take.apply(array, indices, ctx)


def take(
    array: FixedSizeListArray,
    indices: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> FixedSizeListArray:
    return Take.apply(array, indices, ctx)


def take(
    array: DictionaryArray,
    indices: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> DictionaryArray:
    return Take.apply(array, indices, ctx)


def take(
    array: StructArray,
    indices: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> StructArray:
    return Take.apply(array, indices, ctx)


def drop_null[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Create a new array containing only the valid (non-null) elements.

    Uses the array's validity bitmap directly as the filter selection.

    Args:
        array: The input array.
        ctx: Execution context (currently unused — accepted for signature
            uniformity across kernels).

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
    return Filter.apply(array, selection, ctx)


def _drop_null_bool(
    array: BoolArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> BoolArray:
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
    return Filter.apply(array, selection, ctx)
