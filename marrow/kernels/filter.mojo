"""Filter, take, and selection kernels.

``filter``     — select elements where a boolean mask is True.
``take``       — gather elements at arbitrary indices (index -1 → null).
``drop_null``  — remove null elements using the validity bitmap.

All functions support arrays with non-zero offsets (sliced arrays).
"""

from std.bit import count_trailing_zeros
from std.sys import size_of
from std.sys.info import simd_byte_width

from ..arrays import (
    BoolArray,
    PrimitiveArray,
    BinaryLikeArray,
    DynArray,
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
    BinaryLikeBuilder,
)
from ..dtypes import (
    PrimitiveType,
    BinaryLikeType,
    ListLikeType,
    Int32Type,
    bool_,
    int32,
    uint8,
    uint64,
)

from ..views import BitmapView
from .core import Kernel
from ..execution import ExecContext


struct Filter(Kernel):
    """Selection kernel — keep elements where a boolean ``mask`` is True.

    The typed leaves are the ``apply`` overloads below; each operates directly on
    the mask ``BitmapView`` (no ``BoolArray`` wrapping — nested filters recurse by
    handing a child ``BitmapView`` straight to ``dispatch``). ``dispatch``
    resolves a runtime-typed array to the matching leaf.
    """

    comptime name = "filter"

    @staticmethod
    def dispatch(
        array: DynArray,
        mask: BitmapView[_],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Resolve `array`'s runtime dtype and filter it by `mask`."""
        var dt = array.dtype()
        if dt == bool_:
            return Filter.apply(array.as_bool(), mask, ctx).to_dyn()
        elif dt.is_primitive():
            # One arm for every fixed-width type: `apply` is bound on
            # `PrimitiveType`, so numeric, temporal, interval and decimal
            # columns all reach the same leaf and keep their dtype. The
            # numeric and temporal arms this replaces were identical apart
            # from the trait bound, and between them left interval and
            # decimal columns raising `unsupported dtype`.

            def primitive[T: PrimitiveType](d: T) raises {imm} -> DynArray:
                return Filter.apply(array.as_primitive[T](), mask, ctx).to_dyn()

            return dt.dispatch_primitive(primitive)
        elif dt.is_binary_like():

            def binarylike[T: BinaryLikeType](d: T) raises {imm} -> DynArray:
                return Filter.apply(
                    array.as_binary_like[T](), mask, ctx
                ).to_dyn()

            return dt.dispatch_binarylike(binarylike)
        elif dt.is_null():
            return Filter.apply(array.as_null(), mask, ctx).to_dyn()
        elif dt.is_fixed_size_binary():
            return Filter.apply(
                array.as_fixed_size_binary(), mask, ctx
            ).to_dyn()
        elif dt.is_struct():
            return Filter.apply(array.as_struct(), mask, ctx).to_dyn()
        elif dt.is_list_like():

            def listlike[T: ListLikeType](d: T) raises {imm} -> DynArray:
                return Filter.apply(array.as_list_like[T](), mask, ctx).to_dyn()

            return dt.dispatch_listlike(listlike)
        elif dt.is_fixed_size_list():
            return Filter.apply(array.as_fixed_size_list(), mask, ctx).to_dyn()
        elif dt.is_dictionary():
            return Filter.apply(array.as_dictionary(), mask, ctx).to_dyn()
        else:
            raise Self.error(t"unsupported dtype {dt}")

    @staticmethod
    def drop_null[
        T: PrimitiveType
    ](
        array: PrimitiveArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """The valid elements only, keeping the array's type."""
        if not array.bitmap:
            return array.copy()
        return Self.apply(array, array.validity().value(), ctx)

    @staticmethod
    def drop_null(
        array: DynArray, ctx: ExecContext = ExecContext.serial()
    ) raises -> DynArray:
        """Remove null elements: the validity bitmap is itself the keep-mask."""
        if array.dtype().is_null():
            return NullArray(length=0).to_dyn()
        var data = array.to_data()
        if not data.bitmap:
            return array.copy()
        return Filter.dispatch(array, data.validity().value(), ctx)

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        array: PrimitiveArray[T],
        mask: BitmapView[_],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """Filter a primitive array, keeping elements where ``mask`` is set."""
        Self.expect_same_length(len(array), len(mask))
        var out_len, sel_start, sel_end = mask.count_set_bits_with_range()

        if out_len == 0:
            return PrimitiveArray[T].empty(array.dtype)

        # Filter validity bitmap.
        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        if var val_bm := array.validity():
            var filtered_bm, nc = val_bm.value().filter(
                mask, sel_start, sel_end, out_len
            )
            bm = filtered_bm
            null_count = nc

        var result_buf = array.values().filter(
            mask, sel_start, sel_end, out_len
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
        mask: BitmapView[_],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> BoolArray:
        """Filter a bool array, keeping elements where ``mask`` is set."""
        Self.expect_same_length(len(array), len(mask))
        var out_len, sel_start, sel_end = mask.count_set_bits_with_range()

        if out_len == 0:
            return BoolArray.empty()

        # Filter validity bitmap.
        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        if array.bitmap:
            var val_bm = array.validity().value()
            var filtered_bm, nc = val_bm.filter(
                mask, sel_start, sel_end, out_len
            )
            bm = filtered_bm
            null_count = nc

        # Filter data.
        var filtered_data, _ = array.values().filter(
            mask, sel_start, sel_end, out_len
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
        mask: BitmapView[_],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> BinaryLikeArray[T]:
        """Filter a binary-like array (string/binary, 32- or 64-bit offsets),
        keeping elements where ``mask`` is set.

        Word-wise CTZ over the mask visits only set bits (branch-free on
        high-entropy masks); a fully-selected word copies its whole byte span in
        one memcpy (run-merge for dense masks). Validity is filtered in the same
        offset pass.
        """
        Self.expect_same_length(len(array), len(mask))
        comptime O = T.offset
        var n = len(array)
        var out_len = mask.count_set_bits()

        if out_len == 0:
            return BinaryLikeArray[T].empty()

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
                var w = mask.load_bits[DType.uint64](wb)
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
                var w = mask.load_bits[DType.uint64](wb)
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
            var w = mask.load_bits[DType.uint64](wb2)
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
        mask: BitmapView[_],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> NullArray:
        """Filter a null array — the result is a shorter all-null array."""
        Self.expect_same_length(len(array), len(mask))
        return NullArray(length=mask.count_set_bits())

    @staticmethod
    def apply(
        array: FixedSizeBinaryArray,
        mask: BitmapView[_],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> FixedSizeBinaryArray:
        """Filter a fixed-size-binary array by compacting the fixed-width byte
        blocks where ``mask`` is set."""
        Self.expect_same_length(len(array), len(mask))
        var n = len(array)
        var bw = array.byte_width
        var out_len, sel_start, sel_end = mask.count_set_bits_with_range()
        var src = array.buffer.view[DType.uint8]()
        var out = Buffer.alloc_uninit[DType.uint8](out_len * bw)
        var out_view = out.view[DType.uint8]()

        # Compact selected byte blocks — word-wise CTZ over the mask visits only
        # set bits (branch-free on high-entropy masks).
        var dst = 0
        var wb = 0
        while wb < n:
            var w = mask.load_bits[DType.uint64](wb)
            var rem = n - wb
            if rem < 64:
                w &= (UInt64(1) << UInt64(rem)) - 1
            while w != 0:
                var i = wb + Int(count_trailing_zeros(w))
                out_view.slice(dst * bw).copy_from(
                    src.slice((array.offset + i) * bw), bw
                )
                dst += 1
                w &= w - 1
            wb += 64

        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        if array.bitmap:
            var filtered, nc = (
                array.validity()
                .value()
                .filter(mask, sel_start, sel_end, out_len)
            )
            bm = filtered
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
        mask: BitmapView[_],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> ListLikeArray[T]:
        """Filter a list/large-list/map array column-wise: mark each selected
        row's contiguous child range and filter the child recursively, building
        the output offsets in the same pass — no index materialization, no take.
        """
        Self.expect_same_length(len(array), len(mask))
        comptime O = T.offset
        var n = len(array)
        var out_len, sel_start, sel_end = mask.count_set_bits_with_range()
        var offsets = array.offsets.view[O]()
        var child = array.values().copy()

        var new_offsets = Buffer.alloc_uninit[O](out_len + 1)
        var no = new_offsets.view[O]()
        no.unsafe_set(0, Scalar[O](0))
        var child_mask = Bitmap.alloc_zeroed(len(child))
        # Mark each selected row's contiguous child range + build offsets —
        # word-wise CTZ over the mask (branch-free, only set bits visited).
        var total = 0
        var j = 0
        var wb = 0
        while wb < n:
            var w = mask.load_bits[DType.uint64](wb)
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
                j += 1
                no.unsafe_set(j, Scalar[O](total))
                w &= w - 1
            wb += 64

        var new_child = Filter.dispatch(child, child_mask.view(), ctx)
        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        if array.bitmap:
            var filtered, nc = (
                array.validity()
                .value()
                .filter(mask, sel_start, sel_end, out_len)
            )
            bm = filtered
            null_count = nc
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
        mask: BitmapView[_],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> FixedSizeListArray:
        """Filter a fixed-size-list array column-wise: mark each selected row's
        `size` contiguous child slots and filter the child recursively."""
        Self.expect_same_length(len(array), len(mask))
        var n = len(array)
        var size = array.dtype.as_fixed_size_list().size
        var out_len, sel_start, sel_end = mask.count_set_bits_with_range()
        var child = array.values().copy()
        var child_mask = Bitmap.alloc_zeroed(len(child))

        # Mark each selected row's `size` contiguous child slots — word-wise CTZ.
        var wb = 0
        while wb < n:
            var w = mask.load_bits[DType.uint64](wb)
            var rem = n - wb
            if rem < 64:
                w &= (UInt64(1) << UInt64(rem)) - 1
            while w != 0:
                var i = wb + Int(count_trailing_zeros(w))
                child_mask.set_range((array.offset + i) * size, size, True)
                w &= w - 1
            wb += 64

        var new_child = Filter.dispatch(child, child_mask.view(), ctx)
        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        if array.bitmap:
            var filtered, nc = (
                array.validity()
                .value()
                .filter(mask, sel_start, sel_end, out_len)
            )
            bm = filtered
            null_count = nc
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
        mask: BitmapView[_],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DictionaryArray:
        """Filter a dictionary array by compacting its (logical) index codes with
        the fast sequential primitive path and sharing the values unchanged — far
        cheaper than a take (no index materialization, no random gather)."""
        Self.expect_same_length(len(array), len(mask))
        var new_indices = Filter.dispatch(array.indices(), mask, ctx)
        return DictionaryArray(
            dtype=array.type(),
            length=len(new_indices),
            nulls=new_indices.null_count(),
            offset=0,
            indices=new_indices^,
            values=array.dictionary(),
        )

    @staticmethod
    def apply(
        array: StructArray,
        mask: BitmapView[_],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> StructArray:
        """Filter a struct array column-wise: filter every child by the same mask
        and compact the struct-level validity."""
        Self.expect_same_length(len(array), len(mask))
        var n = len(array)
        var out_len, sel_start, sel_end = mask.count_set_bits_with_range()

        var children = [
            Filter.dispatch(array.children[c].slice(array.offset, n), mask, ctx)
            for c in range(len(array.children))
        ]

        var bm: Optional[Bitmap[]] = None
        var null_count = 0
        if array.bitmap:
            var filtered, nc = (
                array.validity()
                .value()
                .filter(mask, sel_start, sel_end, out_len)
            )
            bm = filtered
            null_count = nc

        return StructArray(
            dtype=array.dtype.copy(),
            length=out_len,
            nulls=null_count,
            offset=0,
            bitmap=bm,
            children=children^,
        )


struct Take(Kernel):
    """Gather kernel — collect elements at arbitrary indices (null index → null).

    The typed leaves are the ``apply`` overloads below; ``dispatch`` resolves a
    runtime-typed array to the matching leaf.
    """

    comptime name = "take"

    @staticmethod
    def dispatch(
        array: DynArray,
        indices: Int32Array,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Resolve `array`'s runtime dtype and gather it at `indices`."""
        var dt = array.dtype()
        if dt == bool_:
            return Take.apply(array.as_bool(), indices, ctx).to_dyn()
        elif dt.is_primitive():
            # One arm for every fixed-width type: `apply` is bound on
            # `PrimitiveType`, so numeric, temporal, interval and decimal
            # columns all reach the same leaf and keep their dtype. The
            # numeric and temporal arms this replaces were identical apart
            # from the trait bound, and between them left interval and
            # decimal columns raising `unsupported dtype`.

            def primitive[T: PrimitiveType](d: T) raises {imm} -> DynArray:
                return Take.apply(
                    array.as_primitive[T](), indices, ctx
                ).to_dyn()

            return dt.dispatch_primitive(primitive)
        elif dt.is_binary_like():

            def binarylike[T: BinaryLikeType](d: T) raises {imm} -> DynArray:
                return Take.apply(
                    array.as_binary_like[T](), indices, ctx
                ).to_dyn()

            return dt.dispatch_binarylike(binarylike)
        elif dt.is_null():
            return Take.apply(array.as_null(), indices, ctx).to_dyn()
        elif dt.is_fixed_size_binary():
            return Take.apply(
                array.as_fixed_size_binary(), indices, ctx
            ).to_dyn()
        elif dt.is_struct():
            return Take.apply(array.as_struct(), indices, ctx).to_dyn()
        elif dt.is_list_like():

            def listlike[T: ListLikeType](d: T) raises {imm} -> DynArray:
                return Take.apply(
                    array.as_list_like[T](), indices, ctx
                ).to_dyn()

            return dt.dispatch_listlike(listlike)
        elif dt.is_fixed_size_list():
            return Take.apply(array.as_fixed_size_list(), indices, ctx).to_dyn()
        elif dt.is_dictionary():
            return Take.apply(array.as_dictionary(), indices, ctx).to_dyn()
        else:
            raise Self.error(t"unsupported dtype {dt}")

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        array: PrimitiveArray[T],
        indices: Int32Array,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """Gather elements from a primitive array at the given indices.

        Uses SIMD gather for vectorized collection. Null indices produce
        null output elements (used by outer joins for unmatched rows).
        Source nulls are also propagated.

        The no-null fast path hands its gather loop to ``ctx.stripe``, which
        decides serial versus striped and writes each worker into a disjoint
        output slice. The slow path (null indices or source nulls) stays serial
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
        # Floored at 1: types wider than a SIMD register (decimal256 at 32
        # bytes) would otherwise yield W == 0, which is not a legal store
        # width. At W == 1 the "vector" gather degenerates to a scalar one,
        # which is what those types want anyway.
        comptime W = max(1, simd_byte_width() // size_of[Scalar[native]]())
        var i = 0
        var bitmap = Optional[Bitmap[]](None)
        var null_count = 0

        if not has_null_indices and not has_src_nulls:
            # Fast path: no nulls — pure SIMD gather, no bitmap. Written once;
            # `stripe` decides whether it runs on this thread or across workers.
            # `align=W` keeps every stripe boundary on a vector boundary, so the
            # scalar tail runs at the end of the last stripe rather than once
            # per stripe.
            @always_inline
            def gather(wid: Int, start: Int, end: Int) {imm}:
                var k = start
                while k + W <= end:
                    var offsets = idx.load[W](k).cast[DType.int64]()
                    var vals = src.gather[W](offsets)
                    out.store[W](k, vals)
                    k += W
                while k < end:
                    out.unsafe_set(k, src[Int(idx.unsafe_get(k))])
                    k += 1

            ctx.stripe(n, gather, align=W)
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
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> BoolArray:
        """Gather elements from a bool array at the given indices.

        Null indices produce null output elements.

        Args:
            array: Source bool array.
            indices: Row indices to gather. Null index → null output.
            ctx: Execution context (currently unused — accepted for
                signature parity with the other `take` overloads).

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
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> BinaryLikeArray[T]:
        """Gather elements from a binary-like array at the given indices.

        Null indices produce null output elements.

        Args:
            array: Source binary-like array (string/binary, 32- or 64-bit offsets).
            indices: Row indices to gather. Null index → null output.
            ctx: Execution context (currently unused — accepted for
                signature parity with the other `take` overloads).

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
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> NullArray:
        """Gather from a null array — result is an all-null array of the index count.
        """
        return NullArray(length=len(indices))

    @staticmethod
    def apply(
        array: FixedSizeBinaryArray,
        indices: Int32Array,
        ctx: ExecContext = ExecContext.serial(),
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
        ctx: ExecContext = ExecContext.serial(),
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
        ctx: ExecContext = ExecContext.serial(),
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
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DictionaryArray:
        """Gather rows from a dictionary array: gather its (logical) index array
        with the fast primitive path and share the dictionary values unchanged.
        """
        var new_indices = Take.dispatch(array.indices(), indices, ctx)
        return DictionaryArray(
            dtype=array.type(),
            length=len(indices),
            nulls=new_indices.null_count(),
            offset=0,
            indices=new_indices^,
            values=array.dictionary(),
        )

    @staticmethod
    def apply(
        array: StructArray,
        indices: Int32Array,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> StructArray:
        """Gather rows from a StructArray at the given indices, column-wise.

        Applies ``take`` to each child (sliced to the struct's logical window) and
        gathers the struct-level validity (null index or null source row → null).
        """
        var n = len(array)
        var out_length = len(indices)
        var children = List[DynArray]()
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
# Public API — the three `pc.*` entry points. Everything typed is a method on
# the `Filter` / `Take` kernels; adding an array type means teaching those, not
# writing another delegator here.
# ---------------------------------------------------------------------------


def filter(
    array: DynArray,
    mask: DynArray,
    ctx: ExecContext = ExecContext.serial(),
) raises -> DynArray:
    """Filter `array`, keeping elements where boolean `mask` is True."""
    var m = mask.as_bool().copy()
    return Filter.dispatch(array, m.values(), ctx)


def take(
    array: DynArray,
    indices: Int32Array,
    ctx: ExecContext = ExecContext.serial(),
) raises -> DynArray:
    """Gather elements of `array` at `indices` (null index -> null element)."""
    return Take.dispatch(array, indices, ctx)


def drop_null(
    array: DynArray, ctx: ExecContext = ExecContext.serial()
) raises -> DynArray:
    """Remove null elements using the validity bitmap as the selection."""
    return Filter.drop_null(array, ctx)
