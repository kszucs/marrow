"""Sort kernels — argsort and sort for Marrow arrays.

Phase 1: single-column argsort.
  - Primitive arrays: LSD radix sort (O(N)) for N ≥ 1024; PDQsort for N < 1024.
  - BoolArray: O(N) counting sort.
  - StringArray: stdlib comparison sort.
Phase 2 (future): multi-column permutation refinement on StructArray.
"""

from std.builtin.sort import sort as _sort_impl
from std.algorithm.functional import sync_parallelize
from std.sys import size_of

from ..arrays import (
    BoolArray,
    PrimitiveArray,
    StringArray,
    StructArray,
    AnyArray,
    Int32Array,
)
from ..buffers import Buffer, Bitmap
from ..dtypes import (
    PrimitiveType,
    bool_ as bool_dt,
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
    Int32Type,
)
from .execution import ExecutionContext
from .filter import take as _take


comptime _RADIX_THRESHOLD: Int = 1024
"""Arrays below this size use Mojo stdlib PDQsort instead of LSD radix.

For N < 1024 the 8-pass radix setup cost (3 buffer allocs + encode loop)
exceeds the cost of PDQsort. Above 1024, parallel radix dominates.
"""

comptime _PARALLEL_THRESHOLD: Int = 65_536
"""Minimum element count for parallel histogram/scatter in radix sort."""


# ---------------------------------------------------------------------------
# Key encoding — value → UInt64 for radix sort comparisons
# ---------------------------------------------------------------------------


def _encode_sort_key[T: PrimitiveType](
    val: Scalar[T.native],
    ascending: Bool,
) -> UInt64:
    """Encode val as UInt64 that sorts correctly in unsigned ascending order.

    Float transform: positive → XOR sign bit; negative → XOR all bits
    (NaN becomes uint max, sorting last in ascending order).
    Signed int transform: XOR sign bit so -N < 0 < +N in unsigned order.
    Unsigned: zero-extend cast.
    Descending: complement all bits.
    """
    comptime native = T.native
    var key: UInt64

    comptime if native == DType.float16:
        var bits = val.to_bits().cast[DType.uint16]()
        var sign = bits >> 15
        var flip = (UInt16(0) - sign) | UInt16(0x8000)
        key = (bits ^ flip).cast[DType.uint64]()
    elif native == DType.float32:
        var bits = val.to_bits().cast[DType.uint32]()
        var sign = bits >> 31
        var flip = (UInt32(0) - sign) | UInt32(0x80000000)
        key = (bits ^ flip).cast[DType.uint64]()
    elif native == DType.float64:
        var bits = val.to_bits().cast[DType.uint64]()
        var sign = bits >> 63
        var flip = (UInt64(0) - sign) | UInt64(0x8000000000000000)
        key = bits ^ flip
    elif native == DType.int8:
        key = val.to_bits().cast[DType.uint64]() ^ UInt64(0x80)
    elif native == DType.int16:
        key = val.to_bits().cast[DType.uint64]() ^ UInt64(0x8000)
    elif native == DType.int32:
        key = val.to_bits().cast[DType.uint64]() ^ UInt64(0x80000000)
    elif native == DType.int64:
        key = val.to_bits().cast[DType.uint64]() ^ UInt64(0x8000000000000000)
    else:
        # uint8, uint16, uint32, uint64 — no encoding needed.
        key = val.cast[DType.uint64]()

    if not ascending:
        key = ~key
    return key


# ---------------------------------------------------------------------------
# Comparison sort — Mojo stdlib PDQsort for small arrays (N < _RADIX_THRESHOLD)
# ---------------------------------------------------------------------------


def _comparison_sort_indices[T: PrimitiveType](
    src: PrimitiveArray[T],
    var idx_buf: Buffer[mut=True],
    n: Int,
    ascending: Bool,
) raises -> Buffer[]:
    """Sort idx_buf in-place via Mojo stdlib sort (PDQsort / introsort).

    Uses encoded UInt64 keys so NaN/signed-int order matches the radix path.
    _sort_impl takes a List, so we copy indices in, sort, copy back.
    For N < 1024 the copy overhead is negligible vs 8-pass radix.
    """
    var values = src.values()
    var v = idx_buf.view[DType.int32](0, n)

    var idx_list = List[Int32](capacity=n)
    for i in range(n):
        idx_list.append(v.unsafe_get(i))

    if ascending:

        @parameter
        def cmp_asc(a: Int32, b: Int32) -> Bool:
            return _encode_sort_key[T](
                values.unsafe_get(Int(a)), True
            ) < _encode_sort_key[T](values.unsafe_get(Int(b)), True)

        _sort_impl[cmp_asc](idx_list)
    else:

        @parameter
        def cmp_desc(a: Int32, b: Int32) -> Bool:
            return _encode_sort_key[T](
                values.unsafe_get(Int(a)), False
            ) < _encode_sort_key[T](values.unsafe_get(Int(b)), False)

        _sort_impl[cmp_desc](idx_list)

    for i in range(n):
        v.unsafe_set(i, idx_list[i])
    return idx_buf^.to_immutable()


# ---------------------------------------------------------------------------
# LSD radix sort — O(N), parallel histogram + scatter
# ---------------------------------------------------------------------------


def _radix_sort_indices[T: PrimitiveType](
    src: PrimitiveArray[T],
    var idx_buf: Buffer[mut=True],
    n: Int,
    ascending: Bool,
    ctx: ExecutionContext,
) raises -> Buffer[]:
    """LSD radix sort over encoded UInt64 keys.

    Allocates two alternating (key, index) buffer pairs. Each 8-bit pass reads
    from pair-A and scatters into pair-B, then swaps. After ``num_passes``
    passes the result resides in ``idx_buf`` (or ``idx_b`` when num_passes is
    odd — i.e. for 8-bit types). Returns the sorted original-row indices as an
    immutable Buffer.

    Parallel path (when ``ctx.wants_parallel(n, _PARALLEL_THRESHOLD)``):
    per-thread histograms → partition-major prefix sum → disjoint scatter
    slots, no atomics. Mirrors the ``RadixPartitioner`` pattern in
    ``hashtable.mojo``.
    """
    comptime native = T.native
    comptime num_passes = size_of[Scalar[native]]()

    var values = src.values()

    var key_a = Buffer.alloc_uninit[DType.uint64](n)
    var key_b = Buffer.alloc_uninit[DType.uint64](n)
    var idx_b = Buffer.alloc_uninit[DType.int32](n)

    # Fill key_a: encode each valid element's value.
    var ka_init = key_a.view[DType.uint64](0, n)
    var ia_init = idx_buf.view[DType.int32](0, n)
    for i in range(n):
        ka_init.unsafe_set(
            i,
            _encode_sort_key[T](
                values.unsafe_get(Int(ia_init.unsafe_get(i))), ascending
            ),
        )

    var nt = 1
    if ctx.wants_parallel(n, _PARALLEL_THRESHOLD):
        nt = ctx.resolved_num_threads()
    var chunk = (n + nt - 1) // nt

    for pass_ in range(num_passes):
        var shift = UInt64(pass_ * 8)
        var histograms = List[Int](length=nt * 256, fill=0)

        # Per-thread histogram.
        var ka_h = key_a.view[DType.uint64](0, n)

        @parameter
        def hist_worker(t: Int):
            var start = t * chunk
            if start >= n:
                return
            var end = min(start + chunk, n)
            var base = t * 256
            for i in range(start, end):
                histograms[base + Int((ka_h.unsafe_get(i) >> shift) & 0xFF)] += 1

        sync_parallelize[hist_worker](nt)

        # Partition-major prefix sum → per-thread write offsets.
        var write_offsets = List[Int](length=nt * 256, fill=0)
        var running = 0
        for b in range(256):
            for t in range(nt):
                write_offsets[t * 256 + b] = running
                running += histograms[t * 256 + b]

        # Parallel scatter.
        var ka_s = key_a.view[DType.uint64](0, n)
        var ia_s = idx_buf.view[DType.int32](0, n)
        var kb_s = key_b.view[DType.uint64](0, n)
        var ib_s = idx_b.view[DType.int32](0, n)

        @parameter
        def scatter_worker(t: Int):
            var start = t * chunk
            if start >= n:
                return
            var end = min(start + chunk, n)
            var base = t * 256
            for i in range(start, end):
                var b = Int((ka_s.unsafe_get(i) >> shift) & 0xFF)
                var pos = write_offsets[base + b]
                kb_s.unsafe_set(pos, ka_s.unsafe_get(i))
                ib_s.unsafe_set(pos, ia_s.unsafe_get(i))
                write_offsets[base + b] = pos + 1

        sync_parallelize[scatter_worker](nt)

        # Swap A ↔ B for next pass.
        var tmp_key = key_a^
        key_a = key_b^
        key_b = tmp_key^
        var tmp_idx = idx_buf^
        idx_buf = idx_b^
        idx_b = tmp_idx^

    # After num_passes swaps the result is always in idx_buf (the "current A").
    return idx_buf^.to_immutable()


# ---------------------------------------------------------------------------
# _assemble_output — combine sorted valid + null index lists into Int32Array
# ---------------------------------------------------------------------------


def _assemble_output(
    sorted_valid: Buffer[],
    n_valid: Int,
    null_list: List[Int32],
    n: Int,
    nulls_first: Bool,
) raises -> Int32Array:
    """Merge sorted valid indices and null indices into the final Int32Array."""
    var out = Buffer.alloc_uninit[DType.int32](n)
    var ov = out.view[DType.int32](0, n)
    var sv = sorted_valid.view[DType.int32](0, n_valid)
    var n_null = n - n_valid
    var null_off = 0 if nulls_first else n_valid
    var valid_off = n_null if nulls_first else 0
    for i in range(n_null):
        ov.unsafe_set(null_off + i, null_list[i])
    for i in range(n_valid):
        ov.unsafe_set(valid_off + i, sv.unsafe_get(i))
    return Int32Array(
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=out^.to_immutable(),
    )


# ---------------------------------------------------------------------------
# Typed argsort implementations
# ---------------------------------------------------------------------------


def _argsort_primitive[T: PrimitiveType](
    array: PrimitiveArray[T],
    ascending: Bool = True,
    nulls_first: Bool = True,
    stable: Bool = False,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> Int32Array:
    """Argsort a typed primitive array. Returns sorted row indices."""
    var n = len(array)
    if n == 0:
        return Int32Array(
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=Buffer.alloc_uninit[DType.int32](1).to_immutable(),
        )

    var n_null = array.null_count()
    var n_valid = n - n_null

    # Partition: collect valid and null original row indices in one scan.
    var null_list = List[Int32](capacity=max(n_null, 1))
    var valid_buf = Buffer.alloc_uninit[DType.int32](max(n_valid, 1))
    var vv = valid_buf.view[DType.int32](0, n_valid)
    var vi = 0
    for i in range(n):
        if array.is_valid(i):
            vv.unsafe_set(vi, Int32(i))
            vi += 1
        else:
            null_list.append(Int32(i))

    # Sort valid indices: radix for large arrays, PDQsort for small.
    var sorted_valid: Buffer[]
    if n_valid <= 1:
        sorted_valid = valid_buf^.to_immutable()
    elif n_valid < _RADIX_THRESHOLD:
        sorted_valid = _comparison_sort_indices[T](
            array, valid_buf^, n_valid, ascending
        )
    else:
        sorted_valid = _radix_sort_indices[T](
            array, valid_buf^, n_valid, ascending, ctx
        )

    return _assemble_output(sorted_valid, n_valid, null_list, n, nulls_first)


def _argsort_bool(
    array: BoolArray,
    ascending: Bool = True,
    nulls_first: Bool = True,
) raises -> Int32Array:
    """O(N) counting sort for bool arrays.

    Enumerates indices into three bins (null, false, true) in one pass,
    then assembles the output in the requested order.
    """
    var n = len(array)
    if n == 0:
        return Int32Array(
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=Buffer.alloc_uninit[DType.int32](1).to_immutable(),
        )

    var null_list = List[Int32]()
    var false_list = List[Int32]()
    var true_list = List[Int32]()
    var bv = array.values()  # offset-adjusted BitmapView for data bits
    for i in range(n):
        if not array.is_valid(i):
            null_list.append(Int32(i))
        elif bv.test(i):
            true_list.append(Int32(i))
        else:
            false_list.append(Int32(i))

    var out = Buffer.alloc_uninit[DType.int32](n)
    var ov = out.view[DType.int32](0, n)
    var pos = 0
    if nulls_first:
        for i in range(len(null_list)):
            ov.unsafe_set(pos, null_list[i])
            pos += 1
    if ascending:
        for i in range(len(false_list)):
            ov.unsafe_set(pos, false_list[i])
            pos += 1
        for i in range(len(true_list)):
            ov.unsafe_set(pos, true_list[i])
            pos += 1
    else:
        for i in range(len(true_list)):
            ov.unsafe_set(pos, true_list[i])
            pos += 1
        for i in range(len(false_list)):
            ov.unsafe_set(pos, false_list[i])
            pos += 1
    if not nulls_first:
        for i in range(len(null_list)):
            ov.unsafe_set(pos, null_list[i])
            pos += 1

    return Int32Array(
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=out^.to_immutable(),
    )


def _argsort_string(
    array: StringArray,
    ascending: Bool = True,
    nulls_first: Bool = True,
    stable: Bool = False,
) raises -> Int32Array:
    """Comparison sort for string arrays using the Mojo stdlib sort."""
    var n = len(array)
    if n == 0:
        return Int32Array(
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=Buffer.alloc_uninit[DType.int32](1).to_immutable(),
        )

    var n_null = array.null_count()
    var n_valid = n - n_null

    var valid_list = List[Int32](capacity=max(n_valid, 1))
    var null_list = List[Int32](capacity=max(n_null, 1))
    for i in range(n):
        if array.is_valid(i):
            valid_list.append(Int32(i))
        else:
            null_list.append(Int32(i))

    if n_valid > 1:
        if ascending:
            if stable:

                @parameter
                def cmp_asc_s(a: Int32, b: Int32) -> Bool:
                    return array.unsafe_get(UInt(a)) < array.unsafe_get(UInt(b))

                _sort_impl[cmp_asc_s, stable=True](valid_list)
            else:

                @parameter
                def cmp_asc(a: Int32, b: Int32) -> Bool:
                    return array.unsafe_get(UInt(a)) < array.unsafe_get(UInt(b))

                _sort_impl[cmp_asc](valid_list)
        else:
            if stable:

                @parameter
                def cmp_desc_s(a: Int32, b: Int32) -> Bool:
                    return array.unsafe_get(UInt(b)) < array.unsafe_get(UInt(a))

                _sort_impl[cmp_desc_s, stable=True](valid_list)
            else:

                @parameter
                def cmp_desc(a: Int32, b: Int32) -> Bool:
                    return array.unsafe_get(UInt(b)) < array.unsafe_get(UInt(a))

                _sort_impl[cmp_desc](valid_list)

    var out = Buffer.alloc_uninit[DType.int32](n)
    var ov = out.view[DType.int32](0, n)
    var n_null_ = len(null_list)
    var null_off = 0 if nulls_first else n_valid
    var valid_off = n_null_ if nulls_first else 0
    for i in range(n_null_):
        ov.unsafe_set(null_off + i, null_list[i])
    for i in range(n_valid):
        ov.unsafe_set(valid_off + i, valid_list[i])

    return Int32Array(
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=out^.to_immutable(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def argsort(
    array: AnyArray,
    ascending: Bool = True,
    nulls_first: Bool = True,
    stable: Bool = False,
    limit: Optional[Int] = None,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> Int32Array:
    """Return indices that would sort ``array``.

    Args:
        array: Input array (runtime-typed).
        ascending: Sort direction. ``True`` = smallest first.
        nulls_first: Where to place null elements in the output.
        stable: Preserve relative order of equal elements.
        limit: If set, return only the first ``limit`` indices (top-K).
            Phase 1: implemented as full sort + truncation. Phase 3 will
            add O(N) quickselect.
        ctx: Execution context — controls CPU thread count and GPU device.

    Returns:
        Int32Array of sorted row indices (length = min(limit, len(array))).
    """
    var result: Int32Array

    if array.dtype() == bool_dt:
        result = _argsort_bool(array.as_bool().copy(), ascending, nulls_first)
    elif array.dtype() == int8:
        result = _argsort_primitive(
            array.as_int8(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype() == int16:
        result = _argsort_primitive(
            array.as_int16(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype() == int32:
        result = _argsort_primitive(
            array.as_int32(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype() == int64:
        result = _argsort_primitive(
            array.as_int64(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype() == uint8:
        result = _argsort_primitive(
            array.as_uint8(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype() == uint16:
        result = _argsort_primitive(
            array.as_uint16(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype() == uint32:
        result = _argsort_primitive(
            array.as_uint32(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype() == uint64:
        result = _argsort_primitive(
            array.as_uint64(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype() == float16:
        result = _argsort_primitive(
            array.as_float16(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype() == float32:
        result = _argsort_primitive(
            array.as_float32(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype() == float64:
        result = _argsort_primitive(
            array.as_float64(), ascending, nulls_first, stable, ctx
        )
    elif array.dtype().is_string():
        result = _argsort_string(
            array.as_string(), ascending, nulls_first, stable
        )
    else:
        raise Error(t"argsort: unsupported dtype {array.dtype()}")

    if limit:
        var k = min(limit.value(), len(result))
        return Int32Array(
            length=k,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=result.buffer,
        )
    return result^


def sort(
    array: StructArray,
    key_indices: List[Int],
    ascending: List[Bool],
    nulls_first: Bool = True,
    stable: Bool = False,
    limit: Optional[Int] = None,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> StructArray:
    """Sort a StructArray by the specified key columns.

    Phase 1: single key column only (``len(key_indices) == 1``).
    Phase 2 will add multi-column permutation refinement.

    Args:
        array: Input StructArray.
        key_indices: Column indices to sort by (Phase 1: exactly one).
        ascending: Per-key sort direction.
        nulls_first: Where to place null rows.
        stable: Preserve relative order of equal rows.
        limit: If set, return only the first ``limit`` rows.
        ctx: Execution context.

    Returns:
        A new StructArray with rows in sorted order.
    """
    if len(key_indices) == 0:
        raise Error("sort: key_indices must not be empty")
    if len(key_indices) != len(ascending):
        raise Error("sort: key_indices and ascending must have the same length")
    if len(key_indices) > 1:
        raise Error(
            "sort: multi-column sort not yet implemented (Phase 2); "
            "pass a single key_indices element"
        )

    var key_col = array.field(key_indices[0])
    var indices = argsort(
        key_col, ascending[0], nulls_first, stable, limit, ctx
    )
    return _take(array, indices)
