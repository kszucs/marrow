"""Sort kernels — the `SortIndices` kernel and the `sort` / `sort_indices`
delegators.

`SortIndices.apply` holds the typed leaves:
  - PrimitiveArray[T]: PDQsort (pair-based) for N < 32768; parallel LSD radix
    for N ≥ 32768. Values wider than 64 bits (decimal128/256) have no UInt64
    radix key and always take the comparison path.
  - BoolArray: O(N) counting sort.
  - BinaryLikeArray[T]: stdlib comparison sort (bytewise lexicographic).

`SortIndices.dispatch` resolves a runtime dtype through the `DynType.dispatch_*`
family. Temporal, interval, and decimal32/64 columns sort through their
typed leaf (bound on `PrimitiveType`); dictionary columns sort by their
decoded values. `SortIndices.multi` composes single-column permutations into a
multi-key ordering, and `sort` is `take` under that permutation.

Measured throughput (int64, Apple M-series, parallel where applicable):
  N=1K:    7.9 µs PDQsort  vs Polars  14.7 µs (1.9x faster)
  N=4K:   34.8 µs PDQsort  vs Polars  40.5 µs (1.2x faster)
  N=10K:  155 µs PDQsort   vs Polars  92 µs   (1.7x slower)
  N=100K: 1.87 ms serial   vs Polars 1.27 ms  (1.5x slower)
  N=1M:    6.2 ms parallel vs Polars 18.5 ms  (3.0x faster)
  N=10M:  39.5 ms parallel vs Polars 220 ms   (5.6x faster)

Serial cost breakdown at N=10M (from macOS `sample`, 50 iters, 8-bit baseline):
  scatter × passes   64.8%  ~170 ms  — random writes; primary bottleneck
  hist    × passes   29.0%   ~76 ms  — sequential reads, 8 passes
  take (gather)       2.6%    ~7 ms  — parallel SIMD gather
  assemble            0.9%    ~2 ms  — memcpy-style null placement
  dispatch/encode     2.7%    ~7 ms  — DynArray dispatch + encode loop
"""

from std.builtin.sort import sort as _sort_impl
from std.sys import size_of

from ..arrays import (
    BinaryLikeArray,
    BoolArray,
    PrimitiveArray,
    StructArray,
    DynArray,
    Int32Array,
)
from ..buffers import Buffer
from ..dtypes import (
    BinaryLikeType,
    PrimitiveType,
    bool_ as bool_dt,
    Int32Type,
)
from .cast import cast
from .core import Kernel
from ..execution import ExecContext
from .filter import Take
from .partition import radix_histogram


comptime _RADIX_THRESHOLD: Int = 32_768
"""Arrays below this size use pair-based PDQsort instead of LSD radix.

Measured on Apple M-series (int64):
  PDQsort wins: N ≤ 16K (313 µs vs 438 µs radix at N=16K)
  Radix wins:   N ≥ 32K (578 µs vs 600 µs PDQsort at N=32K)
  Crossover: ~28K elements.
"""

comptime _PARALLEL_THRESHOLD: Int = 524_288
"""Minimum element count for parallel histogram/scatter in radix sort.

At N < 512K, sync_parallelize thread-spawn overhead exceeds the speedup.
Serial radix at N=100K: ~2.6 ms; parallel: ~3.1 ms (20% slower due to overhead).
Parallel pays off above ~512K where the work per thread is large enough.
"""

comptime _BITS_PER_PASS: Int = 11
"""Radix width per pass. 11 bits → 2048-bucket histogram (16 KB per thread).

Measured at N=10M int64, Apple M-series, parallel:
  8-bit:  8 passes, 256-bucket hist  (2 KB/thread)   →  ~263 ms serial (baseline)
  11-bit: 6 passes, 2048-bucket hist (16 KB/thread)  →   ~39 ms parallel (~6.7x vs 8-bit serial)
  12-bit: 6 passes, 4096-bucket hist (32 KB/thread)  →  same pass count, larger hist
  16-bit: 4 passes, 65536-bucket hist (512 KB/thread) →  L1 thrash, slower

11-bit sweet spot: 25% fewer passes for 64-bit types (6 vs 8), histogram still
fits in L1 per thread; wider passes would thrash the 512 KB L2 shared cache.
"""


# ---------------------------------------------------------------------------
# Key encoding — value → UInt64 for radix sort comparisons
# ---------------------------------------------------------------------------


def _encode_sort_key[
    T: PrimitiveType
](val: Scalar[T.native], ascending: Bool,) -> UInt64:
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


struct _SortPair(Copyable, Movable, TrivialRegisterPassable):
    """(encoded_key, original_row_index) pair for comparison sort."""

    var key: UInt64
    var idx: Int32

    def __init__(out self, key: UInt64, idx: Int32):
        self.key = key
        self.idx = idx


def _comparison_sort_indices[
    T: PrimitiveType
](
    src: PrimitiveArray[T],
    var idx_buf: Buffer[mut=True],
    n: Int,
    ascending: Bool,
    stable: Bool,
) raises -> Buffer[]:
    """Sort idx_buf via Mojo stdlib PDQsort.

    Non-float types: sort 4-byte indices directly using `values[a] < values[b]`.
    No encoding, no extra struct. Covers integers, temporal, and decimal types.

    Float types: encoded (key, idx) pairs for NaN-safe total order.
    NaN → uint_max via bit-flip, sorts last (ascending) / first (descending).
    Direct float comparison is not used because `NaN < x` is always false in
    IEEE 754, which would corrupt PDQsort's invariants.

    PDQsort is **not** stable, so `stable` breaks ties on the original row index
    to make the comparator a total order. That costs one extra comparison per
    equal pair and no extra pass — cheaper than sorting twice, and it is what
    `SortIndices.multi` depends on: its column-wise LSD passes preserve a
    less-significant key only if each pass is stable. The radix path below is
    stable by construction and ignores this flag.
    """
    var values = src.values()
    var v = idx_buf.view[DType.int32](0, n)

    comptime if not T.native.is_floating_point():
        var idx_list = List[Int32](capacity=n)
        for i in range(n):
            idx_list.append(v.unsafe_get(i))
        if ascending:
            if stable:

                def cmp_asc_stable(a: Int32, b: Int32) {imm values} -> Bool:
                    var va = values.unsafe_get(Int(a))
                    var vb = values.unsafe_get(Int(b))
                    return va < vb or (va == vb and a < b)

                _sort_impl(idx_list, cmp_asc_stable)
            else:

                def cmp_asc(a: Int32, b: Int32) {imm values} -> Bool:
                    return values.unsafe_get(Int(a)) < values.unsafe_get(Int(b))

                _sort_impl(idx_list, cmp_asc)
        else:
            if stable:

                def cmp_desc_stable(a: Int32, b: Int32) {imm values} -> Bool:
                    var va = values.unsafe_get(Int(a))
                    var vb = values.unsafe_get(Int(b))
                    return va > vb or (va == vb and a < b)

                _sort_impl(idx_list, cmp_desc_stable)
            else:

                def cmp_desc(a: Int32, b: Int32) {imm values} -> Bool:
                    return values.unsafe_get(Int(a)) > values.unsafe_get(Int(b))

                _sort_impl(idx_list, cmp_desc)
        for i in range(n):
            v.unsafe_set(i, idx_list[i])
    else:
        var pairs = List[_SortPair](capacity=n)
        for i in range(n):
            var orig = v.unsafe_get(i)
            pairs.append(
                _SortPair(
                    key=_encode_sort_key[T](
                        values.unsafe_get(Int(orig)), ascending
                    ),
                    idx=orig,
                )
            )

        if stable:

            def cmp_pair_stable(a: _SortPair, b: _SortPair) -> Bool:
                return a.key < b.key or (a.key == b.key and a.idx < b.idx)

            _sort_impl(pairs, cmp_pair_stable)
        else:

            def cmp_pair(a: _SortPair, b: _SortPair) -> Bool:
                return a.key < b.key

            _sort_impl(pairs, cmp_pair)
        for i in range(n):
            v.unsafe_set(i, pairs[i].idx)

    return idx_buf^.to_immutable()


# ---------------------------------------------------------------------------
# LSD radix sort — O(N), parallel histogram + scatter
# ---------------------------------------------------------------------------


def _radix_sort_indices[
    T: PrimitiveType
](
    src: PrimitiveArray[T],
    var idx_buf: Buffer[mut=True],
    n: Int,
    ascending: Bool,
    ctx: ExecContext,
) raises -> Buffer[]:
    """LSD radix sort over encoded UInt64 keys using _BITS_PER_PASS-bit passes.

    Allocates two alternating (key, index) buffer pairs.  Each pass reads from
    pair-A and scatters into pair-B, then swaps A↔B.  The final result always
    resides in idx_buf (the current "A" after the last swap).

    Parallel path (ctx.wants_parallel): per-thread histograms → partition-major
    prefix sum → disjoint scatter slots, no atomics.
    Pass skipping: if histogram has only one non-zero bucket, the pass is a
    no-op (all elements share the same bits for that range) — free for random
    data, significant win for timestamps/sequential IDs.
    """
    comptime native = T.native
    comptime n_bits = size_of[Scalar[native]]() * 8
    # ceil(n_bits / _BITS_PER_PASS): 6 passes for 64-bit, 3 for 32-bit, etc.
    comptime num_passes = (n_bits + _BITS_PER_PASS - 1) // _BITS_PER_PASS
    comptime bucket_count = 1 << _BITS_PER_PASS  # 2048

    var values = src.values()

    # [< 1 ms] Three buffer allocs: key pair A/B + index pair B.
    var key_a = Buffer.alloc_uninit[DType.uint64](n)
    var key_b = Buffer.alloc_uninit[DType.uint64](n)
    var idx_b = Buffer.alloc_uninit[DType.int32](n)

    # [~2 ms parallel, ~7 ms serial at N=10M] Encode all valid elements into key_a.
    # Each value → UInt64 that sorts correctly in unsigned ascending order
    # (sign-flip for signed ints, IEEE 754 bit-flip for floats).
    var ka_init = key_a.view[DType.uint64](0, n)
    var ia_init = idx_buf.view[DType.int32](0, n)
    for i in range(n):
        ka_init.unsafe_set(
            i,
            _encode_sort_key[T](
                values.unsafe_get(Int(ia_init.unsafe_get(i))), ascending
            ),
        )

    # The histogram and the scatter below index `write_offsets` by stripe, so
    # both must stripe identically — same `ctx`, same `_PARALLEL_THRESHOLD`.

    # [~35 ms parallel total, 6 passes × (hist + scatter) at N=10M int64]
    for pass_ in range(num_passes):
        var shift = UInt64(pass_ * _BITS_PER_PASS)
        # Last pass may cover fewer than _BITS_PER_PASS bits.
        var bits_this_pass = min(
            n_bits - pass_ * _BITS_PER_PASS, _BITS_PER_PASS
        )
        var mask = UInt64((1 << bits_this_pass) - 1)

        # [~0.9 ms parallel per pass at N=10M] Per-thread histogram + prefix sum
        # (shared with the radix partitioner, cf. ``radix_histogram``).
        var ka_h = key_a.view[DType.uint64](0, n)

        def bucket_of(i: Int) {imm} -> Int:
            return Int((ka_h.unsafe_get(i) >> shift) & mask)

        var offsets = radix_histogram(
            n, bucket_count, bucket_of, ctx, _PARALLEL_THRESHOLD
        )
        var write_offsets = offsets[0].copy()
        ref bucket_start = offsets[1]

        # Pass skipping: if only one bucket is non-empty every element shares
        # these bits, so the data is already ordered for this pass — skip the
        # scatter. Free for random data, a big win for clustered / timestamp IDs.
        var non_zero = 0
        for b in range(bucket_count):
            if bucket_start[b + 1] > bucket_start[b]:
                non_zero += 1
                if non_zero > 1:
                    break
        if non_zero <= 1:
            continue

        # [~4.9 ms parallel per pass at N=10M] Parallel scatter.
        # Random writes into output buffer; cache-miss rate is the core bottleneck.
        var ka_s = key_a.view[DType.uint64](0, n)
        var ia_s = idx_buf.view[DType.int32](0, n)
        var kb_s = key_b.view[DType.uint64](0, n)
        var ib_s = idx_b.view[DType.int32](0, n)

        @always_inline
        def scatter_worker(
            t: Int, start: Int, end: Int
        ) {mut write_offsets, imm}:
            var base = t * bucket_count
            for i in range(start, end):
                var b = Int((ka_s.unsafe_get(i) >> shift) & mask)
                var pos = write_offsets[base + b]
                kb_s.unsafe_set(pos, ka_s.unsafe_get(i))
                ib_s.unsafe_set(pos, ia_s.unsafe_get(i))
                write_offsets[base + b] = pos + 1

        ctx.stripe(n, scatter_worker, _PARALLEL_THRESHOLD)

        # Swap A ↔ B so the next pass always reads from "current A".
        var tmp_key = key_a^
        key_a = key_b^
        key_b = tmp_key^
        var tmp_idx = idx_buf^
        idx_buf = idx_b^
        idx_b = tmp_idx^

    # Result is always in idx_buf (the current "A" after the final swap).
    return idx_buf^.to_immutable()


struct SortIndices(Kernel):
    """Sort-permutation kernel — the indices that would sort a column.

    The typed leaves are the ``apply`` overloads; ``dispatch`` resolves a
    runtime-typed array to the matching leaf via the ``DynType.dispatch_*`` family
    rather than a per-dtype ladder, so adding a dtype to a family covers it
    without touching this file. ``multi`` composes single-column permutations
    into a multi-key ordering.

    Every entry point returns *indices* — materializing the sorted values is
    ``take`` under the permutation, which is what the free ``sort`` does.

    Not sortable: the nested types (Arrow gives them no total order), `null`,
    and the month-day-nano interval (>64 bits and outside the decimal family).
    """

    comptime name = "sort_indices"

    @staticmethod
    def dispatch(
        array: DynArray,
        ascending: Bool = True,
        nulls_first: Bool = True,
        stable: Bool = False,
        limit: Optional[Int] = None,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int32Array:
        """Return the indices that would sort ``array``.

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
        var dt = array.dtype()
        var result: Int32Array

        if dt == bool_dt:
            result = SortIndices.apply(
                array.as_bool(), ascending, nulls_first, ctx
            )
        elif dt.is_binary_like():

            def binarylike[T: BinaryLikeType](d: T) raises {imm} -> Int32Array:
                return SortIndices.apply(
                    array.as_binary_like[T](),
                    ascending,
                    nulls_first,
                    stable,
                    ctx,
                )

            result = dt.dispatch_binarylike(binarylike)
        elif dt.is_dictionary():
            # Order by the *decoded* values: dictionary index order is an
            # encoding artefact (`ordered=False` is the norm), not a value order.
            result = SortIndices.dispatch(
                cast(array, dt.as_dictionary().value_type().copy(), False, ctx),
                ascending,
                nulls_first,
                stable,
                None,
                ctx,
            )
        elif dt.is_primitive():
            # One arm for every fixed-width type. `apply` is bound on
            # `PrimitiveType`, so numeric, temporal, interval and *all four*
            # decimal widths sort through the typed leaf directly — the sort
            # only ever reads `T.native` and validity, never the logical dtype.
            # No reinterpret to an integer backing is needed.
            #
            # `Decimal128Array` is `PrimitiveArray[Decimal128Type]`, so the
            # separate numeric/decimal128/decimal256 arms this replaces were
            # three more spellings of this one call.
            def primitive[T: PrimitiveType](d: T) raises {imm} -> Int32Array:
                return SortIndices.apply(
                    array.as_primitive[T](), ascending, nulls_first, stable, ctx
                )

            result = dt.dispatch_primitive(primitive)
        else:
            raise Self.error(t"unsupported dtype {dt}")

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

    @staticmethod
    def multi(
        array: StructArray,
        key_indices: List[Int],
        ascending: List[Bool],
        nulls_first: Bool = True,
        stable: Bool = False,
        limit: Optional[Int] = None,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int32Array:
        """The permutation that orders a StructArray by several key columns.

        Multi-column sort is done column-wise (no row comparator): LSD-style
        stable passes, least-significant key first — each pass is one
        ``dispatch`` over a single reordered column plus a gather to compose the
        permutation, so a later (more-significant) key sorts primarily while the
        earlier passes survive as tie-breakers via stability.

        Args:
            array: Input StructArray.
            key_indices: Column indices to sort by, most-significant first.
            ascending: Per-key sort direction.
            nulls_first: Where to place null rows.
            stable: Preserve relative order of equal rows (single-key path only;
                the multi-key path is always stable by construction).
            limit: If set, return only the first ``limit`` indices.
            ctx: Execution context.
        """
        if len(key_indices) == 0:
            raise Self.error("key_indices must not be empty")
        if len(key_indices) != len(ascending):
            raise Self.error(
                "key_indices and ascending must have the same length"
            )

        if len(key_indices) == 1:
            return SortIndices.dispatch(
                array.field(key_indices[0]),
                ascending[0],
                nulls_first,
                stable,
                limit,
                ctx,
            )

        # Multi-column, column-oriented LSD: stable-sort by the least-significant
        # key first, then successively by more-significant keys. Each pass sorts
        # one reordered column and gathers to compose the running permutation;
        # every pass is stable, so a less-significant key's order is preserved as
        # the tie-break under a more-significant one.
        var last = len(key_indices) - 1
        var perm = SortIndices.dispatch(
            array.field(key_indices[last]),
            ascending=ascending[last],
            nulls_first=nulls_first,
            stable=True,
            ctx=ctx,
        )
        for i in reversed(range(last)):
            var reordered = Take.dispatch(array.field(key_indices[i]), perm)
            var local = SortIndices.dispatch(
                reordered,
                ascending=ascending[i],
                nulls_first=nulls_first,
                stable=True,
                ctx=ctx,
            )
            perm = Take.apply(perm, local, ctx)

        if limit:
            var lim = limit.value()
            if lim < len(perm):
                perm = perm.slice(0, lim)
        return perm^

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        arr: PrimitiveArray[T],
        ascending: Bool = True,
        nulls_first: Bool = True,
        stable: Bool = False,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int32Array:
        """Return the indices that would sort a typed primitive array."""
        var n = len(arr)
        if n == 0:
            return Int32Array.empty(Int32Type())

        var n_null = arr.null_count()
        var n_valid = n - n_null

        # Partition: collect valid and null original row indices in one scan.
        var null_list = List[Int32](capacity=max(n_null, 1))
        var valid_buf = Buffer.alloc_uninit[DType.int32](max(n_valid, 1))
        var vv = valid_buf.view[DType.int32](0, n_valid)
        var vi = 0
        for i in range(n):
            if arr.is_valid(i):
                vv.unsafe_set(vi, Int32(i))
                vi += 1
            else:
                null_list.append(Int32(i))

        var sorted_valid = SortIndices._sort_valid[T](
            arr, valid_buf^, n_valid, ascending, stable, ctx
        )
        return SortIndices._assemble(
            sorted_valid, n_valid, null_list, n, nulls_first
        )

    @staticmethod
    def _sort_valid[
        T: PrimitiveType
    ](
        arr: PrimitiveArray[T],
        var valid_buf: Buffer[mut=True],
        n_valid: Int,
        ascending: Bool,
        stable: Bool,
        ctx: ExecContext,
    ) raises -> Buffer[]:
        """Order the `n_valid` non-null row indices held in `valid_buf`.

        A stable request routes to radix wherever a radix key exists, because
        LSD radix is stable *by construction* and costs nothing to make so.
        Teaching the comparison path stability instead means a tie-break on the
        original index, which turns one compare into a compare-plus-branch that
        the predictor cannot learn — measured at **+85%** on a 10k two-key sort
        with a low-cardinality leading key (320 µs -> 594 µs), the exact shape a
        multi-key ORDER BY produces. `_RADIX_THRESHOLD` is tuned for the
        *unstable* comparison sort and does not apply here.

        Only decimal128/256 still need the stable comparator: they exceed the
        UInt64 radix key, so no radix path exists for them at any size.
        """
        if n_valid <= 1:
            return valid_buf^.to_immutable()

        comptime if size_of[Scalar[T.native]]() <= 8:
            if stable or n_valid >= _RADIX_THRESHOLD:
                return _radix_sort_indices[T](
                    arr, valid_buf^, n_valid, ascending, ctx
                )
            else:
                return _comparison_sort_indices[T](
                    arr, valid_buf^, n_valid, ascending, stable
                )
        else:
            # decimal128 / decimal256 exceed the UInt64 radix key, so the
            # comparison path — which compares the native values directly — is
            # the only correct one.
            return _comparison_sort_indices[T](
                arr, valid_buf^, n_valid, ascending, stable
            )

    @staticmethod
    def _assemble(
        sorted_valid: Buffer[],
        n_valid: Int,
        null_list: List[Int32],
        n: Int,
        nulls_first: Bool,
    ) raises -> Int32Array:
        """Merge sorted valid indices and null indices into the final Int32Array.
        """
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

    @staticmethod
    def apply(
        arr: BoolArray,
        ascending: Bool = True,
        nulls_first: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int32Array:
        """O(N) counting sort for bool arrays.

        Enumerates indices into three bins (null, false, true) in one pass,
        then assembles the output in the requested order.
        """
        var n = len(arr)
        if n == 0:
            return Int32Array.empty(Int32Type())

        var null_list = List[Int32]()
        var false_list = List[Int32]()
        var true_list = List[Int32]()
        var bv = arr.values()  # offset-adjusted BitmapView for data bits
        for i in range(n):
            if not arr.is_valid(i):
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

    @staticmethod
    def apply[
        T: BinaryLikeType
    ](
        arr: BinaryLikeArray[T],
        ascending: Bool = True,
        nulls_first: Bool = True,
        stable: Bool = False,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int32Array:
        """Comparison sort for binary-like arrays (string, large_string, binary,
        large_binary) using the Mojo stdlib sort — bytewise lexicographic, which
        is Arrow's ordering for all four."""
        var n = len(arr)
        if n == 0:
            return Int32Array.empty(Int32Type())

        var n_null = arr.null_count()
        var n_valid = n - n_null

        var valid_list = List[Int32](capacity=max(n_valid, 1))
        var null_list = List[Int32](capacity=max(n_null, 1))
        for i in range(n):
            if arr.is_valid(i):
                valid_list.append(Int32(i))
            else:
                null_list.append(Int32(i))

        if n_valid > 1:

            def cmp_asc(a: Int32, b: Int32) {imm arr} -> Bool:
                return arr.unsafe_get(UInt(a)) < arr.unsafe_get(UInt(b))

            def cmp_desc(a: Int32, b: Int32) {imm arr} -> Bool:
                return arr.unsafe_get(UInt(b)) < arr.unsafe_get(UInt(a))

            if ascending:
                if stable:
                    _sort_impl[stable=True](valid_list, cmp_asc)
                else:
                    _sort_impl(valid_list, cmp_asc)
            else:
                if stable:
                    _sort_impl[stable=True](valid_list, cmp_desc)
                else:
                    _sort_impl(valid_list, cmp_desc)

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
# Public API — the two `pc.*`-style entry points (``pc.sort_indices`` /
# ``Table.sort_by``). Three typed `sort_indices` overloads used to sit beside
# them forwarding to `SortIndices.apply`, which is itself typed — so they saved
# no copy and only gave the kernel a second place to be taught a new array type.
# They had already drifted: the `BoolArray` one silently dropped `stable` and
# `limit` from its signature. Call `SortIndices.apply` for a typed array.
# ---------------------------------------------------------------------------


def sort_indices(
    array: DynArray,
    ascending: Bool = True,
    nulls_first: Bool = True,
    stable: Bool = False,
    limit: Optional[Int] = None,
    ctx: ExecContext = ExecContext.serial(),
) raises -> Int32Array:
    """Return the indices that would sort ``array``."""
    return SortIndices.dispatch(
        array, ascending, nulls_first, stable, limit, ctx
    )


def sort(
    array: StructArray,
    key_indices: List[Int],
    ascending: List[Bool],
    nulls_first: Bool = True,
    stable: Bool = False,
    limit: Optional[Int] = None,
    ctx: ExecContext = ExecContext.serial(),
) raises -> StructArray:
    """Sort a StructArray by the specified key columns — ``take`` under the
    permutation from ``SortIndices.multi``."""
    return Take.apply(
        array,
        SortIndices.multi(
            array, key_indices, ascending, nulls_first, stable, limit, ctx
        ),
    )
