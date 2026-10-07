# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Sort kernels — the `SortIndices` kernel and the `sort` / `sort_indices`
delegators.

Every sort is **stable by default**, as Arrow's `sort_indices` is: equal values
keep their input order unless the caller asks for `SortIndices[stable=False]`
(or `sort_indices[stable=False]`). Stability is a compile-time choice, so a
program links only the strategies its sorts use. NaN goes beside the nulls,
as in Arrow, unless `nan_largest` orders it above every number, as SQL does;
`-0.0` ties with `0.0`.

`SortIndices.apply` holds the typed leaves:
  - PrimitiveArray[T]: every value is encoded once into a UInt64 key, and
    `KeySort` orders the rows — PDQsort over composite (key, row) integers
    below the radix crossover, parallel LSD radix above. Both are stable;
    stability only moves the crossover (`_STABLE_RADIX_ROWS_PER_PASS` rows
    per pass when stable, `_RADIX_THRESHOLD` when not). Values wider than 64
    bits (decimal128/256) have no UInt64 key and take a comparison sort.
  - BoolArray: O(N) counting sort, stable by construction.
  - BytesArray (either string layout): stdlib comparison sort (bytewise
    lexicographic).

`SortIndices.dispatch` resolves a runtime dtype through the `DynType.dispatch_*`
family. Temporal, interval, and decimal32/64 columns sort through their
typed leaf (bound on `PrimitiveType`); dictionary columns sort by their
decoded values. `SortIndices.multi` composes single-column permutations into a
multi-key ordering, and `sort` is `take` under that permutation.

Measured throughput of the unstable path (int64, Apple M-series, parallel
where applicable):
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
from std.math import isnan
from std.sys import get_defined_int, size_of

from ..arrays import (
    BinaryLikeArray,
    BytesArray,
    BoolArray,
    PrimitiveArray,
    StructArray,
    DynArray,
    Int32Array,
)
from ..buffers import Buffer
from ..dtypes import (
    PrimitiveType,
    bool_ as bool_dt,
    Int32Type,
)
from .cast import decode_dictionary
from .core import Kernel
from .numeric import EqKernel
from ..codecs import OrderPreserving
from ..execution import ExecContext
from ..errors import InvalidError, TypeError
from .filter import TakeKernel
from .partition import radix_histogram


comptime _RADIX_THRESHOLD: Int = 32_768
"""An *unstable* sort below this size uses the packed PDQsort instead of LSD
radix.

Measured on Apple M-series (int64):
  PDQsort wins: N ≤ 16K (313 µs vs 438 µs radix at N=16K)
  Radix wins:   N ≥ 32K (578 µs vs 600 µs PDQsort at N=32K)
  Crossover: ~28K elements.
"""

# Measured on Apple M-series under load, random keys: int32 composite PDQsort
# 47 us vs radix 85 us at 4K, 190 us vs 166 us at 10K; float64 218 us vs
# 298 us at 10K. A low-cardinality key favours radix, which skips the passes
# whose bits never vary: the 2-key 10K multi sort is 278 us on radix, 611 us
# on the composite sort.
comptime _STABLE_RADIX_ROWS_PER_PASS: Int = get_defined_int[
    "MARROW_SORT_STABLE_RADIX_ROWS_PER_PASS", 2_048
]()
"""A *stable* sort uses LSD radix from this many rows per radix pass up, and a
PDQsort over composite (key, row) integers below: 6,144 rows for a 32-bit key,
12,288 for a 64-bit one. Radix is stable by construction and takes over far
earlier than it does from the unstable PDQsort, but each pass pays a
2,048-bucket histogram, so the crossover grows with the key width."""

comptime _PARALLEL_THRESHOLD: Int = 524_288
"""Minimum element count for parallel histogram/scatter in radix sort.

At N < 512K, the parallel dispatch overhead exceeds the speedup.
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
# KeySort — rows ordered by their encoded keys, independent of the dtype
# ---------------------------------------------------------------------------


struct KeySort[bits: Int]:
    """Orders `Int32` rows by `UInt64` keys whose order lives in their low
    `bits` bits, as `OrderPreserving` encodes them.

    Both algorithms answer the same, stable order: `radix` is stable by
    construction, and `packed` breaks every tie by the row index.
    """

    comptime passes = (Self.bits + _BITS_PER_PASS - 1) // _BITS_PER_PASS
    """Radix passes over a key, `_BITS_PER_PASS` bits each."""

    comptime Composite = DType.uint64 if Self.bits <= 32 else DType.uint128
    """A key with its row index in the low 32 bits."""

    @staticmethod
    @always_inline
    def pack(key: UInt64, row: Int32) -> Scalar[Self.Composite]:
        """`key` above `row`, so composites order by key, then by row."""
        # Bits above `bits` are alike in every key (the descending complement
        # sets them all), so shifting them out loses no order.
        return (key.cast[Self.Composite]() << 32) | row.cast[
            DType.uint32
        ]().cast[Self.Composite]()

    @staticmethod
    def packed(
        mut composites: List[Scalar[Self.Composite]],
        var rows: Buffer[mut=True],
    ) -> Buffer[]:
        """PDQsort over `pack`ed composites, unpacked into `rows`.

        No two composites tie, so the unstable PDQsort answers the stable
        order, comparing plain integers held inline.
        """
        var rv = rows.view[DType.int32](0, len(composites))
        _sort_impl(composites)
        for i in range(len(composites)):
            rv.unsafe_set(
                i, composites[i].cast[DType.uint32]().cast[DType.int32]()
            )
        return rows^.to_immutable()

    @staticmethod
    def radix(
        var keys: Buffer[mut=True],
        var rows: Buffer[mut=True],
        n: Int,
        ctx: ExecContext,
    ) -> Buffer[]:
        """LSD radix sort, `_BITS_PER_PASS` bits per pass.

        Each pass scatters (key, row) from one buffer pair into the other and
        swaps them; a pass whose bits are alike in every key is skipped. Above
        `_PARALLEL_THRESHOLD` rows the histogram and the scatter run per thread,
        into disjoint slots.
        """
        comptime bucket_count = 1 << _BITS_PER_PASS
        var keys_b = Buffer.alloc_uninit[DType.uint64](n)
        var rows_b = Buffer.alloc_uninit[DType.int32](n)

        # The histogram and the scatter below index `write_offsets` by stripe, so
        # both must stripe identically — same `ctx`, same `_PARALLEL_THRESHOLD`.
        for pass_ in range(Self.passes):
            var shift = UInt64(pass_ * _BITS_PER_PASS)
            # The last pass may cover fewer than _BITS_PER_PASS bits.
            var bits_this_pass = min(
                Self.bits - pass_ * _BITS_PER_PASS, _BITS_PER_PASS
            )
            var mask = UInt64((1 << bits_this_pass) - 1)

            var kh = keys.view[DType.uint64](0, n)

            def bucket_of(i: Int) {imm} -> Int:
                return Int((kh.unsafe_get(i) >> shift) & mask)

            var offsets = radix_histogram(
                n, bucket_count, bucket_of, ctx, _PARALLEL_THRESHOLD
            )
            var write_offsets = offsets[0].copy()
            ref bucket_start = offsets[1]

            # One non-empty bucket: every key shares these bits, so the pass
            # would not move anything.
            var non_zero = 0
            for b in range(bucket_count):
                if bucket_start[b + 1] > bucket_start[b]:
                    non_zero += 1
                    if non_zero > 1:
                        break
            if non_zero <= 1:
                continue

            var ka = keys.view[DType.uint64](0, n)
            var ra = rows.view[DType.int32](0, n)
            var kb = keys_b.view[DType.uint64](0, n)
            var rb = rows_b.view[DType.int32](0, n)

            @always_inline
            def scatter_worker(
                t: Int, start: Int, end: Int
            ) {mut write_offsets, imm}:
                var base = t * bucket_count
                for i in range(start, end):
                    var key = ka.unsafe_get(i)
                    var b = Int((key >> shift) & mask)
                    var pos = write_offsets[base + b]
                    kb.unsafe_set(pos, key)
                    rb.unsafe_set(pos, ra.unsafe_get(i))
                    write_offsets[base + b] = pos + 1

            ctx.stripe(n, scatter_worker, _PARALLEL_THRESHOLD)

            var tmp_keys = keys^
            keys = keys_b^
            keys_b = tmp_keys^
            var tmp_rows = rows^
            rows = rows_b^
            rows_b = tmp_rows^

        return rows^.to_immutable()


struct SortIndices[stable: Bool = True, nan_largest: Bool = False](Kernel):
    """Sort-permutation kernel — the indices that would sort a column.

    `stable` keeps equal values in input order; `SortIndices[stable=False]`
    may reorder them; a primitive column differs only in the size at which
    it switches to radix.

    `nan_largest` orders NaN as a value above every number — after them
    ascending, before them descending, with the nulls placed on their own —
    which is SQL's order and the expression layer's. Without it NaN goes
    beside the nulls, which is Arrow's.

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
        nulls_first: Bool = False,
        limit: Optional[Int] = None,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int32Array:
        """Return the indices that would sort ``array``.

        Args:
            array: Input array (runtime-typed).
            ascending: Sort direction. ``True`` = smallest first.
            nulls_first: Where to place null elements in the output.
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
            result = Self.apply(array.as_bool(), ascending, nulls_first, ctx)
        elif dt.is_binary_like() or dt.is_string_view() or dt.is_binary_view():

            def bytes_leaf[A: BytesArray](arr: A) raises {imm} -> Int32Array:
                return Self.apply(arr, ascending, nulls_first, ctx)

            result = array.dispatch_bytes(bytes_leaf)
        elif dt.is_dictionary():
            # Order by the *decoded* values: dictionary index order is an
            # encoding artefact (`ordered=False` is the norm), not a value
            # order. `decode_dictionary`, not `cast`: naming the top-level
            # `cast` links its whole ladder into any binary that sorts
            # *anything*, for a dictionary path most plans never take.
            result = Self.dispatch(
                decode_dictionary(array.as_dictionary(), ctx),
                ascending,
                nulls_first,
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
                return Self.apply(
                    array.as_primitive[T](), ascending, nulls_first, ctx
                )

            result = dt.dispatch_primitive(primitive)
        else:
            raise Self.error[TypeError](t"unsupported dtype {dt}")

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
        nulls_first: Bool = False,
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
            limit: If set, return only the first ``limit`` indices.
            ctx: Execution context.

        `stable` decides only a single key: with several, every pass is
        stable, since the earlier passes survive only as tie-breakers.
        """
        if len(key_indices) == 0:
            raise Self.error[InvalidError]("key_indices must not be empty")
        if len(key_indices) != len(ascending):
            raise Self.error[InvalidError](
                "key_indices and ascending must have the same length",
            )

        if len(key_indices) == 1:
            return Self.dispatch(
                array.field(key_indices[0]),
                ascending[0],
                nulls_first,
                limit,
                ctx,
            )

        # Multi-column, column-oriented LSD: stable-sort by the least-significant
        # key first, then successively by more-significant keys. Each pass sorts
        # one reordered column and gathers to compose the running permutation;
        # every pass is stable, so a less-significant key's order is preserved as
        # the tie-break under a more-significant one.
        var last = len(key_indices) - 1
        comptime Pass = SortIndices[stable=True, nan_largest=Self.nan_largest]
        var perm = Pass.dispatch(
            array.field(key_indices[last]),
            ascending=ascending[last],
            nulls_first=nulls_first,
            ctx=ctx,
        )
        for i in reversed(range(last)):
            var reordered = TakeKernel.dispatch(
                array.field(key_indices[i]), perm
            )
            var local = Pass.dispatch(
                reordered,
                ascending=ascending[i],
                nulls_first=nulls_first,
                ctx=ctx,
            )
            perm = TakeKernel.apply(perm, local, ctx)

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
        nulls_first: Bool = False,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int32Array:
        """Return the indices that would sort a typed primitive array.

        Unless `nan_largest`, NaN rows are set aside with the nulls and land
        between them and the values, in input order — just after the nulls
        with `nulls_first`, just before them otherwise, whichever the
        direction. That is Arrow's placement: NaN is not ordered against a
        number, so it goes to the end the caller chose for values without an
        order.
        """
        var n = len(arr)
        if n == 0:
            return Int32Array.empty(Int32Type())

        var n_null = arr.null_count()

        # Partition: collect valid, NaN and null original row indices in one
        # scan.
        var null_list = List[Int32](capacity=max(n_null, 1))
        var nan_list = List[Int32]()
        var valid_buf = Buffer.alloc_uninit[DType.int32](max(n - n_null, 1))
        var vv = valid_buf.view[DType.int32](0, n - n_null)
        var values = arr.values()
        var vi = 0
        for i in range(n):
            if not arr.is_valid(i):
                null_list.append(Int32(i))
            else:
                comptime if (
                    T.native.is_floating_point() and not Self.nan_largest
                ):
                    if isnan(values.unsafe_get(i)):
                        nan_list.append(Int32(i))
                        continue
                vv.unsafe_set(vi, Int32(i))
                vi += 1

        var sorted_valid = Self._sort_valid[T](
            arr, valid_buf^, vi, ascending, ctx
        )
        return Self._assemble(
            sorted_valid, vi, null_list, nan_list, n, nulls_first
        )

    @staticmethod
    def _sort_valid[
        T: PrimitiveType
    ](
        arr: PrimitiveArray[T],
        var valid_buf: Buffer[mut=True],
        n_valid: Int,
        ascending: Bool,
        ctx: ExecContext,
    ) raises -> Buffer[]:
        """Order the `n_valid` non-null row indices in `valid_buf` -- NaN
        rows among them only under `nan_largest`.

        Values of at most 64 bits are encoded once into `UInt64` keys and
        ordered by `KeySort`: LSD radix from the crossover up, the packed
        PDQsort below it. Both are stable, so `stable` only moves the
        crossover. Decimal128/256 exceed a `UInt64` key and take
        `_comparison_sort`.
        """
        if n_valid <= 1:
            return valid_buf^.to_immutable()

        comptime if size_of[Scalar[T.native]]() <= 8:
            comptime Sorter = KeySort[size_of[Scalar[T.native]]() * 8]
            comptime radix_from = (
                _STABLE_RADIX_ROWS_PER_PASS * Sorter.passes
            ) if Self.stable else _RADIX_THRESHOLD
            var values = arr.values()
            var rows = valid_buf.view[DType.int32](0, n_valid)

            @always_inline
            def key_of(i: Int) {imm} -> UInt64:
                return Self._key[T](
                    values.unsafe_get(Int(rows.unsafe_get(i))), ascending
                )

            if n_valid >= radix_from:
                var keys = Buffer.alloc_uninit[DType.uint64](n_valid)
                var kv = keys.view[DType.uint64](0, n_valid)
                for i in range(n_valid):
                    kv.unsafe_set(i, key_of(i))
                return Sorter.radix(keys^, valid_buf^, n_valid, ctx)
            var composites = List[Scalar[Sorter.Composite]](capacity=n_valid)
            for i in range(n_valid):
                composites.append(Sorter.pack(key_of(i), rows.unsafe_get(i)))
            return Sorter.packed(composites, valid_buf^)
        else:
            return Self._comparison_sort[T](arr, valid_buf^, n_valid, ascending)

    @staticmethod
    @always_inline
    def _key[
        T: PrimitiveType
    ](val: Scalar[T.native], ascending: Bool) -> UInt64:
        """`val`'s sort key, with -0.0 folded onto 0.0 so the two tie, and
        every NaN onto the one positive NaN, which encodes above `+inf`."""
        var key = OrderPreserving.encode_value(
            EqKernel[nan_safe=True].canonical[T.native, 1](val)
        )
        if ascending:
            return key
        else:
            return ~key

    @staticmethod
    def _comparison_sort[
        T: PrimitiveType
    ](
        src: PrimitiveArray[T],
        var idx_buf: Buffer[mut=True],
        n: Int,
        ascending: Bool,
    ) raises -> Buffer[]:
        """Sort `idx_buf` by comparing the values it indexes — for values
        wider than a `UInt64` key (decimal128/256).

        A stable sort takes the stdlib's merge sort, an unstable one PDQsort.
        `SortIndices.multi` depends on the former: its column-wise passes keep
        a less-significant key only if each pass is stable.
        """
        comptime assert (
            not T.native.is_floating_point()
        ), "a float has a UInt64 key"
        var values = src.values()
        var v = idx_buf.view[DType.int32](0, n)
        var idx_list = List[Int32](capacity=n)
        for i in range(n):
            idx_list.append(v.unsafe_get(i))
        if ascending:

            def cmp_asc(a: Int32, b: Int32) {imm values} -> Bool:
                return values.unsafe_get(Int(a)) < values.unsafe_get(Int(b))

            _sort_impl[stable=Self.stable](idx_list, cmp_asc)
        else:

            def cmp_desc(a: Int32, b: Int32) {imm values} -> Bool:
                return values.unsafe_get(Int(a)) > values.unsafe_get(Int(b))

            _sort_impl[stable=Self.stable](idx_list, cmp_desc)
        for i in range(n):
            v.unsafe_set(i, idx_list[i])
        return idx_buf^.to_immutable()

    @staticmethod
    def _assemble(
        sorted_valid: Buffer[],
        n_valid: Int,
        null_list: List[Int32],
        nan_list: List[Int32],
        n: Int,
        nulls_first: Bool,
    ) raises -> Int32Array:
        """Merge sorted valid, NaN and null indices into the final Int32Array:
        `[nulls, NaNs, values]` with `nulls_first`, else `[values, NaNs, nulls]`.
        """
        var out = Buffer.alloc_uninit[DType.int32](n)
        var ov = out.view[DType.int32](0, n)
        var sv = sorted_valid.view[DType.int32](0, n_valid)
        var n_null = len(null_list)
        var n_nan = len(nan_list)
        var null_off = 0 if nulls_first else n_valid + n_nan
        var nan_off = n_null if nulls_first else n_valid
        var valid_off = n_null + n_nan if nulls_first else 0
        for i in range(n_null):
            ov.unsafe_set(null_off + i, null_list[i])
        for i in range(n_nan):
            ov.unsafe_set(nan_off + i, nan_list[i])
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
        nulls_first: Bool = False,
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
    def _sort_bytes[
        A: BytesArray
    ](arr: A, mut rows: List[Int32], ascending: Bool):
        """Sort `rows` of `arr` bytewise, in two phases.

        First by each row's 8-byte `sort_key`, a comparison over one
        contiguous array of integers. Then each run of rows whose keys tie is
        sorted by its values. Rows that differ in their first eight bytes --
        the common case for short or varied values -- never reach the second
        phase; values sharing a long prefix, like URLs, all do, and cost one
        cheap pass over equal keys on top of the value sort.

        Two phases rather than one comparator that falls back on a tie: with
        the value comparison inlined the comparator was 3x slower even when
        no tie occurred, and with it out of line every tie paid a call.
        """
        var keys = List[UInt64](capacity=len(arr))
        for i in range(len(arr)):
            keys.append(arr.sort_key(i))

        def key_asc(a: Int32, b: Int32) {imm keys} -> Bool:
            return keys[Int(a)] < keys[Int(b)]

        def key_desc(a: Int32, b: Int32) {imm keys} -> Bool:
            return keys[Int(b)] < keys[Int(a)]

        def value_asc(a: Int32, b: Int32) {imm arr} -> Bool:
            return arr.unsafe_get(UInt(a)) < arr.unsafe_get(UInt(b))

        def value_desc(a: Int32, b: Int32) {imm arr} -> Bool:
            return arr.unsafe_get(UInt(b)) < arr.unsafe_get(UInt(a))

        if ascending:
            _sort_impl[stable=Self.stable](rows, key_asc)
        else:
            _sort_impl[stable=Self.stable](rows, key_desc)

        var start = 0
        while start < len(rows):
            var end = start + 1
            var key = keys[Int(rows[start])]
            while end < len(rows) and keys[Int(rows[end])] == key:
                end += 1
            if end - start > 1:
                var run = Span(rows)[start:end]
                if ascending:
                    _sort_impl[stable=Self.stable](run, value_asc)
                else:
                    _sort_impl[stable=Self.stable](run, value_desc)
            start = end

    @staticmethod
    def apply[
        A: BytesArray
    ](
        arr: A,
        ascending: Bool = True,
        nulls_first: Bool = False,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int32Array:
        """Comparison sort over byte strings in either layout, using the Mojo
        stdlib sort — bytewise lexicographic, which is Arrow's ordering for
        every binary and string type."""
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
            Self._sort_bytes(arr, valid_list, ascending)

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


def sort_indices[
    stable: Bool = True, nan_largest: Bool = False
](
    array: DynArray,
    ascending: Bool = True,
    nulls_first: Bool = False,
    limit: Optional[Int] = None,
    ctx: ExecContext = ExecContext.serial(),
) raises -> Int32Array:
    """Return the indices that would sort ``array``."""
    return SortIndices[stable, nan_largest].dispatch(
        array, ascending, nulls_first, limit, ctx
    )


def sort[
    stable: Bool = True, nan_largest: Bool = False
](
    array: StructArray,
    key_indices: List[Int],
    ascending: List[Bool],
    nulls_first: Bool = False,
    limit: Optional[Int] = None,
    ctx: ExecContext = ExecContext.serial(),
) raises -> StructArray:
    """Sort a StructArray by the specified key columns — ``take`` under the
    permutation from ``SortIndices.multi``."""
    return TakeKernel.apply(
        array,
        SortIndices[stable, nan_largest].multi(
            array, key_indices, ascending, nulls_first, limit, ctx
        ),
    )
