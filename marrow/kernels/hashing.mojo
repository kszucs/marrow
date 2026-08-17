"""Hashing kernels for Arrow arrays.

Provides column-wise hash computation for use in groupby, joins, and
partitioning. Follows the DuckDB/DataFusion approach of hashing each
column independently and combining hashes across columns.

Public API — the ``RapidHashKernel`` kernel:
  - ``RapidHashKernel.apply``: typed leaves
    - BoolArray: vectorized via precomputed hash + SIMD select
    - PrimitiveArray[T]: vectorized rapidhash (SIMD via elementwise); values
      wider than 64 bits (decimal128/256) fold their 64-bit limbs
    - BinaryLikeArray[T]: per-element AHash (variable-length fallback)
    - StructArray: per-column hash with combining (multi-key)
    - ListLikeArray[T] / FixedSizeListArray: fold the child hashes per row
  - ``RapidHashKernel.dispatch``: runtime-typed dispatch, routed through the
    ``DynType.dispatch_*`` family rather than a hand-written dtype ladder.
    Temporal, interval, and decimal32/64 columns are hashed through their
    typed leaf (bound on ``PrimitiveType``); dictionary columns through their
    decoded values,
    so a dictionary-encoded key hashes identically to the plain column.
  - ``rapidhash``: thin free delegators (PyArrow has no equivalent, but the
    hash-table parameter ``SwissHashTable[hasher]`` binds one as a value).

Rapidhash port follows the C reference at https://github.com/Nicoshev/rapidhash
"""


from std.sys import size_of

from ..arrays import (
    BoolArray,
    BinaryLikeArray,
    PrimitiveArray,
    StructArray,
    ListLikeArray,
    FixedSizeListArray,
    DynArray,
    UInt64Array,
    Int32Array,
)
from ..builders import UInt64Builder, Int32Builder
from ..buffers import Buffer
from ..views import apply
from .filter import take
from .core import Kernel
from ..utils import AHash64, Hasher, RapidHash64, XxHash64
from ..execution import ExecContext, GPU_ENABLED
from ..dtypes import (
    BinaryLikeType,
    IntegerType,
    PrimitiveType,
    ListLikeType,
    bool_,
    int32,
    uint64,
)


def _indices_as_int32(indices: DynArray) raises -> Int32Array:
    """Dictionary indices as `Int32Array`, without reaching `kernels.cast`.

    Arrow allows any signed or unsigned integer as a dictionary index, and
    `take` wants int32. `cast` would do this, but importing it for the
    conversion is what made `kernels.cast` reachable from every binary that
    hashes (Q4.7). Narrow integer widening does not need the cast machinery.
    """
    var dt = indices.dtype()
    if dt == int32:
        return indices.as_int32().copy()

    def widen[T: IntegerType](d: T) raises {imm} -> Int32Array:
        ref src = indices.as_primitive[T]()
        var out = Int32Builder(len(src))
        for i in range(len(src)):
            if src.is_valid(i):
                out.append(Scalar[int32.native](Int(src.unsafe_get(i))))
            else:
                out.append_null()
        return out.finish()

    return dt.dispatch_integer(widen)


comptime NULL_HASH_SENTINEL = UInt64(0x517CC1B727220A95)
"""Fixed hash value used for null elements."""


# ---------------------------------------------------------------------------
# rapidhash — vectorized hash for primitive arrays (SIMD via elementwise)
struct HashKernel[H: Hasher](Kernel):
    """Column hashing kernel — one ``UInt64`` per row, under the hash `H`.

    `H` is comptime, so every call below resolves to a direct `H.hash_lanes` /
    `H.hash` and the generated code is what a hand-written kernel for that one
    algorithm would be. `RapidHashKernel` is the alias the tree uses; `XxHashKernel`
    and `AHashKernel` exist so the choice is a parameter rather than a rewrite.

    The typed leaves are the ``apply`` overloads; ``dispatch`` resolves a
    runtime-typed array to the matching leaf via the ``DynType.dispatch_*`` family
    rather than a per-dtype ladder, so adding a dtype to a family covers it
    without touching this file. Null elements hash to ``NULL_HASH_SENTINEL`` so
    that "null == null" holds for grouping and joining (Arrow's `hash_*`
    semantics).

    Not hashable: `null` and `fixed_size_binary` (no leaf yet), and the
    month-day-nano interval (>64 bits and outside the decimal family).
    """

    comptime name = Self.H.name

    # --- lane helpers ---------------------------------------------------
    #
    # Static methods rather than free functions: they are meaningless without
    # `H`, and `apply` accepts `Self._x[...]` as its comptime function exactly
    # as `BinaryKernel.apply` passes `Self.core[native, _]`. The two that merely
    # forwarded to the trait (`_hash_bits`, `_combine_hashes`) are gone — the
    # call sites say `Self.H.hash_lanes` / `Self.H.combine_lanes` directly.

    @staticmethod
    @always_inline
    def _bool_lanes[
        W: Int
    ](bits: SIMD[DType.bool, W]) -> SIMD[uint64.native, W]:
        """Select between the two precomputed bool digests."""
        comptime bw = size_of[Scalar[bool_.native]]()
        comptime h_false = Self.H.hash_lanes[bw, 1](SIMD[uint64.native, 1](0))[
            0
        ]
        comptime h_true = Self.H.hash_lanes[bw, 1](SIMD[uint64.native, 1](1))[0]
        return bits.select(
            SIMD[uint64.native, W](h_true), SIMD[uint64.native, W](h_false)
        )

    @staticmethod
    @always_inline
    def _bool_lanes_masked[
        W: Int
    ](bits: SIMD[DType.bool, W], valid: SIMD[DType.bool, W]) -> SIMD[
        uint64.native, W
    ]:
        return valid.select(
            Self._bool_lanes[W](bits),
            SIMD[uint64.native, W](NULL_HASH_SENTINEL),
        )

    @staticmethod
    @always_inline
    def _primitive_lanes[
        T: PrimitiveType, W: Int
    ](vals: SIMD[T.native, W]) -> SIMD[uint64.native, W]:
        """Widen a primitive lane to uint64 and hash it."""
        comptime byte_width = size_of[Scalar[T.native]]()

        comptime if byte_width <= 8:
            # Zero-extend (matches C's rapid_read32/rapid_read64); mask to
            # byte_width bits so a narrow signed type does not sign-extend.
            comptime mask = ~UInt64(0) if byte_width >= 8 else (
                UInt64(1) << UInt64(byte_width * 8)
            ) - 1
            return Self.H.hash_lanes[byte_width, W](
                vals.cast[uint64.native]() & SIMD[uint64.native, W](mask)
            )
        else:
            # decimal128 / decimal256 and the month-day-nano interval do not fit
            # a uint64 lane: fold their 64-bit limbs so the high bits
            # participate. A truncating cast would collide every pair differing
            # only above bit 63 — and group-by buckets on the hash alone.
            comptime limbs = byte_width // 8
            var h = SIMD[uint64.native, W](0)
            comptime for i in range(limbs):
                var limb = (vals >> SIMD[T.native, W](i * 64)).cast[
                    uint64.native
                ]()
                h = Self.H.combine_lanes[W](h, Self.H.hash_lanes[8, W](limb))
            return h

    @staticmethod
    @always_inline
    def _primitive_lanes_masked[
        T: PrimitiveType, W: Int
    ](vals: SIMD[T.native, W], valid: SIMD[DType.bool, W]) -> SIMD[
        uint64.native, W
    ]:
        return valid.select(
            Self._primitive_lanes[T, W](vals),
            SIMD[uint64.native, W](NULL_HASH_SENTINEL),
        )

    @staticmethod
    def dispatch(
        keys: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Resolve `keys`'s runtime dtype and hash it."""
        var dt = keys.dtype()
        if dt == bool_:
            return Self.apply(keys.as_bool(), ctx)
        elif dt.is_binary_like():

            def binarylike[T: BinaryLikeType](d: T) raises {imm} -> UInt64Array:
                return Self.apply(keys.as_binary_like[T](), ctx)

            return dt.dispatch_binarylike(binarylike)
        elif dt.is_list_like():

            def listlike[T: ListLikeType](d: T) raises {imm} -> UInt64Array:
                return Self.apply(keys.as_list_like[T](), ctx)

            return dt.dispatch_listlike(listlike)
        elif dt.is_struct():
            return Self.apply(keys.as_struct(), ctx)
        elif dt.is_fixed_size_list():
            return Self.apply(keys.as_fixed_size_list(), ctx)
        elif dt.is_dictionary():
            # Hash the decoded values: a dictionary-encoded key must hash the
            # same as the equivalent plain column, otherwise two batches with
            # different dictionaries would never group together.
            #
            # Decoded by gather rather than through `cast`. `DictionaryCast`
            # does exactly this -- normalise the indices, `take` the values,
            # then a re-cast that is a no-op when the target *is* the value
            # type, which is the only way this called it. Reaching it imported
            # the whole `kernels.cast` fanout, ~797 symbols and roughly a fifth
            # of the fused binary, for one call site. Q4.7.
            ref d = keys.as_dictionary()
            return Self.dispatch(
                take(
                    d.dictionary().copy(), _indices_as_int32(d.indices()), ctx
                ),
                ctx,
            )
        elif dt.is_primitive():
            # One arm for every fixed-width type. `apply` is bound on
            # `PrimitiveType`, so numeric, temporal, interval and *all four*
            # decimal widths hash through the typed leaf directly — the hash
            # only reads the value bytes via `T.native`, never the logical
            # dtype. No reinterpret to an integer backing is needed.
            #
            # `Decimal128Array` is `PrimitiveArray[Decimal128Type]`, so the
            # separate numeric/decimal128/decimal256 arms this replaces were
            # three more spellings of this one call.
            def primitive[T: PrimitiveType](d: T) raises {imm} -> UInt64Array:
                return Self.apply(keys.as_primitive[T](), ctx)

            return dt.dispatch_primitive(primitive)
        else:
            raise Self.error(t"unsupported dtype {dt}")

    @staticmethod
    def apply(
        keys: BoolArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Vectorized rapidhash for bool arrays.

        Precomputes hash(false) and hash(true), loads data bits via the
        bitmap-mask pattern, and uses ``SIMD.select()`` for branchless dispatch.
        Null elements are replaced with ``NULL_HASH_SENTINEL`` inline.

        Parallelism is delegated to ``apply`` via the ``ExecContext`` —
        no per-kernel stripe logic here.
        """
        var n = len(keys)
        var buf: Buffer[mut=True]
        comptime if GPU_ENABLED:
            if ctx.is_gpu():
                buf = Buffer.alloc_device[uint64.native](ctx.device.value(), n)
            else:
                buf = Buffer.alloc_uninit[uint64.native](n)
        else:
            buf = Buffer.alloc_uninit[uint64.native](n)

        var dst = buf.view[uint64.native](0, n)
        var validity = keys.validity()
        if validity:
            apply[uint64.native, Self._bool_lanes_masked[...]](
                keys.values(),
                validity.value(),
                dst,
                ctx,
            )
        else:
            apply[uint64.native, Self._bool_lanes[...]](keys.values(), dst, ctx)

        return UInt64Array(
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf.to_immutable(),
        )

    # FIXME: use the seeding from the Rust implementation
    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        keys: PrimitiveArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Vectorized rapidhash for primitive arrays.

        Each SIMD lane independently computes the rapidhash of one element.
        Null elements are replaced with ``NULL_HASH_SENTINEL`` inline.

        Parallelism is handled uniformly by ``apply`` using the
        ``ExecContext`` — CPU vs GPU is picked from ``ctx.device``, and
        CPU stripe-parallelism is driven by ``ctx.num_threads``.

        Values wider than a uint64 lane (decimal128/decimal256) have no SIMD
        path; they run the same limb-folding core one element at a time.
        """
        var n = len(keys)

        comptime if size_of[Scalar[T.native]]() > 8:
            var values = keys.values()
            var builder = UInt64Builder(capacity=n)
            for i in range(n):
                if keys.is_valid(i):
                    builder.unsafe_append(
                        Self._primitive_lanes[T, 1](values.unsafe_get(i))
                    )
                else:
                    builder.unsafe_append(NULL_HASH_SENTINEL)
            return builder.finish()
        else:
            var buf: Buffer[mut=True]
            comptime if GPU_ENABLED:
                if ctx.is_gpu():
                    buf = Buffer.alloc_device[uint64.native](
                        ctx.device.value(), n
                    )
                else:
                    buf = Buffer.alloc_uninit[uint64.native](n)
            else:
                buf = Buffer.alloc_uninit[uint64.native](n)

            var dst = buf.view[uint64.native](0, n)
            var validity = keys.validity()
            if validity:
                apply[
                    T.native,
                    uint64.native,
                    Self._primitive_lanes_masked[T, ...],
                ](
                    keys.values(),
                    validity.value(),
                    dst,
                    ctx,
                )
            else:
                apply[T.native, uint64.native, Self._primitive_lanes[T, ...]](
                    keys.values(),
                    dst,
                    ctx,
                )

            return UInt64Array(
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=buf.to_immutable(),
            )

    @staticmethod
    def apply[
        T: BinaryLikeType
    ](
        keys: BinaryLikeArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Hash each element of a binary-like array (string, large_string,
        binary, large_binary).

        Hashed with `H` like every other leaf. This used to call
        `std.hashlib.hash` (aHash) regardless of `H`, because the multi-branch
        byte-string path did not exist; a string column and a numeric column
        were therefore hashed by different algorithms. `H.hash` is that path.

        Currently scalar-serial; parallelising variable-length hashing is future
        work — the ``ctx`` parameter exists for API consistency.
        """
        _ = ctx  # TODO: SIMD + parallel string hashing
        var n = len(keys)
        var builder = UInt64Builder(capacity=n)
        var has_bitmap = Bool(keys.bitmap)

        for i in range(n):
            if has_bitmap and not keys.bitmap.value().test(keys.offset + i):
                builder.unsafe_append(NULL_HASH_SENTINEL)
            else:
                builder.unsafe_append(
                    Self.H.hash(keys.unsafe_get(UInt(i)).as_bytes())
                )

        return builder.finish()

    @staticmethod
    def apply(
        keys: StructArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Hash a struct array by combining per-field hashes column-wise.

        Each field is hashed independently via ``dispatch`` and the results are
        combined element-wise using ``_combine_hashes``. The
        ``ExecContext`` is forwarded to per-field hashing and to the
        combine pass — all stripe-parallelism is handled inside ``apply``.

        A struct's own offset/length are *not* propagated to its children by the
        layout, so each field is sliced to the struct's row range first: hashing
        ``keys.slice(start, length)`` then covers exactly those rows.
        """
        var n = len(keys)
        var num_fields = len(keys.children)
        if num_fields == 0:
            raise Self.error("empty struct array")

        var result = Self.dispatch(keys.children[0].slice(keys.offset, n), ctx)

        for k in range(1, num_fields):
            var field_hashes = Self.dispatch(
                keys.children[k].slice(keys.offset, n), ctx
            )

            var buf: Buffer[mut=True]
            comptime if GPU_ENABLED:
                if ctx.is_gpu():
                    buf = Buffer.alloc_device[uint64.native](
                        ctx.device.value(), n
                    )
                else:
                    buf = Buffer.alloc_uninit[uint64.native](n)
            else:
                buf = Buffer.alloc_uninit[uint64.native](n)
            apply[uint64.native, uint64.native, Self.H.combine_lanes[...]](
                result.values(),
                field_hashes.values(),
                buf.view[uint64.native](0, n),
                ctx,
            )
            result = UInt64Array(
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=buf.to_immutable(),
            )

        return result^

    @staticmethod
    def apply[
        T: ListLikeType
    ](
        keys: ListLikeArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Hash a list/large-list/map array row-wise: hash the whole child once,
        then fold the child hashes over each row's element range (column-wise,
        no row-encoding). Null rows hash to ``NULL_HASH_SENTINEL``."""
        var n = len(keys)
        var child_hashes = Self.dispatch(keys.values().copy(), ctx)
        var builder = UInt64Builder(n)
        for i in range(n):
            if not keys.is_valid(i):
                builder.append(NULL_HASH_SENTINEL)
            else:
                var h = UInt64(0)
                var rng = keys.child_range(i)
                for j in range(rng[0], rng[1]):
                    h = Self.H.combine_lanes[1](h, child_hashes.unsafe_get(j))
                builder.append(h)
        return builder.finish()

    @staticmethod
    def apply(
        keys: FixedSizeListArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Hash a fixed-size-list array row-wise: hash the child once, fold each
        row's ``list_size`` child hashes. Null rows hash to
        ``NULL_HASH_SENTINEL``."""
        var n = len(keys)
        var size = keys.dtype.as_fixed_size_list().size
        var child_hashes = Self.dispatch(keys.values().copy(), ctx)
        var builder = UInt64Builder(n)
        for i in range(n):
            if not keys.is_valid(i):
                builder.append(NULL_HASH_SENTINEL)
            else:
                var h = UInt64(0)
                var base = (keys.offset + i) * size
                for j in range(size):
                    h = Self.H.combine_lanes[1](
                        h, child_hashes.unsafe_get(base + j)
                    )
                builder.append(h)
        return builder.finish()


comptime RapidHashKernel = HashKernel[RapidHash64]
"""The default column hash — rapidhash v3, and what every caller in the tree
uses. `HashKernel` is generic only so the choice is expressible."""

comptime XxHashKernel = HashKernel[XxHash64]
comptime AHashKernel = HashKernel[AHash64]
