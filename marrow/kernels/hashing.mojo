"""Hashing kernels for Arrow arrays.

Provides column-wise hash computation for use in groupby, joins, and
partitioning. Follows the DuckDB/DataFusion approach of hashing each
column independently and combining hashes across columns.

Public API — the ``RapidHash`` kernel:
  - ``RapidHash.apply``: typed leaves
    - BoolArray: vectorized via precomputed hash + SIMD select
    - PrimitiveArray[T]: vectorized rapidhash (SIMD via elementwise); values
      wider than 64 bits (decimal128/256) fold their 64-bit limbs
    - BinaryLikeArray[T]: per-element AHash (variable-length fallback)
    - StructArray: per-column hash with combining (multi-key)
    - ListLikeArray[T] / FixedSizeListArray: fold the child hashes per row
  - ``RapidHash.dispatch``: runtime-typed dispatch, routed through the
    ``AnyDataType.dispatch_*`` family rather than a hand-written dtype ladder.
    Temporal, interval, and decimal32/64 columns are hashed through their
    integer ``storage_type()``; dictionary columns through their decoded values,
    so a dictionary-encoded key hashes identically to the plain column.
  - ``rapidhash``: thin free delegators (PyArrow has no equivalent, but the
    hash-table parameter ``SwissHashTable[hasher]`` binds one as a value).

Rapidhash port follows the C reference at https://github.com/Nicoshev/rapidhash
"""

from std.gpu.host import DeviceContext
from std.hashlib import hash as _hash
from std.sys import size_of

from ..arrays import (
    BoolArray,
    BinaryLikeArray,
    PrimitiveArray,
    StructArray,
    ListLikeArray,
    FixedSizeListArray,
    AnyArray,
    UInt64Array,
)
from ..builders import UInt64Builder
from ..buffers import Buffer
from ..views import apply
from .aggregate import reinterpret_array
from .cast import cast
from .execution import ExecutionContext
from .helpers import Kernel
from ..dtypes import (
    BinaryLikeType,
    NumericType,
    PrimitiveType,
    ListLikeType,
    bool_,
    uint64,
)

comptime _h = Scalar[uint64.native]

comptime NULL_HASH_SENTINEL = UInt64(0x517CC1B727220A95)
"""Fixed hash value used for null elements."""


# ---------------------------------------------------------------------------
# Rapidhash primitives — ported from rapidhash.h (v3)
# https://github.com/Nicoshev/rapidhash
# ---------------------------------------------------------------------------


comptime RAPID_SECRET0 = UInt64(0x2D358DCCAA6C78A5)
comptime RAPID_SECRET1 = UInt64(0x8BB84B93962EACC9)
comptime RAPID_SECRET2 = UInt64(0x4B33A62ED433D4A3)
comptime RAPID_SECRET3 = UInt64(0x4D5A2DA51DE1AA47)
comptime RAPID_SECRET4 = UInt64(0xA0761D6478BD642F)
comptime RAPID_SECRET5 = UInt64(0xE7037ED1A0B428DB)
comptime RAPID_SECRET6 = UInt64(0x90ED1765281C388C)
comptime RAPID_SECRET7 = UInt64(0xAAAAAAAAAAAAAAAA)


@always_inline
def _rapid_mum(A: UInt64, B: UInt64) -> Tuple[UInt64, UInt64]:
    """128-bit multiply, return (lo, hi). Port of rapid_mum from rapidhash.h."""
    var r = A.cast[DType.uint128]() * B.cast[DType.uint128]()
    return (UInt64(r), UInt64(r >> 64))


@always_inline
def _rapid_mix(A: UInt64, B: UInt64) -> UInt64:
    """Multiply-mix: 128-bit multiply then XOR halves. Port of rapid_mix."""
    var lo_hi = _rapid_mum(A, B)
    return lo_hi[0] ^ lo_hi[1]


@always_inline
def _rapidhash_fixed[byte_width: Int](value: UInt64) -> UInt64:
    """Rapidhash for a single fixed-width value.

    Exact port of rapidhash_internal() for len=byte_width, seed=0.
    C reference:
      seed ^= rapid_mix(seed ^ secret[2], secret[1])  // seed=0
      seed ^= len  // for len >= 4
      a = value ^ secret[1]
      b = value ^ seed
      rapid_mum(&a, &b)
      return rapid_mix(a ^ secret[7], b ^ secret[1] ^ len)
    """
    var seed = _rapid_mix(RAPID_SECRET2, RAPID_SECRET1) ^ UInt64(byte_width)
    var a = value ^ RAPID_SECRET1
    var b = value ^ seed
    var lo_hi = _rapid_mum(a, b)
    return _rapid_mix(
        lo_hi[0] ^ RAPID_SECRET7,
        lo_hi[1] ^ RAPID_SECRET1 ^ UInt64(byte_width),
    )


# ---------------------------------------------------------------------------
# SIMD-width rapidhash helpers (GPU-compatible, no uint128)
# ---------------------------------------------------------------------------


@always_inline
def _rapid_mum_wide[
    W: Int
](a: SIMD[uint64.native, W], b: SIMD[uint64.native, W]) -> Tuple[
    SIMD[uint64.native, W], SIMD[uint64.native, W]
]:
    """128-bit multiply returning (lo, hi) using 32-bit sub-products.

    GPU-compatible: avoids uint128 which Metal does not support.
    """
    comptime lo32 = SIMD[uint64.native, 1](0xFFFFFFFF)
    var a_lo = a & SIMD[uint64.native, W](0xFFFFFFFF)
    var a_hi = a >> 32
    var b_lo = b & SIMD[uint64.native, W](0xFFFFFFFF)
    var b_hi = b >> 32
    var t0 = a_lo * b_lo
    var t1 = a_lo * b_hi
    var t2 = a_hi * b_lo
    var t3 = a_hi * b_hi
    var mid = (
        (t0 >> 32)
        + (t1 & SIMD[uint64.native, W](0xFFFFFFFF))
        + (t2 & SIMD[uint64.native, W](0xFFFFFFFF))
    )
    var lo = (t0 & SIMD[uint64.native, W](0xFFFFFFFF)) | (mid << 32)
    var hi = t3 + (t1 >> 32) + (t2 >> 32) + (mid >> 32)
    return (lo, hi)


@always_inline
def _rapid_mix_wide[
    W: Int
](a: SIMD[uint64.native, W], b: SIMD[uint64.native, W]) -> SIMD[
    uint64.native, W
]:
    """rapid_mix for SIMD lanes: 128-bit multiply then XOR halves."""
    var lo_hi = _rapid_mum_wide[W](a, b)
    return lo_hi[0] ^ lo_hi[1]


# ---------------------------------------------------------------------------
# rapidhash — vectorized hash for primitive arrays (SIMD via elementwise)
# ---------------------------------------------------------------------------


@always_inline
def _rapidhash_bool[
    W: Int
](bits: SIMD[DType.bool, W]) -> SIMD[uint64.native, W]:
    """Bool rapidhash: select between precomputed hash(0) and hash(1)."""
    comptime hash_false = _rapidhash_fixed[size_of[Scalar[bool_.native]]()](
        UInt64(0)
    )
    comptime hash_true = _rapidhash_fixed[size_of[Scalar[bool_.native]]()](
        UInt64(1)
    )
    return bits.select(
        SIMD[uint64.native, W](hash_true),
        SIMD[uint64.native, W](hash_false),
    )


@always_inline
def _rapidhash_bool_masked[
    W: Int
](bits: SIMD[DType.bool, W], valid: SIMD[DType.bool, W]) -> SIMD[
    uint64.native, W
]:
    """Bool rapidhash with null masking via validity bitmap."""
    return valid.select(
        _rapidhash_bool[W](bits), SIMD[uint64.native, W](NULL_HASH_SENTINEL)
    )


@always_inline
def _rapidhash_bits[
    byte_width: Int, W: Int
](v: SIMD[uint64.native, W]) -> SIMD[uint64.native, W]:
    """Rapidhash core over `byte_width`-wide values already widened to uint64.
    """
    comptime seed = _rapid_mix(RAPID_SECRET2, RAPID_SECRET1) ^ UInt64(
        byte_width
    )
    var a = v ^ SIMD[uint64.native, W](RAPID_SECRET1)
    var b = v ^ SIMD[uint64.native, W](seed)
    var lo_hi = _rapid_mum_wide[W](a, b)
    return _rapid_mix_wide[W](
        lo_hi[0] ^ SIMD[uint64.native, W](RAPID_SECRET7),
        lo_hi[1] ^ SIMD[uint64.native, W](RAPID_SECRET1 ^ UInt64(byte_width)),
    )


@always_inline
def _rapidhash_primitive[
    T: PrimitiveType, W: Int
](vals: SIMD[T.native, W]) -> SIMD[uint64.native, W]:
    """Rapidhash for a SIMD vector of primitive values."""
    comptime byte_width = size_of[Scalar[T.native]]()

    comptime if byte_width <= 8:
        # Zero-extend to uint64 (matches C's rapid_read32/rapid_read64).
        # Mask to byte_width bits to prevent sign-extension for <8-byte types.
        comptime mask = ~UInt64(0) if byte_width >= 8 else (
            UInt64(1) << UInt64(byte_width * 8)
        ) - 1
        return _rapidhash_bits[byte_width, W](
            vals.cast[uint64.native]() & SIMD[uint64.native, W](mask)
        )
    else:
        # decimal128 / decimal256 and the month-day-nano interval do not fit a
        # uint64 lane: fold their 64-bit limbs so the high bits participate. A
        # plain truncating cast would collide every pair of values that differ
        # only above bit 63 — and group-by buckets on the hash alone.
        comptime limbs = byte_width // 8
        var h = SIMD[uint64.native, W](0)
        comptime for i in range(limbs):
            var limb = (vals >> SIMD[T.native, W](i * 64)).cast[uint64.native]()
            h = _combine_hashes[W](h, _rapidhash_bits[8, W](limb))
        return h


@always_inline
def _rapidhash_primitive_masked[
    T: PrimitiveType, W: Int
](vals: SIMD[T.native, W], valid: SIMD[DType.bool, W]) -> SIMD[
    uint64.native, W
]:
    """Rapidhash for primitive values with null masking via validity bitmap."""
    return valid.select(
        _rapidhash_primitive[T, W](vals),
        SIMD[uint64.native, W](NULL_HASH_SENTINEL),
    )


@always_inline
def _combine_hashes[
    W: Int
](existing: SIMD[uint64.native, W], new: SIMD[uint64.native, W]) -> SIMD[
    uint64.native, W
]:
    """Element-wise hash combine using golden ratio constant and rapid_mum."""
    var mixed = existing ^ SIMD[uint64.native, W](0x9E3779B97F4A7C15)
    var lo_hi = _rapid_mum_wide[W](mixed, new)
    return lo_hi[0] ^ lo_hi[1]


struct RapidHash(Kernel):
    """Column hashing kernel — one ``UInt64`` per row.

    The typed leaves are the ``apply`` overloads; ``dispatch`` resolves a
    runtime-typed array to the matching leaf via the ``AnyDataType.dispatch_*`` family
    rather than a per-dtype ladder, so adding a dtype to a family covers it
    without touching this file. Null elements hash to ``NULL_HASH_SENTINEL`` so
    that "null == null" holds for grouping and joining (Arrow's `hash_*`
    semantics).

    Not hashable: `null` and `fixed_size_binary` (no leaf yet), and the
    month-day-nano interval (>64 bits and outside the decimal family).
    """

    comptime name = "rapidhash"

    @staticmethod
    def dispatch(
        keys: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> UInt64Array:
        """Resolve `keys`'s runtime dtype and hash it."""
        var dt = keys.dtype()
        if dt == bool_:
            return RapidHash.apply(keys.as_bool(), ctx)
        elif dt.is_numeric():

            @parameter
            def numeric[T: NumericType](d: T) raises -> UInt64Array:
                return RapidHash.apply(keys.as_primitive[T](), ctx)

            return dt.dispatch_numeric[numeric]()
        elif dt.is_binary_like():

            @parameter
            def binarylike[T: BinaryLikeType](d: T) raises -> UInt64Array:
                return RapidHash.apply(keys.as_binary_like[T](), ctx)

            return dt.dispatch_binarylike[binarylike]()
        elif dt.is_list_like() or dt.is_map():

            @parameter
            def listlike[T: ListLikeType](d: T) raises -> UInt64Array:
                return RapidHash.apply(keys.as_list_like[T](), ctx)

            return dt.dispatch_listlike[listlike]()
        elif dt.is_struct():
            return RapidHash.apply(keys.as_struct(), ctx)
        elif dt.is_fixed_size_list():
            return RapidHash.apply(keys.as_fixed_size_list(), ctx)
        elif dt.is_dictionary():
            # Hash the decoded values: a dictionary-encoded key must hash the
            # same as the equivalent plain column, otherwise two batches with
            # different dictionaries would never group together.
            return RapidHash.dispatch(
                cast(keys, dt.as_dictionary().value_type().copy(), False, ctx),
                ctx,
            )
        elif dt.is_decimal128():
            return RapidHash.apply(keys.as_decimal128(), ctx)
        elif dt.is_decimal256():
            return RapidHash.apply(keys.as_decimal256(), ctx)
        elif dt.is_primitive():
            # Temporal, interval, and decimal32/64 hash bit-identically to their
            # integer `storage_type()`, so reinterpret and reuse the numeric
            # leaf instead of instantiating one hash kernel per logical dtype.
            return RapidHash.dispatch(
                reinterpret_array(keys, dt.storage_type()), ctx
            )
        else:
            raise Error("rapidhash: unsupported dtype ", dt)

    @staticmethod
    def apply(
        keys: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> UInt64Array:
        """Vectorized rapidhash for bool arrays.

        Precomputes hash(false) and hash(true), loads data bits via the
        bitmap-mask pattern, and uses ``SIMD.select()`` for branchless dispatch.
        Null elements are replaced with ``NULL_HASH_SENTINEL`` inline.

        Parallelism is delegated to ``apply`` via the ``ExecutionContext`` —
        no per-kernel stripe logic here.
        """
        var n = len(keys)
        var buf: Buffer[mut=True]
        if ctx.is_gpu():
            buf = Buffer.alloc_device[uint64.native](ctx.device.value(), n)
        else:
            buf = Buffer.alloc_uninit[uint64.native](n)

        var dst = buf.view[uint64.native](0, n)
        var validity = keys.validity()
        if validity:
            apply[uint64.native, _rapidhash_bool_masked](
                keys.values(),
                validity.value(),
                dst,
                ctx,
            )
        else:
            apply[uint64.native, _rapidhash_bool](keys.values(), dst, ctx)

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
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> UInt64Array:
        """Vectorized rapidhash for primitive arrays.

        Each SIMD lane independently computes the rapidhash of one element.
        Null elements are replaced with ``NULL_HASH_SENTINEL`` inline.

        Parallelism is handled uniformly by ``apply`` using the
        ``ExecutionContext`` — CPU vs GPU is picked from ``ctx.device``, and
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
                        _rapidhash_primitive[T, 1](values.unsafe_get(i))
                    )
                else:
                    builder.unsafe_append(_h(NULL_HASH_SENTINEL))
            return builder.finish()
        else:
            var buf: Buffer[mut=True]
            if ctx.is_gpu():
                buf = Buffer.alloc_device[uint64.native](ctx.device.value(), n)
            else:
                buf = Buffer.alloc_uninit[uint64.native](n)

            var dst = buf.view[uint64.native](0, n)
            var validity = keys.validity()
            if validity:
                apply[
                    T.native,
                    uint64.native,
                    _rapidhash_primitive_masked[T, ...],
                ](
                    keys.values(),
                    validity.value(),
                    dst,
                    ctx,
                )
            else:
                apply[T.native, uint64.native, _rapidhash_primitive[T, ...]](
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
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> UInt64Array:
        """Hash each element of a binary-like array (string, large_string,
        binary, large_binary).

        Uses AHash for variable-length values (rapidhash for byte strings
        requires the full multi-branch rapidhash_internal — future work).
        Currently scalar-serial; parallelizing variable-length hashing is future
        work — the ``ctx`` parameter exists for API consistency.
        """
        _ = ctx  # TODO: SIMD + parallel string hashing
        var n = len(keys)
        var builder = UInt64Builder(capacity=n)
        var has_bitmap = Bool(keys.bitmap)

        for i in range(n):
            if has_bitmap and not keys.bitmap.value().test(keys.offset + i):
                builder.unsafe_append(_h(NULL_HASH_SENTINEL))
            else:
                builder.unsafe_append(
                    _h(_hash(String(keys.unsafe_get(UInt(i)))))
                )

        return builder.finish()

    @staticmethod
    def apply(
        keys: StructArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> UInt64Array:
        """Hash a struct array by combining per-field hashes column-wise.

        Each field is hashed independently via ``dispatch`` and the results are
        combined element-wise using ``_combine_hashes``. The
        ``ExecutionContext`` is forwarded to per-field hashing and to the
        combine pass — all stripe-parallelism is handled inside ``apply``.
        """
        var n = len(keys)
        var num_fields = len(keys.children)
        if num_fields == 0:
            raise Error("rapidhash: empty struct array")

        var result = RapidHash.dispatch(keys.children[0], ctx)

        for k in range(1, num_fields):
            var field_hashes = RapidHash.dispatch(keys.children[k], ctx)

            var buf: Buffer[mut=True]
            if ctx.is_gpu():
                buf = Buffer.alloc_device[uint64.native](ctx.device.value(), n)
            else:
                buf = Buffer.alloc_uninit[uint64.native](n)
            apply[uint64.native, uint64.native, _combine_hashes](
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
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> UInt64Array:
        """Hash a list/large-list/map array row-wise: hash the whole child once,
        then fold the child hashes over each row's element range (column-wise,
        no row-encoding). Null rows hash to ``NULL_HASH_SENTINEL``."""
        var n = len(keys)
        var child_hashes = RapidHash.dispatch(keys.values().copy(), ctx)
        var builder = UInt64Builder(n)
        for i in range(n):
            if not keys.is_valid(i):
                builder.append(NULL_HASH_SENTINEL)
            else:
                var h = UInt64(0)
                var rng = keys.child_range(i)
                for j in range(rng[0], rng[1]):
                    h = _combine_hashes[1](h, child_hashes.unsafe_get(j))
                builder.append(h)
        return builder.finish()

    @staticmethod
    def apply(
        keys: FixedSizeListArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> UInt64Array:
        """Hash a fixed-size-list array row-wise: hash the child once, fold each
        row's ``list_size`` child hashes. Null rows hash to
        ``NULL_HASH_SENTINEL``."""
        var n = len(keys)
        var size = keys.dtype.as_fixed_size_list().size
        var child_hashes = RapidHash.dispatch(keys.values().copy(), ctx)
        var builder = UInt64Builder(n)
        for i in range(n):
            if not keys.is_valid(i):
                builder.append(NULL_HASH_SENTINEL)
            else:
                var h = UInt64(0)
                var base = (keys.offset + i) * size
                for j in range(size):
                    h = _combine_hashes[1](h, child_hashes.unsafe_get(base + j))
                builder.append(h)
        return builder.finish()


# ---------------------------------------------------------------------------
# Public API — thin free delegators to the RapidHash kernel.
#
# These exist because `SwissHashTable[hasher]` / `HashJoin[hasher]` bind the
# hasher as a comptime *function value*, which needs a single unambiguous
# symbol rather than an overload set.
# ---------------------------------------------------------------------------


def rapidhash(
    keys: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> UInt64Array:
    """Hash `keys` element-wise; nulls hash to ``NULL_HASH_SENTINEL``."""
    return RapidHash.dispatch(keys, ctx)


def rapidhash(
    keys: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> UInt64Array:
    return RapidHash.apply(keys, ctx)


def rapidhash[
    T: PrimitiveType
](
    keys: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> UInt64Array:
    return RapidHash.apply(keys, ctx)


def rapidhash[
    T: BinaryLikeType
](
    keys: BinaryLikeArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> UInt64Array:
    return RapidHash.apply(keys, ctx)


def rapidhash(
    keys: StructArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> UInt64Array:
    return RapidHash.apply(keys, ctx)
