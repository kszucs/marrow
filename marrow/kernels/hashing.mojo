"""Hashing kernels for Arrow arrays.

Provides column-wise hash computation for use in groupby, joins, and
partitioning. Follows the DuckDB/DataFusion approach of hashing each
column independently and combining hashes across columns.

Public API:
  - ``rapidhash``: hash any array → PrimitiveArray[UInt64Type]
    - BoolArray: vectorized via precomputed hash + SIMD select
    - PrimitiveArray[T]: vectorized rapidhash (SIMD via elementwise)
    - StringArray: per-element AHash (variable-length fallback)
    - StructArray: per-column hash with combining (multi-key)
    - AnyArray: runtime-typed dispatch
  - ``hash_identity``: identity hash for small integer types (bool, uint8, int8)

Rapidhash port follows the C reference at https://github.com/Nicoshev/rapidhash
"""

from std.gpu.host import DeviceContext
from std.hashlib import hash as _hash
from std.sys import size_of

from ..arrays import (
    BoolArray,
    PrimitiveArray,
    StringArray,
    StructArray,
    AnyArray,
)
from ..builders import PrimitiveBuilder
from ..buffers import Buffer
from ..views import apply
from ..dtypes import (
    PrimitiveType,
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
def _rapidhash_primitive[
    T: PrimitiveType, W: Int
](vals: SIMD[T.native, W]) -> SIMD[uint64.native, W]:
    """Rapidhash for a SIMD vector of primitive values."""
    comptime byte_width = size_of[Scalar[T.native]]()
    comptime seed = _rapid_mix(RAPID_SECRET2, RAPID_SECRET1) ^ UInt64(
        byte_width
    )
    # Zero-extend to uint64 (matches C's rapid_read32/rapid_read64).
    # Mask to byte_width bits to prevent sign-extension for <8-byte types.
    comptime mask = ~UInt64(0) if byte_width >= 8 else (
        UInt64(1) << UInt64(byte_width * 8)
    ) - 1
    var v = vals.cast[uint64.native]() & SIMD[uint64.native, W](mask)
    var a = v ^ SIMD[uint64.native, W](RAPID_SECRET1)
    var b = v ^ SIMD[uint64.native, W](seed)
    var lo_hi = _rapid_mum_wide[W](a, b)
    return _rapid_mix_wide[W](
        lo_hi[0] ^ SIMD[uint64.native, W](RAPID_SECRET7),
        lo_hi[1] ^ SIMD[uint64.native, W](RAPID_SECRET1 ^ UInt64(byte_width)),
    )


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


def rapidhash(
    keys: BoolArray,
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[UInt64Type]:
    """Vectorized rapidhash for bool arrays.

    Precomputes hash(false) and hash(true), loads data bits via the
    bitmap-mask pattern, and uses ``SIMD.select()`` for branchless dispatch.
    Null elements are replaced with ``NULL_HASH_SENTINEL`` inline.
    """
    var n = len(keys)
    var buf: Buffer[mut=True]
    if ctx:
        buf = Buffer.alloc_device[uint64.native](ctx.value(), n)
    else:
        buf = Buffer.alloc_uninit[uint64.native](n)

    var dst = buf.view[uint64.native]()
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

    return PrimitiveArray[UInt64Type](
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buf.to_immutable(),
    )


# FIXME: use the seeding from the Rust implementation
def rapidhash[
    T: PrimitiveType
](
    keys: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[UInt64Type]:
    """Vectorized rapidhash for primitive arrays.

    Each SIMD lane independently computes the rapidhash of one element.
    Null elements are replaced with ``NULL_HASH_SENTINEL`` inline.

    Dispatches to GPU when ``ctx`` is provided, CPU SIMD otherwise.
    """
    var n = len(keys)
    var buf: Buffer[mut=True]
    if ctx:
        buf = Buffer.alloc_device[uint64.native](ctx.value(), n)
    else:
        buf = Buffer.alloc_uninit[uint64.native](n)

    var dst = buf.view[uint64.native]()
    var validity = keys.validity()
    if validity:
        apply[T.native, uint64.native, _rapidhash_primitive_masked[T, ...]](
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

    return PrimitiveArray[UInt64Type](
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buf.to_immutable(),
    )


def rapidhash(keys: StringArray) raises -> PrimitiveArray[UInt64Type]:
    """Hash each element of a string array.

    Uses AHash for variable-length strings (rapidhash for strings requires
    the full multi-branch rapidhash_internal — future work).
    """
    var n = len(keys)
    var builder = PrimitiveBuilder[UInt64Type](capacity=n)
    var has_bitmap = Bool(keys.bitmap)

    for i in range(n):
        if has_bitmap and not keys.bitmap.value().test(keys.offset + i):
            builder.unsafe_append(_h(NULL_HASH_SENTINEL))
        else:
            builder.unsafe_append(_h(_hash(String(keys.unsafe_get(UInt(i))))))

    return builder.finish()


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


def rapidhash(
    keys: StructArray,
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[UInt64Type]:
    """Hash a struct array by combining per-field hashes column-wise.

    Each field is hashed independently via ``rapidhash(AnyArray)``
    and the results are combined element-wise using ``_combine_hashes``.
    """
    var n = len(keys)
    var num_fields = len(keys.children)
    if num_fields == 0:
        raise Error("rapidhash: empty struct array")

    var result = rapidhash(keys.children[0], ctx)

    for k in range(1, num_fields):
        var field_hashes = rapidhash(keys.children[k], ctx)

        var buf: Buffer[mut=True]
        if ctx:
            buf = Buffer.alloc_device[uint64.native](ctx.value(), n)
        else:
            buf = Buffer.alloc_uninit[uint64.native](n)
        apply[uint64.native, uint64.native, _combine_hashes](
            result.values(),
            field_hashes.values(),
            buf.view[uint64.native](),
            ctx,
        )
        result = PrimitiveArray[UInt64Type](
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf.to_immutable(),
        )

    return result^


def rapidhash(
    keys: AnyArray,
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[UInt64Type]:
    """Runtime-typed rapidhash dispatch."""
    if keys.dtype() == bool_:
        return rapidhash(keys.as_bool(), ctx)

    if keys.dtype() == int8:
        return rapidhash[Int8Type](keys.as_primitive[Int8Type](), ctx)
    elif keys.dtype() == int16:
        return rapidhash[Int16Type](keys.as_primitive[Int16Type](), ctx)
    elif keys.dtype() == int32:
        return rapidhash[Int32Type](keys.as_primitive[Int32Type](), ctx)
    elif keys.dtype() == int64:
        return rapidhash[Int64Type](keys.as_primitive[Int64Type](), ctx)
    elif keys.dtype() == uint8:
        return rapidhash[UInt8Type](keys.as_primitive[UInt8Type](), ctx)
    elif keys.dtype() == uint16:
        return rapidhash[UInt16Type](keys.as_primitive[UInt16Type](), ctx)
    elif keys.dtype() == uint32:
        return rapidhash[UInt32Type](keys.as_primitive[UInt32Type](), ctx)
    elif keys.dtype() == uint64:
        return rapidhash[UInt64Type](keys.as_primitive[UInt64Type](), ctx)
    elif keys.dtype() == float16:
        return rapidhash[Float16Type](keys.as_primitive[Float16Type](), ctx)
    elif keys.dtype() == float32:
        return rapidhash[Float32Type](keys.as_primitive[Float32Type](), ctx)
    elif keys.dtype() == float64:
        return rapidhash[Float64Type](keys.as_primitive[Float64Type](), ctx)

    if keys.dtype().is_string():
        return rapidhash(keys.as_string())

    if keys.dtype().is_struct():
        return rapidhash(keys.as_struct(), ctx)

    raise Error("rapidhash: unsupported dtype ", keys.dtype())


# ---------------------------------------------------------------------------
# hash_identity — identity hash for small integer types
# ---------------------------------------------------------------------------


def hash_identity[
    T: PrimitiveType
](keys: PrimitiveArray[T]) raises -> PrimitiveArray[UInt64Type]:
    """Identity hash: returns values cast to uint64 with no hash overhead.

    For int8, values are offset by +128 to produce non-negative indices.
    Null elements map to ``NULL_HASH_SENTINEL``.

    Only valid for bool, uint8, and int8 — produces dense values in [0, 255].
    """
    comptime _OFFSET = 128 if T.native == DType.int8 else 0

    var n = len(keys)
    var builder = PrimitiveBuilder[UInt64Type](capacity=n)
    var has_bitmap = Bool(keys.bitmap)

    for i in range(n):
        if has_bitmap and not keys.bitmap.value().test(keys.offset + i):
            builder.unsafe_append(_h(NULL_HASH_SENTINEL))
        else:
            builder.unsafe_append(_h(Int(keys.unsafe_get(i)) + _OFFSET))

    return builder.finish()


def hash_identity(keys: BoolArray) raises -> PrimitiveArray[UInt64Type]:
    """Identity hash for bool arrays (values 0 and 1)."""
    var n = len(keys)
    var builder = PrimitiveBuilder[UInt64Type](capacity=n)
    var has_bitmap = Bool(keys.bitmap)
    for i in range(n):
        if has_bitmap and not keys.bitmap.value().test(keys.offset + i):
            builder.unsafe_append(_h(NULL_HASH_SENTINEL))
        else:
            builder.unsafe_append(_h(Int(keys[i].value())))
    return builder.finish()


def hash_identity(keys: AnyArray) raises -> PrimitiveArray[UInt64Type]:
    """Runtime-typed identity hash dispatch."""
    if keys.dtype() == bool_:
        return hash_identity(keys.as_bool())
    if keys.dtype() == uint8:
        return hash_identity[UInt8Type](keys.as_primitive[UInt8Type]())
    if keys.dtype() == int8:
        return hash_identity[Int8Type](keys.as_primitive[Int8Type]())
    raise Error("hash_identity: only supports bool, uint8, int8")
