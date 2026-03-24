"""Hashing kernels for Arrow arrays.

Provides column-wise hash computation for use in groupby, joins, and
partitioning. Follows the DuckDB/DataFusion approach of hashing each
column independently and combining hashes across columns.

Public API:
  - ``rapidhash``: hash any array → PrimitiveArray[uint64]
    - PrimitiveArray[T]: vectorized rapidhash (SIMD via elementwise)
    - StringArray: per-element AHash (variable-length fallback)
    - StructArray: per-column hash with combining (multi-key)
    - AnyArray: runtime-typed dispatch
  - ``ahash``: original AHash-based per-element hash (scalar)
  - ``hash_identity``: identity hash for small integer types (bool, uint8, int8)

Rapidhash port follows the C reference at https://github.com/Nicoshev/rapidhash
"""

from std.algorithm.functional import elementwise
from std.gpu.host import DeviceContext
from std.hashlib import hash as _hash
from std.sys import size_of
from std.sys.info import simd_byte_width
from std.utils.index import IndexList

from ..arrays import PrimitiveArray, StringArray, StructArray, AnyArray
from ..builders import PrimitiveBuilder
from ..buffers import BufferBuilder
from ..dtypes import (
    DataType,
    uint8,
    int8,
    uint64,
    bool_,
    numeric_dtypes,
    primitive_dtypes,
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


@always_inline
def _combine(existing: UInt64, new: UInt64) -> UInt64:
    """Combine two hash values (rapidhash-based, replaces polynomial combine)."""
    return _rapid_mix(existing ^ UInt64(0x9E3779B97F4A7C15), new)


# ---------------------------------------------------------------------------
# rapidhash — vectorized hash for primitive arrays (SIMD via elementwise)
# ---------------------------------------------------------------------------

# FIXME: use the seeding from the Rust implementation
def rapidhash[
    T: DataType
](keys: PrimitiveArray[T]) raises -> PrimitiveArray[uint64]:
    """Vectorized rapidhash for primitive arrays.

    Uses ``elementwise`` for SIMD processing (same pattern as arithmetic
    kernels). Each SIMD lane independently computes the rapidhash of one
    element. Null elements are overwritten with ``NULL_HASH_SENTINEL`` in
    a post-pass.

    Bool arrays use scalar path since they are bit-packed.
    """
    comptime native = T.native
    comptime byte_width = size_of[Scalar[native]]()
    var n = len(keys)

    # Bool arrays are bit-packed — can't use SIMD load. Use scalar path.
    comptime if T == bool_:
        var builder = PrimitiveBuilder[uint64](capacity=n)
        var has_bitmap = Bool(keys.bitmap)
        for i in range(n):
            if has_bitmap and not keys.bitmap.value().is_valid(keys.offset + i):
                builder.unsafe_append(_h(NULL_HASH_SENTINEL))
            else:
                var v = UInt64(keys.unsafe_get(i))
                builder.unsafe_append(
                    _h(_rapidhash_fixed[byte_width](v))
                )
        return builder.finish()

    var buf = BufferBuilder.alloc_uninit(
        BufferBuilder._aligned_size[uint64.native](n)
    )
    var out_ptr: UnsafePointer[Scalar[uint64.native], MutAnyOrigin]
    out_ptr = buf.ptr.bitcast[Scalar[uint64.native]]()
    var in_ptr = keys.buffer.aligned_unsafe_ptr[native](keys.offset)

    # Pre-compute seed (constant for all elements).
    # C: seed = 0; seed ^= rapid_mix(seed ^ secret[2], secret[1]); seed ^= len
    var seed = _rapid_mix(RAPID_SECRET2, RAPID_SECRET1) ^ UInt64(byte_width)

    @parameter
    @always_inline
    def process[
        W: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]) -> None:
        var i = idx[0]
        # Zero-extend to uint64 (matches C's rapid_read32/rapid_read64).
        # Mask to byte_width bits to prevent sign-extension for <8-byte types.
        comptime mask = ~UInt64(0) if byte_width >= 8 else (UInt64(1) << UInt64(byte_width * 8)) - 1
        var vals = in_ptr.load[width=W](i).cast[DType.uint64]() & mask
        # a = value ^ secret[1]; b = value ^ seed
        var a = vals ^ RAPID_SECRET1
        var b = vals ^ seed
        # rapid_mum(&a, &b): 128-bit multiply per SIMD lane
        var r = a.cast[DType.uint128]() * b.cast[DType.uint128]()
        var lo = r.cast[DType.uint64]()
        var hi = (r >> 64).cast[DType.uint64]()
        # rapid_mix(a ^ secret[7], b ^ secret[1] ^ len)
        var a2 = lo ^ RAPID_SECRET7
        var b2 = hi ^ RAPID_SECRET1 ^ UInt64(byte_width)
        var r2 = a2.cast[DType.uint128]() * b2.cast[DType.uint128]()
        var result = r2.cast[DType.uint64]() ^ (r2 >> 64).cast[DType.uint64]()
        for j in range(W):
            out_ptr[i + j] = Scalar[uint64.native](result[j])

    # TODO: enable running it on gpu as well
    comptime cpu_width = simd_byte_width() // size_of[Scalar[uint64.native]]()
    elementwise[process, cpu_width, target="cpu", use_blocking_impl=True](n)

    # Post-pass: overwrite null positions with sentinel.
    if keys.bitmap:
        for i in range(n):
            if not keys.bitmap.value().is_valid(keys.offset + i):
                (out_ptr + i).store(Scalar[uint64.native](NULL_HASH_SENTINEL))

    return PrimitiveArray[uint64](
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buf.finish(),
    )


# ---------------------------------------------------------------------------
# ahash — original AHash-based scalar hash (retained for strings)
# ---------------------------------------------------------------------------


def ahash[
    T: DataType
](keys: PrimitiveArray[T]) raises -> PrimitiveArray[uint64]:
    """Per-element AHash for a primitive array (scalar, not vectorized)."""
    var n = len(keys)
    var builder = PrimitiveBuilder[uint64](capacity=n)
    var buf = keys.buffer
    var off = keys.offset
    var has_bitmap = Bool(keys.bitmap)

    for i in range(n):
        if has_bitmap and not keys.bitmap.value().is_valid(off + i):
            builder.unsafe_append(_h(NULL_HASH_SENTINEL))
        else:
            builder.unsafe_append(
                _h(_hash(buf.unsafe_get[T.native](off + i)))
            )

    return builder.finish()


def ahash(keys: StringArray) raises -> PrimitiveArray[uint64]:
    """Per-element AHash for a string array."""
    var n = len(keys)
    var builder = PrimitiveBuilder[uint64](capacity=n)
    var has_bitmap = Bool(keys.bitmap)

    for i in range(n):
        if has_bitmap and not keys.bitmap.value().is_valid(keys.offset + i):
            builder.unsafe_append(_h(NULL_HASH_SENTINEL))
        else:
            builder.unsafe_append(
                _h(_hash(String(keys.unsafe_get(UInt(i)))))
            )

    return builder.finish()


def ahash(keys: AnyArray) raises -> PrimitiveArray[uint64]:
    """Runtime-typed AHash dispatch."""
    if keys.dtype() == bool_:
        return ahash[bool_](keys.as_primitive[bool_]())

    comptime for dtype in numeric_dtypes:
        if keys.dtype() == dtype:
            return ahash[dtype](keys.as_primitive[dtype]())

    if keys.dtype().is_string():
        return ahash(keys.as_string())

    if keys.dtype().is_struct():
        return rapidhash(keys.as_struct())

    raise Error("ahash: unsupported dtype ", keys.dtype())


def rapidhash(keys: StringArray) raises -> PrimitiveArray[uint64]:
    """Hash each element of a string array.

    Uses AHash for variable-length strings (rapidhash for strings requires
    the full multi-branch rapidhash_internal — future work).
    """
    return ahash(keys)


def rapidhash(keys: StructArray) raises -> PrimitiveArray[uint64]:
    """Hash a struct array by combining per-field hashes column-wise.

    Each field is hashed independently via ``rapidhash(AnyArray)``
    and the results are combined element-wise using ``_rapid_mix``.
    """
    var n = len(keys)
    var num_fields = len(keys.children)
    if num_fields == 0:
        raise Error("rapidhash: empty struct array")

    var result = rapidhash(keys.children[0])

    for k in range(1, num_fields):
        var field_hashes = rapidhash(keys.children[k])
        var builder = PrimitiveBuilder[uint64](capacity=n)
        for i in range(n):
            builder.unsafe_append(
                _h(
                    _combine(
                        UInt64(result.unsafe_get(i)),
                        UInt64(field_hashes.unsafe_get(i)),
                    )
                )
            )
        result = builder.finish()

    return result^


def rapidhash(keys: AnyArray) raises -> PrimitiveArray[uint64]:
    """Runtime-typed rapidhash dispatch."""
    if keys.dtype() == bool_:
        return rapidhash[bool_](keys.as_primitive[bool_]())

    comptime for dtype in numeric_dtypes:
        if keys.dtype() == dtype:
            return rapidhash[dtype](keys.as_primitive[dtype]())

    if keys.dtype().is_string():
        return rapidhash(keys.as_string())

    if keys.dtype().is_struct():
        return rapidhash(keys.as_struct())

    raise Error("rapidhash: unsupported dtype ", keys.dtype())


# ---------------------------------------------------------------------------
# hash_identity — identity hash for small integer types
# ---------------------------------------------------------------------------


def hash_identity[
    T: DataType
](keys: PrimitiveArray[T]) raises -> PrimitiveArray[uint64]:
    """Identity hash: returns values cast to uint64 with no hash overhead.

    For int8, values are offset by +128 to produce non-negative indices.
    Null elements map to ``NULL_HASH_SENTINEL``.

    Only valid for bool, uint8, and int8 — produces dense values in [0, 255].
    """
    comptime _OFFSET = 128 if T == int8 else 0

    var n = len(keys)
    var builder = PrimitiveBuilder[uint64](capacity=n)
    var has_bitmap = Bool(keys.bitmap)

    for i in range(n):
        if has_bitmap and not keys.bitmap.value().is_valid(keys.offset + i):
            builder.unsafe_append(_h(NULL_HASH_SENTINEL))
        else:
            builder.unsafe_append(_h(Int(keys.unsafe_get(i)) + _OFFSET))

    return builder.finish()


def hash_identity(keys: AnyArray) raises -> PrimitiveArray[uint64]:
    """Runtime-typed identity hash dispatch."""
    if keys.dtype() == bool_:
        return hash_identity[bool_](keys.as_primitive[bool_]())
    if keys.dtype() == uint8:
        return hash_identity[uint8](keys.as_primitive[uint8]())
    if keys.dtype() == int8:
        return hash_identity[int8](keys.as_primitive[int8]())
    raise Error("hash_identity: only supports bool, uint8, int8")
