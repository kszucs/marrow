"""Hashing kernels for Arrow arrays.

Provides column-wise hash computation for use in groupby, joins, and
partitioning. Follows the DuckDB/DataFusion approach of hashing each
column independently and combining hashes across columns.

Public API:
  - ``rapidhash``: hash any array → PrimitiveArray[uint64]
    - BoolArray: vectorized via precomputed hash + SIMD select
    - PrimitiveArray[T]: vectorized rapidhash (SIMD via elementwise)
    - StringArray: per-element AHash (variable-length fallback)
    - StructArray: per-column hash with combining (multi-key)
    - AnyArray: runtime-typed dispatch
  - ``hash_identity``: identity hash for small integer types (bool, uint8, int8)

Rapidhash port follows the C reference at https://github.com/Nicoshev/rapidhash
"""

from std.algorithm.functional import elementwise
from std.gpu.host import DeviceContext, get_gpu_target
from std.hashlib import hash as _hash
from std.math import iota
from std.sys import size_of, has_accelerator
from std.sys.info import simd_byte_width, simd_width_of
from std.utils.index import IndexList

from ..arrays import (
    BoolArray,
    PrimitiveArray,
    StringArray,
    StructArray,
    AnyArray,
)
from ..builders import PrimitiveBuilder
from ..buffers import Buffer
from ..views import BitmapView, BufferView
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


# ---------------------------------------------------------------------------
# rapidhash — vectorized hash for primitive arrays (SIMD via elementwise)
# ---------------------------------------------------------------------------


def _rapidhash_elementwise[
    T: DataType,
    v_origin: Origin[mut=False],
](
    out_ptr: UnsafePointer[Scalar[DType.uint64], MutAnyOrigin],
    in_: BufferView[T.native, _],
    validity: Optional[BitmapView[v_origin]],
    length: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Elementwise rapidhash dispatch — views as params for GPU DevicePassable.

    Computes rapidhash for each element. If ``validity`` is non-empty, null
    elements (bitmap bit unset) are replaced with ``NULL_HASH_SENTINEL``
    using ``SIMD.select()`` for branchless evaluation on both CPU and GPU.
    """
    comptime native = T.native
    comptime byte_width = size_of[Scalar[native]]()

    # Pre-compute seed (constant for all elements).
    # C: seed = 0; seed ^= rapid_mix(seed ^ secret[2], secret[1]); seed ^= len
    comptime seed = _rapid_mix(RAPID_SECRET2, RAPID_SECRET1) ^ UInt64(
        byte_width
    )

    @parameter
    @always_inline
    def _mul128_lo_hi[
        W: Int
    ](a: SIMD[DType.uint64, W], b: SIMD[DType.uint64, W]) -> Tuple[
        SIMD[DType.uint64, W], SIMD[DType.uint64, W]
    ]:
        """128-bit multiply returning (lo, hi) using 32-bit sub-products.

        GPU-compatible: avoids uint128 which Metal does not support.
        Decomposes a*b as: (a_hi*2^32 + a_lo) * (b_hi*2^32 + b_lo)
        """
        comptime lo32 = UInt64(0xFFFFFFFF)
        var a_lo = a & lo32
        var a_hi = a >> 32
        var b_lo = b & lo32
        var b_hi = b >> 32
        var t0 = a_lo * b_lo
        var t1 = a_lo * b_hi
        var t2 = a_hi * b_lo
        var t3 = a_hi * b_hi
        var mid = (t0 >> 32) + (t1 & lo32) + (t2 & lo32)
        var lo = (t0 & lo32) | (mid << 32)
        var hi = t3 + (t1 >> 32) + (t2 >> 32) + (mid >> 32)
        return (lo, hi)

    @parameter
    @always_inline
    def _mix[
        W: Int
    ](a: SIMD[DType.uint64, W], b: SIMD[DType.uint64, W]) -> SIMD[
        DType.uint64, W
    ]:
        """rapid_mix: 128-bit multiply then XOR halves."""
        var lo_hi = _mul128_lo_hi[W](a, b)
        return lo_hi[0] ^ lo_hi[1]

    @parameter
    @always_inline
    def process[
        W: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]) -> None:
        var i = idx[0]
        # Zero-extend to uint64 (matches C's rapid_read32/rapid_read64).
        # Mask to byte_width bits to prevent sign-extension for <8-byte types.
        comptime mask = ~UInt64(0) if byte_width >= 8 else (
            UInt64(1) << UInt64(byte_width * 8)
        ) - 1
        var vals = in_.load[W](i).cast[DType.uint64]() & mask
        # a = value ^ secret[1]; b = value ^ seed
        var a = vals ^ RAPID_SECRET1
        var b = vals ^ seed
        # rapid_mum(&a, &b): 128-bit multiply per SIMD lane
        var lo_hi = _mul128_lo_hi[W](a, b)
        # rapid_mix(a ^ secret[7], b ^ secret[1] ^ len)
        var hashes = _mix[W](
            lo_hi[0] ^ RAPID_SECRET7,
            lo_hi[1] ^ RAPID_SECRET1 ^ UInt64(byte_width),
        )

        # Inline null handling: if bitmap present, select sentinel for null lanes.
        if validity:
            var bv = validity.value()
            var abs_pos = bv.bit_offset() + i
            var byte_idx = abs_pos >> 3
            var bit_off = abs_pos & 7
            var bits = bv.load[DType.uint32](byte_idx)
            bits >>= UInt32(bit_off)
            var valid = (
                (SIMD[DType.uint32, W](bits) >> iota[DType.uint32, W]()) & 1
            ).cast[DType.bool]()
            hashes = valid.select(hashes, NULL_HASH_SENTINEL)

        (out_ptr + i).store(hashes)

    if ctx:
        comptime if has_accelerator():
            comptime gpu_width = simd_width_of[
                DType.uint64, target=get_gpu_target()
            ]()
            elementwise[process, gpu_width, target="gpu"](length, ctx.value())
        else:
            raise Error("rapidhash: no GPU accelerator available")
    else:
        comptime cpu_width = simd_byte_width() // size_of[
            Scalar[DType.uint64]
        ]()
        elementwise[process, cpu_width, target="cpu", use_blocking_impl=True](
            length
        )


def _rapidhash_bool_elementwise[
    out_origin: Origin[mut=True],
    v_origin: Origin[mut=False],
](
    output: BufferView[DType.uint64, out_origin],
    data: BitmapView[_],
    validity: Optional[BitmapView[v_origin]],
    length: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Vectorized bool rapidhash — views as params for GPU DevicePassable.

    Bool arrays are bit-packed so standard SIMD loads don't work. Instead,
    precompute the two possible hashes (for 0 and 1), load W data bits via
    the bitmap-mask pattern, and use ``SIMD.select()`` to pick the right hash.
    A second ``select()`` handles nulls. ``validity`` is None when all values
    are valid.
    """
    comptime hash_false = _rapidhash_fixed[size_of[Scalar[bool_.native]]()](
        UInt64(0)
    )
    comptime hash_true = _rapidhash_fixed[size_of[Scalar[bool_.native]]()](
        UInt64(1)
    )

    @parameter
    @always_inline
    def _load_bits[W: Int](bv: BitmapView[_], i: Int) -> SIMD[DType.bool, W]:
        """Load W consecutive bits from a bit-packed BitmapView."""
        var abs_pos = bv.bit_offset() + i
        var byte_idx = abs_pos >> 3
        var bit_off = abs_pos & 7
        var bits = bv.load[DType.uint32](byte_idx)
        bits >>= UInt32(bit_off)
        return (
            (SIMD[DType.uint32, W](bits) >> iota[DType.uint32, W]()) & 1
        ).cast[DType.bool]()

    @parameter
    @always_inline
    def process[
        W: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]) -> None:
        var i = idx[0]
        var data_bits = _load_bits[W](data, i)
        var hashes = data_bits.select(
            SIMD[DType.uint64, W](hash_true),
            SIMD[DType.uint64, W](hash_false),
        )
        if validity:
            var bv = validity.value()
            var valid = _load_bits[W](bv, i)
            hashes = valid.select(hashes, NULL_HASH_SENTINEL)
        output.store[W](i, hashes)

    if ctx:
        comptime if has_accelerator():
            comptime gpu_width = simd_width_of[
                DType.uint64, target=get_gpu_target()
            ]()
            elementwise[process, gpu_width, target="gpu"](length, ctx.value())
        else:
            raise Error("rapidhash: no GPU accelerator available for bool")
    else:
        comptime cpu_width = simd_byte_width() // size_of[
            Scalar[DType.uint64]
        ]()
        elementwise[process, cpu_width, target="cpu", use_blocking_impl=True](
            length
        )


def rapidhash(
    keys: BoolArray,
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[uint64]:
    """Vectorized rapidhash for bool arrays.

    Precomputes hash(false) and hash(true), loads data bits via the
    bitmap-mask pattern, and uses ``SIMD.select()`` for branchless dispatch.
    Works on both CPU and GPU.
    """
    var n = len(keys)
    var buf: Buffer[mut=True]
    if ctx:
        buf = Buffer.alloc_device[DType.uint64](ctx.value(), n)
    else:
        buf = Buffer.alloc_uninit[uint64.native](n)

    _rapidhash_bool_elementwise(
        buf.view[DType.uint64](), keys.values(), keys.validity(), n, ctx
    )

    return PrimitiveArray[uint64](
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buf.to_immutable(),
    )


# FIXME: use the seeding from the Rust implementation
def rapidhash[
    T: DataType
](
    keys: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[uint64]:
    """Vectorized rapidhash for primitive arrays.

    Uses ``elementwise`` for SIMD processing (same pattern as arithmetic
    kernels). Each SIMD lane independently computes the rapidhash of one
    element. Null elements are replaced with ``NULL_HASH_SENTINEL`` inline
    using ``SIMD.select()``.

    Dispatches to GPU when ``ctx`` is provided, CPU SIMD otherwise.
    """
    var n = len(keys)

    var buf: Buffer[mut=True]
    if ctx:
        buf = Buffer.alloc_device[DType.uint64](ctx.value(), n)
    else:
        buf = Buffer.alloc_uninit[DType.uint64](n)

    var out_ptr = buf.view[DType.uint64]().unsafe_ptr()

    _rapidhash_elementwise[T](out_ptr, keys.values(), keys.validity(), n, ctx)

    return PrimitiveArray[uint64](
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buf.to_immutable(),
    )


def rapidhash(keys: StringArray) raises -> PrimitiveArray[uint64]:
    """Hash each element of a string array.

    Uses AHash for variable-length strings (rapidhash for strings requires
    the full multi-branch rapidhash_internal — future work).
    """
    var n = len(keys)
    var builder = PrimitiveBuilder[uint64](capacity=n)
    var has_bitmap = Bool(keys.bitmap)

    for i in range(n):
        if has_bitmap and not keys.bitmap.value().test(keys.offset + i):
            builder.unsafe_append(_h(NULL_HASH_SENTINEL))
        else:
            builder.unsafe_append(_h(_hash(String(keys.unsafe_get(UInt(i))))))

    return builder.finish()


def _combine_elementwise(
    out_ptr: UnsafePointer[Scalar[DType.uint64], MutAnyOrigin],
    lhs: BufferView[DType.uint64, _],
    rhs: BufferView[DType.uint64, _],
    length: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Element-wise hash combine — views as params for GPU DevicePassable."""

    @parameter
    @always_inline
    def process[
        W: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]) -> None:
        var i = idx[0]
        var existing = lhs.load[W](i)
        var new = rhs.load[W](i)
        var mixed = existing ^ UInt64(0x9E3779B97F4A7C15)
        # GPU-compatible 128-bit multiply (no uint128).
        comptime lo32 = UInt64(0xFFFFFFFF)
        var a_lo = mixed & lo32
        var a_hi = mixed >> 32
        var b_lo = new & lo32
        var b_hi = new >> 32
        var t0 = a_lo * b_lo
        var t1 = a_lo * b_hi
        var t2 = a_hi * b_lo
        var t3 = a_hi * b_hi
        var mid = (t0 >> 32) + (t1 & lo32) + (t2 & lo32)
        var lo = (t0 & lo32) | (mid << 32)
        var hi = t3 + (t1 >> 32) + (t2 >> 32) + (mid >> 32)
        (out_ptr + i).store(lo ^ hi)

    if ctx:
        comptime if has_accelerator():
            comptime gpu_width = simd_width_of[
                DType.uint64, target=get_gpu_target()
            ]()
            elementwise[process, gpu_width, target="gpu"](length, ctx.value())
        else:
            raise Error("_combine_elementwise: no GPU accelerator available")
    else:
        comptime cpu_width = simd_byte_width() // size_of[
            Scalar[DType.uint64]
        ]()
        elementwise[process, cpu_width, target="cpu", use_blocking_impl=True](
            length
        )


def rapidhash(
    keys: StructArray,
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[uint64]:
    """Hash a struct array by combining per-field hashes column-wise.

    Each field is hashed independently via ``rapidhash(AnyArray)``
    and the results are combined element-wise using ``_rapid_mix``.
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
            buf = Buffer.alloc_device[DType.uint64](ctx.value(), n)
        else:
            buf = Buffer.alloc_uninit[uint64.native](n)
        var out_ptr = buf.view[DType.uint64]().unsafe_ptr()

        _combine_elementwise(
            out_ptr,
            result.buffer.view[DType.uint64](),
            field_hashes.buffer.view[DType.uint64](),
            n,
            ctx,
        )
        result = PrimitiveArray[uint64](
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
) raises -> PrimitiveArray[uint64]:
    """Runtime-typed rapidhash dispatch."""
    if keys.dtype() == bool_:
        return rapidhash(keys.as_bool(), ctx)

    comptime for dtype in numeric_dtypes:
        if keys.dtype() == dtype:
            return rapidhash[dtype](keys.as_primitive[dtype](), ctx)

    if keys.dtype().is_string():
        return rapidhash(keys.as_string())

    if keys.dtype().is_struct():
        return rapidhash(keys.as_struct(), ctx)

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
        if has_bitmap and not keys.bitmap.value().test(keys.offset + i):
            builder.unsafe_append(_h(NULL_HASH_SENTINEL))
        else:
            builder.unsafe_append(_h(Int(keys.unsafe_get(i)) + _OFFSET))

    return builder.finish()


def hash_identity(keys: BoolArray) raises -> PrimitiveArray[uint64]:
    """Identity hash for bool arrays (values 0 and 1)."""
    var n = len(keys)
    var builder = PrimitiveBuilder[uint64](capacity=n)
    var has_bitmap = Bool(keys.bitmap)
    for i in range(n):
        if has_bitmap and not keys.bitmap.value().test(keys.offset + i):
            builder.unsafe_append(_h(NULL_HASH_SENTINEL))
        else:
            builder.unsafe_append(_h(Int(keys[i])))
    return builder.finish()


def hash_identity(keys: AnyArray) raises -> PrimitiveArray[uint64]:
    """Runtime-typed identity hash dispatch."""
    if keys.dtype() == bool_:
        return hash_identity(keys.as_bool())
    if keys.dtype() == uint8:
        return hash_identity[uint8](keys.as_primitive[uint8]())
    if keys.dtype() == int8:
        return hash_identity[int8](keys.as_primitive[int8]())
    raise Error("hash_identity: only supports bool, uint8, int8")
