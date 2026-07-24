"""Generic Variant dispatch utilities.

These helpers drive runtime dispatch over a `Variant[*Ts]` without dynamic
dispatch or vtables.  The active type is detected via `v.isa[T]()` in a
compile-time loop; the value is then reinterpreted as *Trait* through
`rebind[downcast[T, Trait]]` (guarded by a `comptime assert conforms_to`
witness) and forwarded to *func*.

Three overloads are provided — distinguished by whether *func* raises and
whether it takes its argument by value or by mutable reference:

  variant_dispatch            — *func* is non-raising, argument by ref
  variant_dispatch_raises     — *func* raises,         argument by ref
  variant_dispatch_raises     — *func* raises,         argument by mut-ref

Note: a single `ref[_] v` overload would unify all three, but the Mojo
compiler currently crashes when `ref[_]` is used here (tracked as a TODO).
"""

from std.utils import Variant
from std.builtin.rebind import downcast
from std.os import abort
from std.sys import has_accelerator, CompilationTarget, size_of

from .dtypes import (
    AnyDataType,
    BinaryLikeType,
    FloatingType,
    NumericType,
    StringLikeType,
)


# ---------------------------------------------------------------------------
# Little-endian byte / bit / varint primitives
#
# The low-level serialization helpers shared by the Arrow IPC (FlatBuffers) and
# Parquet (Thrift / page) codecs. Fixed-width scalars are read/written as
# little-endian bytes independent of the host byte order: the
# `from_bytes[big_endian=False]` read and the shift/mask write both assemble LE
# bytes numerically, so no host byteswap is needed. Stateless — a namespace of
# static methods rather than free functions.
# ---------------------------------------------------------------------------


struct LittleEndian:
    """Little-endian byte, bit, and LEB128-varint reads/writes over a byte span.
    """

    @staticmethod
    def fixed[T: DType](data: Span[UInt8, _], pos: Int) -> Scalar[T]:
        """Read a `T`-width little-endian scalar at byte `pos`. Not bounds-checked
        — callers validate `pos` (matches the raw span reads in the hot decode
        paths)."""
        comptime W = size_of[Scalar[T]]()
        var arr = InlineArray[UInt8, W](fill=0)
        for i in range(W):
            arr[i] = data[pos + i]
        return SIMD[T, 1].from_bytes[big_endian=False](arr)

    @staticmethod
    def write[T: DType](mut buf: List[UInt8], pos: Int, val: Scalar[T]):
        """Write `val` as `T`-width little-endian bytes into `buf` at `pos` (the
        destination slots must already exist)."""
        comptime for i in range(size_of[Scalar[T]]()):
            buf[pos + i] = (val >> Scalar[T](i * 8)).cast[DType.uint8]()

    @staticmethod
    def append[T: DType](mut buf: List[UInt8], val: Scalar[T]):
        """Append `val` as `T`-width little-endian bytes to `buf`."""
        comptime for i in range(size_of[Scalar[T]]()):
            buf.append((val >> Scalar[T](i * 8)).cast[DType.uint8]())

    @staticmethod
    def u32(body: Span[UInt8, _], off: Int) -> Int:
        return Int(Self.fixed[DType.uint32](body, off))

    @staticmethod
    def put_u32(mut out: List[UInt8], v: Int):
        Self.append[DType.uint32](out, UInt32(v))

    @staticmethod
    def put_le(mut out: List[UInt8], bits: UInt64, width: Int):
        """Append the low `width` bytes of `bits`, least-significant first."""
        for i in range(width):
            out.append(UInt8((bits >> UInt64(i * 8)) & 0xFF))

    @staticmethod
    def varint(data: Span[UInt8, _], pos: Int) raises -> Tuple[UInt64, Int]:
        """Read an unsigned LEB128 varint at `pos`; return `(value, next_pos)`.
        """
        var result: UInt64 = 0
        var shift: Int = 0
        var p = pos
        while True:
            if p >= len(data):
                raise Error("varint out of bounds")
            var b = data[p]
            p += 1
            result |= UInt64(b & 0x7F) << UInt64(shift)
            if b & 0x80 == 0:
                break
            shift += 7
            if shift >= 64:
                raise Error("varint too long")
        return (result, p)

    @staticmethod
    def put_varint(mut out: List[UInt8], var v: UInt64):
        """Append `v` as an unsigned LEB128 varint."""
        while True:
            var b = UInt8(v & 0x7F)
            v >>= 7
            if v != 0:
                out.append(b | 0x80)
            else:
                out.append(b)
                break

    @staticmethod
    def bits(data: Span[UInt8, _], bit_offset: Int, nbits: Int) -> UInt64:
        """Read `nbits` starting at absolute `bit_offset`, least-significant
        first."""
        var result: UInt64 = 0
        for i in range(nbits):
            var abs_bit = bit_offset + i
            var byte_idx = abs_bit >> 3
            var bit_idx = abs_bit & 7
            var bit = (UInt64(data[byte_idx]) >> UInt64(bit_idx)) & 1
            result |= bit << UInt64(i)
        return result

    @staticmethod
    def bytes_less(a: Span[UInt8, _], b: Span[UInt8, _]) -> Bool:
        """Unsigned byte-wise lexicographic `a < b` (BYTE_ARRAY ordering)."""
        var n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return a[i] < b[i]
        return len(a) < len(b)


struct Crc32(Copyable, Movable):
    """Standard CRC-32 (reflected, polynomial `0xEDB88320`) — the ISO-3309 /
    zlib / gzip checksum Parquet uses for its optional per-page checksum.
    Incremental: `update` each byte span in order (Parquet v2 pages checksum the
    levels then the compressed values), then read `value`."""

    var _state: UInt32

    def __init__(out self):
        self._state = UInt32(0xFFFFFFFF)

    def update(mut self, data: Span[UInt8, _]):
        var crc = self._state
        for i in range(len(data)):
            crc ^= UInt32(data[i])
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ UInt32(0xEDB88320)
                else:
                    crc = crc >> 1
        self._state = crc

    def value(self) -> UInt32:
        return self._state ^ UInt32(0xFFFFFFFF)

    @staticmethod
    def compute(data: Span[UInt8, _]) -> UInt32:
        """CRC-32 of a single contiguous span."""
        var c = Self()
        c.update(data)
        return c.value()


def has_accelerator_support[*dtypes: DType]() -> Bool:
    """Check if there is accelerator support for all given dtypes.

    For example Metal doesn't support float64 as of April 2026.

    Must use `comptime if`, not runtime `if`: these guards have to eliminate
    the accelerator branches at elaboration time, not at runtime.

    Note `has_accelerator()` is itself `is_gpu() or _accelerator_arch() != ""`,
    so on a machine reporting an accelerator this enables GPU codegen — which
    since 1.0.0b3.dev2026072406 requires a MAX runtime (`lib/libmax.dylib`),
    hence the `max` dependency pinned alongside `mojo` in `pixi.toml`.

    A previous `_accelerator_arch()` check here validated the GPU architecture
    string, working around a toolchain regression that reported a malformed
    target (e.g. 'metal:2-metal4' on an M2 with the Metal 4 API). It has been
    removed; reinstate it if that regression reappears.
    """
    comptime if not has_accelerator():
        return False
    comptime if not CompilationTarget.is_apple_silicon():
        return True
    comptime for dtype in dtypes:
        if dtype == DType.float64:
            return False
    return True


def variant_dispatch[
    R: AnyType,
    //,
    Trait: type_of(AnyType),
    *Ts: Movable,
    func: def[T: Trait](T) capturing[_] -> R,
](ref v: Variant[*Ts]) -> R:
    """Dispatch *func* to the active type in *v*, reinterpreted as *Trait*.

    Only the variant types conforming to *Trait* are dispatched (the rest can't
    satisfy *func*'s `T: Trait` bound); pass a `Trait` every variant type
    conforms to (e.g. the container's own trait) to cover the full variant.
    """
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        comptime if conforms_to(T, Trait):
            if v.isa[T]():
                return func(rebind[downcast[T, Trait]](v[T]))
    abort("unreachable: variant_dispatch")


def variant_dispatch_raises[
    R: AnyType,
    //,
    Trait: type_of(AnyType),
    *Ts: Movable,
    func: def[T: Trait](T) raises capturing[_] -> R,
](ref v: Variant[*Ts]) raises -> R:
    """Like *variant_dispatch* but *func* may raise."""
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        comptime if conforms_to(T, Trait):
            if v.isa[T]():
                return func(rebind[downcast[T, Trait]](v[T]))
    raise Error(
        "variant_dispatch_raises: no arm matched the active variant type"
    )


# TODO: using `ref v` should support both `read` and `mut` args but the compiler crashes
def variant_dispatch_raises[
    R: AnyType,
    //,
    Trait: type_of(AnyType),
    *Ts: Movable,
    func: def[T: Trait](mut T) raises capturing[_] -> R,
](mut v: Variant[*Ts]) raises -> R:
    """Like *variant_dispatch_raises* but *func* takes a mutable reference."""
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        comptime if conforms_to(T, Trait):
            if v.isa[T]():
                return func(rebind[downcast[T, Trait]](v[T]))
    raise Error(
        "variant_dispatch_raises: no arm matched the active variant type"
    )


# ---------------------------------------------------------------------------
# Per-dtype-family dispatch adapters
#
# Thin wrappers over `variant_dispatch_raises` that fix the trait to a dtype
# family, so a kernel's runtime dispatch reads as
# `dispatch_over_numeric[leaf](array.dtype())` instead of an 11-way
# `if dtype == int8 ... elif ...` cascade. `func` receives the runtime dtype
# resolved to its concrete comptime type `T`; the return type `R` is inferred
# from `func`. A dtype outside the family raises (the family trait bound filters
# it out, so no arm matches) — the aggregate boundary relies on this to reject
# non-numeric columns catchably.
# ---------------------------------------------------------------------------


def dispatch_over_numeric[
    R: AnyType,
    //,
    func: def[T: NumericType](T) raises capturing[_] -> R,
](dt: AnyDataType) raises -> R:
    """Resolve a runtime numeric dtype to its comptime type and run `func`."""
    return variant_dispatch_raises[NumericType, func=func](dt._v)


def dispatch_over_floating[
    R: AnyType,
    //,
    func: def[T: FloatingType](T) raises capturing[_] -> R,
](dt: AnyDataType) raises -> R:
    """Resolve a runtime floating dtype to its comptime type and run `func`."""
    return variant_dispatch_raises[FloatingType, func=func](dt._v)


def dispatch_over_stringlike[
    R: AnyType,
    //,
    func: def[T: StringLikeType](T) raises capturing[_] -> R,
](dt: AnyDataType) raises -> R:
    """Resolve a runtime string-like dtype to its comptime type and run `func`.
    """
    return variant_dispatch_raises[StringLikeType, func=func](dt._v)


def dispatch_over_binarylike[
    R: AnyType,
    //,
    func: def[T: BinaryLikeType](T) raises capturing[_] -> R,
](dt: AnyDataType) raises -> R:
    """Resolve a runtime binary-like dtype to its comptime type and run `func`.
    """
    return variant_dispatch_raises[BinaryLikeType, func=func](dt._v)
