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
from std.sys.info import _accelerator_arch


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


# `_TypePredicateGenerator` was moved from a top-level alias in
# `std.builtin.variadics` to a member of `TypeList` in Mojo 1.0.0b1; redeclare
# the underlying MLIR generator type here so our variant-dispatch helpers keep
# accepting an arbitrary type predicate.
comptime _TypePredicateGenerator[T: type_of(AnyType)] = __mlir_type[
    `!lit.generator<<"Type": `,
    T,
    `>`,
    Bool,
    `>`,
]


def has_accelerator_support[*dtypes: DType]() -> Bool:
    """Check if there is accelerator support for all given dtypes.

    For example Metal doesn't support float64 as of April 2026.

    Also guards against Mojo toolchain regressions where the GPU architecture
    string is malformed (e.g. 'metal:2-metal4' on an M2 with Metal 4 API).
    The valid Metal targets are 'metal:1'–'metal:4'; anything else indicates
    the toolchain cannot compile GPU kernels for this device and we fall back
    to CPU.
    """
    if not has_accelerator():
        return False
    if not CompilationTarget.is_apple_silicon():
        return True
    # Validate the GPU architecture string before attempting to compile any
    # GPU kernel.  A malformed target (e.g. 'metal:2-metal4') causes a hard
    # constraint failure deep inside simd_width_of, so we gate it out here.
    comptime arch = _accelerator_arch()
    comptime if (
        arch != "metal:1"
        and arch != "metal:2"
        and arch != "metal:3"
        and arch != "metal:4"
    ):
        return False
    comptime for dtype in dtypes:
        if dtype == DType.float64:
            return False
    return True


comptime _always_true[T: Movable] = True


def variant_dispatch[
    R: AnyType,
    //,
    Trait: type_of(AnyType),
    *Ts: Movable,
    predicate: _TypePredicateGenerator[Movable] = _always_true,
    func: def[T: Trait](T) capturing[_] -> R,
](ref v: Variant[*Ts]) -> R:
    """Dispatch *func* to the active type in *v*, reinterpreted as *Trait*.

    Only types matching *predicate* are dispatched. Defaults to all types,
    so passing no predicate covers the full variant.
    """
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        comptime if predicate[T]:
            if v.isa[T]():
                comptime assert conforms_to(T, Trait)
                return func(rebind[downcast[T, Trait]](v[T]))
    abort("unreachable: variant_dispatch")


def variant_dispatch_raises[
    R: AnyType,
    //,
    Trait: type_of(AnyType),
    *Ts: Movable,
    predicate: _TypePredicateGenerator[Movable] = _always_true,
    func: def[T: Trait](T) raises capturing[_] -> R,
](ref v: Variant[*Ts]) raises -> R:
    """Like *variant_dispatch* but *func* may raise."""
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        comptime if predicate[T]:
            if v.isa[T]():
                comptime assert conforms_to(T, Trait)
                return func(rebind[downcast[T, Trait]](v[T]))
    abort("unreachable: variant_dispatch_raises")


# TODO: using `ref v` should support both `read` and `mut` args but the compiler crashes
def variant_dispatch_raises[
    R: AnyType,
    //,
    Trait: type_of(AnyType),
    *Ts: Movable,
    predicate: _TypePredicateGenerator[Movable] = _always_true,
    func: def[T: Trait](mut T) raises capturing[_] -> R,
](mut v: Variant[*Ts]) raises -> R:
    """Like *variant_dispatch_raises* but *func* takes a mutable reference."""
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        comptime if predicate[T]:
            if v.isa[T]():
                comptime assert conforms_to(T, Trait)
                return func(rebind[downcast[T, Trait]](v[T]))
    abort("unreachable: variant_dispatch_raises")
