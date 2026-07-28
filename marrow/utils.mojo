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
from std.sys import (
    has_accelerator,
    CompilationTarget,
    get_defined_bool,
    size_of,
)


# The single switch for GPU code generation across marrow.  **Off by default**:
# GPU work is opt-in, so build with `-D MARROW_GPU=true` to get it.  With it
# off, every device path is eliminated at elaboration time — device allocations
# in the kernels, the accelerator arms of `_apply_dispatch`, and
# `has_accelerator_support`, which answers False and so makes a GPU
# `ExecutionContext` raise at the dispatch site rather than misbehave.  Applies
# to `mojo build` / `mojo run`; `mojo precompile` rejects `-D` outright.
#
# This is marrow's largest single compile-time lever.  Cold builds (fresh
# `MODULAR_CACHE_DIR` — the transform cache makes a repeated identical compile
# useless as a measurement):
#
#                        GPU off (default)   GPU on
#   cast, numeric x numeric      14.6s        40.1s
#   cast + sort_indices          43.7s        85.0s
#
# **Both halves have to be gated to get any of it.** Device paths only vanish
# when the allocations are wrapped in `comptime if GPU_ENABLED` *and*
# `has_accelerator_support` answers False.  Gating either one alone measures as
# no change at all (45.2s and 84.4s respectively against 42.5s / 84.1s
# baselines), which is why this looked like a dead end for a long time.  So:
# anything that touches device code needs a `comptime if GPU_ENABLED` around
# it; a runtime `if ctx.is_gpu()` cannot be eliminated at elaboration time and
# silently keeps the whole device path alive.
#
# It does not shed the MAX runtime, though — a binary built with the flag off
# still links `libmax` / AsyncRT exactly as one built with it on.
comptime GPU_ENABLED = get_defined_bool["MARROW_GPU", False]()


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
    def checked[T: DType](data: Span[UInt8, _], pos: Int) raises -> Scalar[T]:
        """`fixed`, but raising when the read would run past the end.

        The bounds-checked form belongs here rather than being re-derived by
        each caller: a format parser reads *untrusted* offsets, so "raise rather
        than read past the end" is a property of the read, not of any one
        parser. `ipc.mojo` had its own `_read_le` doing exactly this over a
        `List`, which also pinned its buffers to `List` and made a memory-mapped
        source impossible.
        """
        if pos < 0 or pos + size_of[Scalar[T]]() > len(data):
            raise Error(
                "LittleEndian.checked: ",
                size_of[Scalar[T]](),
                "-byte read at ",
                pos,
                " is out of bounds for ",
                len(data),
                " bytes",
            )
        return Self.fixed[T](data, pos)

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

    GPU codegen is **opt-in**: this answers False unless the build passes
    `-D MARROW_GPU=true` (see `GPU_ENABLED`). Without it a GPU
    `ExecutionContext` raises at the dispatch site rather than misbehaving.
    Applies to `mojo build`/`mojo run` only — `mojo precompile` rejects `-D`.

    Answering False here is one half of eliminating GPU codegen; the other half
    is the `comptime if GPU_ENABLED` guards around the kernels' device
    allocations. Either half alone measures as no improvement whatsoever — this
    call returning False while the allocations stay behind a runtime
    `if ctx.is_gpu()` was 45.2s against a 42.5s baseline. Together they take the
    same build to 14.6s. Do not "simplify" one side away.
    """
    comptime if not GPU_ENABLED:
        return False
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
# `array.dtype().dispatch_numeric[leaf]()` instead of an 11-way
# `if dtype == int8 ... elif ...` cascade. `func` receives the runtime dtype
# resolved to its concrete comptime type `T`; the return type `R` is inferred
# from `func`. A dtype outside the family raises (the family trait bound filters
# it out, so no arm matches) — the aggregate boundary relies on this to reject
# non-numeric columns catchably.
#
# One member per dtype family trait in `dtypes.mojo`, so a kernel never has to
# spell out its own ladder: adding a dtype to `AnyDataType.VariantType` extends
# every family it conforms to at once. Pick the *narrowest* family that covers
# the leaf — each member instantiates `func` once per conforming variant arm, so
# `AnyDataType.dispatch_primitive` costs roughly twice `AnyDataType.dispatch_numeric` in code
# size.
# ---------------------------------------------------------------------------
