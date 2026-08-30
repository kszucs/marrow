"""Decimal casts — one struct per conversion, not one struct per dtype pair.

Split out of `cast.mojo`, which held every cast family in 1,575 lines. This is
the one with a hard internal boundary: six kernels and four helpers that talk
only to each other and to `CastKernel`, and the block carries its own
measured size argument (below) that applies to nothing else in that file.
"""

from .core import Kernel
from ..arrays import ArrayData, DynArray, PrimitiveArray
from ..buffers import Buffer
from ..views import apply, BitmapView
from ..execution import ExecContext
from ..dtypes import (
    DecimalType,
    DynType,
    FloatingType,
    IntegerType,
)
from .cast import CastKernel


# ---------------------------------------------------------------------------
# The decimal family — one struct per conversion, not one struct per dtype pair
# ---------------------------------------------------------------------------
#
# These five were a single `DecimalCastKernel` doing decimal↔decimal, decimal↔float
# and decimal↔integer behind one `_convert[FromN, ToN]`. That function resolved
# *both* sides over "decimal or numeric", so it was monomorphized across the
# full cross product — roughly 16 x 16 pairs — and every line inside it was
# stamped into all of them.
#
# It stayed cheap only while the bodies were three-line unchecked casts. Adding
# the overflow and truncation checks `safe` requires cost **+371,584 bytes of
# `__text` on `query_dynvalue`** for **1,476 bytes of source** — a 250x
# multiplier, and 85% of that commit's whole size regression. The `comptime if`
# ladder elided the dead *branches*; nothing shrank the *matrix*.
#
# Splitting by conversion narrows each kernel's dispatch to the families it can
# actually see, and two of the five collapse a runtime branch outright: an
# integer source is always scale 0, so `IntToDecimalKernel` only ever scales *up*,
# and an integer target is always scale 0, so `DecimalToIntKernel` only ever scales
# *down*. The fat version could not express that — it had to test `delta` at
# runtime on every path.


def _decimal_scale(dt: DynType) raises -> Int:
    """The decimal scale, or 0 for a plain integer/numeric.

    A `DynType` ladder rather than `dispatch_decimal` because it reads a
    *field*, and traits cannot require fields. It is guarded by `is_decimal()`
    so a decimal width the ladder does not know raises instead of silently
    reporting scale 0 — an unscaled decimal would be off by a factor of
    10^scale, with no error anywhere.
    """
    if not dt.is_decimal():
        return 0
    elif dt.is_decimal32():
        return dt.as_decimal32().scale
    elif dt.is_decimal64():
        return dt.as_decimal64().scale
    elif dt.is_decimal128():
        return dt.as_decimal128().scale
    elif dt.is_decimal256():
        return dt.as_decimal256().scale
    else:
        raise Error("decimal_cast: no scale known for decimal type ", dt)


def _pow10[T: DType](n: Int) -> Scalar[T]:
    var r = Scalar[T](1)
    for _ in range(n):
        r = r * 10
    return r


def _map_decimal[
    FromN: DType,
    ToN: DType,
    Op: def(Scalar[FromN]) raises -> Scalar[ToN],
](data: ArrayData, to: DynType, op: Op) raises -> DynArray:
    """Apply ``op`` to each element, writing a fresh ``ToN`` buffer relabelled
    as ``to``. Scalar (int128/256 aren't reliably SIMD-vectorizable).

    ``op`` raises so a checked conversion can report the offending value. That
    costs nothing structurally — this was already a serial loop, so there is no
    parallel or device boundary for the exception to cross, and it measured at
    0 bytes.

    Null lanes are skipped rather than mapped: an invalid slot may hold
    arbitrary junk, and a checked ``op`` would reject an array whose every
    *valid* element converts cleanly. ``NumericCastKernel.apply`` masks its checked
    lanes by validity for the same reason.
    """
    var n = data.length
    var src = data.buffers[0].view[FromN](data.offset, n)
    var out = Buffer.alloc_uninit[ToN](n)
    var dst = out.view[ToN]()
    var validity = Optional[BitmapView[origin_of(data.bitmap._value)]](None)
    if data.bitmap and data.nulls > 0:
        validity = data.bitmap.value().view(data.offset, n)
    for i in range(n):
        if validity and not validity.value()[i]:
            dst.store[1](i, Scalar[ToN](0))
        else:
            dst.store[1](i, op(src.load[1](i)))
    return DynArray.from_data(
        ArrayData(
            dtype=to.copy(),
            length=n,
            nulls=data.nulls,
            offset=0,
            bitmap=data.bitmap,
            buffers=[out.to_immutable()],
            children=[],
        )
    )


def _rescale_up[
    FromN: DType, ToN: DType
](data: ArrayData, to: DynType, delta: Int, safe: Bool) raises -> DynArray:
    """Multiply by 10^delta, widening the scale. Shared by `DecimalRescaleKernel` and
    `IntToDecimalKernel`, which is the only real overlap left between the five.
    """
    var f = _pow10[ToN](delta)

    def up(x: Scalar[FromN]) raises {imm} -> Scalar[ToN]:
        var v = x.cast[ToN]()
        if safe:
            # Two separate losses: narrowing to ToN, then the multiply.
            # Checking only the second would miss a value that never fit the
            # target's backing integer in the first place.
            if v.cast[FromN]() != x:
                raise Error("decimal_cast: value does not fit ", to)
            if v > Scalar[ToN].MAX // f or v < Scalar[ToN].MIN // f:
                raise Error("decimal_cast: rescaling overflows ", to)
        return v * f

    return _map_decimal[FromN, ToN](data, to, up)


def _rescale_down[
    FromN: DType, ToN: DType
](data: ArrayData, to: DynType, delta: Int, safe: Bool) raises -> DynArray:
    """Integer-divide by 10^-delta, narrowing the scale. Shared by
    `DecimalRescaleKernel` and `DecimalToIntKernel`."""
    var f = _pow10[FromN](-delta)

    def down(x: Scalar[FromN]) raises {imm} -> Scalar[ToN]:
        var q = x // f
        if safe:
            if x % f != 0:
                raise Error(
                    "decimal_cast: rescaling to ",
                    to,
                    " would discard a nonzero remainder",
                )
            if q.cast[ToN]().cast[FromN]() != q:
                raise Error("decimal_cast: value does not fit ", to)
        return q.cast[ToN]()

    return _map_decimal[FromN, ToN](data, to, down)


struct DecimalRescaleKernel(CastKernel):
    """Decimal → decimal: an integer rescale by 10^(to_scale − from_scale).

    The only member of the family whose direction is genuinely unknown until
    runtime, so it is the only one that still branches on `delta`."""

    comptime name = "decimal_rescale"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        var data = array.to_data()
        var delta = _decimal_scale(to) - _decimal_scale(array.dtype())

        def on_from[F: DecimalType](s: F) raises {imm} -> DynArray:
            def on_to[T: DecimalType](d: T) raises {imm} -> DynArray:
                if delta >= 0:
                    return _rescale_up[F.native, T.native](
                        data, to, delta, safe
                    )
                return _rescale_down[F.native, T.native](data, to, delta, safe)

            return to.dispatch_decimal(on_to)

        return array.dtype().dispatch_decimal(on_from)


struct IntToDecimalKernel(CastKernel):
    """Integer → decimal. An integer is scale 0, so this only ever scales *up*
    — the `delta < 0` arm the fat kernel carried here was unreachable."""

    comptime name = "int_to_decimal"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        var data = array.to_data()
        var delta = _decimal_scale(to)

        def on_from[F: IntegerType](s: F) raises {imm} -> DynArray:
            def on_to[T: DecimalType](d: T) raises {imm} -> DynArray:
                return _rescale_up[F.native, T.native](data, to, delta, safe)

            return to.dispatch_decimal(on_to)

        return array.dtype().dispatch_integer(on_from)


struct DecimalToIntKernel(CastKernel):
    """Decimal → integer. An integer target is scale 0, so this only ever
    scales *down*, and the remainder check is the whole of `safe` here."""

    comptime name = "decimal_to_int"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        var data = array.to_data()
        var delta = -_decimal_scale(array.dtype())

        def on_from[F: DecimalType](s: F) raises {imm} -> DynArray:
            def on_to[T: IntegerType](d: T) raises {imm} -> DynArray:
                if delta == 0:
                    return _rescale_up[F.native, T.native](data, to, 0, safe)
                return _rescale_down[F.native, T.native](data, to, delta, safe)

            return to.dispatch_integer(on_to)

        return array.dtype().dispatch_decimal(on_from)


struct FloatToDecimalKernel(CastKernel):
    """Float → decimal: multiply by 10^scale in float64 and round.

    float16/32 ↔ int128/256 has no direct compiler-rt path (`__fixhfti` /
    `__floattihf`), so float64 is the intermediary for every source width."""

    comptime name = "float_to_decimal"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        var data = array.to_data()
        var scale = _decimal_scale(to)

        def on_from[F: FloatingType](s: F) raises {imm} -> DynArray:
            def on_to[T: DecimalType](d: T) raises {imm} -> DynArray:
                comptime FromN = F.native
                comptime ToN = T.native
                var f = _pow10[DType.float64](scale)
                # int128/256 do not round-trip through float64 exactly, so this
                # bound is approximate at the last few ULPs. It guards against a
                # value being out by orders of magnitude, not a precision claim.
                var lo = Scalar[ToN].MIN.cast[DType.float64]()
                var hi = Scalar[ToN].MAX.cast[DType.float64]()

                def to_dec(x: Scalar[FromN]) raises {imm} -> Scalar[ToN]:
                    var scaled = round(x.cast[DType.float64]() * f)
                    # NaN fails both comparisons, which is the intent.
                    if safe and not (scaled >= lo and scaled <= hi):
                        raise Error("decimal_cast: value out of range for ", to)
                    return scaled.cast[ToN]()

                return _map_decimal[FromN, ToN](data, to, to_dec)

            return to.dispatch_decimal(on_to)

        return array.dtype().dispatch_floating(on_from)


struct DecimalToFloatKernel(CastKernel):
    """Decimal → float: divide by 10^scale in float64.

    Unchecked on purpose. Decimal → float is inexact by construction and Arrow
    permits it under `safe`, exactly as float → float is permitted
    (`NumericCastKernel.needs_check` answers False there too)."""

    comptime name = "decimal_to_float"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        var data = array.to_data()
        var scale = _decimal_scale(array.dtype())

        def on_from[F: DecimalType](s: F) raises {imm} -> DynArray:
            def on_to[T: FloatingType](d: T) raises {imm} -> DynArray:
                comptime FromN = F.native
                comptime ToN = T.native
                var f = _pow10[DType.float64](scale)

                def to_flt(x: Scalar[FromN]) raises {imm} -> Scalar[ToN]:
                    return (x.cast[DType.float64]() / f).cast[ToN]()

                return _map_decimal[FromN, ToN](data, to, to_flt)

            return to.dispatch_floating(on_to)

        return array.dtype().dispatch_decimal(on_from)


struct DecimalCastKernel(CastKernel):
    """Router for the decimal family: pick the kernel for this pair.

    Kept so `cast()`'s ladder has one decimal arm rather than five, and so the
    "which conversion is this?" question is answered in exactly one place. It
    holds no conversion logic of its own."""

    comptime name = "decimal_cast"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        var src = array.dtype()
        if src.is_decimal() and to.is_decimal():
            return DecimalRescaleKernel.dispatch(array, to, safe, ctx)
        elif src.is_decimal() and to.is_floating_point():
            return DecimalToFloatKernel.dispatch(array, to, safe, ctx)
        elif src.is_decimal() and to.is_integer():
            return DecimalToIntKernel.dispatch(array, to, safe, ctx)
        elif src.is_floating_point() and to.is_decimal():
            return FloatToDecimalKernel.dispatch(array, to, safe, ctx)
        elif src.is_integer() and to.is_decimal():
            return IntToDecimalKernel.dispatch(array, to, safe, ctx)
        else:
            raise Self.error(t"unsupported decimal cast {src} -> {to}")
