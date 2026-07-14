"""Cast kernels — convert an array from one Arrow type to another.

Cast is a **two-level dispatcher**:

- **Level 1 — between families** (the top-level ``cast`` function): inspects the
  source and target type families and delegates to the matching family struct.
- **Level 2 — within a family** (each family struct's ``dispatch``): the typed,
  monomorphized switch that instantiates the concrete ``[From, To]`` kernel.

One struct per family:

- ``NumericCast`` — numeric ↔ numeric via SIMD ``pop.cast``; also carries the
  ``core[In, Out, W]`` lane functor reused by the fused AOT ``Cast`` node.
- ``BoolCast`` — bit-packed bool ↔ numeric.
- ``TemporalCast`` — temporal reinterpret + unit scaling.
- ``StringCast`` — string ↔ numeric/bool (per-element parse / format).
- ``NullCast`` — null → any (all-null array of the target type).

Future families (decimal, dictionary, list/struct, binary, string ↔ temporal)
slot into the level-1 dispatcher as one more ``elif`` each without touching the
existing structs.

Semantics
---------
``safe=False`` is the raw ``SIMD.cast`` fast path: float→int truncates toward
zero, integer narrowing wraps (two's-complement), no overflow saturation
(NaN/inf/out-of-range are undefined), matching ``numpy.astype``. ``safe=True``
(the default, matching PyArrow) runs a separate verification pass that raises on
any lossy conversion. Validity (the null bitmap) is always preserved unchanged.
"""

from std.algorithm.backend.vectorize import vectorize
from std.collections.string import atol, atof, StringSlice
from std.sys import bit_width_of, size_of
from std.sys.info import simd_byte_width

from ..arrays import (
    AnyArray,
    ArrayData,
    BoolArray,
    PrimitiveArray,
    StringArray,
)
from ..buffers import Buffer, Bitmap
from ..builders import AnyBuilder, StringBuilder
from ..utils import variant_dispatch_raises
from ..views import apply
from ..dtypes import (
    AnyDataType,
    DType,
    NumericType,
    TimeUnit,
)
from .helpers import Kernel
from .execution import ExecutionContext


# Predicate restricting ``variant_dispatch_raises`` to the numeric dtype
# variants — resolves a runtime dtype to a concrete ``NumericType`` parameter.
comptime _IsNumeric[T: Movable] = conforms_to(T, NumericType)


# ---------------------------------------------------------------------------
# NumericCast — fixed-width SIMD family
# ---------------------------------------------------------------------------


struct NumericCast(Kernel):
    """Numeric ↔ numeric cast: one ``pop.cast`` per SIMD lane."""

    comptime name = "cast"

    @always_inline
    @staticmethod
    def core[In: DType, Out: DType, W: Int](a: SIMD[In, W]) -> SIMD[Out, W]:
        return a.cast[Out]()

    @staticmethod
    def _needs_check[In: DType, Out: DType]() -> Bool:
        """Whether the (In, Out) pair can lose information and so needs a
        safe-mode verification pass. Provably-exact pairs return False so the
        check is dead-code-eliminated and safe casts stay branchless."""
        comptime if In == Out:
            return False
        elif In.is_floating_point() and Out.is_floating_point():
            return False  # float → float always allowed (may round / go to inf)
        elif In.is_floating_point():
            return True  # float → int: fractional or out-of-range
        elif Out.is_floating_point():
            # int → float: exact iff the integer's bit width fits the mantissa.
            return bit_width_of[In]() > DType.mantissa_width[Out]() + 1
        elif In.is_signed() == Out.is_signed():
            return bit_width_of[Out]() < bit_width_of[In]()  # int→int narrowing
        elif not In.is_signed():
            return bit_width_of[Out]() <= bit_width_of[In]()  # uint → signed
        else:
            return True  # signed → unsigned: negatives always overflow

    @staticmethod
    def _raise_loss[
        From: NumericType, To: NumericType
    ](array: PrimitiveArray[From]) raises:
        """Scan for the first valid value not representable as ``To`` and raise
        with it. Called after a vectorized check flags a failure."""
        comptime In = From.native
        comptime Out = To.native
        for j in range(len(array)):
            if array.is_valid(j):
                var v = array.unsafe_get(j)
                if v.cast[Out]().cast[In]() != v:
                    raise Error(
                        t"cast: value {v} out of range for {AnyDataType(To())}"
                    )
        raise Error(t"cast: value out of range for {AnyDataType(To())}")

    @staticmethod
    def _verify[
        From: NumericType, To: NumericType
    ](array: PrimitiveArray[From]) raises:
        """Raise if any valid element of ``array`` is not exactly representable
        as ``To`` (vectorized round-trip + mask-reduce). Used on the GPU path,
        where the cast runs on-device and the check runs on the host input; the
        CPU path fuses this check into the cast pass (see ``apply``)."""
        comptime In = From.native
        comptime Out = To.native
        comptime if Self._needs_check[In, Out]():
            var length = len(array)
            var src = array.values()
            var validity = array.validity()
            comptime w = max(1, simd_byte_width() // size_of[Scalar[In]]())
            var acc = SIMD[DType.bool, w](fill=False)
            var i = 0
            while i + w <= length:
                var bad = (
                    src.load[w](i).cast[Out]().cast[In]().ne(src.load[w](i))
                )
                if validity:
                    bad = bad & validity.value().mask[w](i)
                acc = acc | bad
                i += w
            var failed = acc.reduce_or()
            while i < length:
                if not validity or validity.value().test(i):
                    var v = src.load[1](i)
                    if v.cast[Out]().cast[In]() != v:
                        failed = True
                i += 1
            if failed:
                Self._raise_loss[From, To](array)

    @staticmethod
    def apply[
        From: NumericType, To: NumericType
    ](
        array: PrimitiveArray[From],
        safe: Bool = True,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[To]:
        comptime In = From.native
        comptime Out = To.native
        var length = len(array)

        var buf: Buffer[mut=True]
        if ctx.is_gpu():
            buf = Buffer.alloc_device[Out](ctx.device.value(), length)
        else:
            buf = Buffer.alloc_uninit[Out](length)
        var dst = buf.view[Out](0, length)

        comptime if Self._needs_check[In, Out]():
            if safe and not ctx.is_gpu():
                # Fused single pass: cast + store + round-trip check, masking
                # null lanes. One memory pass instead of cast-then-verify.
                var src = array.values()
                var validity = array.validity()
                comptime w = max(1, simd_byte_width() // size_of[Scalar[In]]())
                var acc = SIMD[DType.bool, w](fill=False)
                var i = 0
                while i + w <= length:
                    var v = src.load[w](i)
                    var out = v.cast[Out]()
                    dst.store[w](i, out)
                    var bad = out.cast[In]().ne(v)
                    if validity:
                        bad = bad & validity.value().mask[w](i)
                    acc = acc | bad
                    i += w
                var failed = acc.reduce_or()
                while i < length:
                    var v = src.load[1](i)
                    var out = v.cast[Out]()
                    dst.store[1](i, out)
                    if (not validity or validity.value().test(i)) and out.cast[
                        In
                    ]() != v:
                        failed = True
                    i += 1
                if failed:
                    Self._raise_loss[From, To](array)
            else:
                # unsafe, or GPU (device store): branchless cast, then verify
                # separately on the host input when safe.
                apply[In, Out, Self.core[In, Out, _]](array.values(), dst, ctx)
                if safe:
                    Self._verify[From, To](array)
        else:
            # provably exact: never any check
            apply[In, Out, Self.core[In, Out, _]](array.values(), dst, ctx)

        return PrimitiveArray[To](
            length=length,
            nulls=array.nulls,
            offset=0,
            bitmap=array.bitmap,
            buffer=buf.to_immutable(),
        )

    @staticmethod
    def dispatch(
        array: AnyArray,
        to: AnyDataType,
        safe: Bool = True,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        """Level-2 within-family dispatch. Resolves both the runtime source and
        target dtypes to comptime ``NumericType`` parameters via
        ``variant_dispatch_raises`` (no hand-written 11×11 switch), then
        instantiates ``apply[From, To]``."""
        var src_dt = array.dtype()

        @parameter
        def on_source[From: NumericType](src: From) raises -> AnyArray:
            var typed = array.as_primitive[From]().copy()

            @parameter
            def on_target[To: NumericType](dst: To) raises -> AnyArray:
                return NumericCast.apply[From, To](typed, safe, ctx).to_any()

            return variant_dispatch_raises[
                NumericType, predicate=_IsNumeric, func=on_target
            ](to._v)

        return variant_dispatch_raises[
            NumericType, predicate=_IsNumeric, func=on_source
        ](src_dt._v)


# ---------------------------------------------------------------------------
# BoolCast — bit-packed bool ↔ numeric family
# ---------------------------------------------------------------------------


struct BoolCast(Kernel):
    """Cast between the bit-packed ``BoolArray`` and numeric arrays.

    ``numeric → bool`` maps ``x != 0``; ``bool → numeric`` maps ``True→1,
    False→0``. Always lossless, so ``safe`` is ignored. Validity is preserved.

    The ``core`` functors are ``@always_inline`` so a fused AOT expression node
    can inline them into a single vectorize loop, like ``NumericCast.core``.
    """

    comptime name = "cast_bool"

    @always_inline
    @staticmethod
    def core_to_bool[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.ne(0)

    @always_inline
    @staticmethod
    def core_from_bool[
        Out: DType, W: Int
    ](m: SIMD[DType.bool, W]) -> SIMD[Out, W]:
        return m.cast[Out]()

    @staticmethod
    def num_to_bool[
        From: NumericType
    ](array: PrimitiveArray[From], ctx: ExecutionContext) raises -> BoolArray:
        var length = len(array)
        var result = Bitmap.alloc_device(
            ctx.device.value(), length
        ) if ctx.is_gpu() else Bitmap.alloc_uninit(length)
        apply[From.native, Self.core_to_bool[From.native, _]](
            array.values(), result.view(), ctx
        )
        return BoolArray(
            length=length,
            nulls=array.nulls,
            offset=0,
            bitmap=array.bitmap,
            buffer=result.to_immutable(),
        )

    @staticmethod
    def bool_to_num[
        To: NumericType
    ](array: BoolArray, ctx: ExecutionContext) raises -> PrimitiveArray[To]:
        comptime Out = To.native
        var length = len(array)
        var buf: Buffer[mut=True]
        if ctx.is_gpu():
            buf = Buffer.alloc_device[Out](ctx.device.value(), length)
        else:
            buf = Buffer.alloc_uninit[Out](length)
        apply[Out, Self.core_from_bool[Out, _]](
            array.values(), buf.view[Out](0, length), ctx
        )
        return PrimitiveArray[To](
            length=length,
            nulls=array.nulls,
            offset=0,
            bitmap=array.bitmap,
            buffer=buf.to_immutable(),
        )

    @staticmethod
    def dispatch(
        array: AnyArray,
        to: AnyDataType,
        safe: Bool = True,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        if array.dtype().is_bool():
            # bool → numeric (bool → bool identity is handled by the level-1
            # dispatcher's zero-copy fast path).
            var b = array.as_bool().copy()

            @parameter
            def to_numeric[To: NumericType](dst: To) raises -> AnyArray:
                return Self.bool_to_num[To](b, ctx).to_any()

            return variant_dispatch_raises[
                NumericType, predicate=_IsNumeric, func=to_numeric
            ](to._v)
        else:
            # numeric → bool
            if not to.is_bool():
                raise Error(t"cast: expected bool target, got {to}")
            var src_dt = array.dtype()

            @parameter
            def from_numeric[From: NumericType](src: From) raises -> AnyArray:
                return Self.num_to_bool(
                    array.as_primitive[From](), ctx
                ).to_any()

            return variant_dispatch_raises[
                NumericType, predicate=_IsNumeric, func=from_numeric
            ](src_dt._v)


# ---------------------------------------------------------------------------
# TemporalCast — reinterpret + unit scaling family
# ---------------------------------------------------------------------------


struct TemporalCast(Kernel):
    """Cast between temporal types and their underlying integers.

    - **temporal ↔ integer** (matching width): zero-copy reinterpret.
    - **temporal ↔ temporal** at the same resolution and width: reinterpret.
    - **temporal ↔ temporal** with differing units: scale the underlying
      integers by the ratio of their nanosecond resolutions (upscale multiplies,
      downscale integer-divides, truncating toward zero).

    Timezone is metadata only — a naive↔aware relabel changes no values.
    """

    comptime name = "cast_temporal"

    @staticmethod
    def _unit_ns(u: TimeUnit) -> Int64:
        """Nanoseconds per tick for a sub-second time unit."""
        if u.value == 0:
            return 1_000_000_000  # second
        elif u.value == 1:
            return 1_000_000  # millisecond
        elif u.value == 2:
            return 1_000  # microsecond
        else:
            return 1  # nanosecond

    @staticmethod
    def _ns_per_tick(dt: AnyDataType) raises -> Int64:
        """Nanoseconds represented by one tick of a temporal dtype."""
        if dt.is_date32():
            return 86_400_000_000_000  # days
        elif dt.is_date64():
            return 1_000_000  # milliseconds
        elif dt.is_timestamp():
            return Self._unit_ns(dt.as_timestamp().unit)
        elif dt.is_duration():
            return Self._unit_ns(dt.as_duration().unit)
        elif dt.is_time32():
            return Self._unit_ns(dt.as_time32().unit)
        elif dt.is_time64():
            return Self._unit_ns(dt.as_time64().unit)
        raise Error(t"cast: {dt} is not a temporal type")

    @staticmethod
    def _reinterpret(data: ArrayData, to: AnyDataType) raises -> AnyArray:
        """Relabel an integer/temporal buffer as ``to`` without moving data.
        Requires matching physical width."""
        return AnyArray.from_data(
            ArrayData(
                dtype=to.copy(),
                length=data.length,
                nulls=data.nulls,
                offset=data.offset,
                bitmap=data.bitmap,
                buffers=data.buffers.copy(),
                children=[],
            )
        )

    @staticmethod
    def _scale[
        SrcN: DType, DstN: DType
    ](
        data: ArrayData, to: AnyDataType, factor: Int64, up: Bool
    ) raises -> AnyArray:
        """Multiply (up) or integer-divide (down) each element by ``factor``,
        computing in int64 to avoid overflow, then narrow to ``DstN`` and
        relabel as ``to``."""
        var length = data.length
        var buf = Buffer.alloc_uninit[DstN](length)
        var src = data.buffers[0].view[SrcN](data.offset, length)
        var dst = buf.view[DstN](0, length)
        comptime width = max(
            1, simd_byte_width() // size_of[Scalar[DType.int64]]()
        )

        @parameter
        @always_inline
        def process[W: Int](i: Int) -> None:
            var v = src.load[W](i).cast[DType.int64]()
            var r = (v * factor) if up else (v // factor)
            dst.store[W](i, r.cast[DstN]())

        @always_inline
        def lane[W: Int](i: Int):
            process[W](i)

        vectorize[width](length, lane)
        return AnyArray.from_data(
            ArrayData(
                dtype=to.copy(),
                length=length,
                nulls=data.nulls,
                offset=0,
                bitmap=data.bitmap,
                buffers=[buf.to_immutable()],
                children=[],
            )
        )

    @staticmethod
    def dispatch(
        array: AnyArray,
        to: AnyDataType,
        safe: Bool = True,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        var src = array.dtype()
        var data = array.to_data()

        if src.is_integer() or to.is_integer():
            # temporal ↔ integer reinterpret (widths must match)
            if src.byte_width() != to.byte_width():
                raise Error(
                    t"cast: cannot reinterpret {src} as {to} (width mismatch)"
                )
            return Self._reinterpret(data, to)

        # temporal ↔ temporal
        var ns_from = Self._ns_per_tick(src)
        var ns_to = Self._ns_per_tick(to)
        if ns_from == ns_to and src.byte_width() == to.byte_width():
            return Self._reinterpret(data, to)

        var up = ns_from > ns_to
        var factor = (ns_from // ns_to) if up else (ns_to // ns_from)
        if src.byte_width() == 4:
            if to.byte_width() == 4:
                return Self._scale[DType.int32, DType.int32](
                    data, to, factor, up
                )
            else:
                return Self._scale[DType.int32, DType.int64](
                    data, to, factor, up
                )
        else:
            if to.byte_width() == 4:
                return Self._scale[DType.int64, DType.int32](
                    data, to, factor, up
                )
            else:
                return Self._scale[DType.int64, DType.int64](
                    data, to, factor, up
                )


# ---------------------------------------------------------------------------
# StringCast — string ↔ numeric family (designed; parse/format is future work)
# ---------------------------------------------------------------------------


struct StringCast(Kernel):
    """Cast between UTF-8 strings and numeric/bool types.

    Variable-length, so this is builder-based rather than SIMD:

    - **string → numeric**: per-element parse (``atol``/``atof``); an unparseable
      value raises (``safe=True``) or nulls the slot (``safe=False``).
    - **string → bool**: ``"true"``/``"false"``/``"1"``/``"0"`` (case-insensitive).
    - **numeric/bool → string**: per-element format into a ``StringBuilder``
      (``bool`` → ``"true"``/``"false"``).

    ``string ↔ temporal`` / ``decimal`` is not yet implemented.
    """

    comptime name = "cast_string"

    @staticmethod
    def _parse[native: DType](s: StringSlice) raises -> Scalar[native]:
        comptime if native.is_floating_point():
            return atof(s).cast[native]()
        else:
            return Scalar[native](atol(s))

    @staticmethod
    def string_to_num[
        To: NumericType
    ](array: StringArray, safe: Bool) raises -> PrimitiveArray[To]:
        comptime native = To.native
        var n = len(array)
        var buf = Buffer.alloc_zeroed[native](n)
        var view = buf.view[native](0, n)
        var valid = Bitmap.alloc_zeroed(n)
        var nulls = 0
        for i in range(n):
            if not array.is_valid(i):
                nulls += 1
                continue
            var s = array.unsafe_get(UInt(i))
            var value = Scalar[native](0)
            var ok = True
            try:
                value = Self._parse[native](s)
            except:
                ok = False
            if ok:
                view.store[1](i, value)
                valid.set(i)
            elif safe:
                raise Error(t"cast: cannot parse '{s}' as {AnyDataType(To())}")
            else:
                nulls += 1
        var out_bitmap = Optional[Bitmap[]]()
        if nulls > 0:
            out_bitmap = valid.to_immutable()
        return PrimitiveArray[To](
            length=n,
            nulls=nulls,
            offset=0,
            bitmap=out_bitmap,
            buffer=buf.to_immutable(),
        )

    @staticmethod
    def string_to_bool(array: StringArray, safe: Bool) raises -> BoolArray:
        var n = len(array)
        var data = Bitmap.alloc_zeroed(n)
        var valid = Bitmap.alloc_zeroed(n)
        var nulls = 0
        for i in range(n):
            if not array.is_valid(i):
                nulls += 1
                continue
            var s = String(array.unsafe_get(UInt(i))).lower()
            if s == "true" or s == "1":
                data.set(i)
                valid.set(i)
            elif s == "false" or s == "0":
                valid.set(i)
            elif safe:
                raise Error(t"cast: cannot parse '{s}' as bool")
            else:
                nulls += 1
        var out_bitmap = Optional[Bitmap[]]()
        if nulls > 0:
            out_bitmap = valid.to_immutable()
        return BoolArray(
            length=n,
            nulls=nulls,
            offset=0,
            bitmap=out_bitmap,
            buffer=data.to_immutable(),
        )

    @staticmethod
    def num_to_string[
        From: NumericType
    ](array: PrimitiveArray[From]) raises -> StringArray:
        var b = StringBuilder(len(array))
        for i in range(len(array)):
            if array.is_valid(i):
                b.append(String(array.unsafe_get(i)))
            else:
                b.append_null()
        return b.finish()

    @staticmethod
    def bool_to_string(array: BoolArray) raises -> StringArray:
        var b = StringBuilder(len(array))
        for i in range(len(array)):
            if array.is_valid(i):
                b.append("true" if array[i].value() else "false")
            else:
                b.append_null()
        return b.finish()

    @staticmethod
    def dispatch(
        array: AnyArray,
        to: AnyDataType,
        safe: Bool = True,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        var src = array.dtype()
        if src.is_string():
            var s = array.as_string().copy()
            if to.is_bool():
                return Self.string_to_bool(s, safe).to_any()
            elif to.is_numeric():

                @parameter
                def to_num[To: NumericType](dst: To) raises -> AnyArray:
                    return Self.string_to_num[To](s, safe).to_any()

                return variant_dispatch_raises[
                    NumericType, predicate=_IsNumeric, func=to_num
                ](to._v)
            raise Error(t"cast: string → {to} is not supported")
        else:  # target is string
            if src.is_bool():
                return Self.bool_to_string(array.as_bool()).to_any()
            elif src.is_numeric():

                @parameter
                def from_num[From: NumericType](f: From) raises -> AnyArray:
                    return Self.num_to_string(
                        array.as_primitive[From]()
                    ).to_any()

                return variant_dispatch_raises[
                    NumericType, predicate=_IsNumeric, func=from_num
                ](src._v)
            raise Error(t"cast: {src} → string is not supported")


# ---------------------------------------------------------------------------
# NullCast — null → any (all-null of the target type)
# ---------------------------------------------------------------------------


struct NullCast(Kernel):
    """Cast a null array to any target type: an all-null array of that type."""

    comptime name = "cast_null"

    @staticmethod
    def dispatch(
        array: AnyArray,
        to: AnyDataType,
        safe: Bool = True,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        var n = len(array)
        var b = AnyBuilder(to.copy(), capacity=n)
        for _ in range(n):
            b.append_null()
        return b.finish()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cast[
    From: NumericType, To: NumericType
](
    array: PrimitiveArray[From],
    safe: Bool = True,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[To]:
    """Typed numeric cast: ``cast[Int32Type, Float64Type](arr)``."""
    return NumericCast.apply[From, To](array, safe, ctx)


def cast(
    array: AnyArray,
    to: AnyDataType,
    safe: Bool = True,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Cast ``array`` to dtype ``to`` (level-1 family dispatcher).

    ``safe=True`` (default) raises on any lossy conversion; ``safe=False`` uses
    the raw truncating/wrapping ``SIMD.cast`` fast path.
    """
    var src = array.dtype()
    if src == to:
        return array.copy()  # identity → zero-copy
    elif src.is_null():
        return NullCast.dispatch(array, to, safe, ctx)  # null → any
    elif src.is_string() or to.is_string():
        # string family first, so bool↔string / numeric↔string route here
        # rather than to the bool/numeric families.
        return StringCast.dispatch(array, to, safe, ctx)
    elif src.is_numeric() and to.is_numeric():
        return NumericCast.dispatch(array, to, safe, ctx)
    elif src.is_bool() or to.is_bool():
        return BoolCast.dispatch(array, to, safe, ctx)
    elif src.is_temporal() or to.is_temporal():
        return TemporalCast.dispatch(array, to, safe, ctx)
    raise Error(t"cast: unsupported cast {src} -> {to}")
