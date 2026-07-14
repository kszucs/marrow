"""Cast kernels — one slim ``Kernel`` struct per conversion.

Each kernel is a separate struct implementing ``Kernel`` and doing **one**
conversion, so it can be optimized and monomorphized in isolation (and grabbed
directly by the AOT expression layer):

- ``NumericCast`` — numeric ↔ numeric SIMD ``pop.cast``; ``core`` /
  ``core_checked`` are the lane functors (the latter reused by the fused AOT
  node). ``safe`` is a **comptime** parameter selecting checked vs unchecked.
- ``NumToBool`` / ``BoolToNum`` — bit-pack (``x != 0``) / bit-unpack (``True→1``).
- ``TemporalReinterpret`` / ``TemporalScale`` — relabel to the underlying
  integer, or unit-scale it.
- ``StringToNum`` / ``NumToString`` / ``StringToBool`` / ``BoolToString`` —
  per-element ``atol``/``atof`` parse or format (variable-length, builder-based).
- ``NullCast`` — an all-null array of the target type.

Runtime family / dtype / safe **dispatch** over these kernels is the separate
``Cast`` struct at the bottom (``safe`` resolved to a comptime parameter once,
then threaded through); the fused AOT node in ``marrow.expr.values`` bypasses it
and grabs ``NumericCast.core`` directly.
"""

from std.collections.string import atol, atof, StringSlice
from std.collections.string._utf8 import _is_valid_utf8
from std.sys import bit_width_of

from ..arrays import (
    AnyArray,
    ArrayData,
    BinaryLikeArray,
    BoolArray,
    FixedSizeBinaryArray,
    PrimitiveArray,
    StringArray,
)
from ..buffers import Buffer, Bitmap
from ..builders import AnyBuilder, BinaryLikeBuilder, FixedSizeBinaryBuilder
from ..utils import variant_dispatch_raises
from ..views import apply, apply_checked
from ..dtypes import (
    AnyDataType,
    BinaryLikeType,
    DType,
    FixedSizeBinaryType,
    NumericType,
    StringLikeType,
    TimeUnit,
)
from .helpers import Kernel
from .execution import ExecutionContext


# Restrict ``variant_dispatch_raises`` to a dtype family — each resolves a runtime
# dtype to a concrete trait-bound parameter for the nested cast dispatch.
comptime _IsNumeric[T: Movable] = conforms_to(T, NumericType)
comptime _IsBinaryLike[T: Movable] = conforms_to(T, BinaryLikeType)
comptime _IsStringLike[T: Movable] = conforms_to(T, StringLikeType)


# ---------------------------------------------------------------------------
# NumericCast — numeric ↔ numeric
# ---------------------------------------------------------------------------


struct NumericCast(Kernel):
    """Numeric ↔ numeric cast: one ``pop.cast`` per SIMD lane."""

    comptime name = "numeric_cast"

    @always_inline
    @staticmethod
    def core[In: DType, Out: DType, W: Int](a: SIMD[In, W]) -> SIMD[Out, W]:
        """Unchecked cast — one ``pop.cast``. Used by the unsafe / lossless path
        and the fused AOT ``Cast`` node."""
        return a.cast[Out]()

    @always_inline
    @staticmethod
    def core_checked[
        In: DType, Out: DType, W: Int
    ](a: SIMD[In, W]) -> Tuple[SIMD[Out, W], SIMD[DType.bool, W]]:
        """Checked cast — returns ``(out, bad)`` where ``bad`` marks lanes that
        don't round-trip. Casts forward **once** and reuses ``out`` for the
        back-cast, so the safe path does no redundant work."""
        var out = a.cast[Out]()
        return (out, out.cast[In]().ne(a))

    @staticmethod
    def needs_check[In: DType, Out: DType]() -> Bool:
        """Whether the (In, Out) pair can lose information and so needs a
        safe-mode check. Provably-exact pairs return False so the check is
        dead-code-eliminated and safe casts stay branchless."""
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
    def apply[
        From: NumericType, To: NumericType, safe: Bool = True
    ](
        array: PrimitiveArray[From],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[To]:
        """Cast ``array`` to ``To``. ``safe`` is a **comptime** parameter so the
        checked / unchecked kernel is selected at instantiation — usable from
        AOT-compiled fused expressions."""
        comptime In = From.native
        comptime Out = To.native
        var length = len(array)

        var buf: Buffer[mut=True]
        if ctx.is_gpu():
            buf = Buffer.alloc_device[Out](ctx.device.value(), length)
        else:
            buf = Buffer.alloc_uninit[Out](length)
        var dst = buf.view[Out](0, length)

        comptime if safe and Self.needs_check[In, Out]():
            # Fused checked map: cast + validate each lane in a single pass; the
            # map fails on the first unrepresentable value. Serial — a failing
            # block raises, which can't cross parallel / GPU boundaries.
            var validity = array.validity()
            if validity:
                apply_checked[In, Out, Self.core_checked[In, Out, _]](
                    array.values(), validity.value(), dst
                )
            else:
                apply_checked[In, Out, Self.core_checked[In, Out, _]](
                    array.values(), dst
                )
        else:
            # Unsafe, or provably-exact: one branchless, parallel / device pass.
            apply[In, Out, Self.core[In, Out, _]](array.values(), dst, ctx)

        return PrimitiveArray[To](
            length=length,
            nulls=array.nulls,
            offset=0,
            bitmap=array.bitmap,
            buffer=buf.to_immutable(),
        )


# ---------------------------------------------------------------------------
# NumToBool / BoolToNum — bit-packed bool ↔ numeric
# ---------------------------------------------------------------------------


struct NumToBool(Kernel):
    """numeric → bool: ``x != 0``, bit-packed. Lossless; validity preserved."""

    comptime name = "num_to_bool"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.ne(0)

    @staticmethod
    def apply[
        From: NumericType
    ](array: PrimitiveArray[From], ctx: ExecutionContext) raises -> BoolArray:
        var length = len(array)
        var result = Bitmap.alloc_device(
            ctx.device.value(), length
        ) if ctx.is_gpu() else Bitmap.alloc_uninit(length)
        apply[From.native, Self.core[From.native, _]](
            array.values(), result.view(), ctx
        )
        return BoolArray(
            length=length,
            nulls=array.nulls,
            offset=0,
            bitmap=array.bitmap,
            buffer=result.to_immutable(),
        )


struct BoolToNum(Kernel):
    """bool → numeric: ``True→1, False→0``. Lossless; validity preserved."""

    comptime name = "bool_to_num"

    @always_inline
    @staticmethod
    def core[Out: DType, W: Int](m: SIMD[DType.bool, W]) -> SIMD[Out, W]:
        return m.cast[Out]()

    @staticmethod
    def apply[
        To: NumericType
    ](array: BoolArray, ctx: ExecutionContext) raises -> PrimitiveArray[To]:
        comptime Out = To.native
        var length = len(array)
        var buf: Buffer[mut=True]
        if ctx.is_gpu():
            buf = Buffer.alloc_device[Out](ctx.device.value(), length)
        else:
            buf = Buffer.alloc_uninit[Out](length)
        apply[Out, Self.core[Out, _]](
            array.values(), buf.view[Out](0, length), ctx
        )
        return PrimitiveArray[To](
            length=length,
            nulls=array.nulls,
            offset=0,
            bitmap=array.bitmap,
            buffer=buf.to_immutable(),
        )


# ---------------------------------------------------------------------------
# TemporalReinterpret / TemporalScale
# ---------------------------------------------------------------------------


struct TemporalReinterpret(Kernel):
    """Relabel an integer/temporal buffer as ``to`` without moving data.
    Requires matching physical width (zero-copy)."""

    comptime name = "temporal_reinterpret"

    @staticmethod
    def apply(data: ArrayData, to: AnyDataType) raises -> AnyArray:
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


struct TemporalScale(Kernel):
    """Scale a temporal column's underlying integers by ``factor`` (multiply if
    ``up``, else integer-divide), computing in int64 to avoid overflow, then
    narrow to ``DstN`` and relabel as ``to``."""

    comptime name = "temporal_scale"

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
    def ns_per_tick(dt: AnyDataType) raises -> Int64:
        """Nanoseconds represented by one tick of a temporal dtype — drives the
        dispatcher's reinterpret-vs-scale choice and the scale factor."""
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
    def apply[
        SrcN: DType, DstN: DType
    ](
        data: ArrayData,
        to: AnyDataType,
        factor: Int64,
        up: Bool,
        ctx: ExecutionContext,
    ) raises -> AnyArray:
        var length = data.length
        var buf = Buffer.alloc_uninit[DstN](length)
        var src = data.buffers[0].view[SrcN](data.offset, length)

        @parameter
        @always_inline
        def scale[W: Int](v: SIMD[SrcN, W]) -> SIMD[DstN, W]:
            var x = v.cast[DType.int64]()
            return ((x * factor) if up else (x // factor)).cast[DstN]()

        apply[SrcN, DstN, scale](src, buf.view[DstN](0, length), ctx)
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


# ---------------------------------------------------------------------------
# StringToNum / NumToString / StringToBool / BoolToString
# ---------------------------------------------------------------------------


struct StringToNum(Kernel):
    """Parse strings to a numeric type. ``safe`` is comptime: safe=True raises on
    an unparseable value, safe=False nulls it — the dead branch is elided."""

    comptime name = "string_to_num"

    @staticmethod
    def _parse[native: DType](s: StringSlice) raises -> Scalar[native]:
        comptime if native.is_floating_point():
            return atof(s).cast[native]()
        else:
            return Scalar[native](atol(s))

    @staticmethod
    def apply[
        From: StringLikeType, To: NumericType, safe: Bool
    ](array: BinaryLikeArray[From]) raises -> PrimitiveArray[To]:
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
            else:
                comptime if safe:
                    raise Error(
                        t"cast: cannot parse '{s}' as {AnyDataType(To())}"
                    )
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


struct StringToBool(Kernel):
    """Parse ``"true"``/``"false"``/``"1"``/``"0"`` (case-insensitive) to bool.
    ``safe`` comptime: raise vs null on an unrecognized value."""

    comptime name = "string_to_bool"

    @staticmethod
    def apply[
        From: StringLikeType, safe: Bool
    ](array: BinaryLikeArray[From]) raises -> BoolArray:
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
            else:
                comptime if safe:
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


struct NumToString(Kernel):
    """Format a numeric array to strings (per-element ``String(value)``)."""

    comptime name = "num_to_string"

    @staticmethod
    def apply[
        From: NumericType, To: StringLikeType
    ](array: PrimitiveArray[From]) raises -> BinaryLikeArray[To]:
        var b = BinaryLikeBuilder[To](len(array))
        for i in range(len(array)):
            if array.is_valid(i):
                b.append(String(array.unsafe_get(i)))
            else:
                b.append_null()
        return b.finish()


struct BoolToString(Kernel):
    """Format a bool array to ``"true"``/``"false"`` strings."""

    comptime name = "bool_to_string"

    @staticmethod
    def apply[
        To: StringLikeType
    ](array: BoolArray) raises -> BinaryLikeArray[To]:
        var b = BinaryLikeBuilder[To](len(array))
        for i in range(len(array)):
            if array.is_valid(i):
                b.append("true" if array[i].value() else "false")
            else:
                b.append_null()
        return b.finish()


# ---------------------------------------------------------------------------
# BinaryLikeCast — binary/large_binary/utf8/large_utf8 ↔ each other
# ---------------------------------------------------------------------------


struct BinaryLikeCast(Kernel):
    """Cast between the binary-like containers (binary, large_binary, utf8,
    large_utf8). Equal physical offset width → a zero-copy relabel that shares
    the offset and value buffers; differing width (32↔64-bit offsets) → a rebuild
    through a builder. Producing a UTF-8 string from raw bytes validates every
    element under ``safe`` (raising on malformed UTF-8)."""

    comptime name = "binary_like_cast"

    @staticmethod
    def apply[
        From: BinaryLikeType, To: BinaryLikeType, safe: Bool
    ](array: BinaryLikeArray[From]) raises -> BinaryLikeArray[To]:
        comptime bytes_to_text = conforms_to(
            To, StringLikeType
        ) and not conforms_to(From, StringLikeType)
        comptime if safe and bytes_to_text:
            Self._check_utf8(array)
        comptime if From.offset == To.offset:
            # Identical physical layout → relabel only, no allocation.
            return BinaryLikeArray[To](
                length=array.length,
                nulls=array.nulls,
                offset=array.offset,
                bitmap=array.bitmap.copy(),
                offsets=array.offsets.copy(),
                values=array.values.copy(),
            )
        else:
            var b = BinaryLikeBuilder[To](len(array))
            for i in range(len(array)):
                if array.is_valid(i):
                    b.append(array.unsafe_get(UInt(i)))
                else:
                    b.append_null()
            return b.finish()

    @staticmethod
    def _check_utf8[From: BinaryLikeType](array: BinaryLikeArray[From]) raises:
        for i in range(len(array)):
            if array.is_valid(i) and not _is_valid_utf8(
                array.unsafe_get(UInt(i)).as_bytes()
            ):
                raise Error("cast: invalid UTF-8 in binary → string cast")


# ---------------------------------------------------------------------------
# FixedSizeBinaryCast — fixed_size_binary ↔ variable-length binary
# ---------------------------------------------------------------------------


struct FixedSizeBinaryCast(Kernel):
    """Cast fixed-size-binary ↔ variable-length binary. ``to_binary`` derives the
    offset buffer from the fixed width and shares the data bytes; ``from_binary``
    packs each element into a fixed cell, raising when a length ≠ the width."""

    comptime name = "fixed_size_binary_cast"

    @staticmethod
    def to_binary[
        To: BinaryLikeType
    ](array: FixedSizeBinaryArray) raises -> BinaryLikeArray[To]:
        var n = len(array)
        var w = array.byte_width
        var total = array.offset + n  # offsets cover the whole physical prefix
        comptime if To.offset == DType.int32:
            if total * w > Int(Int32.MAX):
                raise Error("cast: byte span too large for 32-bit offsets")
        var out = Buffer.alloc_uninit[To.offset](total + 1)
        var dst = out.view[To.offset]()
        for j in range(total + 1):
            dst.store[1](j, Scalar[To.offset](j * w))
        return BinaryLikeArray[To](
            length=n,
            nulls=array.nulls,
            offset=array.offset,
            bitmap=array.bitmap.copy(),
            offsets=out.to_immutable(),
            values=array.buffer.copy(),
        )

    @staticmethod
    def from_binary[
        From: BinaryLikeType
    ](
        array: BinaryLikeArray[From], byte_width: Int
    ) raises -> FixedSizeBinaryArray:
        var b = FixedSizeBinaryBuilder(byte_width, len(array))
        for i in range(len(array)):
            if array.is_valid(i):
                b.append(
                    array.unsafe_get(UInt(i)).as_bytes()
                )  # raises on width
            else:
                b.append_null()
        return b.finish()


# ---------------------------------------------------------------------------
# NullCast — null → any
# ---------------------------------------------------------------------------


struct NullCast(Kernel):
    """Cast a null array to any target type: an all-null array of that type."""

    comptime name = "null_cast"

    @staticmethod
    def apply(array: AnyArray, to: AnyDataType) raises -> AnyArray:
        var n = len(array)
        var b = AnyBuilder(to.copy(), capacity=n)
        for _ in range(n):
            b.append_null()
        return b.finish()


# ---------------------------------------------------------------------------
# DecimalCast — decimal ↔ decimal (rescale) and decimal ↔ numeric
# ---------------------------------------------------------------------------


struct DecimalCast(Kernel):
    """Cast decimal ↔ decimal (rescale) and decimal ↔ numeric.

    Integer-backed values — decimals, and plain integers taken at scale 0 —
    rescale by ``10^(to_scale − from_scale)``: multiply when the scale widens,
    truncating integer-divide when it narrows. Float conversions divide/multiply
    by ``10^scale`` in floating point. Arithmetic is unchecked (wrapping /
    truncating), matching the unsafe numeric path; validity is preserved.

    ``apply`` owns the whole intra-family dispatch: the decimal side resolves to
    its backing integer via ``_on_dec_native`` (int32/64/128/256), the numeric
    side via ``variant_dispatch_raises``, and the concrete ``[FromN, ToN]`` lane
    kernels (``_rescale`` / ``_to_float`` / ``_from_float``) do the work."""

    comptime name = "decimal_cast"

    @staticmethod
    def apply(
        data: ArrayData, frm: AnyDataType, to: AnyDataType
    ) raises -> AnyArray:
        var from_scale = Self._scale(frm)
        var to_scale = Self._scale(to)
        if frm.is_decimal() and to.is_decimal():

            @parameter
            def on_from[FN: DType]() raises -> AnyArray:
                @parameter
                def on_to[TN: DType]() raises -> AnyArray:
                    return Self._rescale[FN, TN](
                        data, to_scale - from_scale, to
                    )

                return Self._on_dec_native[on_to](to)

            return Self._on_dec_native[on_from](frm)
        elif frm.is_decimal():  # decimal → numeric

            @parameter
            def dec_to_num[FN: DType]() raises -> AnyArray:
                @parameter
                def on_num[To: NumericType](d: To) raises -> AnyArray:
                    comptime TN = To.native
                    comptime if TN.is_floating_point():
                        return Self._to_float[FN, TN](data, from_scale, to)
                    else:
                        return Self._rescale[FN, TN](data, -from_scale, to)

                return variant_dispatch_raises[
                    NumericType, predicate=_IsNumeric, func=on_num
                ](to._v)

            return Self._on_dec_native[dec_to_num](frm)
        elif to.is_decimal():  # numeric → decimal

            @parameter
            def num_to_dec[TN: DType]() raises -> AnyArray:
                @parameter
                def on_num[From: NumericType](s: From) raises -> AnyArray:
                    comptime FN = From.native
                    comptime if FN.is_floating_point():
                        return Self._from_float[FN, TN](data, to_scale, to)
                    else:
                        return Self._rescale[FN, TN](data, to_scale, to)

                return variant_dispatch_raises[
                    NumericType, predicate=_IsNumeric, func=on_num
                ](frm._v)

            return Self._on_dec_native[num_to_dec](to)
        raise Error(t"cast: unsupported decimal cast {frm} -> {to}")

    # -- scale + backing-integer resolution -----------------------------------

    @staticmethod
    def _scale(dt: AnyDataType) -> Int:
        if dt.is_decimal32():
            return dt.as_decimal32().scale
        elif dt.is_decimal64():
            return dt.as_decimal64().scale
        elif dt.is_decimal128():
            return dt.as_decimal128().scale
        elif dt.is_decimal256():
            return dt.as_decimal256().scale
        return 0  # a non-decimal integer is scale 0

    @staticmethod
    def _on_dec_native[
        func: def[N: DType]() raises capturing[_] -> AnyArray
    ](dt: AnyDataType) raises -> AnyArray:
        if dt.is_decimal32():
            return func[DType.int32]()
        elif dt.is_decimal64():
            return func[DType.int64]()
        elif dt.is_decimal128():
            return func[DType.int128]()
        else:
            return func[DType.int256]()

    # -- lane kernels ---------------------------------------------------------

    @staticmethod
    def _pow10[T: DType](n: Int) -> Scalar[T]:
        var r = Scalar[T](1)
        for _ in range(n):
            r = r * 10
        return r

    @staticmethod
    def _finish(
        var out: Buffer[mut=True], data: ArrayData, to: AnyDataType
    ) raises -> AnyArray:
        return AnyArray.from_data(
            ArrayData(
                dtype=to.copy(),
                length=data.length,
                nulls=data.nulls,
                offset=0,
                bitmap=data.bitmap,
                buffers=[out.to_immutable()],
                children=[],
            )
        )

    @staticmethod
    def _rescale[
        FromN: DType, ToN: DType
    ](data: ArrayData, delta: Int, to: AnyDataType) raises -> AnyArray:
        var n = data.length
        var src = data.buffers[0].view[FromN](data.offset, n)
        var out = Buffer.alloc_uninit[ToN](n)
        var dst = out.view[ToN]()
        if delta >= 0:
            var f = Self._pow10[ToN](delta)
            for i in range(n):
                dst.store[1](i, src.load[1](i).cast[ToN]() * f)
        else:
            var f = Self._pow10[FromN](-delta)
            for i in range(n):
                dst.store[1](i, (src.load[1](i) // f).cast[ToN]())
        return Self._finish(out^, data, to)

    @staticmethod
    def _to_float[
        FromN: DType, ToN: DType
    ](data: ArrayData, from_scale: Int, to: AnyDataType) raises -> AnyArray:
        # ``SIMD.cast`` does every conversion, but float16 ↔ int128/int256 needs
        # compiler-rt builtins (``__fixhfti`` / ``__floattihf``) that don't exist
        # for any width, so compute in float64 and let it narrow to float16.
        var n = data.length
        var src = data.buffers[0].view[FromN](data.offset, n)
        var out = Buffer.alloc_uninit[ToN](n)
        var dst = out.view[ToN]()
        var f = Self._pow10[DType.float64](from_scale)
        for i in range(n):
            dst.store[1](
                i, (src.load[1](i).cast[DType.float64]() / f).cast[ToN]()
            )
        return Self._finish(out^, data, to)

    @staticmethod
    def _from_float[
        FromN: DType, ToN: DType
    ](data: ArrayData, to_scale: Int, to: AnyDataType) raises -> AnyArray:
        var n = data.length
        var src = data.buffers[0].view[FromN](data.offset, n)
        var out = Buffer.alloc_uninit[ToN](n)
        var dst = out.view[ToN]()
        var f = Self._pow10[DType.float64](to_scale)
        for i in range(n):
            dst.store[1](
                i, round(src.load[1](i).cast[DType.float64]() * f).cast[ToN]()
            )
        return Self._finish(out^, data, to)


# ---------------------------------------------------------------------------
# Cast — runtime dispatcher over the kernels above
# ---------------------------------------------------------------------------


struct Cast(Kernel):
    """Runtime family / dtype / safe dispatch to the flat cast kernels.

    ``safe`` is resolved to a comptime parameter **once** (in ``apply``) and
    threaded through the family routers, so the checked / unchecked numeric and
    string kernels are picked at compile time with no per-call-site branch.
    """

    comptime name = "cast"

    @staticmethod
    def apply(
        array: AnyArray,
        to: AnyDataType,
        safe: Bool = True,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        var src = array.dtype()
        if src == to:
            return array.copy()  # identity → zero-copy
        elif src.is_null():
            return NullCast.apply(array, to)  # null → any
        elif safe:
            return Self._route[True](array, to, ctx)
        else:
            return Self._route[False](array, to, ctx)

    @staticmethod
    def _route[
        safe: Bool
    ](
        array: AnyArray, to: AnyDataType, ctx: ExecutionContext
    ) raises -> AnyArray:
        var src = array.dtype()
        # Binary-like family first, so bytes ↔ bytes and string ↔ numeric/bool
        # (incl. large_utf8 / fixed_size_binary) all route here.
        if (
            src.is_binary_like()
            or to.is_binary_like()
            or src.is_fixed_size_binary()
            or to.is_fixed_size_binary()
        ):
            return Self._binary[safe](array, to)
        elif src.is_decimal() or to.is_decimal():
            return DecimalCast.apply(array.to_data(), src, to)
        elif src.is_numeric() and to.is_numeric():
            return Self._numeric[safe](array, to, ctx)
        elif src.is_bool() or to.is_bool():
            return Self._bool(array, to, ctx)
        elif src.is_temporal() or to.is_temporal():
            return Self._temporal(array, to, ctx)
        raise Error(t"cast: unsupported cast {src} -> {to}")

    @staticmethod
    def _numeric[
        safe: Bool
    ](
        array: AnyArray, to: AnyDataType, ctx: ExecutionContext
    ) raises -> AnyArray:
        var src_dt = array.dtype()

        @parameter
        def on_source[From: NumericType](s: From) raises -> AnyArray:
            var typed = array.as_primitive[From]().copy()

            @parameter
            def on_target[To: NumericType](d: To) raises -> AnyArray:
                return NumericCast.apply[From, To, safe](typed, ctx).to_any()

            return variant_dispatch_raises[
                NumericType, predicate=_IsNumeric, func=on_target
            ](to._v)

        return variant_dispatch_raises[
            NumericType, predicate=_IsNumeric, func=on_source
        ](src_dt._v)

    @staticmethod
    def _bool(
        array: AnyArray, to: AnyDataType, ctx: ExecutionContext
    ) raises -> AnyArray:
        if array.dtype().is_bool():  # bool → numeric
            var b = array.as_bool().copy()

            @parameter
            def to_num[To: NumericType](d: To) raises -> AnyArray:
                return BoolToNum.apply[To](b, ctx).to_any()

            return variant_dispatch_raises[
                NumericType, predicate=_IsNumeric, func=to_num
            ](to._v)
        else:  # numeric → bool
            if not to.is_bool():
                raise Error(t"cast: expected bool target, got {to}")
            var src_dt = array.dtype()

            @parameter
            def from_num[From: NumericType](s: From) raises -> AnyArray:
                return NumToBool.apply(array.as_primitive[From](), ctx).to_any()

            return variant_dispatch_raises[
                NumericType, predicate=_IsNumeric, func=from_num
            ](src_dt._v)

    @staticmethod
    def _temporal(
        array: AnyArray, to: AnyDataType, ctx: ExecutionContext
    ) raises -> AnyArray:
        var src = array.dtype()
        var data = array.to_data()
        var same_width = src.byte_width() == to.byte_width()
        # temporal ↔ integer, or same-resolution temporal ↔ temporal: reinterpret.
        if src.is_integer() or to.is_integer():
            if not same_width:
                raise Error(
                    t"cast: cannot reinterpret {src} as {to} (width mismatch)"
                )
            return TemporalReinterpret.apply(data, to)
        var ns_from = TemporalScale.ns_per_tick(src)
        var ns_to = TemporalScale.ns_per_tick(to)
        if ns_from == ns_to and same_width:
            return TemporalReinterpret.apply(data, to)
        # otherwise scale the underlying integers by the unit ratio.
        var up = ns_from > ns_to
        var factor = (ns_from // ns_to) if up else (ns_to // ns_from)
        if src.byte_width() == 4:
            if to.byte_width() == 4:
                return TemporalScale.apply[DType.int32, DType.int32](
                    data, to, factor, up, ctx
                )
            return TemporalScale.apply[DType.int32, DType.int64](
                data, to, factor, up, ctx
            )
        else:
            if to.byte_width() == 4:
                return TemporalScale.apply[DType.int64, DType.int32](
                    data, to, factor, up, ctx
                )
            return TemporalScale.apply[DType.int64, DType.int64](
                data, to, factor, up, ctx
            )

    @staticmethod
    def _binary[
        safe: Bool
    ](array: AnyArray, to: AnyDataType) raises -> AnyArray:
        """Binary-like family: bytes ↔ bytes (relabel / offset re-width /
        fixed-size packing) and string ↔ numeric/bool (parse / format)."""
        var src = array.dtype()
        if src.is_binary_like() and to.is_binary_like():
            return Self._relabel_binary[safe](array, to)
        elif src.is_fixed_size_binary() and to.is_binary_like():
            var fsb = array.as_fixed_size_binary().copy()

            @parameter
            def to_bin[To: BinaryLikeType](d: To) raises -> AnyArray:
                return FixedSizeBinaryCast.to_binary[To](fsb).to_any()

            return variant_dispatch_raises[
                BinaryLikeType, predicate=_IsBinaryLike, func=to_bin
            ](to._v)
        elif src.is_binary_like() and to.is_fixed_size_binary():
            var width = to.as_fixed_size_binary().byte_width

            @parameter
            def from_bin[From: BinaryLikeType](s: From) raises -> AnyArray:
                var a = BinaryLikeArray[From](array.to_data())
                return FixedSizeBinaryCast.from_binary[From](a, width).to_any()

            return variant_dispatch_raises[
                BinaryLikeType, predicate=_IsBinaryLike, func=from_bin
            ](src._v)
        elif src.is_string() or src.is_large_string():
            return Self._parse_string[safe](array, to)
        elif to.is_string() or to.is_large_string():
            return Self._format_string(array, to)
        raise Error(t"cast: unsupported cast {src} -> {to}")

    @staticmethod
    def _relabel_binary[
        safe: Bool
    ](array: AnyArray, to: AnyDataType) raises -> AnyArray:
        @parameter
        def on_src[From: BinaryLikeType](s: From) raises -> AnyArray:
            var a = BinaryLikeArray[From](array.to_data())

            @parameter
            def on_to[To: BinaryLikeType](d: To) raises -> AnyArray:
                return BinaryLikeCast.apply[From, To, safe](a).to_any()

            return variant_dispatch_raises[
                BinaryLikeType, predicate=_IsBinaryLike, func=on_to
            ](to._v)

        return variant_dispatch_raises[
            BinaryLikeType, predicate=_IsBinaryLike, func=on_src
        ](array.dtype()._v)

    @staticmethod
    def _parse_string[
        safe: Bool
    ](array: AnyArray, to: AnyDataType) raises -> AnyArray:
        @parameter
        def on_str[From: StringLikeType](s: From) raises -> AnyArray:
            var a = BinaryLikeArray[From](array.to_data())
            if to.is_bool():
                return StringToBool.apply[From, safe](a).to_any()
            elif to.is_numeric():

                @parameter
                def to_num[To: NumericType](d: To) raises -> AnyArray:
                    return StringToNum.apply[From, To, safe](a).to_any()

                return variant_dispatch_raises[
                    NumericType, predicate=_IsNumeric, func=to_num
                ](to._v)
            raise Error(t"cast: {array.dtype()} → {to} is not supported")

        return variant_dispatch_raises[
            StringLikeType, predicate=_IsStringLike, func=on_str
        ](array.dtype()._v)

    @staticmethod
    def _format_string(array: AnyArray, to: AnyDataType) raises -> AnyArray:
        var src = array.dtype()

        @parameter
        def on_target[To: StringLikeType](d: To) raises -> AnyArray:
            if src.is_bool():
                return BoolToString.apply[To](array.as_bool().copy()).to_any()
            elif src.is_numeric():

                @parameter
                def from_num[From: NumericType](s: From) raises -> AnyArray:
                    return NumToString.apply[From, To](
                        array.as_primitive[From]()
                    ).to_any()

                return variant_dispatch_raises[
                    NumericType, predicate=_IsNumeric, func=from_num
                ](src._v)
            raise Error(t"cast: {src} → {to} is not supported")

        return variant_dispatch_raises[
            StringLikeType, predicate=_IsStringLike, func=on_target
        ](to._v)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def cast[
    From: NumericType, To: NumericType
](
    array: PrimitiveArray[From],
    safe: Bool = True,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[To]:
    """Typed numeric cast: ``cast[Int32Type, Float64Type](arr)`` — a runtime
    convenience selecting ``NumericCast.apply``'s comptime kernel. Call
    ``NumericCast.apply[From, To, safe]`` directly for a fully-monomorphized
    (e.g. AOT-fused) cast."""
    if safe:
        return NumericCast.apply[From, To, True](array, ctx)
    return NumericCast.apply[From, To, False](array, ctx)


def cast(
    array: AnyArray,
    to: AnyDataType,
    safe: Bool = True,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Cast ``array`` to dtype ``to`` (runtime family dispatch → ``Cast``)."""
    return Cast.apply(array, to, safe, ctx)
