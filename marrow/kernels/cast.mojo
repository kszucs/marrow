"""Cast kernels — one slim ``Kernel`` struct per conversion.

Each kernel is a separate struct implementing ``Kernel`` and doing **one**
conversion, so it can be optimized and monomorphized in isolation (and grabbed
directly by the AOT expression layer):

- ``NumericCast`` — numeric ↔ numeric SIMD ``pop.cast``; ``core`` /
  ``core_checked`` are the lane functors (the latter reused by the fused AOT
  node). ``safe`` is a **comptime** parameter selecting checked vs unchecked.
- ``NumToBool`` / ``BoolToNum`` — bit-pack (``x != 0``) / bit-unpack (``True→1``).
- ``TemporalCast`` — relabel to the underlying integer, or unit-scale it.
- ``StringToNum`` / ``NumToString`` / ``StringToBool`` / ``BoolToString`` —
  per-element ``atol``/``atof`` parse or format (variable-length, builder-based).
- ``NullCast`` — an all-null array of the target type.

Each kernel exposes a ``dispatch`` static method that resolves the runtime dtypes
of its family (the arithmetic-kernel pattern). The free ``cast`` function at the
bottom picks the target family and delegates to the matching kernel's
``dispatch``; ``safe`` is a plain runtime flag each kernel branches at its leaf
``apply`` call. The fused AOT node in ``marrow.expr.values`` bypasses all of this
and grabs ``NumericCast.core`` directly.
"""

from std.collections.string import atol, atof, StringSlice
from std.collections.string._utf8 import _is_valid_utf8
from std.sys import bit_width_of

from ..arrays import (
    DynArray,
    ArrayData,
    BinaryLikeArray,
    BoolArray,
    FixedSizeBinaryArray,
    PrimitiveArray,
)
from ..buffers import Buffer, Bitmap
from ..builders import (
    DynBuilder,
    BinaryLikeBuilder,
    BoolBuilder,
    FixedSizeBinaryBuilder,
    PrimitiveBuilder,
)
from ..views import apply, apply_checked
from ..dtypes import (
    DynType,
    BinaryLikeType,
    DType,
    NumericType,
    DecimalType,
    StringLikeType,
    TimeUnit,
    int32,
)
from .core import Kernel
from .execution import ExecutionContext
from .filter import take
from ..utils import GPU_ENABLED


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
        comptime if GPU_ENABLED:
            if ctx.is_gpu():
                buf = Buffer.alloc_device[Out](ctx.device.value(), length)
            else:
                buf = Buffer.alloc_uninit[Out](length)
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

    @staticmethod
    def dispatch(
        array: DynArray, to: DynType, safe: Bool, ctx: ExecutionContext
    ) raises -> DynArray:
        """Runtime numeric → numeric: resolve source and target over the numeric
        dtypes, branching ``safe`` into the checked / unchecked ``apply``."""

        @parameter
        def on_source[From: NumericType](s: From) raises -> DynArray:
            var typed = array.as_primitive[From]().copy()

            @parameter
            def on_target[To: NumericType](d: To) raises -> DynArray:
                if safe:
                    return Self.apply[From, To, True](typed, ctx).to_dyn()
                return Self.apply[From, To, False](typed, ctx).to_dyn()

            return to.dispatch_numeric[on_target]()

        return array.dtype().dispatch_numeric[on_source]()


# ---------------------------------------------------------------------------
# NumToBool / BoolToNum — bit-packed bool ↔ numeric
# ---------------------------------------------------------------------------


struct NumToBool(Kernel):
    """Numeric → bool: ``x != 0``, bit-packed. Lossless; validity preserved."""

    comptime name = "num_to_bool"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.ne(0)

    @staticmethod
    def dispatch(array: DynArray, ctx: ExecutionContext) raises -> DynArray:
        """Runtime numeric → bool over the numeric source dtypes."""

        @parameter
        def from_num[From: NumericType](s: From) raises -> DynArray:
            return Self.apply(array.as_primitive[From](), ctx).to_dyn()

        return array.dtype().dispatch_numeric[from_num]()

    @staticmethod
    def apply[
        From: NumericType
    ](array: PrimitiveArray[From], ctx: ExecutionContext) raises -> BoolArray:
        var length = len(array)
        var result: Bitmap[mut=True]
        comptime if GPU_ENABLED:
            result = Bitmap.alloc_device(
                ctx.device.value(), length
            ) if ctx.is_gpu() else Bitmap.alloc_uninit(length)
        else:
            result = Bitmap.alloc_uninit(length)
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
    """Bool → numeric: ``True→1, False→0``. Lossless; validity preserved."""

    comptime name = "bool_to_num"

    @always_inline
    @staticmethod
    def core[Out: DType, W: Int](m: SIMD[DType.bool, W]) -> SIMD[Out, W]:
        return m.cast[Out]()

    @staticmethod
    def dispatch(
        array: DynArray, to: DynType, ctx: ExecutionContext
    ) raises -> DynArray:
        """Runtime bool → numeric over the numeric target dtypes."""
        var b = array.as_bool().copy()

        @parameter
        def to_num[To: NumericType](d: To) raises -> DynArray:
            return Self.apply[To](b, ctx).to_dyn()

        return to.dispatch_numeric[to_num]()

    @staticmethod
    def apply[
        To: NumericType
    ](array: BoolArray, ctx: ExecutionContext) raises -> PrimitiveArray[To]:
        comptime Out = To.native
        var length = len(array)
        var buf: Buffer[mut=True]
        comptime if GPU_ENABLED:
            if ctx.is_gpu():
                buf = Buffer.alloc_device[Out](ctx.device.value(), length)
            else:
                buf = Buffer.alloc_uninit[Out](length)
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
# TemporalCast — temporal ↔ integer / temporal ↔ temporal
# ---------------------------------------------------------------------------


struct TemporalCast(Kernel):
    """Cast temporal ↔ integer / temporal ↔ temporal. Same physical width and
    resolution → a zero-copy relabel (``_reinterpret``); a differing unit → scale
    the underlying integers by the unit ratio (``_scale``)."""

    comptime name = "temporal_cast"

    @staticmethod
    def dispatch(
        array: DynArray, to: DynType, ctx: ExecutionContext
    ) raises -> DynArray:
        var src = array.dtype()
        var data = array.to_data()
        var same_width = src.byte_width() == to.byte_width()
        # temporal ↔ integer, or same-resolution temporal ↔ temporal: reinterpret.
        if src.is_integer() or to.is_integer():
            if not same_width:
                raise Error(
                    t"cast: cannot reinterpret {src} as {to} (width mismatch)"
                )
            return Self._reinterpret(data, to)
        var ns_from = Self.ns_per_tick(src)
        var ns_to = Self.ns_per_tick(to)
        if ns_from == ns_to and same_width:
            return Self._reinterpret(data, to)
        # otherwise scale the underlying integers by the unit ratio.
        var up = ns_from > ns_to
        var factor = (ns_from // ns_to) if up else (ns_to // ns_from)
        # Every temporal type Arrow defines is int32- or int64-backed, so this
        # is unreachable today. Checked rather than assumed because the branches
        # below treat "not 4" as "8": a wider temporal type would be scaled
        # through the wrong lane width with no error anywhere.
        var sw = src.byte_width()
        var tw = to.byte_width()
        if (sw != 4 and sw != 8) or (tw != 4 and tw != 8):
            raise Error(
                t"cast: cannot scale {src} ({sw}B) to {to} ({tw}B): ",
                "temporal storage must be 4 or 8 bytes",
            )
        if src.byte_width() == 4:
            if to.byte_width() == 4:
                return Self._scale[DType.int32, DType.int32](
                    data, to, factor, up, ctx
                )
            return Self._scale[DType.int32, DType.int64](
                data, to, factor, up, ctx
            )
        else:
            if to.byte_width() == 4:
                return Self._scale[DType.int64, DType.int32](
                    data, to, factor, up, ctx
                )
            return Self._scale[DType.int64, DType.int64](
                data, to, factor, up, ctx
            )

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
    def ns_per_tick(dt: DynType) raises -> Int64:
        """Nanoseconds represented by one tick of a temporal dtype — drives the
        reinterpret-vs-scale choice and the scale factor."""
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

    # TODO: remove this
    @staticmethod
    def _reinterpret(data: ArrayData, to: DynType) raises -> DynArray:
        """Relabel an integer/temporal buffer as ``to`` without moving data."""
        return DynArray.from_data(
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
        data: ArrayData,
        to: DynType,
        factor: Int64,
        up: Bool,
        ctx: ExecutionContext,
    ) raises -> DynArray:
        """Scale the underlying integers by ``factor`` (multiply if ``up``, else
        integer-divide), computing in int64 to avoid overflow, then narrow to
        ``DstN`` and relabel as ``to``."""
        var length = data.length
        var buf = Buffer.alloc_uninit[DstN](length)
        var src = data.buffers[0].view[SrcN](data.offset, length)

        @parameter
        @always_inline
        def scale[W: Int](v: SIMD[SrcN, W]) -> SIMD[DstN, W]:
            var x = v.cast[DType.int64]()
            return ((x * factor) if up else (x // factor)).cast[DstN]()

        apply[SrcN, DstN, scale](src, buf.view[DstN](0, length), ctx)
        return DynArray.from_data(
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
    def dispatch(
        array: DynArray, to: DynType, safe: Bool, ctx: ExecutionContext
    ) raises -> DynArray:
        """Runtime string-like → numeric: resolve source string kind and numeric
        target, branching ``safe`` into the raising / nulling ``apply``."""

        @parameter
        def on_str[From: StringLikeType](s: From) raises -> DynArray:
            var a = BinaryLikeArray[From](array.to_data())

            @parameter
            def to_num[To: NumericType](d: To) raises -> DynArray:
                if safe:
                    return Self.apply[From, To, True](a).to_dyn()
                return Self.apply[From, To, False](a).to_dyn()

            return to.dispatch_numeric[to_num]()

        return array.dtype().dispatch_stringlike[on_str]()

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
        var b = PrimitiveBuilder[To](len(array))
        for i in range(len(array)):
            if not array.is_valid(i):
                b.append_null()
                continue
            var s = array.unsafe_get(UInt(i))
            try:
                b.append(Self._parse[To.native](s))
            except:
                comptime if safe:
                    raise Error(t"cast: cannot parse '{s}' as {DynType(To())}")
                else:
                    b.append_null()
        return b.finish()


struct StringToBool(Kernel):
    """Parse ``"true"``/``"false"``/``"1"``/``"0"`` (case-insensitive) to bool.
    ``safe`` comptime: raise vs null on an unrecognized value."""

    comptime name = "string_to_bool"

    @staticmethod
    def dispatch(
        array: DynArray, safe: Bool, ctx: ExecutionContext
    ) raises -> DynArray:
        """Runtime string-like → bool over the source string kinds."""

        @parameter
        def on_str[From: StringLikeType](s: From) raises -> DynArray:
            var a = BinaryLikeArray[From](array.to_data())
            if safe:
                return Self.apply[From, True](a).to_dyn()
            return Self.apply[From, False](a).to_dyn()

        return array.dtype().dispatch_stringlike[on_str]()

    @staticmethod
    def apply[
        From: StringLikeType, safe: Bool
    ](array: BinaryLikeArray[From]) raises -> BoolArray:
        var b = BoolBuilder(len(array))
        for i in range(len(array)):
            if not array.is_valid(i):
                b.append_null()
                continue
            var s = String(array.unsafe_get(UInt(i))).lower()
            if s == "true" or s == "1":
                b.append(True)
            elif s == "false" or s == "0":
                b.append(False)
            else:
                comptime if safe:
                    raise Error(t"cast: cannot parse '{s}' as bool")
                else:
                    b.append_null()
        return b.finish()


struct NumToString(Kernel):
    """Format a numeric array to strings (per-element ``String(value)``)."""

    comptime name = "num_to_string"

    @staticmethod
    def dispatch(array: DynArray, to: DynType) raises -> DynArray:
        """Runtime numeric → string-like: resolve target string kind and numeric
        source."""

        @parameter
        def on_target[To: StringLikeType](d: To) raises -> DynArray:
            @parameter
            def from_num[From: NumericType](s: From) raises -> DynArray:
                return Self.apply[From, To](array.as_primitive[From]()).to_dyn()

            return array.dtype().dispatch_numeric[from_num]()

        return to.dispatch_stringlike[on_target]()

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
    def dispatch(array: DynArray, to: DynType) raises -> DynArray:
        """Runtime bool → string-like over the target string kinds."""
        var b = array.as_bool().copy()

        @parameter
        def on_target[To: StringLikeType](d: To) raises -> DynArray:
            return Self.apply[To](b).to_dyn()

        return to.dispatch_stringlike[on_target]()

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
    def dispatch(array: DynArray, to: DynType, safe: Bool) raises -> DynArray:
        """Runtime bytes ↔ bytes: resolve the source and target binary-like kinds,
        branching ``safe`` into the UTF-8-validating / trusting ``apply``."""

        @parameter
        def on_src[From: BinaryLikeType](s: From) raises -> DynArray:
            var a = BinaryLikeArray[From](array.to_data())

            @parameter
            def on_to[To: BinaryLikeType](d: To) raises -> DynArray:
                if safe:
                    return Self.apply[From, To, True](a).to_dyn()
                return Self.apply[From, To, False](a).to_dyn()

            return to.dispatch_binarylike[on_to]()

        return array.dtype().dispatch_binarylike[on_src]()

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
    def dispatch(array: DynArray, to: DynType) raises -> DynArray:
        """Runtime fixed_size_binary ↔ binary, in whichever direction applies.
        """
        if array.dtype().is_fixed_size_binary():  # fsb → binary
            var fsb = array.as_fixed_size_binary().copy()

            @parameter
            def to_bin[To: BinaryLikeType](d: To) raises -> DynArray:
                return Self.to_binary[To](fsb).to_dyn()

            return to.dispatch_binarylike[to_bin]()
        else:  # binary → fsb
            var width = to.as_fixed_size_binary().byte_width

            @parameter
            def from_bin[From: BinaryLikeType](s: From) raises -> DynArray:
                var a = BinaryLikeArray[From](array.to_data())
                return Self.from_binary[From](a, width).to_dyn()

            return array.dtype().dispatch_binarylike[from_bin]()

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
    def dispatch(array: DynArray, to: DynType) raises -> DynArray:
        var n = len(array)
        var b = DynBuilder(to.copy(), capacity=n)
        for _ in range(n):
            b.append_null()
        return b.finish()


# ---------------------------------------------------------------------------
# DecimalCast — decimal ↔ decimal (rescale) and decimal ↔ numeric
# ---------------------------------------------------------------------------


struct DecimalCast(Kernel):
    """Cast decimal ↔ decimal (rescale) and decimal ↔ numeric.

    Both sides resolve uniformly to a scalar native and a scale — a decimal to its
    backing integer (int32/64/128/256) and its scale, a plain numeric to its own
    native at scale 0. One ``_convert[FromN, ToN]`` then covers every case: an
    integer rescale by ``10^(to_scale − from_scale)`` (multiply up, truncating
    integer-divide down) when neither side is float, else a divide/multiply by
    ``10^scale`` in float64 — float16 ↔ int128/256 has no direct compiler-rt path
    (``__fixhfti`` / ``__floattihf``), so float64 is the intermediary. Arithmetic
    is unchecked (wrapping / truncating); validity is preserved."""

    comptime name = "decimal_cast"

    @staticmethod
    def dispatch(array: DynArray, to: DynType) raises -> DynArray:
        var data = array.to_data()
        var from_scale = Self._scale(array.dtype())
        var to_scale = Self._scale(to)

        @parameter
        def on_from[FromN: DType]() raises -> DynArray:
            @parameter
            def on_to[ToN: DType]() raises -> DynArray:
                return Self._convert[FromN, ToN](data, from_scale, to_scale, to)

            return Self._on_native[on_to](to)

        return Self._on_native[on_from](array.dtype())

    @staticmethod
    def _scale(dt: DynType) raises -> Int:
        """The decimal scale, or 0 for a plain integer/numeric.

        A `DynType` ladder rather than `dispatch_decimal` because it reads a
        *field*, and traits cannot require fields. It is guarded by
        `is_decimal()` so a decimal width the ladder does not know raises
        instead of silently reporting scale 0 — an unscaled decimal would be
        off by a factor of 10^scale, with no error anywhere.
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
            raise Self.error(t"no scale known for decimal type {dt}")

    @staticmethod
    def _on_native[
        func: def[N: DType]() raises capturing[_] -> DynArray
    ](dt: DynType) raises -> DynArray:
        """Resolve a decimal to its backing integer, or a numeric to its own
        native, then run ``func`` with that scalar ``DType``."""
        if dt.is_decimal():

            @parameter
            def by_dec[T: DecimalType](x: T) raises -> DynArray:
                return func[T.native]()

            return dt.dispatch_decimal[by_dec]()
        else:

            @parameter
            def by_num[T: NumericType](x: T) raises -> DynArray:
                return func[T.native]()

            return dt.dispatch_numeric[by_num]()

    @staticmethod
    def _convert[
        FromN: DType, ToN: DType
    ](
        data: ArrayData, from_scale: Int, to_scale: Int, to: DynType
    ) raises -> DynArray:
        """Per-element conversion for the resolved native pair."""
        comptime if FromN.is_floating_point():  # float → decimal
            var f = Self._pow10[DType.float64](to_scale)

            @parameter
            def to_dec(x: Scalar[FromN]) -> Scalar[ToN]:
                return round(x.cast[DType.float64]() * f).cast[ToN]()

            return Self._map[FromN, ToN, to_dec](data, to)
        elif ToN.is_floating_point():  # decimal → float
            var f = Self._pow10[DType.float64](from_scale)

            @parameter
            def to_flt(x: Scalar[FromN]) -> Scalar[ToN]:
                return (x.cast[DType.float64]() / f).cast[ToN]()

            return Self._map[FromN, ToN, to_flt](data, to)
        else:  # integer rescale by 10^(to_scale − from_scale)
            var delta = to_scale - from_scale
            if delta >= 0:
                var f = Self._pow10[ToN](delta)

                @parameter
                def up(x: Scalar[FromN]) -> Scalar[ToN]:
                    return x.cast[ToN]() * f

                return Self._map[FromN, ToN, up](data, to)
            else:
                var f = Self._pow10[FromN](-delta)

                @parameter
                def down(x: Scalar[FromN]) -> Scalar[ToN]:
                    return (x // f).cast[ToN]()

                return Self._map[FromN, ToN, down](data, to)

    @staticmethod
    def _pow10[T: DType](n: Int) -> Scalar[T]:
        var r = Scalar[T](1)
        for _ in range(n):
            r = r * 10
        return r

    @staticmethod
    def _map[
        FromN: DType,
        ToN: DType,
        op: def(Scalar[FromN]) capturing[_] -> Scalar[ToN],
    ](data: ArrayData, to: DynType) raises -> DynArray:
        """Apply ``op`` to each element, writing a fresh ``ToN`` buffer relabelled
        as ``to``. Scalar (int128/256 aren't reliably SIMD-vectorizable)."""
        var n = data.length
        var src = data.buffers[0].view[FromN](data.offset, n)
        var out = Buffer.alloc_uninit[ToN](n)
        var dst = out.view[ToN]()
        for i in range(n):
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


# ---------------------------------------------------------------------------
# ListCast / StructCast / DictionaryCast — nested + dictionary
# ---------------------------------------------------------------------------


struct ListCast(Kernel):
    """Cast a list-like array (list / large_list) to another of the same kind by
    recursively casting its child values to the target's value type; the offset
    buffer and validity are shared unchanged."""

    comptime name = "list_cast"

    @staticmethod
    def dispatch(
        array: DynArray, to: DynType, safe: Bool, ctx: ExecutionContext
    ) raises -> DynArray:
        var data = array.to_data()
        var child = DynArray.from_data(data.children[0].copy())
        var target = (
            to.as_large_list()
            .value_type()
            .copy() if to.is_large_list() else to.as_list()
            .value_type()
            .copy()
        )
        var new_child = cast(child, target, safe, ctx)
        return DynArray.from_data(
            ArrayData(
                dtype=to.copy(),
                length=data.length,
                nulls=data.nulls,
                offset=data.offset,
                bitmap=data.bitmap,
                buffers=data.buffers.copy(),
                children=[new_child.to_data()],
            )
        )


struct StructCast(Kernel):
    """Cast struct → struct by recursively casting each field to the target
    field's type (matched by position); the field counts must match."""

    comptime name = "struct_cast"

    @staticmethod
    def dispatch(
        array: DynArray, to: DynType, safe: Bool, ctx: ExecutionContext
    ) raises -> DynArray:
        var data = array.to_data()
        ref fields = to.as_struct().fields
        if len(fields) != len(data.children):
            raise Error(
                t"cast: struct field count mismatch {array.dtype()} -> {to}"
            )
        var children = List[ArrayData]()
        for i in range(len(data.children)):
            var field_arr = DynArray.from_data(data.children[i].copy())
            var casted = cast(field_arr, fields[i].dtype, safe, ctx)
            children.append(casted.to_data())
        return DynArray.from_data(
            ArrayData(
                dtype=to.copy(),
                length=data.length,
                nulls=data.nulls,
                offset=data.offset,
                bitmap=data.bitmap,
                buffers=data.buffers.copy(),
                children=children^,
            )
        )


struct DictionaryCast(Kernel):
    """Decode a dictionary array — gather its values by index (``take``) — then
    cast the decoded values to the target type when it differs."""

    comptime name = "dictionary_cast"

    @staticmethod
    def dispatch(
        array: DynArray, to: DynType, safe: Bool, ctx: ExecutionContext
    ) raises -> DynArray:
        ref d = array.as_dictionary()
        var indices = cast(d.indices(), int32, False, ctx).as_int32().copy()
        var decoded = take(d.dictionary().copy(), indices, ctx)
        if decoded.dtype() == to:
            return decoded^
        return cast(decoded, to, safe, ctx)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def cast(
    array: DynArray,
    to: DynType,
    safe: Bool = True,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> DynArray:
    """Cast ``array`` to dtype ``to``: pick the target family and delegate to the
    matching kernel's ``dispatch``. ``safe`` is a runtime flag each kernel resolves
    at its leaf ``apply`` call (raise vs. null/truncate)."""
    var src = array.dtype()
    if src == to:
        return array.copy()  # identity → zero-copy
    elif src.is_null():
        return NullCast.dispatch(array, to)  # null → any
    elif src.is_dictionary():
        return DictionaryCast.dispatch(array, to, safe, ctx)  # decode first
    elif src.is_binary_like() and to.is_binary_like():
        return BinaryLikeCast.dispatch(array, to, safe)  # bytes ↔ bytes
    elif (src.is_fixed_size_binary() and to.is_binary_like()) or (
        src.is_binary_like() and to.is_fixed_size_binary()
    ):
        return FixedSizeBinaryCast.dispatch(array, to)
    elif src.is_string() or src.is_large_string():  # string-like → numeric/bool
        if to.is_bool():
            return StringToBool.dispatch(array, safe, ctx)
        elif to.is_numeric():
            return StringToNum.dispatch(array, to, safe, ctx)
        raise Error(t"cast: unsupported cast {src} -> {to}")
    elif to.is_string() or to.is_large_string():  # numeric/bool → string-like
        if src.is_bool():
            return BoolToString.dispatch(array, to)
        elif src.is_numeric():
            return NumToString.dispatch(array, to)
        raise Error(t"cast: unsupported cast {src} -> {to}")
    elif src.is_decimal() or to.is_decimal():
        return DecimalCast.dispatch(array, to)
    elif src.is_numeric() and to.is_numeric():
        return NumericCast.dispatch(array, to, safe, ctx)
    elif to.is_bool():
        return NumToBool.dispatch(array, ctx)  # numeric → bool
    elif src.is_bool():
        return BoolToNum.dispatch(array, to, ctx)  # bool → numeric
    elif src.is_temporal() or to.is_temporal():
        return TemporalCast.dispatch(array, to, ctx)
    elif (src.is_list() and to.is_list()) or (
        src.is_large_list() and to.is_large_list()
    ):
        return ListCast.dispatch(array, to, safe, ctx)
    elif src.is_struct() and to.is_struct():
        return StructCast.dispatch(array, to, safe, ctx)
    raise Error(t"cast: unsupported cast {src} -> {to}")
