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

Every kernel conforms to ``CastKernel`` and so exposes the **same**
``dispatch(array, to, safe, ctx)``, resolving the runtime dtypes of its family
(the arithmetic-kernel pattern). The free ``cast`` function at the bottom picks
the target family and delegates to the matching kernel's ``dispatch``; ``safe``
is a plain runtime flag each kernel branches at its leaf ``apply`` call. The
fused AOT node in ``marrow/expr/comptime/casts.mojo`` bypasses all of this and
grabs ``NumericCast.core`` directly.

The uniform signature is the point of the trait, not code reuse: the ladder used
to call six different shapes and silently dropped whichever argument the target
arm did not accept. ``CastKernel``'s docstring has the detail.
"""

from std.collections.string import atol, atof, StringSlice
from std.collections.string._utf8 import _is_valid_utf8
from std.sys import bit_width_of, simd_width_of

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
from ..views import apply, apply_checked, BufferView, BitmapView
from ..dtypes import (
    DynType,
    BinaryLikeType,
    DType,
    NumericType,
    DecimalType,
    IntegerType,
    FloatingType,
    StringLikeType,
    int32,
)
from .core import Kernel
from .temporal import ticks_per_second
from ..execution import ExecContext, GPU_ENABLED
from .filter import take


# ---------------------------------------------------------------------------
# The family trait
# ---------------------------------------------------------------------------


trait CastKernel(Kernel):
    """A cast kernel: one erased entry point, identical across the family.

    Declared abstract rather than defaulted, because no two cast kernels share
    a resolution strategy — the point of the trait is not code reuse, it is
    that every arm takes the *same four arguments*, so `cast()`'s ladder cannot
    hand one arm fewer than it hands another.

    It could, and did. Six of the fifteen kernels had no `ctx` parameter and
    seven no `safe`, so the ladder silently dropped whichever the target arm
    did not accept: `cast(x, decimal128(38, 2), safe=True)` wrapped on overflow
    and a millisecond→second cast discarded its remainder, both under the
    default that promises to raise. Widening the signatures is what makes that
    class of defect unrepresentable; the two kernels that had to *implement*
    `safe` are the separate half of the fix.

    An arm with nothing to do with an argument still takes it and says why in
    its docstring — a total conversion cannot fail, so `safe` is inert there.
    That is a deliberate wart: a uniform signature that is occasionally
    redundant beats fifteen signatures nobody can check against the call site.

    Note this trait buys signature uniformity and **not** a shared dispatcher.
    `cast()` stays a hand-written ladder calling each struct by name: a closure
    generic over its own trait bound needs a narrowing adapter that inlines
    into every arm, measured at +662,740 bytes on one gate (§0).
    """

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        ...


# ---------------------------------------------------------------------------
# NumericCast — numeric ↔ numeric
# ---------------------------------------------------------------------------


struct NumericCast(CastKernel):
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
        ctx: ExecContext = ExecContext.serial(),
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
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Runtime numeric → numeric: resolve source and target over the numeric
        dtypes, branching ``safe`` into the checked / unchecked ``apply``."""

        def on_source[From: NumericType](s: From) raises {imm} -> DynArray:
            var typed = array.as_primitive[From]().copy()

            def on_target[To: NumericType](d: To) raises {imm} -> DynArray:
                if safe:
                    return Self.apply[From, To, True](typed, ctx).to_dyn()
                return Self.apply[From, To, False](typed, ctx).to_dyn()

            return to.dispatch_numeric(on_target)

        return array.dtype().dispatch_numeric(on_source)


# ---------------------------------------------------------------------------
# NumToBool / BoolToNum — bit-packed bool ↔ numeric
# ---------------------------------------------------------------------------


struct NumToBool(CastKernel):
    """Numeric → bool: ``x != 0``, bit-packed. Lossless; validity preserved."""

    comptime name = "num_to_bool"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.ne(0)

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Runtime numeric → bool over the numeric source dtypes.

        `to` is checked rather than assumed: `cast()` only routes here when it
        is bool, but that was a positional invariant no reader of this kernel
        could verify. `safe` is inert — `x != 0` is total.
        """
        if not to.is_bool():
            raise Self.error(t"target must be bool, got {to}")

        def from_num[From: NumericType](s: From) raises {imm} -> DynArray:
            return Self.apply(array.as_primitive[From](), ctx).to_dyn()

        return array.dtype().dispatch_numeric(from_num)

    @staticmethod
    def apply[
        From: NumericType
    ](array: PrimitiveArray[From], ctx: ExecContext) raises -> BoolArray:
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


struct BoolToNum(CastKernel):
    """Bool → numeric: ``True→1, False→0``. Lossless; validity preserved."""

    comptime name = "bool_to_num"

    @always_inline
    @staticmethod
    def core[Out: DType, W: Int](m: SIMD[DType.bool, W]) -> SIMD[Out, W]:
        return m.cast[Out]()

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Runtime bool → numeric over the numeric target dtypes.

        `safe` is inert: `True→1, False→0` is representable in every numeric
        type, so this conversion cannot fail.
        """
        var b = array.as_bool().copy()

        def to_num[To: NumericType](d: To) raises {imm} -> DynArray:
            return Self.apply[To](b, ctx).to_dyn()

        return to.dispatch_numeric(to_num)

    @staticmethod
    def apply[
        To: NumericType
    ](array: BoolArray, ctx: ExecContext) raises -> PrimitiveArray[To]:
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


struct TemporalCast(CastKernel):
    """Cast temporal ↔ integer / temporal ↔ temporal. Same physical width and
    resolution → a zero-copy relabel (``_reinterpret``); a differing unit → scale
    the underlying integers by the unit ratio (``_scale``)."""

    comptime name = "temporal_cast"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
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
                    data, to, factor, up, safe, ctx
                )
            return Self._scale[DType.int32, DType.int64](
                data, to, factor, up, safe, ctx
            )
        else:
            if to.byte_width() == 4:
                return Self._scale[DType.int64, DType.int32](
                    data, to, factor, up, safe, ctx
                )
            return Self._scale[DType.int64, DType.int64](
                data, to, factor, up, safe, ctx
            )

    @staticmethod
    def ns_per_tick(dt: DynType) raises -> Int64:
        """Nanoseconds represented by one tick of a temporal dtype — drives the
        reinterpret-vs-scale choice and the scale factor.

        The reciprocal of `temporal.ticks_per_second`, which already walks the
        same five dtypes and the same four `TimeUnit`s. Every resolution Arrow
        defines divides 1e9 exactly, so the division is not lossy. `date32` is
        the one dtype that helper rejects -- a day is not a sub-second tick --
        so it is answered here.

        Folding `_scale_checked` into `_map_scalar` was tried alongside this and
        **reverted**: it removed 49 source lines and cost **+10,560 bytes of
        `__text`** on `query_sort`, `query_dynvalue` and `query_runtime`, because
        `_map_scalar` carries the allocation and the `ArrayData` relabel inside
        every `[SrcN, DstN]` instantiation while `_scale` has them once. Same
        shape as the `DecimalCast._convert` merge recorded below. This change,
        measured alone, is +64 bytes on one gate and +0 on the other eleven."""
        if dt.is_date32():
            return 86_400_000_000_000  # days
        if not dt.is_temporal():
            raise Error(t"cast: {dt} is not a temporal type")
        return 1_000_000_000 // Int64(ticks_per_second(dt))

    @staticmethod
    def _scale_checked[
        SrcN: DType, DstN: DType
    ](
        data: ArrayData,
        src: BufferView[SrcN, _],
        dst: BufferView[mut=True, DstN, _],
        factor: Int64,
        up: Bool,
    ) raises:
        """Serial checked scale: raise on the first tick that cannot round-trip.

        Serial by construction, not by preference — this raises, and an
        exception cannot cross a `ctx.stripe` worker or a GPU kernel boundary
        (§0). `NumericCast.apply`'s checked arm is serial for the same reason.

        Null lanes are skipped: an invalid slot may hold arbitrary junk that
        need not be representable, and failing on it would reject a perfectly
        valid array."""
        var n = data.length
        var validity = Optional[BitmapView[origin_of(data.bitmap._value)]](None)
        if data.bitmap and data.nulls > 0:
            validity = data.bitmap.value().view(data.offset, n)
        for i in range(n):
            if validity and not validity.value()[i]:
                dst.store[1](i, Scalar[DstN](0))
                continue
            var x = src.load[1](i).cast[DType.int64]()
            var scaled: Int64
            if up:
                if x > Int64.MAX // factor or x < Int64.MIN // factor:
                    raise Self.error(
                        t"scaling {data.dtype} would overflow: tick {x} times"
                        t" {factor} is out of range for the target unit"
                    )
                scaled = x * factor
            else:
                if x % factor != 0:
                    raise Self.error(
                        t"scaling {data.dtype} would lose data: tick {x} is"
                        t" not a whole multiple of {factor}"
                    )
                scaled = x // factor
            var out = scaled.cast[DstN]()
            if out.cast[DType.int64]() != scaled:
                raise Self.error(
                    t"scaling {data.dtype} would overflow: {scaled} does not"
                    t" fit the target's storage width"
                )
            dst.store[1](i, out)

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
        safe: Bool,
        ctx: ExecContext,
    ) raises -> DynArray:
        """Scale the underlying integers by ``factor`` (multiply if ``up``, else
        integer-divide), computing in int64 to avoid overflow, then narrow to
        ``DstN`` and relabel as ``to``.

        Under ``safe`` a scale that cannot round-trip raises rather than
        silently losing data: multiplying up can exceed int64 or the target
        width, and dividing down discards any sub-tick remainder. Arrow C++'s
        ``ShiftTime`` draws the same two lines, under ``allow_time_overflow``
        and ``allow_time_truncate``."""
        var length = data.length
        var buf = Buffer.alloc_uninit[DstN](length)
        var src = data.buffers[0].view[SrcN](data.offset, length)

        if safe:
            Self._scale_checked[SrcN, DstN](
                data, src, buf.view[DstN](0, length), factor, up
            )
        else:

            @always_inline
            def scale[W: Int](v: SIMD[SrcN, W]) {imm} -> SIMD[DstN, W]:
                var x = v.cast[DType.int64]()
                return ((x * factor) if up else (x // factor)).cast[DstN]()

            apply[SrcN, DstN](src, buf.view[DstN](0, length), scale, ctx)
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


struct StringToNum(CastKernel):
    """Parse strings to a numeric type. ``safe`` is comptime: safe=True raises on
    an unparseable value, safe=False nulls it — the dead branch is elided."""

    comptime name = "string_to_num"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Runtime string-like → numeric: resolve source string kind and numeric
        target, branching ``safe`` into the raising / nulling ``apply``."""

        def on_str[From: StringLikeType](s: From) raises {imm} -> DynArray:
            var a = BinaryLikeArray[From](array.to_data())

            def to_num[To: NumericType](d: To) raises {imm} -> DynArray:
                if safe:
                    return Self.apply[From, To, True](a).to_dyn()
                return Self.apply[From, To, False](a).to_dyn()

            return to.dispatch_numeric(to_num)

        return array.dtype().dispatch_stringlike(on_str)

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


struct StringToBool(CastKernel):
    """Parse ``"true"``/``"false"``/``"1"``/``"0"`` (case-insensitive) to bool.
    ``safe`` comptime: raise vs null on an unrecognized value."""

    comptime name = "string_to_bool"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Runtime string-like → bool over the source string kinds.

        `to` is checked rather than assumed — see `NumToBool.dispatch`.
        """
        if not to.is_bool():
            raise Self.error(t"target must be bool, got {to}")

        def on_str[From: StringLikeType](s: From) raises {imm} -> DynArray:
            var a = BinaryLikeArray[From](array.to_data())
            if safe:
                return Self.apply[From, True](a).to_dyn()
            return Self.apply[From, False](a).to_dyn()

        return array.dtype().dispatch_stringlike(on_str)

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


struct NumToString(CastKernel):
    """Format a numeric array to strings (per-element ``String(value)``)."""

    comptime name = "num_to_string"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Runtime numeric → string-like: resolve target string kind and numeric
        source.

        `safe` is inert (formatting a number is total) and `ctx` unused: this
        builds through a `BinaryLikeBuilder`, one variable-length element at a
        time, so there is no lane to stripe or upload.
        """

        def on_target[To: StringLikeType](d: To) raises {imm} -> DynArray:
            def from_num[From: NumericType](s: From) raises {imm} -> DynArray:
                return Self.apply[From, To](array.as_primitive[From]()).to_dyn()

            return array.dtype().dispatch_numeric(from_num)

        return to.dispatch_stringlike(on_target)

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


struct BoolToString(CastKernel):
    """Format a bool array to ``"true"``/``"false"`` strings."""

    comptime name = "bool_to_string"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Runtime bool → string-like over the target string kinds.

        `safe` and `ctx` are inert — see `NumToString.dispatch`.
        """
        var b = array.as_bool().copy()

        def on_target[To: StringLikeType](d: To) raises {imm} -> DynArray:
            return Self.apply[To](b).to_dyn()

        return to.dispatch_stringlike(on_target)

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


def _all_ascii(window: BufferView[DType.uint8, _]) -> Bool:
    """True when every byte in `window` is < 0x80.

    ASCII is a subset of UTF-8 that is closed under slicing, so an all-ASCII
    buffer makes every element of *any* offset layout over it valid UTF-8 —
    which is what lets `BinaryLikeCast._check_utf8` skip both its offset scan
    and its per-element loop.

    Four accumulators, reduced once per 4 KiB chunk. The accumulators are for
    throughput — the loop is a pure load-and-OR chain, latency-bound on one
    accumulator and bandwidth-bound on enough of them to cover the load-to-use
    distance. The chunking is for the *failure* case: a caller that gets False
    goes on to do more work over the same bytes, so bailing within 4 KiB of the
    first non-ASCII byte is what keeps this probe from costing a wasted pass
    over the whole buffer.
    """
    comptime W = simd_width_of[DType.uint8]()
    comptime CHUNK = 4096  # a whole multiple of 4 * W on every target here
    var n = len(window)
    var base = 0

    while base < n:
        var stop = min(base + CHUNK, n)
        var acc0 = SIMD[DType.uint8, W](0)
        var acc1 = SIMD[DType.uint8, W](0)
        var acc2 = SIMD[DType.uint8, W](0)
        var acc3 = SIMD[DType.uint8, W](0)

        var i = base
        while i + 4 * W <= stop:
            acc0 |= window.load[W](i)
            acc1 |= window.load[W](i + W)
            acc2 |= window.load[W](i + 2 * W)
            acc3 |= window.load[W](i + 3 * W)
            i += 4 * W
        while i + W <= stop:
            acc0 |= window.load[W](i)
            i += W

        var tail = UInt8(0)
        while i < stop:
            tail |= window.unsafe_get(i)
            i += 1

        if ((((acc0 | acc1) | (acc2 | acc3)).reduce_or() | tail) & 0x80) != 0:
            return False
        base = stop

    return True


def _validate_utf8_window(window: BufferView[DType.uint8, _]) -> Bool:
    """Validate `window` as UTF-8, skipping runs of pure-ASCII blocks.

    The reason this exists rather than one `_is_valid_utf8` call over the whole
    window: on the columns that motivated this code the bytes are *mostly*
    ASCII but not entirely. ClickBench's `URL` is 4.5% non-ASCII by byte, and
    those bytes are clustered — 92.7% of 16-byte blocks are pure ASCII. A
    single call pays the slow validator's per-byte throughput on all of it; this
    pays it only on the blocks that actually contain a multi-byte sequence.

    The block skipping is safe because of where the region boundaries land. A
    pure-ASCII block cannot contain any part of a multi-byte sequence, so no
    sequence can straddle one: the byte after a skipped block is a character
    start, and so is the byte at the window start. Every region handed to
    `_is_valid_utf8` therefore begins and ends on a character boundary and can
    be validated independently of its neighbours.
    """
    comptime W = simd_width_of[DType.uint8]()
    var n = len(window)
    var i = 0

    while i < n:
        if i + W <= n and (window.load[W](i) & 0x80).reduce_or() == 0:
            i += W
        else:
            var start = i
            while i < n:
                if i + W <= n and (window.load[W](i) & 0x80).reduce_or() == 0:
                    break
                if i + W <= n:
                    i += W
                else:
                    i = n
            if not _is_valid_utf8(window[start:i].as_span()):
                return False

    return True


# ---------------------------------------------------------------------------
# BinaryLikeCast — binary/large_binary/utf8/large_utf8 ↔ each other
# ---------------------------------------------------------------------------


struct BinaryLikeCast(CastKernel):
    """Cast between the binary-like containers (binary, large_binary, utf8,
    large_utf8). Equal physical offset width → a zero-copy relabel that shares
    the offset and value buffers; differing width (32↔64-bit offsets) → a rebuild
    through a builder. Producing a UTF-8 string from raw bytes validates every
    element under ``safe`` (raising on malformed UTF-8)."""

    comptime name = "binary_like_cast"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Runtime bytes ↔ bytes: resolve the source and target binary-like kinds,
        branching ``safe`` into the UTF-8-validating / trusting ``apply``.

        `ctx` is unused: the equal-offset-width case is a pure relabel and the
        differing-width case rebuilds through a builder.
        """

        def on_src[From: BinaryLikeType](s: From) raises {imm} -> DynArray:
            var a = BinaryLikeArray[From](array.to_data())

            def on_to[To: BinaryLikeType](d: To) raises {imm} -> DynArray:
                if safe:
                    return Self.apply[From, To, True](a).to_dyn()
                return Self.apply[From, To, False](a).to_dyn()

            return to.dispatch_binarylike(on_to)

        return array.dtype().dispatch_binarylike(on_src)

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
        """Reject a binary array that does not hold valid UTF-8 in every
        non-null element.

        Two whole-buffer fast paths sit in front of the exact per-element loop.
        Both are *sufficient* conditions only: whenever either one fails the
        loop still runs, so the accept/reject decision is unchanged and only
        the cost of reaching it moves.

        1. **All-ASCII.** Every byte < 0x80 makes every element valid UTF-8 no
           matter where the offsets cut the buffer, and no matter what a null
           slot holds. One SIMD pass, memory-bandwidth bound.
        2. **Valid window + element starts on character boundaries.** If the
           whole byte window validates *and* no element begins on a
           continuation byte (0b10xxxxxx), every element is a whole number of
           characters and therefore validates on its own. The boundary scan is
           what keeps this from being weaker than the loop: without it a
           multi-byte character split across two elements would validate as a
           concatenation while each half is individually malformed.

        The fall-through matters as much as the fast paths. A null slot is
        allowed to hold arbitrary bytes, so a whole-window check can fail on an
        array the loop accepts; falling back rather than raising is what keeps
        that from becoming a false rejection.

        Arrow C++ validates per element too (`Utf8Validator::VisitValue` in
        `scalar_cast_string.cc`) — its `ValidateUTF8Inline` is cheap enough per
        call that it never needed a buffer-wide pre-check.
        """
        if len(array) == 0:
            return

        var start = Int(array.offsets.unsafe_get[From.offset](array.offset))
        var end = Int(
            array.offsets.unsafe_get[From.offset](array.offset + array.length)
        )
        var window = array.values.view[DType.uint8](start, end - start)

        if _all_ascii(window):
            return
        if _validate_utf8_window(window) and Self._starts_on_boundaries(
            array, start, end
        ):
            return

        for i in range(len(array)):
            if array.is_valid(i) and not _is_valid_utf8(
                array.unsafe_get(UInt(i)).as_bytes()
            ):
                raise Error("cast: invalid UTF-8 in binary → string cast")

    @staticmethod
    def _starts_on_boundaries[
        From: BinaryLikeType
    ](array: BinaryLikeArray[From], start: Int, end: Int) -> Bool:
        """True when no element in the window begins on a UTF-8 continuation
        byte, i.e. every offset cuts the buffer at a character boundary.

        Element 0 starts at `start`, which is where validation began, so it is
        a boundary by construction and is skipped. An offset sitting at `end`
        belongs to a trailing empty element and has no byte to inspect."""
        var window = array.values.view[DType.uint8](start, end - start)
        for k in range(1, array.length):
            var off = Int(
                array.offsets.unsafe_get[From.offset](array.offset + k)
            )
            if off < end and (window.unsafe_get(off - start) & 0xC0) == 0x80:
                return False
        return True


# ---------------------------------------------------------------------------
# FixedSizeBinaryCast — fixed_size_binary ↔ variable-length binary
# ---------------------------------------------------------------------------


struct FixedSizeBinaryCast(CastKernel):
    """Cast fixed-size-binary ↔ variable-length binary. ``to_binary`` derives the
    offset buffer from the fixed width and shares the data bytes; ``from_binary``
    packs each element into a fixed cell, raising when a length ≠ the width."""

    comptime name = "fixed_size_binary_cast"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Runtime fixed_size_binary ↔ binary, in whichever direction applies.

        `safe` is inert *by reference*, not by oversight: Arrow C++ has no flag
        for a binary → fixed-size-binary width mismatch, so `from_binary`
        raises on a wrong width unconditionally. `ctx` is unused (builder).
        """
        if array.dtype().is_fixed_size_binary():  # fsb → binary
            var fsb = array.as_fixed_size_binary().copy()

            def to_bin[To: BinaryLikeType](d: To) raises {imm} -> DynArray:
                return Self.to_binary[To](fsb).to_dyn()

            return to.dispatch_binarylike(to_bin)
        else:  # binary → fsb
            var width = to.as_fixed_size_binary().byte_width

            def from_bin[
                From: BinaryLikeType
            ](s: From) raises {imm} -> DynArray:
                var a = BinaryLikeArray[From](array.to_data())
                return Self.from_binary[From](a, width).to_dyn()

            return array.dtype().dispatch_binarylike(from_bin)

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


struct NullCast(CastKernel):
    """Cast a null array to any target type: an all-null array of that type."""

    comptime name = "null_cast"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Runtime null → any.

        `safe` and `ctx` are inert: the output is all-null whatever the target,
        so nothing can fail and there is nothing to parallelize.
        """
        var n = len(array)
        var b = DynBuilder(to.copy(), capacity=n)
        for _ in range(n):
            b.append_null()
        return b.finish()


# ---------------------------------------------------------------------------
# The decimal family — one struct per conversion, not one struct per dtype pair
# ---------------------------------------------------------------------------
#
# These five were a single `DecimalCast` doing decimal↔decimal, decimal↔float
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
# integer source is always scale 0, so `IntToDecimal` only ever scales *up*,
# and an integer target is always scale 0, so `DecimalToInt` only ever scales
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
    *valid* element converts cleanly. ``NumericCast.apply`` masks its checked
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
    """Multiply by 10^delta, widening the scale. Shared by `DecimalRescale` and
    `IntToDecimal`, which is the only real overlap left between the five."""
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
    `DecimalRescale` and `DecimalToInt`."""
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


struct DecimalRescale(CastKernel):
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


struct IntToDecimal(CastKernel):
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


struct DecimalToInt(CastKernel):
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


struct FloatToDecimal(CastKernel):
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


struct DecimalToFloat(CastKernel):
    """Decimal → float: divide by 10^scale in float64.

    Unchecked on purpose. Decimal → float is inexact by construction and Arrow
    permits it under `safe`, exactly as float → float is permitted
    (`NumericCast.needs_check` answers False there too)."""

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


struct DecimalCast(CastKernel):
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
            return DecimalRescale.dispatch(array, to, safe, ctx)
        elif src.is_decimal() and to.is_floating_point():
            return DecimalToFloat.dispatch(array, to, safe, ctx)
        elif src.is_decimal() and to.is_integer():
            return DecimalToInt.dispatch(array, to, safe, ctx)
        elif src.is_floating_point() and to.is_decimal():
            return FloatToDecimal.dispatch(array, to, safe, ctx)
        elif src.is_integer() and to.is_decimal():
            return IntToDecimal.dispatch(array, to, safe, ctx)
        else:
            raise Self.error(t"unsupported decimal cast {src} -> {to}")


# ---------------------------------------------------------------------------
# ListCast / StructCast / DictionaryCast — nested + dictionary
# ---------------------------------------------------------------------------


struct ListCast(CastKernel):
    """Cast a list-like array (list / large_list / map) to another of the same
    kind by recursively casting its child values to the target's value type; the
    offset buffer and validity are shared unchanged.

    `map` rides this path rather than needing its own kernel: physically it is a
    list whose single child is the non-nullable `entries` struct, so casting a
    `map<k1, v1>` to `map<k2, v2>` is casting that struct — which `StructCast`
    then does field by field. Only the *target child type* differs, which is why
    the three cases meet here and nowhere else."""

    comptime name = "list_cast"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        var data = array.to_data()
        var child = DynArray.from_data(data.children[0].copy())
        var target: DynType
        if to.is_map():
            target = to.as_map().entries_field().dtype.copy()
        elif to.is_large_list():
            target = to.as_large_list().value_type().copy()
        else:
            target = to.as_list().value_type().copy()
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


struct StructCast(CastKernel):
    """Cast struct → struct by recursively casting each field to the target
    field's type (matched by position); the field counts must match."""

    comptime name = "struct_cast"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
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


struct DictionaryCast(CastKernel):
    """Decode a dictionary array — gather its values by index (``take``) — then
    cast the decoded values to the target type when it differs."""

    comptime name = "dictionary_cast"

    @staticmethod
    def dispatch(
        array: DynArray,
        to: DynType,
        safe: Bool = True,
        ctx: ExecContext = ExecContext.serial(),
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
    ctx: ExecContext = ExecContext.serial(),
) raises -> DynArray:
    """Cast ``array`` to dtype ``to``: pick the target family and delegate to the
    matching kernel's ``dispatch``. ``safe`` is a runtime flag each kernel resolves
    at its leaf ``apply`` call (raise vs. null/truncate)."""
    var src = array.dtype()
    if src == to:
        return array.copy()  # identity → zero-copy
    elif src.is_null():
        return NullCast.dispatch(array, to, safe, ctx)  # null → any
    elif src.is_dictionary():
        return DictionaryCast.dispatch(array, to, safe, ctx)  # decode first
    elif src.is_binary_like() and to.is_binary_like():
        return BinaryLikeCast.dispatch(array, to, safe, ctx)  # bytes ↔ bytes
    elif (src.is_fixed_size_binary() and to.is_binary_like()) or (
        src.is_binary_like() and to.is_fixed_size_binary()
    ):
        return FixedSizeBinaryCast.dispatch(array, to, safe, ctx)
    elif src.is_string() or src.is_large_string():  # string-like → numeric/bool
        if to.is_bool():
            return StringToBool.dispatch(array, to, safe, ctx)
        elif to.is_numeric():
            return StringToNum.dispatch(array, to, safe, ctx)
        raise Error(t"cast: unsupported cast {src} -> {to}")
    elif to.is_string() or to.is_large_string():  # numeric/bool → string-like
        if src.is_bool():
            return BoolToString.dispatch(array, to, safe, ctx)
        elif src.is_numeric():
            return NumToString.dispatch(array, to, safe, ctx)
        raise Error(t"cast: unsupported cast {src} -> {to}")
    elif src.is_decimal() or to.is_decimal():
        return DecimalCast.dispatch(array, to, safe, ctx)
    elif src.is_numeric() and to.is_numeric():
        return NumericCast.dispatch(array, to, safe, ctx)
    elif to.is_bool():
        return NumToBool.dispatch(array, to, safe, ctx)  # numeric → bool
    elif src.is_bool():
        return BoolToNum.dispatch(array, to, safe, ctx)  # bool → numeric
    elif src.is_temporal() or to.is_temporal():
        return TemporalCast.dispatch(array, to, safe, ctx)
    elif (
        (src.is_list() and to.is_list())
        or (src.is_large_list() and to.is_large_list())
        or (src.is_map() and to.is_map())
    ):
        return ListCast.dispatch(array, to, safe, ctx)
    elif src.is_struct() and to.is_struct():
        return StructCast.dispatch(array, to, safe, ctx)
    raise Error(t"cast: unsupported cast {src} -> {to}")
