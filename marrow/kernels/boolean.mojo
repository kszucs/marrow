"""Boolean and bitwise kernels.

Three tiers per kernel (same scheme as ``arithmetic.mojo``):

- **Tier 0 (core)** — ``Kernel.core[W]``: raw SIMD predicate on
  ``SIMD[DType.bool, W]``. Used by ``faszom.mojo`` expression nodes for
  compile-time kernel fusion.
- **Tier 1 (apply)** — ``Kernel.apply``: typed ``BoolArray`` API.  Operates
  directly on bit-packed bitmaps via 64-bit word operations — more efficient
  than element-wise SIMD for packed bits.
- **Tier 2 (dispatch)** — ``Kernel.dispatch``: type-erased ``AnyArray`` entry.
  Default implementation in the trait; concrete structs only define ``core``
  and ``apply``.
"""


from std.math import isnan, isinf

from ..arrays import BoolArray, PrimitiveArray, AnyArray
from ..buffers import Bitmap
from ..builders import PrimitiveBuilder
from ..dtypes import (
    NumericType,
    PrimitiveType,
    FloatingType,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    Float16Type,
    Float32Type,
    Float64Type,
    bool_ as bool_dt,
)
from ..utils import dispatch_over_numeric, dispatch_over_floating
from ..views import BitmapView, apply
from .helpers import Kernel
from .execution import ExecutionContext


# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------


trait BoolBinaryKernel(Kernel):
    """Element-wise binary boolean kernel (BoolArray × BoolArray → BoolArray).

    Concrete structs define ``core`` and ``apply``; ``dispatch`` has a default
    implementation that type-checks inputs and delegates to ``apply``.
    """

    @staticmethod
    def core[
        W: Int
    ](a: SIMD[DType.bool, W], b: SIMD[DType.bool, W]) -> SIMD[DType.bool, W]:
        ...

    @staticmethod
    def apply(
        left: BoolArray,
        right: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        ...

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        if left.dtype() != bool_dt or right.dtype() != bool_dt:
            raise Error(t"{Self.name}: inputs must be bool arrays")
        return Self.apply(
            left.as_bool().copy(), right.as_bool().copy(), ctx
        ).to_any()


trait BoolUnaryKernel(Kernel):
    """Element-wise unary boolean kernel (BoolArray → BoolArray).

    Concrete structs define ``core`` and ``apply``; ``dispatch`` has a default
    implementation that type-checks the input and delegates to ``apply``.
    """

    @staticmethod
    def core[W: Int](a: SIMD[DType.bool, W]) -> SIMD[DType.bool, W]:
        ...

    @staticmethod
    def apply(
        arr: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        ...

    @staticmethod
    def dispatch(
        arr: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        if arr.dtype() != bool_dt:
            raise Error(t"{Self.name}: input must be a bool array")
        return Self.apply(arr.as_bool().copy(), ctx).to_any()


trait UnaryPredicateKernel(Kernel):
    """Element-wise `array -> bool` predicate over a runtime-typed `AnyArray`.

    Unlike `BoolUnaryKernel` (bool -> bool) these take *any* array and produce a
    `BoolArray`. Validity predicates (`is_null`/`not_null`) read only the bitmap;
    value predicates (`is_nan`/`is_inf`) scan the values. Concrete kernels define
    `apply` (which owns type resolution — over any dtype for the validity ones,
    over floating dtypes for the value ones); `dispatch` defaults to
    `apply(...).to_any()`, mirroring the other kernel traits.
    """

    @staticmethod
    def apply(
        arr: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        ...

    @staticmethod
    def dispatch(
        arr: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return Self.apply(arr, ctx).to_any()


# ---------------------------------------------------------------------------
# Kleene 3-valued logic helpers
# ---------------------------------------------------------------------------


def _validity_ones(arr: BoolArray) raises -> Bitmap[mut=False]:
    """The array's offset-adjusted validity as an owned, offset-0 bitmap, or an
    all-ones bitmap when the array carries no validity buffer (every value
    valid). Lets the Kleene formulas treat both operands uniformly."""
    var v = arr.validity()
    if v:
        # OR the view with itself to materialize a fresh offset-0 copy.
        return v.value().union(v.value()).to_immutable()
    else:
        return (~Bitmap.alloc_zeroed(len(arr)).view()).to_immutable()


def _kleene[
    is_and: Bool
](left: BoolArray, right: BoolArray, name: String) raises -> BoolArray:
    """SQL Kleene 3-valued AND (`is_and=True`) / OR (`is_and=False`).

    Data is the plain bitwise `a & b` / `a | b`. Validity matches Arrow C++'s
    `KleeneAnd` / `KleeneOr`:

    - AND valid iff `(a_valid & ~a) | (b_valid & ~b) | (a_valid & b_valid)`
      — a known-false operand forces a valid (false) result.
    - OR  valid iff `(a_valid &  a) | (b_valid &  b) | (a_valid & b_valid)`
      — a known-true operand forces a valid (true) result.
    """
    var n = len(left)
    if len(right) != n:
        raise Error(t"{name}: input arrays must have equal length")
    var a_data = left.values()
    var b_data = right.values()
    var data: Bitmap[mut=False]
    comptime if is_and:
        data = (a_data & b_data).to_immutable()
    else:
        data = (a_data | b_data).to_immutable()

    # Both operands fully valid → result fully valid (keeps the None convention).
    if not left.bitmap and not right.bitmap:
        return BoolArray(length=n, nulls=0, offset=0, bitmap=None, buffer=data^)

    var a_valid = _validity_ones(left)
    var b_valid = _validity_ones(right)
    var av = a_valid.view()
    var bv = b_valid.view()

    # term_a / term_b: positions each operand alone can decide (valid & deciding).
    var term_a: Bitmap[mut=True]
    var term_b: Bitmap[mut=True]
    comptime if is_and:
        # term = valid AND NOT data, in one fused pass (no ~data temporaries).
        term_a = av.difference(a_data)
        term_b = bv.difference(b_data)
    else:
        term_a = av & a_data
        term_b = bv & b_data
    var both_valid = av & bv
    var partial = term_a.view() | term_b.view()
    var valid = partial.view() | both_valid.view()

    var valid_bm = valid.to_immutable()
    var nulls = n - valid_bm.view().count_set_bits()
    return BoolArray(
        length=n, nulls=nulls, offset=0, bitmap=valid_bm^, buffer=data^
    )


# ---------------------------------------------------------------------------
# Kernel structs
# ---------------------------------------------------------------------------


struct AndKernel(BoolBinaryKernel):
    comptime name = "and_"

    @always_inline
    @staticmethod
    def core[
        W: Int
    ](a: SIMD[DType.bool, W], b: SIMD[DType.bool, W]) -> SIMD[DType.bool, W]:
        return a & b

    @staticmethod
    def apply(
        left: BoolArray,
        right: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        return _kleene[is_and=True](left, right, "and_")


struct OrKernel(BoolBinaryKernel):
    comptime name = "or_"

    @always_inline
    @staticmethod
    def core[
        W: Int
    ](a: SIMD[DType.bool, W], b: SIMD[DType.bool, W]) -> SIMD[DType.bool, W]:
        return a | b

    @staticmethod
    def apply(
        left: BoolArray,
        right: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        return _kleene[is_and=False](left, right, "or_")


struct NotKernel(BoolUnaryKernel):
    comptime name = "not_"

    @always_inline
    @staticmethod
    def core[W: Int](a: SIMD[DType.bool, W]) -> SIMD[DType.bool, W]:
        return ~a

    @staticmethod
    def apply(
        arr: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        # NOT flips the data; nulls propagate unchanged (NOT NULL = NULL).
        var bm: Optional[Bitmap[mut=False]] = None
        if arr.bitmap:
            bm = _validity_ones(arr)
        return BoolArray(
            length=len(arr),
            nulls=arr.null_count(),
            offset=0,
            bitmap=bm^,
            buffer=(~arr.values()).to_immutable(),
        )


struct XorKernel(BoolBinaryKernel):
    comptime name = "xor"

    @always_inline
    @staticmethod
    def core[
        W: Int
    ](a: SIMD[DType.bool, W], b: SIMD[DType.bool, W]) -> SIMD[DType.bool, W]:
        return a ^ b

    @staticmethod
    def apply(
        left: BoolArray,
        right: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        var n = len(left)
        if len(right) != n:
            raise Error("xor: input arrays must have equal length")
        var data = (left.values() ^ right.values()).to_immutable()
        # XOR is fully determined only when both operands are valid.
        if not left.bitmap and not right.bitmap:
            return BoolArray(
                length=n, nulls=0, offset=0, bitmap=None, buffer=data^
            )
        var a_valid = _validity_ones(left)
        var b_valid = _validity_ones(right)
        var valid = a_valid.view() & b_valid.view()
        var valid_bm = valid.to_immutable()
        var nulls = n - valid_bm.view().count_set_bits()
        return BoolArray(
            length=n, nulls=nulls, offset=0, bitmap=valid_bm^, buffer=data^
        )


# ---------------------------------------------------------------------------
# Unary predicates — is_null / not_null (validity) and is_nan / is_inf (values)
# ---------------------------------------------------------------------------


trait NullPredicateKernel(UnaryPredicateKernel):
    """`is_null` / `not_null` — reads only the validity bitmap. `negate` picks the
    polarity (`is_null` inverts validity, `not_null` copies it); the shared
    `apply` runs the mask through the offset-aware byte-level bitmap `apply`."""

    comptime negate: Bool

    @always_inline
    @staticmethod
    def bits[W: Int](v: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
        comptime if Self.negate:
            return ~v
        else:
            return v

    @staticmethod
    def apply(
        arr: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> BoolArray:
        var n = len(arr)
        var validity = arr.to_data().validity()
        var buf: Bitmap[mut=False]
        if validity:
            var out = Bitmap.alloc_uninit(n)
            apply[Self.bits](validity.value(), out.view(), ctx)
            buf = out.to_immutable()
        else:
            var zeroed = Bitmap.alloc_zeroed(n)
            comptime if Self.negate:
                buf = zeroed.to_immutable()  # all-valid -> is_null all False
            else:
                buf = (~zeroed.view()).to_immutable()  # not_null all True
        return BoolArray(length=n, nulls=0, offset=0, bitmap=None, buffer=buf)


trait ValuePredicateKernel(UnaryPredicateKernel):
    """`is_nan` / `is_inf` — scans floating values with the SIMD `core` predicate,
    bit-packing via the shared `apply` (CPU serial/parallel + GPU dispatch by
    `ctx`) and propagating input nulls (result is null where the input is)."""

    @staticmethod
    def core[T: DType, W: Int](x: SIMD[T, W]) -> SIMD[DType.bool, W]:
        ...

    @always_inline
    @staticmethod
    def keep[W: Int](v: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
        return v

    @staticmethod
    def apply(
        arr: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> BoolArray:
        @parameter
        def leaf[T: FloatingType](d: T) raises -> BoolArray:
            ref prim = arr.as_primitive[T]()
            var n = len(prim)
            var result = Bitmap.alloc_device(
                ctx.device.value(), n
            ) if ctx.is_gpu() else Bitmap.alloc_uninit(n)
            apply[T.native, Self.core[T.native, _]](
                prim.values(), result.view(), ctx
            )
            var vbm: Optional[Bitmap[]] = None
            var validity = prim.validity()
            if validity:
                var vout = Bitmap.alloc_uninit(n)
                apply[Self.keep](validity.value(), vout.view())
                vbm = vout.to_immutable()
            return BoolArray(
                length=n,
                nulls=prim.null_count(),
                offset=0,
                bitmap=vbm^,
                buffer=result.to_immutable(),
            )

        return dispatch_over_floating[leaf](arr.dtype())


struct IsNullKernel(NullPredicateKernel):
    comptime name = "is_null"
    comptime negate = True


struct NotNullKernel(NullPredicateKernel):
    comptime name = "not_null"
    comptime negate = False


struct IsNanKernel(ValuePredicateKernel):
    comptime name = "is_nan"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](x: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return isnan(x)


struct IsInfKernel(ValuePredicateKernel):
    comptime name = "is_inf"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](x: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return isinf(x)


# ---------------------------------------------------------------------------
# is_null
# ---------------------------------------------------------------------------


# TODO: it should return with the bitmap from the input array instead of creating a new one, but that requires
def is_null[
    T: PrimitiveType
](
    arr: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Return a bool array that is True where arr has a null value."""
    var length = len(arr)
    if not arr.bitmap:
        return BoolArray(
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=Bitmap.alloc_zeroed(length).to_immutable(),
        )
    var bm = (~arr.bitmap.value().view()).to_immutable()
    return BoolArray(
        length=length, nulls=0, offset=arr.offset, bitmap=None, buffer=bm
    )


def is_null(
    arr: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed is_null."""

    @parameter
    def leaf[T: NumericType](d: T) raises -> AnyArray:
        return is_null(arr.as_primitive[T](), ctx).to_any()

    return dispatch_over_numeric[leaf](arr.dtype())


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def select[
    T: PrimitiveType
](
    mask: BoolArray,
    then_: PrimitiveArray[T],
    else_: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise select: result[i] = then_[i] if mask[i] else else_[i]."""
    var length = len(then_)
    if len(mask) != length or len(else_) != length:
        raise Error("select: input arrays must have equal length")
    var builder = PrimitiveBuilder[T](then_.dtype, length)
    var data_bv = mask.values()
    for i in range(length):
        if data_bv.test(mask.offset + i):
            builder.unsafe_set(i, then_.unsafe_get(i))
        else:
            builder.unsafe_set(i, else_.unsafe_get(i))
    builder.set_length(length)
    return builder.finish()


# TODO: use SIMD select instead of naive element-wise loop when possible
def select(
    mask: AnyArray,
    then_: AnyArray,
    else_: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed select."""
    if then_.dtype() != else_.dtype():
        raise Error(
            t"select: dtype mismatch: {then_.dtype()} vs {else_.dtype()}"
        )

    @parameter
    def leaf[T: NumericType](d: T) raises -> AnyArray:
        return select(
            mask.as_bool(),
            then_.as_primitive[T](),
            else_.as_primitive[T](),
            ctx,
        ).to_any()

    return dispatch_over_numeric[leaf](then_.dtype())
