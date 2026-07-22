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


from ..arrays import BoolArray, PrimitiveArray, AnyArray
from ..buffers import Bitmap
from ..builders import PrimitiveBuilder
from ..dtypes import (
    NumericType,
    PrimitiveType,
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
from ..utils import dispatch_over_numeric
from ..views import BitmapView
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
        var length = len(left)
        if len(right) != length:
            raise Error("and_: input arrays must have equal length")
        return BoolArray(
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=(left.values() & right.values()).to_immutable(),
        )


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
        var length = len(left)
        if len(right) != length:
            raise Error("or_: input arrays must have equal length")
        return BoolArray(
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=(left.values() | right.values()).to_immutable(),
        )


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
        return BoolArray(
            length=len(arr),
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=(~arr.values()).to_immutable(),
        )


# Bool-result markers NOT IMPLEMENTED yet (compute `core`/`apply` are TODO) —
# named so the typed expression layer can reference them.
struct XorKernel(Kernel):
    comptime name = "xor"


struct IsNullKernel(Kernel):
    comptime name = "is_null"


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
