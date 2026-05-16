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
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    bool_ as bool_dt,
)
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
    def core[W: Int](
        a: SIMD[DType.bool, W], b: SIMD[DType.bool, W]
    ) -> SIMD[DType.bool, W]: ...

    @staticmethod
    def apply(
        left: BoolArray,
        right: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray: ...

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        if left.dtype() != bool_dt or right.dtype() != bool_dt:
            raise Error(t"{Self.name}: inputs must be bool arrays")
        return Self.apply(left.as_bool().copy(), right.as_bool().copy(), ctx).to_any()


trait BoolUnaryKernel(Kernel):
    """Element-wise unary boolean kernel (BoolArray → BoolArray).

    Concrete structs define ``core`` and ``apply``; ``dispatch`` has a default
    implementation that type-checks the input and delegates to ``apply``.
    """

    @staticmethod
    def core[W: Int](a: SIMD[DType.bool, W]) -> SIMD[DType.bool, W]: ...

    @staticmethod
    def apply(
        arr: BoolArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray: ...

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
    def core[W: Int](
        a: SIMD[DType.bool, W], b: SIMD[DType.bool, W]
    ) -> SIMD[DType.bool, W]:
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
    def core[W: Int](
        a: SIMD[DType.bool, W], b: SIMD[DType.bool, W]
    ) -> SIMD[DType.bool, W]:
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


# ---------------------------------------------------------------------------
# Public API — thin wrappers
# ---------------------------------------------------------------------------


def and_(
    lhs: BoolArray,
    rhs: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Bitwise AND of two bit-packed bool arrays."""
    return AndKernel.apply(lhs, rhs, ctx)


def or_(
    lhs: BoolArray,
    rhs: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Bitwise OR of two bit-packed bool arrays."""
    return OrKernel.apply(lhs, rhs, ctx)


def not_(
    arr: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Bitwise NOT of a bit-packed bool array."""
    return NotKernel.apply(arr, ctx)


def and_(
    lhs: AnyArray,
    rhs: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed AND: dispatches to the typed BoolArray overload."""
    return AndKernel.dispatch(lhs, rhs, ctx)


def or_(
    lhs: AnyArray,
    rhs: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed OR: dispatches to the typed BoolArray overload."""
    return OrKernel.dispatch(lhs, rhs, ctx)


def not_(
    arr: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed NOT: dispatches to the typed BoolArray overload."""
    return NotKernel.dispatch(arr, ctx)


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
    if arr.dtype() == int8:
        return is_null(arr.as_int8(), ctx).to_any()
    elif arr.dtype() == int16:
        return is_null(arr.as_int16(), ctx).to_any()
    elif arr.dtype() == int32:
        return is_null(arr.as_int32(), ctx).to_any()
    elif arr.dtype() == int64:
        return is_null(arr.as_int64(), ctx).to_any()
    elif arr.dtype() == uint8:
        return is_null(arr.as_uint8(), ctx).to_any()
    elif arr.dtype() == uint16:
        return is_null(arr.as_uint16(), ctx).to_any()
    elif arr.dtype() == uint32:
        return is_null(arr.as_uint32(), ctx).to_any()
    elif arr.dtype() == uint64:
        return is_null(arr.as_uint64(), ctx).to_any()
    elif arr.dtype() == float16:
        return is_null(arr.as_float16(), ctx).to_any()
    elif arr.dtype() == float32:
        return is_null(arr.as_float32(), ctx).to_any()
    elif arr.dtype() == float64:
        return is_null(arr.as_float64(), ctx).to_any()
    raise Error(t"is_null: unsupported dtype {arr.dtype()}")


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
    ref bool_mask = mask.as_bool()
    if then_.dtype() == int8:
        return select(bool_mask, then_.as_int8(), else_.as_int8(), ctx).to_any()
    elif then_.dtype() == int16:
        return select(
            bool_mask, then_.as_int16(), else_.as_int16(), ctx
        ).to_any()
    elif then_.dtype() == int32:
        return select(
            bool_mask, then_.as_int32(), else_.as_int32(), ctx
        ).to_any()
    elif then_.dtype() == int64:
        return select(
            bool_mask, then_.as_int64(), else_.as_int64(), ctx
        ).to_any()
    elif then_.dtype() == uint8:
        return select(
            bool_mask, then_.as_uint8(), else_.as_uint8(), ctx
        ).to_any()
    elif then_.dtype() == uint16:
        return select(
            bool_mask, then_.as_uint16(), else_.as_uint16(), ctx
        ).to_any()
    elif then_.dtype() == uint32:
        return select(
            bool_mask, then_.as_uint32(), else_.as_uint32(), ctx
        ).to_any()
    elif then_.dtype() == uint64:
        return select(
            bool_mask, then_.as_uint64(), else_.as_uint64(), ctx
        ).to_any()
    elif then_.dtype() == float16:
        return select(
            bool_mask, then_.as_float16(), else_.as_float16(), ctx
        ).to_any()
    elif then_.dtype() == float32:
        return select(
            bool_mask, then_.as_float32(), else_.as_float32(), ctx
        ).to_any()
    elif then_.dtype() == float64:
        return select(
            bool_mask, then_.as_float64(), else_.as_float64(), ctx
        ).to_any()
    raise Error(t"select: unsupported dtype {then_.dtype()}")
