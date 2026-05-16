"""Element-wise comparison kernels.

Each kernel compares two ``PrimitiveArray[T]`` values element-wise and returns
a ``BoolArray`` following the Arrow boolean layout.

Null propagation: if either input has a null at position ``i``, the output is
null at ``i`` (validity = ``bitmap_and(left.bitmap, right.bitmap)``).  Data
bits for null positions are set to the comparison result of the underlying
values (undefined per Arrow spec, but branch-free for performance).

Available kernels
-----------------
* ``equal``          — left[i] == right[i]
* ``not_equal``      — left[i] != right[i]
* ``less``           — left[i] <  right[i]
* ``less_equal``     — left[i] <= right[i]
* ``greater``        — left[i] >  right[i]
* ``greater_equal``  — left[i] >= right[i]

Each has a typed overload ``def[T: DataType](PrimitiveArray[T], PrimitiveArray[T])``
and a runtime-typed overload ``def(AnyArray, AnyArray)`` that dispatches via
``bool_array_dispatch``.

Three tiers per kernel (same scheme as ``arithmetic.mojo``):
- **Tier 0 (core)** — ``KernelStruct.core[T, W]``: raw SIMD predicate.
- **Tier 1 (apply)** — ``KernelStruct.apply[T]``: typed array API.
- **Tier 2 (dispatch)** — ``KernelStruct.dispatch(AnyArray)``: type-erased entry.
"""

from ..arrays import (
    BoolArray,
    PrimitiveArray,
    StringArray,
    AnyArray,
    StructArray,
)
from ..buffers import Bitmap
from ..views import apply
from ..dtypes import PrimitiveType, bool_ as bool_dt
from .helpers import bitmap_and, bool_array_dispatch
from .execution import ExecutionContext


# ---------------------------------------------------------------------------
# Generic comparison kernel helper — compare + bit-pack via apply
# ---------------------------------------------------------------------------


def _binary_cmp[
    T: PrimitiveType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) thin -> SIMD[
        DType.bool, W
    ],
    name: StringLiteral = "",
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Binary comparison kernel — compare + bit-pack via apply."""
    if len(left) != len(right):
        raise Error(
            t"{name} arrays must have the same length, got {len(left)} and"
            t" {len(right)}"
        )

    comptime native = T.native
    var length = len(left)
    var bm = bitmap_and(left.bitmap, right.bitmap) if (
        left.bitmap or right.bitmap
    ) else Optional[Bitmap[]]()

    var result = Bitmap.alloc_device(
        ctx.device.value(), length
    ) if ctx.is_gpu() else Bitmap.alloc_uninit(length)
    apply[native, func](left.values(), right.values(), result.view(), ctx)
    return BoolArray(
        length=length,
        nulls=length - bm.value().view().count_set_bits() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=result.to_immutable(),
    )


# ---------------------------------------------------------------------------
# Kernel trait
# ---------------------------------------------------------------------------


trait BinaryCompareKernel:
    """Element-wise binary comparison kernel producing a BoolArray."""

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W], b: SIMD[T, W]
    ) -> SIMD[DType.bool, W]: ...

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext,
    ) raises -> BoolArray: ...

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext,
    ) raises -> AnyArray: ...


# ---------------------------------------------------------------------------
# Kernel structs
# ---------------------------------------------------------------------------


struct EqKernel(BinaryCompareKernel):
    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W], b: SIMD[T, W]
    ) -> SIMD[DType.bool, W]:
        return a.eq(b)

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        return _binary_cmp[T, func=EqKernel.core[T.native, _], name="equal"](left, right, ctx)

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return bool_array_dispatch["equal", EqKernel.apply[_]](left, right, ctx)


struct NeKernel(BinaryCompareKernel):
    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W], b: SIMD[T, W]
    ) -> SIMD[DType.bool, W]:
        return a.ne(b)

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        return _binary_cmp[T, func=NeKernel.core[T.native, _], name="not_equal"](left, right, ctx)

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return bool_array_dispatch["not_equal", NeKernel.apply[_]](left, right, ctx)


struct LtKernel(BinaryCompareKernel):
    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W], b: SIMD[T, W]
    ) -> SIMD[DType.bool, W]:
        return a.lt(b)

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        return _binary_cmp[T, func=LtKernel.core[T.native, _], name="less"](left, right, ctx)

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return bool_array_dispatch["less", LtKernel.apply[_]](left, right, ctx)


struct LeKernel(BinaryCompareKernel):
    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W], b: SIMD[T, W]
    ) -> SIMD[DType.bool, W]:
        return a.le(b)

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        return _binary_cmp[T, func=LeKernel.core[T.native, _], name="less_equal"](left, right, ctx)

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return bool_array_dispatch["less_equal", LeKernel.apply[_]](left, right, ctx)


struct GtKernel(BinaryCompareKernel):
    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W], b: SIMD[T, W]
    ) -> SIMD[DType.bool, W]:
        return a.gt(b)

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        return _binary_cmp[T, func=GtKernel.core[T.native, _], name="greater"](left, right, ctx)

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return bool_array_dispatch["greater", GtKernel.apply[_]](left, right, ctx)


struct GeKernel(BinaryCompareKernel):
    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W], b: SIMD[T, W]
    ) -> SIMD[DType.bool, W]:
        return a.ge(b)

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        return _binary_cmp[T, func=GeKernel.core[T.native, _], name="greater_equal"](left, right, ctx)

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return bool_array_dispatch["greater_equal", GeKernel.apply[_]](left, right, ctx)


# ---------------------------------------------------------------------------
# Typed public API — thin wrappers
# ---------------------------------------------------------------------------


def equal[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Element-wise equality: result[i] = left[i] == right[i]."""
    return EqKernel.apply[T](left, right, ctx)


def not_equal[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Element-wise inequality: result[i] = left[i] != right[i]."""
    return NeKernel.apply[T](left, right, ctx)


def less[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Element-wise less-than: result[i] = left[i] < right[i]."""
    return LtKernel.apply[T](left, right, ctx)


def less_equal[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Element-wise less-or-equal: result[i] = left[i] <= right[i]."""
    return LeKernel.apply[T](left, right, ctx)


def greater[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Element-wise greater-than: result[i] = left[i] > right[i]."""
    return GtKernel.apply[T](left, right, ctx)


def greater_equal[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Element-wise greater-or-equal: result[i] = left[i] >= right[i]."""
    return GeKernel.apply[T](left, right, ctx)


# ---------------------------------------------------------------------------
# String overloads
# ---------------------------------------------------------------------------


def equal(
    left: StringArray,
    right: StringArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Element-wise string equality."""
    var n = len(left)
    if len(right) != n:
        raise Error("equal: string arrays must have the same length")
    var bm = bitmap_and(left.bitmap, right.bitmap)
    var bm_builder = Bitmap.alloc_zeroed(n)
    for i in range(n):
        var eq = String(left.unsafe_get(UInt(i))) == String(
            right.unsafe_get(UInt(i))
        )
        if eq:
            bm_builder.set(i)
    return BoolArray(
        length=n,
        nulls=n - bm.value().view().count_set_bits() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=bm_builder.to_immutable(),
    )


# ---------------------------------------------------------------------------
# Runtime-typed overloads
# ---------------------------------------------------------------------------


def equal(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed equal."""
    if left.dtype().is_string():
        return equal(left.as_string(), right.as_string(), ctx).to_any()
    return EqKernel.dispatch(left, right, ctx)


def equal(
    left: StructArray,
    right: StructArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Element-wise struct equality: all corresponding columns must match.

    Returns a boolean array where element ``i`` is True iff
    ``left[i] == right[i]`` across every child column.
    """
    from .boolean import and_

    var n_keys = len(left.children)
    var mask = (
        equal(left.children[0].copy(), right.children[0].copy(), ctx)
        .as_bool()
        .copy()
    )
    for k in range(1, n_keys):
        mask = and_(
            mask,
            equal(left.children[k].copy(), right.children[k].copy(), ctx)
            .as_bool()
            .copy(),
            ctx,
        )
    return mask^


def not_equal(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed not_equal."""
    return NeKernel.dispatch(left, right, ctx)


def less(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed less."""
    return LtKernel.dispatch(left, right, ctx)


def less_equal(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed less_equal."""
    return LeKernel.dispatch(left, right, ctx)


def greater(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed greater."""
    return GtKernel.dispatch(left, right, ctx)


def greater_equal(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed greater_equal."""
    return GeKernel.dispatch(left, right, ctx)
