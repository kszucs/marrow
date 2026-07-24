"""Element-wise comparison kernels.

Each kernel compares two ``PrimitiveArray[T]`` values element-wise and returns
a ``BoolArray`` following the Arrow boolean layout.

Null propagation: if either input has a null at position ``i``, the output is
null at ``i`` (validity = ``bitmap_and(left.bitmap, right.bitmap)``).  Data
bits for null positions are set to the comparison result of the underlying
values (undefined per Arrow spec, but branch-free for performance).

Available kernels
-----------------
* ``EqKernel``  — left[i] == right[i]
* ``NeKernel``  — left[i] != right[i]
* ``LtKernel``  — left[i] <  right[i]
* ``LeKernel``  — left[i] <= right[i]
* ``GtKernel``  — left[i] >  right[i]
* ``GeKernel``  — left[i] >= right[i]

Three tiers per kernel (same scheme as ``arithmetic.mojo``):
- **Tier 0 (core)** — ``KernelStruct.core[T, W]``: raw SIMD predicate.
- **Tier 1 (apply)** — ``KernelStruct.apply[T]``: typed array API.
- **Tier 2 (dispatch)** — ``KernelStruct.dispatch(AnyArray)``: type-erased entry.

All six kernels also implement ``str_predicate`` / ``apply_string``: ``dispatch``
routes ``string`` / ``large_string`` inputs to a lexicographic byte comparison
(UTF-8 byte order equals codepoint order), so ``<`` ``<=`` ``>`` ``>=`` ``==``
``!=`` all work on strings. ``equal`` additionally keeps free-function overloads
for ``StringArray`` / ``StructArray`` / ``AnyArray`` (nested struct equality is
not folded into ``EqKernel``).
"""

from ..arrays import (
    BoolArray,
    PrimitiveArray,
    StringArray,
    BinaryLikeArray,
    AnyArray,
    StructArray,
)
from ..buffers import Bitmap
from ..views import apply
from ..dtypes import (
    NumericType,
    PrimitiveType,
    StringLikeType,
    bool_ as bool_dt,
)
from ..utils import dispatch_over_numeric
from .helpers import Kernel, bitmap_and
from .execution import ExecutionContext


# ---------------------------------------------------------------------------
# Generic comparison kernel helper — compare + bit-pack via apply
# ---------------------------------------------------------------------------


def _binary_cmp[
    T: PrimitiveType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) thin -> SIMD[
        DType.bool, W
    ],
    name: String = "",
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
    var bm = bitmap_and(left.bitmap.copy(), right.bitmap.copy()) if (
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


trait BinaryCompareKernel(Kernel):
    """Element-wise binary comparison kernel producing a BoolArray.

    Concrete structs define ``comptime name`` and ``core``; ``apply`` and
    ``dispatch`` have default implementations.
    """

    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        ...

    @staticmethod
    def str_predicate[
        o1: Origin, o2: Origin
    ](a: StringSlice[o1], b: StringSlice[o2]) -> Bool:
        """Scalar per-element string comparison (lexicographic byte ordering).
        """
        ...

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        return _binary_cmp[T, func=Self.core[T.native, _], name=Self.name](
            left, right, ctx
        )

    @staticmethod
    def apply_string[
        L: StringLikeType, R: StringLikeType
    ](left: BinaryLikeArray[L], right: BinaryLikeArray[R]) raises -> BoolArray:
        """Element-wise string comparison producing a bit-packed BoolArray.

        Lexicographic byte ordering (UTF-8 byte order equals codepoint order).
        Validity is the AND of the operand bitmaps, matching the numeric path.
        """
        var n = len(left)
        if len(right) != n:
            raise Error(
                t"{Self.name}: arrays must have the same length, got {n} and"
                t" {len(right)}"
            )
        var bm = bitmap_and(left.bitmap.copy(), right.bitmap.copy())
        var data = Bitmap.alloc_zeroed(n)
        for i in range(n):
            if left.is_valid(i) and right.is_valid(i):
                if Self.str_predicate(
                    left.unsafe_get(UInt(i)), right.unsafe_get(UInt(i))
                ):
                    data.set(i)
        return BoolArray(
            length=n,
            nulls=n - bm.value().view().count_set_bits() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=data.to_immutable(),
        )

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        if left.dtype() != right.dtype():
            raise Error(
                t"{Self.name}: dtype mismatch: {left.dtype()} vs"
                t" {right.dtype()}"
            )

        @parameter
        def leaf[T: NumericType](d: T) raises -> AnyArray:
            return Self.apply(
                left.as_primitive[T](), right.as_primitive[T](), ctx
            ).to_any()

        if left.dtype().is_string():
            return Self.apply_string(
                left.as_string(), right.as_string()
            ).to_any()
        elif left.dtype().is_large_string():
            return Self.apply_string(
                left.as_large_string(), right.as_large_string()
            ).to_any()
        else:
            return dispatch_over_numeric[leaf](left.dtype())


# ---------------------------------------------------------------------------
# Kernel structs
# ---------------------------------------------------------------------------


struct EqKernel(BinaryCompareKernel):
    comptime name = "equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.eq(b)

    @always_inline
    @staticmethod
    def str_predicate[
        o1: Origin, o2: Origin
    ](a: StringSlice[o1], b: StringSlice[o2]) -> Bool:
        return a == b


struct NeKernel(BinaryCompareKernel):
    comptime name = "not_equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.ne(b)

    @always_inline
    @staticmethod
    def str_predicate[
        o1: Origin, o2: Origin
    ](a: StringSlice[o1], b: StringSlice[o2]) -> Bool:
        return a != b


struct LtKernel(BinaryCompareKernel):
    comptime name = "less"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.lt(b)

    @always_inline
    @staticmethod
    def str_predicate[
        o1: Origin, o2: Origin
    ](a: StringSlice[o1], b: StringSlice[o2]) -> Bool:
        return a < b


struct LeKernel(BinaryCompareKernel):
    comptime name = "less_equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.le(b)

    @always_inline
    @staticmethod
    def str_predicate[
        o1: Origin, o2: Origin
    ](a: StringSlice[o1], b: StringSlice[o2]) -> Bool:
        return a <= b


struct GtKernel(BinaryCompareKernel):
    comptime name = "greater"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.gt(b)

    @always_inline
    @staticmethod
    def str_predicate[
        o1: Origin, o2: Origin
    ](a: StringSlice[o1], b: StringSlice[o2]) -> Bool:
        return a > b


struct GeKernel(BinaryCompareKernel):
    comptime name = "greater_equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.ge(b)

    @always_inline
    @staticmethod
    def str_predicate[
        o1: Origin, o2: Origin
    ](a: StringSlice[o1], b: StringSlice[o2]) -> Bool:
        return not (a < b)  # StringSlice has no __ge__(StringSlice) overload


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
    var bm = bitmap_and(left.bitmap.copy(), right.bitmap.copy())
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
    from .boolean import AndKernel

    var n_keys = len(left.children)
    var mask = (
        equal(left.children[0].copy(), right.children[0].copy(), ctx)
        .as_bool()
        .copy()
    )
    for k in range(1, n_keys):
        mask = AndKernel.apply(
            mask,
            equal(left.children[k].copy(), right.children[k].copy(), ctx)
            .as_bool()
            .copy(),
            ctx,
        )
    return mask^
