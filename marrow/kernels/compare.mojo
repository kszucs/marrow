"""Element-wise comparison kernels.

Each kernel compares two ``PrimitiveArray[T]`` values element-wise and returns
a ``BoolArray`` following the Arrow boolean layout.

Null propagation: if either input has a null at position ``i``, the output is
null at ``i`` (validity = ``Bitmap.intersect(left.bitmap, right.bitmap)``).  Data
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

These are **numeric** comparisons. The same operators over ``string`` /
``large_string`` are a separate family — ``StringPredicateKernel`` and its
``StringLtKernel``/``StringEqKernel``/… conformers in ``string.mojo`` — because
a variable-width predicate is elementwise and cannot vectorize. Whoever
interprets ``a < b`` picks the family from the operand dtype; see
``_compare`` in ``marrow/expr/dynamic.mojo``.

``EqKernel`` additionally overloads ``apply`` for ``StructArray``: row equality
is every child column agreeing, which is how the hash table verifies key rows.
"""

from ..arrays import (
    BoolArray,
    PrimitiveArray,
    AnyArray,
    StructArray,
)
from ..buffers import Bitmap
from ..views import apply
from ..dtypes import NumericType, PrimitiveType
from .core import Kernel
from .boolean import AndKernel
from .execution import ExecutionContext
from ..utils import GPU_ENABLED


# ---------------------------------------------------------------------------
# Generic comparison kernel helper — compare + bit-pack via apply
# ---------------------------------------------------------------------------


def _binary_cmp[
    T: PrimitiveType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) thin -> SIMD[
        DType.bool, W
    ],
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Binary comparison kernel — compare + bit-pack via apply."""
    comptime native = T.native
    var length = len(left)
    var bm = Bitmap.intersect(left.bitmap.copy(), right.bitmap.copy()) if (
        left.bitmap or right.bitmap
    ) else Optional[Bitmap[]]()

    var result: Bitmap[mut=True]
    comptime if GPU_ENABLED:
        result = Bitmap.alloc_device(
            ctx.device.value(), length
        ) if ctx.is_gpu() else Bitmap.alloc_uninit(length)
    else:
        result = Bitmap.alloc_uninit(length)
    apply[native, func](left.values(), right.values(), result.view(), ctx)
    return BoolArray(
        length=length,
        nulls=bm.value().unset_count() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=result.to_immutable(),
    )


# ---------------------------------------------------------------------------
# Kernel trait
# ---------------------------------------------------------------------------


trait NumericCompareKernel(Kernel):
    """Element-wise comparison over fixed-width lanes, producing a BoolArray.

    Concrete structs define ``comptime name`` and ``core`` (the SIMD predicate);
    ``apply`` and ``dispatch`` are defaulted.

    **Numeric only.** Comparing strings is a different kernel family —
    `StringPredicateKernel` in `string.mojo`, whose core is elementwise over
    variable-width data and cannot vectorize. This trait used to carry a
    `comptime StringKernel` naming its string counterpart, so every numeric
    comparison had to know about strings and `dispatch` branched on dtype at run
    time to pick between two unrelated implementations. Which family `a < b`
    means is a question about the operands, and it belongs to whoever is
    interpreting the operator, not to the SIMD kernel.
    """

    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        ...

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        Self.expect_same_length(len(left), len(right))
        return _binary_cmp[T, func=Self.core[T.native, _]](left, right, ctx)

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        Self.expect_same_dtype(left.dtype(), right.dtype())

        @parameter
        def leaf[T: NumericType](d: T) raises -> AnyArray:
            return Self.apply(
                left.as_primitive[T](), right.as_primitive[T](), ctx
            ).to_any()

        return left.dtype().dispatch_numeric[leaf]()


# ---------------------------------------------------------------------------
# Kernel structs
# ---------------------------------------------------------------------------


struct EqKernel(NumericCompareKernel):
    comptime name = "equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.eq(b)

    @staticmethod
    def apply(
        left: StructArray,
        right: StructArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        """Row equality: element ``i`` is True iff every child column agrees.

        This is the comparison the hash table verifies key rows with, so it
        recurses through `dispatch` and picks up each child's own dtype arm."""
        Self.expect_same_dtype(left.dtype, right.dtype)
        var mask = (
            Self.dispatch(
                left.children[0].copy(), right.children[0].copy(), ctx
            )
            .as_bool()
            .copy()
        )
        for k in range(1, len(left.children)):
            mask = AndKernel.apply(
                mask,
                Self.dispatch(
                    left.children[k].copy(), right.children[k].copy(), ctx
                )
                .as_bool()
                .copy(),
                ctx,
            )
        return mask^


struct NeKernel(NumericCompareKernel):
    comptime name = "not_equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.ne(b)


struct LtKernel(NumericCompareKernel):
    comptime name = "less"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.lt(b)


struct LeKernel(NumericCompareKernel):
    comptime name = "less_equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.le(b)


struct GtKernel(NumericCompareKernel):
    comptime name = "greater"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.gt(b)


struct GeKernel(NumericCompareKernel):
    comptime name = "greater_equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return a.ge(b)
