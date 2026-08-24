"""Element-wise kernels over numeric arrays — arithmetic and comparison.

One module because they are one kind of thing: both are numeric-only, both are
three-tier, both resolve a runtime dtype through `DynType.dispatch_numeric`.
What separates an `AddKernel` from an `LtKernel` is the `core` functor and the
output layout — values for arithmetic, a bit-packed `BoolArray` for comparison.

Comparison stopped being "the string-aware one" when `NumericCompareKernel`
dropped its `comptime StringKernel`: comparing strings is a separate family
(`StringPredicateKernel` in `string.mojo`) whose core is elementwise over
variable-width data and cannot vectorize. Whoever interprets `a < b` picks the
family from the operand dtype — see `_compare` in `marrow/exprold/dynamic.mojo`.

Three tiers per operation:

- **Tier 0 (core)** — `KernelStruct.core[T: DType, W: Int]`: raw SIMD functor,
  no allocation; called directly by expression-node `exec_core[W](idx)` for
  kernel fusion.
- **Tier 1 (apply)** — `KernelStruct.apply[T: PrimitiveType]`: allocates an
  output buffer, propagates null bitmaps, dispatches CPU/GPU via `apply()`.
- **Tier 2 (dispatch)** — `KernelStruct.dispatch(DynArray)`: runtime-typed entry
  point; resolves the dtype to the typed `apply` via `DynType.dispatch_numeric`
  / `.dispatch_floating`.

Structural kernels (filter, sort, concat, …) operate on array layout rather than
element values and are **not** part of this tier scheme.

Null propagation (comparison): if either input is null at `i` the output is null
at `i` (validity = `Bitmap.intersect(left.bitmap, right.bitmap)`). Data bits for
null positions hold the comparison of the underlying values — undefined per the
Arrow spec, but branch-free.

`EqKernel` additionally overloads `apply` for `StructArray`: row equality is
every child column agreeing, which is how the hash table verifies key rows.
"""

import std.math as math

from ..arrays import (
    PrimitiveArray,
    BinaryLikeArray,
    DynArray,
    BoolArray,
    StructArray,
)
from ..buffers import Buffer, Bitmap
from ..views import apply
from ..dtypes import (
    PrimitiveType,
    NumericType,
    FloatingType,
    BinaryLikeType,
    bool_ as bool_dt,
)
from .core import Kernel
from .boolean import AndKernel, NotKernel, XorKernel
from ..execution import ExecContext, GPU_ENABLED


# ---------------------------------------------------------------------------
# Kernel traits — 3-tier interface
# ---------------------------------------------------------------------------


trait BinaryKernel(Kernel):
    """Base for element-wise binary kernels: ``core`` (abstract) + ``apply`` (default).

    Concrete structs define ``comptime name`` and ``core``; subtraits add ``dispatch``.
    """

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        ...

    @staticmethod
    def dispatch(
        left: DynArray,
        right: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Erased entry point. Declared here rather than only on the sub-traits
        so a node generic over `BinaryKernel` can reach it — `FloatBinary` takes
        `DivKernel` (a `BinaryNumericKernel`) and `PowKernel` (a
        `BinaryFloatKernel`), which have no common sub-trait. Both sub-traits
        already default it, and no struct conforms to `BinaryKernel` directly.
        """
        ...

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        Self.expect_same_length(len(left), len(right))
        comptime native = T.native
        var length = len(left)
        var bm = Bitmap.intersect_views(left.validity(), right.validity())
        var buf: Buffer[mut=True]
        comptime if GPU_ENABLED:
            if ctx.is_gpu():
                buf = Buffer.alloc_device[native](ctx.device.value(), length)
            else:
                buf = Buffer.alloc_zeroed[native](length)
        else:
            buf = Buffer.alloc_zeroed[native](length)
        apply[native, native, Self.core[native, _]](
            left.values(), right.values(), buf.view[native](0, length), ctx
        )
        return PrimitiveArray[T](
            dtype=left.dtype.copy(),
            length=length,
            nulls=bm.value().unset_count() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=buf.to_immutable(),
        )


trait BinaryNumericKernel(BinaryKernel):
    """Binary kernel dispatching over all numeric dtypes."""

    @staticmethod
    def dispatch(
        left: DynArray,
        right: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        Self.expect_same_dtype(left.dtype(), right.dtype())

        def leaf[T: NumericType](d: T) raises {imm} -> DynArray:
            return Self.apply(
                left.as_primitive[T](), right.as_primitive[T](), ctx
            ).to_dyn()

        return left.dtype().dispatch_numeric(leaf)


trait BinaryFloatKernel(BinaryKernel):
    """Binary kernel dispatching over floating-point dtypes only."""

    @staticmethod
    def dispatch(
        left: DynArray,
        right: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        Self.expect_same_dtype(left.dtype(), right.dtype())

        def leaf[T: FloatingType](d: T) raises {imm} -> DynArray:
            return Self.apply(
                left.as_primitive[T](), right.as_primitive[T](), ctx
            ).to_dyn()

        return left.dtype().dispatch_floating(leaf)


trait UnaryKernel(Kernel):
    """Base for element-wise unary kernels: ``core`` (abstract) + ``apply`` (default).

    Concrete structs define ``comptime name`` and ``core``; subtraits add ``dispatch``.
    """

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        ...

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        array: PrimitiveArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        comptime native = T.native
        var length = len(array)
        var buf: Buffer[mut=True]
        comptime if GPU_ENABLED:
            if ctx.is_gpu():
                buf = Buffer.alloc_device[native](ctx.device.value(), length)
            else:
                buf = Buffer.alloc_zeroed[native](length)
        else:
            buf = Buffer.alloc_zeroed[native](length)
        apply[native, native, Self.core[native, _]](
            array.values(), buf.view[native](0, length), ctx
        )
        return PrimitiveArray[T](
            dtype=array.dtype.copy(),
            length=length,
            nulls=array.null_count(),
            offset=0,
            bitmap=array.bitmap,
            buffer=buf.to_immutable(),
        )


trait UnaryNumericKernel(UnaryKernel):
    """Unary kernel dispatching over all numeric dtypes."""

    @staticmethod
    def dispatch(
        array: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        def leaf[T: NumericType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_primitive[T](), ctx).to_dyn()

        return array.dtype().dispatch_numeric(leaf)


trait UnaryFloatKernel(UnaryKernel):
    """Unary kernel dispatching over floating-point dtypes only."""

    @staticmethod
    def dispatch(
        array: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        def leaf[T: FloatingType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_primitive[T](), ctx).to_dyn()

        return array.dtype().dispatch_floating(leaf)


# ---------------------------------------------------------------------------
# Kernel structs — BinaryKernel
# ---------------------------------------------------------------------------


struct AddKernel(BinaryNumericKernel):
    comptime name = "add"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b


struct SubKernel(BinaryNumericKernel):
    comptime name = "subtract"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a - b


struct MulKernel(BinaryNumericKernel):
    comptime name = "multiply"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a * b


struct DivKernel(BinaryNumericKernel):
    comptime name = "divide"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        # Replace zeros with 1 to avoid SIGFPE; null positions are masked by bitmap.
        return a / b.eq(0).select(SIMD[T, W](1), b)


struct FloordivKernel(BinaryNumericKernel):
    comptime name = "floordiv"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a // b.eq(0).select(SIMD[T, W](1), b)


struct ModKernel(BinaryNumericKernel):
    comptime name = "modulo"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a % b.eq(0).select(SIMD[T, W](1), b)


struct MinKernel(BinaryNumericKernel):
    comptime name = "min_element_wise"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.min(a, b)


struct MaxKernel(BinaryNumericKernel):
    comptime name = "max_element_wise"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.max(a, b)


# ---------------------------------------------------------------------------
# Kernel structs — UnaryKernel
# ---------------------------------------------------------------------------


struct NegKernel(UnaryNumericKernel):
    comptime name = "negate"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__neg__()


struct AbsKernel(UnaryNumericKernel):
    comptime name = "abs"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__abs__()


struct SignKernel(UnaryNumericKernel):
    comptime name = "sign"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.gt(SIMD[T, W](0)).cast[T]() - a.lt(SIMD[T, W](0)).cast[T]()


struct FloorKernel(UnaryNumericKernel):
    comptime name = "floor"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__floor__()


struct CeilKernel(UnaryNumericKernel):
    comptime name = "ceil"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__ceil__()


struct TruncKernel(UnaryNumericKernel):
    comptime name = "trunc"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__trunc__()


struct RoundKernel(UnaryNumericKernel):
    comptime name = "round"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__round__()


# ---------------------------------------------------------------------------
# Kernel structs — BinaryFloatKernel
# ---------------------------------------------------------------------------


struct PowKernel(BinaryFloatKernel):
    comptime name = "power"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.pow(a, b)


# ---------------------------------------------------------------------------
# Kernel structs — UnaryFloatKernel
# ---------------------------------------------------------------------------


struct SqrtKernel(UnaryFloatKernel):
    comptime name = "sqrt"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return math.sqrt(a)


struct ExpKernel(UnaryFloatKernel):
    comptime name = "exp"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.exp(a)


struct Exp2Kernel(UnaryFloatKernel):
    comptime name = "exp2"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.exp2(a)


struct LogKernel(UnaryFloatKernel):
    comptime name = "ln"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.log(a)


struct Log2Kernel(UnaryFloatKernel):
    comptime name = "log2"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.log2(a)


struct Log10Kernel(UnaryFloatKernel):
    comptime name = "log10"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.log10(a)


struct Log1pKernel(UnaryFloatKernel):
    comptime name = "log1p"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        # std.math.log1p upcasts to float64 in recent nightlies — use log(1+a) instead.
        comptime assert T.is_floating_point()
        return math.log(a + 1)


struct SinKernel(UnaryFloatKernel):
    comptime name = "sin"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.sin(a)


struct CosKernel(UnaryFloatKernel):
    comptime name = "cos"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.cos(a)


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
    ctx: ExecContext = ExecContext.serial(),
) raises -> BoolArray:
    """Binary comparison kernel — compare + bit-pack via apply."""
    comptime native = T.native
    var length = len(left)
    var bm = Bitmap.intersect_views(left.validity(), right.validity())

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
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> BoolArray:
        Self.expect_same_length(len(left), len(right))
        return _binary_cmp[T, func=Self.core[T.native, _]](left, right, ctx)

    @staticmethod
    def dispatch(
        left: DynArray,
        right: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        Self.expect_same_dtype(left.dtype(), right.dtype())

        # `apply` is bound on `PrimitiveType`, so dispatch on that family, not
        # the narrower `NumericType`: temporal, interval and decimal columns all
        # reach the same leaf. Narrowing here made runtime comparison raise on
        # those dtypes, took `equal` -- and with it hash-join key
        # verification and `nullif` -- down too, and left `pruning.mojo` unable
        # to prune a single row group on a date or decimal predicate. CLAUDE.md's
        # "dispatch on the widest family the typed leaf accepts" rule is for
        # exactly this; `filter`/`take` and `sort` were already fixed.
        def leaf[T: PrimitiveType](d: T) raises {imm} -> DynArray:
            return Self.apply(
                left.as_primitive[T](), right.as_primitive[T](), ctx
            ).to_dyn()

        return left.dtype().dispatch_primitive(leaf)


# ---------------------------------------------------------------------------
# Kernel structs
# ---------------------------------------------------------------------------


def equal(
    left: DynArray,
    right: DynArray,
    ctx: ExecContext = ExecContext.serial(),
) raises -> BoolArray:
    """Equality over any comparable dtype, picking the kernel family.

    Fixed-width and variable-width equality are separate kernels — SIMD over
    fixed-width lanes versus an elementwise walk — and `NumericCompareKernel`
    deliberately knows nothing about the latter. Two callers nonetheless need
    equality as a *primitive over an arbitrary dtype* rather than as an operator
    they are interpreting: hash-join row verification, where a key row is an
    arbitrary schema, and `nullif`, which is defined for any dtype with an
    equality. This names that once instead of open-coding the same two-line
    branch at each.

    The split is `binarylike` vs everything else, not `stringlike` vs
    everything else: what decides the kernel is whether the payload is
    variable-width, and `binary` is as variable-width as `string`.

    Not to be confused with the erased arm in `NumericCompare`, which answers a different
    question — which kernel the *user's* `==` meant — and lives in the
    expression layer for that reason.
    """
    if left.dtype() != right.dtype():
        # Checked before either arm, and before the downcast below: `leaf`
        # resolves `T` from the *left* dtype and then reads `right` at that same
        # `T`, so a mismatch here would be one more wrong `as_type` — the exact
        # failure this function's binarylike arm was added to fix.
        raise Error(
            "equal: dtype mismatch, ", left.dtype(), " vs ", right.dtype()
        )
    elif left.dtype() == bool_dt:
        # Booleans are bit-packed, so `BoolArray` is not a `PrimitiveArray` and
        # `dispatch_primitive` raised "dtype is not primitive" on them — a
        # `bool` join key was impossible for the same reason a `binary` one was.
        # Equality over packed bits is XNOR, which the boolean kernels already
        # spell: `Xor` gives "the two differ" with Arrow's validity (valid only
        # where both operands are), and `Not` flips the data while propagating
        # those nulls. Both work a word at a time on the bitmaps, so this is not
        # a slower path than the SIMD one it could not use.
        return NotKernel.apply(
            XorKernel.apply(left.as_bool().copy(), right.as_bool().copy(), ctx),
            ctx,
        )
    elif left.dtype().is_binary_like():
        # `is_binary_like`, not `is_string_like`. `binary` and `large_binary`
        # are perfectly ordinary hash-join key columns, but they are *not*
        # stringlike, so the old `is_string() or is_large_string()` test dropped
        # them into the numeric arm and `dispatch_primitive` raised "dtype is
        # not primitive" — joining on a `binary` key was impossible while the
        # same join on `string` worked.
        def leaf[T: BinaryLikeType](d: T) raises {imm} -> BoolArray:
            return _bytes_equal(
                left.as_binary_like[T](), right.as_binary_like[T]()
            )

        return left.dtype().dispatch_binarylike(leaf)
    else:
        return EqKernel.dispatch(left, right, ctx).as_bool().copy()


def _bytes_equal[
    T: BinaryLikeType
](left: BinaryLikeArray[T], right: BinaryLikeArray[T]) raises -> BoolArray:
    """Element-wise byte equality over a `binarylike` pair.

    `StringEqKernel` computes exactly this for text, and its body is already
    byte-level — but the whole `StringPredicateKernel` family is deliberately
    bound on `StringLikeType`, because `LIKE`, `upper` and `startswith` *are*
    text operations and their `is_string_like` guards exist to say so. Row
    equality is not a text operation: `equal` has to compare whatever a key
    column happens to hold. Widening the text family to reach `binary` would
    have made `upper(binary)` type-check, so the byte-level bound lives here
    instead, next to the one caller that needs it.

    Null semantics match `StringPredicateKernel.apply`: null on either side
    yields null out, and the data bit at a null position is left clear.
    """
    if len(left) != len(right):
        raise Error("equal: length mismatch, ", len(left), " vs ", len(right))
    var n = len(left)
    var bm = Bitmap.intersect_views(left.validity(), right.validity())
    var data = Bitmap.alloc_zeroed(n)
    for i in range(n):
        if left.is_valid(i) and right.is_valid(i):
            if left.unsafe_get(UInt(i)) == right.unsafe_get(UInt(i)):
                data.set(i)
    return BoolArray(
        length=n,
        nulls=bm.value().unset_count() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=data.to_immutable(),
    )


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
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> BoolArray:
        """Row equality: element ``i`` is True iff every child column agrees.

        This is the comparison the hash table verifies key rows with. A key row
        is an arbitrary schema, so the children span dtype families and each one
        goes through `equal` rather than this kernel's own numeric
        `dispatch`."""
        Self.expect_same_dtype(left.dtype, right.dtype)
        var mask = equal(left.children[0].copy(), right.children[0].copy(), ctx)
        for k in range(1, len(left.children)):
            mask = AndKernel.apply(
                mask,
                equal(left.children[k].copy(), right.children[k].copy(), ctx),
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
