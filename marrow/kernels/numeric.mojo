# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Element-wise kernels over numeric arrays — arithmetic and comparison.

One module because they are one kind of thing: both are numeric-only, both are
three-tier, both resolve a runtime dtype through `DynType.dispatch_numeric`.
What separates an `AddKernel` from an `LtKernel` is the `core` functor and the
output layout — values for arithmetic, a bit-packed `BoolArray` for comparison.

Comparison stopped being "the string-aware one" when `NumericCompareKernel`
dropped its `comptime StringKernel`: comparing strings is a separate family
(`StringPredicateKernel` in `string.mojo`) whose core is elementwise over
variable-width data and cannot vectorize. Whoever interprets `a < b` picks the
family from the operand dtype — see the runtime lane in
`marrow/expr/runtime/values.mojo`.

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

The six comparison kernels carry a `nan_safe` parameter, because marrow needs
two answers about NaN — see `EqKernel`.
"""

import std.math as math

from ..arrays import (
    PrimitiveArray,
    BinaryLikeArray,
    BytesArray,
    DynArray,
    BoolArray,
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
from .boolean import NotKernel, XorKernel
from .string import StringEqKernel
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
    def overflows[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        """The lanes of a signed integer division whose quotient does not fit:
        `MIN / -1`. All false for unsigned and floating types."""
        comptime if T.is_integral() and T.is_signed():
            return a.eq(SIMD[T, W](Scalar[T].MIN)) & b.eq(SIMD[T, W](-1))
        else:
            return SIMD[DType.bool, W](fill=False)

    @always_inline
    @staticmethod
    def divisor[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        """`b` with 1 in every zero lane and, for a signed integer, every
        `MIN / -1` lane -- the two an integer division cannot take: x86 `idiv`
        traps on both and LLVM leaves both undefined. Dividing by 1 there
        leaves `MIN`, the two's-complement wrap of `MIN // -1`, and a
        remainder of 0, which is exact. `FloordivKernel` and `ModKernel`
        divide by it too."""
        return (b.eq(0) | Self.overflows(a, b)).select(SIMD[T, W](1), b)

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        """True division: floats divide by zero, integers cannot.

        The float lane is left alone, so `10.0 / 0.0` is `inf`, `-10.0 / 0.0`
        is `-inf` and `0.0 / 0.0` is `nan` — IEEE 754's answers, and what
        DuckDB 1.5.5 and `pyarrow.compute.divide` both give, measured
        2026-09-22. Getting them is a *deletion* rather than a rule: the
        answer depends on the dividend's sign and on whether it is zero, so no
        substituted divisor can produce all three. That is not the rule `//`
        and `%` follow — those answer NULL, because division by zero has a
        value in the reals' completion and integer division by zero does not.

        **Integers keep the substituted divisor**, because a lane has nothing
        else to answer with: `inf` is not an `int64`, SIMD can neither raise
        nor write a null, and `idiv` by zero traps on x86. The 1 is a harmless
        value rather than an answer, the same device as
        `FloordivKernel.core`'s — and, as there, the layer above supplies the
        meaning. No marrow query reaches this arm, both expression lanes being
        `float64` before `Div`; the one caller that can is `pc.divide`, and
        `python/bindings/compute.mojo` checks the divisor there and raises,
        which is what `pyarrow.compute.divide` does.

        **The signed `MIN / -1` lane answers 0**, as `pyarrow.compute.divide`
        does: its quotient does not fit, and `idiv` traps on x86 rather than
        wrapping, so that lane divides by 1 too and the answer is substituted.
        """
        comptime if T.is_integral():
            var q = a / Self.divisor(a, b)
            comptime if T.is_signed():
                return Self.overflows(a, b).select(SIMD[T, W](0), q)
            else:
                return q
        else:
            return a / b


struct FloordivKernel(BinaryNumericKernel):
    comptime name = "floordiv"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        """`//`: SQL's truncating quotient on integers, Python's floor on floats.

        Two things happen here, and only one of them is conditional.

        - **The zero divisor**, always. A lane can neither raise nor produce
          a null, so it divides by 1 — a value, not an answer. Turning that
          row into SQL's NULL is the expression layer's job, not this one's:
          `pyarrow.compute`, whose names these mirror, does not null either.
          Both lanes do it, `DivisionBinary` and `RuntimeValue._null_zeros`.
        - **The rounding, integers only.** Floored and truncated division
          differ by one quotient step, and only where the signs differ and the
          division is inexact; `a - q * d` is that remainder, a multiply rather
          than a second division.

        **Floats keep Python's floor deliberately.** Neither convention is
        SQL's: DuckDB's `//` on DOUBLE is plain division — `-1.5 // 3.0` is
        `-0.5`, `-7.5 // 2.0` is `-3.75` (measured 2026-09-04) — so truncating
        would trade one divergence for another while breaking `//`'s meaning.
        No golden case asks, and `//` reads as floor division everywhere else
        in the language.
        """
        var d = DivKernel.divisor(a, b)
        var q = a // d
        comptime if T.is_integral():
            var inexact = (a - q * d).ne(0) & (a.lt(0) ^ d.lt(0))
            return inexact.select(q + 1, q)
        else:
            return q


struct ModKernel(BinaryNumericKernel):
    comptime name = "modulo"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        """`%`: on integers the remainder takes the sign of the **dividend**.

        `FloordivKernel.core` documents both corrections and the reason floats
        are excluded from the second; stepping the remainder back by one
        divisor is what keeps `a == (a // b) * b + a % b` true under the
        truncating rule, so the two must be conditional together or the
        identity breaks.
        """
        var d = DivKernel.divisor(a, b)
        var r = a % d
        comptime if T.is_integral():
            var inexact = r.ne(0) & (a.lt(0) ^ d.lt(0))
            return inexact.select(r - d, r)
        else:
            return r


struct MinKernel(BinaryNumericKernel):
    comptime name = "minimum"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.min(a, b)


struct MaxKernel(BinaryNumericKernel):
    comptime name = "maximum"

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
        # verification and `nullif` -- down too, and left the runtime lane
        # unable to prune a row group on a date or decimal predicate, which is
        # still the only lane that can. CLAUDE.md's
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
    """The user's `=` over any comparable dtype, picking the kernel family: a
    NULL on either side gives NULL, and a NaN equals nothing. Key identity,
    where both are equal to themselves, is `KeyCompare` (`hashing.mojo`).

    Fixed-width and variable-width equality are separate kernels — SIMD over
    fixed-width lanes versus an elementwise walk — and `NumericCompareKernel`
    deliberately knows nothing about the latter. `nullif` nonetheless needs
    equality as a *primitive over an arbitrary dtype* rather than as an
    operator it is interpreting, and this names that once.

    The split is `primitive` vs everything else: what decides the kernel is
    whether the payload is fixed-width, and every byte string -- `binary` as
    much as `string`, in either layout -- is not.

    Not to be confused with the erased arm in `NumericCompare`, which answers a different
    question — which kernel the *user's* `==` meant — and lives in the
    expression layer for that reason.
    """
    # Checked before every arm, and before the downcast below: `leaf` resolves
    # `T` from the *left* dtype and then reads `right` at that same `T`, so a
    # mismatch here would be one more wrong `as_type` — the exact failure this
    # function's binarylike arm was added to fix. `expect_same_dtype` rather
    # than a hand-rolled raise, so one condition has one message whichever
    # entry point reached it.
    EqKernel.expect_same_dtype(left.dtype(), right.dtype())

    if left.dtype() == bool_dt:
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
    elif left.dtype().is_primitive():
        return EqKernel.dispatch(left, right, ctx).as_bool().copy()
    else:
        # Every byte-string dtype, not just the text ones. `binary` and
        # `large_binary` are perfectly ordinary hash-join key columns, but an
        # earlier `is_string() or is_large_string()` test dropped them into the
        # numeric arm and `dispatch_primitive` raised "dtype is not primitive"
        # -- joining on a `binary` key was impossible while the same join on
        # `string` worked. Anything else (a nested key) raises in the dispatch.
        # `apply` through a local leaf, not `StringEqKernel.dispatch`, which
        # round-trips the result through an erased array.
        def leaf[A: BytesArray](arr: A) raises {imm} -> BoolArray:
            return StringEqKernel.apply(arr, right.as_type[A]())

        return left.dispatch_bytes(leaf)


struct EqKernel[nan_safe: Bool = False](NumericCompareKernel):
    """Equality. `nan_safe=True` additionally makes a NaN equal itself.

    The parameter exists because marrow needs both answers and they differ on
    exactly one input. The default is the user's `=`: `EqKernel` is
    `pyarrow.compute.equal`, so a NaN equals nothing, itself included.
    `EqKernel[nan_safe=True]` is the NaN half of key identity: `KeyCompare`
    (`hashing.mojo`) compares fixed-width keys with its `core`.

    Not IEEE `totalOrder`, which is a bitwise compare: that separates `-0.0`
    from `0.0` and one NaN payload from another, where both parameterizations
    here agree `-0.0 == 0.0` and this one folds every NaN together.
    """

    # Constant, and deliberately not read off `nan_safe`: this is the
    # *operator's* name and both parameterizations are equality. It reaches plan
    # rendering (`NumericCompare.write_to`) and the pruning rule's
    # `Self.K.name == EqKernel.name` test, so making it vary printed
    # `equal_nan_safe(a, 1)` in `explain()` and silently dropped zone-map
    # pruning for every equality predicate. A comptime member that does *not*
    # read the struct parameter resolves off the unbound name too, which is why
    # the siblings can say `GtKernel.name`.
    comptime name = "equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        comptime if Self.nan_safe and T.is_floating_point():
            return a.eq(b) | (math.isnan(a) & math.isnan(b))
        else:
            return a.eq(b)

    @always_inline
    @staticmethod
    def canonical[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        """One representative of each class of values `core` calls equal:
        `-0.0` becomes `0.0` and, under `nan_safe`, every NaN one NaN, so
        equal values share one bit pattern. Integers are their own.

        `+ 0.0` folds the zeros in one instruction: adding zero is exact for
        every float, and `-0.0 + 0.0` is `+0.0`."""
        comptime if T.is_floating_point():
            var zeroed = a + SIMD[T, W](0)
            comptime if Self.nan_safe:
                return math.isnan(a).select(SIMD[T, W](math.nan[T]()), zeroed)
            else:
                return zeroed
        else:
            return a


struct NeKernel[nan_safe: Bool = False](NumericCompareKernel):
    """`equal`'s complement, under whichever NaN rule `nan_safe` selects.

    `a.ne(b)` is *not* the IEEE answer: it lowers to an ordered compare, so it
    answered False whenever either operand was a NaN — `nan <> 1.0` came out
    False where pyarrow says True. Negating `eq` is correct under both rules by
    construction, and keeps `<>` the exact complement of `=` when `nan_safe`
    makes the latter total."""

    comptime name = "not_equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        return ~EqKernel[Self.nan_safe].core[T, W](a, b)


struct LtKernel[nan_safe: Bool = False](NumericCompareKernel):
    """Ordering. `nan_safe=True` adopts SQL's total order, where NaN is greater
    than every number and than `inf` — DuckDB 1.5.5, DataFusion 54 and Polars
    1.43 all answer `nan > 1.0` true."""

    comptime name = "less"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        comptime if Self.nan_safe and T.is_floating_point():
            # `~(a >= b)` rather than `a < b | …`: one `isnan`, not two, and it
            # makes `<` the exact complement of `>=` under the total order.
            return ~(a.ge(b) | math.isnan(a))
        else:
            return a.lt(b)


struct LeKernel[nan_safe: Bool = False](NumericCompareKernel):
    """Ordering — see `LtKernel`. Under the total order a NaN is also `<=` a NaN,
    which IEEE denies."""

    comptime name = "less_equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        comptime if Self.nan_safe and T.is_floating_point():
            return a.le(b) | math.isnan(b)
        else:
            return a.le(b)


struct GtKernel[nan_safe: Bool = False](NumericCompareKernel):
    comptime name = "greater"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        comptime if Self.nan_safe and T.is_floating_point():
            # The complement of `<=`, for the same reason as `LtKernel`.
            return ~(a.le(b) | math.isnan(b))
        else:
            return a.gt(b)


struct GeKernel[nan_safe: Bool = False](NumericCompareKernel):
    """Ordering — see `LtKernel`. Under the total order a NaN is also `>=` a NaN.
    """

    comptime name = "greater_equal"

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        comptime if Self.nan_safe and T.is_floating_point():
            return a.ge(b) | math.isnan(a)
        else:
            return a.ge(b)
