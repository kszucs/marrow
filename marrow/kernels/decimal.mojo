# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Decimal arithmetic — Arrow C++'s result types over the unscaled integers.

A decimal is an integer plus a column-level `(precision, scale)`, so every
operation is two questions: **what type is the answer**, and **what integer
operation computes it**. The first is `DecimalPromotion`, the second a
`DecimalBinaryKernel`, and the two are separate because the comptime
expression lane answers the first at bind time and fuses the second into a
lane, while the erased entry point here runs them back to back.

The rules are Arrow C++'s (`CastBinaryDecimalArgs` and the
`ResolveDecimal*Output` resolvers in `compute/kernels/`), which is what makes
the answers bit-identical to `pyarrow.compute`:

| op | operands are rescaled to | result |
|---|---|---|
| `+` `-` | `s = max(s1, s2)` | `(max(p1 - s1, p2 - s2) + s + 1, s)` |
| `*` | unchanged | `(p1 + p2 + 1, s1 + s2)` |
| `/` | dividend up by `max(4, s1 + p2 - s2 + 1) + s2 - s1` | `(p1 + up, s1 + up - s2)`, truncated |
| compare | both to the max scale | bool |

An integer operand is a decimal of scale 0 with as many digits as its type
holds (`int32` is `(10, 0)`); a float operand is not decimal arithmetic at all
and the caller casts both sides to `float64`, as Arrow does.

**The result is 128-bit unless an operand is 256-bit**, whatever the input
widths — decimal32 and decimal64 promote exactly as Arrow C++ promotes them.

**No wide intermediate is needed, and that is the rule's doing.** A result
whose precision would exceed the width's maximum (38, or 76) raises
*before* any row is read, so every rescaled operand and every product fits the
backing integer by construction: `(10, 2) * (10, 2)` is `(21, 4)`, and a
21-digit product cannot overflow an int128. That is why these kernels are the
plain lane operators over the unscaled values, and why nothing here needs a
256-bit multiply or a long division.

**Division truncates toward zero**, as Arrow C++, arrow-rs and ClickHouse all
do; Polars rounds and DuckDB answers `DOUBLE` (both measured 2026-09-27). A
zero divisor is the caller's to handle, as for `DivKernel`: the expression
lanes null that row, as they do for `//`.
"""

from ..arrays import DynArray, PrimitiveArray
from ..buffers import Bitmap, Buffer
from ..dtypes import (
    Decimal128Type,
    Decimal256Type,
    DecimalType,
    DynType,
    bool_,
    decimal128,
    decimal256,
    float64,
)
from ..views import apply
from .core import Kernel
from ..errors import InvalidError, TypeError
from .cast import cast
from .numeric import FloordivKernel
from ..execution import ExecContext


# ---------------------------------------------------------------------------
# DecimalPromotion — what each operand becomes, and what the answer is
# ---------------------------------------------------------------------------


@fieldwise_init
struct DecimalPromotion(Copyable, Movable):
    """Arrow C++'s decimal promotion: what each operand is cast to (`left`,
    `right`), the answer's type (`output`), and how many digits each operand's
    scale grows by (C++'s `left_scaleup` / `right_scaleup`). Types and digit
    counts only; each constructor raises where Arrow's `DecimalType::Make`
    would, which is why the arithmetic cannot overflow.
    """

    var left: DynType
    var right: DynType
    var output: DynType
    var left_up: Int
    var right_up: Int

    @staticmethod
    def accepts(l: DynType, r: DynType) -> Bool:
        """Whether `l op r` is decimal arithmetic: a decimal on at least one
        side, and a decimal or an integer on the other — Arrow C++'s rule,
        where an integer is a decimal of scale 0."""
        if not (l.is_decimal() or r.is_decimal()):
            return False
        return (l.is_decimal() or l.is_integer()) and (
            r.is_decimal() or r.is_integer()
        )

    @staticmethod
    def common(l: DynType, r: DynType) raises -> DynType:
        """The type both sides of a comparison take when one is a decimal:
        `float64` against a float, otherwise `compare`'s common decimal."""
        if l.is_floating_point() or r.is_floating_point():
            return DynType(float64)
        return Self.compare(l, r).left.copy()

    @staticmethod
    def _operand(dt: DynType) raises -> Tuple[Int, Int]:
        """`(precision, scale)` of a decimal or an integer operand — an
        integer as a scale-0 decimal of as many digits as its type holds,
        Arrow C++'s `MaxDecimalDigitsForInteger`."""
        if dt.is_decimal():

            def params[T: DecimalType](d: T) raises {} -> Tuple[Int, Int]:
                return (d.precision(), d.scale())

            return dt.dispatch_decimal(params)
        elif dt.is_int8() or dt.is_uint8():
            return (3, 0)
        elif dt.is_int16() or dt.is_uint16():
            return (5, 0)
        elif dt.is_int32() or dt.is_uint32():
            return (10, 0)
        elif dt.is_int64():
            return (19, 0)
        elif dt.is_uint64():
            return (20, 0)
        raise TypeError(t"decimal: {dt} is neither a decimal nor an integer")

    @staticmethod
    def _wide(l: DynType, r: DynType) -> Bool:
        """Whether the operation runs in 256 bits: exactly when an operand
        is a decimal256."""
        return l.is_decimal256() or r.is_decimal256()

    @staticmethod
    def _make(wide: Bool, precision: Int, scale: Int) raises -> DynType:
        """`decimal128(precision, scale)`, or `decimal256` when `wide`,
        raising when the width cannot hold that many digits."""
        var limit = (
            Decimal256Type.max_precision if wide else Decimal128Type.max_precision
        )
        if precision < 1 or precision > limit:
            raise InvalidError(
                t"decimal precision out of range [1, {limit}]: {precision}"
            )
        if wide:
            return decimal256(precision, scale)
        return decimal128(precision, scale)

    @staticmethod
    def add(l: DynType, r: DynType) raises -> Self:
        """`+` and `-`: both operands meet at the larger scale, and the
        integer part gains one digit for the carry."""
        var p1, s1 = Self._operand(l)
        var p2, s2 = Self._operand(r)
        var wide = Self._wide(l, r)
        var s = max(s1, s2)
        return Self(
            left=Self._make(wide, p1 + s - s1, s),
            right=Self._make(wide, p2 + s - s2, s),
            output=Self._make(wide, max(p1 - s1, p2 - s2) + s + 1, s),
            left_up=s - s1,
            right_up=s - s2,
        )

    @staticmethod
    def multiply(l: DynType, r: DynType) raises -> Self:
        """`*`: no rescale at all — the scales add, and so do the digits."""
        var p1, s1 = Self._operand(l)
        var p2, s2 = Self._operand(r)
        var wide = Self._wide(l, r)
        return Self(
            left=Self._make(wide, p1, s1),
            right=Self._make(wide, p2, s2),
            output=Self._make(wide, p1 + p2 + 1, s1 + s2),
            left_up=0,
            right_up=0,
        )

    @staticmethod
    def divide(l: DynType, r: DynType) raises -> Self:
        """`/`: the dividend is scaled up first, so the truncated integer
        quotient carries `max(4, s1 + p2 - s2 + 1)` fractional digits."""
        var p1, s1 = Self._operand(l)
        var p2, s2 = Self._operand(r)
        var wide = Self._wide(l, r)
        var up = max(4, s1 + p2 - s2 + 1) + s2 - s1
        return Self(
            left=Self._make(wide, p1 + up, s1 + up),
            right=Self._make(wide, p2, s2),
            output=Self._make(wide, p1 + up, s1 + up - s2),
            left_up=up,
            right_up=0,
        )

    @staticmethod
    def compare(l: DynType, r: DynType) raises -> Self:
        """A comparison: both operands at the larger scale in one common type
        (`left` and `right`), answering `bool`. Unlike arithmetic this does
        not raise past 38 digits: Arrow C++ moves the comparison to decimal256
        instead."""
        var p1, s1 = Self._operand(l)
        var p2, s2 = Self._operand(r)
        var s = max(s1, s2)
        var p = max(p1 + s - s1, p2 + s - s2)
        var common = Self._make(
            Self._wide(l, r) or p > Decimal128Type.max_precision, p, s
        )
        return Self(
            left=common.copy(),
            right=common^,
            output=DynType(bool_),
            left_up=s - s1,
            right_up=s - s2,
        )


# ---------------------------------------------------------------------------
# DecimalBinaryKernel — the integer operation, and the erased entry point
# ---------------------------------------------------------------------------


trait DecimalBinaryKernel(Kernel):
    """One decimal operator: its result-type rule and its lane.

    `core` is the operation over the *rescaled* unscaled integers, which is
    all a lane ever sees — the comptime lane calls it directly with the
    promotion's powers of ten applied, and `apply` calls it through `views`.

    Not a `BinaryKernel`, whose `apply` labels the answer with an operand's
    dtype: that is exactly what a decimal operator must not do.
    """

    @staticmethod
    def promotion(l: DynType, r: DynType) raises -> DecimalPromotion:
        """The types this operator runs in, for operands of these types."""
        ...

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        ...

    @staticmethod
    def apply[
        T: DecimalType
    ](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        output: T,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """Operands already cast to the promotion's `left` and `right`; the
        answer is labelled `output`. Null-in, null-out."""
        Self.expect_same_length(len(left), len(right))
        comptime native = T.native
        var length = len(left)
        var bm = Bitmap.intersect_views(left.validity(), right.validity())
        var buf = Buffer.alloc_zeroed[native](length)
        apply[native, native, Self.core[native, _]](
            left.values(), right.values(), buf.view[native](0, length), ctx
        )
        return PrimitiveArray[T](
            dtype=output,
            length=length,
            nulls=bm.value().unset_count() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=buf.to_immutable(),
        )

    @staticmethod
    def dispatch(
        left: DynArray,
        right: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Erased entry: decimal or integer operands, in either position."""
        var promo = Self.promotion(left.dtype(), right.dtype())
        var l = cast(left, promo.left, True, ctx)
        var r = cast(right, promo.right, True, ctx)

        # Two arms, not `dispatch_decimal`'s four: `DecimalPromotion` only
        # ever answers decimal128 or decimal256.
        if promo.output.is_decimal256():
            return Self.apply(
                l.as_decimal256(),
                r.as_decimal256(),
                promo.output.as_decimal256(),
                ctx,
            ).to_dyn()
        return Self.apply(
            l.as_decimal128(),
            r.as_decimal128(),
            promo.output.as_decimal128(),
            ctx,
        ).to_dyn()


struct DecimalAddKernel(DecimalBinaryKernel):
    comptime name = "add"

    @staticmethod
    def promotion(l: DynType, r: DynType) raises -> DecimalPromotion:
        return DecimalPromotion.add(l, r)

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b


struct DecimalSubKernel(DecimalBinaryKernel):
    comptime name = "subtract"

    @staticmethod
    def promotion(l: DynType, r: DynType) raises -> DecimalPromotion:
        return DecimalPromotion.add(l, r)

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a - b


struct DecimalMulKernel(DecimalBinaryKernel):
    comptime name = "multiply"

    @staticmethod
    def promotion(l: DynType, r: DynType) raises -> DecimalPromotion:
        return DecimalPromotion.multiply(l, r)

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a * b


struct DecimalDivKernel(DecimalBinaryKernel):
    comptime name = "divide"

    @staticmethod
    def promotion(l: DynType, r: DynType) raises -> DecimalPromotion:
        return DecimalPromotion.divide(l, r)

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        """The truncating integer quotient, a zero divisor substituted by 1.

        Exactly `FloordivKernel.core`'s integer arm, which is SQL's `//`:
        truncation toward zero is what Arrow's decimal division does. The
        substituted 1 is a value, not an answer, as it is for `DivKernel`: a
        lane can neither raise nor write a null, so the layer above supplies
        the meaning — both expression lanes null a zero divisor first.
        """
        return FloordivKernel.core[T, W](a, b)
