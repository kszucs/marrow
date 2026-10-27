# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The `Kernel` base trait, and the properties a kernel may claim.

Every kernel family in `marrow.kernels` — element-wise arithmetic, comparison,
boolean, string, temporal, aggregate — descends from `Kernel`, so a kernel is
nameable without knowing which family it belongs to. Family traits add the
call shape (`core`/`apply`/`dispatch`, `combine`/`finalize`, ...).

`Kernel` also owns the argument checks every family repeats, so the diagnostic
a caller sees is the same sentence whichever kernel raised it, and a new family
inherits it by conforming rather than by copying the message.

A property trait (`Partial`, `Prunable`) states one law about a kernel's
values. A kernel lists every property it has, a caller's bound lists every
property it needs, and `kernels/tests/test_laws.mojo` proves each law for
every kernel that claims it.
"""

from std.builtin.variadics import TypeList

from ..arrays import BoolArray, DynArray, PrimitiveArray
from ..buffers import Bitmap, Buffer
from ..dtypes import DecimalType, DynType, NumericType, PrimitiveType
from ..errors import ArrowError, InvalidError, TypeError
from ..views import apply


comptime conforms[F: type_of(AnyType), T: AnyType]: Bool = conforms_to(T, F)
"""Whether dtype `T` belongs to family `F`."""

comptime family[F: type_of(AnyType)] = conforms[F, ...]
"""A kernel's `accepts`: every dtype of family `F` —
`comptime accepts = family[FloatingType]`."""

comptime _member[L: TypeList[Trait=AnyType, ...], T: AnyType]: Bool = (
    L.contains[T]()
)

comptime one_of[*Ts: AnyType] = _member[TypeList.of[Trait=AnyType, *Ts](), ...]
"""A kernel's `accepts`: exactly the dtypes `Ts` —
`comptime accepts = one_of[Date32Type, Date64Type, TimestampType]`."""


trait Kernel:
    """Base trait for all compute kernels."""

    comptime name: String
    """This kernel's identity — for display and diagnostics, never dispatch."""

    @staticmethod
    def error[E: ArrowError, M: Writable](message: M) -> E:
        """This kernel's failure, attributed to it: `Self.error[TypeError](...)`
        reads `"TypeError: <name>: <message>"`."""
        return E(t"{Self.name}: {message}")

    @staticmethod
    def expect_same_length(left: Int, right: Int) raises:
        """Raise unless both operands hold the same number of elements."""
        if left != right:
            raise Self.error[InvalidError](
                t"arrays must have the same length, got {left} and {right}"
            )

    @staticmethod
    def expect_same_dtype(left: DynType, right: DynType) raises:
        """Raise unless both operands carry the same dtype."""
        if left != right:
            raise Self.error[TypeError](t"dtype mismatch: {left} vs {right}")

    @staticmethod
    def expect_same_dtype(left: DynArray, right: DynArray) raises:
        """Raise unless both arrays carry the same dtype."""
        if left.dtype() != right.dtype():
            Self.expect_same_dtype(left.dtype(), right.dtype())


trait Partial:
    """A kernel undefined for some right operands: where `defined` is false,
    the kernel's result is undefined. The kernel computes a value there anyway
    (a substituted divisor); whoever presents SQL semantics nulls that row,
    through `null_undefined` over an array or from `defined` in a fused loop.

    `defined` is the rule and its one statement; `null_undefined` applies it
    to a whole array.

    **`defined` must not change under promotion.** The expression lanes
    evaluate `defined` on the divisor's own dtype and `core` on the promoted
    one, so for every widening they perform from `A` to `P`,
    `defined[A](x) == defined[P](x.cast[P]())`. Under the zero rule every
    conformer shares, a non-zero must stay non-zero and a zero must stay a
    zero.

    `DivKernel` does not claim it: float division is total (IEEE 754 answers
    `inf` and `nan`), and an integer division by zero is an error, raised by
    the Python binding.
    """

    @always_inline
    @staticmethod
    def defined[T: DType, W: Int](b: SIMD[T, W]) -> SIMD[DType.bool, W]:
        """Per lane, whether the kernel is defined for the right operand `b`."""
        ...

    @staticmethod
    def null_undefined[
        DivisorType: PrimitiveType
    ](b: PrimitiveArray[DivisorType]) raises -> PrimitiveArray[DivisorType]:
        """`b` with every row where `defined` is false made null: values
        untouched, the existing nulls kept, correct at any offset. Runs on the
        CPU.

        `b` itself when no valid row is undefined. Otherwise a new array at
        offset 0 holding a copy of `b`'s values: one offset addresses both an
        array's values and its validity, and the bits computed here start at
        `b`'s first row.
        """
        comptime native = DivisorType.native
        var length = len(b)
        var bits = Bitmap.alloc_uninit(length)
        apply[native, Self.defined[native, _]](b.values(), bits.view())
        var defined = bits^.to_immutable()
        var undefined = defined.unset_count()
        var validity: Bitmap[mut=False]
        var nulls: Int
        if undefined == 0:
            return b.copy()
        elif b.bitmap:
            validity = _valid_and(b, defined)
            nulls = validity.unset_count()
        else:
            validity = defined^
            nulls = undefined
        if nulls == b.null_count():
            return b.copy()
        else:
            var values = Buffer.alloc_uninit[native](length)
            values.view[native](0, length).copy_from(b.values(), length)
            return PrimitiveArray[DivisorType](
                dtype=b.dtype.copy(),
                length=length,
                nulls=nulls,
                offset=0,
                bitmap=validity,
                buffer=values^.to_immutable(),
            )

    @staticmethod
    def null_undefined(b: DynArray) raises -> DynArray:
        """The erased form, for a numeric or a decimal `b`: the two families
        a partial kernel takes, so no other dtype is instantiated.

        It shares the typed form's name rather than being a `dispatch`: it
        only resolves `b`'s dtype through those two narrow arms, the typed
        form holds the logic, and `dispatch` keeps its one meaning on the
        kernels that claim this property.
        """

        def numeric[T: NumericType](d: T) raises {imm} -> DynArray:
            return Self.null_undefined(b.as_primitive[T]()).to_dyn()

        def decimal[T: DecimalType](d: T) raises {imm} -> DynArray:
            return Self.null_undefined(b.as_primitive[T]()).to_dyn()

        if b.dtype().is_decimal():
            return b.dtype().try_visit[DynArray, DecimalType](
                decimal, "dtype is not decimal"
            )
        else:
            return b.dtype().try_visit[DynArray, NumericType](
                numeric, "dtype is not numeric"
            )


def _valid_and[
    T: PrimitiveType
](b: PrimitiveArray[T], bits: Bitmap[mut=False]) raises -> Bitmap[mut=False]:
    """`b`'s validity AND `bits`, both indexed from `b`'s first row."""
    # A function rather than inline code: `intersect_views` needs views with
    # an immutable origin, which a borrowed argument has and a local does not.
    return Bitmap.intersect_views(b.validity(), Optional(bits.view())).value()


trait Prunable:
    """A comparison that can tell from `[min, max]` bounds whether any pair
    inside them may satisfy it. Sound: never false when some pair does.

    The bounds are the least and greatest value under the total order (NaN
    greatest), and `possible` compares them under that order whatever the
    kernel's own NaN rule: every pair the IEEE comparison accepts, the total
    order accepts too, so one rule is sound for both.

    `possible` is the rule and its one statement; `possible_segments` applies
    it to a zone map, one set of bounds per segment, a consecutive run of
    rows.
    """

    @always_inline
    @staticmethod
    def possible[
        T: DType, W: Int
    ](
        lmin: SIMD[T, W], lmax: SIMD[T, W], rmin: SIMD[T, W], rmax: SIMD[T, W]
    ) -> SIMD[DType.bool, W]:
        """Per lane, whether some `a` in `[lmin, lmax]` and `b` in
        `[rmin, rmax]` may satisfy the comparison: false only where none can.
        """
        ...

    @staticmethod
    def possible_segments[
        BoundType: PrimitiveType
    ](
        lmin: PrimitiveArray[BoundType],
        lmax: PrimitiveArray[BoundType],
        rmin: PrimitiveArray[BoundType],
        rmax: PrimitiveArray[BoundType],
    ) raises -> BoolArray:
        """`possible` per segment; null where any bound is null (an absent
        statistic means read it). Runs on the CPU, one segment at a time:
        there are as many segments as row groups or pages, not rows."""
        comptime native = BoundType.native
        var segments = len(lmin)
        if (
            len(lmax) != segments
            or len(rmin) != segments
            or len(rmax) != segments
        ):
            raise InvalidError(
                t"possible_segments: bounds must have the same length, got"
                t" {segments}, {len(lmax)}, {len(rmin)} and {len(rmax)}"
            )
        # Per row, not `Bitmap.intersect`: the bitmap join links into every
        # AOT binary that prunes, and grows each dtype arm of the erased form.
        var bits = Bitmap.alloc_zeroed(segments)
        var validity = Bitmap.alloc_zeroed(segments)
        var nulls = 0
        for i in range(segments):
            if (
                lmin.is_valid(i)
                and lmax.is_valid(i)
                and rmin.is_valid(i)
                and rmax.is_valid(i)
            ):
                validity.unsafe_set(i)
                if Self.possible[native, 1](
                    lmin.unsafe_get(i),
                    lmax.unsafe_get(i),
                    rmin.unsafe_get(i),
                    rmax.unsafe_get(i),
                )[0]:
                    bits.unsafe_set(i)
            else:
                nulls += 1
        var bitmap: Optional[Bitmap[mut=False]] = None
        if nulls > 0:
            bitmap = validity^.to_immutable()
        return BoolArray(
            length=segments,
            nulls=nulls,
            offset=0,
            bitmap=bitmap^,
            buffer=bits^.to_immutable(),
        )

    @staticmethod
    def possible_segments(
        lmin: DynArray, lmax: DynArray, rmin: DynArray, rmax: DynArray
    ) raises -> BoolArray:
        """The erased form, over every `PrimitiveType`, so temporal and
        decimal bounds prune in the runtime lane as numeric ones do. The four
        bounds must share one dtype.

        It shares the typed form's name rather than being a `dispatch`, as
        `Partial.null_undefined` does: it only resolves the bounds' dtype, and
        the typed form holds the logic.
        """
        var dtype = lmin.dtype()
        if (
            lmax.dtype() != dtype
            or rmin.dtype() != dtype
            or rmax.dtype() != dtype
        ):
            raise TypeError(
                t"possible_segments: bounds must share one dtype, got {dtype},"
                t" {lmax.dtype()}, {rmin.dtype()} and {rmax.dtype()}"
            )

        def leaf[T: PrimitiveType](d: T) raises {imm} -> BoolArray:
            return Self.possible_segments(
                lmin.as_primitive[T](),
                lmax.as_primitive[T](),
                rmin.as_primitive[T](),
                rmax.as_primitive[T](),
            )

        return dtype.try_visit[BoolArray, PrimitiveType](
            leaf, "dtype is not primitive"
        )


# TODO: have vectorwise and elementwise kernels conform to a common trait
