# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The `Kernel` base trait, and the properties a kernel may claim.

Every kernel family in `marrow.kernels` — element-wise arithmetic, comparison,
boolean, string, temporal, aggregate — descends from `Kernel`, so a kernel is
nameable without knowing which family it belongs to. `UnaryKernel` and
`BinaryKernel` add the register function `core` that fused loops and folds call,
with `Out`, the dtype it answers, and `BinaryKernel` the `fold_lanes` a fold
reduces one register with; family traits add the call shape over arrays
(`apply`/`dispatch`, `combine`/`finalize`, ...).

`Kernel` also owns the argument checks every family repeats, so the diagnostic
a caller sees is the same sentence whichever kernel raised it, and a new family
inherits it by conforming rather than by copying the message.

`ScanKernel` is a stream stage that carries state from register to register,
and `PackKernel` / `UnpackKernel` turn values into a typed `Packed` and back
(`compose.mojo`, `packing.mojo`). Each keeps what one pass needs in a `State`
value its caller owns, so the kernel itself holds only its parameters.

A property trait (`Partial`, `Prunable`, `Identity`, `Associative`,
`Commutative`) states one law about a kernel's values. A kernel lists every
property it has, a caller's bound lists every property it needs, and
`kernels/tests/test_laws.mojo` proves each law for every kernel that claims it.
"""

from std.builtin.variadics import TypeList

from ..arrays import BoolArray, DynArray, PrimitiveArray
from ..buffers import Bitmap, Buffer
from ..dtypes import DecimalType, DynType, NumericType, PrimitiveType
from ..errors import ArrowError, InvalidError, TypeError
from ..views import BufferView, apply
from .packing import BLOCK


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


trait UnaryKernel(Kernel):
    """A kernel that maps a register of lanes to a register of lanes, one
    operand in: `core`, the function a fused loop calls directly and every
    array entry point applies lane by lane.

    `Out[T]` is the dtype `core` answers for operands of `T`, and
    `same_core` is `core` for a kernel that keeps the dtype, typed as
    answering `T`.
    """

    comptime Out[T: DType]: DType = T
    """The dtype `core` answers for operands of `T`; `T` unless the kernel
    says otherwise."""

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        """Per lane, the kernel's result for `a`."""
        ...

    @always_inline
    @staticmethod
    def same_core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        """`core`, typed as answering `T`: for a kernel whose `Out[T]` is
        `T`, and refused at compile time for any other."""
        comptime assert Self.Out[T] == T, "the kernel keeps the dtype"
        return rebind[SIMD[T, W]](Self.core[T, W](a))


trait BinaryKernel(Kernel):
    """A kernel that maps two registers of lanes to a register of lanes, two
    operands in: `core`, the function a fused loop or a fold calls directly and
    every array entry point applies lane by lane.

    `Out[T]` is the dtype `core` answers for operands of `T`, and
    `same_core` is `core` typed as answering `T`, as on `UnaryKernel`.
    """

    comptime Out[T: DType]: DType = T
    """The dtype `core` answers for operands of `T`; `T` unless the kernel
    says otherwise."""

    @always_inline
    @staticmethod
    def core[
        T: DType, W: Int
    ](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        """Per lane, the kernel's result for `a` and `b`."""
        ...

    @always_inline
    @staticmethod
    def same_core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        """`core`, typed as answering `T`: for a kernel whose `Out[T]` is
        `T`, and refused at compile time for any other."""
        comptime assert Self.Out[T] == T, "the kernel keeps the dtype"
        return rebind[SIMD[T, W]](Self.core[T, W](a, b))

    @always_inline
    @staticmethod
    def fold_lanes[T: DType, W: Int](v: SIMD[T, W]) -> Scalar[T]:
        """One register folded across its lanes: the left fold
        `core(... core(core(v[0], v[1]), v[2]) ..., v[W - 1])`.

        A kernel that claims `Associative` and `Commutative` may answer it in
        any order, and only a fold, which needs both, calls it. The default
        is the left fold itself, for a kernel that keeps the dtype.
        """
        var acc = v[0]
        comptime for k in range(1, W):
            acc = Self.same_core[T, 1](acc, v[k])
        return acc


trait Identity:
    """A binary kernel with a neutral value: `identity[T]()` is an `e` for which
    `core(e, x)` and `core(x, e)` are both `x`, for every `x` of dtype `T`.
    Equal as values: a NaN equals a NaN, and `-0.0` equals `0.0`.

    A fold seeds its accumulators with `e` (every lane, every partial result,
    every group), so a segment or group no value reached holds `e`; what an
    aggregate answers for an empty group is its own rule (`empty_is_null`).
    That is why `e` must leave every value unchanged, a NaN included.
    """

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        """The neutral value of dtype `T`."""
        ...


trait Associative:
    """A binary kernel whose grouping does not matter:
    `core(core(a, b), c) == core(a, core(b, c))` for every `a`, `b` and `c`. A
    fold may regroup its operands, lanes and partial results alike.

    Equal as values: a NaN equals a NaN, and `-0.0` equals `0.0`.

    **Float `+` and `*` are declared, not proved.** They are associative over
    the real numbers and only approximately so in floating point, where a
    regrouped sum can round differently. SQL engines treat them as
    associative, and so do marrow's SIMD folds. `kernels/tests/test_laws.mojo`
    checks them where the operation is exact.
    """

    pass


trait Commutative:
    """A binary kernel whose operand order does not matter:
    `core(a, b) == core(b, a)` for every `a` and `b`. A fold may reorder its
    operands.

    Equal as values: a NaN equals a NaN, and `-0.0` equals `0.0`.
    """

    pass


trait ScanKernel(Deinitable, Kernel, Movable):
    """A stream stage that carries state through an in-order pass: a running
    fold, a delay line, a fixed operand.

    The instance holds the stage's parameters, and a `State[T]` value holds
    what one pass has seen; the caller owns the state and passes it to every
    call, so one stage serves any number of passes, and a pass may stop and
    go on. `core` answers `Out[T]` for a register of `T`.
    """

    comptime Out[T: DType]: DType = T
    """The dtype `core` answers for values of `T`."""

    comptime State[T: DType]: Movable & Deinitable
    """What a pass over values of `T` carries from one register to the
    next."""

    def start[T: DType](self) -> Self.State[T]:
        """The state before the first row."""
        ...

    def reset[T: DType](self, mut state: Self.State[T]):
        """Back to the state before the first row, at a segment start."""
        state = self.start[T]()

    @always_inline
    def core[
        T: DType, W: Int
    ](self, v: SIMD[T, W], mut state: Self.State[T]) -> SIMD[Self.Out[T], W]:
        """The stage's answer for the register `v`, the rows after the ones
        `state` has seen; `state` then includes `v`."""
        ...


trait PackKernel(Deinitable, Kernel, Movable):
    """Values in, a `Packed` out: the bytes, and whatever a decoder needs
    besides. A column is packed a block at a time: `start` once, `store`
    each block in order, then `finish`.

    `Packed` may hold another packer's `Packed`: that is how a packer that
    keeps more than its values carries it.
    """

    comptime Packed: Copyable & Deinitable
    """What a column is packed as."""

    comptime State[T: DType]: Movable & Deinitable
    """What packing values of `T` keeps between blocks."""

    def start[T: DType](self, count: Int) raises -> Self.State[T]:
        """Before `count` values of `T` are stored; refuses parameters these
        values cannot be packed by."""
        ...

    def store[
        T: DType
    ](self, mut state: Self.State[T], block: BufferView[T, _]) raises:
        """The next block of values: `BLOCK` of them (`packing.mojo`), or
        fewer in the last block only. A packer may keep a block of scratch
        and start a segment at each block's first value; one that does
        refuses any other block with an `InvalidError`."""
        ...

    def finish[T: DType](self, var state: Self.State[T]) raises -> Self.Packed:
        """The column's `Packed`, once every block is stored."""
        ...

    # Out of line: its message is built in no packer's `store`.
    @no_inline
    @staticmethod
    def block_error(count: Int) -> InvalidError:
        """What `store` raises for a block of `count` values it does not
        take."""
        return Self.error[InvalidError](
            t"a block of {count} values, where every block but the last"
            t" holds {BLOCK}"
        )


trait UnpackKernel(Deinitable, Kernel, Movable):
    """A `Packed` in, its values out, a register at a time and in order.

    `start` refuses a `Packed` whose parts do not fit together. `load`
    never raises: what it finds wrong it notes in the state, and `check`
    raises it once the column is read.
    """

    comptime Packed: Copyable & Deinitable
    """What the values were packed as."""

    comptime State[T: DType]: Movable & Deinitable
    """What reading values of `T` keeps between registers."""

    def start[
        T: DType
    ](self, var packed: Self.Packed, count: Int) raises -> Self.State[T]:
        """Before the `count` values `packed` holds are read as `T`."""
        ...

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: Self.State[T], i: Int) -> SIMD[T, W]:
        """The `W` values from value `i`, the registers read in order."""
        ...

    def check[T: DType](self, state: Self.State[T]) raises:
        """Raise what `load` noted wrong; by default, nothing."""
        pass
