# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Fused decimal operators: arithmetic, comparison and casts over `DecimalValue`.

A decimal lane is the unscaled integer, so every node here is a numeric node
**plus a power of ten per operand**. The result type is Arrow C++'s
(`DecimalPromotion` in `kernels/decimal.mojo`), and it depends on the
operands' precision and scale — runtime values on the dtype instance — so each
node resolves its promotion in `bind`, once per batch, and carries the power
of ten each operand is scaled by in its `Bound`. The lane then multiplies and
operates: `col("a", decimal128(10, 2)) + col("b", decimal128(5, 1))` is one
loop computing `a + b * 10`, with no rescaled intermediate column.

**The output width is comptime and the precision is not.** `Type` is
`WideDecimalType` of the wider operand — decimal128 unless an operand is
decimal256 — the same width `DecimalPromotion` picks at run time. The two are
one rule written twice, once per phase, and the tests pin both. Precision and
scale are only known from the schema, which is why every node here overrides
`dtype`.

**A comparison is the one place the widths can part.** Arrow C++ moves a
comparison to decimal256 when aligning the scales needs more than 38 digits
(`decimal128(38, 0)` against `decimal128(38, 10)`); a comptime lane cannot
change its register width per batch, so `DecimalCompare` raises there instead,
naming the types. The runtime lane follows Arrow.

**Division by zero is NULL**, as `//` is in both lanes and as DuckDB does —
not an error, which is what `pc.divide` answers. The expression layer has one
rule for a zero divisor and decimals follow it.

**The casts fuse where a lane can express them.** Rescaling is a multiply or a
truncating divide, and decimal -> float a divide in `float64`, so those three
stay in the loop. Text crosses the fixed-width boundary and breaks, as
`NumToString` does. Like every cast in this lane they are unchecked:
`safe=True` is refused, and the runtime lane is where a checked cast lives.
"""

from ...arrays import BinaryLikeArray, PrimitiveArray, StructArray
from ...buffers import Bitmap
from ...views import apply
from ...dtypes import (
    BoolType,
    DecimalType,
    DynType,
    NumericType,
    StringLikeType,
    WideDecimalType,
)
from ...kernels.cast import DecimalToStringKernel, StringToDecimalKernel
from ...kernels.decimal import (
    DecimalAddKernel,
    DecimalBinaryKernel,
    DecimalDivKernel,
    DecimalMulKernel,
    DecimalPromotion,
    DecimalSubKernel,
)
from ...kernels.numeric import (
    EqKernel,
    FloordivKernel,
    GeKernel,
    GtKernel,
    LeKernel,
    LtKernel,
    NeKernel,
    NumericCompareKernel,
)
from ...errors import NotImplementedError
from ...schema import Schema
from ..logical import Shape
from ..logical import Bindings
from .rules import wider, widest_shape
from .core import (
    BoolValue,
    ColumnBound,
    DecimalValue,
    NumericValue,
    StringValue,
    Unnamed,
)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
struct DecimalBinary[K: DecimalBinaryKernel, L: DecimalValue, R: DecimalValue](
    DecimalValue, Unnamed
):
    """`+`, `-` and `*` over two decimal operands, fused.

    `K` supplies both halves: `K.promotion` the result type and how far each
    operand is scaled up, `K.core` the integer operation over the rescaled
    lanes. When the promotion scales neither operand — always for `*`, and
    for `+`/`-` over equal scales — the lane skips the two multiplies.
    """

    comptime Type = WideDecimalType[
        wider[Self.L.Type.native, Self.R.Type.native]
    ]
    comptime N = Self.Type.native
    """The lane's integer — named once, since a chained projection does not
    reduce at a call site (see `NumericCompare.ArgType`)."""

    comptime shape = widest_shape[Self.L, Self.R]
    comptime Bound = Tuple[
        Self.L.Bound, Self.R.Bound, Scalar[Self.N], Scalar[Self.N], Bool
    ]
    """The operands' bounds, the power of ten each is rescaled by, and
    whether either factor is not 1 — constant for the batch, so the lane's
    branch on it always predicts."""

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def dtype(self, schema: Schema) raises -> DynType:
        return Self.K.promotion(
            self.l.dtype(schema), self.r.dtype(schema)
        ).output.copy()

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var schema = Schema.from_dtype(batch.dtype)
        var promo = Self.K.promotion(self.l.dtype(schema), self.r.dtype(schema))
        return (
            self.l.bind(batch, bindings),
            self.r.bind(batch, bindings),
            pow(Scalar[Self.N](10), promo.left_up),
            pow(Scalar[Self.N](10), promo.right_up),
            promo.left_up != 0 or promo.right_up != 0,
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return Bitmap.intersect(
            self.l.validity(bound[0]), self.r.validity(bound[1])
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[Self.N, W]:
        var a = self.l.lane[W](bound[0], idx).cast[Self.N]()
        var b = self.r.lane[W](bound[1], idx).cast[Self.N]()
        if bound[4]:
            return Self.K.core[Self.N, W](a * bound[2], b * bound[3])
        return Self.K.core[Self.N, W](a, b)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime DecimalAdd = DecimalBinary[DecimalAddKernel, _, _]
comptime DecimalSub = DecimalBinary[DecimalSubKernel, _, _]
comptime DecimalMul = DecimalBinary[DecimalMulKernel, _, _]


struct DecimalDiv[L: DecimalValue, R: DecimalValue](DecimalValue, Unnamed):
    """`/` over two decimal operands: the dividend scaled up, a truncating
    quotient, and NULL where the divisor is zero.

    Its own node rather than a `DecimalBinary` for the reason `DivisionBinary`
    is not a `NumericBinary`: the zero-divisor bits are a third slot in the
    `Bound`, which every `+` would otherwise carry too.
    """

    comptime Type = WideDecimalType[
        wider[Self.L.Type.native, Self.R.Type.native]
    ]
    comptime N = Self.Type.native
    comptime DivisorType = Self.R.Type.native
    """The divisor's lane type, which the zero scan reads in."""

    comptime shape = Shape.columnar
    """Columnar regardless of the operands, as `DivisionBinary` — a scalar
    shape has no validity to carry the zero divisor's NULL."""

    comptime Bound = Tuple[
        Self.L.Bound, Self.R.Bound, Scalar[Self.N], Optional[Bitmap[mut=False]]
    ]
    """The operands' bounds, the dividend's factor, and the rows whose
    divisor is non-zero — `None` when none is zero."""

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def dtype(self, schema: Schema) raises -> DynType:
        return DecimalDivKernel.promotion(
            self.l.dtype(schema), self.r.dtype(schema)
        ).output.copy()

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var schema = Schema.from_dtype(batch.dtype)
        var promo = DecimalDivKernel.promotion(
            self.l.dtype(schema), self.r.dtype(schema)
        )
        var lb = self.l.bind(batch, bindings)
        var rb = self.r.bind(batch, bindings)
        var length = len(batch)
        var mask = Bitmap.alloc_uninit(length)

        @always_inline
        def nonzero[W: Int](i: Int) {imm} -> SIMD[DType.bool, W]:
            return self.r.lane[W](rb, i).ne(0)

        apply[Self.DivisorType](mask.view(), nonzero)
        var bits = mask^.to_immutable()
        var nonzero_rows = Optional[Bitmap[mut=False]](None)
        if bits.unset_count() > 0:
            nonzero_rows = Optional(bits^)
        return (lb^, rb^, pow(Scalar[Self.N](10), promo.left_up), nonzero_rows^)

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return Bitmap.intersect(
            Bitmap.intersect(
                self.l.validity(bound[0]), self.r.validity(bound[1])
            ),
            bound[3].copy(),
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[Self.N, W]:
        return DecimalDivKernel.core[Self.N, W](
            self.l.lane[W](bound[0], idx).cast[Self.N]() * bound[2],
            self.r.lane[W](bound[1], idx).cast[Self.N](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(DecimalDivKernel.name, "(", self.l, ", ", self.r, ")")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
struct DecimalCompare[
    K: NumericCompareKernel, L: DecimalValue, R: DecimalValue
](BoolValue, Unnamed):
    """A comparison over two decimal operands, at their common scale.

    `1.5` as `decimal(5, 1)` and `1.50` as `decimal(10, 2)` are equal: both
    are brought to the larger scale before the integers are compared, which
    is `DecimalPromotion.compare`. Different scales are the ordinary case
    here, not an error — unlike `TemporalCompare`, whose units have no rule.

    Statistics pruning is not implemented: `mask` keeps every chunk, the
    default. The runtime lane prunes decimal predicates.
    """

    comptime ArgType = WideDecimalType[
        wider[Self.L.Type.native, Self.R.Type.native]
    ]
    comptime NativeType = Self.ArgType.native
    comptime shape = widest_shape[Self.L, Self.R]
    comptime Bound = Tuple[
        Self.L.Bound,
        Self.R.Bound,
        Scalar[Self.NativeType],
        Scalar[Self.NativeType],
    ]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    def _promotion(self, schema: Schema) raises -> DecimalPromotion:
        """The common type, refused when it is wider than this lane — see the
        module docstring."""
        var ldt = self.l.dtype(schema)
        var rdt = self.r.dtype(schema)
        var promo = DecimalPromotion.compare(ldt, rdt)
        if promo.left.is_decimal256() and Self.NativeType != DType.int256:
            raise NotImplementedError(
                t"decimal comparison of {ldt} and {rdt} needs {promo.left},"
                t" wider than this lane's register; compare them in the"
                t" runtime lane"
            )
        return promo^

    # -- Value --------------------------------------------------------------

    def dtype(self, schema: Schema) raises -> DynType:
        """`bool` — after checking the comparison fits the lane, so a query
        that cannot run fails when it is planned, as the arithmetic nodes'
        precision errors do."""
        _ = self._promotion(schema)
        return DynType(BoolType())

    # -- BoolValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var promo = self._promotion(Schema.from_dtype(batch.dtype))
        return (
            self.l.bind(batch, bindings),
            self.r.bind(batch, bindings),
            pow(Scalar[Self.NativeType](10), promo.left_up),
            pow(Scalar[Self.NativeType](10), promo.right_up),
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return Bitmap.intersect(
            self.l.validity(bound[0]), self.r.validity(bound[1])
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return Self.K.core[Self.NativeType, W](
            self.l.lane[W](bound[0], idx).cast[Self.NativeType]() * bound[2],
            self.r.lane[W](bound[1], idx).cast[Self.NativeType]() * bound[3],
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime DecimalEq = DecimalCompare[EqKernel[nan_safe=False], _, _]
comptime DecimalNe = DecimalCompare[NeKernel[nan_safe=False], _, _]
comptime DecimalLt = DecimalCompare[LtKernel[nan_safe=False], _, _]
comptime DecimalLe = DecimalCompare[LeKernel[nan_safe=False], _, _]
comptime DecimalGt = DecimalCompare[GtKernel[nan_safe=False], _, _]
comptime DecimalGe = DecimalCompare[GeKernel[nan_safe=False], _, _]


# ---------------------------------------------------------------------------
# Casts that fuse
# ---------------------------------------------------------------------------
struct DecimalRescale[To: DecimalType, A: DecimalValue](DecimalValue, Unnamed):
    """Decimal -> decimal: a rescale by `10^(to_scale - from_scale)` — a
    multiply up, or a divide down that truncates toward zero, as Arrow's
    unchecked cast does. Computed in the wider of the two integers, then
    narrowed, so a decimal128 rescaled into a decimal32 divides before it
    wraps. The fused counterpart of `DecimalRescaleKernel`."""

    comptime Type = Self.To
    comptime From = Self.A.Type.native
    comptime M = wider[Self.From, Self.To.native]
    """The integer the rescale runs in."""

    comptime shape = Self.A.shape
    comptime Bound = Tuple[Self.A.Bound, Scalar[Self.M], Bool]
    """The operand's bound, `10^|delta|`, and whether the scale grows."""

    var a: Self.A
    var _dtype: Self.To

    def __init__(out self, var a: Self.A, dtype: Self.To):
        self.a = a^
        self._dtype = dtype

    # -- Value --------------------------------------------------------------

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var source = self.a.dtype(Schema.from_dtype(batch.dtype))
        var delta = self._dtype.scale() - source.as_type[Self.A.Type]().scale()
        return (
            self.a.bind(batch, bindings),
            pow(Scalar[Self.M](10), abs(delta)),
            delta >= 0,
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(bound[0])

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        var v = self.a.lane[W](bound[0], idx).cast[Self.M]()
        var f = SIMD[Self.M, W](bound[1])
        var out = v * f if bound[2] else FloordivKernel.core[Self.M, W](v, f)
        return out.cast[Self.Type.native]()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.a, ", ", self._dtype, ")")


struct NumToDecimal[To: DecimalType, A: NumericValue](DecimalValue, Unnamed):
    """Integer or float -> decimal: `x * 10^scale`. A number is scale 0, so
    the scale only grows. An integer multiplies in the wider of the two
    integers; a float multiplies in `float64` and rounds, as
    `FloatToDecimalKernel` does. The mirror of `DecimalToNum`."""

    comptime Type = Self.To
    comptime From = Self.A.Type.native
    comptime F = DType.float64 if Self.From.is_floating_point() else wider[
        Self.From, Self.To.native
    ]
    """The factor's type: `float64` for a float operand, the wider integer
    otherwise."""

    comptime shape = Self.A.shape
    comptime Bound = Tuple[Self.A.Bound, Scalar[Self.F]]

    var a: Self.A
    var _dtype: Self.To

    def __init__(out self, var a: Self.A, dtype: Self.To):
        self.a = a^
        self._dtype = dtype

    # -- Value --------------------------------------------------------------

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return (
            self.a.bind(batch, bindings),
            pow(Scalar[Self.F](10), self._dtype.scale()),
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(bound[0])

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        var x = self.a.lane[W](bound[0], idx).cast[Self.F]()
        var scaled = x * SIMD[Self.F, W](bound[1])
        comptime if Self.From.is_floating_point():
            return scaled.__round__().cast[Self.Type.native]()
        else:
            return scaled.cast[Self.Type.native]()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.a, ", ", self._dtype, ")")


struct DecimalToNum[To: NumericType, A: DecimalValue](NumericValue, Unnamed):
    """Decimal -> numeric: `x / 10^scale`, in `float64` for a float target and
    truncating toward zero for an integer one."""

    comptime Type = Self.To
    comptime From = Self.A.Type.native
    comptime F = DType.float64 if Self.To.native.is_floating_point() else Self.From
    """The divisor's type: `float64` for a float target, the decimal's integer
    otherwise."""

    comptime shape = Self.A.shape
    comptime Bound = Tuple[Self.A.Bound, Scalar[Self.F]]

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var source = self.a.dtype(Schema.from_dtype(batch.dtype))
        var scale = source.as_type[Self.A.Type]().scale()
        return (self.a.bind(batch, bindings), pow(Scalar[Self.F](10), scale))

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(bound[0])

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        var x = self.a.lane[W](bound[0], idx).cast[Self.From]()
        comptime if Self.Type.native.is_floating_point():
            var f = bound[1].cast[DType.float64]()
            return (x.cast[DType.float64]() / f).cast[Self.Type.native]()
        else:
            return FloordivKernel.core[Self.From, W](
                x, SIMD[Self.From, W](bound[1].cast[Self.From]())
            ).cast[Self.Type.native]()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.a, ", ", Self.Type(), ")")


# ---------------------------------------------------------------------------
# Casts that break
# ---------------------------------------------------------------------------
struct DecimalToString[To: StringLikeType, A: DecimalValue](
    ColumnBound, StringValue, Unnamed
):
    """Format decimal -> text, every digit the scale declares kept — `150` at
    scale 2 is `1.50`."""

    comptime Type = Self.To
    comptime From = Self.A.Type
    comptime shape = Shape.columnar
    comptime Bound = BinaryLikeArray[Self.To]

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- StringValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var arr = self.a.evaluate(batch, bindings).to_array(len(batch))
        return DecimalToStringKernel.apply[Self.From, Self.To](
            arr.as_type[PrimitiveArray[Self.From]]()
        )

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        # As `NumToString.lane`: `unsafe_get` because `lane` cannot raise.
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.a, ", ", Self.Type(), ")")


struct StringToDecimal[To: DecimalType, A: StringValue](
    ColumnBound, DecimalValue, Unnamed
):
    """Parse text -> decimal, nulling what will not parse or fit — the
    unchecked `StringToDecimalKernel`, as `StringToNum` is unchecked."""

    comptime Type = Self.To
    comptime From = Self.A.Type
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.To]

    var a: Self.A
    var _dtype: Self.To

    def __init__(out self, var a: Self.A, dtype: Self.To):
        self.a = a^
        self._dtype = dtype

    # -- Value --------------------------------------------------------------

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var arr = self.a.evaluate(batch, bindings).to_array(len(batch))
        return StringToDecimalKernel.apply[Self.From, Self.To](
            arr.as_type[BinaryLikeArray[Self.From]](), self._dtype, False
        )

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.a, ", ", self._dtype, ")")
