"""Fused numeric operators: nodes built from other numeric nodes.

Each is generic over its operands' *types*, so the whole subtree is one type
and the driver inlines it into one loop. `Add[Column[Int64Type],
Literal[Int64Type]]` names its kernel, its operand types and its output type at
compile time; nothing is looked up when it runs.

The kernel is a parameter rather than a field for the same reason `_eval` is a
pointer in the runtime lane: routing on a name would put every arithmetic
kernel in every binary that builds any expression.
"""

from ...dtypes import DataType, DynType, Float64Type, NumericType
from ...kernels.numeric import (
    AbsKernel,
    AddKernel,
    BinaryKernel,
    CeilKernel,
    DivKernel,
    EqKernel,
    ExpKernel,
    FloorKernel,
    FloordivKernel,
    GeKernel,
    GtKernel,
    LeKernel,
    LogKernel,
    LtKernel,
    ModKernel,
    NeKernel,
    NegKernel,
    NumericCompareKernel,
    BinaryNumericKernel,
    MulKernel,
    PowKernel,
    RoundKernel,
    SignKernel,
    SqrtKernel,
    SubKernel,
    TruncKernel,
    UnaryKernel,
    UnaryNumericKernel,
)
from ...arrays import StructArray, BoolArray, DynArray, PrimitiveArray
from ...kernels.conditional import (
    BinaryConditionalKernel,
    CoalesceKernel,
    FillNullKernel,
    NullifKernel,
    case_when,
)
from ...schema import Schema
from ...tabular import RecordBatch
from ...buffers import Bitmap
from ..logical import Shape, merged
from ..params import Bindings
from ...kernels.bounds import (
    EqBounds,
    GeBounds,
    GtBounds,
    LeBounds,
    LtBounds,
    NeBounds,
)
from ..pruning import PruneStats, Truth
from ..physical import Datum

from .rules import promote, wider, widest_shape
from .core import (
    BoolValue,
    ColumnBound,
    NumericValue,
    TemporalValue,
    Unnamed,
)


struct NumericBinary[K: BinaryNumericKernel, L: NumericValue, R: NumericValue](
    NumericValue, Unnamed
):
    """A binary arithmetic node over two fused operands."""

    comptime Type = promote[Self.L.Type, Self.R.Type]
    """The wider operand wins: `Add(int32, int64)` is `int64`."""

    comptime shape = widest_shape[Self.L, Self.R]

    comptime Bound = Tuple[Self.L.Bound, Self.R.Bound]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(self.l.columns(), self.r.columns())

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return (self.l.bind(batch, bindings), self.r.bind(batch, bindings))

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """Arithmetic is null-in, null-out: the result is valid where both
        operands are. A missing bitmap means "cannot be null", so an absent
        side contributes nothing to the intersection."""
        return Bitmap.intersect(
            self.l.validity(bound[0]), self.r.validity(bound[1])
        )

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return Self.K.core[Self.Type.native, W](
            self.l.lane[W](bound[0], idx).cast[Self.Type.native](),
            self.r.lane[W](bound[1], idx).cast[Self.Type.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


# ---------------------------------------------------------------------------
# Named shapes
# ---------------------------------------------------------------------------
# Partial specialisations, so a call site writes `Add[L, R]` rather than
# restating the kernel. The kernel is still a comptime parameter — these are
# names for a binding, not a registry, and an unused one costs nothing.
comptime Add = NumericBinary[AddKernel, _, _]
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]
comptime Mod = NumericBinary[ModKernel, _, _]
comptime Floordiv = NumericBinary[FloordivKernel, _, _]
"""`%` and `//` — **Python's**, not SQL's, and coherently so.

`ModKernel` takes the sign of the divisor, so `-1 % 3` is 2 and
`a == (a // b) * b + a % b` holds. SQL, PyArrow and arrow-rs all truncate
toward zero and answer -1. That divergence is deliberate and recorded in
`golden/cases/math_mod_int64.mojo`, whose SQL twin spells floored modulo as
`((n % 3) + 3) % 3` rather than asserting SQL's convention against marrow's.
"""


# ---------------------------------------------------------------------------
# NumericUnary — one operand, the operand's type
# ---------------------------------------------------------------------------
struct NumericUnary[K: UnaryNumericKernel, A: NumericValue](
    NumericValue, Unnamed
):
    """A unary arithmetic node that keeps its operand's dtype — `neg`, `abs`,
    `sign`, `floor`, `ceil`, `round`, `trunc`.

    The dtype is preserved rather than promoted because every one of these is
    closed over its input: `abs(int32)` is an `int32` and `floor(float64)` is a
    `float64` — Arrow C++'s rule for the same functions, and PyArrow's. The
    two exceptions are `sqrt`/`exp`/`ln`, which are not closed over the
    integers and therefore live on `FloatUnary` below.

    `validity` forwards rather than intersecting: with one operand there is
    nothing to intersect, and none of these kernels manufactures or removes a
    null.
    """

    comptime Type = Self.A.Type
    comptime shape = Self.A.shape
    comptime Bound = Self.A.Bound

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self.a.bind(batch, bindings)

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(bound)

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        # The cast is a no-op — `Self.Type` *is* `Self.A.Type` — but it has to
        # be spelled: a chained projection is not canonicalised, so the
        # compiler reports `SIMD[A.Type.native, W]` "cannot be converted" to
        # `SIMD[NumericUnary[K, A].Type.native, W]`, which is the same type.
        # `TemporalCompare.lane` spells two casts for exactly this reason.
        return Self.K.core[Self.Type.native, W](
            self.a.lane[W](bound, idx).cast[Self.Type.native]()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ")")


comptime Neg = NumericUnary[NegKernel, _]
comptime Abs = NumericUnary[AbsKernel, _]
comptime Sign = NumericUnary[SignKernel, _]
comptime Floor = NumericUnary[FloorKernel, _]
comptime Ceil = NumericUnary[CeilKernel, _]
comptime Round = NumericUnary[RoundKernel, _]
comptime Trunc = NumericUnary[TruncKernel, _]


# ---------------------------------------------------------------------------
# FloatBinary / FloatUnary — the nodes whose output is float64 whatever went in
# ---------------------------------------------------------------------------
struct FloatBinary[K: BinaryKernel, L: NumericValue, R: NumericValue](
    NumericValue, Unnamed
):
    """`/` and `**` — binary operators whose result is always `float64`.

    Separate from `NumericBinary` because `promote[L, R]` is the wrong rule
    here: `5 / 2` is 2.5, not 2, so the output type does not follow from the
    operands at all. Both operands are cast up to `float64` *before* the
    kernel, which is what makes integer true division exact rather than an
    integer division silently widened afterwards.

    `K` is bound on `BinaryKernel`, not on `BinaryNumericKernel` or
    `BinaryFloatKernel`: `DivKernel` is the former and `PowKernel` the latter,
    and the two have no common sub-trait. That is also why `BinaryKernel`
    declares `dispatch` — see its docstring.

    **PyArrow diverges here and marrow does not follow it.** `pc.divide` on two
    integer arrays returns an integer, so `-1 / 3` is 0; marrow answers -0.333.
    The rule is Python's, and it is the same rule that makes `Floordiv` and
    `Mod` above agree with `//` and `%`.
    """

    comptime Type = Float64Type
    comptime shape = widest_shape[Self.L, Self.R]
    comptime Bound = Tuple[Self.L.Bound, Self.R.Bound]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(self.l.columns(), self.r.columns())

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return (self.l.bind(batch, bindings), self.r.bind(batch, bindings))

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """Null-in, null-out — as `NumericBinary`. Division by zero is *not* a
        null: `DivKernel` substitutes 1 for a zero divisor to dodge SIGFPE and
        the float result is an infinity, which is a value."""
        return Bitmap.intersect(
            self.l.validity(bound[0]), self.r.validity(bound[1])
        )

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return Self.K.core[Self.Type.native, W](
            self.l.lane[W](bound[0], idx).cast[Self.Type.native](),
            self.r.lane[W](bound[1], idx).cast[Self.Type.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime Div = FloatBinary[DivKernel, _, _]
comptime Pow = FloatBinary[PowKernel, _, _]


struct FloatUnary[K: UnaryKernel, A: NumericValue](NumericValue, Unnamed):
    """`sqrt`, `exp`, `ln` — unary operators whose result is always `float64`.

    The operand is cast up before the kernel for the same reason
    `FloatBinary`'s are: `sqrt` of an integer is not an integer, so preserving
    the operand's dtype the way `NumericUnary` does would truncate the answer
    rather than widen it.

    `K` is bound on `UnaryKernel` rather than `UnaryFloatKernel` so the same
    node can carry a kernel from either family, matching `FloatBinary`.
    """

    comptime Type = Float64Type
    comptime shape = Self.A.shape
    comptime Bound = Self.A.Bound

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self.a.bind(batch, bindings)

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """Forwarded. `sqrt(-1)` is NaN, not null — a domain error is a value
        in IEEE 754, and `is_nan` is how a caller asks about it."""
        return self.a.validity(bound)

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return Self.K.core[Self.Type.native, W](
            self.a.lane[W](bound, idx).cast[Self.Type.native]()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ")")


comptime Sqrt = FloatUnary[SqrtKernel, _]
comptime Exp = FloatUnary[ExpKernel, _]
comptime Ln = FloatUnary[LogKernel, _]


struct NumericCompare[
    K: NumericCompareKernel, L: NumericValue, R: NumericValue
](BoolValue, Unnamed):
    """A comparison over two numeric operands, producing packed bits.

    `Type` is not declared: `BoolValue` fixes it, because a comparison has no
    choice about what it produces.
    """

    comptime shape = widest_shape[Self.L, Self.R]

    comptime ArgType = promote[Self.L.Type, Self.R.Type]
    """The common type both operands are cast to before comparing.

    `int32 < int64` compares in `int64`; without this the narrower operand
    would be compared against a truncated view of the wider one.
    """

    comptime NativeType = wider[Self.L.Type.native, Self.R.Type.native]
    """Sizes the lane. The *operands*' width, not the output's — see
    `BoolValue.NativeType`."""

    comptime Bound = Tuple[Self.L.Bound, Self.R.Bound]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(self.l.columns(), self.r.columns())

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return (self.l.bind(batch, bindings), self.r.bind(batch, bindings))

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """A comparison is null-in, null-out: valid where both operands are.

        The data bit is computed regardless — a SIMD lane compares whatever is
        in the slot — so this bitmap is the *only* thing that records the
        result is meaningless there. Reading the data bit without it is the
        defect that made NULL join keys match each other.
        """
        return Bitmap.intersect(
            self.l.validity(bound[0]), self.r.validity(bound[1])
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return Self.K.core[Self.ArgType.native, W](
            self.l.lane[W](bound[0], idx).cast[Self.ArgType.native](),
            self.r.lane[W](bound[1], idx).cast[Self.ArgType.native](),
        )

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        """`prune` is `lane` in the interval domain, and reads the same way:
        both operands cast to `ArgType`, then the kernel.

        The reading is selected by a `comptime if` over `Self.K.name` rather
        than by a second struct parameter. That keeps this purely additive —
        no arity change, no alias change, no call site touched — and its
        failure mode is conservative: an operator with no arm falls through to
        `maybe`, which is always correct. Only the taken arm is emitted, so a
        `Gt` node links `GtBounds.decide` and nothing else.
        """
        comptime dt = Self.ArgType.native
        var lb = self.l.bounds(stats, bindings).cast[dt]()
        var rb = self.r.bounds(stats, bindings).cast[dt]()
        comptime if Self.K.name == LtKernel.name:
            return Truth(LtBounds.maybe[dt](lb, rb))
        elif Self.K.name == LeKernel.name:
            return Truth(LeBounds.maybe[dt](lb, rb))
        elif Self.K.name == GtKernel.name:
            return Truth(GtBounds.maybe[dt](lb, rb))
        elif Self.K.name == GeKernel.name:
            return Truth(GeBounds.maybe[dt](lb, rb))
        elif Self.K.name == EqKernel.name:
            return Truth(EqBounds.maybe[dt](lb, rb))
        elif Self.K.name == NeKernel.name:
            return Truth(NeBounds.maybe[dt](lb, rb))
        else:
            return Truth.maybe

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime Eq = NumericCompare[EqKernel, _, _]
comptime Ne = NumericCompare[NeKernel, _, _]
comptime Lt = NumericCompare[LtKernel, _, _]
comptime Le = NumericCompare[LeKernel, _, _]
comptime Gt = NumericCompare[GtKernel, _, _]
comptime Ge = NumericCompare[GeKernel, _, _]


struct CaseWhen[C: BoolValue, T: NumericValue, E: NumericValue](
    ColumnBound, NumericValue, Unnamed
):
    """`CASE WHEN cond THEN then ELSE otherwise END`, over numeric branches.

    **Not element-wise fused, deliberately.** `bind` computes the whole result
    through `kernels.conditional.case_when` and `lane` reads it back, so the
    `Bound` is the answer rather than the operands. `expr/`'s `CaseWhen` does
    the same thing for the same reason: which branch supplies a row depends on
    the condition's *validity* as well as its value — a null condition counts
    as false, and a selected-but-null value stays null — and that three-way
    rule is the kernel's, not a lane's. Reimplementing it per lane would be the
    `_kleene` mistake, where per-lane boolean validity measured 4-10x worse
    than the kernel's bitmap algebra.

    What it does keep is the *type*: the branches are comptime nodes, so
    `then` and `otherwise` are still fused subtrees evaluated without
    materialising their own intermediates, and the output type is `T`'s with no
    dispatch.
    """

    comptime Type = Self.T.Type
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.Type]
    """The computed result. `bind` is where the work happens."""

    var cond: Self.C
    var then: Self.T
    var otherwise: Self.E

    def __init__(
        out self, var cond: Self.C, var then: Self.T, var otherwise: Self.E
    ):
        self.cond = cond^
        self.then = then^
        self.otherwise = otherwise^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(
            merged(self.cond.columns(), self.then.columns()),
            self.otherwise.columns(),
        )

    def dtype(self, schema: Schema) raises -> DynType:
        return self.then.dtype(schema)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var n = len(batch)
        var conditions = List[BoolArray]()
        conditions.append(
            self.cond.evaluate(batch, bindings).to_array(n).as_bool().copy()
        )
        var values = List[DynArray]()
        values.append(self.then.evaluate(batch, bindings).to_array(n))
        var else_ = Optional(
            self.otherwise.evaluate(batch, bindings).to_array(n)
        )
        return (
            case_when(conditions, values, else_^)
            .as_primitive[Self.Type]()
            .copy()
        )

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "if_else(", self.cond, ", ", self.then, ", ", self.otherwise, ")"
        )


struct TemporalCompare[
    K: NumericCompareKernel, L: TemporalValue, R: TemporalValue
](BoolValue, Unnamed):
    """A comparison over two temporal operands, producing packed bits.

    Separate from `NumericCompare` for one reason: `NumericCompare.ArgType` is
    `promote[L.Type, R.Type]`, and `promote` encodes *numeric* widening —
    signedness and int-to-float. Those rules are wrong for temporal, where the
    question is not width but **unit**: `timestamp[s]` against `timestamp[ms]`
    needs one side *scaled*, not widened.

    **So this compares only operands that already share a representation, and
    checks it rather than assuming.** The comptime assert catches a width
    mismatch (`date32` against `timestamp[ns]`) at compile time; `bind` checks
    the dtypes are actually equal, which is what catches a *unit* mismatch that
    the widths agree on. Cross-unit comparison is deliberately not silently
    wrong — it raises, and adding it means choosing coercion rules (Arrow C++'s
    `common_temporal_resolution` is the prior art). Backlog S21.

    **Merging this with `NumericCompare` into one node parameterised over the
    argument type was tried three ways and none is clean** (measured against
    `mojo precompile`, 2026-08-24), so the shared bodies stay duplicated:

    - `ArgType = L.Type if TEMPORAL else promote[L.Type, R.Type]` over a common
      `L: PrimitiveValue` bound fails at the *declaration* — `parameter 'L' has
      'NumericType' type, but value has type 'PrimitiveType'`. The branch that
      never runs still has to be well-formed.
    - Making it total by widening `promote` to `PrimitiveType` does compile,
      and the conditional does reduce and does carry the `NumericType` bound.
      But that is exactly the change `TemporalColumn`'s docstring defers — a
      decision about promotion semantics, not a bound to widen — and it makes
      `Gt(date32_col, time32_col)` compile, where today it is rejected.
    - Passing the argument type as a parameter, `Compare[K, A, L, R]` with
      narrow-bounded aliases supplying `A`, keeps every bound and compiles, but
      `A` appears nowhere in `__init__` and so cannot be inferred: `Gt(col(...),
      lit(...))` fails with *value passed to 'l' cannot be converted from
      'Column[...]' to 'GtKernel'*. Only the explicit `Gt[L, R](...)` spelling
      survives, and every call site uses the inferred one.

    So what separates the two nodes is the operand *bounds*, not the bodies:
    they are what make `NumericCompare` over temporal operands — and a
    temporal-versus-numeric comparison — unrepresentable rather than merely
    wrong. Both are rejected at compile time today; every unification that
    preserves the call syntax demotes at least one to a runtime error.
    """

    comptime ArgType = Self.L.Type
    """The shared representation, named as a member.

    `Self.L.Type.native` used directly at a call site does not reduce — the
    compiler reports a type "cannot be converted" to *itself*, because a
    chained projection is not canonicalised. Binding it here makes every later
    use a single projection off `Self.ArgType`, which is exactly why
    `NumericCompare` has an `ArgType` too.
    """

    comptime NativeType = Self.ArgType.native
    comptime shape = widest_shape[Self.L, Self.R]
    comptime Bound = Tuple[Self.L.Bound, Self.R.Bound]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        comptime assert (
            Self.L.Type.native == Self.R.Type.native
        ), "temporal comparison needs one representation on both sides"
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(self.l.columns(), self.r.columns())

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        # Same width is not the same type: `date32` and `time32[s]` are both
        # int32, and `timestamp[s]` and `timestamp[ms]` differ only in a unit
        # the width cannot see. Checked once per batch, not per row.
        if self.l.dtype(Schema.from_dtype(batch.dtype)) != self.r.dtype(
            Schema.from_dtype(batch.dtype)
        ):
            raise Error(
                "temporal comparison between ",
                String(self.l.dtype(Schema.from_dtype(batch.dtype))),
                " and ",
                String(self.r.dtype(Schema.from_dtype(batch.dtype))),
                ": units must match, coercion is not implemented",
            )
        return (self.l.bind(batch, bindings), self.r.bind(batch, bindings))

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """Null-in, null-out — as `NumericCompare`. The lane compares whatever
        bytes are present, so this bitmap is the only record that the answer is
        meaningless."""
        return Bitmap.intersect(
            self.l.validity(bound[0]), self.r.validity(bound[1])
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        # The representations are equal by construction, so this is the same
        # instruction a numeric comparison of the backing integers would be.
        # Both casts are no-ops — the constructor's assert guarantees the two
        # natives are identical — but the compiler cannot see that through the
        # projection, so they are spelled.
        return Self.K.core[Self.ArgType.native, W](
            self.l.lane[W](bound[0], idx).cast[Self.ArgType.native](),
            self.r.lane[W](bound[1], idx).cast[Self.ArgType.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime TemporalEq = TemporalCompare[EqKernel, _, _]
comptime TemporalNe = TemporalCompare[NeKernel, _, _]
comptime TemporalLt = TemporalCompare[LtKernel, _, _]
comptime TemporalLe = TemporalCompare[LeKernel, _, _]
comptime TemporalGt = TemporalCompare[GtKernel, _, _]
comptime TemporalGe = TemporalCompare[GeKernel, _, _]


# ---------------------------------------------------------------------------
# ConditionalBinary — coalesce / nullif / fill_null
# ---------------------------------------------------------------------------
struct ConditionalBinary[
    K: BinaryConditionalKernel, L: NumericValue, R: NumericValue
](ColumnBound, NumericValue, Unnamed):
    """`coalesce` / `nullif` / `fill_null` over two same-dtype numeric operands.

    A breaker, and the one whose validity is the point rather than a detail.
    These nodes decide their nulls from the operands' *values* — `coalesce`
    takes the second operand exactly where the first was null, `nullif`
    manufactures a null where two values are equal — so validity cannot be
    derived by intersecting bitmaps and must come from the computed result.

    **That is why this conforms to `ColumnBound`**, and it is what
    `comptime/core.mojo` is referring to when it says the previous expression
    layer needed two validity methods and evaluated
    `coalesce`/`nullif`/`case_when` twice per fused pass: its `validity` took
    the batch, so it re-ran `combine` over both operands to recover a bitmap
    the result already carried. Here `bind` computes once and `validity` reads
    the bound.
    """

    comptime Type = Self.L.Type
    comptime shape = Shape.columnar
    """Columnar regardless of the operands. The kernel runs over columns, so
    this reports what it does — as `CaseWhen` does."""

    comptime Bound = PrimitiveArray[Self.Type]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(self.l.columns(), self.r.columns())

    def dtype(self, schema: Schema) raises -> DynType:
        return self.l.dtype(schema)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var n = len(batch)
        return (
            Self.K.combine(
                self.l.evaluate(batch, bindings).to_array(n),
                self.r.evaluate(batch, bindings).to_array(n),
            )
            .as_primitive[Self.Type]()
            .copy()
        )

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime Coalesce = ConditionalBinary[CoalesceKernel, _, _]
comptime Nullif = ConditionalBinary[NullifKernel, _, _]
comptime FillNull = ConditionalBinary[FillNullKernel, _, _]
"""`fill_null(l, r)` — `l` with its nulls taken from `r`.

For two operands this computes what `Coalesce` does, and it is kept as its own
name because that is the verb PyArrow and Polars users reach for, and because
the kernel additionally pins both operands to one dtype."""
