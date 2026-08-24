"""Fused numeric operators: nodes built from other numeric nodes.

Each is generic over its operands' *types*, so the whole subtree is one type
and the driver inlines it into one loop. `Add[Column[Int64Type],
Literal[Int64Type]]` names its kernel, its operand types and its output type at
compile time; nothing is looked up when it runs.

The kernel is a parameter rather than a field for the same reason `_eval` is a
pointer in the runtime lane: routing on a name would put every arithmetic
kernel in every binary that builds any expression.
"""

from ...dtypes import DataType, DynType, NumericType
from ...kernels.numeric import (
    AddKernel,
    GtKernel,
    LtKernel,
    NumericCompareKernel,
    BinaryNumericKernel,
    MulKernel,
    SubKernel,
)
from ...arrays import StructArray, BoolArray, DynArray, PrimitiveArray
from ...kernels.conditional import case_when
from ...schema import Schema
from ...tabular import RecordBatch
from ...buffers import Bitmap
from ..logical import Shape, merged
from ..params import Bindings
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

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime Gt = NumericCompare[GtKernel, _, _]
comptime Lt = NumericCompare[LtKernel, _, _]


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


comptime TemporalGt = TemporalCompare[GtKernel, _, _]
comptime TemporalLt = TemporalCompare[LtKernel, _, _]
