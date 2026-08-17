"""Expression execution — staged, strategy-pluggable fusion (see
`docs/architecture.md`).

Model
-----
- `execute(batch) -> Datum` is the **one universal verb** every node has.
  `Datum = Scalar | Array` (Arrow's Datum / DataFusion's ColumnarValue) is the
  strategy-agnostic wire format between stages.
- Fusion is a **pluggable strategy**, not a single primitive. A strategy = a
  composable per-element `core` + a driver that runs a whole same-strategy subtree
  in one pass. Every fused node implements the same two-method protocol:

      comptime State: Copyable & Deinitable   # per-node, comptime
      def state(self, batch) raises -> Self.State       # once per pass
      def lane[W](self, state, idx) -> ...              # once per SIMD chunk

  `state()` resolves everything the loop needs — a column lookup, a `Variant`
  unwrap, a materialized breaker stage — *before* the loop starts; `lane()` reads
  only `state` and `idx`. The three families differ only in what `lane` returns:
    * `NumericValue` — vectorized: `lane[W] -> SIMD[native, W]`, driver fills a `Buffer`.
    * `BoolValue`    — vectorized: `lane[W] -> SIMD[bool, W]`, driver bit-packs a `Bitmap`.
    * `StringValue`  — elementwise: `lane(idx) -> String`, driver appends to a builder
      (variable-width UTF-8 has no W-wide lane, so `col || "a" || "b"` fuses one
      row at a time — no intermediate `StringArray`).
- A composite's `State` is built out of its children's: a unary node's is its
  operand's, a binary node's is `Pair[L.State, R.State]`. So one `state()` call
  at the root resolves the whole subtree, and the lane loop touches nothing else.
- **Pipeline breakers** — cross-row ops (`Reduction`, `WindowFunction`) that can't
  fuse in any strategy — cut the tree into a forest of fused stages. A breaker's
  `State` **is** its materialized stage result, computed once by `state()`; it then
  behaves as an ordinary fused leaf: a scalar reduction **splats** it (like a
  literal), a columnar one **loads** from it (like a column). So the stage above
  still fuses through the single `NumericBinary` — there is no separate
  "materialized" binary.
- Expressions are **immutable**: all per-execute state lives in the `State` value
  the driver owns for the duration of one pass, keyed by *type* rather than by
  position. That is what replaced a shared `Context` of positionally-addressed
  slots, where a `prepare` walk and a `core` walk had to agree on a DFS order.
"""

from std.sys import bit_width_of
from std.builtin.rebind import downcast
from std.utils import Variant

from ..schema import Schema
from ..tabular import RecordBatch
from ..arrays import (
    DynArray,
    PrimitiveArray,
    Int32Array,
    BoolArray,
    BinaryLikeArray,
)
from ..scalars import DynScalar, BoolScalar, PrimitiveScalar, StringScalar
from ..buffers import Buffer, Bitmap
from ..builders import Int64Builder, BinaryLikeBuilder
from ..views import apply
from ..dtypes import (
    FloatingType,
    DynType,
    DataType,
    NumericType,
    DType,
    Int32Type,
    Int64Type,
    Float64Type,
    BoolType,
    StringLikeType,
    StringType,
    ListLikeType,
    TemporalType,
)
from ..kernels.numeric import (
    NumericCompareKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
    EqKernel,
    NeKernel,
)
from ..kernels.numeric import (
    BinaryKernel,
    BinaryNumericKernel,
    UnaryKernel,
    UnaryNumericKernel,
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    FloordivKernel,
    ModKernel,
    NegKernel,
    AbsKernel,
    SignKernel,
    FloorKernel,
    CeilKernel,
    RoundKernel,
    PowKernel,
    SqrtKernel,
    ExpKernel,
    LogKernel,
)
from ..kernels.aggregate import (
    Aggregation,
    NumericAgg,
    TemporalMinMax,
    StringMinMax,
    CountAgg,
    DistinctAgg,
    MinOp,
    MaxOp,
    AggKernel,
    SumKernel,
    MeanKernel,
    MinKernel,
    MaxKernel,
    ProductKernel,
    CountKernel,
    BoolReduceKernel,
    AnyKernel,
    AllKernel,
)
from ..kernels.string import (
    LengthKernel,
    ConcatKernel,
    StringMapKernel,
    UpperKernel,
    LowerKernel,
    StripKernel,
    LStripKernel,
    RStripKernel,
    ReverseKernel,
    CapitalizeKernel,
    StringPredicateKernel,
    StartsWithKernel,
    EndsWithKernel,
    ContainsKernel,
    StringEqKernel,
    StringNeKernel,
    StringLtKernel,
    StringLeKernel,
    StringGtKernel,
    StringGeKernel,
    LikeKernel,
    ILikeKernel,
)
from ..kernels.membership import IsInKernel
from ..kernels.conditional import (
    case_when as case_when_kernel,
    BinaryConditionalKernel,
    CoalesceKernel,
    NullifKernel,
)
from ..kernels.temporal import (
    TemporalExtractKernel,
    YearKernel,
    MonthKernel,
    DayKernel,
    HourKernel,
    MinuteKernel,
    SecondKernel,
    DayOfWeekKernel,
    QuarterKernel,
    DayOfYearKernel,
    CalendarUnit,
    DateTruncKernel,
)
from ..kernels.boolean import (
    BoolBinaryKernel,
    BoolUnaryKernel,
    AndKernel,
    OrKernel,
    XorKernel,
    NotKernel,
    UnaryPredicateKernel,
    ValuePredicateKernel,
    IsNullKernel,
    NotNullKernel,
    IsNanKernel,
    IsInfKernel,
)
from ..kernels.nested import ArrayLengthKernel, ArrayContainsKernel
from ..kernels.interval import (
    Interval,
    IntervalKernel,
    LtInterval,
    LeInterval,
    GtInterval,
    GeInterval,
    EqInterval,
    NeInterval,
    AndInterval,
    OrInterval,
    XorInterval,
)
from .pruning import PruneStats
from .dynamic import DynAgg, DynValue, _promote_operands
from .relations import BoxedValue
from .aggregates import AggFunc
from ..kernels.cast import (
    cast as cast_array,
    NumericCast as NumericCastKernel,
    NumToBool as NumToBoolKernel,
    BoolToNum as BoolToNumKernel,
    StringToNum as StringToNumKernel,
    StringToBool as StringToBoolKernel,
    NumToString as NumToStringKernel,
    BoolToString as BoolToStringKernel,
    BinaryLikeCast as StringToStringKernel,
)


# ---------------------------------------------------------------------------
# Datum — Scalar | Array, the uniform `execute` result.
# ---------------------------------------------------------------------------
comptime Datum = Variant[DynScalar, DynArray]


def into_array(d: Datum, n: Int) raises -> DynArray:
    """Force `d` to an array of length `n` — broadcasting a scalar (lazy until here).
    """
    if d.isa[DynScalar]():
        return d[DynScalar].repeat(n)
    return d[DynArray].copy()


# Known follow-ups (flagged during design; not yet addressed):
#  - CSE: identical breaker subtrees (`sum(a)` used twice) each build their own
#    `State`, so they recompute. A keyed memo over subtree identity restores it.
#  - SCHEDULER: independent breakers materialize sequentially inside `state()`;
#    they are independent stages and can be scheduled to run concurrently.
#  - `DynScalar.repeat` has no string support, so a string *scalar* cannot broadcast
#    to a column yet (core-array machinery, orthogonal to fusion).


# ---------------------------------------------------------------------------
# Pair — a binary node's `State`: its two operands' states, under names.
# ---------------------------------------------------------------------------
@fieldwise_init
struct Pair[
    L: Copyable & Deinitable,
    R: Copyable & Deinitable,
](Copyable, Deinitable, Movable):
    """The state of a two-operand fused node.

    `Tuple[L.State, R.State]` is the spelling CLAUDE.md records as verified and
    it works here too; this is a plain struct instead because named fields read
    better at the use site — `state.l` / `state.r` against `state[0]` /
    `state[1]` — and because it carries none of `Tuple`'s variadic-pack storage.
    """

    var l: Self.L
    var r: Self.R


# ---------------------------------------------------------------------------
# Promotion — output dtype is the wider operand (Add(int32,int64) -> int64)
# ---------------------------------------------------------------------------
def _rank[T: DataType]() -> Int:
    comptime if conforms_to(T, NumericType):
        comptime N = downcast[T, NumericType]()
        return bit_width_of[N.native]() + (
            1000 if N.native.is_floating_point() else 0
        )
    else:
        return 0


comptime promote[L: NumericType, R: NumericType] = L if (
    _rank[L]() >= _rank[R]()
) else R

# Lane width, a different question from `promote`: the bit-packing driver sizes
# `W` from a DType, and a narrower one yields a *larger* `W`, so a wider operand's
# load would overflow the SIMD register. Every bool node whose operands may differ
# in width picks `wider` of the two; `promote` (where floats outrank ints
# regardless of bit width) decides the *value* domain, not the register size.
comptime wider[L: DType, R: DType] = L if (
    bit_width_of[L]() >= bit_width_of[R]()
) else R


# ---------------------------------------------------------------------------
# Plan analysis — order-preserving dedup union of column-name lists, so a
# composite node can fold its children's `referenced_columns()` without repeats.
# ---------------------------------------------------------------------------
def _union_columns(var acc: List[String], names: List[String]) -> List[String]:
    for i in range(len(names)):
        var seen = False
        for j in range(len(acc)):
            if acc[j] == names[i]:
                seen = True
                break
        if not seen:
            acc.append(names[i].copy())
    return acc^


# ---------------------------------------------------------------------------
# Validity — the fused lane threads result nulls alongside the data buffer. Each
# node exposes `validity(batch)` returning an offset-0 owned bitmap (`None` = all
# valid); the numeric/bool drivers combine it into the finished array. Propagating
# ops AND-combine children validities via the same `bitmap_and` helper the kernels
# use; Kleene `And`/`Or` reuse the null-correct `and_`/`or_` kernels; column leaves
# read their own validity; literals / `is_null` results are always valid.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The fused drivers. Each takes the node's `State` — resolved once by the caller
# — and runs one pass, reading it through a producer closure handed to
# `views.apply`.
#
# Free functions rather than trait default methods so that `V.State` is a single
# projection off a *directly trait-bound parameter*, the form CLAUDE.md records
# as reducing reliably, rather than `Self.State` inside a trait default.
# ---------------------------------------------------------------------------
def _drive_numeric[
    V: NumericValue
](node: V, batch: RecordBatch, state: V.State) raises -> Datum:
    """One fused numeric pass: fill a `Buffer` from `node.lane`."""
    comptime native = V.OutType.native
    comptime if V.OutShape == 0:  # scalar → evaluate the lane once, then splat
        return PrimitiveScalar[V.OutType](node.lane[1](state, 0)[0]).to_dyn()
    else:  # columnar → one fused vectorized pass
        var length = batch.num_rows()
        var buf = Buffer.alloc_uninit[native](length)

        @always_inline
        def producer[W: Int](i: Int) {imm} -> SIMD[native, W]:
            return node.lane[W](state, i)

        apply[native](buf.view[native](0, length), producer)
        var v = node.state_validity(batch, state)
        var arr = PrimitiveArray[V.OutType](
            dtype=V.OutType(),
            length=length,
            nulls=v.value().unset_count() if v else 0,
            offset=0,
            bitmap=v^,
            buffer=buf.to_immutable(),
        )
        return arr^.to_dyn()


def _drive_bool[
    V: BoolValue
](node: V, batch: RecordBatch, state: V.State) raises -> Datum:
    """One fused bool pass: bit-pack a `Bitmap` from `node.lane`."""
    var length = batch.num_rows()
    var bm = Bitmap.alloc_uninit(length)

    @always_inline
    def producer[W: Int](i: Int) {imm} -> SIMD[DType.bool, W]:
        return node.lane[W](state, i)

    apply[V.NativeType](bm.view(), producer)  # bit-packing overload
    var v = node.state_validity(batch, state)
    return BoolArray(
        length=length,
        nulls=v.value().unset_count() if v else 0,
        offset=0,
        bitmap=v^,
        buffer=bm.to_immutable(),
    ).to_dyn()


def _drive_string[
    V: StringValue
](node: V, batch: RecordBatch, state: V.State) raises -> Datum:
    """One fused string pass: append `node.lane` into a builder. No closure
    here, but it lives beside the other two so the three drivers read alike."""
    comptime if V.OutShape == 0:
        return StringScalar(node.lane(state, 0)).to_dyn()
    else:
        var n = batch.num_rows()
        # Validity is threaded here for the same reason the numeric and bool
        # drivers thread it: `lane` reads values through `unsafe_get`, which
        # does not consult the bitmap, so without this every string
        # *transformation* returned an all-valid column. A bare column keeps
        # its nulls (that path returns the column as-is), which is what hid
        # this.
        var v = node.state_validity(batch, state)
        var builder = BinaryLikeBuilder[V.OutType](capacity=n)
        for i in range(n):
            if v and not v.value().test(i):
                builder.append_null()
            else:
                builder.append(node.lane(state, i))
        return builder.finish().to_dyn()


# ---------------------------------------------------------------------------
# Value — every node. `materialize` is abstract; the family traits default it to
# their fused driver and the breakers override it with their stage result.
# ---------------------------------------------------------------------------
trait Value(Copyable, Deinitable, Movable):
    """A node. Every member is a runtime method.

    Both comptime members this once carried are gone. `OutType` went first — no
    `[V: Value]` code read it. `OutShape` followed: its real readers are the
    numeric and string drivers, which reach it through the *family* traits, and
    one node (`NullPredicate`) that propagated its operand's shape while always
    materializing a column regardless. It is declared on `NumericValue`,
    `BoolValue` and `StringValue` instead.

    That leaves a trait of runtime methods only — exactly the property
    `7d57398` requires of anything the boxes erase into, now with no
    qualification."""

    def materialize(self, batch: RecordBatch) raises -> Datum:
        """Produce this node's result `Datum` — the family driver: a numeric `Buffer`,
        a bool `Bitmap`, a string builder, or (for a leaf like `ListColumn`) just its
        column."""
        ...

    def execute(self, batch: RecordBatch) raises -> Datum:
        """Top-level entry — `materialize` under the name callers use.

        The two cannot be collapsed into one, though the bodies now say the same
        thing: `DynValue` needs an `execute(batch) -> DynArray` for the
        relational engine, and Mojo will not overload on return type alone. Two
        names is what lets it satisfy the trait with `materialize -> Datum` and
        still expose its own `execute`. `relations.BoxedValue` calls
        `materialize` for exactly that reason."""
        return self.materialize(batch)

    def name(self) -> String:
        """Best-effort output name — a column returns its name; most nodes are
        anonymous (`""`). Used by the relational engine for output schema fields.
        """
        return String()

    # --- plan analysis (projection / predicate pushdown) --------------------
    def referenced_columns(self) -> List[String]:
        """The column names this node reads. A column-ref returns its own name; a
        literal returns `[]`; a composite returns the order-preserving deduped
        union of its children. Abstract — every node fixes it concretely so an
        optimizer can push projections/predicates through the fused subtree."""
        ...

    def render(self) -> String:
        """How this node prints inside a plan.

        The interpreter rendered a whole tree by switching on tags. A node
        renders itself instead, so a new node cannot be missed by the printer —
        it either says something or falls back to its name."""
        return self.name()

    def bound_column(self, schema: Schema) raises -> Int:
        """This node's column position if it is a bare column reference, else
        `-1`.

        Join and group keys must be plain columns, and the relational layer used
        to establish that by asking the interpreter for its tag (`kind() ==
        LOAD`) and then its payload. That let a caller reach into a
        representation it should not know about. A node answers for itself
        instead, and every non-column node inherits "no"."""
        return -1

    def prune(self, stats: PruneStats) raises -> Interval:
        """What this node's value can be, given per-column statistics.

        Defaults to "no information", which is always sound — a caller only ever
        skips data it has *proven* cannot match. Nodes that can say something
        useful (columns, literals, comparisons, `and`/`or`) override it.

        This used to be a 9-arm switch on the interpreter's tag. Putting it on
        the node means a new node cannot be forgotten by it: it either says
        something or inherits the conservative answer."""
        return Interval.unknown()

    # --- validity (null tracking) -------------------------------------------
    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity as an offset-0 owned bitmap, or `None`
        when every element is valid. Default `None` (all-valid) covers literals,
        `is_null`/`not_null` results, windows, and reductions; column leaves read
        their own validity and propagating ops AND-combine their children's."""
        return None

    # --- fluent surface available in every family (reads only validity) ------
    #
    # `count_distinct`/`approx_count_distinct` live here rather than once per
    # family: `AggExpr.of` is bound on `Value`, so neither copy needed a family
    # trait. They were byte-identical in `NumericValue` and `StringValue`, and
    # that duplication was not inert — a struct conforming to both inherits two
    # conflicting defaults and will not compile, which is how `DynValue` found
    # them.
    def count_distinct(self) -> AggExpr:
        return AggExpr.of[DistinctAgg[True]](self.copy())

    def approx_count_distinct(self) -> AggExpr:
        return AggExpr.of[DistinctAgg[False]](self.copy())

    def isnull(self) -> IsNull[Self]:
        return IsNull(self.copy())

    def notnull(self) -> NotNull[Self]:
        return NotNull(self.copy())


trait NumericValue(Value):
    comptime OutShape: Int  # 0 scalar, 1 columnar
    comptime OutType: NumericType

    comptime State: Copyable & Deinitable
    """Everything this node's lane needs, resolved once per pass.

    A column leaf's is its typed column; a literal's is nothing; a composite's is
    built from its children's (`Pair[L.State, R.State]` for a binary node); a
    breaker's is its materialized stage. Declared per concrete struct — a trait
    default here cannot be reduced at a `-> Self.State` return site unless the
    bound is `ImplicitlyCopyable`, which array states deliberately are not."""

    def state(self, batch: RecordBatch) raises -> Self.State:
        """Resolve this subtree against `batch`, once, before the lane loop."""
        ...

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        """One SIMD chunk. Reads `state` and `idx` and nothing else — that
        removal is the optimization, worth 30x on `a + 1` over 1M rows."""
        ...

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity, given the state the driver already has.

        Defaults to `validity(batch)`: most nodes derive validity from their
        operands and have no use for the state. A **breaker** whose `State` is
        its materialized result overrides this to read the bitmap straight off
        that array. Without the hook the driver's two calls — `state()` and
        `validity()` — are independent, and for `coalesce`, `nullif` and
        `case_when` each one ran the whole selection kernel, so every fused pass
        over them did the work twice (FU-7a)."""
        return self.validity(batch)

    # --- fluent surface: arithmetic, comparison, unary, reductions -----------
    def __add__[Rhs: NumericValue](self, o: Rhs) -> Add[Self, Rhs]:
        return Add(self.copy(), o.copy())

    def __sub__[Rhs: NumericValue](self, o: Rhs) -> Sub[Self, Rhs]:
        return Sub(self.copy(), o.copy())

    def __mul__[Rhs: NumericValue](self, o: Rhs) -> Mul[Self, Rhs]:
        return Mul(self.copy(), o.copy())

    def __truediv__[Rhs: NumericValue](self, o: Rhs) -> Div[Self, Rhs]:
        return Div(self.copy(), o.copy())

    def __mod__[Rhs: NumericValue](self, o: Rhs) -> Mod[Self, Rhs]:
        return Mod(self.copy(), o.copy())

    def __floordiv__[Rhs: NumericValue](self, o: Rhs) -> Floordiv[Self, Rhs]:
        return Floordiv(self.copy(), o.copy())

    def __pow__[Rhs: NumericValue](self, o: Rhs) -> Pow[Self, Rhs]:
        return Pow(self.copy(), o.copy())

    def __neg__(self) -> Neg[Self]:
        return Neg(self.copy())

    def __lt__[Rhs: NumericValue](self, o: Rhs) -> Lt[Self, Rhs]:
        return Lt(self.copy(), o.copy())

    def __le__[Rhs: NumericValue](self, o: Rhs) -> Le[Self, Rhs]:
        return Le(self.copy(), o.copy())

    def __gt__[Rhs: NumericValue](self, o: Rhs) -> Gt[Self, Rhs]:
        return Gt(self.copy(), o.copy())

    def __ge__[Rhs: NumericValue](self, o: Rhs) -> Ge[Self, Rhs]:
        return Ge(self.copy(), o.copy())

    def __eq__[Rhs: NumericValue](self, o: Rhs) -> Eq[Self, Rhs]:
        return Eq(self.copy(), o.copy())

    def __ne__[Rhs: NumericValue](self, o: Rhs) -> Ne[Self, Rhs]:
        return Ne(self.copy(), o.copy())

    def abs(self) -> Abs[Self]:
        return Abs(self.copy())

    def sign(self) -> Sign[Self]:
        return Sign(self.copy())

    def floor(self) -> Floor[Self]:
        return Floor(self.copy())

    def ceil(self) -> Ceil[Self]:
        return Ceil(self.copy())

    def round(self) -> Round[Self]:
        return Round(self.copy())

    def isnan(self) -> IsNan[Self]:
        return IsNan(self.copy())

    def isinf(self) -> IsInf[Self]:
        return IsInf(self.copy())

    def sqrt(self) -> Sqrt[Self]:
        return Sqrt(self.copy())

    def exp(self) -> Exp[Self]:
        return Exp(self.copy())

    def ln(self) -> Ln[Self]:
        return Ln(self.copy())

    def sum(self) -> Sum[Self]:
        return Sum(self.copy())

    def mean(self) -> Mean[Self]:
        return Mean(self.copy())

    def min(self) -> Min[Self]:
        return Min(self.copy())

    def max(self) -> Max[Self]:
        return Max(self.copy())

    def product(self) -> Product[Self]:
        return Product(self.copy())

    def count(self) -> Count[Self]:
        return Count(self.copy())

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return _drive_numeric(self, batch, self.state(batch))

    # --- aggregates without a fused reduction -------------------------------
    #
    # `sum`/`mean`/`min`/`max`/`product`/`count` are already above: they build a
    # `Reduction`, which is both a scalar value inside an expression and — via
    # `AggExpr`'s conversion — an aggregate in a `GROUP BY`. The distinct counts
    # have no fused form (their state is a hash set / HLL sketch, not a scalar
    # accumulator), so they go straight to an `AggExpr`.


struct NumericColumn[T: NumericType](NumericValue):
    """A numeric column, resolved by name against `batch.schema` **once per
    pass** — `state()` does the lookup and the unwrap, `lane()` only loads."""

    comptime OutType = Self.T
    comptime OutShape = 1
    comptime State = PrimitiveArray[Self.T]

    var _name: String

    def referenced_columns(self) -> List[String]:
        return [self._name.copy()]

    def bound_column(self, schema: Schema) raises -> Int:
        var i = schema.get_field_index(self._name)
        if i == -1:
            raise Error("column '", self._name, "' not found")
        return i

    def prune(self, stats: PruneStats) raises -> Interval:
        var iv = stats.by_name(self._name)
        return Interval.bounds(iv[0].copy(), iv[1].copy())

    def __init__(out self, var name: String):
        self._name = name^

    def state(self, batch: RecordBatch) raises -> Self.State:
        # The schema lookup and the `Variant` unwrap both live here, out of the
        # lane. `RecordBatch.column(name)` owns the missing-name diagnostic —
        # `get_field_index` answers -1, and indexing the column list with that
        # trips a bounds assert that aborts the runner instead of reporting the
        # name. Every column leaf goes through it for that reason.
        return batch.column(self._name).as_primitive[Self.T]().copy()

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return state.values().load[W](idx)

    def materialize(self, batch: RecordBatch) raises -> Datum:
        # a leaf column returns as-is (keeps validity; the fused loop drops
        # nulls).
        return batch.column(self._name).copy()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return batch.column(self._name).to_data().owned_validity()

    def name(self) -> String:
        return self._name.copy()


@fieldwise_init
struct NumericLiteral[T: NumericType](NumericValue):
    """A numeric constant, broadcast into every lane."""

    comptime OutType = Self.T
    comptime OutShape = 0
    comptime State = NoneType
    """Nothing to resolve — the value is in the node, so the lane splats it."""

    def render(self) -> String:
        return String("literal(", self._value, ")")

    def prune(self, stats: PruneStats) raises -> Interval:
        var v = PrimitiveScalar[Self.T](self._value).to_dyn()
        return Interval.bounds(Optional(v.copy()), Optional(v^))

    var _value: Scalar[Self.OutType.native]

    def referenced_columns(self) -> List[String]:
        return List[String]()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return NoneType()

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return SIMD[Self.OutType.native, W](self._value)


@fieldwise_init
struct NumericBinary[K: BinaryNumericKernel, L: NumericValue, R: NumericValue](
    NumericValue
):
    """Fused arithmetic over two operands, widening to the wider dtype. There is no
    "materialized" counterpart: a breaker operand is itself a fused leaf (it reads
    its stage result from `ctx`), so it composes here like any column/literal.
    """

    comptime OutType = promote[Self.L.OutType, Self.R.OutType]
    comptime OutShape = max(Self.L.OutShape, Self.R.OutShape)
    comptime State = Pair[Self.L.State, Self.R.State]

    var l: Self.L
    var r: Self.R

    def render(self) -> String:
        return String(
            Self.K.name, "(", self.l.render(), ", ", self.r.render(), ")"
        )

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    def state(self, batch: RecordBatch) raises -> Self.State:
        return Pair(self.l.state(batch), self.r.state(batch))

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        var a = self.l.lane[W](state.l, idx).cast[Self.OutType.native]()
        var b = self.r.lane[W](state.r, idx).cast[Self.OutType.native]()
        return Self.K.core[Self.OutType.native, W](a, b)

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return Bitmap.intersect(self.l.validity(batch), self.r.validity(batch))

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree rather than re-deriving from `batch`.

        Without this the chain stops at the first composite: the driver asks
        the root, the root's default falls back to `validity(batch)`, and a
        breaker operand re-runs its whole kernel — which is the FU-7a cost this
        was meant to remove."""
        return Bitmap.intersect(
            self.l.state_validity(batch, state.l),
            self.r.state_validity(batch, state.r),
        )


@fieldwise_init
struct NumericUnary[K: UnaryNumericKernel, A: NumericValue](NumericValue):
    """Fused unary op preserving the operand dtype — `neg`, `abs`, …."""

    comptime OutType = Self.A.OutType
    comptime OutShape = Self.A.OutShape
    comptime State = Self.A.State
    var a: Self.A

    def render(self) -> String:
        return String(Self.K.name, "(", self.a.render(), ")")

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self.a.state(batch)

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return Self.K.core[Self.OutType.native, W](self.a.lane[W](state, idx))

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree — see `NumericBinary.state_validity`.
        """
        return self.a.state_validity(batch, state)


@fieldwise_init
struct NumericCast[To: NumericType, A: NumericValue](NumericValue):
    """Fused numeric → numeric cast — reinterprets the operand's SIMD lane at the
    target dtype, so `col.cast(int64) + other` stays a single fused pass."""

    comptime OutType = Self.To
    comptime OutShape = Self.A.OutShape
    comptime State = Self.A.State
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self.a.state(batch)

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return NumericCastKernel.core[
            Self.A.OutType.native, Self.OutType.native, W
        ](self.a.lane[W](state, idx))

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree — see `NumericBinary.state_validity`.
        """
        return self.a.state_validity(batch, state)


@fieldwise_init
struct FloatBinary[K: BinaryKernel, L: NumericValue, R: NumericValue](
    NumericValue
):
    """Binary op whose result is always float64 — `Div` (true division), `Pow`.
    Operands cast up to float64 before the kernel, so `5 / 2 == 2.5`."""

    comptime OutType = Float64Type
    comptime OutShape = max(Self.L.OutShape, Self.R.OutShape)
    comptime State = Pair[Self.L.State, Self.R.State]
    var l: Self.L
    var r: Self.R

    def render(self) -> String:
        return String(
            Self.K.name, "(", self.l.render(), ", ", self.r.render(), ")"
        )

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    def state(self, batch: RecordBatch) raises -> Self.State:
        return Pair(self.l.state(batch), self.r.state(batch))

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        var a = self.l.lane[W](state.l, idx).cast[Self.OutType.native]()
        var b = self.r.lane[W](state.r, idx).cast[Self.OutType.native]()
        return Self.K.core[Self.OutType.native, W](a, b)

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return Bitmap.intersect(self.l.validity(batch), self.r.validity(batch))

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree rather than re-deriving from `batch`.

        Without this the chain stops at the first composite: the driver asks
        the root, the root's default falls back to `validity(batch)`, and a
        breaker operand re-runs its whole kernel — which is the FU-7a cost this
        was meant to remove."""
        return Bitmap.intersect(
            self.l.state_validity(batch, state.l),
            self.r.state_validity(batch, state.r),
        )


@fieldwise_init
struct FloatUnary[K: UnaryKernel, A: NumericValue](NumericValue):
    """Unary op whose result is always float64 — `sqrt`, `exp`, `log`."""

    comptime OutType = Float64Type
    comptime OutShape = Self.A.OutShape
    comptime State = Self.A.State
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self.a.state(batch)

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return Self.K.core[Self.OutType.native, W](
            self.a.lane[W](state, idx).cast[Self.OutType.native]()
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree — see `NumericBinary.state_validity`.
        """
        return self.a.state_validity(batch, state)


comptime Add = NumericBinary[AddKernel, _, _]
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]
comptime Mod = NumericBinary[ModKernel, _, _]
comptime Floordiv = NumericBinary[FloordivKernel, _, _]
comptime Div = FloatBinary[DivKernel, _, _]
comptime Pow = FloatBinary[PowKernel, _, _]
comptime Neg = NumericUnary[NegKernel, _]
comptime Abs = NumericUnary[AbsKernel, _]
comptime Sign = NumericUnary[SignKernel, _]
comptime Floor = NumericUnary[FloorKernel, _]
comptime Ceil = NumericUnary[CeilKernel, _]
comptime Round = NumericUnary[RoundKernel, _]
comptime Sqrt = FloatUnary[SqrtKernel, _]
comptime Exp = FloatUnary[ExpKernel, _]
comptime Ln = FloatUnary[LogKernel, _]


# ---------------------------------------------------------------------------
# BoolValue — the vectorized bool strategy. Same SIMD `core`, but the driver
# bit-packs a `Bitmap` (the one physical difference from the numeric lane).
# ---------------------------------------------------------------------------
trait BoolValue(Value):
    comptime OutShape: Int  # 0 scalar, 1 columnar
    comptime NativeType: DType  # operand width (sizes the SIMD lane), not the output

    comptime State: Copyable & Deinitable
    """Resolved once per pass — see `NumericValue.State`."""

    def state(self, batch: RecordBatch) raises -> Self.State:
        ...

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        ...

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity, given the state the driver already has.

        Defaults to `validity(batch)`: most nodes derive validity from their
        operands and have no use for the state. A **breaker** whose `State` is
        its materialized result overrides this to read the bitmap straight off
        that array. Without the hook the driver's two calls — `state()` and
        `validity()` — are independent, and for `coalesce`, `nullif` and
        `case_when` each one ran the whole selection kernel, so every fused pass
        over them did the work twice (FU-7a)."""
        return self.validity(batch)

    # --- fluent surface: boolean logic + reductions --------------------------
    def __and__[Rhs: BoolValue](self, o: Rhs) -> And[Self, Rhs]:
        return And(self.copy(), o.copy())

    def __or__[Rhs: BoolValue](self, o: Rhs) -> Or[Self, Rhs]:
        return Or(self.copy(), o.copy())

    def __xor__[Rhs: BoolValue](self, o: Rhs) -> Xor[Self, Rhs]:
        return Xor(self.copy(), o.copy())

    def __invert__(self) -> Not[Self]:
        return Not(self.copy())

    def any(self) -> Any[Self]:
        return Any(self.copy())

    def all(self) -> All[Self]:
        return All(self.copy())

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return _drive_bool(self, batch, self.state(batch))


@fieldwise_init
struct NumericCompare[
    K: NumericCompareKernel,
    P: IntervalKernel,
    L: NumericValue,
    R: NumericValue,
](BoolValue):
    """Fused numeric comparison → a bit-packed `BoolArray`.

    Carries **both** kernels of the operator: `K` for fixed-width lanes and `S`
    for strings. `NumericCompareKernel`'s own docstring is explicit that this
    pairing belongs here — "which family `a < b` means is a question about the
    operands, and it belongs to whoever is interpreting the operator, not to the
    SIMD kernel" — which is why the kernel's former `comptime StringKernel` was
    removed.

    The fused lane only ever uses `K`: its operands are `NumericValue`, so `S` is
    named but never instantiated. It exists for the erased arm, where the dtype
    is not known until the operands materialize."""

    comptime OutType = BoolType
    comptime OutShape = max(Self.L.OutShape, Self.R.OutShape)
    # Compare in the promoted domain of BOTH operands — the same rule as
    # `NumericBinary`, so `a > b` and `a + b` never disagree about widening.
    comptime ArgType = promote[Self.L.OutType, Self.R.OutType]
    comptime NativeType = wider[Self.L.OutType.native, Self.R.OutType.native]
    comptime State = Pair[Self.L.State, Self.R.State]
    var l: Self.L
    var r: Self.R

    def prune(self, stats: PruneStats) raises -> Interval:
        """`P` is this operator read over intervals — see `kernels.interval`.

        This used to branch on `Self.K.name`, matching the SIMD kernel's
        *display* string against five literals. `Kernel.name` is documented as
        "for display and diagnostics, never dispatch", and the mismatch was not
        theoretical: renaming a kernel silently dropped through to
        `unknown()`, which is sound, so pruning switched itself off with no
        error and no failing test."""
        return Interval.truth(
            Self.P.apply(self.l.prune(stats), self.r.prune(stats))
        )

    def render(self) -> String:
        return String(
            Self.K.name, "(", self.l.render(), ", ", self.r.render(), ")"
        )

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    def state(self, batch: RecordBatch) raises -> Self.State:
        return Pair(self.l.state(batch), self.r.state(batch))

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        var a = self.l.lane[W](state.l, idx).cast[Self.ArgType.native]()
        var b = self.r.lane[W](state.r, idx).cast[Self.ArgType.native]()
        return Self.K.core[Self.ArgType.native, W](a, b)

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return Bitmap.intersect(self.l.validity(batch), self.r.validity(batch))

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree rather than re-deriving from `batch`.

        Without this the chain stops at the first composite: the driver asks
        the root, the root's default falls back to `validity(batch)`, and a
        breaker operand re-runs its whole kernel — which is the FU-7a cost this
        was meant to remove."""
        return Bitmap.intersect(
            self.l.state_validity(batch, state.l),
            self.r.state_validity(batch, state.r),
        )


comptime Lt = NumericCompare[LtKernel, LtInterval, _, _]
comptime Le = NumericCompare[LeKernel, LeInterval, _, _]
comptime Gt = NumericCompare[GtKernel, GtInterval, _, _]
comptime Ge = NumericCompare[GeKernel, GeInterval, _, _]
comptime Eq = NumericCompare[EqKernel, EqInterval, _, _]
comptime Ne = NumericCompare[NeKernel, NeInterval, _, _]


# ---------------------------------------------------------------------------
# Boolean logic — one fused lane over bit-packed masks (bitwise SIMD). Rather
# than materializing bool children, these stay in the bool lane:
# `(a < 3) & (b > 15)` is one fused pass. Compute lives in the boolean kernels.
# ---------------------------------------------------------------------------
@fieldwise_init
struct BoolBinary[
    K: BoolBinaryKernel, P: IntervalKernel, L: BoolValue, R: BoolValue
](BoolValue):
    """Fused `and`/`or`/`xor` over two bool masks."""

    comptime OutType = BoolType
    comptime OutShape = max(Self.L.OutShape, Self.R.OutShape)
    # size the SIMD width by the WIDER operand — a narrow one (e.g. an int32 bool
    # breaker) must not shrink W below what a wider sibling's load (int64) needs, or
    # `SIMD[int64, W]` overflows the register.
    comptime NativeType = wider[Self.L.NativeType, Self.R.NativeType]
    comptime State = Pair[Self.L.State, Self.R.State]
    var l: Self.L
    var r: Self.R

    def prune(self, stats: PruneStats) raises -> Interval:
        """`P` is this operator read over intervals — see `NumericCompare.prune`
        for why this is a kernel rather than a match on `Self.K.name`."""
        return Interval.truth(
            Self.P.apply(self.l.prune(stats), self.r.prune(stats))
        )

    def render(self) -> String:
        return String(
            Self.K.name, "(", self.l.render(), ", ", self.r.render(), ")"
        )

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    def state(self, batch: RecordBatch) raises -> Self.State:
        return Pair(self.l.state(batch), self.r.state(batch))

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        var a = self.l.lane[W](state.l, idx)
        var b = self.r.lane[W](state.r, idx)
        return Self.K.core[W](a, b)

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # The fused `lane` above already produces the correct DATA (plain
        # bitwise `a & b` / `a | b` / `a ^ b`, matching the kernels). Only the
        # 3-valued *validity* is data-dependent, so reuse the null-correct kernel
        # `apply` (`and_`/`or_` Kleene, `xor` two-valued) to derive it — but only
        # when an operand can carry nulls; when both are fully valid the result is
        # fully valid, keeping the all-valid fast path allocation-free.
        var lv = self.l.validity(batch)
        var rv = self.r.validity(batch)
        if not lv and not rv:
            return None
        else:
            var n = batch.num_rows()
            var la = into_array(self.l.execute(batch), n).as_bool().copy()
            var ra = into_array(self.r.execute(batch), n).as_bool().copy()
            var res = Self.K.apply(la, ra)
            if res.bitmap:
                return res.bitmap.value().copy()
            else:
                return None


@fieldwise_init
struct BoolUnary[K: BoolUnaryKernel, A: BoolValue](BoolValue):
    """Fused `not` over a bool mask."""

    comptime OutType = BoolType
    comptime OutShape = Self.A.OutShape
    comptime NativeType = Self.A.NativeType
    comptime State = Self.A.State
    var a: Self.A

    def render(self) -> String:
        return String(Self.K.name, "(", self.a.render(), ")")

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self.a.state(batch)

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        return Self.K.core[W](self.a.lane[W](state, idx))

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree — see `NumericBinary.state_validity`.
        """
        return self.a.state_validity(batch, state)


comptime And = BoolBinary[AndKernel, AndInterval, _, _]
comptime Or = BoolBinary[OrKernel, OrInterval, _, _]
comptime Xor = BoolBinary[XorKernel, XorInterval, _, _]
comptime Not = BoolUnary[NotKernel, _]


@fieldwise_init
struct BoolReduce[K: BoolReduceKernel, A: BoolValue](BoolValue):
    """Fold a bool column to a scalar bool (`any`/`all`) — a scalar bool breaker;
    once folded it splats.

    KNOWN LIMITATION: fusing this splat directly under bool logic *beside a wider
    numeric load* (e.g. `all(mask) & (int64_col > 0)`) currently trips a Mojo
    backend codegen crash ("failed to run the pass manager"). Standalone `any`/`all`
    and same-width compositions are fine. Follow-up when the backend is fixed.
    """

    comptime OutType = BoolType
    comptime OutShape = 0
    comptime NativeType = DType.int32
    comptime State = Bool
    """The folded scalar — the lane splats it, like a literal's."""
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        var arr = (
            into_array(self.a.execute(batch), batch.num_rows()).as_bool().copy()
        )
        return Self.K.reduce(arr)

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return BoolScalar(self.state(batch)).to_dyn()

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        return SIMD[DType.bool, W](state)


comptime Any = BoolReduce[AnyKernel, _]
comptime All = BoolReduce[AllKernel, _]


# ---------------------------------------------------------------------------
# Unary predicates. `is_nan`/`is_inf` fuse — a per-lane SIMD predicate over a float
# operand. `is_null`/`not_null` read only validity
# (no value lane), so they're bool breakers over any family. Compute in the kernels.
# ---------------------------------------------------------------------------
@fieldwise_init
struct NumericPredicate[K: ValuePredicateKernel, A: NumericValue](BoolValue):
    """Fused `is_nan`/`is_inf` — a per-lane SIMD predicate over a numeric operand.
    """

    comptime OutType = BoolType
    comptime OutShape = Self.A.OutShape
    comptime NativeType = Self.A.OutType.native
    comptime State = Self.A.State
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self.a.state(batch)

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        return Self.K.core[Self.NativeType, W](self.a.lane[W](state, idx))

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree — see `NumericBinary.state_validity`.
        """
        return self.a.state_validity(batch, state)


@fieldwise_init
struct NullPredicate[K: UnaryPredicateKernel, A: Value](BoolValue):
    """`is_null`/`not_null` — reads the operand's validity (any family), so a bool
    breaker: materialize the operand, run the kernel into a `BoolArray`, load it.
    """

    comptime OutType = BoolType
    comptime OutShape = 1
    """Always columnar. `state` runs `into_array(..., batch.num_rows())`, so this
    node materializes a length-N `BoolArray` whatever its operand's shape was —
    propagating `Self.A.OutShape` claimed a scalar result it never produced."""
    comptime NativeType = DType.int32  # lane width for the bit-pack driver
    comptime State = BoolArray
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return Self.K.apply(arr)

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        return state.values().load[W](idx)


comptime IsNan = NumericPredicate[IsNanKernel, _]
comptime IsInf = NumericPredicate[IsInfKernel, _]
comptime IsNull = NullPredicate[IsNullKernel, _]
comptime NotNull = NullPredicate[NotNullKernel, _]


# ---------------------------------------------------------------------------
# Cross-family casts. Fixed-width -> fixed-width fuses (num <-> bool via a per-lane
# kernel `core`). String parses (string -> num/bool) have
# no value lane, so they're breakers. All compute lives in `kernels.cast`.
# (Currently only `string` operands, not `large_string`.)
# ---------------------------------------------------------------------------
@fieldwise_init
struct NumToBool[A: NumericValue](BoolValue):
    """Fused numeric -> bool (`x != 0`)."""

    comptime OutType = BoolType
    comptime OutShape = Self.A.OutShape
    comptime NativeType = Self.A.OutType.native
    comptime State = Self.A.State
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self.a.state(batch)

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        return NumToBoolKernel.core[Self.NativeType, W](
            self.a.lane[W](state, idx)
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree — see `NumericBinary.state_validity`.
        """
        return self.a.state_validity(batch, state)


@fieldwise_init
struct BoolToNum[To: NumericType, A: BoolValue](NumericValue):
    """Fused bool -> numeric (`True->1, False->0`)."""

    comptime OutType = Self.To
    comptime OutShape = Self.A.OutShape
    comptime State = Self.A.State
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self.a.state(batch)

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return BoolToNumKernel.core[Self.OutType.native, W](
            self.a.lane[W](state, idx)
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree — see `NumericBinary.state_validity`.
        """
        return self.a.state_validity(batch, state)


@fieldwise_init
struct StringToNum[To: NumericType, A: StringValue](NumericValue):
    """Parse string -> numeric (nulling on unparseable). No value lane, so a breaker:
    parse the whole column once via the kernel, then load per lane."""

    comptime OutType = Self.To
    comptime OutShape = 1
    comptime State = PrimitiveArray[Self.To]
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        var s = (
            into_array(self.a.execute(batch), batch.num_rows())
            .as_string()
            .copy()
        )
        return StringToNumKernel.apply[StringType, Self.To, False](s)

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # A parse failure is a null the *input* does not have ("x" -> null), so
        # validity comes from the parsed column, not from `a`. Inheriting the
        # all-valid default made `to_int(s) + 1` yield 0 where it should be null.
        #
        # Re-runs the parse. Only reached when this node is not the one being
        # driven; a fused parent takes `state_validity` below, which reads the
        # bitmap off the state it already has.
        return self.state(batch).to_data().owned_validity()

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Read off the materialized result rather than re-running the kernel —
        see the trait's `state_validity`."""
        return state.to_data().owned_validity()

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return state.values().load[W](idx)


@fieldwise_init
struct StringToBool[A: StringValue](BoolValue):
    """Parse string -> bool (`"true"`/`"false"`/`"1"`/`"0"`). A bool breaker."""

    comptime OutType = BoolType
    comptime OutShape = 1
    comptime NativeType = DType.int32
    comptime State = BoolArray
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        var s = (
            into_array(self.a.execute(batch), batch.num_rows())
            .as_string()
            .copy()
        )
        return StringToBoolKernel.apply[StringType, False](s)

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        return state.values().load[W](idx)


# ---------------------------------------------------------------------------
# StringValue — the elementwise string strategy. No W-wide lane: `core(idx)`
# yields one row's `String`, and the driver appends them into a builder, so a
# concat chain fuses without materializing intermediate string arrays.
# ---------------------------------------------------------------------------
trait StringValue(Value):
    comptime OutShape: Int  # 0 scalar, 1 columnar
    comptime OutType: StringLikeType

    comptime State: Copyable & Deinitable
    """Resolved once per pass — see `NumericValue.State`."""

    def state(self, batch: RecordBatch) raises -> Self.State:
        ...

    @always_inline
    def lane(self, state: Self.State, idx: Int) -> String:
        """One row. Variable-width UTF-8 has no W-wide lane, so this is the
        elementwise counterpart of the SIMD families' `lane[W]`."""
        ...

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity, given the state the driver already has.

        Defaults to `validity(batch)`: most nodes derive validity from their
        operands and have no use for the state. A **breaker** whose `State` is
        its materialized result overrides this to read the bitmap straight off
        that array. Without the hook the driver's two calls — `state()` and
        `validity()` — are independent, and for `coalesce`, `nullif` and
        `case_when` each one ran the whole selection kernel, so every fused pass
        over them did the work twice (FU-7a)."""
        return self.validity(batch)

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return _drive_string(self, batch, self.state(batch))

    # --- fluent surface: maps, predicates, length, concat -------------------
    def length(self) -> StringLength[Self]:
        return StringLength(self.copy())

    def upper(self) -> Upper[Self]:
        return Upper(self.copy())

    def lower(self) -> Lower[Self]:
        return Lower(self.copy())

    def strip(self) -> Strip[Self]:
        return Strip(self.copy())

    def lstrip(self) -> LStrip[Self]:
        return LStrip(self.copy())

    def rstrip(self) -> RStrip[Self]:
        return RStrip(self.copy())

    def reverse(self) -> Reverse[Self]:
        return Reverse(self.copy())

    def capitalize(self) -> Capitalize[Self]:
        return Capitalize(self.copy())

    def __add__[Rhs: StringValue](self, o: Rhs) -> Concat[Self, Rhs]:
        return Concat(self.copy(), o.copy())

    def startswith[Rhs: StringValue](self, o: Rhs) -> StartsWith[Self, Rhs]:
        return StartsWith(self.copy(), o.copy())

    def endswith[Rhs: StringValue](self, o: Rhs) -> EndsWith[Self, Rhs]:
        return EndsWith(self.copy(), o.copy())

    def contains[Rhs: StringValue](self, o: Rhs) -> StrContains[Self, Rhs]:
        return StrContains(self.copy(), o.copy())

    def __eq__[Rhs: StringValue](self, o: Rhs) -> StrEq[Self, Rhs]:
        return StrEq(self.copy(), o.copy())

    def __ne__[Rhs: StringValue](self, o: Rhs) -> StrNe[Self, Rhs]:
        return StrNe(self.copy(), o.copy())

    def __lt__[Rhs: StringValue](self, o: Rhs) -> StrLt[Self, Rhs]:
        return StrLt(self.copy(), o.copy())

    def __le__[Rhs: StringValue](self, o: Rhs) -> StrLe[Self, Rhs]:
        return StrLe(self.copy(), o.copy())

    def __gt__[Rhs: StringValue](self, o: Rhs) -> StrGt[Self, Rhs]:
        return StrGt(self.copy(), o.copy())

    def __ge__[Rhs: StringValue](self, o: Rhs) -> StrGe[Self, Rhs]:
        return StrGe(self.copy(), o.copy())

    def like[Rhs: StringValue](self, o: Rhs) -> Like[Self, Rhs]:
        return Like(self.copy(), o.copy())

    def ilike[Rhs: StringValue](self, o: Rhs) -> ILike[Self, Rhs]:
        return ILike(self.copy(), o.copy())

    # --- aggregates (marrow.expr.aggregates) --------------------------------

    def min(self) -> AggExpr:
        return AggExpr.of[StringMinMax[MinOp, Self.OutType]](self.copy())

    def max(self) -> AggExpr:
        return AggExpr.of[StringMinMax[MaxOp, Self.OutType]](self.copy())

    def count(self) -> AggExpr:
        return AggExpr.of[CountAgg](self.copy())


struct StringColumn[T: StringLikeType](StringValue):
    """A string column, resolved by name against `batch.schema` **once per
    pass** — `state()` does the lookup, `lane()` only reads a row."""

    comptime OutType = Self.T
    comptime OutShape = 1
    comptime State = BinaryLikeArray[Self.T]
    var _name: String

    def referenced_columns(self) -> List[String]:
        return [self._name.copy()]

    def bound_column(self, schema: Schema) raises -> Int:
        var i = schema.get_field_index(self._name)
        if i == -1:
            raise Error("column '", self._name, "' not found")
        return i

    def prune(self, stats: PruneStats) raises -> Interval:
        """A string column reports its bounds like any other.

        These two were on `NumericColumn` alone, so in the fused lane a string
        column could not be a join key and a string predicate pruned nothing —
        while the runtime lane, which keys on `_tag == "column"` regardless of
        dtype, pruned it. `Interval.compare` has ordered strings the whole
        time; only these overrides were missing."""
        var iv = stats.by_name(self._name)
        return Interval.bounds(iv[0].copy(), iv[1].copy())

    def __init__(out self, var name: String):
        self._name = name^

    def state(self, batch: RecordBatch) raises -> Self.State:
        # `RecordBatch.column(name)` owns the missing-name diagnostic — see
        # `NumericColumn.state`.
        return batch.column(self._name).as_type[Self.State]().copy()

    @always_inline
    def lane(self, state: Self.State, idx: Int) -> String:
        return String(state.unsafe_get(UInt(idx)))

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return batch.column(self._name).copy()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return batch.column(self._name).to_data().owned_validity()

    def name(self) -> String:
        return self._name.copy()


@fieldwise_init
struct StringLiteral[T: StringLikeType](StringValue):
    """A string constant, broadcast into every row."""

    comptime OutType = Self.T
    comptime OutShape = 0
    comptime State = NoneType
    var _value: String

    def prune(self, stats: PruneStats) raises -> Interval:
        var v = StringScalar(self._value).to_dyn()
        return Interval.bounds(Optional(v.copy()), Optional(v^))

    def referenced_columns(self) -> List[String]:
        return List[String]()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return NoneType()

    @always_inline
    def lane(self, state: Self.State, idx: Int) -> String:
        return self._value.copy()


@fieldwise_init
struct Concat[L: StringValue, R: StringValue](StringValue):
    """Fused elementwise concatenation — `col || "a" || "b"` builds each row once,
    no intermediate `StringArray` for `col || "a"`."""

    comptime OutType = Self.L.OutType
    comptime OutShape = max(Self.L.OutShape, Self.R.OutShape)
    comptime State = Pair[Self.L.State, Self.R.State]
    var l: Self.L
    var r: Self.R

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    def state(self, batch: RecordBatch) raises -> Self.State:
        return Pair(self.l.state(batch), self.r.state(batch))

    @always_inline
    def lane(self, state: Self.State, idx: Int) -> String:
        return ConcatKernel.combine(
            self.l.lane(state.l, idx),
            self.r.lane(state.r, idx),
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # a null operand poisons the row, as it does in `ConcatKernel`
        return Bitmap.intersect(self.l.validity(batch), self.r.validity(batch))

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree rather than re-deriving from `batch`.

        Without this the chain stops at the first composite: the driver asks
        the root, the root's default falls back to `validity(batch)`, and a
        breaker operand re-runs its whole kernel — which is the FU-7a cost this
        was meant to remove."""
        return Bitmap.intersect(
            self.l.state_validity(batch, state.l),
            self.r.state_validity(batch, state.r),
        )


@fieldwise_init
struct StringUnary[K: StringMapKernel, A: StringValue](StringValue):
    """Fused elementwise `string -> string` (`upper`/`lower`/`strip`/…). Composes in
    one builder pass with concat: `upper(col) || "!"` never materializes `upper(col)`.
    The transform lives in the kernel."""

    comptime OutType = Self.A.OutType
    comptime OutShape = Self.A.OutShape
    comptime State = Self.A.State
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self.a.state(batch)

    @always_inline
    def lane(self, state: Self.State, idx: Int) -> String:
        # bound to a local first: a `StringSlice` over a temporary `String`
        # would dangle the moment the temporary is destroyed.
        var s = self.a.lane(state, idx)
        return Self.K.transform(StringSlice(s))

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # a map transforms values, never validity — `upper(null)` is null
        return self.a.validity(batch)

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Propagate down the state tree — see `NumericBinary.state_validity`.
        """
        return self.a.state_validity(batch, state)


comptime Upper = StringUnary[UpperKernel, _]
comptime Lower = StringUnary[LowerKernel, _]
comptime Strip = StringUnary[StripKernel, _]
comptime LStrip = StringUnary[LStripKernel, _]
comptime RStrip = StringUnary[RStripKernel, _]
comptime Reverse = StringUnary[ReverseKernel, _]
comptime Capitalize = StringUnary[CapitalizeKernel, _]


# ---------------------------------------------------------------------------
# Casts *to* string — a materialized string result, so a string breaker: `state`
# folds the operand to a string column via `kernels.cast` and `lane` reads that
# column per row, so `cast(x, string) || "!"` still fuses in the elementwise
# builder pass. Compute lives in the cast kernels.
# ---------------------------------------------------------------------------
@fieldwise_init
struct NumToString[To: StringLikeType, A: NumericValue](StringValue):
    """Format numeric -> string."""

    comptime OutType = Self.To
    comptime OutShape = 1
    comptime State = BinaryLikeArray[Self.To]
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return (
            NumToStringKernel.dispatch(arr, Self.To())
            .as_type[Self.State]()
            .copy()
        )

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane(self, state: Self.State, idx: Int) -> String:
        return String(state.unsafe_get(UInt(idx)))


@fieldwise_init
struct BoolToString[To: StringLikeType, A: BoolValue](StringValue):
    """`True`/`False` -> string."""

    comptime OutType = Self.To
    comptime OutShape = 1
    comptime State = BinaryLikeArray[Self.To]
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return (
            BoolToStringKernel.dispatch(arr, Self.To())
            .as_type[Self.State]()
            .copy()
        )

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane(self, state: Self.State, idx: Int) -> String:
        return String(state.unsafe_get(UInt(idx)))


@fieldwise_init
struct StringToString[To: StringLikeType, A: StringValue](StringValue):
    """Cast between string containers (`string` <-> `large_string`)."""

    comptime OutType = Self.To
    comptime OutShape = 1
    comptime State = BinaryLikeArray[Self.To]
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return (
            StringToStringKernel.dispatch(arr, Self.To(), False)
            .as_type[Self.State]()
            .copy()
        )

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane(self, state: Self.State, idx: Int) -> String:
        return String(state.unsafe_get(UInt(idx)))


# ---------------------------------------------------------------------------
# String predicates — `string × string -> bool`. Variable-width comparison has no
# SIMD lane, so this is a bool *breaker*: `state` materializes both string stages
# and runs the kernel into a `BoolArray`; `lane` reads that mask, so a predicate
# still fuses under boolean logic. The comparison lives in the kernel.
# ---------------------------------------------------------------------------
@fieldwise_init
struct StringPredicate[
    K: StringPredicateKernel,
    P: IntervalKernel,
    L: StringValue,
    R: StringValue,
](BoolValue):
    comptime OutType = BoolType
    comptime OutShape = max(Self.L.OutShape, Self.R.OutShape)
    comptime NativeType = DType.int32  # lane width for the bit-pack driver
    comptime State = BoolArray
    var l: Self.L
    var r: Self.R

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    def prune(self, stats: PruneStats) raises -> Interval:
        """`P` reads this operator over the operands' string bounds.

        Ordering predicates carry a real rule; `startswith`/`like` and friends
        pass `NeInterval`, whose answer is always "maybe" — the conservative
        default they had implicitly by inheriting `Value.prune`."""
        return Interval.truth(
            Self.P.apply(self.l.prune(stats), self.r.prune(stats))
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # A string predicate is null exactly where an operand is: the kernel
        # already ANDs the operand bitmaps, and `lane` reads only the data bits,
        # so the node has to carry the validity itself.
        return Bitmap.intersect(self.l.validity(batch), self.r.validity(batch))

    def state(self, batch: RecordBatch) raises -> Self.State:
        var n = batch.num_rows()
        var la = into_array(self.l.execute(batch), n).as_string().copy()
        comptime if Self.R.OutShape == 0:
            # A scalar right operand: evaluate it once. `into_array` would
            # broadcast it into n copies of the same string, and the array x
            # array kernel would then re-read — and for LIKE, recompile — that
            # constant on every row.
            #
            # `apply_scalar`'s validity comes from the left operand alone
            # (`Bitmap.intersect(l, None)` reduces to exactly that), which is
            # only correct because no `OutShape == 0` string node can be null.
            # `StringLiteral` is the only leaf with `OutShape == 0`, and it
            # holds a plain `String` with no validity flag. The two composites
            # that could also reach `OutShape == 0` forward the property
            # rather than break it: `Concat.OutShape` is
            # `max(L.OutShape, R.OutShape)`, so it is 0 only when both operands
            # are themselves all-literal, and its `validity` intersects theirs
            # (both `None`); `StringUnary.OutShape` is `A.OutShape`, so it is 0
            # only when its single operand is, and its `validity` just forwards
            # `a.validity(batch)` unchanged. So reaching `OutShape == 0` at all,
            # through any nesting of these three, forces every leaf to be a
            # `StringLiteral`, and the intersection/forwarding chain reduces to
            # `None` all the way up. If a nullable string literal is ever
            # added, this assumption breaks and needs revisiting here.
            var rst = self.r.state(batch)
            var pat = self.r.lane(rst, 0)
            return Self.K.apply_scalar(la, pat)
        else:
            var ra = into_array(self.r.execute(batch), n).as_string().copy()
            return Self.K.apply(la, ra)

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        return state.values().load[W](idx)


comptime StartsWith = StringPredicate[StartsWithKernel, NeInterval, _, _]
comptime EndsWith = StringPredicate[EndsWithKernel, NeInterval, _, _]
comptime StrContains = StringPredicate[ContainsKernel, NeInterval, _, _]
comptime StrEq = StringPredicate[StringEqKernel, EqInterval, _, _]
comptime StrNe = StringPredicate[StringNeKernel, NeInterval, _, _]
# SQL LIKE / ILIKE — same breaker shape (`LikeKernel`/`ILikeKernel` are
# `StringPredicateKernel`s), so they slot straight into `StringPredicate`.
comptime Like = StringPredicate[LikeKernel, NeInterval, _, _]
comptime ILike = StringPredicate[ILikeKernel, NeInterval, _, _]


# String ordering comparisons — `string < <= > >=` -> bool. The compare kernels
# name their string counterpart, so ordering is the same node as every other
# string predicate; only the kernel differs.
comptime StrLt = StringPredicate[StringLtKernel, LtInterval, _, _]
comptime StrLe = StringPredicate[StringLeKernel, LeInterval, _, _]
comptime StrGt = StringPredicate[StringGtKernel, GtInterval, _, _]
comptime StrGe = StringPredicate[StringGeKernel, GeInterval, _, _]


# ---------------------------------------------------------------------------
# is_in — SQL `x IN (...)`. A bool breaker over any value family: `state`
# hashes the captured value-set once and probes the operand column (reusing
# `kernels.membership.IsInKernel`), then `lane` loads the mask. The output is
# always valid (PyArrow `is_in` never nulls), so validity defaults to `None`.
# ---------------------------------------------------------------------------
@fieldwise_init
struct IsIn[A: Value](BoolValue):
    comptime OutType = BoolType
    comptime OutShape = 1
    comptime NativeType = DType.int32
    comptime State = BoolArray
    var a: Self.A
    var _value_set: DynArray

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return IsInKernel.dispatch(arr, self._value_set.copy())

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        return state.values().load[W](idx)


# ---------------------------------------------------------------------------
# Strategy transition (string -> numeric) — modelled as a plain breaker. The two
# strategies don't compose, so the string (elementwise) stage materializes; the
# `LengthKernel` folds it to the int32 length column (vectorized offset subtraction,
# handling string / large_string). `lane` then just loads that column, so the
# arithmetic above fuses — exactly like a window: `length(s) + 1` is one numeric pass
# over the materialized lengths. Same shape as every other breaker.
# ---------------------------------------------------------------------------
@fieldwise_init
struct StringLength[A: StringValue](NumericValue):
    """Byte length of a string value → int32. `state` materializes the string
    stage and folds it to the length column via `LengthKernel`; `lane` loads
    that column per chunk."""

    comptime OutType = Int32Type
    comptime OutShape = 1
    comptime State = Int32Array
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # a null string has a null length — validity passes through unchanged.
        return self.a.validity(batch)

    def state(self, batch: RecordBatch) raises -> Self.State:
        var s = into_array(self.a.execute(batch), batch.num_rows())
        return LengthKernel.dispatch(s).as_int32().copy()

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return state.values().load[W](idx)


# ---------------------------------------------------------------------------
# Pipeline breakers — cross-row `Value`s that cut the tree into stages. Each
# materializes its operand in `state()`, then acts as a fused leaf: a scalar
# reduction splats its state, a columnar window loads from it. Nesting needs no
# coordination — a breaker's state is its own value, not a slot in a shared
# context that an outer walk could perturb.
struct AggExpr(Copyable, Movable, Writable):
    """An aggregate over an input expression, under an output column name.

    Two ways in:

    - **fused / AOT** — ``col("amount", int64).sum()``. The ``Aggregation`` is
      computed from the node's own ``OutType``, so nothing is interpreted: the
      plan points straight at ``AggState[SumKernel, Int64Type]``.
    - **dynamic** — ``col("amount").sum()`` on a ``DynValue``, which carries the
      function's *name* until the plan is built and the input's dtype is known.

    ``input_for(schema)`` hands back the input expression ready to execute:
    the dynamic lane resolves its column names against the schema first (a
    fused node addresses columns by name at execute time and needs nothing)."""

    var out_name: String
    """The output column name — the aggregate's own name unless aliased."""

    var input: BoxedValue
    var _unresolved: Optional[DynValue]
    var _func: String
    var _of: Optional[def(DynType) thin raises -> AggFunc]

    @implicit
    def __init__[
        K: AggKernel, In: NumericValue
    ](out self, var reduction: Reduction[K, In]):
        """From a fused reduction: ``col("amount", int64).sum()`` is a scalar
        `Reduction` inside an expression and this aggregate in a `GROUP BY` —
        the same node, read two ways.

        When the operand is typed, the kernel and its `OutType` name the
        `Aggregation` outright and nothing is resolved later. When it is erased
        there is no `OutType` to name one with, so the aggregate is carried by
        name and resolved against the column's dtype at plan build — the same
        split as `Reduction.alias`."""
        self = AggExpr.of[K.Grouped[In.OutType]](reduction.a.copy())

    @implicit
    def __init__(out self, var agg: DynAgg):
        """From the dynamic lane: keep the name, resolve it at plan build."""
        var out_name = agg.out_name.copy()
        if not out_name:
            out_name = agg.func.copy()
        self.out_name = out_name^
        self.input = agg.input.copy()
        self._unresolved = agg.input.copy()
        self._func = agg.func.copy()
        self._of = None

    def __init__(
        out self,
        *,
        var out_name: String,
        var input: BoxedValue,
        of: def(DynType) thin raises -> AggFunc,
    ):
        self.out_name = out_name^
        self.input = input^
        self._unresolved = None
        self._func = String()
        self._of = of

    @staticmethod
    def of[A: Aggregation, In: Value](var input: In) -> AggExpr:
        """From the fused lane: the aggregation is named, not looked up."""
        return AggExpr(
            out_name=String(A.name),
            input=BoxedValue(input^),
            of=AggFunc.of[A],
        )

    def alias(self, var name: String) -> AggExpr:
        """Name this aggregate's output column."""
        var out = self.copy()
        out.out_name = name^
        return out^

    def input_for(self, schema: Schema) raises -> BoxedValue:
        """The input expression, ready to execute against ``schema``."""
        if self._unresolved:
            return BoxedValue(self._unresolved.value().resolve_names(schema))
        return self.input.copy()

    def resolve(self, in_dtype: DynType) raises -> AggFunc:
        """The aggregate, bound to the dtype its input turned out to have."""
        if self._of:
            return self._of.value()(in_dtype)
        return AggFunc(self._func, in_dtype)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._func if self._func else self.out_name, "(")
        self.input.write_to(writer)
        writer.write(")")


# ---------------------------------------------------------------------------
@fieldwise_init
struct Reduction[K: AggKernel, A: NumericValue](NumericValue):
    """Whole-array reduction → a scalar. Output dtype is the kernel's accumulator
    algebra `K.AccType[A.OutType]` (sum widens, mean → float64, min/max keep it).
    """

    comptime OutType = Self.K.AccType[Self.A.OutType]
    comptime OutShape = 0
    comptime State = PrimitiveScalar[Self.OutType]
    """The folded scalar, kept whole rather than unwrapped to its value: an
    empty or all-null input folds to a *null* scalar, and `materialize` has to
    hand that null back."""
    var a: Self.A

    def alias(self, var name: String) -> AggExpr:
        """Name this reduction's output column, making it a GROUP BY aggregate.

        `x.sum()` is a `Reduction` in both lanes — a whole-array fold — and
        `.alias(...)` is what turns it into an aggregate expression. Putting it
        here rather than on the box is what keeps one spelling: the box cannot
        override `sum` (it is defaulted in exactly one family, so a second
        candidate is ambiguous rather than an override), and it does not need
        to."""
        return AggExpr.of[NumericAgg[Self.K, Self.A.OutType]](
            self.a.copy()
        ).alias(name^)

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        # The operand's dtype is `A.OutType`, so the reduce is the fully typed
        # one — no dtype dispatch and nothing erased.
        var arg = into_array(self.a.execute(batch), batch.num_rows())
        return Self.K.reduce(arg.as_primitive[Self.A.OutType]())

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return SIMD[Self.OutType.native, W](state.value())


comptime Sum = Reduction[SumKernel, _]
comptime Mean = Reduction[MeanKernel, _]
comptime Min = Reduction[MinKernel, _]
comptime Max = Reduction[MaxKernel, _]
comptime Product = Reduction[ProductKernel, _]
comptime Count = Reduction[CountKernel, _]


@fieldwise_init
struct FrameBound(Copyable, ImplicitlyCopyable, Movable):
    var kind: UInt8
    var offset: Int64


@fieldwise_init
struct WindowSpec(Copyable, Movable):
    """Partition/order/frame — the toy carries frame bounds only."""

    var start: FrameBound
    var end: FrameBound


trait WindowKernel:
    comptime name: String
    comptime OutType: NumericType

    @staticmethod
    def evaluate_all(values: DynArray) raises -> DynArray:
        ...


struct RowNumberKernel(WindowKernel):
    comptime name = "row_number"
    comptime OutType = Int64Type

    @staticmethod
    def evaluate_all(values: DynArray) raises -> DynArray:
        # frame-independent (DataFusion `evaluate_all`): row_number = 1..n.
        var n = len(values)
        var b = Int64Builder(n)
        for i in range(n):
            b.append(Int64(i + 1))
        return b.finish().to_dyn()


@fieldwise_init
struct WindowFunction[Func: WindowKernel, A: Value](NumericValue):
    """`func.over(spec)` → a columnar breaker. `state` materializes the whole
    output column; `lane` then loads it per chunk like a column."""

    comptime OutType = Self.Func.OutType
    comptime OutShape = 1
    comptime State = PrimitiveArray[Self.OutType]
    var a: Self.A
    var spec: WindowSpec

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def state(self, batch: RecordBatch) raises -> Self.State:
        var v = into_array(self.a.execute(batch), batch.num_rows())
        return Self.Func.evaluate_all(v).as_primitive[Self.OutType]().copy()

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return state.values().load[W](idx)


comptime RowNumber = WindowFunction[RowNumberKernel, _]


# ---------------------------------------------------------------------------
# Conditional / null-handling — `coalesce`, `nullif`, `case_when` over the numeric
# family. Value selection is data-dependent (which candidate per row), so these are
# numeric breakers: `state` runs the selection kernel once into a column, then
# `lane` loads it. Their result validity is data-dependent too (coalesce nulls
# only where every operand is null; nullif adds nulls on equality; case_when depends
# on the branch), so `validity` re-runs the kernel and reads the materialized
# bitmap — the one case where operand validities alone can't reconstruct the result.
# ---------------------------------------------------------------------------
# Conditional binary breakers — `coalesce(l, r)` and `nullif(l, r)` differ only
# in the kernel they call, so one generic node parameterized by a tiny op struct
# covers both (mirrors `Reduction[K]` / `TemporalExtract[K]`). Both are
# data-dependent-validity breakers over two same-dtype numeric operands.
@fieldwise_init
struct ConditionalBinary[
    K: BinaryConditionalKernel, L: NumericValue, R: NumericValue
](NumericValue):
    """`coalesce`/`nullif` over two same-dtype numeric operands; `K` picks the
    kernel. `state` materializes the result once; `lane` loads it."""

    comptime OutType = Self.L.OutType
    comptime OutShape = 1
    comptime State = PrimitiveArray[Self.OutType]
    var l: Self.L
    var r: Self.R

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    def _result(self, batch: RecordBatch) raises -> DynArray:
        var n = batch.num_rows()
        var la = into_array(self.l.execute(batch), n)
        var ra = into_array(self.r.execute(batch), n)
        return Self.K.combine(la, ra)

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        """Asked only when this node is *not* the one being driven — a fused
        parent takes `state_validity` instead, which costs nothing extra."""
        return self._result(batch).to_data().owned_validity()

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Read off the materialized result rather than re-running the kernel —
        see the trait's `state_validity`."""
        return state.to_data().owned_validity()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self._result(batch).as_primitive[Self.OutType]().copy()

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self._result(batch)

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return state.values().load[W](idx)


comptime Coalesce = ConditionalBinary[CoalesceKernel, _, _]
comptime Nullif = ConditionalBinary[NullifKernel, _, _]


@fieldwise_init
struct CaseWhen[C: BoolValue, T: NumericValue, E: NumericValue](NumericValue):
    """Single-branch `CASE WHEN cond THEN then ELSE otherwise` over numeric
    values — `then`/`otherwise` share a dtype. A null condition counts as false
    (Arrow semantics), so `otherwise` is chosen there."""

    comptime OutType = Self.T.OutType
    comptime OutShape = 1
    comptime State = PrimitiveArray[Self.OutType]
    var cond: Self.C
    var then: Self.T
    var otherwise: Self.E

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            _union_columns(
                self.cond.referenced_columns(),
                self.then.referenced_columns(),
            ),
            self.otherwise.referenced_columns(),
        )

    def _result(self, batch: RecordBatch) raises -> DynArray:
        var n = batch.num_rows()
        var ca = into_array(self.cond.execute(batch), n).as_bool().copy()
        var conditions = List[BoolArray]()
        conditions.append(ca^)
        var values = List[DynArray]()
        values.append(into_array(self.then.execute(batch), n))
        var else_ = Optional[DynArray](
            into_array(self.otherwise.execute(batch), n)
        )
        return case_when_kernel(conditions, values, else_^)

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        """Asked only when this node is *not* the one being driven — a fused
        parent takes `state_validity` instead, which costs nothing extra."""
        return self._result(batch).to_data().owned_validity()

    def state_validity(
        self, batch: RecordBatch, state: Self.State
    ) raises -> Optional[Bitmap[mut=False]]:
        """Read off the materialized result rather than re-running the kernel —
        see the trait's `state_validity`."""
        return state.to_data().owned_validity()

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self._result(batch).as_primitive[Self.OutType]().copy()

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self._result(batch)

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return state.values().load[W](idx)


# ---------------------------------------------------------------------------
# Temporal family — a temporal column is materialize-only (like a list column):
# no fused lane, `materialize` hands back the array. Component extraction
# (`year`/`month`/…/`day_of_year`) is a breaker → int32 `NumericValue` (same shape
# as `StringLength`); `date_trunc` floors to a unit boundary, staying temporal.
# All compute lives in `kernels.temporal`.
# ---------------------------------------------------------------------------
trait TemporalValue(Value):
    comptime OutType: TemporalType

    def year(self) -> TemporalExtract[YearKernel, Self]:
        return TemporalExtract[YearKernel](self.copy())

    def month(self) -> TemporalExtract[MonthKernel, Self]:
        return TemporalExtract[MonthKernel](self.copy())

    def day(self) -> TemporalExtract[DayKernel, Self]:
        return TemporalExtract[DayKernel](self.copy())

    def hour(self) -> TemporalExtract[HourKernel, Self]:
        return TemporalExtract[HourKernel](self.copy())

    def minute(self) -> TemporalExtract[MinuteKernel, Self]:
        return TemporalExtract[MinuteKernel](self.copy())

    def second(self) -> TemporalExtract[SecondKernel, Self]:
        return TemporalExtract[SecondKernel](self.copy())

    def day_of_week(self) -> TemporalExtract[DayOfWeekKernel, Self]:
        return TemporalExtract[DayOfWeekKernel](self.copy())

    def quarter(self) -> TemporalExtract[QuarterKernel, Self]:
        return TemporalExtract[QuarterKernel](self.copy())

    def day_of_year(self) -> TemporalExtract[DayOfYearKernel, Self]:
        return TemporalExtract[DayOfYearKernel](self.copy())

    def date_trunc(self, unit: String) raises -> DateTrunc[Self]:
        """Floor to *unit*. Parsed here, so the fused node carries a validated
        `CalendarUnit` and an unsupported spelling fails at construction."""
        return DateTrunc(self.copy(), CalendarUnit.parse(unit))

    # --- aggregates (marrow.expr.aggregates) --------------------------------

    def min(self) -> AggExpr:
        return AggExpr.of[TemporalMinMax[MinOp, Self.OutType]](self.copy())

    def max(self) -> AggExpr:
        return AggExpr.of[TemporalMinMax[MaxOp, Self.OutType]](self.copy())

    def count(self) -> AggExpr:
        return AggExpr.of[CountAgg](self.copy())


struct TemporalColumn[T: TemporalType](TemporalValue):
    """A temporal column, resolved by name. No fused lane — `materialize` hands
    back the column."""

    comptime OutType = Self.T
    comptime OutShape = 1
    var _name: String

    def referenced_columns(self) -> List[String]:
        return [self._name.copy()]

    def __init__(out self, var name: String):
        self._name = name^

    def materialize(self, batch: RecordBatch) raises -> Datum:
        # `RecordBatch.column(name)` owns the missing-name diagnostic — see
        # `NumericColumn.state`.
        return batch.column(self._name).copy()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return batch.column(self._name).to_data().owned_validity()

    def name(self) -> String:
        return self._name.copy()


@fieldwise_init
struct TemporalExtract[K: TemporalExtractKernel, A: TemporalValue](
    NumericValue
):
    """Extract a calendar/clock field from a temporal value → int32. A breaker,
    same shape as `StringLength`."""

    comptime OutType = Int32Type
    comptime OutShape = 1
    comptime State = Int32Array
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # a null temporal value has a null field — validity passes through.
        return self.a.validity(batch)

    def state(self, batch: RecordBatch) raises -> Self.State:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return Self.K.dispatch(arr).as_int32().copy()

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return state.values().load[W](idx)


@fieldwise_init
struct DateTrunc[A: TemporalValue](TemporalValue):
    """Floor a temporal value to a unit boundary (`second`/`minute`/`hour`/
    `day`), keeping the same temporal type. Materialize-only, like a column."""

    comptime OutType = Self.A.OutType
    comptime OutShape = 1
    var a: Self.A
    var _unit: CalendarUnit

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def materialize(self, batch: RecordBatch) raises -> Datum:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return DateTruncKernel.apply(arr, self._unit)


comptime Year = TemporalExtract[YearKernel, _]
comptime Month = TemporalExtract[MonthKernel, _]
comptime Day = TemporalExtract[DayKernel, _]
comptime Hour = TemporalExtract[HourKernel, _]
comptime Minute = TemporalExtract[MinuteKernel, _]
comptime Second = TemporalExtract[SecondKernel, _]
comptime DayOfWeek = TemporalExtract[DayOfWeekKernel, _]
comptime Quarter = TemporalExtract[QuarterKernel, _]
comptime DayOfYear = TemporalExtract[DayOfYearKernel, _]


# ---------------------------------------------------------------------------
# List family — nested, variable-length. Lists don't fuse, so a list column is a
# prepare-only `Value` (execute -> the list array). The useful ops produce
# fixed-width results and are breakers: `length` (-> int32, like StringLength) and
# `contains` (-> bool). Both delegate to kernels.nested.
# ---------------------------------------------------------------------------
trait ListValue(Value):
    comptime OutType: ListLikeType

    def length(self) -> ListLength[Self]:
        return ListLength(self.copy())

    def contains[E: NumericValue](self, elem: E) -> ListContains[Self, E]:
        return ListContains(self.copy(), elem.copy())


struct ListColumn[T: ListLikeType](ListValue):
    """A list column, resolved by name. No fused lane — `materialize` hands back
    the column."""

    comptime OutType = Self.T
    comptime OutShape = 1
    var _name: String

    def referenced_columns(self) -> List[String]:
        return [self._name.copy()]

    def __init__(out self, var name: String):
        self._name = name^

    def materialize(self, batch: RecordBatch) raises -> Datum:
        # `RecordBatch.column(name)` owns the missing-name diagnostic — see
        # `NumericColumn.state`.
        return batch.column(self._name).copy()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return batch.column(self._name).to_data().owned_validity()

    def name(self) -> String:
        return self._name.copy()


@fieldwise_init
struct ListLength[A: ListValue](NumericValue):
    """List element count -> int32. A breaker, same shape as `StringLength`."""

    comptime OutType = Int32Type
    comptime OutShape = 1
    comptime State = Int32Array
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # a null list has a null length — validity passes through unchanged.
        return self.a.validity(batch)

    def state(self, batch: RecordBatch) raises -> Self.State:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return ArrayLengthKernel.dispatch(arr).as_int32().copy()

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return state.values().load[W](idx)


@fieldwise_init
struct ListContains[A: ListValue, E: NumericValue](BoolValue):
    """Element-wise membership: `elem[i] in list[i]` -> bool. A breaker (a literal
    element broadcasts). Numeric element types."""

    comptime OutType = BoolType
    comptime OutShape = 1
    comptime NativeType = DType.int32
    comptime State = BoolArray
    var a: Self.A
    var elem: Self.E

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.a.referenced_columns(), self.elem.referenced_columns()
        )

    def state(self, batch: RecordBatch) raises -> Self.State:
        var n = batch.num_rows()
        var la = into_array(self.a.execute(batch), n)
        var ea = into_array(self.elem.execute(batch), n)
        return ArrayContainsKernel.dispatch(la, ea).as_bool().copy()

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return self.state(batch).to_dyn()

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[DType.bool, W]:
        return state.values().load[W](idx)


# ---------------------------------------------------------------------------
# NOTE: erasure lives in `relations.mojo` (`BoxedValue`) — because `execute`
# already returns a concrete `Datum`, it is a plain fn-pointer trampoline, and
# it belongs beside the operators that hold one.
# ---------------------------------------------------------------------------
# NOTE: `Table[T]` (the `t.a` schema-struct sugar over a plain struct of
# dtype-tag fields) is deferred — the parametric
# `comptime _dtype[name] = reflect[T].field[name].T` alias hits the documented
# `reflect` resolution bug (see schema.mojo). `col("a", int64)` is the working
# column-reference API in the meantime.


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def col[T: NumericType](var name: String, dtype: T) -> NumericColumn[T]:
    """Reference a numeric column by name — `col("a", int64)`."""
    return NumericColumn[T](name^)


def col[T: StringLikeType](var name: String, dtype: T) -> StringColumn[T]:
    """Reference a string column by name — `col("s", string)`."""
    return StringColumn[T](name^)


def col[T: ListLikeType](var name: String, dtype: T) -> ListColumn[T]:
    """Reference a list column by name — `col("l", list_(int64))`."""
    return ListColumn[T](name^)


def col[T: TemporalType](var name: String, dtype: T) -> TemporalColumn[T]:
    """Reference a temporal column by name — `col("ts", timestamp(second))`."""
    return TemporalColumn[T](name^)


def lit[T: NumericType](value: Int, dtype: T) -> NumericLiteral[T]:
    """An integral constant — `lit(10, int64)`."""
    return NumericLiteral[T](Scalar[T.native](value))


def lit[T: FloatingType](value: Float64, dtype: T) -> NumericLiteral[T]:
    """A fractional constant — `lit(3.5, float64)`.

    Without this overload the only spelling took an `Int`, so `lit(3.5,
    float64)` was unrepresentable: it truncated to 3."""
    return NumericLiteral[T](Scalar[T.native](value))


def lit(value: String) -> StringLiteral[StringType]:
    """A string constant — `lit("suffix")`. Same verb as the numeric ones; the
    argument type picks the literal."""
    return StringLiteral[StringType](value)


# ---------------------------------------------------------------------------
# AggExpr — one aggregate in a query
#
# `col("x").sum()` on a fused node produces one of these with its `Aggregation`
# already chosen; the same call on a `DynValue` produces a `DynAgg`, which
# converts into one and resolves its function name when the plan is built. Both
# arrive at the same `AggFunc`, which is why `aggregate(...)` takes a single
# list and does not care which lane each member came from.
# ---------------------------------------------------------------------------


def col(var name: String) -> DynValue:
    """Reference a column whose dtype is not known here — `col("a")`.

    Same verb as `col(name, dtype)`, one argument shorter, and that argument is
    the whole difference between the lanes: with a dtype the fused
    `NumericColumn[T]` leaf is built, without one the column's type is found on
    the batch and this is a runtime-lane node."""
    return DynValue.column(name^)


def lit[T: NumericType](value: Scalar[T.native]) -> DynValue:
    """A scalar constant for the runtime lane — `lit[Int64Type](3)`.

    A literal always knows its type where it is written; what is erased here is
    the *expression*, so the value goes in as a `DynScalar` payload."""
    return DynValue.literal(PrimitiveScalar[T](value))


def if_else(cond: DynValue, then_: DynValue, else_: DynValue) -> DynValue:
    """Element-wise conditional — the single-branch `CaseWhen`."""
    return DynValue.if_else(cond, then_, else_)


def coalesce(values: List[DynValue]) raises -> DynValue:
    """First non-null across N expressions.

    Folds the binary `Coalesce` node rather than introducing an n-ary one:
    `coalesce(a, b, c)` is `Coalesce(Coalesce(a, b), c)`, which is the same
    result because the operation is associative and null-propagating."""
    if len(values) == 0:
        raise Error("coalesce: needs at least one value")
    var acc = values[0].copy()
    for k in range(1, len(values)):
        acc = acc.coalesce(values[k])
    return acc^


def case_when(
    conditions: List[DynValue],
    values: List[DynValue],
    var else_: Optional[DynValue] = None,
) raises -> DynValue:
    """Multi-branch `CASE WHEN`, built by nesting the single-branch `CaseWhen`.

    `conditions[k]` selects `values[k]` for the first branch that is
    valid-and-true. Nesting right-to-left gives first-match-wins, which is what
    the interpreter's interleaved-args form computed."""
    if len(conditions) != len(values):
        raise Error("case_when: len(conditions) != len(values)")
    if len(conditions) == 0:
        raise Error("case_when: needs at least one branch")
    # No `else_` means "null where nothing matched". `CaseWhen` always has a
    # third operand, so the null is built from an existing node rather than a new
    # one: `Nullif(v, v)` is `v` with every element equal to itself removed — an
    # all-null column of the right dtype.
    var acc = else_.value().copy() if else_ else values[0].nullif(values[0])
    for k in range(len(conditions) - 1, -1, -1):
        acc = DynValue.if_else(conditions[k], values[k], acc)
    return acc^
