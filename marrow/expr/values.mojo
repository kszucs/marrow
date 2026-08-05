"""Expression execution — staged, strategy-pluggable fusion (see
`docs/architecture.md`).

Model
-----
- `execute(batch, ctx) -> Datum` is the **one universal verb** every node has.
  `Datum = Scalar | Array` (Arrow's Datum / DataFusion's ColumnarValue) is the
  strategy-agnostic wire format between stages.
- Fusion is a **pluggable strategy**, not a single primitive. A strategy = a
  composable per-element `core` + a driver that runs a whole same-strategy subtree
  in one pass:
    * `NumericValue` — vectorized: `vectorwise[W] -> SIMD[native, W]`, driver fills a `Buffer`.
    * `BoolValue`    — vectorized: `vectorwise[W] -> SIMD[bool, W]`, driver bit-packs a `Bitmap`.
    * `StringValue`  — elementwise: `elementwise(idx) -> String`, driver appends to a builder
      (variable-width UTF-8 has no W-wide lane, so `col || "a" || "b"` fuses one
      row at a time — no intermediate `StringArray`).
- **Pipeline breakers** — cross-row ops (`Reduction`, `WindowFunction`) that can't
  fuse in any strategy — cut the tree into a forest of fused stages. A breaker
  materializes its stage once in `prepare` into a shared `Context`, then behaves as
  an ordinary fused leaf: a scalar reduction reads the context and **splats** (like
  a literal), a columnar window reads it and **loads** (like a column). So the stage
  above still fuses through the single `NumericBinary` — there is no separate
  "materialized" binary.
- Expressions are **immutable**: all per-execute state lives in the `Context`.
  Breaker results are stored positionally; `prepare` (which fills them) and `core`
  (which reads them) visit breakers in the same DFS order, so a plain integer
  `slot` matches reads to writes — no per-lane keying in the hot loop. A breaker
  materializes its operand through a *fresh* sub-context, so nested breakers
  never perturb the outer slot order.
"""

from std.sys import bit_width_of
from std.builtin.rebind import downcast
from std.utils import Variant
from std.memory import ArcPointer

from ..schema import Schema
from ..tabular import RecordBatch
from ..arrays import (
    Array,
    DynArray,
    PrimitiveArray,
    Int32Array,
    BoolArray,
    BinaryLikeArray,
    StringArray,
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
from .pruning import PruneStats, PruneBound
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


# ---------------------------------------------------------------------------
# Context — per-execute shared scratch. Pipeline-breaker stage results live here,
# positionally: `prepare` appends them in DFS order and `core` reads them back in
# the same order via a `slot`. Keeping results here (not on the nodes) is what
# makes expressions immutable.
# ---------------------------------------------------------------------------
struct Context(Copyable, Movable):
    var _slots: List[Datum]

    def __init__(out self):
        self._slots = List[Datum]()

    def append(mut self, var d: Datum):
        self._slots.append(d^)

    def get(self, i: Int) -> Datum:
        # a `Datum` copy is a ref-count bump (no heap); cheap enough per lane.
        return self._slots[i].copy()

    def get[A: Array](self, i: Int) -> A:
        """Typed slot read — `ctx.get[BoolArray](i)`. Pulls the typed array straight
        out of the slot's `Datum` (a ref-count bump), skipping the `as_xxx().copy()`
        dance at every breaker read."""
        return self._slots[i][DynArray].as_type[A]().copy()

    def size(self) -> Int:
        return len(self._slots)


# Known follow-ups (flagged during design; not yet addressed):
#  - PERF: `Context.get` copies a `Datum` per lane for a fused breaker. A pass could
#    hoist scalar splats out of the loop or intern slots to plain scalars/pointers.
#  - CSE: positional slots forgo dedup — identical breakers (`sum(a)` used twice)
#    recompute. A keyed dedup in `prepare` (map subtree-key -> slot) can restore it.
#  - SCHEDULER: independent breakers run sequentially in `prepare`; they are
#    independent stages and can be scheduled to run concurrently.
#  - `DynScalar.repeat` has no string support, so a string *scalar* cannot broadcast
#    to a column yet (core-array machinery, orthogonal to fusion).


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
# Value — every node. `execute` is abstract; `prepare` defaults to a no-op (only
# composites recurse and breakers prepare).
# ---------------------------------------------------------------------------
trait Value(Copyable, ImplicitlyDeletable, Movable):
    comptime OutType: DataType
    comptime OutShape: Int  # 0 scalar, 1 columnar
    # A pipeline breaker (cross-row / materializing) — the family drivers prepare
    # it and read the slot straight back instead of running the fused loop. Default off.

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        """Produce this node's result `Datum` — the family driver: a numeric `Buffer`,
        a bool `Bitmap`, a string builder, or (for a leaf like `ListColumn`) just its
        column. Abstract; a breaker never runs it (the `else` below elides it).
        """
        ...

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        """The one dispatch, shared by every family: a breaker `prepare`s its stage
        and reads the slot straight back; everything else `materialize`s the result.
        """
        comptime if conforms_to(Self, Breaker):
            var i = ctx.size()
            self.prepare(batch, ctx)
            return ctx.get(i)
        else:
            return self.materialize(batch, ctx)

    def execute(self, batch: RecordBatch) raises -> Datum:
        """Top-level entry — execute against a fresh context (also the fresh
        sub-context each breaker uses to prepare its operand)."""
        var ctx = Context()
        return self.execute(batch, ctx)

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        """Pre-pass before a fused loop: run breaker stages into `ctx` in DFS
        order. A breaker runs its own; composites override to recurse; a leaf
        does nothing."""
        comptime if conforms_to(Self, Breaker):
            ctx.append(self.materialize(batch, ctx))

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

    def prune(self, stats: PruneStats) raises -> PruneBound:
        """What this node's value can be, given per-column statistics.

        Defaults to "no information", which is always sound — a caller only ever
        skips data it has *proven* cannot match. Nodes that can say something
        useful (columns, literals, comparisons, `and`/`or`) override it.

        This used to be a 9-arm switch on the interpreter's tag. Putting it on
        the node means a new node cannot be forgotten by it: it either says
        something or inherits the conservative answer."""
        return PruneBound.unknown()

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


trait Breaker(Value):
    """A node that ends a fused pipeline: it cannot be evaluated a lane at a
    time, so it materializes into `Context` and its consumer reads the slot.

    Conformance *is* the marker. It replaces a `comptime IsBreaker: Bool` that
    every node had to set by hand, the same hazard `IsErased` posed before it
    was deleted.

    It adds no method. `materialize` is the one strategy hook every node already
    implements; conforming here says only *when* it runs: as a pre-pass, into a
    `Context` slot, rather than inline in a fused loop.
    """

    pass


trait NumericValue(Value):
    comptime OutType: NumericType

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        ...

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

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        self.prepare(batch, ctx)
        comptime native = Self.OutType.native
        comptime if Self.OutShape == 0:  # scalar → evaluate the lane once, then splat
            var slot = 0
            var v = self.vectorwise[1](batch, ctx, slot, 0)[0].cast[
                Self.OutType.native
            ]()
            return PrimitiveScalar[Self.OutType](v).to_dyn()
        else:  # columnar → one fused vectorized pass
            var length = batch.num_rows()
            var buf = Buffer.alloc_uninit[native](length)

            @parameter
            @always_inline
            def producer[W: Int](i: Int) -> SIMD[native, W]:
                var slot = 0
                return self.vectorwise[W](batch, ctx, slot, i)

            apply[native, producer](buf.view[native](0, length))
            var v = self.validity(batch)
            var arr = PrimitiveArray[Self.OutType](
                dtype=Self.OutType(),
                length=length,
                nulls=v.value().unset_count() if v else 0,
                offset=0,
                bitmap=v^,
                buffer=buf.to_immutable(),
            )
            return arr^.to_dyn()

    # --- aggregates without a fused reduction -------------------------------
    #
    # `sum`/`mean`/`min`/`max`/`product`/`count` are already above: they build a
    # `Reduction`, which is both a scalar value inside an expression and — via
    # `AggExpr`'s conversion — an aggregate in a `GROUP BY`. The distinct counts
    # have no fused form (their state is a hash set / HLL sketch, not a scalar
    # accumulator), so they go straight to an `AggExpr`.


struct NumericColumn[T: NumericType](NumericValue):
    """A numeric column, resolved by name against `batch.schema` each pass.
    PERF: `get_field_index` runs per pass; resolve-once is a follow-up."""

    comptime OutType = Self.T
    comptime OutShape = 1

    var _name: String

    def referenced_columns(self) -> List[String]:
        return [self._name.copy()]

    def bound_column(self, schema: Schema) raises -> Int:
        var i = schema.get_field_index(self._name)
        if i == -1:
            raise Error("column '", self._name, "' not found")
        return i

    def prune(self, stats: PruneStats) raises -> PruneBound:
        var iv = stats.by_name(self._name)
        return PruneBound.interval(iv[0].copy(), iv[1].copy())

    def __init__(out self, var name: String):
        self._name = name^

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        return (
            batch.columns[batch.schema.get_field_index(self._name)]
            .as_primitive[Self.T]()
            .values()
            .load[W](idx)
        )

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        # a leaf column returns as-is (keeps validity; the fused loop drops
        # nulls). `RecordBatch.column(name)` owns the missing-name diagnostic —
        # `get_field_index` answers -1, and indexing the column list with that
        # trips a bounds assert that aborts the runner instead of reporting the
        # name. Every column leaf goes through it for that reason.
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

    def render(self) -> String:
        return String("literal(", self._value, ")")

    def prune(self, stats: PruneStats) raises -> PruneBound:
        var v = PrimitiveScalar[Self.T](self._value).to_dyn()
        return PruneBound.interval(Optional(v.copy()), Optional(v^))

    var _value: Scalar[Self.OutType.native]

    def referenced_columns(self) -> List[String]:
        return List[String]()

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
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

    """Propagated, not defaulted: an operand that resolves its types at run time
    cannot be fused into, so this whole node takes the dispatch arm."""

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

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var a = self.l.vectorwise[W](batch, ctx, slot, idx).cast[
            Self.OutType.native
        ]()
        var b = self.r.vectorwise[W](batch, ctx, slot, idx).cast[
            Self.OutType.native
        ]()
        return Self.K.core[Self.OutType.native, W](a, b)

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return Bitmap.intersect(self.l.validity(batch), self.r.validity(batch))

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.l.prepare(batch, ctx)
        self.r.prepare(batch, ctx)


@fieldwise_init
struct NumericUnary[K: UnaryNumericKernel, A: NumericValue](NumericValue):
    """Fused unary op preserving the operand dtype — `neg`, `abs`, …."""

    comptime OutType = Self.A.OutType
    comptime OutShape = Self.A.OutShape
    var a: Self.A

    def render(self) -> String:
        return String(Self.K.name, "(", self.a.render(), ")")

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        return Self.K.core[Self.OutType.native, W](
            self.a.vectorwise[W](batch, ctx, slot, idx)
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.prepare(batch, ctx)


@fieldwise_init
struct NumericCast[To: NumericType, A: NumericValue](NumericValue):
    """Fused numeric → numeric cast — reinterprets the operand's SIMD lane at the
    target dtype, so `col.cast(int64) + other` stays a single fused pass."""

    comptime OutType = Self.To
    comptime OutShape = Self.A.OutShape
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        return NumericCastKernel.core[
            Self.A.OutType.native, Self.OutType.native, W
        ](self.a.vectorwise[W](batch, ctx, slot, idx))

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.prepare(batch, ctx)


@fieldwise_init
struct FloatBinary[K: BinaryKernel, L: NumericValue, R: NumericValue](
    NumericValue
):
    """Binary op whose result is always float64 — `Div` (true division), `Pow`.
    Operands cast up to float64 before the kernel, so `5 / 2 == 2.5`."""

    comptime OutType = Float64Type
    comptime OutShape = max(Self.L.OutShape, Self.R.OutShape)
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

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var a = self.l.vectorwise[W](batch, ctx, slot, idx).cast[
            Self.OutType.native
        ]()
        var b = self.r.vectorwise[W](batch, ctx, slot, idx).cast[
            Self.OutType.native
        ]()
        return Self.K.core[Self.OutType.native, W](a, b)

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return Bitmap.intersect(self.l.validity(batch), self.r.validity(batch))

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.l.prepare(batch, ctx)
        self.r.prepare(batch, ctx)


@fieldwise_init
struct FloatUnary[K: UnaryKernel, A: NumericValue](NumericValue):
    """Unary op whose result is always float64 — `sqrt`, `exp`, `log`."""

    comptime OutType = Float64Type
    comptime OutShape = Self.A.OutShape
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        return Self.K.core[Self.OutType.native, W](
            self.a.vectorwise[W](batch, ctx, slot, idx).cast[
                Self.OutType.native
            ]()
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.prepare(batch, ctx)


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
    comptime NativeType: DType  # operand width (sizes the SIMD lane), not the output

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        ...

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

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        self.prepare(batch, ctx)
        var length = batch.num_rows()
        var bm = Bitmap.alloc_uninit(length)

        @parameter
        @always_inline
        def producer[W: Int](i: Int) -> SIMD[DType.bool, W]:
            var slot = 0
            return self.vectorwise[W](batch, ctx, slot, i)

        apply[Self.NativeType, producer](bm.view())  # bit-packing overload
        var v = self.validity(batch)
        return BoolArray(
            length=length,
            nulls=v.value().unset_count() if v else 0,
            offset=0,
            bitmap=v^,
            buffer=bm.to_immutable(),
        ).to_dyn()


@fieldwise_init
struct NumericCompare[
    K: NumericCompareKernel,
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
    var l: Self.L
    var r: Self.R

    def prune(self, stats: PruneStats) raises -> PruneBound:
        # The interpreter had one switch arm per operator here. `K` names the
        # operator, so the rule is selected at elaboration instead.
        var l = self.l.prune(stats)
        var r = self.r.prune(stats)
        comptime n = Self.K.name
        comptime if n == "equal":
            return PruneBound.boolean(l.maybe_eq(r))
        elif n == "less":
            return PruneBound.boolean(l.maybe_lt(r))
        elif n == "less_equal":
            return PruneBound.boolean(l.maybe_le(r))
        elif n == "greater":
            return PruneBound.boolean(l.maybe_gt(r))
        elif n == "greater_equal":
            return PruneBound.boolean(l.maybe_ge(r))
        else:  # not_equal carries no usable interval rule
            return PruneBound.unknown()

    def render(self) -> String:
        return String(
            Self.K.name, "(", self.l.render(), ", ", self.r.render(), ")"
        )

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        var a = self.l.vectorwise[W](batch, ctx, slot, idx).cast[
            Self.ArgType.native
        ]()
        var b = self.r.vectorwise[W](batch, ctx, slot, idx).cast[
            Self.ArgType.native
        ]()
        return Self.K.core[Self.ArgType.native, W](a, b)

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return Bitmap.intersect(self.l.validity(batch), self.r.validity(batch))

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.l.prepare(batch, ctx)
        self.r.prepare(batch, ctx)


comptime Lt = NumericCompare[LtKernel, _, _]
comptime Le = NumericCompare[LeKernel, _, _]
comptime Gt = NumericCompare[GtKernel, _, _]
comptime Ge = NumericCompare[GeKernel, _, _]
comptime Eq = NumericCompare[EqKernel, _, _]
comptime Ne = NumericCompare[NeKernel, _, _]


# ---------------------------------------------------------------------------
# Boolean logic — fused vectorwise over bit-packed masks (bitwise SIMD). Rather
# than materializing bool children, these stay in the bool lane:
# `(a < 3) & (b > 15)` is one fused pass. Compute lives in the boolean kernels.
# ---------------------------------------------------------------------------
@fieldwise_init
struct BoolBinary[K: BoolBinaryKernel, L: BoolValue, R: BoolValue](BoolValue):
    """Fused `and`/`or`/`xor` over two bool masks."""

    comptime OutType = BoolType
    comptime OutShape = max(Self.L.OutShape, Self.R.OutShape)
    # size the SIMD width by the WIDER operand — a narrow one (e.g. an int32 bool
    # breaker) must not shrink W below what a wider sibling's load (int64) needs, or
    # `SIMD[int64, W]` overflows the register.
    comptime NativeType = wider[Self.L.NativeType, Self.R.NativeType]
    var l: Self.L
    var r: Self.R

    def prune(self, stats: PruneStats) raises -> PruneBound:
        var l = self.l.prune(stats).maybe_true
        var r = self.r.prune(stats).maybe_true
        comptime n = Self.K.name
        comptime if n == "and_":
            return PruneBound.boolean(l and r)
        elif n == "or_":
            return PruneBound.boolean(l or r)
        else:  # xor tells us nothing about the row group
            return PruneBound.unknown()

    def render(self) -> String:
        return String(
            Self.K.name, "(", self.l.render(), ", ", self.r.render(), ")"
        )

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        var a = self.l.vectorwise[W](batch, ctx, slot, idx)
        var b = self.r.vectorwise[W](batch, ctx, slot, idx)
        return Self.K.core[W](a, b)

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # The fused `vectorwise` above already produces the correct DATA (plain
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

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.l.prepare(batch, ctx)
        self.r.prepare(batch, ctx)


@fieldwise_init
struct BoolUnary[K: BoolUnaryKernel, A: BoolValue](BoolValue):
    """Fused `not` over a bool mask."""

    comptime OutType = BoolType
    comptime OutShape = Self.A.OutShape
    comptime NativeType = Self.A.NativeType
    var a: Self.A

    def render(self) -> String:
        return String(Self.K.name, "(", self.a.render(), ")")

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        return Self.K.core[W](self.a.vectorwise[W](batch, ctx, slot, idx))

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.prepare(batch, ctx)


comptime And = BoolBinary[AndKernel, _, _]
comptime Or = BoolBinary[OrKernel, _, _]
comptime Xor = BoolBinary[XorKernel, _, _]
comptime Not = BoolUnary[NotKernel, _]


@fieldwise_init
struct BoolReduce[K: BoolReduceKernel, A: BoolValue](BoolValue, Breaker):
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
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var arr = (
            into_array(self.a.execute(batch), batch.num_rows()).as_bool().copy()
        )
        return Datum(BoolScalar(Self.K.reduce(arr)).to_dyn())

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        var d = ctx.get(slot)
        slot += 1
        return SIMD[DType.bool, W](d[DynScalar].as_bool().value())


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
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        return Self.K.core[Self.NativeType, W](
            self.a.vectorwise[W](batch, ctx, slot, idx)
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.prepare(batch, ctx)


@fieldwise_init
struct NullPredicate[K: UnaryPredicateKernel, A: Value](BoolValue, Breaker):
    """`is_null`/`not_null` — reads the operand's validity (any family), so a bool
    breaker: prepare the operand, run the kernel into a `BoolArray`, read it."""

    comptime OutType = BoolType
    comptime OutShape = Self.A.OutShape
    comptime NativeType = DType.int32  # lane width for the bit-pack driver
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return Datum(Self.K.apply(arr).to_dyn())

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        var s = slot
        slot += 1
        return ctx.get[BoolArray](s).values().load[DType.bool, W](idx)


comptime IsNan = NumericPredicate[IsNanKernel, _]
comptime IsInf = NumericPredicate[IsInfKernel, _]
comptime IsNull = NullPredicate[IsNullKernel, _]
comptime NotNull = NullPredicate[NotNullKernel, _]


# ---------------------------------------------------------------------------
# Cross-family casts. Fixed-width -> fixed-width fuses (num <-> bool via a per-lane
# kernel `core`). String parses (string -> num/bool) have
# no value lane, so they're breakers. All compute lives in `kernels.cast`.
# (Casts *to* string need the string lane to thread a slot for a materialized
# result — a follow-up. Currently only `string` operands, not `large_string`.)
# ---------------------------------------------------------------------------
@fieldwise_init
struct NumToBool[A: NumericValue](BoolValue):
    """Fused numeric -> bool (`x != 0`)."""

    comptime OutType = BoolType
    comptime OutShape = Self.A.OutShape
    comptime NativeType = Self.A.OutType.native
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        return NumToBoolKernel.core[Self.NativeType, W](
            self.a.vectorwise[W](batch, ctx, slot, idx)
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.prepare(batch, ctx)


@fieldwise_init
struct BoolToNum[To: NumericType, A: BoolValue](NumericValue):
    """Fused bool -> numeric (`True->1, False->0`)."""

    comptime OutType = Self.To
    comptime OutShape = Self.A.OutShape
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        return BoolToNumKernel.core[Self.OutType.native, W](
            self.a.vectorwise[W](batch, ctx, slot, idx)
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(batch)

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.prepare(batch, ctx)


@fieldwise_init
struct StringToNum[To: NumericType, A: StringValue](Breaker, NumericValue):
    """Parse string -> numeric (nulling on unparseable). No value lane, so a breaker:
    parse the whole column once via the kernel, then load per lane."""

    comptime OutType = Self.To
    comptime OutShape = 1
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var s = (
            into_array(self.a.execute(batch), batch.num_rows())
            .as_string()
            .copy()
        )
        return Datum(
            StringToNumKernel.apply[StringType, Self.To, False](s).to_dyn()
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # A parse failure is a null the *input* does not have ("x" -> null), so
        # validity comes from the parsed column, not from `a`. Inheriting the
        # all-valid default made `to_int(s) + 1` yield 0 where it should be null.
        #
        # This re-runs the parse: `validity` has no access to the `Context` the
        # stage result already sits in. That is the protocol defect
        # `docs/design-expression-evaluation.md` removes by carrying validity in
        # the node's state; correctness first, and the extra pass goes away with
        # it.
        var s = (
            into_array(self.a.execute(batch), batch.num_rows())
            .as_string()
            .copy()
        )
        return (
            StringToNumKernel.apply[StringType, Self.To, False](s)
            .to_data()
            .owned_validity()
        )

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var i = slot
        slot += 1
        return ctx.get[PrimitiveArray[Self.To]](i).values().load[W](idx)


@fieldwise_init
struct StringToBool[A: StringValue](BoolValue, Breaker):
    """Parse string -> bool (`"true"`/`"false"`/`"1"`/`"0"`). A bool breaker."""

    comptime OutType = BoolType
    comptime OutShape = 1
    comptime NativeType = DType.int32
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var s = (
            into_array(self.a.execute(batch), batch.num_rows())
            .as_string()
            .copy()
        )
        return Datum(StringToBoolKernel.apply[StringType, False](s).to_dyn())

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        var i = slot
        slot += 1
        return ctx.get[BoolArray](i).values().load[DType.bool, W](idx)


# ---------------------------------------------------------------------------
# StringValue — the elementwise string strategy. No W-wide lane: `core(idx)`
# yields one row's `String`, and the driver appends them into a builder, so a
# concat chain fuses without materializing intermediate string arrays.
# ---------------------------------------------------------------------------
trait StringValue(Value):
    comptime OutType: StringLikeType

    @always_inline
    def elementwise(
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> String:
        ...

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        self.prepare(batch, ctx)
        comptime if Self.OutShape == 0:
            var slot = 0
            return StringScalar(self.elementwise(batch, ctx, slot, 0)).to_dyn()
        else:
            var n = batch.num_rows()
            # Validity is threaded here for the same reason the numeric and bool
            # drivers thread it: `elementwise` reads values through `unsafe_get`,
            # which does not consult the bitmap, so without this every string
            # *transformation* returned an all-valid column. A bare column keeps
            # its nulls (that path returns the column as-is), which is what hid
            # this.
            var v = self.validity(batch)
            var builder = BinaryLikeBuilder[Self.OutType](capacity=n)
            for i in range(n):
                var slot = 0
                if v and not v.value().test(i):
                    builder.append_null()
                else:
                    builder.append(self.elementwise(batch, ctx, slot, i))
            return builder.finish().to_dyn()

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
    """A string column, resolved by name against `batch.schema` each pass."""

    comptime OutType = Self.T
    comptime OutShape = 1
    var _name: String

    def referenced_columns(self) -> List[String]:
        return [self._name.copy()]

    def __init__(out self, var name: String):
        self._name = name^

    @always_inline
    def elementwise(
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> String:
        return String(
            batch.columns[batch.schema.get_field_index(self._name)]
            .as_string()
            .unsafe_get(UInt(idx))
        )

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        # `RecordBatch.column(name)` owns the missing-name diagnostic — see
        # `NumericColumn.materialize`.
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
    var _value: String

    def referenced_columns(self) -> List[String]:
        return List[String]()

    @always_inline
    def elementwise(
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> String:
        return self._value.copy()


@fieldwise_init
struct Concat[L: StringValue, R: StringValue](StringValue):
    """Fused elementwise concatenation — `col || "a" || "b"` builds each row once,
    no intermediate `StringArray` for `col || "a"`."""

    comptime OutType = Self.L.OutType
    comptime OutShape = max(Self.L.OutShape, Self.R.OutShape)
    var l: Self.L
    var r: Self.R

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    @always_inline
    def elementwise(
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> String:
        return ConcatKernel.combine(
            self.l.elementwise(batch, ctx, slot, idx),
            self.r.elementwise(batch, ctx, slot, idx),
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # a null operand poisons the row, as it does in `ConcatKernel`
        return Bitmap.intersect(self.l.validity(batch), self.r.validity(batch))

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.l.prepare(batch, ctx)
        self.r.prepare(batch, ctx)


@fieldwise_init
struct StringUnary[K: StringMapKernel, A: StringValue](StringValue):
    """Fused elementwise `string -> string` (`upper`/`lower`/`strip`/…). Composes in
    one builder pass with concat: `upper(col) || "!"` never materializes `upper(col)`.
    The transform lives in the kernel."""

    comptime OutType = Self.A.OutType
    comptime OutShape = Self.A.OutShape
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    @always_inline
    def elementwise(
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> String:
        var s = self.a.elementwise(batch, ctx, slot, idx)
        return Self.K.transform(StringSlice(s))

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # a map transforms values, never validity — `upper(null)` is null
        return self.a.validity(batch)

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.prepare(batch, ctx)


comptime Upper = StringUnary[UpperKernel, _]
comptime Lower = StringUnary[LowerKernel, _]
comptime Strip = StringUnary[StripKernel, _]
comptime LStrip = StringUnary[LStripKernel, _]
comptime RStrip = StringUnary[RStripKernel, _]
comptime Reverse = StringUnary[ReverseKernel, _]
comptime Capitalize = StringUnary[CapitalizeKernel, _]


# ---------------------------------------------------------------------------
# Casts *to* string — a materialized string result, so a string breaker: `prepare`
# folds the operand to a string column via `kernels.cast`; `elementwise` reads that
# column per row (threading `slot`), so `cast(x, string) || "!"` still fuses in the
# elementwise builder pass. Compute lives in the cast kernels.
# ---------------------------------------------------------------------------
@fieldwise_init
struct NumToString[To: StringLikeType, A: NumericValue](Breaker, StringValue):
    """Format numeric -> string."""

    comptime OutType = Self.To
    comptime OutShape = 1
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return Datum(NumToStringKernel.dispatch(arr, Self.To()))

    @always_inline
    def elementwise(
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> String:
        var i = slot
        slot += 1
        return String(
            ctx.get[BinaryLikeArray[Self.To]](i).unsafe_get(UInt(idx))
        )


@fieldwise_init
struct BoolToString[To: StringLikeType, A: BoolValue](Breaker, StringValue):
    """`True`/`False` -> string."""

    comptime OutType = Self.To
    comptime OutShape = 1
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return Datum(BoolToStringKernel.dispatch(arr, Self.To()))

    @always_inline
    def elementwise(
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> String:
        var i = slot
        slot += 1
        return String(
            ctx.get[BinaryLikeArray[Self.To]](i).unsafe_get(UInt(idx))
        )


@fieldwise_init
struct StringToString[To: StringLikeType, A: StringValue](Breaker, StringValue):
    """Cast between string containers (`string` <-> `large_string`)."""

    comptime OutType = Self.To
    comptime OutShape = 1
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return Datum(StringToStringKernel.dispatch(arr, Self.To(), False))

    @always_inline
    def elementwise(
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> String:
        var i = slot
        slot += 1
        return String(
            ctx.get[BinaryLikeArray[Self.To]](i).unsafe_get(UInt(idx))
        )


# ---------------------------------------------------------------------------
# String predicates — `string × string -> bool`. Variable-width comparison has no
# vectorwise lane, so this is a bool *breaker*: `prepare` materializes both string
# stages and runs the kernel into a `BoolArray`; `vectorwise` reads that mask, so a
# predicate still fuses under boolean logic. The comparison lives in the kernel.
# ---------------------------------------------------------------------------
@fieldwise_init
struct StringPredicate[
    K: StringPredicateKernel, L: StringValue, R: StringValue
](BoolValue, Breaker):
    comptime OutType = BoolType
    comptime OutShape = max(Self.L.OutShape, Self.R.OutShape)
    comptime NativeType = DType.int32  # lane width for the bit-pack driver
    var l: Self.L
    var r: Self.R

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.l.referenced_columns(), self.r.referenced_columns()
        )

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # A string predicate is null exactly where an operand is: the kernel
        # already ANDs the operand bitmaps, and `vectorwise` reads only the
        # data bits, so the node has to carry the validity itself.
        return Bitmap.intersect(self.l.validity(batch), self.r.validity(batch))

    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
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
            var rctx = Context()
            self.r.prepare(batch, rctx)
            var rslot = 0
            var pat = self.r.elementwise(batch, rctx, rslot, 0)
            ctx.append(Self.K.apply_scalar(la, pat).to_dyn())
        else:
            var ra = into_array(self.r.execute(batch), n).as_string().copy()
            ctx.append(Self.K.apply(la, ra).to_dyn())

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        var s = slot
        slot += 1
        return ctx.get[BoolArray](s).values().load[DType.bool, W](idx)


comptime StartsWith = StringPredicate[StartsWithKernel, _, _]
comptime EndsWith = StringPredicate[EndsWithKernel, _, _]
comptime StrContains = StringPredicate[ContainsKernel, _, _]
comptime StrEq = StringPredicate[StringEqKernel, _, _]
comptime StrNe = StringPredicate[StringNeKernel, _, _]
# SQL LIKE / ILIKE — same breaker shape (`LikeKernel`/`ILikeKernel` are
# `StringPredicateKernel`s), so they slot straight into `StringPredicate`.
comptime Like = StringPredicate[LikeKernel, _, _]
comptime ILike = StringPredicate[ILikeKernel, _, _]


# String ordering comparisons — `string < <= > >=` -> bool. The compare kernels
# name their string counterpart, so ordering is the same node as every other
# string predicate; only the kernel differs.
comptime StrLt = StringPredicate[StringLtKernel, _, _]
comptime StrLe = StringPredicate[StringLeKernel, _, _]
comptime StrGt = StringPredicate[StringGtKernel, _, _]
comptime StrGe = StringPredicate[StringGeKernel, _, _]


# ---------------------------------------------------------------------------
# is_in — SQL `x IN (...)`. A bool breaker over any value family: `prepare`
# hashes the captured value-set once and probes the operand column (reusing
# `kernels.membership.IsInKernel`), then `vectorwise` loads the mask. The output is
# always valid (PyArrow `is_in` never nulls), so validity defaults to `None`.
# ---------------------------------------------------------------------------
@fieldwise_init
struct IsIn[A: Value](BoolValue, Breaker):
    comptime OutType = BoolType
    comptime OutShape = 1
    comptime NativeType = DType.int32
    var a: Self.A
    var _value_set: DynArray

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return Datum(IsInKernel.dispatch(arr, self._value_set.copy()).to_dyn())

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        var s = slot
        slot += 1
        return ctx.get[BoolArray](s).values().load[DType.bool, W](idx)


# ---------------------------------------------------------------------------
# Strategy transition (string -> numeric) — modelled as a plain breaker. The two
# strategies don't compose, so the string (elementwise) stage materializes; the
# `LengthKernel` folds it to the int32 length column (vectorized offset subtraction,
# handling string / large_string). `vectorwise` then just loads that column, so the
# arithmetic above fuses — exactly like a window: `length(s) + 1` is one numeric pass
# over the materialized lengths. Same shape as every other breaker.
# ---------------------------------------------------------------------------
@fieldwise_init
struct StringLength[A: StringValue](Breaker, NumericValue):
    """Byte length of a string value → int32. `prepare` materializes the string
    stage and folds it to the length column via `LengthKernel`; `vectorwise` loads
    that column per lane."""

    comptime OutType = Int32Type
    comptime OutShape = 1
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # a null string has a null length — validity passes through unchanged.
        return self.a.validity(batch)

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var s = into_array(self.a.execute(batch), batch.num_rows())
        return Datum(LengthKernel.dispatch(s))

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var s = slot
        slot += 1
        return ctx.get[Int32Array](s).values().load[W](idx)


# ---------------------------------------------------------------------------
# Pipeline breakers — cross-row `Value`s that cut the tree into stages. They
# prepare their operand through a *fresh* sub-context (`run`, so nested
# breakers don't perturb the outer slot order), then act as fused leaves: a scalar
# reduction splats, a columnar window loads. `prepare` appends the result; `core`
# reads it back positionally via `slot`.
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
struct Reduction[K: AggKernel, A: NumericValue](Breaker, NumericValue):
    """Whole-array reduction → a scalar. Output dtype is the kernel's accumulator
    algebra `K.AccType[A.OutType]` (sum widens, mean → float64, min/max keep it).
    """

    comptime OutType = Self.K.AccType[Self.A.OutType]
    comptime OutShape = 0
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

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        # The operand's dtype is `A.OutType`, so the reduce is the fully typed
        # one — no dtype dispatch, and the erasure is only the `Context` slot.
        var arg = into_array(self.a.execute(batch), batch.num_rows())
        return Datum(Self.K.reduce(arg.as_primitive[Self.A.OutType]()).to_dyn())

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var d = ctx.get(slot)
        slot += 1
        return SIMD[Self.OutType.native, W](
            d[DynScalar].as_primitive[Self.OutType]().value()
        )


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
struct WindowFunction[Func: WindowKernel, A: Value](Breaker, NumericValue):
    """`func.over(spec)` → a columnar breaker. `prepare` materializes the whole
    output column into `ctx`; `core` then loads it per lane like a column."""

    comptime OutType = Self.Func.OutType
    comptime OutShape = 1
    var a: Self.A
    var spec: WindowSpec

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var v = into_array(self.a.execute(batch), batch.num_rows())
        return Datum(Self.Func.evaluate_all(v))

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var s = slot
        slot += 1
        return ctx.get[PrimitiveArray[Self.OutType]](s).values().load[W](idx)


comptime RowNumber = WindowFunction[RowNumberKernel, _]


# ---------------------------------------------------------------------------
# Conditional / null-handling — `coalesce`, `nullif`, `case_when` over the numeric
# family. Value selection is data-dependent (which candidate per row), so these are
# numeric breakers: `prepare` runs the selection kernel once into a column, then
# `vectorwise` loads it. Their result validity is data-dependent too (coalesce nulls
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
](Breaker, NumericValue):
    """`coalesce`/`nullif` over two same-dtype numeric operands; `K` picks the
    kernel. `prepare` materializes the result once; `vectorwise` loads it."""

    comptime OutType = Self.L.OutType
    comptime OutShape = 1
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
        return self._result(batch).to_data().owned_validity()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        return Datum(self._result(batch))

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var i = slot
        slot += 1
        return ctx.get[PrimitiveArray[Self.OutType]](i).values().load[W](idx)


comptime Coalesce = ConditionalBinary[CoalesceKernel, _, _]
comptime Nullif = ConditionalBinary[NullifKernel, _, _]


@fieldwise_init
struct CaseWhen[C: BoolValue, T: NumericValue, E: NumericValue](
    Breaker, NumericValue
):
    """Single-branch `CASE WHEN cond THEN then ELSE otherwise` over numeric
    values — `then`/`otherwise` share a dtype. A null condition counts as false
    (Arrow semantics), so `otherwise` is chosen there."""

    comptime OutType = Self.T.OutType
    comptime OutShape = 1
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
        return self._result(batch).to_data().owned_validity()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        return Datum(self._result(batch))

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var i = slot
        slot += 1
        return ctx.get[PrimitiveArray[Self.OutType]](i).values().load[W](idx)


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

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        # `RecordBatch.column(name)` owns the missing-name diagnostic — see
        # `NumericColumn.materialize`.
        return batch.column(self._name).copy()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return batch.column(self._name).to_data().owned_validity()

    def name(self) -> String:
        return self._name.copy()


@fieldwise_init
struct TemporalExtract[K: TemporalExtractKernel, A: TemporalValue](
    Breaker, NumericValue
):
    """Extract a calendar/clock field from a temporal value → int32. A breaker,
    same shape as `StringLength`."""

    comptime OutType = Int32Type
    comptime OutShape = 1
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # a null temporal value has a null field — validity passes through.
        return self.a.validity(batch)

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return Datum(Self.K.dispatch(arr))

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var i = slot
        slot += 1
        return ctx.get[Int32Array](i).values().load[W](idx)


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

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
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

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        # `RecordBatch.column(name)` owns the missing-name diagnostic — see
        # `NumericColumn.materialize`.
        return batch.column(self._name).copy()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        return batch.column(self._name).to_data().owned_validity()

    def name(self) -> String:
        return self._name.copy()


@fieldwise_init
struct ListLength[A: ListValue](Breaker, NumericValue):
    """List element count -> int32. A breaker, same shape as `StringLength`."""

    comptime OutType = Int32Type
    comptime OutShape = 1
    var a: Self.A

    def referenced_columns(self) -> List[String]:
        return self.a.referenced_columns()

    def validity(
        self, batch: RecordBatch
    ) raises -> Optional[Bitmap[mut=False]]:
        # a null list has a null length — validity passes through unchanged.
        return self.a.validity(batch)

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var arr = into_array(self.a.execute(batch), batch.num_rows())
        return Datum(ArrayLengthKernel.dispatch(arr))

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var i = slot
        slot += 1
        return ctx.get[Int32Array](i).values().load[W](idx)


@fieldwise_init
struct ListContains[A: ListValue, E: NumericValue](BoolValue, Breaker):
    """Element-wise membership: `elem[i] in list[i]` -> bool. A breaker (a literal
    element broadcasts). Numeric element types."""

    comptime OutType = BoolType
    comptime OutShape = 1
    comptime NativeType = DType.int32
    var a: Self.A
    var elem: Self.E

    def referenced_columns(self) -> List[String]:
        return _union_columns(
            self.a.referenced_columns(), self.elem.referenced_columns()
        )

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var n = batch.num_rows()
        var la = into_array(self.a.execute(batch), n)
        var ea = into_array(self.elem.execute(batch), n)
        return Datum(ArrayContainsKernel.dispatch(la, ea))

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        DType.bool, W
    ]:
        var i = slot
        slot += 1
        return ctx.get[BoolArray](i).values().load[DType.bool, W](idx)


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
