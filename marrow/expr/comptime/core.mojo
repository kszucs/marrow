"""The comptime lane: expressions whose structure is their type.

`And[Gt[Column[Int64Type], Literal[Int64Type]], …]` is a type, not data. A whole
subtree therefore compiles to one inlined SIMD loop with no dispatch anywhere
inside it — worth a measured **3.4x** on binary size against the same plan built
from runtime expressions (1.46 MB vs 4.91 MB), because the fused form
instantiates only the kernels it names.

Two consequences follow, and both are load-bearing elsewhere:

- **Nothing outside can inspect the structure.** A rewriter cannot open a type,
  so every question it asks is answered by the node itself — which is what
  `Analyzable` exists for, and why its methods are total.
- **LLVM already optimises the interior.** Constant folding, GVN/CSE and
  instcombine all apply to an inlined loop, so an expression-level rewriter has
  nothing to find here. Interior rewrites belong to the runtime lane alone.

This module holds only what the lane *shares* — the base trait and the family
traits that refine it. The nodes themselves live beside it: `leaves.mojo`,
`operators.mojo`, `reductions.mojo`.

`ComptimeValue` is the base every node here shares. It does *not* declare
`lane` — that returns `SIMD[Type.native, W]`, which only means something for
a fixed-width type, so each family refines the base with its own. `NumericValue`
is the first such family; string, bool, temporal and list follow the same shape.
"""

from ...buffers import Bitmap
from ...dtypes import (
    BoolType,
    PrimitiveType,
    TemporalType,
    DataType,
    ListLikeType,
    DynType,
    NumericType,
    StringLikeType,
)
from ...schema import Schema
from ...arrays import (
    Array,
    StructArray,
    BinaryLikeArray,
    BoolArray,
    ListLikeArray,
    PrimitiveArray,
)
from ...buffers import Bitmap, Buffer
from ...builders import BinaryLikeBuilder
from ...scalars import PrimitiveScalar
from ...tabular import RecordBatch
from ...views import apply
from ..logical import Shape, Value
from ..bindings import Bindings
from ...kernels.aggregate import (
    COUNT,
    MAX,
    MEAN,
    MIN,
    PRODUCT,
    STDDEV,
    SUM,
    VARIANCE,
)
from ...kernels.bounds import Bounds
from ..pruning import PruneStats, Prunable, Truth
from .aggregates import (
    ApproxCountDistinct,
    StringApproxCountDistinct,
    StringCountDistinct,
    Count,
    CountDistinct,
    Max,
    Mean,
    Min,
    Product,
    StdDev,
    StringMax,
    StringMin,
    Sum,
    Variance,
)
from .numeric import (
    Abs,
    Add,
    Ceil,
    Div,
    Eq,
    Exp,
    Floor,
    Floordiv,
    Ge,
    Gt,
    Le,
    Ln,
    Lt,
    Mod,
    Mul,
    Ne,
    Neg,
    Pow,
    Round,
    Sign,
    Sqrt,
    Sub,
    TemporalEq,
    TemporalGe,
    TemporalGt,
    TemporalLe,
    TemporalLt,
    TemporalNe,
    Trunc,
)
from .boolean import And, IsInf, IsNan, Not, Or, Xor
from .strings import (
    Capitalize,
    EndsWith,
    ILike,
    LStrip,
    Like,
    Lower,
    Reverse,
    RStrip,
    StartsWith,
    StrContains,
    StrEq,
    StrGe,
    StrGt,
    StrLe,
    StrLt,
    StrNe,
    StringLength,
    Strip,
    Upper,
)
from .temporal import (
    DateTrunc,
    Day,
    DayOfWeek,
    DayOfYear,
    Hour,
    Minute,
    Month,
    Quarter,
    Second,
    Year,
)
from ...kernels.temporal import CalendarUnit
from ..physical import Datum
from ..physical import Evaluable, DynOperator, EvalOperator


# ---------------------------------------------------------------------------
# ComptimeValue — what every node in this lane shares
# ---------------------------------------------------------------------------
trait ComptimeValue(Evaluable, Prunable, Value):
    """A `Value` whose type states its output type and its per-batch state.

    This is where `evaluate` lives — **not** on `Value`. A logical node is
    stateless and has no business exposing a way to run itself; `evaluate` here
    is the *lane's fused driver*, the thing this lane's operator calls once it
    has been handed a batch. It is invisible to `DynValue` and to every
    consumer outside the lane, which reach a value only through
    `to_operator`.

    The two comptime members are what the runtime lane cannot supply, and are
    therefore the whole reason this is a separate trait rather than a naming
    convention: a `RuntimeValue` learns its type from a schema at run time and
    materialises a `DynArray` per node, so it can answer neither.
    """

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        """One fused pass over the batch. The lane's driver, called by
        `EvalOperator`; each family below supplies the default body."""
        ...

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        """Every comptime node becomes the same operator — one that forwards
        each batch to the fused driver. `grouped` is ignored: an elementwise
        value has no placement. Aggregates override this with a `FusedAggregateOperator`.

        `bindings` is handed to the operator, which passes it back down through
        `evaluate` -> `bind`. A `Param` reads it there, which is also the only
        place it can reach a parameter nested inside a fused subtree: this call
        copies the node without descending into it.
        """
        return EvalOperator[Self](self.copy(), bindings.copy())

    # -- the cardinality surface, per family --------------------------------
    #
    # These used to be two defaults on this trait, on the argument that "a
    # cardinality is an int64 whatever was counted, so there is no per-family
    # variation to express". That is true of the *answer* and false of the
    # *access*: `DistinctCount[exact, A]` hashes an `A`, and a hash is faster
    # typed. Each family knows the array its values evaluate to — a
    # `PrimitiveValue` to `PrimitiveArray[Self.Type]`, a `StringValue` to
    # `BinaryLikeArray[Self.Type]` — so the method belongs where that is known.
    #
    # This is the one-way door the old comment named: a trait default whose
    # return type a conformer must change becomes an ambiguous overload at
    # every call site, so the defaults had to be *removed* here rather than
    # specialised below.
    #
    # They still return an `Aggregate` that cannot fuse, which materialises the
    # aggregate but **not the operand**: `count_distinct(upper(region))` still
    # compiles `upper(region)` into one fused loop and only the distinct count
    # runs over a column. `count_distinct` has no fold algebra — no identity,
    # no combine, no finalize — so there is no `K` a fully fused node could be
    # parameterised on, and that is the one thing given up.

    comptime Type: DataType
    """This node's output type, known without a schema.

    `Analyzable.dtype(schema)` ignores its argument in this lane and answers
    from here. The runtime lane is the reason that method takes a schema at all.
    """

    # `Bound`, `bind` and `validity` are deliberately **not** here. All
    # three are *fusion* machinery, and this trait does not mean "fuses" --
    # it means "knows its output type without a schema". A Kleene `AND`
    # knows it produces `bool` and does not fuse at all; making it invent a
    # `Bound` it never reads, to satisfy a base it only needs `Type` from,
    # is how a trait starts describing its first implementer instead of its
    # concept.
    #
    # They are declared on the fusing families, which also do not agree on
    # what validity needs.
    #
    # Structural validity — numeric, comparison, string — is null-in-null-out
    # and answers from the `Bound` alone. Data-dependent validity does not:
    # Kleene `AND` decides its nulls from the operand *values* (`NULL AND
    # FALSE` is `FALSE`), so it needs the batch, and measurement says it should
    # not compute the rule per lane at all — the bitmap algebra in
    # `kernels.boolean._kleene` runs 64 bits per instruction against a SIMD
    # lane's one bit per byte, and beat a fused per-lane version by 4-10x
    # (`bench_boolean.mojo`, 2026-08-22).
    #
    # One signature on the base would therefore have to serve both, which is
    # how `expr/` ended up with two methods (`validity` and `state_validity`)
    # and evaluated `coalesce`/`nullif`/`case_when` twice per fused pass.

    # ---------------------------------------------------------------------------
    # NumericValue — the first family, refining the base with a lane
    # ---------------------------------------------------------------------------

    # -- boolean operators --------------------------------------------------
    # On `ComptimeValue`, not `BoolValue`: `And`/`Or`/`Xor` bind their operands
    # on `ComptimeValue`, and `BoolValue` means *fusable* — which boolean
    # binaries deliberately are not, because `_kleene`'s bitmap algebra beats
    # per-lane Kleene by 4-10x.

    def __and__[Rhs: ComptimeValue](self, o: Rhs) -> And[Self, Rhs]:
        return And(self.copy(), o.copy())

    def __or__[Rhs: ComptimeValue](self, o: Rhs) -> Or[Self, Rhs]:
        return Or(self.copy(), o.copy())

    def __xor__[Rhs: ComptimeValue](self, o: Rhs) -> Xor[Self, Rhs]:
        return Xor(self.copy(), o.copy())

    def __invert__(self) -> Not[Self]:
        return Not(self.copy())


trait PrimitiveValue(ComptimeValue):
    """A comptime node producing a fixed-width column.

    **The machinery, not a domain claim.** Everything below — `Bound`, `bind`,
    `lane[W]`, `validity`, the fused `evaluate` — needs only that the output is
    fixed-width: a `native` dtype, a buffer, a SIMD lane. It says nothing about
    which *operations* the type supports.

    That distinction used to be missing: this trait was `NumericValue`, and
    `Type: NumericType` was a single bound standing for both "I can be read by
    a lane loop" and "I support arithmetic". Two claims in one bound meant a
    temporal column could not be read at all, because dates are not numeric —
    even though reading one is the same instruction. `expr/` answered with a
    second `TemporalColumn` and a duplicated set of comparison arms.

    The domains are now markers on top: `NumericValue` and `TemporalValue` add
    no members and exist so a node can require the *operations* it needs. A
    comparison binds on this trait and serves both; arithmetic binds on
    `NumericValue` and rejects dates at compile time. Same split as
    `FoldKernel`'s `ArithmeticAgg` marker, for the same reason.
    """

    comptime Type: PrimitiveType

    def bounds(
        self, stats: PruneStats, bindings: Bindings
    ) -> Bounds[Self.Type.native]:
        """What this sub-expression's value can be over one granule. Default:
        unknown.

        `bounds` is to `prune` what `lane` is to `evaluate`: the typed half a
        composite reads from its operands. `Bounds[Self.Type.native]` is the
        same projection `lane`'s `SIMD[Self.Type.native, W]` already makes, and
        a *defaulted body* at that return type reduces — verified, so the
        fallback of declaring this abstract and writing a four-line body on
        every conformer is not needed.
        """
        return Bounds[Self.Type.native].unknown()

    def count_distinct(self) -> CountDistinct[Self]:
        """`COUNT(DISTINCT self)` — exact, nulls excluded (SQL semantics).

        Names the array its values evaluate to, `PrimitiveArray[Self.Type]`,
        which is what lets `DistinctCount` hash a typed column instead of
        walking an erased one."""
        return CountDistinct[Self](self.copy())

    def approx_count_distinct(self) -> ApproxCountDistinct[Self]:
        """`APPROX_COUNT_DISTINCT(self)` — a HyperLogLog estimate, ~0.65%
        standard error, nulls excluded."""
        return ApproxCountDistinct[Self](self.copy())

    comptime Bound: Copyable & Deinitable
    """Everything the lane loop needs, resolved once per batch.

    A column leaf's is its typed column; a literal's is nothing; a binary node's
    is `Tuple[L.Bound, R.Bound]`. Declared per concrete struct rather than
    defaulted: a trait default cannot reduce at a `-> Self.Bound` return site
    unless the bound is `ImplicitlyCopyable`, and marrow's array types
    deliberately are not.

    `expr/` called this `State`, which could mean anything. It is specifically
    *this subtree's column references, bound to this batch* — the stage between
    an expression and a per-element read.
    """

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        """Resolve this subtree against `batch`, once, before the lane loop.

        Every schema lookup and every `Variant` unwrap happens here so that
        `lane` does none. That removal is the optimisation.
        """
        ...

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        """One fused pass over the batch — `bind` once, then `lane` per chunk.

        A trait **default**, not a free driver, because it is the same for
        every numeric node and it is precisely what this trait means by
        evaluating. `expr/` kept it as a free `_drive_numeric`; CLAUDE.md
        records that re-defaulting a base trait's abstract method in a
        sub-trait recurses, but that limit is about returning
        `Self.ArrayType`, and `Datum` is concrete.

        Leaves override it: a column returns itself rather than copying through
        a fresh buffer, and a literal stays a scalar.
        """
        comptime native = Self.Type.native
        var bound = self.bind(batch, bindings)

        comptime if Self.shape == Shape.scalar:
            # Nothing to iterate — evaluate the lane once and stay lazy, which
            # is what `Shape.scalar` promises its caller.
            return Datum(
                PrimitiveScalar[Self.Type](
                    Optional(self.lane[1](bound, 0)[0]),
                    self.dtype(Schema.from_dtype(batch.dtype)).as_type[
                        Self.Type
                    ](),
                ).to_dyn()
            )
        else:
            var length = len(batch)
            var buf = Buffer.alloc_uninit[native](length)

            @always_inline
            def producer[W: Int](i: Int) {imm} -> SIMD[native, W]:
                return self.lane[W](bound, i)

            apply[native](buf.view[native](0, length), producer)

            # Validity once, from the bound — never per lane.
            var v = self.validity(bound)
            var arr = PrimitiveArray[Self.Type](
                dtype=self.dtype(Schema.from_dtype(batch.dtype)).as_type[
                    Self.Type
                ](),
                length=length,
                nulls=v.value().unset_count() if v else 0,
                offset=0,
                bitmap=v^,
                buffer=buf.to_immutable(),
            )
            return arr^.to_dyn()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity, or `None` when it cannot be null.

        Structural: a numeric node is null exactly where an operand is, so this
        intersects the operands' bitmaps and never reads their values. That is
        what lets it take the `Bound` and not the batch — a leaf's `Bound` *is*
        its column, bitmap included. `expr/` needed a second method
        (`state_validity`) precisely because its first one took the batch and
        re-ran the whole selection kernel for `coalesce`, `nullif` and
        `case_when`.

        `lane` produces the **data** bits only, so this stays a separate
        question from evaluation. It is a second recursive walk, not a free
        one, but it is affordable: a bitmap is one bit per row against a data
        element's 32 or 64, so a depth-`d` tree costs `d` passes over
        `rows/64` words next to one pass over `rows` elements.
        """
        ...

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        """One SIMD chunk.

        Reads `bound` and `idx` and nothing else — not `self`. That is what
        lets the whole subtree inline into a single loop, and it is why `bind`
        exists as a separate stage rather than the lane reaching through
        `self`.
        """
        ...


# ---------------------------------------------------------------------------
# StringValue — the family whose lane has no width
# ---------------------------------------------------------------------------
trait StringValue(ComptimeValue):
    """A comptime node producing a variable-width string column.

    The family that breaks the SIMD shape, and it is worth being explicit about
    why rather than treating it as an exception. `NumericValue.lane[W]` and
    `BoolValue.lane[W]` both answer `W` elements at once because their storage
    is fixed-width: element `i` is at a computable offset. UTF-8 is not — a
    string's position depends on every string before it — so **`lane` here takes
    no `W` and answers one `String`**. There is no vector to widen to.

    Everything else is unchanged, and that is the point: `bind` still resolves
    the subtree once per batch, `validity` is still structural and still reads
    the `Bound` rather than the batch, and a string subtree still fuses into one
    loop. Fusion is about eliminating dispatch, not about SIMD width, so it
    survives a family that cannot vectorise.
    """

    comptime Type: StringLikeType

    def dtype(self, schema: Schema) raises -> DynType:
        """The output type, from `Type` — the schema is the runtime lane's
        concern. Defaulted here rather than on `ComptimeValue` because
        constructing a `Type()` needs it to be `Defaultable`, and the base
        bounds it only by `DataType`. `TemporalValue` keeps its own override
        for the same reason: a timestamp's unit and timezone live in the value,
        not the type."""
        return DynType(Self.Type())

    def count_distinct(self) -> StringCountDistinct[Self]:
        """`COUNT(DISTINCT self)` — exact, nulls excluded (SQL semantics)."""
        return StringCountDistinct[Self](self.copy())

    def approx_count_distinct(self) -> StringApproxCountDistinct[Self]:
        """`APPROX_COUNT_DISTINCT(self)` — a HyperLogLog estimate, ~0.65%
        standard error, nulls excluded."""
        return StringApproxCountDistinct[Self](self.copy())

    comptime Bound: Copyable & Deinitable
    """This subtree's column references, bound to this batch — as
    `NumericValue.Bound`, and declared per concrete struct for the same
    reason."""

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        """Resolve this subtree against `batch`, once, before the lane loop."""
        ...

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        """One fused pass — `bind` once, then `lane` per row.

        A builder rather than `apply`: `apply` writes fixed-width elements into
        a preallocated buffer, and neither the width nor the total byte count of
        a string result is known before the loop runs. The offsets buffer is
        built as it goes, which is what a `BinaryLikeBuilder` already does.

        Leaves override this: a column hands back its own array rather than
        copying every byte through a fresh builder.
        """
        var bound = self.bind(batch, bindings)
        var length = len(batch)
        var builder = BinaryLikeBuilder[Self.Type](length)
        var v = self.validity(bound)
        if v:
            ref bm = v.value()
            var bits = bm.view()
            for i in range(length):
                if bits[i]:
                    builder.append(self.lane(bound, i))
                else:
                    builder.append_null()
        else:
            for i in range(length):
                builder.append(self.lane(bound, i))
        return builder.finish().to_dyn()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity, or `None` when it cannot be null.

        Structural, exactly as in `NumericValue`: intersect the operands'
        bitmaps, never read their values.
        """
        ...

    @always_inline
    def lane(self, bound: Self.Bound, idx: Int) -> String:
        """One row. The elementwise counterpart of the SIMD families'
        `lane[W]`, and the only shape a variable-width encoding admits."""
        ...

    # -- operators ----------------------------------------------------------

    def __eq__[Rhs: StringValue](self, o: Rhs) -> StrEq[Self, Rhs]:
        return StrEq(self.copy(), o.copy())

    def __ne__[Rhs: StringValue](self, o: Rhs) -> StrNe[Self, Rhs]:
        return StrNe(self.copy(), o.copy())

    def __lt__[Rhs: StringValue](self, o: Rhs) -> StrLt[Self, Rhs]:
        return StrLt(self.copy(), o.copy())

    def __gt__[Rhs: StringValue](self, o: Rhs) -> StrGt[Self, Rhs]:
        return StrGt(self.copy(), o.copy())

    def __le__[Rhs: StringValue](self, o: Rhs) -> StrLe[Self, Rhs]:
        return StrLe(self.copy(), o.copy())

    def __ge__[Rhs: StringValue](self, o: Rhs) -> StrGe[Self, Rhs]:
        return StrGe(self.copy(), o.copy())

    # -- transforms ---------------------------------------------------------
    #
    # Every one of these wraps a node that already existed in `strings.mojo`
    # and had no way to be *reached*: `Upper` … `Capitalize` were aliased,
    # `StartsWith` … `ILike` were aliased, and the only spelling was to name
    # the node — `Upper(col("region", string))` — which is what
    # `golden/prelude.mojo` re-exports them for. Seven of the golden cases
    # spell the fluent form instead and were skipped for want of these
    # thirteen one-line methods.
    #
    # Trait defaults rather than methods on each leaf, for the same reason
    # `NumericValue`'s operators are: a default costs nothing in a binary that
    # never calls it, and repeating them per conformer is how `StringColumn`
    # and `StringLiteral` drift apart.

    def upper(self) -> Upper[Self]:
        """ASCII upper-case. Non-ASCII bytes pass through — `UpperKernel`
        does not decode UTF-8, matching Arrow's `ascii_upper`."""
        return Upper[Self](self.copy())

    def lower(self) -> Lower[Self]:
        """ASCII lower-case, the counterpart of `upper`."""
        return Lower[Self](self.copy())

    def strip(self) -> Strip[Self]:
        """Whitespace removed from both ends."""
        return Strip[Self](self.copy())

    def lstrip(self) -> LStrip[Self]:
        """Whitespace removed from the left."""
        return LStrip[Self](self.copy())

    def rstrip(self) -> RStrip[Self]:
        """Whitespace removed from the right."""
        return RStrip[Self](self.copy())

    def reverse(self) -> Reverse[Self]:
        """Bytes reversed. Byte-wise, not code-point-wise: reversing a
        multi-byte UTF-8 sequence produces invalid UTF-8, which is what
        Arrow's `binary_reverse` does and why it is named for binary."""
        return Reverse[Self](self.copy())

    def capitalize(self) -> Capitalize[Self]:
        """First byte upper-cased, the rest lower-cased."""
        return Capitalize[Self](self.copy())

    def length(self) -> StringLength[Self]:
        """Byte length as `int32`. A null string has a null length, not 0 —
        the kernel records it and `StringLength` reads it back."""
        return StringLength[Self](self.copy())

    # -- predicates ---------------------------------------------------------

    def startswith[Rhs: StringValue](self, o: Rhs) -> StartsWith[Self, Rhs]:
        """`self LIKE o || '%'`, without the pattern compile."""
        return StartsWith[Self, Rhs](self.copy(), o.copy())

    def endswith[Rhs: StringValue](self, o: Rhs) -> EndsWith[Self, Rhs]:
        """`self LIKE '%' || o`, without the pattern compile."""
        return EndsWith[Self, Rhs](self.copy(), o.copy())

    def contains[Rhs: StringValue](self, o: Rhs) -> StrContains[Self, Rhs]:
        """Substring search. Literal, not a pattern — `%` and `_` are
        ordinary characters here, which is the whole difference from
        `like`."""
        return StrContains[Self, Rhs](self.copy(), o.copy())

    def like[Rhs: StringValue](self, o: Rhs) -> Like[Self, Rhs]:
        """SQL `LIKE`. A `Shape.scalar` pattern is compiled once per batch
        rather than once per row — see `StringPredicate`."""
        return Like[Self, Rhs](self.copy(), o.copy())

    def ilike[Rhs: StringValue](self, o: Rhs) -> ILike[Self, Rhs]:
        """SQL `ILIKE` — `like`, case-insensitively."""
        return ILike[Self, Rhs](self.copy(), o.copy())

    # -- aggregates ---------------------------------------------------------
    #
    # This cannot fuse: `min` over a string is a
    # bytewise scan keeping the index of the best row, not a scalar fold, so
    # there is no `FoldKernel` to parameterise a fused node on. The **operand**
    # stays typed, so `min(upper(name))` still fuses `upper(name)`.

    def min(self) -> StringMin[Self]:
        """`MIN(self)` — lexicographic (bytewise), matching Arrow's
        `hash_min`. Keeps the input's type."""
        return StringMin[Self](self.copy())

    def max(self) -> StringMax[Self]:
        """`MAX(self)` — lexicographic (bytewise), matching Arrow's
        `hash_max`."""
        return StringMax[Self](self.copy())


trait NumericValue(PrimitiveValue):
    """This value supports arithmetic.

    No members — it exists only so a node can say it needs `+` rather than
    merely a readable lane. Mojo has no conditional conformance, so a single
    leaf cannot be numeric for `int64` and temporal for `date32`; the leaves
    therefore differ while everything above them is shared.
    """

    comptime Type: NumericType
    """Narrowed from `PrimitiveValue`. A sub-trait *can* narrow an associated
    type — a conformer cannot, which is why the domains are traits and not a
    bound on the leaf."""

    def dtype(self, schema: Schema) raises -> DynType:
        """The output type, from `Type` — the schema is the runtime lane's
        concern. Defaulted here rather than on `ComptimeValue` because
        constructing a `Type()` needs it to be `Defaultable`, and the base
        bounds it only by `DataType`. `TemporalValue` keeps its own override
        for the same reason: a timestamp's unit and timezone live in the value,
        not the type."""
        return DynType(Self.Type())

    # -- the aggregate surface ----------------------------------------------
    #
    # `col("amount", int64).sum()` rather than naming a kernel. Trait defaults,
    # so every numeric node gets them for free and no leaf repeats them.
    #
    # They are here and not on `PrimitiveValue` because a fused `Aggregate`
    # binds its input on `NumericValue`. `min`/`max` are ordered rather than
    # arithmetic and belong one level up the moment `AggState` accepts a
    # non-numeric accumulator; until then they would not compile there.

    def sum(self) -> Sum[Self]:
        """`SUM(self)`. Integers widen to int64; floats stay float64."""
        return Sum[Self](self.copy(), String(SUM))

    def product(self) -> Product[Self]:
        """`PRODUCT(self)`."""
        return Product[Self](self.copy(), String(PRODUCT))

    def mean(self) -> Mean[Self]:
        """`AVG(self)`. Accumulates in float64 and divides by the valid count,
        so nulls are excluded rather than counted as zero."""
        return Mean[Self](self.copy(), String(MEAN))

    def min(self) -> Min[Self]:
        """`MIN(self)`. Keeps the input's type."""
        return Min[Self](self.copy(), String(MIN))

    def max(self) -> Max[Self]:
        """`MAX(self)`."""
        return Max[Self](self.copy(), String(MAX))

    def variance[ddof: Int = 0](self) -> Variance[ddof, Self]:
        """`VAR_POP(self)` by default, `VAR_SAMP(self)` at `ddof=1`.

        The divisor is `n - ddof` over the *non-null* values, so `ddof=0` is
        the population variance and `ddof=1` the sample variance. Zero is the
        default because it is Arrow's, and therefore PyArrow's.

        **This does not fuse, and the reason is worth knowing.** Welford's
        recurrence is a fold, but its accumulator is a triple — count, mean,
        `M2` — where `AggState` holds one accumulator column plus one count.
        So `Dispersion` is not a `FoldKernel`, does not conform to `Foldable`,
        and `Aggregate.fuses` answers False. The operand still fuses:
        `(col("a", int64) * 2).variance()` compiles the multiply into one loop
        and only the dispersion materialises.
        """
        return Variance[ddof, Self](self.copy(), String(VARIANCE))

    def stddev[ddof: Int = 0](self) -> StdDev[ddof, Self]:
        """`STDDEV_POP(self)` by default, `STDDEV_SAMP(self)` at `ddof=1`.

        The square root of `variance[ddof]()`, computed from the same Welford
        state rather than by aggregating twice.
        """
        return StdDev[ddof, Self](self.copy(), String(STDDEV))

    def count(self) -> Count[Self]:
        """`COUNT(self)` — the *non-null* values of `self`, not the row count.

        `COUNT(*)` is `count_star()` in `builders.mojo`, which is this same
        aggregate over a literal.
        """
        return Count[Self](self.copy(), String(COUNT))

    # -- operators ----------------------------------------------------------
    # The fluent surface CLAUDE.md mandates: `col("a", int64) > lit(2, int64)`
    # rather than `Gt(...)` by hand. `Rhs`, not `R`, because a trait default's
    # parameter must not collide with a conformer's struct parameter — the
    # binary nodes already bind `L`/`R`.

    def __add__[Rhs: NumericValue](self, o: Rhs) -> Add[Self, Rhs]:
        return Add(self.copy(), o.copy())

    def __sub__[Rhs: NumericValue](self, o: Rhs) -> Sub[Self, Rhs]:
        return Sub(self.copy(), o.copy())

    def __mul__[Rhs: NumericValue](self, o: Rhs) -> Mul[Self, Rhs]:
        return Mul(self.copy(), o.copy())

    # All six comparisons, not just the ordering pair. `Eq` in particular is
    # the shape statistics pruning and Parquet bloom filters both key on, so a
    # lane missing it cannot prune the most common predicate there is.

    def __eq__[Rhs: NumericValue](self, o: Rhs) -> Eq[Self, Rhs]:
        return Eq(self.copy(), o.copy())

    def __ne__[Rhs: NumericValue](self, o: Rhs) -> Ne[Self, Rhs]:
        return Ne(self.copy(), o.copy())

    def __lt__[Rhs: NumericValue](self, o: Rhs) -> Lt[Self, Rhs]:
        return Lt(self.copy(), o.copy())

    def __le__[Rhs: NumericValue](self, o: Rhs) -> Le[Self, Rhs]:
        return Le(self.copy(), o.copy())

    def __gt__[Rhs: NumericValue](self, o: Rhs) -> Gt[Self, Rhs]:
        return Gt(self.copy(), o.copy())

    def __ge__[Rhs: NumericValue](self, o: Rhs) -> Ge[Self, Rhs]:
        return Ge(self.copy(), o.copy())

    # The three operators whose answer is *not* the wider operand. `/` and
    # `**` produce `float64` whatever went in, so they route to `FloatBinary`
    # rather than `NumericBinary`; `%` and `//` stay in the operand type but
    # take Python's sign convention rather than SQL's — see `Mod`.

    def __truediv__[Rhs: NumericValue](self, o: Rhs) -> Div[Self, Rhs]:
        """True division: `5 / 2` is 2.5, not 2. PyArrow's `divide` answers 2
        on integer input; marrow follows Python so that `/`, `//` and `%`
        agree with each other."""
        return Div(self.copy(), o.copy())

    def __floordiv__[Rhs: NumericValue](self, o: Rhs) -> Floordiv[Self, Rhs]:
        """Floored division, the `//` that makes
        `a == (a // b) * b + a % b` hold."""
        return Floordiv(self.copy(), o.copy())

    def __mod__[Rhs: NumericValue](self, o: Rhs) -> Mod[Self, Rhs]:
        """Remainder taking the sign of the **divisor**: `-1 % 3` is 2."""
        return Mod(self.copy(), o.copy())

    def __pow__[Rhs: NumericValue](self, o: Rhs) -> Pow[Self, Rhs]:
        """`self ** o`, in `float64`. `PowKernel` asserts a floating lane, so
        integer operands are widened rather than rejected."""
        return Pow(self.copy(), o.copy())

    # -- unary --------------------------------------------------------------
    #
    # The kernels for all of these have been in `kernels/numeric.mojo` since
    # long before this lane existed and were entirely unreferenced from it.
    # These methods and the two nodes they name (`NumericUnary`, `FloatUnary`)
    # are the wiring, not new compute.

    def __neg__(self) -> Neg[Self]:
        """`-self`, keeping the operand's dtype."""
        return Neg[Self](self.copy())

    def abs(self) -> Abs[Self]:
        """`|self|`. Keeps the dtype, so `abs(INT64_MIN)` wraps rather than
        widening — the same overflow C and Arrow C++ have."""
        return Abs[Self](self.copy())

    def sign(self) -> Sign[Self]:
        """-1, 0 or 1, in the operand's dtype."""
        return Sign[Self](self.copy())

    def floor(self) -> Floor[Self]:
        """Round toward negative infinity, keeping the dtype. On an integer
        operand this is the identity."""
        return Floor[Self](self.copy())

    def ceil(self) -> Ceil[Self]:
        """Round toward positive infinity, keeping the dtype."""
        return Ceil[Self](self.copy())

    def round(self) -> Round[Self]:
        """Round to nearest, keeping the dtype."""
        return Round[Self](self.copy())

    def trunc(self) -> Trunc[Self]:
        """Round toward zero, keeping the dtype. The one that differs from
        `floor` on negatives."""
        return Trunc[Self](self.copy())

    def sqrt(self) -> Sqrt[Self]:
        """Square root, in `float64` — not closed over the integers, so
        unlike `abs` this cannot keep the operand's dtype."""
        return Sqrt[Self](self.copy())

    def exp(self) -> Exp[Self]:
        """`e ** self`, in `float64`."""
        return Exp[Self](self.copy())

    def ln(self) -> Ln[Self]:
        """Natural logarithm, in `float64`. `ln(0)` is `-inf` and `ln(-1)` is
        NaN — IEEE 754 values, not nulls."""
        return Ln[Self](self.copy())

    # -- value predicates ---------------------------------------------------

    def is_nan(self) -> IsNan[Self]:
        """True where the value is NaN, **NULL where the value is null**.

        That second half is the whole distinction from `is_null`:
        `is_null(NULL)` is TRUE, `is_nan(NULL)` is NULL. Floating operands
        only — the node asserts it at compile time.
        """
        return IsNan[Self](self.copy())

    def is_inf(self) -> IsInf[Self]:
        """True where the value is an infinity of either sign, NULL where it
        is null — as `is_nan`."""
        return IsInf[Self](self.copy())


trait TemporalValue(PrimitiveValue):
    """Marker: ordered and comparable, but not arithmetic.

    `date + date` is meaningless, so temporal values are deliberately not
    accepted by `NumericBinary`. Comparison, `min`/`max` and grouping all work,
    because those bind on `PrimitiveValue`.
    """

    comptime Type: TemporalType

    # -- aggregates ---------------------------------------------------------
    #
    # `min`/`max` over a temporal column *is* a genuine `AggState` fold — the
    # accumulator keeps the input dtype and `MinMax.acc_dtype` carries its unit
    # and timezone through. It cannot be the **fused** one today:
    # `FusedAggregateOperator.__init__` builds its accumulator dtype from `Self.A.Type()`
    # and a `TemporalType` is not `Defaultable`. Routing through the runtime
    # lane is correct but materialises; moving the accumulator dtype out of
    # `__init__` and into first push is what would fuse it, and is owed.

    def min(self) -> Min[Self]:
        """`MIN(self)`. Keeps the input's dtype — unit and timezone
        included."""
        return Min[Self](self.copy())

    def max(self) -> Max[Self]:
        """`MAX(self)`. Keeps the input's dtype — unit and timezone
        included."""
        return Max[Self](self.copy())

    # -- comparison ---------------------------------------------------------
    #
    # `TemporalCompare` and its six aliases existed with **no callers at
    # all**: naming `TemporalGt(a, b)` by hand was the only way to reach the
    # node, and `TemporalNe`/`TemporalLe`/`TemporalGe` had not even that. A
    # family whose whole documented purpose is "ordered and comparable, but
    # not arithmetic" could not be compared.
    #
    # `Rhs`, not `R`: a trait default's parameter must not collide with a
    # conformer's struct parameter, and `TemporalCompare` binds `L`/`R`.
    # Both sides must already share a representation — `TemporalCompare`
    # asserts the width at compile time and checks the unit per batch, because
    # cross-unit coercion is a promotion decision this lane has not made.

    def __eq__[Rhs: TemporalValue](self, o: Rhs) -> TemporalEq[Self, Rhs]:
        return TemporalEq(self.copy(), o.copy())

    def __ne__[Rhs: TemporalValue](self, o: Rhs) -> TemporalNe[Self, Rhs]:
        return TemporalNe(self.copy(), o.copy())

    def __lt__[Rhs: TemporalValue](self, o: Rhs) -> TemporalLt[Self, Rhs]:
        return TemporalLt(self.copy(), o.copy())

    def __le__[Rhs: TemporalValue](self, o: Rhs) -> TemporalLe[Self, Rhs]:
        return TemporalLe(self.copy(), o.copy())

    def __gt__[Rhs: TemporalValue](self, o: Rhs) -> TemporalGt[Self, Rhs]:
        return TemporalGt(self.copy(), o.copy())

    def __ge__[Rhs: TemporalValue](self, o: Rhs) -> TemporalGe[Self, Rhs]:
        return TemporalGe(self.copy(), o.copy())

    # -- field extraction ---------------------------------------------------
    #
    # Nine kernels that `kernels/temporal.mojo` has carried unreferenced.
    # `int32` for every one, which is Arrow C++'s and PyArrow's answer.

    def year(self) -> Year[Self]:
        """The proleptic Gregorian year."""
        return Year[Self](self.copy())

    def month(self) -> Month[Self]:
        """Month of year, 1-12."""
        return Month[Self](self.copy())

    def day(self) -> Day[Self]:
        """Day of month, 1-31."""
        return Day[Self](self.copy())

    def hour(self) -> Hour[Self]:
        """Hour of day, 0-23."""
        return Hour[Self](self.copy())

    def minute(self) -> Minute[Self]:
        """Minute of hour, 0-59."""
        return Minute[Self](self.copy())

    def second(self) -> Second[Self]:
        """Second of minute, 0-59. Sub-second digits are dropped, not
        rounded."""
        return Second[Self](self.copy())

    def quarter(self) -> Quarter[Self]:
        """Quarter of year, 1-4."""
        return Quarter[Self](self.copy())

    def day_of_week(self) -> DayOfWeek[Self]:
        """ISO weekday, **Monday = 0** — PyArrow's default, not DuckDB's
        Sunday = 0."""
        return DayOfWeek[Self](self.copy())

    def day_of_year(self) -> DayOfYear[Self]:
        """Day of year, 1-366."""
        return DayOfYear[Self](self.copy())

    # -- truncation ---------------------------------------------------------

    def date_trunc(self, unit: String) raises -> DateTrunc[Self]:
        """Floor to a `second`/`minute`/`hour`/`day`/`month`/`quarter`/`year`
        boundary, keeping this value's own type.

        The unit is parsed **here**, at plan-construction time, so a
        misspelling raises when the query is built rather than on the first row
        that evaluates it. That is the entire reason `CalendarUnit` is a type
        and not a `String`: past this call an unsupported unit is
        unrepresentable.
        """
        return DateTrunc[Self](self.copy(), CalendarUnit.parse(unit))


# ---------------------------------------------------------------------------
# ListValue — the family with no lane
# ---------------------------------------------------------------------------
trait ListValue(ComptimeValue):
    """A comptime node producing a list column.

    **Declares no `lane`, and that is the point.** Every other family answers
    `lane` with something a register holds — `SIMD` for numeric and bool, a
    `String` for text. A list *element* is a whole sub-array, so there is no
    per-element value to return and nothing a fused loop could do with one.

    So this family exists to be **consumed** rather than to be an operand:
    `bind` resolves the column, and the operations over it — `length`,
    `contains` — are nodes of *other* families that read the bound list and
    produce a fixed-width lane. `ListLength` is a `NumericValue`; a list never
    flows through arithmetic itself.

    That is why the trait is small: `Type`, `Bound`, `bind`, `validity`. There
    is no fused `evaluate` default either, because without a lane there is
    nothing to drive — a list leaf hands back its own column.
    """

    comptime Type: ListLikeType

    # No associated `Bound`. Every other family has one because a *composite*
    # node's bound is a tuple of its operands' — `Add`'s is
    # `Tuple[L.Bound, R.Bound]`. There are no composite list nodes: a list is
    # only ever read from a column, so the bound is always the column itself
    # and naming it as a variable would be a variable with one value. It also
    # keeps `ArrayLengthKernel` able to infer its own `T`, which an opaque
    # associated type defeats.

    def bind(
        self, batch: StructArray, bindings: Bindings
    ) raises -> ListLikeArray[Self.Type]:
        ...

    def validity(
        self, bound: ListLikeArray[Self.Type]
    ) raises -> Optional[Bitmap[mut=False]]:
        ...


# ---------------------------------------------------------------------------
# BoolValue — the family whose lane is bit-packed
# ---------------------------------------------------------------------------
trait BoolValue(ComptimeValue):
    """A comptime node producing a bit-packed boolean column.

    Separate from `NumericValue` because the output is *packed*: a lane yields
    `SIMD[DType.bool, W]` and the driver writes bits, not elements. That is a
    different destination, not a different dtype, which is why `Type` is fixed
    here rather than declared per node — a bool node has no choice about what
    it produces.
    """

    comptime Type = BoolType

    def dtype(self, schema: Schema) raises -> DynType:
        """The output type, from `Type` — the schema is the runtime lane's
        concern. Defaulted here rather than on `ComptimeValue` because
        constructing a `Type()` needs it to be `Defaultable`, and the base
        bounds it only by `DataType`. `TemporalValue` keeps its own override
        for the same reason: a timestamp's unit and timezone live in the value,
        not the type."""
        return DynType(Self.Type())

    comptime Bound: Copyable & Deinitable
    """Everything the lane loop needs, resolved once per batch.

    A column leaf's is its typed column; a literal's is nothing; a binary node's
    is `Tuple[L.Bound, R.Bound]`. Declared per concrete struct rather than
    defaulted: a trait default cannot reduce at a `-> Self.Bound` return site
    unless the bound is `ImplicitlyCopyable`, and marrow's array types
    deliberately are not.

    `expr/` called this `State`, which could mean anything. It is specifically
    *this subtree's column references, bound to this batch* — the stage between
    an expression and a per-element read.
    """

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        """Resolve this subtree against `batch`, once, before the lane loop.

        Every schema lookup and every `Variant` unwrap happens here so that
        `lane` does none. That removal is the optimisation.
        """
        ...

    comptime NativeType: DType
    """The **operand** width, which sizes the SIMD lane — not the output.

    A comparison over `int64` iterates 64-bit lanes even though it emits one
    bit per row, so `W` follows the operands. Sizing it from the output would
    give a `W` wide enough to overflow the register the operands are loaded
    into. A node with two operands of different widths takes the wider.
    """

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        """One fused bool pass: bit-pack a `Bitmap` from `lane`.

        The numeric default's sibling, and separate for the one reason above —
        the destination is a bitmap, so `apply` takes its bit-packing overload
        and the lane width comes from `NativeType`.
        """
        var length = len(batch)
        var bound = self.bind(batch, bindings)
        var bits = Bitmap.alloc_uninit(length)

        @always_inline
        def producer[W: Int](i: Int) {imm} -> SIMD[DType.bool, W]:
            return self.lane[W](bound, i)

        apply[Self.NativeType](bits.view(), producer)

        var v = self.validity(bound)
        return Datum(
            BoolArray(
                length=length,
                nulls=v.value().unset_count() if v else 0,
                offset=0,
                bitmap=v^,
                buffer=bits.to_immutable(),
            ).to_dyn()
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity, or `None` when it cannot be null.

        Structural, exactly as `NumericValue.validity` is: a comparison is null
        where an operand is, and never because of what the operands *say*.

        **Kleene `AND`/`OR` do not belong to this family**, and that is the
        reason this signature is allowed to stay simple. Three-valued logic
        decides nulls from operand values, so it can answer neither from the
        `Bound` nor per lane — measured at 4-10x slower per-lane than the
        bitmap algebra in `kernels.boolean._kleene`, which runs 64 bits per
        instruction (`bench_boolean.mojo`, 2026-08-22). They get their own
        family, whose `evaluate` calls the kernel rather than driving a lane.
        """
        ...

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        ...


# ---------------------------------------------------------------------------
# Unnamed, ColumnBound — what the families cut across
# ---------------------------------------------------------------------------
# The two traits below are **not** families. A family answers "what shape does
# this produce" and every node belongs to exactly one; these answer two
# narrower questions that recur *across* families, and a node opts into each
# independently. `NumericCompare` is a `BoolValue` and `Unnamed`; `ListLength`
# is a `NumericValue`, `Unnamed` *and* `ColumnBound`.
#
# They exist because the answers were copied: nine nodes across four files
# spelled `return String()`, and seven spelled
# `return bound.to_data().owned_validity()` — byte-identical bodies a reader
# has to diff to know are the same. A trait default states it once, and the
# conformance list says which nodes mean it.
#
# **What could not be factored, and why.** `columns()` duplicates just as
# widely — five nodes spell `[self._name.copy()]`, five spell
# `merged(self.l.columns(), self.r.columns())` — and it stays duplicated,
# because both bodies read a *field*. Mojo rejects a `var` requirement on a
# trait outright ("traits do not support 'var' fields; use 'comptime' to
# declare associated types"), so no default can reach `self._name` or
# `self.l`; routing through an abstract accessor would only trade one one-line
# body per node for another. The two below are factorable precisely because
# neither reads `self`: `Unnamed.name` reads nothing, and
# `ColumnBound.validity` reads only its `bound` argument.


trait Unnamed(ComptimeValue):
    """This node has no name of its own.

    Every computed node -- arithmetic, comparison, boolean, `CASE WHEN`,
    `array_length` -- answers `name()` with the empty string, because a name is
    something a *reference* carries and a computation does not. The planner
    supplies one where output needs a label.

    The conformance is the documentation. Before this trait a reader could only
    learn that `NumericBinary` and `StringCompare` agree by comparing their
    bodies; now the two say so in their conformance list, and a node that
    genuinely has a name (`Column`, `Literal`) is visibly absent from it.

    A same-signature override of a trait default is ordinary -- `Column` and
    `Literal` simply do not conform here rather than overriding. What is
    forbidden, and what this deliberately does not do, is a default whose
    *return type* a conformer must change: those become competing overloads and
    every call site reports `ambiguous call to 'name'`.
    """

    def name(self) -> String:
        return String()


trait ColumnBound(ComptimeValue):
    """This node's `Bound` is an array, so its validity is that array's.

    The families disagree about where validity comes from, which is why
    `validity` is declared on each of them rather than on `ComptimeValue`.
    This trait names the case where the answer is trivial: the bound *is* a
    materialised column, so the result's nulls are already recorded in it and
    there is nothing to intersect.

    Two quite different nodes land here, and it is worth being explicit that
    they are the same case:

    - **A leaf** -- `Column`, `TemporalColumn`, `BoolColumn`, `StringColumn`,
      `ListColumn`. `bind` resolved the column out of the batch, and the column
      carries its own bitmap. No second lookup, no re-read.
    - **A node that computed its result in `bind`** -- `ListLength` and
      `CaseWhen`. Neither fuses element-wise: each runs a kernel over the whole
      batch and `lane` reads the answer back. `CaseWhen`'s validity is
      emphatically *not* structural -- a row is null when the **selected**
      branch was null -- but that is exactly why it belongs here: the rule was
      already applied by the kernel, so the answer lives in the bound result
      and nowhere else.

    That is the honest boundary. A node whose validity is *structural* --
    `NumericBinary`, `NumericCompare`, `TemporalCompare`, `StringCompare` --
    intersects its operands' bitmaps and does not conform, because its `Bound`
    is a tuple of its operands' bounds and not a column at all.

    `Bound` is narrowed from the family's `Copyable & Deinitable` to `Array`,
    which is what lets the default body call `to_data()`. A sub-trait may
    narrow an associated type; a conformer may not, which is why this is a
    trait rather than a bound written on each node.
    """

    comptime Bound: Array

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return bound.to_data().owned_validity()
