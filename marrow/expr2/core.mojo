"""The spine of the expression layer: what an expression *is*, and its box.

There is deliberately **no `Value` trait**. An expression is the composition of
three traits, each named for who asks:

    Analyzable   a rewriter asks these
    Evaluable    the executor asks this
    Writable     a human asks this (stdlib)

`expr/` carried one `Value` trait that executed, materialised, pruned, named
itself, rendered itself, listed its columns, reported whether it was a bare
column, reported its validity, *and* constructed two kinds of distinct-count
aggregate. Nine responsibilities behind one name is how seven trampoline slots
accreted without anyone deciding to add them. Splitting by asker makes "should
this method exist?" answerable: it exists if someone named can be pointed at.

The split has a second effect, which is the point of doing it here rather than
as a tidy-up. **A rewriter holds an `Analyzable` and therefore cannot execute
it** — not by convention, but because the type has no `evaluate`. The one rule
that would want to (constant folding) is also the one LLVM already performs for
the comptime lane, so it belongs to the runtime lane alone and gets its
evaluator explicitly.

Both lanes satisfy this common surface, which is what lets `DynValue` hold
either. Lane mixing already worked this way in `expr/` — `IsIn[A: Value]` and
its two siblings bind on the *common* trait, never on a family — so naming the
lanes changes nothing about how they meet.

This module is a **leaf**: it imports the containers and the interval kernel,
and nothing else under `marrow.expr2`.
"""

from std.memory import ArcPointer
from std.utils import Variant

from ..arrays import DynArray
from ..scalars import DynScalar
from ..dtypes import DynType
from ..kernels.interval import Interval
from ..schema import Schema
from ..tabular import RecordBatch
from .pruning import PruneStats


# ---------------------------------------------------------------------------
# Datum — the wire format between stages
# ---------------------------------------------------------------------------
comptime Datum = Variant[DynScalar, DynArray]
"""`Scalar | Array`, Arrow's Datum and DataFusion's ColumnarValue.

A literal stays a scalar until something needs it as a column, so a predicate
over a constant never allocates one.
"""


def into_array(d: Datum, n: Int) raises -> DynArray:
    """Force `d` to a column of length `n`, broadcasting a scalar.

    The single place laziness ends. Keeping it here rather than in either lane
    is what lets the lanes stop importing each other for it.
    """
    if d.isa[DynScalar]():
        return d[DynScalar].repeat(n)
    return d[DynArray].copy()


# ---------------------------------------------------------------------------
# Shape — scalar or columnar
# ---------------------------------------------------------------------------
struct Shape(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """Whether an expression yields one value or one per row.

    A value type rather than a bare `Int`, for the reason `JoinKind` is one:
    `0` and `1` are interchangeable to the compiler and to a reader, and the
    two callers who ask this — `into_array`, deciding whether to broadcast, and
    the planner, deciding whether a projection needs materialising — would each
    be re-deriving the convention from a comment.
    """

    var _code: UInt8

    comptime scalar = Shape(0)
    """One value for the whole batch. A literal; a reduction's result."""

    comptime columnar = Shape(1)
    """One value per row."""

    def __init__(out self, code: UInt8):
        self._code = code

    def __eq__(self, other: Self) -> Bool:
        return self._code == other._code

    def __ne__(self, other: Self) -> Bool:
        return self._code != other._code

    def write_to[W: Writer](self, mut writer: W):
        writer.write("scalar" if self == Shape.scalar else "columnar")


# ---------------------------------------------------------------------------
# Analyzable — what a rewriter asks
# ---------------------------------------------------------------------------
trait Analyzable:
    """Questions a rewriter asks of an expression it cannot open.

    The comptime lane's structure *is* its type, so nothing outside can inspect
    it; a node must answer for itself. That constraint is real and is why these
    methods exist at all. What is *not* forced is scattering them across the
    expression's own interface as though answering a rewriter were part of
    being an expression — hence this trait, named for the asker.

    Every method is **total**: a node that is not a column still answers
    `as_column`, a node with no bounds still answers `interval`. Totality is
    what lets a caller compose answers without asking what kind of node it
    holds — and, at the type level, it is what makes a conditional rewrite
    reduce, since neither branch can name something that does not exist.
    """

    def columns(self) -> List[String]:
        """Every column name this expression reads, first-seen order, no repeats.

        Projection pushdown is exactly this question asked of a predicate.
        """
        ...

    def name(self) -> String:
        """What this expression is called, or `""` if it has no name of its own.

        A column is called by its column name; a literal by how it reads, so
        `lit(1)` is `"1"` — matching SQL, where `SELECT 1` yields a column
        named `1`. A computed expression has no name and answers `""`; the
        planner supplies `key0`, `sum`, or whatever the caller aliased.

        **Bare-column-ness is a composition, not a second method.** Four
        callers need to know whether an expression is exactly a column
        reference — a projected pass-through must carry its source `Field`
        whole (dtype, `nullable`, metadata), `GROUP BY d` must name its output
        `d` rather than `key0`, and a join must reject a computed key. All four
        ask:

            value.name() != "" and len(value.columns()) == 1

        which separates the three cases without a slot of its own, because
        `columns()` is empty for a literal and `name()` is empty for anything
        computed.

        A **name**, not a position: a node does not know the schema it will be
        resolved against. `expr/` spelled this `bound_column(schema) raises ->
        Int`, which conflated *are you a column?* with *where is it?* and made
        the first able to raise. The planner holds the schema and does the
        lookup; the node only says what it is.
        """
        ...

    def dtype(self, schema: Schema) raises -> DynType:
        """The type this expression produces over `schema`.

        A planner needs this to build a `Project`'s or an `Aggregate`'s output
        schema. `expr/` had no such method and instead **evaluated every
        expression against a zero-row batch** to see what came back — which
        works, but makes schema computation depend on execution, allocates, and
        can raise from deep inside a kernel for what is a static question.

        It takes a `schema` because only the comptime lane knows its type
        outright: a `RuntimeValue` column reference learns its type by looking
        itself up. The comptime lane ignores the argument and answers from
        `Type`.
        """
        ...

    def interval(self, stats: PruneStats) raises -> Interval:
        """The range this expression can produce, given per-column `[min, max]`.

        Drives statistics pruning: a definite "cannot be true" skips a row
        group without decoding it. Correctness never depends on the answer — a
        scan still applies the exact predicate — so an imprecise `Interval`
        costs time, never a wrong row.
        """
        ...


# ---------------------------------------------------------------------------
# Evaluable — what the executor asks
# ---------------------------------------------------------------------------
trait Evaluable:
    """Produce a column from a batch.

    One method, because that is the entire physical contract at this level. The
    comptime lane satisfies it by binding its column references once per batch
    and running a fused per-element loop; the runtime lane satisfies it by
    materialising a `DynArray` per node. The difference is invisible here,
    which is why a `Relation` can hold either.
    """

    comptime shape: Shape
    """`Shape.scalar` or `Shape.columnar`.

    Lets a caller know whether `evaluate` will broadcast before it calls, so a
    literal-only expression need not materialise a column to find out.

    Lowercase because it is a comptime *value*, not a type — the same spelling
    kernels use for `comptime name`.
    """

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        ...


comptime Value = Analyzable & Evaluable & Writable & Copyable & Deinitable
"""What every expression is, in both lanes — a *name for the composition*, not
a trait of its own.

`expr/` had a `Value` **trait** carrying nine responsibilities. This keeps the
name, which the tree and its docs already use, while the substance moves into
`Analyzable` and `Evaluable`. Nothing conforms to `Value`; things conform to
the traits it names.

`Copyable & Deinitable` are here because `DynValue` boxes into an
`ArcPointer`, which requires them (`Copyable` already implies `Movable`). They
are a storage requirement, not part of what it means to be an expression, which
is why they sit in the alias rather than in `Analyzable` or `Evaluable`.
"""


# ---------------------------------------------------------------------------
# DynValue — the box, and the only place the two lanes meet
# ---------------------------------------------------------------------------
struct DynValue(Copyable, Movable, Writable):
    """An expression of either lane, erased.

    **This box is the feature, not overhead.** It is what lets a dynamically
    composed plan hold comptime-fused expressions — measured at 1.46 MB against
    4.91 MB for the same plan with runtime expressions. Removing it would force
    a choice between a fully comptime plan (which instantiates per plan shape
    and no Python frontend can build) and runtime expressions everywhere (which
    is the 4.91 MB configuration).

    Six slots, each traceable to a named asker: four for `Analyzable`, one for
    `Evaluable`, one for `Writable`. `expr/` carried seven and had no `dtype`,
    computing output types by evaluating against a zero-row batch instead.
    Dropped: `name()`, which duplicated what `name` and `write` already
    answered, and `resolve_names`, a rewrite that is a no-op in the comptime
    lane and therefore belongs on the runtime value rather than on every boxed
    expression in every binary.

    Deliberately **not** conforming to the traits it erases. A box may hold a
    trait-bound value; it should not be one. `DynValue` exposes the same
    surface as its own API, and nothing in the tree asks it to substitute for a
    typed value in generic code.
    """

    var _boxed: ArcPointer[NoneType]
    var _evaluate: def (ArcPointer[NoneType], RecordBatch) thin raises -> Datum
    var _columns: def (ArcPointer[NoneType]) thin -> List[String]
    var _name: def (ArcPointer[NoneType]) thin -> String
    var _dtype: def (ArcPointer[NoneType], Schema) thin raises -> DynType
    var _interval: def (ArcPointer[NoneType], PruneStats) thin raises -> Interval
    var _write: def (ArcPointer[NoneType]) thin -> String
    var _shape: Shape

    # -- trampolines --------------------------------------------------------
    # One instantiation per boxed type, wired at construction. There is no
    # registry and nothing names every value type in one place, so a type that
    # is never boxed costs nothing in the binary.

    @staticmethod
    def _evaluate_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> Datum:
        return rebind[ArcPointer[V]](ptr)[].evaluate(batch)

    @staticmethod
    def _columns_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType]) -> List[String]:
        return rebind[ArcPointer[V]](ptr)[].columns()

    @staticmethod
    def _name_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
        return rebind[ArcPointer[V]](ptr)[].name()

    @staticmethod
    def _dtype_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], schema: Schema) raises -> DynType:
        return rebind[ArcPointer[V]](ptr)[].dtype(schema)

    @staticmethod
    def _interval_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], stats: PruneStats) raises -> Interval:
        return rebind[ArcPointer[V]](ptr)[].interval(stats)

    @staticmethod
    def _write_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType]) -> String:
        return String(rebind[ArcPointer[V]](ptr)[])

    @implicit
    def __init__[V: Value](out self, value: V):
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._evaluate = Self._evaluate_tramp[V]
        self._columns = Self._columns_tramp[V]
        self._name = Self._name_tramp[V]
        self._dtype = Self._dtype_tramp[V]
        self._interval = Self._interval_tramp[V]
        self._write = Self._write_tramp[V]
        self._shape = V.shape

    # -- the erased surface -------------------------------------------------

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        return self._evaluate(self._boxed, batch)

    def columns(self) -> List[String]:
        return self._columns(self._boxed)

    def name(self) -> String:
        return self._name(self._boxed)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._dtype(self._boxed, schema)

    def interval(self, stats: PruneStats) raises -> Interval:
        return self._interval(self._boxed, stats)

    def shape(self) -> Shape:
        """The boxed value's `shape`, read at construction.

        A field rather than a seventh trampoline: it is a constant per boxed
        type, so calling through a pointer to fetch it would pay a call to
        learn something already known.
        """
        return self._shape

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._write(self._boxed))
