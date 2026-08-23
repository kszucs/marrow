"""The logical layer: an immutable description of a query.

Paired with `physical.mojo`, which holds what these become when they run.

A `Relation` says *what* to compute. It owns nothing that runs, so a plan is
freely copyable, shareable, inspectable and — once the optimizer exists —
rewritable. `to_operator(ctx)` turns it into the physical operator that owns
the running state.

**Two methods, not eight.** `expr/`'s `DynRelation` carries `schema`,
`to_operator`, `write`, `drop`, `kind`, `with_predicate`, `with_projection`
and `children`. The last four exist for an optimizer that was never finished,
and two of them cannot express any rewrite that changes a node's type or arity
— which is every rewrite worth having. They are not reproduced here. When a
rule needs to walk or rebuild a plan, `children` and its inverse arrive with
that rule and are shaped by it.

The one property that must survive whatever arrives: **nothing may name every
node type in one place.** `DynRelation.__init__[T]` wires trampolines per
constructed type, with no registry, which is why `kernels::sort` occupies zero
bytes in a binary that never sorts. A `match` over all kinds would make every
operator reachable from any plan.
"""

from std.memory import ArcPointer

from ..execution import ExecContext
from ..schema import Schema, schema
from ..tabular import RecordBatch
from ..dtypes import DynType, Field, field
from .physical import (
    Datum,
    AggregateOperator,
    BatchSourceOperator,
    DynOperator,
    LimitOperator,
    Pipeline,
    FilterOperator,
    ProjectOperator,
    SortOperator,
)


# ---------------------------------------------------------------------------
# Shape — scalar or columnar
# ---------------------------------------------------------------------------
struct Shape(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """Whether an expression yields one value or one per row.

    A value type rather than a bare `Int`, for the reason `JoinKind` is one:
    `0` and `1` are interchangeable to the compiler and to a reader, and the
    two callers who ask this — `Datum.to_array`, deciding whether to broadcast, and
    the planner, deciding whether a projection needs materialising — would each
    be re-deriving the convention from a comment.
    """

    var _code: UInt8

    comptime scalar = Shape(0)
    """One value for the whole batch. A literal; an aggregate's result."""

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
    `name`, with `""`. Totality is what lets a caller compose answers without
    asking what kind of node it holds — and, at the type level, it is what
    makes a conditional rewrite reduce, since neither branch can name
    something that does not exist.
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


# ---------------------------------------------------------------------------
# Executable — what the executor asks
# ---------------------------------------------------------------------------
trait Executable(Copyable, Deinitable):
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

    def to_operator(self, grouped: Bool) raises -> DynOperator[Datum]:
        """The stateful thing that runs this value.

        **One method for every logical node.** A `Relation` becomes a pipeline,
        and a value — elementwise or reduction alike — becomes an `Operator`
        producing a `Datum`. The two differ only in *when* they answer `Some`:
        an elementwise value answers from `push`, an aggregate accumulates and
        answers from `finish`. That is the same distinction `Filter` and
        `Aggregate` already make one layer up, so it needs no second trait and
        no second box.

        `grouped` picks a fold's placement and is ignored by everything else.
        It is the one asymmetry left, and it goes away when placement becomes
        part of the value's type rather than a plan-time flag.

        **There is deliberately no `evaluate` beside this.** A logical node is
        stateless — it describes an expression and nothing more — so it has no
        business exposing a way to run itself. Anything that runs owns state,
        and that is what a processor is for. `evaluate` survives only *inside*
        each lane, as the fused driver the lane's processor calls, and it is
        not visible on `Value` or on `DynValue`.

        No default: a node cannot supply one without naming its lane's
        processor, and the two lanes have different ones. Each lane defaults it
        for its own family.
        """
        ...


comptime Value = Analyzable & Executable & Writable
"""What every expression is, in both lanes — a *name for the composition*, not
a trait of its own.

`expr/` had a `Value` **trait** carrying nine responsibilities. This keeps the
name, which the tree and its docs already use, while the substance moves into
`Analyzable` and `Executable`. Nothing conforms to `Value`; things conform to
the traits it names.

`Copyable & Deinitable` are here because `DynValue` boxes into an
`ArcPointer`, which requires them (`Copyable` already implies `Movable`). They
are a storage requirement, not part of what it means to be an expression, which
is why they sit in the alias rather than in `Analyzable` or `Executable`.
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
    `Executable`, one for `Writable`. `expr/` carried seven and had no `dtype`,
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
    var _columns: def(ArcPointer[NoneType]) thin -> List[String]
    var _name: def(ArcPointer[NoneType]) thin -> String
    var _dtype: def(ArcPointer[NoneType], Schema) thin raises -> DynType
    var _write: def(ArcPointer[NoneType]) thin -> String
    var _to_processor: def(
        ArcPointer[NoneType], Bool
    ) thin raises -> DynOperator[Datum]
    var _shape: Shape

    # -- trampolines --------------------------------------------------------
    # One instantiation per boxed type, wired at construction. There is no
    # registry and nothing names every value type in one place, so a type that
    # is never boxed costs nothing in the binary.

    @staticmethod
    def _columns_tramp[V: Value](ptr: ArcPointer[NoneType]) -> List[String]:
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
    def _to_processor_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], grouped: Bool) raises -> DynOperator[Datum]:
        return rebind[ArcPointer[V]](ptr)[].to_operator(grouped)

    @staticmethod
    def _write_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
        return String(rebind[ArcPointer[V]](ptr)[])

    @implicit
    def __init__[V: Value](out self, value: V):
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._columns = Self._columns_tramp[V]
        self._name = Self._name_tramp[V]
        self._dtype = Self._dtype_tramp[V]
        self._write = Self._write_tramp[V]
        self._to_processor = Self._to_processor_tramp[V]
        self._shape = V.shape

    # -- the erased surface -------------------------------------------------

    def columns(self) -> List[String]:
        return self._columns(self._boxed)

    def name(self) -> String:
        return self._name(self._boxed)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._dtype(self._boxed, schema)

    def to_operator(self, grouped: Bool) raises -> DynOperator[Datum]:
        """The stateful thing that runs this value.

        The slot `DynAggValue._acc` used to occupy, on the one box that now
        holds every value. An aggregate reaches its `FoldOperator` through here; an
        elementwise value reaches an `EvalOperator`. The caller cannot tell,
        which is the point.
        """
        return self._to_processor(self._boxed, grouped)

    def shape(self) -> Shape:
        """The boxed value's `shape`, read at construction.

        A field rather than a seventh trampoline: it is a constant per boxed
        type, so calling through a pointer to fetch it would pay a call to
        learn something already known.
        """
        return self._shape

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._write(self._boxed))


trait Relation(Copyable, Deinitable, Movable):
    """An immutable description of a query."""

    def schema(self) -> Schema:
        """The columns this relation produces.

        Computed at construction and stored, not derived on demand: a caller
        asks for it once per plan node while building the node above, and a
        `Filter` would otherwise re-derive its input's schema every time.
        """
        ...

    def to_operator(self, ctx: ExecContext) raises -> Pipeline:
        """The running operator for this description.

        Takes a context because this is the seam where one logical operator may
        become different physical ones — a hash join or a merge join, a CPU or
        a GPU pass. Nothing exercises that yet; the argument is here because
        removing the seam is the part that would be hard to undo.
        """
        ...


struct DynRelation(Copyable, Movable, Writable):
    """A `Relation` of any operator, erased.

    Copyable, unlike `Pipeline`: a plan owns nothing that runs, so sharing
    one is an `ArcPointer` bump and two consumers cannot interfere.
    """

    var _data: ArcPointer[NoneType]
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_to_processor: def(
        ArcPointer[NoneType], ExecContext
    ) thin raises -> Pipeline
    var _virt_write: def(ArcPointer[NoneType]) thin -> String

    @staticmethod
    def _schema_tramp[R: Relation](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[R]](ptr)[].schema()

    @staticmethod
    def _to_processor_tramp[
        R: Relation
    ](ptr: ArcPointer[NoneType], ctx: ExecContext) raises -> Pipeline:
        return rebind[ArcPointer[R]](ptr)[].to_operator(ctx)

    @staticmethod
    def _write_tramp[
        R: Relation & Writable
    ](ptr: ArcPointer[NoneType]) -> String:
        return String(rebind[ArcPointer[R]](ptr)[])

    @implicit
    def __init__[R: Relation & Writable](out self, value: R):
        var ptr = ArcPointer[R](value.copy())
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_schema = Self._schema_tramp[R]
        self._virt_to_processor = Self._to_processor_tramp[R]
        self._virt_write = Self._write_tramp[R]

    def schema(self) -> Schema:
        return self._virt_schema(self._data)

    def to_operator(self, ctx: ExecContext) raises -> Pipeline:
        return self._virt_to_processor(self._data, ctx)

    def execute(
        self, ctx: ExecContext = ExecContext.auto()
    ) raises -> RecordBatch:
        """Run this plan and drain it into one batch."""
        var p = self.to_operator(ctx)
        return p.collect(self.schema())

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._virt_write(self._data))


struct InMemoryTable(Relation, Writable):
    """A batch already in memory, as a source."""

    var _batch: RecordBatch

    def __init__(out self, var batch: RecordBatch):
        self._batch = batch^

    def schema(self) -> Schema:
        return self._batch.schema.copy()

    def to_operator(self, ctx: ExecContext) raises -> Pipeline:
        """The one relation that *creates* a pipeline; every other appends."""
        return Pipeline(BatchSourceOperator(self._batch.copy()))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("InMemoryTable(", self._batch.num_rows(), " rows)")


struct Filter(Relation, Writable):
    """Rows of `input` where `predicate` is true.

    Schema-preserving: a filter changes which rows survive, never which columns
    exist. That is why it can hold its input's schema rather than computing
    one, and it is also why a filter cannot narrow the columns it compacts —
    knowing what is read downstream is a *physical* property this layer does
    not have.
    """

    var _input: DynRelation
    var _predicate: DynValue

    def __init__(out self, var input: DynRelation, var predicate: DynValue):
        self._input = input^
        self._predicate = predicate^

    def schema(self) -> Schema:
        return self._input.schema()

    def to_operator(self, ctx: ExecContext) raises -> Pipeline:
        var pipe = self._input.to_operator(ctx)
        pipe.append(
            FilterOperator(self._predicate.to_operator(False), ctx.copy())
        )
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Filter(", self._input, ", ", self._predicate, ")")


struct Project(Relation, Writable):
    """`SELECT <values> AS <names>` — a new set of columns over the same rows.

    This is the node that gives `Analyzable.dtype` and `Analyzable.name` their
    callers: the output schema is one field per value, and a field needs both a
    type and a name.
    """

    var _input: DynRelation
    var _names: List[String]
    var _values: List[DynValue]
    var _schema: Schema

    def __init__(
        out self,
        var input: DynRelation,
        var names: List[String],
        var values: List[DynValue],
    ) raises:
        if len(names) != len(values):
            raise Error(
                "project: ", len(names), " names but ", len(values), " values"
            )
        self._schema = Self._output_schema(input.schema(), names, values)
        self._input = input^
        self._names = names^
        self._values = values^

    @staticmethod
    def _output_schema(
        input: Schema, names: List[String], values: List[DynValue]
    ) raises -> Schema:
        """One field per value, computed once at construction.

        A value that is **exactly a bare column** carries its source `Field`
        over whole — dtype, `nullable` and metadata — rather than being
        rebuilt from its dtype alone. Rebuilding loses `nullable`, so
        projecting a column produced a *different* schema for it than
        selecting the same column did; `expr/` records that as a real
        divergence, with `nullable` False becoming True.

        Bare-column-ness is the composition `name() != "" and
        len(columns()) == 1`: a literal is named but reads nothing, and
        anything computed has no name. This is its first caller, which is why
        it is spelled here rather than kept as a method nobody used.
        """
        var fields = List[Field](capacity=len(values))
        for i in range(len(values)):
            ref v = values[i]
            var is_column = v.name() != "" and len(v.columns()) == 1
            var carried = -1
            if is_column:
                carried = input.get_field_index(v.name())
            if carried >= 0:
                ref src = input.fields[carried]
                fields.append(
                    Field(
                        names[i].copy(),
                        src.dtype.copy(),
                        src.nullable,
                        src.metadata.copy(),
                    )
                )
            else:
                fields.append(field(names[i].copy(), v.dtype(input)))
        return schema(fields^)

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(self, ctx: ExecContext) raises -> Pipeline:
        var pipe = self._input.to_operator(ctx)
        var values = List[DynOperator[Datum]](capacity=len(self._values))
        for ref v in self._values:
            values.append(v.to_operator(False))
        pipe.append(ProjectOperator(values^, self._schema.copy()))
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Project(", self._input, ", ")
        for i in range(len(self._names)):
            if i > 0:
                writer.write(", ")
            writer.write(self._names[i], "=", self._values[i])
        writer.write(")")


struct Aggregate(Relation, Writable):
    """`SELECT <keys>, <aggs> FROM input GROUP BY <keys>`.

    The output schema is the key fields followed by the aggregate fields, in
    that order. Everything downstream depends on that ordering — the
    processor reads its key fields back off the front of it, and a `Filter`
    above this node is exactly `HAVING`.

    An empty `keys` is **not** a different node: it is `SELECT sum(x) FROM t`,
    one implicit group. The only thing it changes is which fold each aggregate
    starts, and that is decided here, at plan-build time, because it is known
    here. `to_state(grouped)` compiles two loops out of one struct and running
    the grouped one over a single group measured 14.6x worse — a runtime
    branch could not have made that choice.
    """

    var _input: DynRelation
    var _keys: List[DynValue]
    var _aggs: List[DynValue]
    var _schema: Schema

    def __init__(
        out self,
        var input: DynRelation,
        var keys: List[DynValue],
        var aggs: List[DynValue],
    ) raises:
        self._schema = Self._output_schema(input.schema(), keys, aggs)
        self._input = input^
        self._keys = keys^
        self._aggs = aggs^

    @staticmethod
    def _output_schema(
        input: Schema, keys: List[DynValue], aggs: List[DynValue]
    ) raises -> Schema:
        """Keys first, then aggregates, computed once at construction.

        A key that is a bare column keeps its own name; anything computed has
        none and is called `key0`, `key1`, … by position. That rule is not
        cosmetic: `expr/` shipped a defect where one lane answered `d` and the
        other `key0` for the same `GROUP BY d`, so one query had two output
        schemas depending on which lane built it.
        """
        var fields = List[Field](capacity=len(keys) + len(aggs))
        for i in range(len(keys)):
            ref k = keys[i]
            var name = k.name()
            if name == "":
                name = "key" + String(i)
            fields.append(field(name^, k.dtype(input)))
        for ref a in aggs:
            fields.append(field(a.name(), a.dtype(input)))
        return schema(fields^)

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(self, ctx: ExecContext) raises -> Pipeline:
        var grouped = len(self._keys) > 0
        var folds = List[DynOperator[Datum]](capacity=len(self._aggs))
        for ref a in self._aggs:
            folds.append(a.to_operator(grouped))
        var pipe = self._input.to_operator(ctx)
        var keys = List[DynOperator[Datum]](capacity=len(self._keys))
        for ref k in self._keys:
            keys.append(k.to_operator(False))
        pipe.append(
            AggregateOperator(keys^, folds^, self._schema.copy(), ctx.copy())
        )
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Aggregate(", self._input)
        for ref k in self._keys:
            writer.write(", by=", k)
        for ref a in self._aggs:
            writer.write(", ", a)
        writer.write(")")


struct Limit(Relation, Writable):
    """`OFFSET n LIMIT m` — schema-preserving and streaming.

    Reads no column of its own, so it neither adds nor removes fields. The
    operator it builds reports `done` once it has its rows, which is what stops
    the source: in a push engine nothing downstream can otherwise halt a scan.
    """

    var _input: DynRelation
    var _offset: Int
    var _length: Int

    def __init__(out self, var input: DynRelation, offset: Int, length: Int):
        self._input = input^
        self._offset = offset
        self._length = length

    def schema(self) -> Schema:
        return self._input.schema()

    def to_operator(self, ctx: ExecContext) raises -> Pipeline:
        var pipe = self._input.to_operator(ctx)
        pipe.append(LimitOperator(self._offset, self._length))
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Limit(",
            self._input,
            ", offset=",
            self._offset,
            ", length=",
            self._length,
            ")",
        )


struct Sort(Relation, Writable):
    """`ORDER BY` — schema-preserving, and a pipeline breaker.

    Sorting is blocking by nature: no prefix of the input determines the first
    output row, so the operator buffers every morsel and orders once at
    `finish`. That the engine expresses this with the same two methods a filter
    uses is the point of the push interface.
    """

    var _input: DynRelation
    var _keys: List[DynValue]
    var _ascending: List[Bool]
    var _nulls_first: Bool

    def __init__(
        out self,
        var input: DynRelation,
        var keys: List[DynValue],
        var ascending: List[Bool],
        nulls_first: Bool = True,
    ) raises:
        if len(keys) != len(ascending):
            raise Error(
                "sort: ",
                len(keys),
                " keys but ",
                len(ascending),
                " directions",
            )
        if len(keys) == 0:
            raise Error("sort: needs at least one key")
        self._input = input^
        self._keys = keys^
        self._ascending = ascending^
        self._nulls_first = nulls_first

    def schema(self) -> Schema:
        return self._input.schema()

    def to_operator(self, ctx: ExecContext) raises -> Pipeline:
        var pipe = self._input.to_operator(ctx)
        pipe.append(
            SortOperator(
                _to_operators(self._keys),
                self._ascending.copy(),
                self._nulls_first,
                ctx.copy(),
            )
        )
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Sort(", self._input)
        for i in range(len(self._keys)):
            writer.write(", ", self._keys[i])
            writer.write(" asc" if self._ascending[i] else " desc")
        writer.write(")")


def _to_operators(values: List[DynValue]) raises -> List[DynOperator[Datum]]:
    """Turn a list of logical values into the operators that run them.

    Built **once**, when the plan becomes physical — not per batch. That is the
    whole point of the split: anything a value needs to resolve or cache before
    the first row arrives has a place to live now.
    """
    var out = List[DynOperator[Datum]](capacity=len(values))
    for ref v in values:
        out.append(v.to_operator(False))
    return out^
