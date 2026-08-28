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

from std.collections import Set
from std.memory import ArcPointer

from ..execution import ExecContext
from ..kernels.join import JoinKind, JOIN_INNER
from ..schema import Schema, schema
from ..tabular import RecordBatch
from ..dtypes import DynType, Field, field
from .params import Bindings
from .runtime.values import column
from .physical import (
    Datum,
    GroupByOperator,
    BatchSourceOperator,
    DynOperator,
    LimitOperator,
    Pipeline,
    FilterOperator,
    JoinOperator,
    ParquetScanOperator,
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
trait Value(Copyable, Deinitable, Writable):
    """What every expression is, in both lanes.

    Five members: what it reads, what it is called, what type it produces,
    whether it yields one value or one per row, and how to turn it into
    something that runs.

    **One trait, not three.** This was `Analyzable & Executable & Writable`, a
    composite alias split that way in reaction to `expr/`'s nine-responsibility
    `Value` trait. The reaction overshot: nothing ever bound on `Analyzable`
    alone, and `Executable` was bound alone in exactly one place, for `shape`.
    Two names that only ever appeared composed back together are not two
    abstractions — five members in one trait is the honest count, and it is
    still not nine.

    `dtype` takes a `Schema` because the runtime lane learns its type from one.
    The comptime lane ignores the argument and answers from its own type; that
    asymmetry is the price of one box holding both lanes.

    `to_operator` is the only way to *run* a value — see `physical.mojo`. There
    is deliberately no `evaluate` here. That is not a claim that a node is
    inert: in the comptime lane the node **is** the executable form, and the
    fused `bind`/`lane` machinery lives on the family traits precisely so a
    subtree stays one type and inlines into one loop. What this trait says is
    narrower and true — a node carries no *state*, so nothing outside a lane
    can run one, and two executions of the same plan cannot interfere.
    """

    comptime aggregates: Bool = False
    """Whether this value answers from `Operator.drain` rather than from a
    batch — that is, whether it is an aggregate.

    An aggregate is an ordinary `Value` in every other respect, so nothing
    structural distinguishes it and the relations that cannot accept one had no
    way to say so. `Project` and `Filter` read this and raise; before it,
    `project([col("a").sum()])` reached `ProjectOperator.push`, which called
    `.value()` on the `None` an aggregate answers with and **aborted the
    process**.

    A defaulted `comptime` rather than a marker trait, because a trait
    constraining nothing documents nothing: every value would still satisfy
    `Value` either way, and only the *answer* differs.
    """

    comptime shape: Shape
    """`Shape.scalar` or `Shape.columnar` — whether this yields one value or
    one per row. Known without running, which is why it lives here and not on
    the operator."""

    def columns(self) -> List[String]:
        """Which columns this expression reads, deduplicated, first-seen
        order."""
        ...

    def name(self) -> String:
        """This expression's name, or empty when it has none."""
        ...

    def dtype(self, schema: Schema) raises -> DynType:
        """The type this produces, without running anything."""
        ...

    def to_operator(
        self, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        """The stateful thing that runs this value.

        `grouped` picks a fold's placement and is ignored by everything else.
        `bindings` supplies this execution's parameter values — the operator
        carries them and hands them back down to `bind`, where a `Param` reads
        them. That is why a plan holds no parameter state and two executions
        of it cannot interfere.
        """
        ...


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

    Five function slots — `columns`, `name`, `dtype`, `write`, `to_operator` —
    plus `shape`, read once at construction because it is a comptime constant.
    `expr/` carried seven and had no `dtype`, computing output types by
    evaluating against a zero-row batch instead.

    `expr/`'s two extra slots were `name()`, which duplicated what `name` and
    `write` already answered, and `resolve_names` — a *rewrite*, carried by
    every boxed expression in every binary though it is a no-op in the comptime
    lane. Nothing here is a rewrite: parameter values travel *through* an
    execution rather than being substituted into a copy of the plan, so the
    box never has to hand back a re-boxed `DynValue`.

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
    var _to_operator: def(
        ArcPointer[NoneType], Bool, Bindings
    ) thin raises -> DynOperator
    var _shape: Shape
    var _aggregates: Bool
    var _drop: def(var ArcPointer[NoneType]) thin
    """Erasure forgets the pointee's destructor; this carries it. See
    `DynOperator._virt_drop` for why the release has to happen at the true
    type, and for the probe that measured it."""

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
    def _to_operator_tramp[
        V: Value
    ](
        ptr: ArcPointer[NoneType], grouped: Bool, bindings: Bindings
    ) raises -> DynOperator:
        return rebind[ArcPointer[V]](ptr)[].to_operator(grouped, bindings)

    @staticmethod
    def _write_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
        return String(rebind[ArcPointer[V]](ptr)[])

    @staticmethod
    def _drop_tramp[V: Value](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[V]](ptr)
        _ = ptr^
        _ = typed^

    @implicit
    def __init__[V: Value](out self, value: V):
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._drop = Self._drop_tramp[V]
        self._aggregates = V.aggregates
        self._columns = Self._columns_tramp[V]
        self._name = Self._name_tramp[V]
        self._dtype = Self._dtype_tramp[V]
        self._write = Self._write_tramp[V]
        self._to_operator = Self._to_operator_tramp[V]
        self._shape = V.shape

    def __deinit__(deinit self):
        self._drop(self._boxed^)

    # -- the erased surface -------------------------------------------------

    def columns(self) -> List[String]:
        return self._columns(self._boxed)

    def name(self) -> String:
        return self._name(self._boxed)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._dtype(self._boxed, schema)

    def to_operator(
        self, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        """The stateful thing that runs this value.

        The slot `DynAggValue._acc` used to occupy, on the one box that now
        holds every value. An aggregate reaches its `FusedAggregateOperator` through here; an
        elementwise value reaches an `EvalOperator`. The caller cannot tell,
        which is the point.
        """
        return self._to_operator(self._boxed, grouped, bindings)

    def aggregates(self) -> Bool:
        """Whether the boxed value answers from `drain` rather than per batch.

        A field for the same reason `shape` is one: a constant per boxed value,
        so a trampoline would pay an indirect call to read something fixed at
        construction."""
        return self._aggregates

    def shape(self) -> Shape:
        """The boxed value's `shape`, read at construction.

        A field rather than a seventh trampoline: it is a constant per boxed
        type, so calling through a pointer to fetch it would pay a call to
        learn something already known.
        """
        return self._shape

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._write(self._boxed))


# ---------------------------------------------------------------------------
# merged — order-preserving union, the shape `columns()` folds with
# ---------------------------------------------------------------------------
# `columns()` folds a node's children into a list that is deduplicated but
# **order-preserving**: first-seen order is part of the contract
# (`test_runtime_columns_are_deduped_in_first_seen_order` asserts it), so a
# plain `Set` cannot answer it on its own.
#
# A `List` carries the order and a `Set` answers membership, which also makes
# this linear — the loop it replaces rescanned the accumulated list once per
# candidate, so a wide expression was quadratic in its own column count.
#
# `bind` and `validity` fold too and are deliberately *not* here: each is
# already a single expression — a tuple and an intersect — so there is nothing
# to extract. They could not be defaulted onto a trait anyway, since a default
# returning `Self.Bound` needs that type to be `ImplicitlyCopyable`, which
# marrow's array types deliberately are not.
def merged(var into: List[String], extra: List[String]) -> List[String]:
    """`into`, followed by whatever in `extra` it does not already contain."""
    var seen = Set[String]()
    for ref n in into:
        seen.add(n.copy())
    for ref n in extra:
        if n not in seen:
            seen.add(n.copy())
            into.append(n.copy())
    return into^


trait Relation(Copyable, Deinitable, Movable):
    """An immutable description of a query."""

    def schema(self) -> Schema:
        """The columns this relation produces.

        Computed at construction and stored, not derived on demand: a caller
        asks for it once per plan node while building the node above, and a
        `Filter` would otherwise re-derive its input's schema every time.
        """
        ...

    def to_operator(
        self, ctx: ExecContext, bindings: Bindings = Bindings()
    ) raises -> Pipeline:
        """The running operator for this description.

        Takes a context because this is the seam where one logical operator may
        become different physical ones — a hash join or a merge join, a CPU or
        a GPU pass. Nothing exercises that yet; the argument is here because
        removing the seam is the part that would be hard to undo.

        `bindings` is threaded down to the values this plan contains and is
        consumed there, by `Param.bind`. A relation never reads one.
        """
        ...


struct DynRelation(Copyable, Movable, Writable):
    """A `Relation` of any operator, erased.

    Copyable, unlike `Pipeline`: a plan owns nothing that runs, so sharing
    one is an `ArcPointer` bump and two consumers cannot interfere.
    """

    var _data: ArcPointer[NoneType]
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_to_operator: def(
        ArcPointer[NoneType], ExecContext, Bindings
    ) thin raises -> Pipeline
    var _virt_write: def(ArcPointer[NoneType]) thin -> String
    var _drop: def(var ArcPointer[NoneType]) thin
    """Erasure forgets the pointee's destructor; this carries it. See
    `DynOperator._virt_drop` for why the release has to happen at the true
    type, and for the probe that measured it."""

    @staticmethod
    def _schema_tramp[
        R: Relation & Writable
    ](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[R]](ptr)[].schema()

    @staticmethod
    def _to_operator_tramp[
        R: Relation
    ](
        ptr: ArcPointer[NoneType], ctx: ExecContext, bindings: Bindings
    ) raises -> Pipeline:
        return rebind[ArcPointer[R]](ptr)[].to_operator(ctx, bindings)

    @staticmethod
    def _write_tramp[
        R: Relation & Writable
    ](ptr: ArcPointer[NoneType]) -> String:
        return String(rebind[ArcPointer[R]](ptr)[])

    @staticmethod
    def _drop_tramp[R: Relation & Writable](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[R]](ptr)
        _ = ptr^
        _ = typed^

    @implicit
    def __init__[R: Relation & Writable](out self, value: R):
        var ptr = ArcPointer[R](value.copy())
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_schema = Self._schema_tramp[R]
        self._virt_to_operator = Self._to_operator_tramp[R]
        self._virt_write = Self._write_tramp[R]
        self._drop = Self._drop_tramp[R]

    def __deinit__(deinit self):
        self._drop(self._data^)

    def schema(self) -> Schema:
        return self._virt_schema(self._data)

    def to_operator(
        self, ctx: ExecContext, bindings: Bindings = Bindings()
    ) raises -> Pipeline:
        return self._virt_to_operator(self._data, ctx, bindings)

    # -- the plan-building API ----------------------------------------------
    #
    # These are the surface a caller actually writes. Every one returns a
    # `DynRelation`, so plans compose left to right —
    # `t.filter(...).aggregate(...).sort_by(...)` — and no caller ever names a
    # node type or wraps anything in `DynRelation` by hand.
    #
    # They live on the box rather than on `Relation` because that is where
    # composition happens: a verb needs a *boxed* input to build its node with,
    # and `self` already is one. Putting them on the trait would make every
    # conformer implement eight methods it does not care about.

    def filter(self, var predicate: DynValue) raises -> DynRelation:
        """Rows where `predicate` is true. Schema-preserving."""
        return Filter(self.copy(), predicate^)

    def select(self, names: List[String]) raises -> DynRelation:
        """Keep these columns, in this order.

        Sugar over `project`: the values are runtime column reads, so this
        needs no dtype from the caller. That is the runtime lane earning its
        keep — a fused `Column[T]` would force `select` to be generic over
        every column's type.
        """
        var values = List[DynValue](capacity=len(names))
        for ref n in names:
            values.append(column(n.copy()))
        return Project(self.copy(), names.copy(), values^)

    def project(
        self, var names: List[String], var values: List[DynValue]
    ) raises -> DynRelation:
        """`SELECT <values> AS <names>` — new columns over the same rows."""
        return Project(self.copy(), names^, values^)

    def limit(self, length: Int, offset: Int = 0) raises -> DynRelation:
        """`OFFSET offset LIMIT length`."""
        return Limit(self.copy(), offset, length)

    def sort_by(
        self,
        var keys: List[DynValue],
        var ascending: List[Bool],
        nulls_first: Bool = True,
    ) raises -> DynRelation:
        """`ORDER BY` — a pipeline breaker, so it buffers and sorts at the
        end."""
        return Sort(self.copy(), keys^, ascending^, nulls_first)

    def aggregate(
        self,
        var aggs: List[DynValue],
        var keys: List[DynValue] = List[DynValue](),
    ) raises -> DynRelation:
        """`SELECT <keys>, <aggs> ... GROUP BY <keys>`.

        `keys` defaults to empty, which is a whole-table aggregate rather than
        a special node — `sum(x)` with no `GROUP BY` is one implicit group.
        Aggregates come first because they are the part a caller always
        supplies.
        """
        return Aggregate(self.copy(), keys^, aggs^)

    def join(
        self,
        var right: DynRelation,
        var left_keys: List[Int],
        var right_keys: List[Int],
        kind: JoinKind = JOIN_INNER,
    ) raises -> DynRelation:
        """Equijoin. `self` is the build side and `right` streams."""
        return Join(self.copy(), right^, left_keys^, right_keys^, kind)

    def execute(
        self,
        ctx: ExecContext = ExecContext.auto(),
        bindings: Bindings = Bindings(),
    ) raises -> RecordBatch:
        """Run this plan and drain it into one batch."""
        var p = self.to_operator(ctx, bindings)
        # The shim: operators work in struct arrays, the public API hands back
        # a batch. Cheap — children move, schema comes off the struct dtype.
        return RecordBatch.from_struct_array(p.collect(self.schema()))

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._virt_write(self._data))


struct InMemoryTable(Relation, Writable):
    """A batch already in memory, as a source."""

    var _batch: RecordBatch

    def __init__(out self, var batch: RecordBatch):
        self._batch = batch^

    def schema(self) -> Schema:
        return self._batch.schema.copy()

    def to_operator(
        self, ctx: ExecContext, bindings: Bindings = Bindings()
    ) raises -> Pipeline:
        """The one relation that *creates* a pipeline; every other appends."""
        return Pipeline(BatchSourceOperator(self._batch.to_struct_array()))

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

    def __init__(
        out self, var input: DynRelation, var predicate: DynValue
    ) raises:
        if predicate.aggregates():
            raise Error(
                "filter: '",
                predicate.name(),
                (
                    "' is an aggregate, which has no value per row; put it in"
                    " .aggregate() and filter the result (HAVING)"
                ),
            )
        self._input = input^
        self._predicate = predicate^

    def schema(self) -> Schema:
        return self._input.schema()

    def to_operator(
        self, ctx: ExecContext, bindings: Bindings = Bindings()
    ) raises -> Pipeline:
        var pipe = self._input.to_operator(ctx, bindings)
        pipe.append(
            FilterOperator(
                self._predicate.to_operator(False, bindings), ctx.copy()
            )
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
        for i in range(len(values)):
            if values[i].aggregates():
                raise Error(
                    "project: '",
                    names[i],
                    (
                        "' is an aggregate, which has no value per row; use"
                        " .aggregate() instead"
                    ),
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

    def to_operator(
        self, ctx: ExecContext, bindings: Bindings = Bindings()
    ) raises -> Pipeline:
        var pipe = self._input.to_operator(ctx, bindings)
        var values = List[DynOperator](capacity=len(self._values))
        for ref v in self._values:
            values.append(v.to_operator(False, bindings))
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
    operator reads its key fields back off the front of it, and a `Filter`
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

    def to_operator(
        self, ctx: ExecContext, bindings: Bindings = Bindings()
    ) raises -> Pipeline:
        var grouped = len(self._keys) > 0
        var folds = List[DynOperator](capacity=len(self._aggs))
        for ref a in self._aggs:
            folds.append(a.to_operator(grouped, bindings))
        var pipe = self._input.to_operator(ctx, bindings)
        var keys = List[DynOperator](capacity=len(self._keys))
        for ref k in self._keys:
            keys.append(k.to_operator(False, bindings))
        pipe.append(
            GroupByOperator(keys^, folds^, self._schema.copy(), ctx.copy())
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

    def to_operator(
        self, ctx: ExecContext, bindings: Bindings = Bindings()
    ) raises -> Pipeline:
        var pipe = self._input.to_operator(ctx, bindings)
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

    def to_operator(
        self, ctx: ExecContext, bindings: Bindings = Bindings()
    ) raises -> Pipeline:
        var pipe = self._input.to_operator(ctx, bindings)
        pipe.append(
            SortOperator(
                _to_operators(self._keys, bindings),
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


def _to_operators(
    values: List[DynValue], bindings: Bindings
) raises -> List[DynOperator]:
    """Turn a list of logical values into the operators that run them.

    Built **once**, when the plan becomes physical — not per batch. That is the
    whole point of the split: anything a value needs to resolve or cache before
    the first row arrives has a place to live now.
    """
    var out = List[DynOperator](capacity=len(values))
    for ref v in values:
        out.append(v.to_operator(False, bindings))
    return out^


struct Join(Relation, Writable):
    """An equijoin over two sub-plans.

    The **only** node with two inputs, and the reason `Pipeline` had to be an
    `Operator`: the build side is a whole plan, and it is handed to the
    operator as an ordinary boxed stage. Before that, a chain of stages was a
    different kind of thing from a stage, and there was nowhere to put a second
    one.

    `left` is the build side and `right` streams. That is the usual convention
    and it is not arbitrary — the build side is materialised and indexed, so it
    should be the smaller one. Choosing it automatically is an optimiser's job
    and this layer does not have one.
    """

    var _left: DynRelation
    var _right: DynRelation
    var _left_keys: List[Int]
    var _right_keys: List[Int]
    var _kind: JoinKind
    var _strictness: UInt8
    var _schema: Schema

    def __init__(
        out self,
        var left: DynRelation,
        var right: DynRelation,
        var left_keys: List[Int],
        var right_keys: List[Int],
        kind: JoinKind = JOIN_INNER,
        strictness: UInt8 = 0,
    ) raises:
        if len(left_keys) != len(right_keys):
            raise Error(
                "join: ",
                len(left_keys),
                " left keys but ",
                len(right_keys),
                " right keys",
            )
        if len(left_keys) == 0:
            raise Error("join: needs at least one key pair")
        self._schema = Self._output_schema(left.schema(), right.schema(), kind)
        self._left = left^
        self._right = right^
        self._left_keys = left_keys^
        self._right_keys = right_keys^
        self._kind = kind
        self._strictness = strictness

    @staticmethod
    def _output_schema(
        left: Schema, right: Schema, kind: JoinKind
    ) raises -> Schema:
        """Left fields then right fields — except for the kinds that emit only
        the left side.

        `SEMI` and `ANTI` answer "which left rows had a match", so the right
        side contributes nothing to the output. `JoinKind.emits_right_columns`
        owns that rule; asking it here keeps the schema and the kernel from
        disagreeing about the shape of the same result.
        """
        var fields = List[Field]()
        for ref f in left.fields:
            fields.append(f.copy())
        if kind.emits_right_columns():
            for ref f in right.fields:
                fields.append(f.copy())
        return schema(fields^)

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self, ctx: ExecContext, bindings: Bindings = Bindings()
    ) raises -> Pipeline:
        """The probe side is the pipeline; the build side is a stage's cargo."""
        var probe = self._right.to_operator(ctx, bindings)
        probe.append(
            JoinOperator(
                self._left.to_operator(ctx, bindings),
                self._left_keys.copy(),
                self._right_keys.copy(),
                self._kind,
                self._strictness,
                self._schema.copy(),
                ctx.copy(),
            )
        )
        return probe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Join(", self._left, ", ", self._right, ", ", self._kind, ")"
        )


struct ParquetScan(Relation, Writable):
    """A Parquet file as a source, read one row group at a time.

    **The schema is the projection.** The scan reads only its own columns out
    of the file, so narrowing a scan's schema *is* how a projection gets pushed
    into it — no separate mechanism, and nothing to keep in sync.

    The schema is supplied rather than read from the file, so building the plan
    touches no I/O: a `Relation` is a description, and a description that has
    to open a file to exist cannot be constructed for a file that is not there
    yet. The operator opens it on first `drain`.
    """

    var _path: String
    var _schema: Schema

    def __init__(out self, var path: String, var schema: Schema):
        self._path = path^
        self._schema = schema^

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self, ctx: ExecContext, bindings: Bindings = Bindings()
    ) raises -> Pipeline:
        return Pipeline(
            ParquetScanOperator(self._path.copy(), self._schema.copy())
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("ParquetScan(", self._path, ")")
