"""The logical layer: an immutable description of a query.

Paired with `physical.mojo`, which holds what these become when they run.

A `Relation` says *what* to compute. It owns nothing that runs, so a plan is
freely copyable, shareable, inspectable and — once the optimizer exists —
rewritable. `to_processor(ctx)` turns it into the physical operator that owns
the running state.

**Two methods, not eight.** `expr/`'s `DynRelation` carries `schema`,
`to_processor`, `write`, `drop`, `kind`, `with_predicate`, `with_projection`
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
from .core import DynAggValue, DynValue
from ..dtypes import Field, field
from .physical import (
    AggregateOperator,
    BatchSource,
    LimitOperator,
    DynAggregateState,
    DynProcessor,
    FilterOperator,
    ProjectOperator,
    SortOperator,
)


trait Relation(Copyable, Deinitable, Movable):
    """An immutable description of a query."""

    def schema(self) -> Schema:
        """The columns this relation produces.

        Computed at construction and stored, not derived on demand: a caller
        asks for it once per plan node while building the node above, and a
        `Filter` would otherwise re-derive its input's schema every time.
        """
        ...

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        """The running operator for this description.

        Takes a context because this is the seam where one logical operator may
        become different physical ones — a hash join or a merge join, a CPU or
        a GPU pass. Nothing exercises that yet; the argument is here because
        removing the seam is the part that would be hard to undo.
        """
        ...


struct DynRelation(Copyable, Movable, Writable):
    """A `Relation` of any operator, erased.

    Copyable, unlike `DynProcessor`: a plan owns nothing that runs, so sharing
    one is an `ArcPointer` bump and two consumers cannot interfere.
    """

    var _data: ArcPointer[NoneType]
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_to_processor: def(
        ArcPointer[NoneType], ExecContext
    ) thin raises -> DynProcessor
    var _virt_write: def(ArcPointer[NoneType]) thin -> String

    @staticmethod
    def _schema_tramp[R: Relation](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[R]](ptr)[].schema()

    @staticmethod
    def _to_processor_tramp[
        R: Relation
    ](ptr: ArcPointer[NoneType], ctx: ExecContext) raises -> DynProcessor:
        return rebind[ArcPointer[R]](ptr)[].to_processor(ctx)

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

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        return self._virt_to_processor(self._data, ctx)

    def execute(
        self, ctx: ExecContext = ExecContext.auto()
    ) raises -> RecordBatch:
        """Run this plan and drain it into one batch."""
        var p = self.to_processor(ctx)
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

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        """The one relation that *creates* a pipeline; every other appends."""
        return DynProcessor(BatchSource(self._batch.copy()))

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

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        var pipe = self._input.to_processor(ctx)
        pipe.append(FilterOperator(self._predicate.copy(), ctx.copy()))
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

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        var pipe = self._input.to_processor(ctx)
        pipe.append(ProjectOperator(self._values.copy(), self._schema.copy()))
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
    var _aggs: List[DynAggValue]
    var _schema: Schema

    def __init__(
        out self,
        var input: DynRelation,
        var keys: List[DynValue],
        var aggs: List[DynAggValue],
    ) raises:
        self._schema = Self._output_schema(input.schema(), keys, aggs)
        self._input = input^
        self._keys = keys^
        self._aggs = aggs^

    @staticmethod
    def _output_schema(
        input: Schema, keys: List[DynValue], aggs: List[DynAggValue]
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

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        var grouped = len(self._keys) > 0
        var states = List[DynAggregateState](capacity=len(self._aggs))
        for ref a in self._aggs:
            states.append(a.to_state(grouped))
        var pipe = self._input.to_processor(ctx)
        pipe.append(
            AggregateOperator(
                self._keys.copy(), states^, self._schema.copy(), ctx.copy()
            )
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

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        var pipe = self._input.to_processor(ctx)
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

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        var pipe = self._input.to_processor(ctx)
        pipe.append(
            SortOperator(
                self._keys.copy(),
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
