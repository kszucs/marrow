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
from .core import DynValue
from ..dtypes import Field, field
from .physical import (
    BatchSource,
    DynProcessor,
    FilterProcessor,
    ProjectProcessor,
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
        return p.collect()

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
        return DynProcessor(
            FilterProcessor(
                self._input.to_processor(ctx),
                self._predicate.copy(),
                ctx.copy(),
            )
        )

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
        return DynProcessor(
            ProjectProcessor(
                self._input.to_processor(ctx),
                self._values.copy(),
                self._schema.copy(),
            )
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Project(", self._input, ", ")
        for i in range(len(self._names)):
            if i > 0:
                writer.write(", ")
            writer.write(self._names[i], "=", self._values[i])
        writer.write(")")
