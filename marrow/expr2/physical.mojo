"""The physical layer: operators that own execution state and pull morsels.

A `Relation` describes a query and is immutable, shareable and rewritable. A
`Processor` is what that description becomes when it runs, and it owns
everything mutable — a scan's offset, a grouper's table, a sort's buffer.

The split is deliberate and it is the cleaner half of `expr/`'s design:
`DynProcessor` accreted **three** slots to `DynRelation`'s eight, because a
processor was only ever asked two things. It is kept here for that reason and
for one more — `to_processor(ctx)` is the seam where one logical operator could
become different physical ones (hash join or merge join, CPU or GPU). Fusing
the layers would weld that shut.

`pull` raises `Exhausted` rather than returning an `Optional[RecordBatch]`:
end-of-stream is not a value a caller should be able to ignore, and an operator
that forgets to propagate it hangs rather than silently truncating.
"""

from std.memory import ArcPointer

from ..arrays import DynArray, Int32Array, StructArray
from ..builders import DynBuilder, Int32Builder
from ..dtypes import Field, struct_
from ..kernels.concat import concat
from ..execution import ExecContext
from ..kernels.filter import filter
from ..kernels.groupby import HashGrouper
from ..scalars import DynScalar
from ..schema import Schema
from ..tabular import RecordBatch
from .core import DynValue


struct Exhausted(TrivialRegisterPassable, Writable):
    """Raised by `pull()` when a processor has no more morsels."""

    def __init__(out self):
        pass

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Exhausted")


trait Processor(Deinitable, Movable):
    """Something that yields morsels until it is empty.

    Two methods, and that is the whole physical contract. Every operator —
    scan, filter, sort, join — is one of these, and the erased box carries
    exactly these two plus a destructor.
    """

    def schema(self) -> Schema:
        ...

    def pull(mut self) raises -> RecordBatch:
        """The next morsel, or raise `Exhausted`."""
        ...


struct DynProcessor(Movable):
    """A `Processor` of any operator, erased.

    Move-only: a processor owns mutable state, so copying one would fork an
    execution mid-stream. `DynRelation` copies freely for the opposite reason —
    it owns nothing that runs.
    """

    var _data: ArcPointer[NoneType]
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_pull: def(ArcPointer[NoneType]) thin raises -> RecordBatch

    @staticmethod
    def _schema_tramp[P: Processor](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[P]](ptr)[].schema()

    @staticmethod
    def _pull_tramp[
        P: Processor
    ](ptr: ArcPointer[NoneType]) raises -> RecordBatch:
        return rebind[ArcPointer[P]](ptr)[].pull()

    @implicit
    def __init__[P: Processor](out self, var value: P):
        var ptr = ArcPointer[P](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_schema = Self._schema_tramp[P]
        self._virt_pull = Self._pull_tramp[P]

    def schema(self) -> Schema:
        return self._virt_schema(self._data)

    def pull(mut self) raises -> RecordBatch:
        return self._virt_pull(self._data)

    def collect(mut self) raises -> RecordBatch:
        """Drain every morsel into one batch.

        The only thing that catches `Exhausted`, so an operator never has to
        decide what end-of-stream means.
        """
        var batches = List[RecordBatch]()
        while True:
            try:
                batches.append(self.pull())
            except e:
                if String(e) == "Exhausted":
                    break
                raise e
        if len(batches) == 0:
            # An empty result must still be a *well-formed* batch: one
            # zero-length column per field. A schema naming fields beside an
            # empty column list leaves `num_columns()` at 0, so anything
            # walking columns by schema index runs off the end.
            var s = self.schema()
            var cols = List[DynArray](capacity=len(s.fields))
            for ref f in s.fields:
                var b = DynBuilder(f.dtype)
                cols.append(b.finish())
            return RecordBatch(schema=s^, columns=cols^)
        if len(batches) == 1:
            return batches.pop()

        # Concatenate per column: `concat` joins arrays, and a batch is its
        # columns plus a schema every morsel already agrees on.
        var s = batches[0].schema.copy()
        var cols = List[DynArray](capacity=len(s.fields))
        for i in range(len(s.fields)):
            var parts = List[DynArray]()
            for ref b in batches:
                parts.append(b.columns[i].copy())
            # Serial: this runs once at the end over already-materialised
            # morsels, so there is nothing for workers to overlap with.
            cols.append(concat(parts, ExecContext.serial()))
        return RecordBatch(schema=s^, columns=cols^)


struct BatchSource(Processor):
    """Yields one in-memory batch, once."""

    var _batch: RecordBatch
    var _done: Bool

    def __init__(out self, var batch: RecordBatch):
        self._batch = batch^
        self._done = False

    def schema(self) -> Schema:
        return self._batch.schema.copy()

    def pull(mut self) raises -> RecordBatch:
        if self._done:
            raise Exhausted()
        self._done = True
        return self._batch.copy()


struct FilterProcessor(Processor):
    """Keeps rows where the predicate is true."""

    var _input: DynProcessor
    var _predicate: DynValue
    var _ctx: ExecContext

    def __init__(
        out self,
        var input: DynProcessor,
        var predicate: DynValue,
        var ctx: ExecContext,
    ):
        self._input = input^
        self._predicate = predicate^
        self._ctx = ctx^

    def schema(self) -> Schema:
        return self._input.schema()

    def pull(mut self) raises -> RecordBatch:
        """Evaluate the predicate, then compact every column.

        A morsel that filters to nothing is skipped rather than yielded: an
        empty batch is a legitimate *result* but a useless *morsel*, and
        forwarding it makes every operator above handle a case that carries no
        rows.
        """
        while True:
            var batch = self._input.pull()
            var mask = self._predicate.evaluate(batch).to_array(
                batch.num_rows()
            )
            var cols = List[DynArray]()
            for i in range(batch.num_columns()):
                cols.append(
                    filter(batch.columns[i].copy(), mask.copy(), self._ctx)
                )
            var out = RecordBatch(schema=batch.schema.copy(), columns=cols^)
            if out.num_rows() > 0:
                return out^


struct ProjectProcessor(Processor):
    """Evaluates each projected value against every morsel."""

    var _input: DynProcessor
    var _values: List[DynValue]
    var _schema: Schema

    def __init__(
        out self,
        var input: DynProcessor,
        var values: List[DynValue],
        var schema: Schema,
    ):
        self._input = input^
        self._values = values^
        self._schema = schema^

    def schema(self) -> Schema:
        return self._schema.copy()

    def pull(mut self) raises -> RecordBatch:
        var batch = self._input.pull()
        var cols = List[DynArray](capacity=len(self._values))
        for ref v in self._values:
            # `Datum.to_array` is where a scalar-shaped value stops being lazy: a
            # projection of a constant materialises here and nowhere earlier.
            cols.append(v.evaluate(batch).to_array(batch.num_rows()))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)


# ---------------------------------------------------------------------------
# AggregateState — the physical half of a aggregate
# ---------------------------------------------------------------------------
trait AggregateState(Deinitable, Movable):
    """A fold in progress. The aggregate counterpart of `Processor`.

    Move-only for the same reason: it owns mutable state, so copying one would
    fork a fold halfway through and double-count whatever came before.
    """

    def update(
        mut self, batch: RecordBatch, groups: Int32Array, num_groups: Int
    ) raises:
        """Fold one morsel in, at the given group assignment.

        Takes the **batch**, not a column: a fused aggregate state binds its own
        input subtree and reads lanes, so `sum(a * 2 + b)` never materialises
        `a * 2 + b`. That is the one thing DataFusion, Polars and ClickHouse
        cannot express — all three take an already-computed column, because
        none has comptime types.

        An ungrouped fold ignores `groups`; there is one group by construction,
        and building a zero vector to say so is exactly the cost it avoids.
        """
        ...

    def finish(mut self, num_groups: Int) raises -> DynArray:
        """One value per group, once every morsel has been folded.

        A column rather than a scalar even when ungrouped — `num_groups` is
        then 1 — so the grouped and ungrouped folds answer the same shape and
        `Aggregate` does not branch on which it holds.
        """
        ...


struct DynAggregateState(Movable):
    """An `AggregateState` of any aggregate, erased."""

    var _data: ArcPointer[NoneType]
    var _virt_update: def(
        ArcPointer[NoneType], RecordBatch, Int32Array, Int
    ) thin raises
    var _virt_finish: def(ArcPointer[NoneType], Int) thin raises -> DynArray

    @staticmethod
    def _update_tramp[
        A: AggregateState
    ](
        ptr: ArcPointer[NoneType],
        batch: RecordBatch,
        groups: Int32Array,
        num_groups: Int,
    ) raises:
        rebind[ArcPointer[A]](ptr)[].update(batch, groups, num_groups)

    @staticmethod
    def _finish_tramp[
        A: AggregateState
    ](ptr: ArcPointer[NoneType], num_groups: Int) raises -> DynArray:
        return rebind[ArcPointer[A]](ptr)[].finish(num_groups)

    @implicit
    def __init__[A: AggregateState](out self, var value: A):
        var ptr = ArcPointer[A](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_update = Self._update_tramp[A]
        self._virt_finish = Self._finish_tramp[A]

    def update(
        mut self, batch: RecordBatch, groups: Int32Array, num_groups: Int
    ) raises:
        self._virt_update(self._data, batch, groups, num_groups)

    def finish(mut self, num_groups: Int) raises -> DynArray:
        return self._virt_finish(self._data, num_groups)


struct AggregateProcessor(Processor):
    """Blocking: fold every morsel as it arrives, then emit one row per group.

    **Nothing is buffered.** `AggregateState.update` takes the whole
    `RecordBatch`, so each state binds its own input subtree and folds lanes
    straight out of the morsel — `sum(a * 2 + b)` never materialises
    `a * 2 + b`, and no per-aggregate chunk list is ever built. `expr/`'s
    processor buffers one evaluated column per aggregate per morsel and
    `concat`s them at emit time; this one keeps only the grouper's key
    builders, which grow with the number of *groups* rather than the number of
    rows.

    Group ids are dense and stable across morsels, so a state that has already
    folded batch N keeps its slots when batch N+1 introduces new groups —
    `AggState._grow` extends them in place rather than reallocating a fold.

    `HAVING` needs no node of its own: a `Filter` above this operator evaluates
    its predicate against the aggregate's *output* batch.
    """

    var _input: DynProcessor
    var _keys: List[DynValue]
    var _states: List[DynAggregateState]
    var _schema: Schema
    var _grouper: HashGrouper
    var _ctx: ExecContext
    var _emitted: Bool

    def __init__(
        out self,
        var input: DynProcessor,
        var keys: List[DynValue],
        var states: List[DynAggregateState],
        var schema: Schema,
        var ctx: ExecContext,
    ):
        self._input = input^
        self._keys = keys^
        self._states = states^
        self._schema = schema^
        self._grouper = HashGrouper()
        self._ctx = ctx^
        self._emitted = False

    def schema(self) -> Schema:
        return self._schema.copy()

    def _key_fields(self) -> List[Field]:
        """The output schema is keys then aggregates, so the group keys are its
        first `len(self._keys)` fields."""
        var fields = List[Field](capacity=len(self._keys))
        for i in range(len(self._keys)):
            fields.append(self._schema.fields[i].copy())
        return fields^

    def _group(mut self, batch: RecordBatch) raises -> Int32Array:
        """This morsel's rows, resolved to dense group ids."""
        var children = List[DynArray](capacity=len(self._keys))
        for ref k in self._keys:
            children.append(k.evaluate(batch).to_array(batch.num_rows()))
        var keys = StructArray(
            dtype=struct_(self._key_fields()),
            length=batch.num_rows(),
            nulls=0,
            offset=0,
            bitmap=None,
            children=children^,
        )
        return self._grouper.consume_keys(keys)

    def pull(mut self) raises -> RecordBatch:
        if self._emitted:
            raise Exhausted()
        self._emitted = True

        var keyless = len(self._keys) == 0
        # An ungrouped fold ignores `groups` entirely, so this is never read.
        # Building one zero per row to say "everything is group 0" is exactly
        # the cost the `grouped=False` instantiation exists to avoid.
        var empty = Int32Builder(0)
        var no_groups = empty.finish()
        # One implicit group when there are no keys — including over an input
        # that yields nothing, where `sum` must still answer one null rather
        # than no rows.
        var num_groups = 1 if keyless else 0

        while True:
            try:
                var batch = self._input.pull()
                if keyless:
                    for i in range(len(self._states)):
                        self._states[i].update(batch, no_groups, 1)
                else:
                    var gids = self._group(batch)
                    num_groups = self._grouper.num_groups()
                    for i in range(len(self._states)):
                        self._states[i].update(batch, gids, num_groups)
            except e:
                if String(e) != "Exhausted":
                    raise e
                break

        var cols = List[DynArray](capacity=len(self._schema.fields))
        if not keyless:
            cols = self._grouper.key_columns(self._key_fields())
        # Indexed rather than `for ref`: a state is move-only, and iterating a
        # `List` by reference requires its element to be `Copyable`.
        for i in range(len(self._states)):
            cols.append(self._states[i].finish(num_groups))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)
