"""The physical layer: operators that transform batches as they are pushed.

A `Relation` describes a query and is immutable, shareable and rewritable. This
is what that description becomes when it runs, and it owns everything mutable —
a grouper's table, an accumulator's slots.

**The engine pushes.** An operator is handed a batch and answers with what it
produced, if anything:

    push(batch) -> Optional[RecordBatch]
    finish()    -> Optional[RecordBatch]

That one interface covers streaming and blocking alike, because **blocking
stops being a type distinction and becomes *when you return `Some`***. `Filter`
and `Project` answer from `push` and nothing from `finish`; an aggregate
accumulates through every `push`, answers `None`, and produces its whole result
from `finish`. Under the old pull design those were two different shapes and
therefore two different erased boxes.

**Sources stay pull, and drive.** A scan is I/O and naturally a generator, so
`Source.next()` produces batches and `DynProcessor.collect` pushes them through
the chain. That keeps a reader unchanged, and it is the same split DuckDB
makes. `Exhausted` is **gone**: end of stream is `next()` answering `None`, not
an exception, which also removes the `String(e) == "Exhausted"` comparison the
old `collect` needed.

`DynProcessor` is the assembled pipeline — the driving source plus the operator
chain above it — rather than a single erased operator. That is what lets
`Relation.to_processor` stay compositional without a `children()` walk over the
plan: a source relation creates the pipeline, and every relation above it
appends one stage.
"""

from std.memory import ArcPointer

from ..arrays import DynArray, Int32Array, StructArray
from ..builders import DynBuilder, Int32Builder
from ..dtypes import Field, struct_
from ..kernels.concat import concat
from ..execution import ExecContext
from ..kernels.filter import filter
from ..kernels.groupby import HashGrouper
from ..schema import Schema
from ..tabular import RecordBatch
from .core import DynValue


# ---------------------------------------------------------------------------
# Source — the driver
# ---------------------------------------------------------------------------
trait Source(Deinitable, Movable):
    """Produces batches until it has none left.

    Pull, deliberately, and the only pull in the engine: a scan is a generator
    over I/O, so inverting it would buy nothing and cost every reader.
    """

    def next(mut self) raises -> Optional[RecordBatch]:
        """The next batch, or `None` at end of stream."""
        ...


struct DynSource(Movable):
    """A `Source` of any kind, erased. Move-only: it owns a position."""

    var _data: ArcPointer[NoneType]
    var _virt_next: def(ArcPointer[NoneType]) thin raises -> Optional[
        RecordBatch
    ]

    @staticmethod
    def _next_tramp[
        S: Source
    ](ptr: ArcPointer[NoneType]) raises -> Optional[RecordBatch]:
        return rebind[ArcPointer[S]](ptr)[].next()

    @implicit
    def __init__[S: Source](out self, var value: S):
        var ptr = ArcPointer[S](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_next = Self._next_tramp[S]

    def next(mut self) raises -> Optional[RecordBatch]:
        return self._virt_next(self._data)


struct BatchSource(Source):
    """Yields one in-memory batch, once."""

    var _batch: RecordBatch
    var _done: Bool

    def __init__(out self, var batch: RecordBatch):
        self._batch = batch^
        self._done = False

    def next(mut self) raises -> Optional[RecordBatch]:
        if self._done:
            return None
        self._done = True
        return self._batch.copy()


# ---------------------------------------------------------------------------
# Operator — one stage of the pipeline
# ---------------------------------------------------------------------------
trait Operator(Deinitable, Movable):
    """A stage that transforms pushed batches.

    Two methods, and that is the whole physical contract. Streaming and
    blocking operators differ only in *when* they answer `Some`.

    `Out` is an **associated type** rather than a fixed `RecordBatch`, because
    the two things this trait must cover do not produce the same thing: a
    relational stage produces a batch, a *value*'s stage produces a column.
    Fixing `Out = RecordBatch` would force every value to wrap its column in a
    one-column `RecordBatch` — allocating a `Schema` per value per batch — only
    for `ProjectOperator` to unwrap N of them and reassemble one. That is a
    real runtime cost paid for a nominal unification.
    """

    comptime Out: Copyable
    """What this stage produces — `RecordBatch` relationally, a column for a
    value."""

    def push(mut self, batch: RecordBatch) raises -> Optional[Self.Out]:
        """Consume one batch; answer what it produced, if anything."""
        ...

    def finish(mut self) raises -> Optional[Self.Out]:
        """Flush at end of stream. A streaming operator answers `None`."""
        ...


struct DynOperator[Out: Copyable](Movable):
    """An `Operator` of any stage, erased — **one box, parameterised on `Out`**.

    `DynOperator[RecordBatch]` carries the relational chain and
    `DynOperator[Datum]` carries a value's, but they are two instantiations of
    one definition rather than two hand-written boxes, so the erasure surface
    stays single.

    Move-only: an operator owns mutable state, so copying one would fork an
    execution mid-stream. `DynRelation` copies freely for the opposite reason —
    it owns nothing that runs.
    """

    var _data: ArcPointer[NoneType]
    var _virt_push: def(
        ArcPointer[NoneType], RecordBatch
    ) thin raises -> Optional[Self.Out]
    var _virt_finish: def(ArcPointer[NoneType]) thin raises -> Optional[
        Self.Out
    ]

    @staticmethod
    def _push_tramp[
        O: Operator
    ](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> Optional[O.Out]:
        return rebind[ArcPointer[O]](ptr)[].push(batch)

    @staticmethod
    def _finish_tramp[
        O: Operator
    ](ptr: ArcPointer[NoneType]) raises -> Optional[O.Out]:
        return rebind[ArcPointer[O]](ptr)[].finish()

    @implicit
    def __init__[O: Operator](out self: DynOperator[O.Out], var value: O):
        var ptr = ArcPointer[O](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_push = Self._push_tramp[O]
        self._virt_finish = Self._finish_tramp[O]

    def push(mut self, batch: RecordBatch) raises -> Optional[Self.Out]:
        return self._virt_push(self._data, batch)

    def finish(mut self) raises -> Optional[Self.Out]:
        return self._virt_finish(self._data)


# ---------------------------------------------------------------------------
# DynProcessor — the assembled pipeline
# ---------------------------------------------------------------------------
struct DynProcessor(Movable):
    """A driving source plus the operator chain above it.

    A pipeline rather than a single operator, so `Relation.to_processor` can
    stay compositional: `InMemoryTable` creates one, and `Filter`, `Project`
    and `Aggregate` each `append` a stage to their input's. Nothing has to walk
    the plan, which matters because `DynRelation` deliberately exposes no
    `children()`.
    """

    var _source: DynSource
    var _ops: List[DynOperator[RecordBatch]]

    def __init__(out self, var source: DynSource):
        self._source = source^
        self._ops = List[DynOperator[RecordBatch]]()

    def append(mut self, var op: DynOperator[RecordBatch]):
        self._ops.append(op^)

    def _flow(
        mut self,
        var batch: RecordBatch,
        start: Int,
        mut out: List[RecordBatch],
    ) raises:
        """Push one batch through stages `start..` and collect what survives.

        A stage answering `None` ends this batch's journey — it was consumed
        (an aggregate) or it emptied (a filter), and either way there is
        nothing for the stages above to see.
        """
        var cur = Optional[RecordBatch](batch^)
        for i in range(start, len(self._ops)):
            if cur:
                cur = self._ops[i].push(cur.value().copy())
            else:
                return
        if cur:
            out.append(cur.value().copy())

    def collect(mut self, schema: Schema) raises -> RecordBatch:
        """Drive the source, then flush, and drain everything into one batch.

        The flush is a **cascade**, not a loop of independent calls: when stage
        `i` finally produces its result from `finish`, that batch has still
        never been seen by stages `i+1..`, so it is pushed through them before
        stage `i+1` is itself finished. An aggregate under a projection depends
        on exactly this ordering.
        """
        var out = List[RecordBatch]()
        while True:
            var b = self._source.next()
            if b:
                self._flow(b.value().copy(), 0, out)
            else:
                break

        for i in range(len(self._ops)):
            var f = self._ops[i].finish()
            if f:
                self._flow(f.value().copy(), i + 1, out)

        if len(out) == 0:
            # An empty result must still be a *well-formed* batch: one
            # zero-length column per field. A schema naming fields beside an
            # empty column list leaves `num_columns()` at 0, so anything
            # walking columns by schema index runs off the end.
            var cols = List[DynArray](capacity=len(schema.fields))
            for ref f in schema.fields:
                var b = DynBuilder(f.dtype)
                cols.append(b.finish())
            return RecordBatch(schema=schema.copy(), columns=cols^)
        if len(out) == 1:
            return out.pop()

        # Concatenate per column: `concat` joins arrays, and a batch is its
        # columns plus a schema every morsel already agrees on.
        var s = out[0].schema.copy()
        var cols = List[DynArray](capacity=len(s.fields))
        for i in range(len(s.fields)):
            var parts = List[DynArray]()
            for ref b in out:
                parts.append(b.columns[i].copy())
            # Serial: this runs once at the end over already-materialised
            # morsels, so there is nothing for workers to overlap with.
            cols.append(concat(parts, ExecContext.serial()))
        return RecordBatch(schema=s^, columns=cols^)


# ---------------------------------------------------------------------------
# Streaming operators
# ---------------------------------------------------------------------------
struct FilterOperator(Operator):
    """Keeps rows where the predicate is true."""

    comptime Out = RecordBatch

    var _predicate: DynValue
    var _ctx: ExecContext

    def __init__(out self, var predicate: DynValue, var ctx: ExecContext):
        self._predicate = predicate^
        self._ctx = ctx^

    def push(mut self, batch: RecordBatch) raises -> Optional[RecordBatch]:
        """Evaluate the predicate, then compact every column.

        A morsel that filters to nothing answers `None` rather than an empty
        batch: an empty batch is a legitimate *result* but a useless *morsel*,
        and forwarding it makes every stage above handle a case carrying no
        rows.
        """
        var mask = self._predicate.evaluate(batch).to_array(batch.num_rows())
        var cols = List[DynArray]()
        for i in range(batch.num_columns()):
            cols.append(filter(batch.columns[i].copy(), mask.copy(), self._ctx))
        var out = RecordBatch(schema=batch.schema.copy(), columns=cols^)
        if out.num_rows() > 0:
            return out^
        return None

    def finish(mut self) raises -> Optional[RecordBatch]:
        return None


struct ProjectOperator(Operator):
    """Evaluates each projected value against every morsel."""

    comptime Out = RecordBatch

    var _values: List[DynValue]
    var _schema: Schema

    def __init__(out self, var values: List[DynValue], var schema: Schema):
        self._values = values^
        self._schema = schema^

    def push(mut self, batch: RecordBatch) raises -> Optional[RecordBatch]:
        var cols = List[DynArray](capacity=len(self._values))
        for ref v in self._values:
            # `Datum.to_array` is where a scalar-shaped value stops being lazy:
            # a projection of a constant materialises here and nowhere earlier.
            cols.append(v.evaluate(batch).to_array(batch.num_rows()))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)

    def finish(mut self) raises -> Optional[RecordBatch]:
        return None


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


struct AggregateOperator(Operator):
    """Blocking: fold every pushed morsel, then emit one row per group.

    The shape the push interface exists for — `push` answers `None` all the way
    through the stream and `finish` answers the whole result. Under the old
    pull design this needed a different trait from `Filter` and `Project`, and
    therefore a second erased box.

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

    `HAVING` needs no node of its own: a `FilterOperator` above this stage sees
    the aggregate's *output* batch, which is exactly what the flush cascade in
    `DynProcessor.collect` delivers.
    """

    comptime Out = RecordBatch

    var _keys: List[DynValue]
    var _states: List[DynAggregateState]
    var _schema: Schema
    var _grouper: HashGrouper
    var _ctx: ExecContext
    var _num_groups: Int
    var _keyless: Bool
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynValue],
        var states: List[DynAggregateState],
        var schema: Schema,
        var ctx: ExecContext,
    ):
        self._keyless = len(keys) == 0
        self._keys = keys^
        self._states = states^
        self._schema = schema^
        self._grouper = HashGrouper()
        self._ctx = ctx^
        # One implicit group when there are no keys — including over an input
        # that yields nothing, where `sum` must still answer one null rather
        # than no rows.
        self._num_groups = 1 if self._keyless else 0
        self._emitted = False

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

    def push(mut self, batch: RecordBatch) raises -> Optional[RecordBatch]:
        if self._keyless:
            # An ungrouped fold ignores `groups` entirely, so the empty array
            # is never read. Building one zero per row to say "everything is
            # group 0" is exactly the cost `grouped=False` exists to avoid.
            var empty = Int32Builder(0)
            var no_groups = empty.finish()
            # Indexed rather than `for ref`: a state is move-only, and
            # iterating a `List` by reference requires `Copyable`.
            for i in range(len(self._states)):
                self._states[i].update(batch, no_groups, 1)
        else:
            var gids = self._group(batch)
            self._num_groups = self._grouper.num_groups()
            for i in range(len(self._states)):
                self._states[i].update(batch, gids, self._num_groups)
        return None

    def finish(mut self) raises -> Optional[RecordBatch]:
        if self._emitted:
            return None
        self._emitted = True
        var cols = List[DynArray](capacity=len(self._schema.fields))
        if not self._keyless:
            cols = self._grouper.key_columns(self._key_fields())
        for i in range(len(self._states)):
            cols.append(self._states[i].finish(self._num_groups))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)
