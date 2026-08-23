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
from ..kernels.filter import filter, take
from ..kernels.core import Groups
from ..kernels.groupby import HashGrouping
from ..kernels.sort import sort_indices
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
# Morsel — the unit that flows through the pipeline
# ---------------------------------------------------------------------------
struct Morsel(Copyable, Movable):
    """A batch, plus the group each of its rows belongs to.

    Carrying the grouping *with* the batch is what collapses three executor
    shapes into one. A fold needs to know which slot a row contributes to;
    `push(batch)` alone cannot tell it, which is why an aggregate used to need
    its own trait (`AggregateState`) and its own erased box. Put the assignment
    in the morsel and a fold is simply an `Operator` whose `Out` is a column.

    Relational stages ignore `groups` entirely. They pay nothing for it: the
    ungrouped assignment holds an **empty** id array, because a fold that does
    not scatter never reads the ids and materialising one `Int32` per row to
    say "everything is group 0" is exactly the cost `ScalarGrouping` avoids.
    """

    var batch: RecordBatch
    var groups: Groups

    def __init__(out self, var batch: RecordBatch, var groups: Groups):
        self.batch = batch^
        self.groups = groups^

    @staticmethod
    def ungrouped(var batch: RecordBatch) raises -> Morsel:
        """A morsel with the trivial one-slot assignment, for the relational
        chain and for a query with no `GROUP BY`."""
        var empty = Int32Builder(0)
        return Morsel(batch^, Groups(empty.finish(), 1))


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

    def push(mut self, morsel: Morsel) raises -> Optional[Self.Out]:
        """Consume one morsel; answer what it produced, if anything."""
        ...

    def finish(mut self) raises -> Optional[Self.Out]:
        """Flush at end of stream. A streaming operator answers `None`."""
        ...

    def done(self) -> Bool:
        """Whether this stage will never produce anything again.

        The signal a push engine needs and a pull engine gets for free. Without
        it `LIMIT 10` over a billion-row scan still reads a billion rows: the
        source drives, so nothing downstream can stop it. `collect` stops
        pulling as soon as any stage answers True.

        Defaults to False because almost nothing finishes early — only a
        bounded operator like `Limit` does.
        """
        return False


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
    var _virt_push: def(ArcPointer[NoneType], Morsel) thin raises -> Optional[
        Self.Out
    ]
    var _virt_finish: def(ArcPointer[NoneType]) thin raises -> Optional[
        Self.Out
    ]
    var _virt_done: def(ArcPointer[NoneType]) thin -> Bool

    @staticmethod
    def _push_tramp[
        O: Operator
    ](ptr: ArcPointer[NoneType], morsel: Morsel) raises -> Optional[O.Out]:
        return rebind[ArcPointer[O]](ptr)[].push(morsel)

    @staticmethod
    def _finish_tramp[
        O: Operator
    ](ptr: ArcPointer[NoneType]) raises -> Optional[O.Out]:
        return rebind[ArcPointer[O]](ptr)[].finish()

    @staticmethod
    def _done_tramp[O: Operator](ptr: ArcPointer[NoneType]) -> Bool:
        return rebind[ArcPointer[O]](ptr)[].done()

    @implicit
    def __init__[O: Operator](out self: DynOperator[O.Out], var value: O):
        var ptr = ArcPointer[O](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_push = Self._push_tramp[O]
        self._virt_finish = Self._finish_tramp[O]
        self._virt_done = Self._done_tramp[O]

    def push(mut self, morsel: Morsel) raises -> Optional[Self.Out]:
        return self._virt_push(self._data, morsel)

    def finish(mut self) raises -> Optional[Self.Out]:
        return self._virt_finish(self._data)

    def done(self) -> Bool:
        return self._virt_done(self._data)


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
        var morsel: Morsel,
        start: Int,
        mut out: List[RecordBatch],
    ) raises:
        """Push one batch through stages `start..` and collect what survives.

        A stage answering `None` ends this batch's journey — it was consumed
        (an aggregate) or it emptied (a filter), and either way there is
        nothing for the stages above to see.
        """
        var cur = Optional[Morsel](morsel^)
        for i in range(start, len(self._ops)):
            if cur:
                var produced = self._ops[i].push(cur.value())
                if produced:
                    cur = Morsel.ungrouped(produced.value().copy())
                else:
                    cur = None
            else:
                return
        if cur:
            out.append(cur.value().batch.copy())

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
                self._flow(Morsel.ungrouped(b.value().copy()), 0, out)
                # Early termination. A `Limit` that has its rows stops the
                # source rather than letting it drain — the one thing a push
                # engine must add back that a pull engine got for free.
                var stop = False
                for i in range(len(self._ops)):
                    if self._ops[i].done():
                        stop = True
                if stop:
                    break
            else:
                break

        for i in range(len(self._ops)):
            var f = self._ops[i].finish()
            if f:
                self._flow(Morsel.ungrouped(f.value().copy()), i + 1, out)

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

    def push(mut self, morsel: Morsel) raises -> Optional[RecordBatch]:
        """Evaluate the predicate, then compact every column.

        A morsel that filters to nothing answers `None` rather than an empty
        batch: an empty batch is a legitimate *result* but a useless *morsel*,
        and forwarding it makes every stage above handle a case carrying no
        rows.
        """
        ref batch = morsel.batch
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

    def push(mut self, morsel: Morsel) raises -> Optional[RecordBatch]:
        ref batch = morsel.batch
        var cols = List[DynArray](capacity=len(self._values))
        for ref v in self._values:
            # `Datum.to_array` is where a scalar-shaped value stops being lazy:
            # a projection of a constant materialises here and nowhere earlier.
            cols.append(v.evaluate(batch).to_array(batch.num_rows()))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)

    def finish(mut self) raises -> Optional[RecordBatch]:
        return None


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
    var _folds: List[DynOperator[DynArray]]
    var _schema: Schema
    var _grouping: HashGrouping
    var _keyless: Bool
    var _ctx: ExecContext
    var _num_groups: Int
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynValue],
        var folds: List[DynOperator[DynArray]],
        var schema: Schema,
        var ctx: ExecContext,
    ):
        self._keyless = len(keys) == 0
        self._keys = keys^
        self._folds = folds^
        self._schema = schema^
        self._grouping = HashGrouping()
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

    def _key_columns(mut self, batch: RecordBatch) raises -> List[DynArray]:
        """The key expressions, evaluated against this morsel.

        Evaluated **once** and handed to the grouping, not once per aggregate:
        that is why placement is the operator's business rather than each
        fold's, and it is what a fold-shaped-as-an-independent-operator design
        would have to give up.
        """
        var children = List[DynArray](capacity=len(self._keys))
        for ref k in self._keys:
            children.append(k.evaluate(batch).to_array(batch.num_rows()))
        return children^

    def push(mut self, morsel: Morsel) raises -> Optional[RecordBatch]:
        """Assign the rows to slots **once**, then hand the same morsel to
        every fold.

        This is why placement belongs to the operator rather than to each fold:
        N aggregates over one `GROUP BY` hash the keys once between them, where
        N folds each owning a grouping would hash N times.

        Placement is a runtime choice here and a comptime one inside the fold,
        and that split is measured. Parameterising *this* operator on
        `Grouping` instantiates it once per conformer for **+24,432 bytes** and
        buys nothing: its branch runs once per batch, while the 14.6x
        register-fold win lives in `Fold`, already monomorphised on `G`.
        """
        # Indexed rather than `for ref`: a fold is move-only, and iterating a
        # `List` by reference requires `Copyable`.
        if self._keyless:
            # The morsel already carries the trivial one-slot assignment, so
            # there is nothing to compute and nothing to rebuild.
            for i in range(len(self._folds)):
                _ = self._folds[i].push(morsel)
            return None

        ref batch = morsel.batch
        var groups = self._grouping.assign(
            self._key_columns(batch), batch.num_rows()
        )
        self._num_groups = groups.num_groups
        var forwarded = Morsel(batch.copy(), groups^)
        for i in range(len(self._folds)):
            _ = self._folds[i].push(forwarded)
        return None

    def finish(mut self) raises -> Optional[RecordBatch]:
        if self._emitted:
            return None
        self._emitted = True
        var cols = List[DynArray]()
        if not self._keyless:
            cols = self._grouping.key_columns(self._key_fields())
        for i in range(len(self._folds)):
            var col = self._folds[i].finish()
            if col:
                cols.append(col.value().copy())
        return RecordBatch(schema=self._schema.copy(), columns=cols^)


struct LimitOperator(Operator):
    """`OFFSET`/`LIMIT` — streaming, and the reason `Operator.done` exists.

    Skips `offset` rows, emits at most `length`, and then reports `done` so the
    driver stops pulling. Without that signal the source would drain in full and
    a `LIMIT 10` over a large scan would cost the whole scan.

    Slicing is zero-copy, so a limited batch shares its parent's buffers rather
    than compacting.
    """

    comptime Out = RecordBatch

    var _offset: Int
    var _length: Int
    var _skipped: Int
    var _emitted: Int

    def __init__(out self, offset: Int, length: Int):
        self._offset = offset
        self._length = length
        self._skipped = 0
        self._emitted = 0

    def push(mut self, morsel: Morsel) raises -> Optional[RecordBatch]:
        ref batch = morsel.batch
        var n = batch.num_rows()
        var start = 0
        if self._skipped < self._offset:
            var skip = min(self._offset - self._skipped, n)
            self._skipped += skip
            start = skip
        var available = n - start
        if available <= 0:
            return None
        var wanted = min(available, self._length - self._emitted)
        if wanted <= 0:
            return None
        self._emitted += wanted
        return batch.slice(start, wanted)

    def finish(mut self) raises -> Optional[RecordBatch]:
        return None

    def done(self) -> Bool:
        return self._emitted >= self._length


struct SortOperator(Operator):
    """`ORDER BY` — blocking, because a global order needs every row.

    Buffers each morsel and sorts once at `finish`. That is not a limitation of
    the engine but of the operation: no prefix of the input determines the
    first output row.

    Multiple keys are handled by sorting **stably, last key first**, which is
    the standard decomposition — each pass preserves the order the previous one
    established, so the composition orders by key 0, ties broken by key 1, and
    so on. It costs one pass per key and needs no comparator over tuples, which
    the single-column `sort_indices` kernel could not express anyway.
    """

    comptime Out = RecordBatch

    var _keys: List[DynValue]
    var _ascending: List[Bool]
    var _nulls_first: Bool
    var _batches: List[RecordBatch]
    var _ctx: ExecContext
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynValue],
        var ascending: List[Bool],
        nulls_first: Bool,
        var ctx: ExecContext,
    ):
        self._keys = keys^
        self._ascending = ascending^
        self._nulls_first = nulls_first
        self._batches = List[RecordBatch]()
        self._ctx = ctx^
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[RecordBatch]:
        self._batches.append(morsel.batch.copy())
        return None

    def finish(mut self) raises -> Optional[RecordBatch]:
        if self._emitted or len(self._batches) == 0:
            return None
        self._emitted = True

        var schema = self._batches[0].schema.copy()
        var cols = List[DynArray](capacity=len(schema.fields))
        for i in range(len(schema.fields)):
            var parts = List[DynArray]()
            for ref b in self._batches:
                parts.append(b.columns[i].copy())
            cols.append(concat(parts, self._ctx))
        var whole = RecordBatch(schema=schema^, columns=cols^)

        var order = Optional[Int32Array](None)
        for k in range(len(self._keys) - 1, -1, -1):
            var key = self._keys[k].evaluate(whole).to_array(whole.num_rows())
            if order:
                key = take(key, order.value(), self._ctx)
            var pass_order = sort_indices(
                key,
                ascending=self._ascending[k],
                nulls_first=self._nulls_first,
                stable=True,
                ctx=self._ctx,
            )
            if order:
                # Compose: this pass permutes the previous order, it does not
                # replace it. Skipping this is the classic multi-key sort bug —
                # the last key wins and every earlier one is discarded.
                var prev: DynArray = order.value().copy()
                order = take(prev, pass_order, self._ctx).as_int32().copy()
            else:
                order = pass_order^

        if not order:
            return whole^
        var sorted = List[DynArray](capacity=whole.num_columns())
        for i in range(whole.num_columns()):
            sorted.append(
                take(whole.columns[i].copy(), order.value(), self._ctx)
            )
        return RecordBatch(schema=whole.schema.copy(), columns=sorted^)
