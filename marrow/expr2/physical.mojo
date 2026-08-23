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
the source operator answers from `drain` and `Pipeline.collect` pushes what
it yields through
the chain. That keeps a reader unchanged, and it is the same split DuckDB
makes. `Exhausted` is **gone**: end of stream is `next()` answering `None`, not
an exception, which also removes the `String(e) == "Exhausted"` comparison the
old `collect` needed.

`Pipeline` is the assembled pipeline — the driving source plus the operator
chain above it — rather than a single erased operator. That is what lets
`Relation.to_operator` stay compositional without a `children()` walk over the
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
from .core import Datum, DynValue


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

    def drain(mut self) raises -> Optional[Self.Out]:
        """Produce whatever is available **without new input**; `None` when
        there is nothing left.

        Repeatable, and that is what lets one trait cover sources too. A source
        is simply the operator whose `push` is never called and whose `drain`
        keeps answering until its input runs out; a filter answers `None`
        immediately; an aggregate answers its result once, then `None`.

        It also lets an operator emit *several* batches at end of stream — a
        chunking sort, a fanning join — which a one-shot `finish` could not
        express at all.
        """
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
    var _virt_drain: def(ArcPointer[NoneType]) thin raises -> Optional[Self.Out]
    var _virt_done: def(ArcPointer[NoneType]) thin -> Bool

    @staticmethod
    def _push_tramp[
        O: Operator
    ](ptr: ArcPointer[NoneType], morsel: Morsel) raises -> Optional[O.Out]:
        return rebind[ArcPointer[O]](ptr)[].push(morsel)

    @staticmethod
    def _drain_tramp[
        O: Operator
    ](ptr: ArcPointer[NoneType]) raises -> Optional[O.Out]:
        return rebind[ArcPointer[O]](ptr)[].drain()

    @staticmethod
    def _done_tramp[O: Operator](ptr: ArcPointer[NoneType]) -> Bool:
        return rebind[ArcPointer[O]](ptr)[].done()

    @implicit
    def __init__[O: Operator](out self: DynOperator[O.Out], var value: O):
        var ptr = ArcPointer[O](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_push = Self._push_tramp[O]
        self._virt_drain = Self._drain_tramp[O]
        self._virt_done = Self._done_tramp[O]

    def push(mut self, morsel: Morsel) raises -> Optional[Self.Out]:
        return self._virt_push(self._data, morsel)

    def drain(mut self) raises -> Optional[Self.Out]:
        return self._virt_drain(self._data)

    def done(self) -> Bool:
        return self._virt_done(self._data)


trait Evaluable(Copyable, Deinitable):
    """A lane's fused driver — the one thing a processor needs to call.

    Deliberately *not* `Value`. `evaluate` is execution, so it does not belong
    on a logical trait; this names the internal capability each lane offers its
    own processor, and nothing outside a lane is bound on it.
    """

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        ...


struct EvalOperator[V: Evaluable](Operator):
    """An elementwise value, as an `Operator`.

    The adapter that lets one method — `to_operator` — serve every logical
    node. An elementwise value has all its output ready as soon as it sees a
    batch, so it answers from `push` and has nothing to flush; an aggregate is
    the mirror image. Neither needs a trait the other does not have.

    Generic over the node type rather than holding a `DynValue`, so a fused
    subtree stays one type through the boundary and is still inlined into one
    loop. That is the same reason `FoldOperator` is parameterised on its input.

    This is also where a value would keep state that outlives a batch — a
    compiled `LIKE` automaton, an `IsIn` hash set. It has none today, and the
    slot existing is the point: `evaluate(batch)` alone had nowhere to put one.
    """

    comptime Out = Datum

    var _value: Self.V

    def __init__(out self, var value: Self.V):
        self._value = value^

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        return self._value.evaluate(morsel.batch)

    def drain(mut self) raises -> Optional[Datum]:
        return None


# ---------------------------------------------------------------------------
# Pipeline — the assembled pipeline
# ---------------------------------------------------------------------------
struct Pipeline(Movable):
    """A chain of operators, stage 0 being the one that drives.

    Not a second abstraction beside `Operator` — just a `List` of them, so
    `Relation.to_operator` stays compositional: `InMemoryTable` creates a
    pipeline holding its source operator, and `Filter`, `Project`, `Sort`,
    `Limit` and `Aggregate` each `append` a stage to their input's. Nothing has
    to walk the plan, which matters because `DynRelation` deliberately exposes
    no `children()`.

    It was called `DynProcessor`, which was wrong twice over: it erases
    nothing, and "processor" was a second word for what `Operator` already
    names.
    """

    var _ops: List[DynOperator[RecordBatch]]

    def __init__(out self, var source: DynOperator[RecordBatch]):
        self._ops = List[DynOperator[RecordBatch]]()
        self._ops.append(source^)

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
        """Run the chain to completion and drain it into one batch.

        **One loop, not two.** Driving the source and flushing the chain used
        to be separate code paths because a source had a different method
        (`next`) from an operator (`finish`). With a repeatable `drain` they
        are the same act — "produce without new input" — so the driver walks
        the stages in order, drains each until it runs dry, and pushes whatever
        comes out through the stages above it.

        That ordering is what makes the flush a **cascade**: when stage `i`
        finally yields its result, no later stage has seen it, so it must flow
        through `i+1..` before stage `i+1` is drained. A projection over an
        aggregate returns nothing without this.
        """
        var out = List[RecordBatch]()
        for i in range(len(self._ops)):
            while True:
                var produced = self._ops[i].drain()
                if not produced:
                    break
                self._flow(
                    Morsel.ungrouped(produced.value().copy()), i + 1, out
                )
                # Early termination. A `Limit` that has its rows stops the
                # chain rather than letting the source drain — the one thing a
                # push engine must add back that a pull engine got for free.
                var stop = False
                for j in range(len(self._ops)):
                    if self._ops[j].done():
                        stop = True
                if stop:
                    break

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

        var sch = out[0].schema.copy()
        var cols = List[DynArray](capacity=len(sch.fields))
        for i in range(len(sch.fields)):
            var parts = List[DynArray]()
            for ref b in out:
                parts.append(b.columns[i].copy())
            # Serial: this runs once at the end over already-materialised
            # morsels, so there is nothing for workers to overlap with.
            cols.append(concat(parts, ExecContext.serial()))
        return RecordBatch(schema=sch^, columns=cols^)


# ---------------------------------------------------------------------------
# Streaming operators
# ---------------------------------------------------------------------------
struct FilterOperator(Operator):
    """Keeps rows where the predicate is true."""

    comptime Out = RecordBatch

    var _predicate: DynOperator[Datum]
    var _ctx: ExecContext

    def __init__(
        out self, var predicate: DynOperator[Datum], var ctx: ExecContext
    ):
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
        var produced = self._predicate.push(morsel)
        if not produced:
            return None
        var mask = produced.value().to_array(batch.num_rows())
        var cols = List[DynArray]()
        for i in range(batch.num_columns()):
            cols.append(filter(batch.columns[i].copy(), mask.copy(), self._ctx))
        var out = RecordBatch(schema=batch.schema.copy(), columns=cols^)
        if out.num_rows() > 0:
            return out^
        return None

    def drain(mut self) raises -> Optional[RecordBatch]:
        return None


struct ProjectOperator(Operator):
    """Evaluates each projected value against every morsel."""

    comptime Out = RecordBatch

    var _values: List[DynOperator[Datum]]
    var _schema: Schema

    def __init__(
        out self, var values: List[DynOperator[Datum]], var schema: Schema
    ):
        self._values = values^
        self._schema = schema^

    def push(mut self, morsel: Morsel) raises -> Optional[RecordBatch]:
        ref batch = morsel.batch
        var cols = List[DynArray](capacity=len(self._values))
        # Indexed: an operator is move-only, so a `List` of them cannot be
        # iterated by reference.
        for i in range(len(self._values)):
            var d = self._values[i].push(morsel)
            # `Datum.to_array` is where a scalar-shaped value stops being lazy:
            # a projection of a constant materialises here and nowhere earlier.
            cols.append(d.value().to_array(batch.num_rows()))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)

    def drain(mut self) raises -> Optional[RecordBatch]:
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
    `Pipeline.collect` delivers.
    """

    comptime Out = RecordBatch

    var _keys: List[DynOperator[Datum]]
    var _folds: List[DynOperator[Datum]]
    var _schema: Schema
    var _grouping: HashGrouping
    var _keyless: Bool
    var _ctx: ExecContext
    var _num_groups: Int
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynOperator[Datum]],
        var folds: List[DynOperator[Datum]],
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

    def _key_columns(mut self, morsel: Morsel) raises -> List[DynArray]:
        """The key expressions, evaluated against this morsel.

        Evaluated **once** and handed to the grouping, not once per aggregate:
        that is why placement is the operator's business rather than each
        fold's, and it is what a fold-shaped-as-an-independent-operator design
        would have to give up.
        """
        var n = morsel.batch.num_rows()
        var children = List[DynArray](capacity=len(self._keys))
        for i in range(len(self._keys)):
            var d = self._keys[i].push(morsel)
            children.append(d.value().to_array(n))
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
        register-fold win lives in `FoldOperator`, already monomorphised on `G`.
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
            self._key_columns(morsel), batch.num_rows()
        )
        self._num_groups = groups.num_groups
        var forwarded = Morsel(batch.copy(), groups^)
        for i in range(len(self._folds)):
            _ = self._folds[i].push(forwarded)
        return None

    def drain(mut self) raises -> Optional[RecordBatch]:
        if self._emitted:
            return None
        self._emitted = True
        var cols = List[DynArray]()
        if not self._keyless:
            cols = self._grouping.key_columns(self._key_fields())
        for i in range(len(self._folds)):
            var col = self._folds[i].drain()
            if col:
                cols.append(col.value().to_array(self._num_groups))
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

    def drain(mut self) raises -> Optional[RecordBatch]:
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

    var _keys: List[DynOperator[Datum]]
    var _ascending: List[Bool]
    var _nulls_first: Bool
    var _batches: List[RecordBatch]
    var _ctx: ExecContext
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynOperator[Datum]],
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

    def drain(mut self) raises -> Optional[RecordBatch]:
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
            var key = (
                self._keys[k]
                .push(Morsel.ungrouped(whole.copy()))
                .value()
                .to_array(whole.num_rows())
            )
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


struct BatchSourceOperator(Operator):
    """Yields one in-memory batch, once — **an ordinary `Operator`**.

    There is no `Source` trait. A source is just the operator that never has
    `push` called on it and answers from `drain` until it runs dry, which is
    exactly what a repeatable `drain` means. Sources are still *pull* in the
    sense that matters — a scan is a generator over I/O and nothing asks it to
    invert — but that is a property of this conformer, not a second
    abstraction the whole layer has to carry.
    """

    comptime Out = RecordBatch

    var _batch: RecordBatch
    var _done: Bool

    def __init__(out self, var batch: RecordBatch):
        self._batch = batch^
        self._done = False

    def push(mut self, morsel: Morsel) raises -> Optional[RecordBatch]:
        # A source consumes nothing; the driver never calls this.
        return None

    def drain(mut self) raises -> Optional[RecordBatch]:
        if self._done:
            return None
        self._done = True
        return self._batch.copy()
