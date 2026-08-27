"""The physical layer: operators that transform batches as they are pushed.

A `Relation` describes a query and is immutable, shareable and rewritable. This
is what that description becomes when it runs, and it owns everything mutable —
a grouper's table, an accumulator's slots.

**The engine pushes.** An operator is handed a batch and answers with what it
produced, if anything:

    push(batch) -> Optional[StructArray]
    finish()    -> Optional[StructArray]

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
from ..scalars import DynScalar
from std.utils import Variant
from ..builders import nulls
from ..dtypes import Field, struct_
from ..kernels.concat import concat
from ..execution import ExecContext
from ..kernels.filter import filter, take
from ..kernels.core import Groups
from ..kernels.aggregate import AggregateFn
from ..kernels.groupby import HashGrouping
from ..dtypes import DynType
from ..parquet.reader import LeafSet, ParquetFile
from ..parquet.source import MappedFile
from ..kernels.join import HashJoin, JoinKind
from ..utils import RapidHash64
from .runtime.aggregates import RuntimeAggregate
from .params import Bindings
from ..kernels.sort import sort_indices
from ..schema import Schema
from ..tabular import RecordBatch


# ---------------------------------------------------------------------------
# Datum — the wire format between stages
# ---------------------------------------------------------------------------
struct Datum(Copyable, Movable):
    """`Scalar | Array` — Arrow's Datum, DataFusion's ColumnarValue.

    What every `evaluate` returns. A struct rather than a bare `Variant` alias
    so that the one operation callers actually perform on it — *give me a
    column* — is a method on the thing rather than a free function beside it,
    and so the variant's members stay closed: nothing outside can reach in and
    handle the two cases differently.

    The scalar case is what makes it worth having. `lit(1)` evaluates to one
    value, not a million copies of one value, and stays that way until
    something needs a column. A literal-only subtree therefore costs nothing
    until it meets a batch.
    """

    var _v: Variant[DynScalar, DynArray]

    @implicit
    def __init__(out self, var value: DynArray):
        self._v = Variant[DynScalar, DynArray](value^)

    @implicit
    def __init__(out self, var value: DynScalar):
        self._v = Variant[DynScalar, DynArray](value^)

    def is_scalar(self) -> Bool:
        """Whether this is still one value.

        The planner asks so it knows whether a projection needs materialising;
        `Shape` is the *static* answer to the same question, and this is the
        one that survives erasure.
        """
        return self._v.isa[DynScalar]()

    def struct_array(self) raises -> StructArray:
        """The batch a relational stage produced.

        Relational stages put their struct array here and this takes it back
        out; both directions are refcount bumps. It raises if a scalar-shaped
        datum reaches a relational position — that is a bug in the plan, not a
        shape worth handling.
        """
        if self.is_scalar():
            raise Error("struct_array: expected a batch, got a scalar")
        return self._v[DynArray].as_struct().copy()

    def to_array(self, n: Int) raises -> DynArray:
        """This value as a column of length `n`, broadcasting a scalar.

        The single place laziness ends. `n` is why this cannot be an implicit
        conversion: a scalar does not know how many rows it is about to become,
        and only the caller holding the batch does.
        """
        if self._v.isa[DynScalar]():
            return self._v[DynScalar].repeat(n)
        return self._v[DynArray].copy()


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

    var batch: StructArray
    var groups: Groups

    def __init__(out self, var batch: StructArray, var groups: Groups):
        self.batch = batch^
        self.groups = groups^

    @staticmethod
    def ungrouped(var batch: StructArray) raises -> Morsel:
        """A morsel with the trivial one-slot assignment, for the relational
        chain and for a query with no `GROUP BY`."""
        var n = len(batch)
        return Morsel(batch^, Groups.single(n))


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

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Consume one morsel; answer what it produced, if anything."""
        ...

    def drain(mut self) raises -> Optional[Datum]:
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


struct DynOperator(Movable):
    """An `Operator` of any stage, erased — **one box, parameterised on `Out`**.

    `DynOperator` carries the relational chain and
    `DynOperator` carries a value's, but they are two instantiations of
    one definition rather than two hand-written boxes, so the erasure surface
    stays single.

    Move-only: an operator owns mutable state, so copying one would fork an
    execution mid-stream. `DynRelation` copies freely for the opposite reason —
    it owns nothing that runs.
    """

    var _data: ArcPointer[NoneType]
    var _virt_push: def(ArcPointer[NoneType], Morsel) thin raises -> Optional[
        Datum
    ]
    var _virt_drain: def(ArcPointer[NoneType]) thin raises -> Optional[Datum]
    var _virt_done: def(ArcPointer[NoneType]) thin -> Bool

    @staticmethod
    def _push_tramp[
        O: Operator
    ](ptr: ArcPointer[NoneType], morsel: Morsel) raises -> Optional[Datum]:
        return rebind[ArcPointer[O]](ptr)[].push(morsel)

    @staticmethod
    def _drain_tramp[
        O: Operator
    ](ptr: ArcPointer[NoneType]) raises -> Optional[Datum]:
        return rebind[ArcPointer[O]](ptr)[].drain()

    @staticmethod
    def _done_tramp[O: Operator](ptr: ArcPointer[NoneType]) -> Bool:
        return rebind[ArcPointer[O]](ptr)[].done()

    @implicit
    def __init__[O: Operator](out self, var value: O):
        var ptr = ArcPointer[O](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_push = Self._push_tramp[O]
        self._virt_drain = Self._drain_tramp[O]
        self._virt_done = Self._done_tramp[O]

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        return self._virt_push(self._data, morsel)

    def drain(mut self) raises -> Optional[Datum]:
        return self._virt_drain(self._data)

    def done(self) -> Bool:
        return self._virt_done(self._data)


trait Evaluable(Copyable, Deinitable):
    """A lane's fused driver — the one thing an operator needs to call.

    Deliberately *not* `Value`. `evaluate` is execution, so it does not belong
    on a logical trait; this names the internal capability each lane offers its
    own operator, and nothing outside a lane is bound on it.
    """

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
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

    var _value: Self.V
    var _bindings: Bindings
    """This execution's parameter values. Held by the *operator*, not the node:
    that is what keeps a plan a description and lets two executions bind
    different values."""

    def __init__(out self, var value: Self.V, var bindings: Bindings):
        self._value = value^
        self._bindings = bindings^

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        return self._value.evaluate(morsel.batch, self._bindings)

    def drain(mut self) raises -> Optional[Datum]:
        return None


# ---------------------------------------------------------------------------
# Pipeline — the assembled pipeline
# ---------------------------------------------------------------------------
struct Pipeline(Operator):
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

    **It is itself an `Operator`** — a composite one. A chain of stages pushes,
    drains and finishes exactly like a single stage, so it needs no concept of
    its own. That is not decoration: `Join` has *two* inputs, and each of them
    is a whole sub-plan. Only if a pipeline is an operator can a join hold two
    of them.

    Relations still build it *concretely* rather than through the box, because
    `append` needs the concrete type. Composing through `DynOperator` would
    nest one pipeline per stage and charge every batch an extra trampoline per
    level; appending keeps a plan's stages in one flat list. Composite where it
    buys something, flat where it does not.
    """

    var _ops: List[DynOperator]
    var _stage: Int
    """How far `drain` has walked. A composite operator has to remember its own
    position, because `drain` answers one batch per call."""

    var _pending: List[StructArray]

    def __init__(out self, var source: DynOperator):
        self._ops = List[DynOperator]()
        self._ops.append(source^)
        self._stage = 0
        self._pending = List[StructArray]()

    def append(mut self, var op: DynOperator):
        self._ops.append(op^)

    def _flow(
        mut self,
        var morsel: Morsel,
        start: Int,
        mut out: List[StructArray],
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
                    cur = Morsel.ungrouped(produced.value().struct_array())
                else:
                    cur = None
            else:
                return
        if cur:
            out.append(cur.value().batch.copy())

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Push a morsel through every stage.

        **Vestigial today, and the docstring should say so rather than imply a
        capability the type does not have.** `__init__` requires a source as
        stage 0, and a source ignores `push`, so for every pipeline that can
        currently be constructed this walks the chain and answers `None`. It
        exists to satisfy `Operator`.

        It becomes meaningful the moment a pipeline can be built *without* a
        source — a join's probe side fed by its parent rather than scanning.
        """
        var out = List[StructArray]()
        self._flow(morsel.copy(), 0, out)
        if len(out) == 0:
            return None
        for i in range(1, len(out)):
            self._pending.append(out[i].copy())
        return Datum(out[0].copy().to_dyn())

    def done(self) -> Bool:
        for i in range(len(self._ops)):
            if self._ops[i].done():
                return True
        return False

    def drain(mut self) raises -> Optional[Datum]:
        """One batch of whatever the chain still has, `None` when spent.

        Walks the stages in order, draining each dry and cascading what it
        yields through the stages above — the same cascade `collect` needs, but
        resumable, because an `Operator` answers one batch at a time. `_stage`
        is the resume point.
        """
        while True:
            if len(self._pending) > 0:
                return Datum(self._pending.pop(0).to_dyn())
            if self._stage >= len(self._ops):
                return None
            var produced = self._ops[self._stage].drain()
            if produced:
                var out = List[StructArray]()
                self._flow(
                    Morsel.ungrouped(produced.value().struct_array()),
                    self._stage + 1,
                    out,
                )
                for ref b in out:
                    self._pending.append(b.copy())
                # Early termination: a `Limit` with its rows stops the chain
                # rather than letting the source drain — the one thing a push
                # engine must add back that a pull engine got for free.
                if self.done():
                    self._stage = len(self._ops)
            else:
                self._stage += 1

    def collect(mut self, schema: Schema) raises -> StructArray:
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
        var out = List[StructArray]()
        while True:
            var b = self.drain()
            if b:
                out.append(b.value().struct_array())
            else:
                break

        # Serial: this runs once at the end over already-materialised morsels,
        # so there is nothing for workers to overlap with. An empty result is
        # still a *well-formed* batch — `RecordBatch.empty` gives one
        # zero-length column per field, so anything walking columns by schema
        # index stays in bounds.
        return _concat_batches(out, schema, ExecContext.serial())


# ---------------------------------------------------------------------------
# Streaming operators
# ---------------------------------------------------------------------------
struct FilterOperator(Operator):
    """Keeps rows where the predicate is true."""

    var _predicate: DynOperator
    var _ctx: ExecContext

    def __init__(out self, var predicate: DynOperator, var ctx: ExecContext):
        self._predicate = predicate^
        self._ctx = ctx^

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
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
        var mask = produced.value().to_array(len(batch))
        var out = (
            filter(batch.copy().to_dyn(), mask.copy(), self._ctx)
            .as_struct()
            .copy()
        )
        if len(out) > 0:
            return Datum(out^.to_dyn())
        return None

    def drain(mut self) raises -> Optional[Datum]:
        return None


struct ProjectOperator(Operator):
    """Evaluates each projected value against every morsel."""

    var _values: List[DynOperator]
    var _schema: Schema

    def __init__(out self, var values: List[DynOperator], var schema: Schema):
        self._values = values^
        self._schema = schema^

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        ref batch = morsel.batch
        var cols = List[DynArray](capacity=len(self._values))
        # Indexed: an operator is move-only, so a `List` of them cannot be
        # iterated by reference.
        for i in range(len(self._values)):
            var d = self._values[i].push(morsel)
            # `Datum.to_array` is where a scalar-shaped value stops being lazy:
            # a projection of a constant materialises here and nowhere earlier.
            cols.append(d.value().to_array(len(batch)))
        return Datum(_struct_of(self._schema, cols^, len(batch)).to_dyn())

    def drain(mut self) raises -> Optional[Datum]:
        return None


struct AggregateOperator(Operator):
    """Blocking: fold every pushed morsel, then emit one row per group.

    The shape the push interface exists for — `push` answers `None` all the way
    through the stream and `finish` answers the whole result. Under the old
    pull design this needed a different trait from `Filter` and `Project`, and
    therefore a second erased box.

    **A fused fold buffers nothing; a `AggregateFn` buffers O(rows).** Which of
    the two a query pays for is decided per aggregate, not here.
    `FoldOperator` binds its own input subtree and folds lanes straight out of
    the morsel, so `sum(a * 2 + b)` never materialises `a * 2 + b` and this
    stage keeps only the grouper's key builders, which grow with the number of
    *groups*. `ColumnAggregateOperator` cannot: a `AggregateFn` is one-shot
    over the whole input, so it accumulates each morsel's evaluated columns
    and ids and calls the fold once at `drain`. Any query containing a
    `count_distinct`, or a `min`/`max` over a string or temporal column, is
    therefore **O(rows)** in memory for that aggregate's input. There is
    precedent — `SortOperator._batches`, `JoinOperator._buffered` — and it is
    the price of an aggregate whose state is not a scalar accumulator.

    Group ids are dense and stable across morsels, which is what makes both
    halves work: a state that has already folded batch N keeps its slots when
    batch N+1 introduces new groups (`AggState._grow` extends them in place),
    and concatenating N morsels' id arrays end to end is a valid assignment
    over the concatenated input rather than N unrelated numberings.

    A fold that answers `None` from `drain` never saw a morsel at all — the
    input filtered to nothing — and had no dtype to resolve against. Its slot
    is filled with a null column typed from this stage's own output schema,
    which is the only place that type is still known.

    `HAVING` needs no node of its own: a `FilterOperator` above this stage sees
    the aggregate's *output* batch, which is exactly what the flush cascade in
    `Pipeline.collect` delivers.
    """

    var _keys: List[DynOperator]
    var _folds: List[DynOperator]
    var _schema: Schema
    var _grouping: HashGrouping
    var _keyless: Bool
    var _ctx: ExecContext
    var _num_groups: Int
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynOperator],
        var folds: List[DynOperator],
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
        var n = len(morsel.batch)
        var children = List[DynArray](capacity=len(self._keys))
        for i in range(len(self._keys)):
            var d = self._keys[i].push(morsel)
            children.append(d.value().to_array(n))
        return children^

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Assign the rows to slots **once**, then hand the same morsel to
        every fold.

        This is why placement belongs to the operator rather than to each grouped:
        N aggregates over one `GROUP BY` hash the keys once between them, where
        N folds each owning a grouping would hash N times.

        Placement is a runtime choice here and a comptime one inside the grouped,
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
            self._key_columns(morsel), len(batch)
        )
        self._num_groups = groups.num_groups
        var forwarded = Morsel(batch.copy(), groups^)
        for i in range(len(self._folds)):
            _ = self._folds[i].push(forwarded)
        return None

    def drain(mut self) raises -> Optional[Datum]:
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
            else:
                # The output schema is keys then aggregates, so aggregate `i`
                # is field `len(self._keys) + i`. Appending nothing instead
                # would hand `_struct_of` fewer children than its dtype has
                # fields — a batch that runs and mis-indexes.
                cols.append(
                    nulls(
                        self._num_groups,
                        self._schema.fields[len(self._keys) + i].dtype,
                    )
                )
        return Datum(_struct_of(self._schema, cols^, self._num_groups).to_dyn())


struct ColumnAggregateOperator(Operator):
    """An aggregate that consumes **materialised columns** rather than lanes.

    The counterpart of `FoldOperator`, and named for what it consumes. Both are
    *value* operators — two of the things `AggregateOperator` holds in its
    `_folds` list — and both answer a `Datum` of one value per slot. They
    differ in what they can do with a morsel: a fold reads `lane[W]` out of a
    fused subtree and keeps a scalar accumulator; this one has an aggregate
    whose state is a hash set, a sketch, or a best-row index, so its input has
    to exist as a column first.

    **Zero type parameters, shared by both lanes.** The comptime lane hands it
    `Agg.grouped` — a statically known method taken as a thin pointer — and the
    runtime lane hands it the `RuntimeAggregate` node to resolve on first push.
    The
    operand keeps its fusion either way, because this operator does not
    evaluate the operand: `_inputs` holds the operand's *own* operator, which
    for a comptime subtree is a fully monomorphised `EvalOperator[A]`.

    **It buffers, and that is inherent.** A `AggregateFn` is one-shot over the
    whole input, so each morsel's evaluated columns and group ids are kept and
    `concat`ed at `drain`. Concatenating ids across morsels is sound because
    `HashGrouping` assigns dense ids that are stable across batches — batch
    N+1 extends the numbering rather than restarting it.

    Buffering the ids per aggregate is a known, bounded redundancy: three
    column-aggregates in one query keep three copies of the same `Int32` per
    row. `AggregateOperator` sees every morsel and could keep one, but only by
    special-casing this operator among the values it holds, which is the
    uniformity that made "every value answers `to_operator`" work at all.

    **Exactly one of `_grouped` and `_node` is set at construction, and which
    one says which lane built this.** The comptime lane knows its `AggKernel`
    at compile time and supplies `_grouped` directly — which is what keeps a
    binary that only ever aggregates a comptime expression from linking the
    name ladder at all, and that DCE property is what the whole two-node design
    turns on. The runtime lane knows only a name, so it supplies `_node` and
    `_grouped` is filled on **first push**, from the morsel's real dtypes; it
    cannot be filled earlier, because a `RuntimeValue` operand has no dtype
    until a schema is in hand. `_grouped` is therefore `Some` from the first
    push onward in both cases, which is the invariant `drain` reads.

    Two nullable fields rather than a `Variant`: the alignment hazard
    CLAUDE.md records for `Variant` is not worth paying for two small members,
    so the invariant is stated here instead of encoded.
    """

    var _inputs: List[DynOperator]
    var _grouped: Optional[AggregateFn]
    """The implementation. `Some` from the first push onward, in both lanes."""

    var _node: Optional[RuntimeAggregate]
    """The node itself, for the lane that has a name instead of a type — an
    operator here already holds its node (`FoldOperator._input`,
    `EvalOperator._value`), so the ladder needs no object of its own."""

    var _empty: Optional[DynArray]
    """This aggregate's answer over an input that produced no column at all.
    Known without any dtype, so it is computed at construction rather than
    being a third code path at `drain`."""

    var _scatters: Bool
    """Whether the query has `GROUP BY` keys.

    A runtime `Bool` rather than a per-morsel `Groups.is_single()` read: a
    grouped query whose first batch happens to hold exactly one group would
    otherwise be mistaken for an ungrouped one, and that batch's ids dropped.
    """

    var _chunks: List[List[DynArray]]
    var _ids: List[DynArray]
    var _num_groups: Int
    var _rows: Int
    var _emitted: Bool

    def __init__(
        out self,
        var inputs: List[DynOperator],
        grouped: AggregateFn,
        var empty: Optional[DynArray],
        scatters: Bool,
    ):
        """The comptime lane's constructor: the aggregate is a type, so its
        implementation is already known."""
        self._chunks = List[List[DynArray]]()
        for _ in range(len(inputs)):
            self._chunks.append(List[DynArray]())
        self._inputs = inputs^
        self._grouped = grouped
        self._node = None
        self._empty = empty^
        self._scatters = scatters
        self._ids = List[DynArray]()
        self._num_groups = 0 if scatters else 1
        self._rows = 0
        self._emitted = False

    def __init__(
        out self,
        var inputs: List[DynOperator],
        var node: RuntimeAggregate,
        scatters: Bool,
    ) raises:
        """The runtime lane's constructor: the aggregate is a name, so the
        implementation waits for the first morsel's dtypes."""
        self._chunks = List[List[DynArray]]()
        for _ in range(len(inputs)):
            self._chunks.append(List[DynArray]())
        self._inputs = inputs^
        self._grouped = None
        self._empty = node.empty()
        self._node = node^
        self._scatters = scatters
        self._ids = List[DynArray]()
        self._num_groups = 0 if scatters else 1
        self._rows = 0
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Evaluate this aggregate's inputs against the morsel and keep them.

        Empty batches are kept too rather than skipped: an empty column still
        carries its dtype, which is the one thing a name-resolved fold needs
        and the one thing a later batch cannot supply retroactively.
        """
        var n = len(morsel.batch)
        # Indexed: an operator is move-only, so a `List` of them cannot be
        # iterated by reference.
        for i in range(len(self._inputs)):
            var d = self._inputs[i].push(morsel)
            self._chunks[i].append(d.value().to_array(n))
        if not self._grouped:
            var in_dtypes = List[DynType](capacity=len(self._chunks))
            for i in range(len(self._chunks)):
                in_dtypes.append(self._chunks[i][0].dtype())
            self._grouped = self._node.value().resolve(in_dtypes).grouped
        self._rows += n
        if self._scatters:
            self._ids.append(morsel.groups.ids.copy().to_dyn())
            self._num_groups = max(self._num_groups, morsel.groups.num_groups)
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        if len(self._chunks) == 0 or len(self._chunks[0]) == 0:
            # No morsel ever arrived, so there is no dtype to resolve against.
            # A distinct count still has an answer; an extremum does not, and
            # `AggregateOperator` fills its slot from the output schema.
            if self._empty:
                return Datum(self._empty.value().copy())
            return None
        var columns = List[DynArray](capacity=len(self._chunks))
        for i in range(len(self._chunks)):
            columns.append(concat(self._chunks[i], ExecContext.serial()))
        var groups: Groups
        if self._scatters:
            groups = Groups(
                concat(self._ids, ExecContext.serial()).as_int32().copy(),
                self._num_groups,
            )
        else:
            groups = Groups.single(self._rows)
        return Datum(self._grouped.value()(groups, columns))


struct LimitOperator(Operator):
    """`OFFSET`/`LIMIT` — streaming, and the reason `Operator.done` exists.

    Skips `offset` rows, emits at most `length`, and then reports `done` so the
    driver stops pulling. Without that signal the source would drain in full and
    a `LIMIT 10` over a large scan would cost the whole scan.

    Slicing is zero-copy, so a limited batch shares its parent's buffers rather
    than compacting.
    """

    var _offset: Int
    var _length: Int
    var _skipped: Int
    var _emitted: Int

    def __init__(out self, offset: Int, length: Int):
        self._offset = offset
        self._length = length
        self._skipped = 0
        self._emitted = 0

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        ref batch = morsel.batch
        var n = len(batch)
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
        return Datum(batch.slice(start, wanted).to_dyn())

    def drain(mut self) raises -> Optional[Datum]:
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

    var _keys: List[DynOperator]
    var _ascending: List[Bool]
    var _nulls_first: Bool
    var _batches: List[StructArray]
    var _ctx: ExecContext
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynOperator],
        var ascending: List[Bool],
        nulls_first: Bool,
        var ctx: ExecContext,
    ):
        self._keys = keys^
        self._ascending = ascending^
        self._nulls_first = nulls_first
        self._batches = List[StructArray]()
        self._ctx = ctx^
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        self._batches.append(morsel.batch.copy())
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted or len(self._batches) == 0:
            return None
        self._emitted = True

        var schema = Schema.from_dtype(self._batches[0].dtype)
        var whole = _concat_batches(self._batches, schema, self._ctx)

        var order = Optional[Int32Array](None)
        for k in range(len(self._keys) - 1, -1, -1):
            var key = (
                self._keys[k]
                .push(Morsel.ungrouped(whole.copy()))
                .value()
                .to_array(len(whole))
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
            return Datum(whole^.to_dyn())
        return Datum(take(whole.copy().to_dyn(), order.value(), self._ctx))


struct BatchSourceOperator(Operator):
    """Yields one in-memory batch, once — **an ordinary `Operator`**.

    There is no `Source` trait. A source is just the operator that never has
    `push` called on it and answers from `drain` until it runs dry, which is
    exactly what a repeatable `drain` means. Sources are still *pull* in the
    sense that matters — a scan is a generator over I/O and nothing asks it to
    invert — but that is a property of this conformer, not a second
    abstraction the whole layer has to carry.
    """

    var _batch: StructArray
    var _done: Bool

    def __init__(out self, var batch: StructArray):
        self._batch = batch^
        self._done = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        # A source consumes nothing; the driver never calls this.
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._done:
            return None
        self._done = True
        return Datum(self._batch.copy().to_dyn())


struct JoinOperator(Operator):
    """Equijoin — **the operator with two inputs**, and the one the push
    interface had to be checked against.

    Its build side is a whole sub-plan, held as a `DynOperator`.
    That only became expressible when `Pipeline` started conforming to
    `Operator`: before, a chain of stages was a different kind of thing from a
    stage, so there was nowhere to put a second one.

    **The build side is drained to completion before the first probe.** A hash
    join is a pipeline breaker on one input and streaming on the other, and the
    two methods say exactly that — the build happens inside the first `push`,
    and probe morsels stream through afterwards.

    Some kinds also block on the *probe* side, and that is not a shortcut.
    LEFT, FULL, SEMI and ANTI each emit rows determined by every probe row
    taken together: LEFT/FULL/ANTI have a tail of unmatched build rows, and
    SEMI emits a build row once no matter how many probe rows hit it. Probing
    morsel-by-morsel recomputes those per morsel, so LEFT, FULL and ANTI would
    re-emit their tail once per morsel and SEMI would duplicate. Those kinds
    therefore buffer and probe once at `drain`. RIGHT is *not* among them: its
    extra rows are unmatched probe rows, and each probe row belongs to exactly
    one morsel, so it streams correctly. `expr/` reached the same conclusion
    and this carries it over deliberately.
    """

    var _build: DynOperator
    var _left_keys: List[Int]
    var _right_keys: List[Int]
    var _kind: JoinKind
    var _strictness: UInt8
    var _schema: Schema
    var _ctx: ExecContext
    var _index: Optional[HashJoin[RapidHash64]]
    var _buffered: List[StructArray]
    var _emitted: Bool

    def __init__(
        out self,
        var build: DynOperator,
        var left_keys: List[Int],
        var right_keys: List[Int],
        kind: JoinKind,
        strictness: UInt8,
        var schema: Schema,
        var ctx: ExecContext,
    ):
        self._build = build^
        self._left_keys = left_keys^
        self._right_keys = right_keys^
        self._kind = kind
        self._strictness = strictness
        self._schema = schema^
        self._ctx = ctx^
        self._index = None
        self._buffered = List[StructArray]()
        self._emitted = False

    def _blocks_on_probe_side(self) -> Bool:
        """Whether this kind's output depends on the whole probe side."""
        return (
            self._kind.emits_unmatched_left()
            or not self._kind.emits_right_columns()
        )

    def _ensure_built(mut self) raises:
        """Drain the build side and index it, once."""
        if self._index:
            return
        var parts = List[StructArray]()
        while True:
            var b = self._build.drain()
            if b:
                parts.append(b.value().struct_array())
            else:
                break
        var left = _concat_batches(parts, self._build_schema(), self._ctx)
        var index = HashJoin[RapidHash64](self._ctx.copy())
        index.build(left.copy(), self._left_keys)
        self._index = index^

    def _build_schema(self) -> Schema:
        """The build side's fields are the first `len(left_keys)`-agnostic
        prefix of the output schema for every kind that emits them."""
        return self._schema.copy()

    def _probe(mut self, batch: StructArray) raises -> StructArray:
        var result = self._index.value().probe(
            batch.copy(),
            self._right_keys,
            self._kind,
            self._strictness,
        )
        # Re-typed with the *declared* schema: the kernel names its output from
        # the arrays it joined, while `Join.schema()` is what the plan promised
        # its consumers. Handing back the kernel's dtype makes
        # `plan.schema() != plan.execute().schema`.
        return _struct_of(self._schema, result.children.copy(), len(result))

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        self._ensure_built()
        if self._blocks_on_probe_side():
            self._buffered.append(morsel.batch.copy())
            return None
        return Datum(self._probe(morsel.batch).to_dyn())

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        # Reached even when the probe side yielded nothing, which is what makes
        # `LEFT JOIN` over an empty right side emit the left rows at all.
        self._ensure_built()
        if not self._blocks_on_probe_side():
            return None
        var whole = _concat_batches(
            self._buffered, self._schema.copy(), self._ctx
        )
        return Datum(self._probe(whole).to_dyn())


def _struct_of(
    schema: Schema, var columns: List[DynArray], length: Int
) raises -> StructArray:
    """A struct array over `columns`, typed by `schema`.

    The execution layer works in struct arrays; this is how an operator that
    produced a fresh column list says so. `RecordBatch` appears only at the
    boundary (`BatchSourceOperator` in, `Pipeline.collect` out).
    """
    return StructArray(
        dtype=struct_(schema.fields.copy()),
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        children=columns^,
    )


def _concat_batches(
    batches: List[StructArray], schema: Schema, ctx: ExecContext
) raises -> StructArray:
    """Join both phases need one contiguous side; this is where that happens."""
    if len(batches) == 1:
        return batches[0].copy()
    if len(batches) == 0:
        return RecordBatch.empty(schema).to_struct_array()
    var parts = List[DynArray](capacity=len(batches))
    for ref b in batches:
        parts.append(b.copy().to_dyn())
    return concat(parts^, ctx).as_struct().copy()


struct ParquetScanOperator(Operator):
    """Reads a Parquet file, one **row group** per `drain`.

    A source like `BatchSourceOperator`, and the reason sources stayed pull:
    this is a generator over I/O, and nothing would be gained by inverting it.
    `drain` answering repeatedly until `None` is exactly a scan's shape — which
    is why merging `Source` into `Operator` cost nothing here.

    One row group at a time rather than the whole file, so resident data is
    bounded by a row group rather than by the file. `read()` decodes only the
    requested group, so an unread group is never touched.

    **The schema doubles as the projection.** Only its columns are read, so
    narrowing a scan's schema is how a projection is pushed into it.

    **No pruning.** `expr/`'s scan skips row groups whose statistics prove no
    row can match, and windows several groups at once. Both need
    `expr/pruning.mojo`, which has no `expr2` counterpart yet. Their absence
    costs speed and never correctness — a `Filter` above the scan applies the
    predicate exactly — so this is a smaller scan, not a wrong one.
    """

    var _path: String
    var _schema: Schema
    var _file: Optional[ParquetFile[MappedFile, LeafSet.all()]]
    var _next_group: Int
    var _pending: List[StructArray]

    def __init__(out self, var path: String, var schema: Schema):
        self._path = path^
        self._schema = schema^
        self._file = None
        self._next_group = 0
        self._pending = List[StructArray]()

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        # A source consumes nothing; the driver never calls this.
        return None

    def drain(mut self) raises -> Optional[Datum]:
        while len(self._pending) == 0:
            if not self._file:
                # Opened on first use, not at plan time: a `Relation` is a
                # description and must not touch the filesystem to exist.
                self._file = ParquetFile[MappedFile, LeafSet.all()](
                    self._path.copy()
                )
            if self._next_group >= self._file.value().num_row_groups():
                return None

            var names = List[String](capacity=len(self._schema.fields))
            for ref f in self._schema.fields:
                names.append(f.name.copy())
            var groups = List[Int]()
            groups.append(self._next_group)
            self._next_group += 1

            var table = self._file.value().read(
                columns=Optional(names^), row_groups=Optional(groups^)
            )
            # A row group can decode to several chunks; each becomes a morsel
            # rather than being concatenated back together.
            for ref b in table.to_batches():
                self._pending.append(b.to_struct_array())
        return Datum(self._pending.pop(0).to_dyn())
