"""The physical layer: operators that transform batches as they are pushed.

A `Relation` describes a query and is immutable, shareable and rewritable. This
is what that description becomes when it runs, and it owns everything mutable —
a grouper's table, an accumulator's slots.

**The engine pushes.** An operator is handed a morsel and answers with what it
produced, if anything:

    push(morsel: Morsel) -> Optional[Datum]
    drain()              -> Optional[Datum]
    done()               -> Bool

That one interface covers streaming and blocking alike, because **blocking
stops being a type distinction and becomes *when you return `Some`***. `Filter`
and `Project` answer from `push` and nothing from `drain`; an aggregate
accumulates through every `push`, answers `None`, and produces its whole result
from `drain`. Under the old pull design those were two different shapes and
therefore two different erased boxes.

`drain` rather than a one-shot `finish`: it is repeatable, which is what lets
the same trait cover sources, and it lets an operator emit *several* batches at
end of stream — a chunking sort, a fanning join.

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
from ..builders import Int32Builder, nulls
from ..dtypes import Field, field, struct_
from ..kernels.aggregate import AggKernel
from ..kernels.concat import concat
from ..execution import ExecContext
from ..kernels.filter import filter, take
from ..kernels.groups import Groups
from ..kernels.groupby import HashGrouping
from ..dtypes import DynType
from ..parquet.reader import LeafSet, ParquetFile
from ..parquet.source import MappedFile
from ..kernels.join import HashJoin, JoinKind
from ..utils import RapidHash64
from .bindings import Bindings
from .logical import DynValue, WindowExpr
from .pushdown import Pushdown, read_plan, row_group_stats
from ..kernels.sort import SortIndices, sort_indices
from ..kernels.window import WindowExtents, mark_changes
from ..schema import Schema, schema
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

        `Shape` is the *static* answer to the same question and is what the
        planner actually reads; this is the one that survives erasure, and its
        only caller is `struct_array`'s own guard below. Kept public rather
        than folded into that guard because the distinction is part of what a
        `Datum` *is* — a caller holding one has no other way to ask.
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

        **`n` binds the array case too** — backlog AG-2. It used to be read
        only on the scalar branch and ignored entirely on the array one, so a
        column of the wrong length passed straight through into
        `_struct_of(schema, cols, n)`, which trusts its `length` argument and
        never looks at its children. The result is a `StructArray` claiming `n`
        rows over a child that has fewer: not a crash, but an out-of-bounds
        read the moment anything indexes it. An aggregate over an input that
        produced no morsel at all reached exactly that shape. Checking here
        turns the whole class into a raise naming both numbers.
        """
        if self._v.isa[DynScalar]():
            return self._v[DynScalar].repeat(n)
        ref arr = self._v[DynArray]
        if len(arr) != n:
            raise Error(
                "to_array: expected a column of ",
                n,
                " rows, got ",
                len(arr),
            )
        return arr.copy()


# ---------------------------------------------------------------------------
# Morsel — the unit that flows through the pipeline
# ---------------------------------------------------------------------------
struct Morsel(Copyable, Movable):
    """A batch, plus the group each of its rows belongs to.

    Carrying the grouping *with* the batch is what collapses three executor
    shapes into one. A fold needs to know which slot a row contributes to;
    `push(batch)` alone cannot tell it — which would force an aggregate into
    its own trait and its own erased box. Put the assignment in the morsel and
    a fold is simply an `Operator` producing a column.

    Relational stages ignore `groups` entirely. They pay nothing for it: the
    ungrouped assignment holds an **empty** id array, because a fold that does
    not scatter never reads the ids and materialising one `Int32` per row to
    say "everything is group 0" is exactly the cost `Groups.single` avoids.
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

    Three methods — `push`, `drain` and `done` — and that is the whole physical
    contract. Streaming and blocking operators differ only in *when* they
    answer `Some`.

    Both answer a `Datum`, which is what lets one trait cover the two things
    that produce different shapes: a relational stage produces a batch, a
    *value*'s stage produces a column, and `Datum` holds either. Fixing the
    output at `RecordBatch` instead would force every value to wrap its column
    in a one-column batch — allocating a `Schema` per value per batch — only
    for `ProjectOperator` to unwrap N of them and reassemble one. That is a
    real runtime cost paid for a nominal unification.

    Deliberately not an associated `Out`: nothing would constrain it and
    nothing would read it.
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
    """An `Operator` of any stage, erased — **one box for both chains**.

    The relational chain and a value's chain go through the same box rather
    than two hand-written ones, because both stages answer a `Datum`, so the
    erasure surface stays single.

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
    var _virt_drop: def(var ArcPointer[NoneType]) thin
    """Erasure drops the pointee's destructor, so it has to be carried
    separately. `rebind[ArcPointer[NoneType]]` keeps the allocation and the
    refcount but forgets the type, so the final release runs `NoneType`'s
    destructor and **`O.__deinit__` never runs** — every operator's state leaks.
    Measured, not reasoned: a 200k-iteration probe over a triple-shared box
    reports zero destructions without this field and zero live objects with it.
    """

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

    @staticmethod
    def _drop_tramp[O: Operator](var ptr: ArcPointer[NoneType]):
        """Release the box's reference at the operator's *true* type.

        `rebind` takes a second reference, so the erased one can be released
        without reaching zero; the typed reference then falls out of scope and
        its release is the one that runs `O.__deinit__`. Sharing still works —
        a box that holds one of several references simply decrements.
        """
        var typed = rebind[ArcPointer[O]](ptr)
        _ = ptr^
        _ = typed^

    @implicit
    def __init__[O: Operator](out self, var value: O):
        var ptr = ArcPointer[O](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_push = Self._push_tramp[O]
        self._virt_drain = Self._drain_tramp[O]
        self._virt_done = Self._done_tramp[O]
        self._virt_drop = Self._drop_tramp[O]

    def __deinit__(deinit self):
        self._virt_drop(self._data^)

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
    loop. That is the same reason the fused aggregate operators are
    parameterised on their input.

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
        return self._first_done() >= 0

    def _first_done(self) -> Int:
        """The lowest stage that has everything it needs, or `-1`.

        `done` answers whether *any* stage is finished; `drain` needs to know
        *which*, because a finished stage stops the ones below it and says
        nothing about the ones above. Collapsing the two is what made an
        aggregate over a `Limit` return no rows.
        """
        for i in range(len(self._ops)):
            if self._ops[i].done():
                return i
        return -1

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
                # Early termination: a `Limit` with its rows stops the
                # stages **below** it from producing more — the one thing a
                # push engine must add back that a pull engine got for free.
                #
                # It must not skip the stages **above** it. This read
                # `self._stage = len(self._ops)`, which jumped past every
                # remaining stage, so an `Aggregate` over a `Limit` never had
                # `drain` called and emitted nothing at all: `SELECT sum(a)
                # FROM (SELECT * FROM t LIMIT 3)` returned zero rows where an
                # ungrouped aggregate must always return one. Anything that
                # answers only from `drain` — every aggregate, and a `Sort` —
                # was silently dropped the moment a `Limit` sat below it.
                var stop = self._first_done()
                if stop >= 0 and stop + 1 > self._stage:
                    self._stage = stop + 1
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


struct GroupByOperator(Operator):
    """Blocking: fold every pushed morsel, then emit one row per group.

    The shape the push interface exists for — `push` answers `None` all the way
    through the stream and `drain` answers the whole result. Under the old
    pull design this needed a different trait from `Filter` and `Project`, and
    therefore a second erased box.

    **Which aggregate machine a query pays for is decided per aggregate, not
    here.** `ScatteredAggregateOperator` and `RegisterAggregateOperator` bind
    their own input subtree and fold lanes straight out of the morsel, so
    `sum(a * 2 + b)` never materialises `a * 2 + b`;
    `BufferedAggregateOperator` evaluates its operand to a column per morsel
    and hands that to the kernel. None of the three buffers rows — every
    `AggKernel` is streaming, so all of them keep O(groups) state — and this
    stage keeps only the grouper's key builders, which grow with the number of
    *groups* too. That was not always so: a `count_distinct` used to hold its
    whole input column, and the docstring here recorded it as the price of a
    non-scalar accumulator long after `DistinctCount` became incremental.

    Group ids are dense and stable across morsels, which is what makes all of
    them work: a state that has already folded batch N keeps its slots when
    batch N+1 introduces new groups (`AggState.reserve` extends them in place),
    and concatenating N morsels' id arrays end to end is a valid assignment
    over the concatenated input rather than N unrelated numberings.

    A fold that answers `None` from `drain` would have its slot filled with a
    null column typed from this stage's own output schema, the only place that
    type is still known. No aggregate operator does — all three answer `Some`
    from their first `drain`, seeding one slot when the query has no keys — so
    the branch is a guard against a future `Operator` in this position rather
    than a path any query takes.

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

        This is why placement belongs to the operator rather than to each fold:
        N aggregates over one `GROUP BY` hash the keys once between them, where
        N folds each owning a grouping would hash N times.

        Placement is a runtime choice here and a comptime one inside the fold,
        and that split is measured. Parameterising *this* operator on a
        grouping trait instantiated it once per conformer for **+24,432
        bytes** and bought nothing: its branch runs once per batch, while the
        14.6x register-fold win lives one level down, in the split between
        `RegisterAggregateOperator` and `ScatteredAggregateOperator`. That
        measurement is also why the trait is gone — the one place a future
        conformer would have plugged in had already been tried and rejected.
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

    Buffers each morsel and sorts once at `drain`. That is not a limitation of
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
    var _limit: Optional[Int]
    """How many ordered rows the plan above actually needs, or `None` for all.

    Set only by `optimizer.mojo`'s `TopN` rule, which alone knows that nothing
    between the `Limit` and this `Sort` drops rows."""
    var _batches: List[StructArray]
    var _ctx: ExecContext
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynOperator],
        var ascending: List[Bool],
        nulls_first: Bool,
        limit: Optional[Int],
        var ctx: ExecContext,
    ):
        self._keys = keys^
        self._ascending = ascending^
        self._nulls_first = nulls_first
        self._limit = limit
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
            # **The bound applies to the primary key only.** The multi-key
            # decomposition sorts stably from the least significant key to the
            # most, composing each pass onto the previous permutation, so every
            # pass but the last must return a *full* permutation for the next
            # one to permute. Truncating an earlier pass discards rows the
            # later keys still have to order, which loses answers rather than
            # reordering them. `k == 0` is the final, most significant pass.
            var pass_order = sort_indices(
                key,
                ascending=self._ascending[k],
                nulls_first=self._nulls_first,
                stable=True,
                limit=self._limit if k == 0 else None,
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


struct WindowOperator(Operator):
    """`OVER (...)` — blocking, because a window function needs its partition.

    Buffers every morsel and answers once at `drain`, the same shape as
    `SortOperator` and for the same reason: no prefix of the input determines
    any output row, since the row that ends up adjacent may not have arrived.

    **The partitioning is a prefix of the sort key, not a second mechanism.**
    `PARTITION BY k ORDER BY v` is the ordering `[k, v]`, so one
    `SortIndices.multi` answers both questions, and `kernels/window.mojo` reads
    partition and peer boundaries straight off the result. Hash-partitioning
    first and sorting each bucket would need `HashGrouping`, a gather per
    bucket, and a second null convention to keep consistent with `GROUP BY`'s
    — three moving parts to express what the sort already expresses.

    **Rows come out in input order.** The sort is an internal device, so the
    computed columns are scattered back through the inverse permutation rather
    than the batch being reordered to match them. `with_columns` means
    `SELECT *, f() OVER ()`, and a verb that silently reordered its input
    would be a surprising thing for one added column to do — every golden case
    sorts afterwards and so could not tell, which is exactly why it is worth
    getting right here rather than relying on the consumer.
    """

    var _names: List[String]
    var _exprs: List[WindowExpr]
    """Every expression on this node shares one window spec — `with_columns`
    stacks a separate node per distinct spec, so one sort serves them all."""

    var _input_schema: Schema
    var _output_schema: Schema
    var _bindings: Bindings
    var _ctx: ExecContext
    var _batches: List[StructArray]
    var _emitted: Bool

    def __init__(
        out self,
        var names: List[String],
        var exprs: List[WindowExpr],
        var input_schema: Schema,
        var output_schema: Schema,
        var bindings: Bindings,
        var ctx: ExecContext,
    ):
        self._names = names^
        self._exprs = exprs^
        self._input_schema = input_schema^
        self._output_schema = output_schema^
        self._bindings = bindings^
        self._ctx = ctx^
        self._batches = List[StructArray]()
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        self._batches.append(morsel.batch.copy())
        return None

    def _eval(self, value: DynValue, batch: StructArray) raises -> DynArray:
        """`value` as a column over `batch`, the way `SortOperator` reads a
        key."""
        var op = value.to_operator(self._input_schema, False, self._bindings)
        var produced = op.push(Morsel.ungrouped(batch.copy()))
        return produced.value().to_array(len(batch))

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted or len(self._batches) == 0:
            return None
        self._emitted = True

        var whole = _concat_batches(
            self._batches, self._input_schema, self._ctx
        )
        var n = len(whole)

        # -- one ordering answers both questions ----------------------------
        var num_partition = len(self._exprs[0].partition_by)
        var keys = List[DynArray]()
        var ascending = List[Bool]()
        for ref k in self._exprs[0].partition_by:
            keys.append(self._eval(k, whole))
            # Direction is irrelevant to a partition: it groups equal rows,
            # and every order puts equal rows together.
            ascending.append(True)
        for i in range(len(self._exprs[0].order_by)):
            keys.append(self._eval(self._exprs[0].order_by[i], whole))
            ascending.append(self._exprs[0].ascending[i])

        var perm = self._permutation(
            keys, ascending, self._exprs[0].nulls_first, n
        )

        # -- boundaries ------------------------------------------------------
        var sorted_keys = List[DynArray](capacity=len(keys))
        for ref k in keys:
            sorted_keys.append(take(k.copy(), perm, self._ctx))

        var new_partition = List[Bool](length=n, fill=False)
        if n > 0:
            new_partition[0] = True
        for i in range(num_partition):
            mark_changes(sorted_keys[i], new_partition, self._ctx)
        var new_peer = new_partition.copy()
        for i in range(num_partition, len(sorted_keys)):
            mark_changes(sorted_keys[i], new_peer, self._ctx)
        var extents = WindowExtents(new_partition^, new_peer^)

        # -- the sorted batch, and the way back ------------------------------
        var sorted_columns = List[DynArray]()
        for i in range(len(self._input_schema.fields)):
            sorted_columns.append(take(whole.field(i), perm, self._ctx))
        var sorted_batch = _struct_of(self._input_schema, sorted_columns^, n)
        var inverse = self._inverse(perm, n)

        # -- one column per expression ---------------------------------------
        var columns = List[DynArray]()
        for i in range(len(self._input_schema.fields)):
            columns.append(whole.field(i))
        for ref e in self._exprs:
            var computed = self._compute(e, extents, sorted_batch)
            columns.append(take(computed^, inverse, self._ctx))

        return Datum(_struct_of(self._output_schema, columns^, n).to_dyn())

    def _permutation(
        self,
        keys: List[DynArray],
        ascending: List[Bool],
        nulls_first: Bool,
        n: Int,
    ) raises -> Int32Array:
        """The order the window is evaluated in.

        Identity when the window names no keys at all — `OVER ()` is one
        partition in input order, and `SortIndices.multi` rejects an empty key
        list rather than answering that.
        """
        if len(keys) == 0:
            var identity = Int32Builder(n)
            for i in range(n):
                identity.append(Int32(i))
            return identity.finish()
        var fields = List[Field](capacity=len(keys))
        var indices = List[Int](capacity=len(keys))
        for i in range(len(keys)):
            fields.append(field(String("k", i), keys[i].dtype()))
            indices.append(i)
        var key_batch = _struct_of(schema(fields^), keys.copy(), n)
        return SortIndices.multi(
            key_batch,
            indices,
            ascending,
            nulls_first=nulls_first,
            stable=True,
            ctx=self._ctx,
        )

    def _inverse(self, perm: Int32Array, n: Int) raises -> Int32Array:
        """`inv[perm[j]] = j` — where each input row landed in the sort.

        Gathering a sorted result under this puts it back beside the row it
        describes, which is what lets the batch itself stay in input order.
        """
        var positions = List[Int](length=n, fill=0)
        for j in range(n):
            positions[Int(perm[j].value())] = j
        var out = Int32Builder(n)
        for i in range(n):
            out.append(Int32(positions[i]))
        return out.finish()

    def _compute(
        self,
        expr: WindowExpr,
        extents: WindowExtents,
        sorted_batch: StructArray,
    ) raises -> DynArray:
        """One window expression's column, in **sorted** order."""
        # **Two branches, not eight.** Which window function this is lives in
        # `WindowExpr`'s slot, instantiated where the verb named it, so the
        # bodies of the ones this binary never writes are not linked. The
        # aggregate is the one kind with no slot: its argument is an ordinary
        # aggregate `Value`, so it runs through that value's own operator.
        if expr.is_aggregate():
            return self._framed_aggregate(expr, extents, sorted_batch)

        # The ranking functions read no column; the rest gather one.
        var argument: Optional[DynArray] = None
        if not expr.fixed_dtype:
            argument = self._eval(expr.argument.value(), sorted_batch)
        return expr.compute.value()(
            extents,
            argument^,
            expr.offset,
            expr.frame.is_rows,
            expr.frame.preceding,
            expr.frame.following,
            self._ctx,
        )

    def _framed_aggregate(
        self,
        expr: WindowExpr,
        extents: WindowExtents,
        sorted_batch: StructArray,
    ) raises -> DynArray:
        """An aggregate evaluated once per frame.

        **The aggregate runs through its own operator**, on a slice of the
        sorted batch. That is what makes every aggregate a window aggregate at
        once — `SUM`, `MIN`, `COUNT`, `AVG` and anything added later — with the
        kernel's own null semantics rather than a second implementation of
        them: `SUM` over an all-null frame answers null here because `SumFold`
        answers null, not because this file decided it should.

        The cost is one operator per row, since an aggregate accumulates and
        frames overlap, so nothing can be carried from one frame to the next
        through this interface. That is O(rows) operator constructions and it
        is the honest price of the reuse; a running accumulator would be a
        per-aggregate, per-dtype kernel and is what to write when this shows up
        in a profile.

        Slicing rather than gathering keeps the per-frame cost at O(1) for the
        batch itself — `StructArray.slice` is zero-copy and `field()` pushes
        the offset down to each child — so only the aggregate's own scan is
        linear in the frame.
        """
        var n = len(extents)
        var dtype = expr.dtype(self._input_schema)
        if n == 0:
            return nulls(0, dtype^)
        var parts = List[DynArray](capacity=n)
        for j in range(n):
            var start: Int
            var stop: Int
            if expr.frame.is_rows:
                start = max(
                    extents.partition_start[j], j + expr.frame.preceding
                )
                stop = min(
                    extents.partition_end[j], j + expr.frame.following + 1
                )
            else:
                # The default `RANGE` frame ends at the current row's peer
                # group, not at the current row — which is why `LAST_VALUE`
                # is not the partition's last value.
                start = extents.partition_start[j]
                stop = extents.peer_end[j]
            # **An empty frame still runs the aggregate**, on a zero-row
            # slice, rather than short-circuiting to null. The identity of the
            # empty set is the aggregate's to decide, not this loop's:
            # `COUNT` over no rows is 0 and `MIN` over no rows is NULL, and
            # the aggregate operator already answers both: a fused ungrouped
            # `count` lands in `RegisterAggregateOperator`, whose `drain`
            # emits `AggState.finish`'s `c == 0` case, and `CountFold` sets
            # `empty_is_null = False`. (`BufferedAggregateOperator` covers
            # `count_distinct` and the string extrema and agrees.)
            # Short-circuiting discarded
            # that and made every `COUNT(*) OVER (... ROWS BETWEEN 30
            # PRECEDING AND 1 PRECEDING)` report NULL for a partition's first
            # row where SQL reports 0.
            #
            # The upper bounds are clamped into the batch first: a frame
            # lying wholly past the end (`rows=(5, 10)` on a 3-row partition)
            # gives `start > len`, and slicing there would be out of range.
            # `max(0, ...)` cannot fire -- both branches above give a
            # non-negative `start` -- and is kept only so the two bounds read
            # symmetrically.
            var limit = len(sorted_batch)
            var lo = max(0, min(start, limit))
            var hi = max(lo, min(stop, limit))
            var frame = sorted_batch.slice(lo, hi - lo)
            var op = expr.argument.value().to_operator(
                self._input_schema, False, self._bindings
            )
            var produced = op.push(Morsel.ungrouped(frame^))
            if not produced:
                produced = op.drain()
            if produced:
                parts.append(produced.value().to_array(1))
            else:
                parts.append(nulls(1, dtype.copy()))
        return concat(parts^, self._ctx)


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
    one morsel, so it streams correctly. the previous expression package
    reached the same conclusion
    and this carries it over deliberately.
    """

    var _build: DynOperator
    var _left_keys: List[Int]
    var _right_keys: List[Int]
    var _kind: JoinKind
    var _strictness: UInt8
    var _schema: Schema
    """What this stage promises its consumers: left fields then right, or left
    alone for the kinds that emit no right columns."""

    var _build_schema: Schema
    """The **build** input's own fields — the left side's, not this stage's.

    A separate field because the two genuinely differ and confusing them is a
    wrong answer rather than a crash. It was a `_build_schema()` *method*
    returning `self._schema.copy()`, whose docstring described a
    "`len(left_keys)`-agnostic prefix of the output schema" that neither it nor
    anything else computed. `_concat_batches` reads a schema only when it is
    handed **no** batches, and then builds an empty batch from it — so a build
    side that yielded nothing was indexed as if it had the join's output
    columns. For an inner or left join that is the left fields plus a run of
    all-null right ones, and `probe` then hands `_struct_of` more children than
    the output dtype has fields.

    Invisible today only because `_concat_batches` short-circuits at
    `len(batches) == 1`, which is every build side that produced exactly one
    batch."""

    var _probe_schema: Schema
    """The **probe** input's own fields — the right side's. Same argument, and
    the same latent defect: `drain` concatenates the buffered probe morsels,
    and for `SEMI`/`ANTI` the output schema is the *left* side's, so an empty
    probe side was described with the wrong columns entirely."""

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
        var build_schema: Schema,
        var probe_schema: Schema,
        var ctx: ExecContext,
    ):
        self._build = build^
        self._left_keys = left_keys^
        self._right_keys = right_keys^
        self._kind = kind
        self._strictness = strictness
        self._schema = schema^
        self._build_schema = build_schema^
        self._probe_schema = probe_schema^
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
        var left = _concat_batches(parts, self._build_schema.copy(), self._ctx)
        var index = HashJoin[RapidHash64](self._ctx.copy())
        index.build(left.copy(), self._left_keys)
        self._index = index^

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
            self._buffered, self._probe_schema.copy(), self._ctx
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
    """Join both phases need one contiguous side; this is where that happens.

    `schema` describes **the side being concatenated**, not the join's output,
    and it is read only when `batches` is empty — the one case where there is
    no array to take a dtype from. Handing it the output schema is what
    `JoinOperator._build_schema` used to do, and it is wrong whenever the two
    differ."""
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

    **Pruning.** A predicate pushed down from a `Filter` is evaluated once per
    row group against its statistics, and a group proven to hold no matching
    row is never decoded. Speed only: the `Filter` above still applies the
    predicate exactly, so an empty `Pushdown` and a full one return the same
    rows. Row-group *windowing* is still absent and is a separate change.
    """

    var _path: String
    var _schema: Schema
    var _file: Optional[ParquetFile[MappedFile, LeafSet.all()]]
    var _pushed: Pushdown
    var _bindings: Bindings
    var _plan: List[Int]
    """Row groups this scan will read, computed on the first `drain`.

    Empty until the file is opened, because a `Relation` is a description and
    must not touch the filesystem to exist — so the plan cannot be built where
    the operator is."""

    var _next: Int
    var _pending: List[StructArray]

    def __init__(
        out self,
        var path: String,
        var schema: Schema,
        var pushed: Pushdown = Pushdown(),
        var bindings: Bindings = Bindings(),
    ):
        self._path = path^
        self._schema = schema^
        self._file = None
        self._pushed = pushed^
        self._bindings = bindings^
        self._plan = List[Int]()
        self._next = 0
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
                self._plan = read_plan(
                    row_group_stats(self._file.value()),
                    self._pushed,
                    self._bindings,
                )
            if self._next >= len(self._plan):
                return None

            var names = List[String](capacity=len(self._schema.fields))
            for ref f in self._schema.fields:
                names.append(f.name.copy())
            var groups = List[Int](capacity=1)
            groups.append(self._plan[self._next])
            self._next += 1

            var table = self._file.value().read(
                columns=Optional(names^), row_groups=Optional(groups^)
            )
            # A row group can decode to several chunks; each becomes a morsel
            # rather than being concatenated back together.
            for ref b in table.to_batches():
                self._pending.append(b.to_struct_array())
        return Datum(self._pending.pop(0).to_dyn())


struct BufferedAggregateOperator[Agg: AggKernel, A: Evaluable](Operator):
    """The aggregate that cannot fold lanes: evaluate the operand to a column,
    hand it to the kernel.

    Reached when `Agg` has no lane algebra (`count_distinct` keeps a hash set
    or a sketch, `min`/`max` over a string is a bytewise scan, a dispersion
    keeps Welford's triple) or the operand is not lane-readable.

    **The operand still stays typed.** Only the aggregation step
    materialises: `count_distinct(upper(region))` still compiles
    `upper(region)` into one fused loop, and `A` is a type parameter rather
    than a `DynValue` precisely so that fusion is not thrown away along with
    the aggregate's.

    **Not O(rows).** This buffered *columns* once, calling a one-shot
    `grouped` at `drain`; every `AggKernel` is now streaming, so each morsel is
    absorbed into per-slot state and released. The name is historical, and what
    it now buffers is one evaluated column at a time.

    **Here, not in `comptime/`, because it is not lane-specific.** Its two
    siblings bind on `PrimitiveValue` and belong to the fused lane; this one
    needs only that the operand can be evaluated to a column, which both lanes
    can do. Hence the bound is `Evaluable` alone -- `physical.mojo` does not
    import `logical.mojo`, so naming `Value` here would create a cycle.
    """

    var _input: Self.A
    var _bindings: Bindings
    """This execution's parameter values, held by the *operator* rather than
    the node — which is what keeps the plan immutable and lets two executions
    of it bind different values."""

    var _state: Self.Agg
    """The accumulator, built at construction from the operand's dtype."""

    var _scatters: Bool
    """Whether the query has `GROUP BY` keys.

    A field and not a parameter, unlike the fused operator's: this arm reads it
    once per morsel to build a `Groups`, not once per row, so specialising on
    it would double the instantiation for a branch that never reaches the inner
    loop — measured at +4.6%.
    """

    var _emitted: Bool

    def __init__(
        out self,
        var input: Self.A,
        var bindings: Bindings,
        scatters: Bool,
        in_dtype: DynType,
    ) raises:
        self._input = input^
        self._bindings = bindings^
        self._state = Self.Agg(in_dtype)
        self._scatters = scatters
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        var n = len(morsel.batch)
        var column = self._input.evaluate(
            morsel.batch, self._bindings
        ).to_array(n)
        var groups = morsel.groups.copy() if self._scatters else Groups.single(
            n
        )
        # The one narrowing in this lane, and it is comptime-resolved:
        # `Agg` is a parameter here, so `InArray` is a concrete type and
        # this is a conversion rather than a dispatch.
        self._state.update(groups, Self.Agg.InArray(column.to_data()))
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        if not self._scatters:
            # One implicit slot, and an input that produced no morsel at all
            # never grew it. `count_distinct` of nothing is one 0 and `min` of
            # nothing is one NULL — both one row, which is what the stage above
            # builds its output batch from.
            self._state.reserve(1)
        return Datum(self._state.finish())
