# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

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

**Sources are pulled, by the engine.** A plan lowers into phases
(`marrow.engine`), each a source and one `Pipeline` per worker; the driver reads
the source's morsels, `feed`s each through a worker's pipeline, and drains the
pipeline once the source is spent. A scan operator answers from `drain`, so
`OperatorSource` reads it a batch at a time.

`Pipeline` is one worker's operator chain rather than a single erased
operator. That is what lets `Relation.lower` stay compositional without a
`children()` walk over the plan: a source relation creates the phase, and every
relation above it appends one stage to each worker's pipeline.
"""

from std.collections import Set
from std.memory import ArcPointer
from std.time import perf_counter_ns

from ..errors import DynError, InternalError
from ..arrays import BoolArray, DynArray, Int32Array, StructArray
from ..buffers import Bitmap
from ..scalars import ArrowScalar, DynScalar, NullScalar
from std.builtin.rebind import downcast
from std.utils import Variant
from ..builders import DynBuilder, Int32Builder, StructBuilder, arange, nulls
from ..dtypes import Field, Int32Type, field, null, struct_
from ..kernels.aggregate import AggKernel, Windowable
from ..kernels.conditional import CaseWhenKernel
from ..execution import PARALLEL_ENABLED, ExecContext
from ..kernels.concat import concat
from ..kernels.filter import FilterKernel, TakeKernel
from ..kernels.groupby import Groups
from ..kernels.dictionary import DictionaryEncoder
from ..dtypes import DynType
from ..formats.parquet.reader import LeafSet, ParquetFile, RowSelection
from ..io import ByteSource, DynSource, StorageOptions
from ..formats import Format
from ..iceberg.catalog import IcebergTable
from ..iceberg.manifest import DataFile
from ..iceberg.planning import scan_tasks, snapshot_files
from ..iceberg.projection import NameMapping
from ..iceberg.scan import FileScanTask, IcebergFileReader
from ..kernels.join import (
    BUILD_LEFT,
    BuildMatches,
    HashJoin,
    JOIN_ALL,
    JOIN_ANTI,
    JOIN_SEMI,
    JoinBuildSide,
    JoinKind,
)
from ..utils import KeyHash
from ..utils.threads import Thread
from .logical import Bindings
from .estimates import Cost, Estimate
from .logical import (
    DynValue,
    Filter,
    JoinChain,
    JoinPricing,
    JoinRules,
    JoinClasses,
    JoinFilter,
    JoinLink,
    JoinRef,
    Multiplicity,
    Value,
    WindowSpec,
)
from .index import Index, page_selections
from ..kernels.sort import SortIndices
from ..kernels.window import WindowExtents, WindowFrame, WindowKernel
from ..schema import Schema, schema
from ..tabular import RecordBatch, Table


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
    var _to_array: def(DynScalar, Int) thin raises -> DynArray
    """How the scalar case broadcasts, fixed where its type was still known.

    A typed scalar points at its own `to_array`, so a binary links the
    broadcast of exactly the scalars it builds; only an already-erased scalar
    reaches `DynScalar.to_array`, which covers every dtype. Never called for
    an array."""

    @implicit
    def __init__(out self, var value: DynArray):
        self._v = Variant[DynScalar, DynArray](value^)
        self._to_array = Self._broadcast[NullScalar]

    @implicit
    def __init__(out self, var value: DynScalar):
        self._v = Variant[DynScalar, DynArray](value^)
        self._to_array = DynScalar.to_array

    @implicit
    def __init__[S: ArrowScalar](out self, var value: S):
        self._v = Variant[DynScalar, DynArray](DynScalar(value^))
        self._to_array = Self._broadcast[S]

    @staticmethod
    def _broadcast[
        S: ArrowScalar
    ](scalar: DynScalar, rows: Int) raises -> DynArray:
        return scalar.as_type[S]().to_array(rows)

    def is_scalar(self) -> Bool:
        """Whether this is still one value.

        `Shape` is the *static* answer to the same question and is what the
        planner actually reads; this is the one that survives erasure, and its
        only caller is `struct_array`'s own guard below. Kept public rather
        than folded into that guard because the distinction is part of what a
        `Datum` *is* — a caller holding one has no other way to ask.
        """
        return self._v.isa[DynScalar]()

    def length(self, scalar: Int) -> Int:
        """Its rows: an array's length, or `scalar` for one value."""
        if self._v.isa[DynScalar]():
            return scalar
        return len(self._v[DynArray])

    def struct_array(self) raises -> StructArray:
        """The batch a relational stage produced.

        Relational stages put their struct array here and this takes it back
        out; both directions are refcount bumps. It raises if a scalar-shaped
        datum reaches a relational position — that is a bug in the plan, not a
        shape worth handling.
        """
        if self.is_scalar():
            raise InternalError("struct_array: expected a batch, got a scalar")
        return self._v[DynArray].as_struct().copy()

    def to_array(self, n: Int) raises -> DynArray:
        """This value as a column of length `n`, broadcasting a scalar.

        The single place laziness ends. `n` is why this cannot be an implicit
        conversion: a scalar does not know how many rows it is about to become,
        and only the caller holding the batch does.

        **`n` binds the array case too.** It used to be read
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
            return self._to_array(self._v[DynScalar], n)
        ref arr = self._v[DynArray]
        if len(arr) != n:
            raise InternalError(
                t"to_array: expected a column of {n} rows, got {len(arr)}"
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
        source drives, so nothing downstream can stop it. The driver stops
        reading a phase's source as soon as any stage answers True.

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
@fieldwise_init
struct StageStats(Copyable, ImplicitlyCopyable, Movable):
    """What one stage of one worker's pipeline did, while it was observed."""

    var rows_in: Int
    var rows_out: Int
    var batches_in: Int
    var batches_out: Int
    var ns: Int
    """Time in the stage's own `push` and `drain`, the stages above it
    excluded."""

    @staticmethod
    def zero() -> Self:
        return Self(0, 0, 0, 0, 0)


comptime Observation = ArcPointer[List[StageStats]]
"""A pipeline's per-stage counts, shared with whoever asked to observe it."""


@fieldwise_init
struct TraceEvent(Copyable, Movable):
    """One slice of a timeline: what ran, on which thread, from when and for
    how long, over how many rows (-1 when none)."""

    var name: String
    var thread: Int
    """The OS thread it ran on (`Thread.current_id`)."""
    var start: Int
    """`perf_counter_ns` when it began."""
    var ns: Int
    var rows: Int


comptime Trail = ArcPointer[List[TraceEvent]]
"""One thread's timeline slices, appended as it runs: a pipeline's, or a
source's, which keeps one slot per morsel."""


struct ObservedOperator(Operator):
    """A stage of an observed pipeline: the operator it wraps, and the
    stage's entry of the pipeline's counts, updated on every call. One worker
    owns a pipeline, so nothing else writes them while it runs."""

    var _op: DynOperator
    var _stats: Observation
    var _stage: Int
    var _trail: Optional[Trail]
    """Where each call goes as a timeline slice, when traced."""
    var _label: String

    def __init__(
        out self,
        var op: DynOperator,
        stats: Observation,
        stage: Int,
        trail: Optional[Trail],
        var label: String,
    ):
        self._op = op^
        self._stats = stats
        self._stage = stage
        self._trail = trail
        self._label = label^

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        var since = Int(perf_counter_ns())
        var produced = self._op.push(morsel)
        var ns = Int(perf_counter_ns()) - since
        ref s = self._stats[][self._stage]
        s.rows_in += len(morsel.batch)
        s.batches_in += 1
        s.ns += ns
        if produced:
            s.rows_out += produced.value().length(len(morsel.batch))
            s.batches_out += 1
        if self._trail:
            self._trail.value()[].append(
                TraceEvent(
                    self._label,
                    Thread.current_id(),
                    since,
                    ns,
                    len(morsel.batch),
                )
            )
        return produced^

    def drain(mut self) raises -> Optional[Datum]:
        var since = Int(perf_counter_ns())
        var produced = self._op.drain()
        var ns = Int(perf_counter_ns()) - since
        ref s = self._stats[][self._stage]
        s.ns += ns
        if produced:
            s.rows_out += produced.value().length(1)
            s.batches_out += 1
        if self._trail:
            self._trail.value()[].append(
                TraceEvent(
                    self._label + " drain", Thread.current_id(), since, ns, -1
                )
            )
        return produced^

    def done(self) -> Bool:
        return self._op.done()


struct Pipeline(Movable):
    """One worker's chain of operators.

    Not a second abstraction beside `Operator` — just a `List` of them, so
    `Relation.lower` stays compositional: a source relation creates a phase of
    empty pipelines, and `Filter`, `Project`, `Sort`, `Limit` and `Aggregate`
    each `append` a stage to every worker's. Nothing has to walk the plan,
    which matters because `DynRelation` deliberately exposes no `children()`.

    It was called `DynProcessor`, which was wrong twice over: it erases
    nothing, and "processor" was a second word for what `Operator` already
    names.

    Appending keeps a plan's stages in one flat list: composing through
    `DynOperator` would nest one pipeline per stage and charge every batch an
    extra trampoline per level.
    """

    var _ops: List[DynOperator]
    var _stage: Int
    """How far `drain` has walked. A composite operator has to remember its own
    position, because `drain` answers one batch per call."""

    var _pending: List[StructArray]

    def __init__(out self):
        """No stages: a worker's pipeline, fed by its phase's source."""
        self._ops = List[DynOperator]()
        self._stage = 0
        self._pending = List[StructArray]()

    def append(mut self, var op: DynOperator):
        self._ops.append(op^)

    def observe(
        mut self,
        trail: Optional[Trail] = None,
        labels: List[String] = List[String](),
    ) -> Observation:
        """Count what every stage does from now on, and with a `trail` record
        each call as a timeline slice named by `labels`: each stage is wrapped
        in an `ObservedOperator`, so a pipeline nobody observes runs as it
        is."""
        var stats = Observation(
            List[StageStats](length=len(self._ops), fill=StageStats.zero())
        )
        var ops = List[DynOperator](capacity=len(self._ops))
        var stage = 0
        while len(self._ops) > 0:
            var label = labels[stage].copy() if stage < len(labels) else String(
                "stage ", stage
            )
            ops.append(
                ObservedOperator(self._ops.pop(0), stats, stage, trail, label^)
            )
            stage += 1
        self._ops = ops^
        return stats^

    @no_inline
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

    def feed(
        mut self, var batch: StructArray, mut out: List[StructArray]
    ) raises:
        """Push `batch` through every stage, and append to `out` what comes
        out of the last one. Nothing is drained: a blocking stage keeps what
        it absorbed until `drain`."""
        self._flow(Morsel.ungrouped(batch^), 0, out)

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
        yields through the stages above: when stage `i` finally yields, no
        later stage has seen it, so it flows through `i+1..` before `i+1` is
        drained — a projection over an aggregate returns nothing without this.
        Resumable, because an `Operator` answers one batch at a time; `_stage`
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


@no_inline
def finish_batches(
    batches: List[StructArray], schema: Schema, ctx: ExecContext
) raises -> StructArray:
    """A plan's batches as its one result of `schema`, concatenated on
    `ctx`.

    An empty result is still a well-formed batch: one zero-length column per
    field, so anything walking columns by schema index stays in bounds.
    """
    var result = _concat_batches(batches, schema, ctx)
    # Several batches were concatenated, and that compacts sparse view columns
    # already; a single one is returned as it is, so its view columns are
    # compacted here -- a result must not keep the pages it was filtered out
    # of alive. An explicit ladder rather than `dispatch_binaryview` with a
    # closure: every plan collects, and the closure form measured +13.8 KB of
    # `__text` on each fused size gate.
    for i in range(len(result.children)):
        ref child = result.children[i]
        if child.dtype().is_string_view():
            result.children[i] = child.as_string_view().compact().to_dyn()
        elif child.dtype().is_binary_view():
            result.children[i] = child.as_binary_view().compact().to_dyn()
    return result^


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
            FilterKernel.dispatch(
                batch.copy().to_dyn(), mask.as_bool(), self._ctx
            )
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


struct SelectOperator(Operator):
    """Picks columns by position and names them by `schema`: a join chain's
    output when a filter lands on its root, and so runs after the root's
    join rather than inside it, and a union's later inputs, relabelled to
    the first's names. Positional because two inputs may share a name;
    zero-copy, and `StructArray.field` applies a sliced batch's offset.
    """

    var _indices: List[Int]
    var _schema: Schema

    def __init__(out self, var indices: List[Int], var schema: Schema):
        self._indices = indices^
        self._schema = schema^

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        return Datum(
            morsel.batch.select(self._indices)
            .with_fields(self._schema.fields)
            .to_dyn()
        )

    def drain(mut self) raises -> Optional[Datum]:
        return None


struct RenamedPredicate(Operator):
    """A predicate evaluated over a view of its input: the columns at
    `indices`, named by `schema`.

    What lets a join chain evaluate a residual filter between two joins. A
    predicate binds its columns by name when it runs, and the columns of a
    chain's intermediate batch are only known by position, so the filter sees
    exactly the columns it reads under exactly the names it was written
    against. Hand the result to a `FilterOperator`, which filters the whole
    batch by the mask this answers.
    """

    var _predicate: DynOperator
    var _indices: List[Int]
    var _schema: Schema

    def __init__(
        out self,
        var predicate: DynOperator,
        var indices: List[Int],
        var schema: Schema,
    ):
        self._predicate = predicate^
        self._indices = indices^
        self._schema = schema^

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        var view = morsel.batch.select(self._indices).with_fields(
            self._schema.fields
        )
        return self._predicate.push(Morsel.ungrouped(view^))

    def drain(mut self) raises -> Optional[Datum]:
        return None


# Appends rather than returning a `DynOperator`: this type is a field of
# `Aggregate`, so it is spelled into the symbol name of everything instantiated
# over `DynRelation`, and a `DynOperator`'s spelling exceeds `ld`'s symbol
# length limit. A `Pipeline` spells as a pointer and two integers.
comptime AggregateLowering = def(
    mut Pipeline,
    var List[DynOperator],
    var List[DynOperator],
    var Schema,
    var ExecContext,
    Bool,
) thin
"""Appends an `Aggregate`'s grouping stage, over its lowered keys and folds, to
a pipeline — `GroupedAggregateOperator.append_to`, or
`UngroupedAggregateOperator.append_to` when there are no keys. The flag says
the folds answer partial states, which the stage emits as they are."""


@fieldwise_init
struct FoldMode(Copyable, ImplicitlyCopyable, Movable):
    """What an aggregate's operator does with the morsels it is pushed, and
    answers at `drain`.

    `final()` folds rows and answers the aggregate. `partial()` folds rows and
    answers the state so far, as `AggKernel.partials` columns in one struct
    column, so the states of the workers that split an input can be combined.
    `combine(column)` is pushed those states — the struct column at
    `column` — and answers the aggregate of everything they folded."""

    var _kind: UInt8
    var column: Int
    """Where a combining operator finds the partial states it is pushed."""

    @staticmethod
    def final() -> Self:
        return Self(0, -1)

    @staticmethod
    def partial() -> Self:
        return Self(1, -1)

    @staticmethod
    def combine(column: Int) -> Self:
        return Self(2, column)

    def is_partial(self) -> Bool:
        return self._kind == 1

    def is_combine(self) -> Bool:
        return self._kind == 2


@no_inline
def _partial_struct(
    var columns: List[DynArray], names: List[String]
) raises -> StructArray:
    """`columns` as one struct column named by `names`, typed by the columns
    themselves: partial states have no declared schema."""
    var fields = List[Field](capacity=len(columns))
    for i in range(len(columns)):
        fields.append(field(names[i], columns[i].dtype()))
    return StructArray.from_arrays(columns^, fields)


def emit_fold[
    Agg: AggKernel
](mut state: Agg, slots: Int, partial: Bool) raises -> Datum:
    """Seed `slots` slots, then answer the aggregate, or with `partial` the
    state as one struct column — the tail every fold operator shares.

    `reserve` is what makes an aggregate over **zero** morsels answer at all:
    a keyless query still owes exactly one row. `reserve(0)` is a no-op in
    every conformer, so a grouped query that saw no rows answers no rows."""
    state.reserve(slots)
    comptime if PARALLEL_ENABLED:
        if partial:
            var columns = state.partials()
            var names = List[String](capacity=len(columns))
            for i in range(len(columns)):
                names.append(String("p", i))
            return Datum(_partial_struct(columns^, names).to_dyn())
    return Datum(state.finish().to_dyn())


struct CombineAggregateOperator[Agg: AggKernel](Operator):
    """The aggregate of partial states: each pushed morsel carries, at
    `column`, the struct of `AggKernel.partials` columns some workers'
    operators answered, one row per group they saw, and `drain` answers the
    aggregate of everything those workers folded."""

    var _state: Self.Agg
    var _column: Int
    var _scatters: Bool
    """Whether the query has keys: a keyless one owes one row of nothing."""
    var _emitted: Bool

    def __init__(
        out self, column: Int, scatters: Bool, in_dtype: DynType
    ) raises:
        self._state = Self.Agg(in_dtype)
        self._column = column
        self._scatters = scatters
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        # Ungrouped, the stage forwards the one-slot assignment as it is.
        var states = morsel.batch.field(self._column).as_struct().flatten()
        self._state.combine(morsel.groups, states)
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        return emit_fold(self._state, 0 if self._scatters else 1, False)


struct ColumnOperator(Operator):
    """A batch's column at a fixed position."""

    var _index: Int

    def __init__(out self, index: Int):
        self._index = index

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        return Datum(morsel.batch.field(self._index))

    def drain(mut self) raises -> Optional[Datum]:
        return None


comptime _PASS_MIN_ROWS = 4096
"""Rows a worker's partial table places before it may judge them too
distinct to group."""


struct GroupedAggregateOperator(Operator):
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
    stage keeps only the encoder's key values, which grow with the number of
    *groups* too. That was not always so: a `count_distinct` used to hold its
    whole input column, and the docstring here recorded it as the price of a
    non-scalar accumulator long after `DistinctCount` became incremental.

    Group ids are dense and stable across morsels, which is what makes all of
    them work: a state that has already folded batch N keeps its slots when
    batch N+1 introduces new groups (`AggState.reserve` extends them in place),
    and concatenating N morsels' id arrays end to end is a valid assignment
    over the concatenated input rather than N unrelated numberings.

    A fold that answers `None` from `drain` has its slot filled with a null
    column typed from this stage's output schema — see `_emit_folds`.

    `HAVING` needs no node of its own: a `FilterOperator` above this stage sees
    the aggregate's *output* batch, which is exactly what the flush cascade in
    `Pipeline.drain` delivers.

    **There is always a key.** A query without one is a single implicit group
    and lowers to `UngroupedAggregateOperator`, which has no encoder to build.
    """

    var _keys: List[DynOperator]
    var _folds: List[DynOperator]
    var _schema: Schema
    var _encoder: DictionaryEncoder[KeyHash]
    """Group keys to group ids: a group id is the code of its key."""
    var _partial: Bool
    """Whether the folds answer partial states."""
    var _may_pass: Bool
    """Whether rows may be handed on unencoded: a partial stage, and no key
    dictionary-encoded, so the rows' keys concatenate with the encoder's."""
    var _passing: Bool
    """Whether this stage hands its rows on unencoded, each its own slot,
    having seen its table take mostly new keys."""
    var _rows_seen: Int
    """Rows the encoder has placed."""
    var _passed: Int
    """Rows handed on, each a slot after the encoder's codes; the encoder
    keeps their keys, unindexed, after its codes' keys."""
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynOperator],
        var folds: List[DynOperator],
        var schema: Schema,
        var ctx: ExecContext,
        partial: Bool = False,
    ):
        # The output schema is keys then aggregates, so the key types are its
        # first `len(keys)` fields.
        var key_types = List[DynType](capacity=len(keys))
        var may_pass = partial
        for i in range(len(keys)):
            key_types.append(schema.fields[i].dtype.copy())
            may_pass = may_pass and not key_types[i].is_dictionary()
        self._keys = keys^
        self._folds = folds^
        self._schema = schema^
        # The encoder gets the caller's context: it is what selects radix
        # placement over the single-table path, and what stripes the key
        # hashing.
        self._encoder = DictionaryEncoder[KeyHash](key_types^, ctx^)
        self._partial = partial
        self._may_pass = may_pass
        self._passing = False
        self._rows_seen = 0
        self._passed = 0
        self._emitted = False

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
        """
        ref batch = morsel.batch
        var keys = self._key_columns(morsel)
        var groups: Groups
        comptime if PARALLEL_ENABLED:
            if self._passing:
                groups = self._hand_on(keys^, len(batch))
            else:
                groups = self._place(keys, len(batch))
        else:
            groups = self._place(keys, len(batch))
        var forwarded = Morsel(batch.copy(), groups^)
        # Indexed rather than `for ref`: a fold is move-only, and iterating a
        # `List` by reference requires `Copyable`.
        for i in range(len(self._folds)):
            _ = self._folds[i].push(forwarded)
        return None

    def _place(mut self, keys: List[DynArray], n: Int) raises -> Groups:
        """The slots of `n` rows by their `keys`' codes. A partial stage
        whose keys have almost all been new, over enough rows to judge it
        by, reduces little, and the combine groups the same keys again: it
        hands its rows on as they are from then on."""
        var placed = self._encoder.encode(keys)
        comptime if PARALLEL_ENABLED:
            self._rows_seen += n
            if (
                self._may_pass
                and self._rows_seen >= _PASS_MIN_ROWS
                and 8 * len(self._encoder) > 7 * self._rows_seen
            ):
                self._passing = True
        return Groups(placed.ids.copy(), len(self._encoder))

    def _hand_on(mut self, var keys: List[DynArray], n: Int) raises -> Groups:
        """Each of `n` rows its own slot, after every slot so far, its keys
        kept after the codes' keys: the combine downstream groups them."""
        var first = len(self._encoder) + self._passed
        self._passed += n
        self._encoder.extend_unindexed(keys^)
        return Groups(arange[Int32Type](first, first + n), first + n)

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        return _emit_folds(
            self._folds,
            self._encoder.values(),
            self._schema,
            len(self._encoder) + self._passed,
            self._partial,
        )

    @staticmethod
    def append_to(
        mut pipe: Pipeline,
        var keys: List[DynOperator],
        var folds: List[DynOperator],
        var schema: Schema,
        var ctx: ExecContext,
        partial: Bool,
    ):
        """Append this operator over `keys` and `folds` to `pipe` — the
        `AggregateLowering` a plan's `Aggregate` points at, so the plan names
        its encoder by the operator it builds."""
        pipe.append(Self(keys^, folds^, schema^, ctx^, partial))


struct UngroupedAggregateOperator(Operator):
    """Blocking: fold every pushed morsel into one implicit group, then emit
    its one row — `SELECT sum(x) FROM t`, the aggregate without a key.

    Separate from `GroupedAggregateOperator` because it has nothing to place:
    the morsel already carries the one-slot assignment (`Morsel.ungrouped`), so
    it goes to every fold as it is, and there is no encoder to build.

    One row even over an input that yields nothing — `sum` of nothing is one
    NULL, not no rows.
    """

    var _folds: List[DynOperator]
    var _schema: Schema
    var _partial: Bool
    """Whether the folds answer partial states."""
    var _emitted: Bool

    def __init__(
        out self,
        var folds: List[DynOperator],
        var schema: Schema,
        partial: Bool = False,
    ):
        self._folds = folds^
        self._schema = schema^
        self._partial = partial
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        # Indexed rather than `for ref`: a fold is move-only, and iterating a
        # `List` by reference requires `Copyable`.
        for i in range(len(self._folds)):
            _ = self._folds[i].push(morsel)
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        return _emit_folds(
            self._folds, List[DynArray](), self._schema, 1, self._partial
        )

    @staticmethod
    def append_to(
        mut pipe: Pipeline,
        var keys: List[DynOperator],
        var folds: List[DynOperator],
        var schema: Schema,
        var ctx: ExecContext,
        partial: Bool,
    ):
        """Append this operator over `folds` to `pipe` — the
        `AggregateLowering` of an `Aggregate` without keys, so `keys` is
        empty."""
        debug_assert(len(keys) == 0, "an ungrouped aggregate has no keys")
        pipe.append(Self(folds^, schema^, partial))


def _emit_folds(
    mut folds: List[DynOperator],
    var columns: List[DynArray],
    schema: Schema,
    rows: Int,
    partial: Bool,
) raises -> Datum:
    """The aggregate stage's one batch: the key `columns`, then every fold's
    answer, `rows` long.

    A fold answering `None` gets a null column typed from `schema`, the only
    place that type is still known. The output schema is keys then
    aggregates, so aggregate `i` is field `len(columns) + i`; appending
    nothing instead would hand `_struct_of` fewer children than its dtype has
    fields — a batch that runs and mis-indexes. No aggregate operator answers
    `None` today, so the branch guards a future `Operator` in this position.
    """
    var num_keys = len(columns)
    # Indexed rather than `for ref`: a fold is move-only.
    for i in range(len(folds)):
        var col = folds[i].drain()
        if col:
            columns.append(col.value().to_array(rows))
        else:
            columns.append(nulls(rows, schema.fields[num_keys + i].dtype))
    comptime if PARALLEL_ENABLED:
        if partial:
            # Partial states are typed by what the folds answered.
            return Datum(_partial_struct(columns^, schema.names()).to_dyn())
    return Datum(_struct_of(schema, columns^, rows).to_dyn())


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
    the single-column `SortIndices` kernel could not express anyway.

    `per_morsel` sorts each morsel on its own, as it is pushed, and answers its
    first `limit` rows: the cut a top-N takes of every morsel before the rows
    that survive are sorted again together.
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
    var _per_morsel: Bool
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynOperator],
        var ascending: List[Bool],
        nulls_first: Bool,
        limit: Optional[Int],
        var ctx: ExecContext,
        *,
        per_morsel: Bool = False,
    ):
        self._keys = keys^
        self._ascending = ascending^
        self._nulls_first = nulls_first
        self._limit = limit
        self._batches = List[StructArray]()
        self._ctx = ctx^
        self._per_morsel = per_morsel
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        if self._per_morsel:
            return self._sorted(morsel.batch.copy())
        self._batches.append(morsel.batch.copy())
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted or len(self._batches) == 0:
            return None
        self._emitted = True

        var schema = Schema.from_dtype(self._batches[0].dtype)
        return self._sorted(_concat_batches(self._batches, schema, self._ctx))

    def _sorted(mut self, var whole: StructArray) raises -> Datum:
        """`whole` in key order, cut to `limit` rows when there is one."""
        var order = Optional[Int32Array](None)
        for k in range(len(self._keys) - 1, -1, -1):
            var key = (
                self._keys[k]
                .push(Morsel.ungrouped(whole.copy()))
                .value()
                .to_array(len(whole))
            )
            if order:
                key = TakeKernel.dispatch(key, order.value(), self._ctx)
            # **The bound applies to the primary key only.** The multi-key
            # decomposition sorts stably from the least significant key to the
            # most, composing each pass onto the previous permutation, so every
            # pass but the last must return a *full* permutation for the next
            # one to permute. Truncating an earlier pass discards rows the
            # later keys still have to order, which loses answers rather than
            # reordering them. `k == 0` is the final, most significant pass.
            var pass_order = SortIndices[nan_largest=True].dispatch(
                key,
                ascending=self._ascending[k],
                nulls_first=self._nulls_first,
                limit=self._limit if k == 0 else None,
                ctx=self._ctx,
            )
            if order:
                # Compose: this pass permutes the previous order, it does not
                # replace it. Skipping this is the classic multi-key sort bug —
                # the last key wins and every earlier one is discarded.
                var prev: DynArray = order.value().copy()
                order = (
                    TakeKernel.dispatch(prev, pass_order, self._ctx)
                    .as_int32()
                    .copy()
                )
            else:
                order = pass_order^

        if not order:
            return Datum(whole^.to_dyn())
        return Datum(
            TakeKernel.dispatch(whole.copy().to_dyn(), order.value(), self._ctx)
        )


struct WindowOperator(Operator):
    """`OVER (...)` — blocking, because a window function needs its partition.

    Buffers every morsel and answers once at `drain`, the same shape as
    `SortOperator` and for the same reason: no prefix of the input determines
    any output row, since the row that ends up adjacent may not have arrived.

    **The partitioning is a prefix of the sort key, not a second mechanism.**
    `PARTITION BY k ORDER BY v` is the ordering `[k, v]`, so one
    `SortIndices.multi` answers both questions, and `WindowExtents` reads
    partition and peer boundaries straight off the result. Hash-partitioning
    first and sorting each bucket would need a `DictionaryEncoder`, a gather per
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

    var _values: List[DynValue]
    """Window values; those sharing a spec share one sort."""

    var _input_schema: Schema
    var _output_schema: Schema
    var _bindings: Bindings
    var _ctx: ExecContext
    var _batches: List[StructArray]
    var _emitted: Bool

    def __init__(
        out self,
        var values: List[DynValue],
        var input_schema: Schema,
        var output_schema: Schema,
        var bindings: Bindings,
        var ctx: ExecContext,
    ):
        self._values = values^
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

        var whole = StructBuilder.concat(self._batches)
        var n = len(whole)

        var columns = List[DynArray]()
        for i in range(len(self._input_schema.fields)):
            columns.append(whole.field(i))
        # One sort per distinct spec, in first-seen order; `order[k]` is the
        # value `computed[k]` answers.
        var computed = List[DynArray](capacity=len(self._values))
        var order = List[Int](capacity=len(self._values))
        for i in range(len(self._values)):
            var seen = False
            for ref j in order:
                if j == i:
                    seen = True
            if not seen:
                var spec = self._values[i].window_spec()
                var same = List[Int]()
                for j in range(i, len(self._values)):
                    if self._values[j].window_spec() == spec:
                        same.append(j)
                computed.extend(self._compute(spec, same, whole))
                order.extend(same^)
        for i in range(len(self._values)):
            for k in range(len(order)):
                if order[k] == i:
                    columns.append(computed[k].copy())
        return Datum(_struct_of(self._output_schema, columns^, n).to_dyn())

    def _compute(
        self, spec: WindowSpec, which: List[Int], whole: StructArray
    ) raises -> List[DynArray]:
        """The columns of the values at `which`, all over `spec`, from one
        sort — each back in input order."""
        var n = len(whole)
        # -- one ordering answers both questions ----------------------------
        var keys = List[DynArray]()
        var ascending = List[Bool]()
        for ref k in spec.partition_by:
            keys.append(self._eval(k, whole))
            # Direction is irrelevant to a partition: it groups equal rows,
            # and every order puts equal rows together.
            ascending.append(True)
        for i in range(len(spec.order_by)):
            keys.append(self._eval(spec.order_by[i], whole))
            ascending.append(spec.ascending[i])
        var perm = self._permutation(keys, ascending, spec.nulls_first, n)

        var sorted_keys = List[DynArray](capacity=len(keys))
        for ref k in keys:
            sorted_keys.append(TakeKernel.dispatch(k.copy(), perm, self._ctx))
        var extents = WindowExtents.of_sorted(
            sorted_keys, len(spec.partition_by), self._ctx
        )

        # -- the sorted batch, and the way back ------------------------------
        var sorted_columns = List[DynArray]()
        for i in range(len(self._input_schema.fields)):
            sorted_columns.append(
                TakeKernel.dispatch(whole.field(i), perm, self._ctx)
            )
        var sorted_batch = _struct_of(self._input_schema, sorted_columns^, n)
        var inverse = self._inverse(perm, n)

        var out = List[DynArray](capacity=len(which))
        for ref i in which:
            var column = self._values[i].window_column(
                sorted_batch,
                extents,
                self._input_schema,
                self._bindings,
                self._ctx,
            )
            out.append(TakeKernel.dispatch(column^, inverse, self._ctx))
        return out^

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
        return SortIndices[nan_largest=True].multi(
            key_batch,
            indices,
            ascending,
            nulls_first=nulls_first,
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


struct JoinBuild(Movable):
    """One equijoin's build side, indexed once, and what is needed to emit
    its rows: shared by the `JoinOperator` of every worker that probes it, and
    by the phase that emits the build rows the probes owe.

    **The build side runs to completion before the first probe.** A hash join
    is a pipeline breaker on one input and streaming on the other: lowering
    runs the build side through the driver and indexes it here, and probe
    morsels stream through the workers' operators afterwards.

    **Every kind streams on the probe side.** Each morsel is probed as it
    arrives, and the build rows that LEFT, FULL, SEMI and ANTI owe for the
    probe side as a whole — the unmatched ones, or SEMI's matched ones — are
    emitted once, by `unmatched`, from the build keys every probing operator
    marked. So the probe side is never held.

    Every field is in build/probe terms except `_schema`, the column order
    the kernel's output is relabelled to, and `_build_side`, which says which
    logical side was built.

    It emits the positions `_select` of `_schema`, named by `_output` — by
    position, since two inputs of a join chain may share a name.
    """

    var _index: HashJoin[KeyHash]
    var _probe_keys: List[Int]
    """Key column indices into the **probe** input's own schema."""
    var _kind: JoinKind
    var _strictness: UInt8
    var _build_side: JoinBuildSide
    """Which logical side was built; the kernel uses it to emit the columns
    in `_schema`'s order."""
    var _schema: Schema
    """What the join promises its consumers: left fields then right, or one
    side alone for the kinds that emit a single side. Independent of
    `_build_side`."""
    var _probe_schema: Schema
    """The **probe** input's own fields: the types of the all-null probe
    columns the unmatched build rows are padded with. Not the output schema,
    which for `SEMI`/`ANTI` is one side's alone."""
    var _select: List[Int]
    """The positions in `_schema` the join emits, as `_output`. An empty
    list emits no column, every row."""
    var _output: Schema
    var _marks: Bool
    """Whether the build rows emitted depend on the matches: LEFT, FULL,
    SEMI and ANTI, read against `(build, probe)`."""
    var _marks_only: Bool
    """Whether a probe morsel contributes nothing but marks: SEMI and ANTI."""
    var _matches: List[ArcPointer[BuildMatches]]
    """The build keys each probing operator has matched."""

    def __init__(
        out self,
        var build: List[StructArray],
        build_keys: List[Int],
        var probe_keys: List[Int],
        kind: JoinKind,
        strictness: UInt8,
        build_side: JoinBuildSide,
        var schema: Schema,
        var build_schema: Schema,
        var probe_schema: Schema,
        var select: List[Int],
        var output: Schema,
        ctx: ExecContext,
    ) raises:
        # `build_schema` is the build input's own: read only when the side
        # produced no batch, to type the empty side.
        var side = _concat_batches(build, build_schema, ctx)
        self._index = HashJoin[KeyHash](side, build_keys, ctx.copy())
        self._probe_keys = probe_keys^
        self._kind = kind
        self._strictness = strictness
        self._build_side = build_side
        self._schema = schema^
        self._probe_schema = probe_schema^
        self._select = select^
        self._output = output^
        var physical = build_side.physical(kind)
        self._marks = physical.needs_left_matches()
        self._marks_only = physical == JOIN_SEMI or physical == JOIN_ANTI
        self._matches = List[ArcPointer[BuildMatches]]()

    def owes_rows(self) -> Bool:
        """Whether build rows are emitted after the probe side, by whether
        they matched."""
        return self._marks

    def probe_matches(mut self) -> ArcPointer[BuildMatches]:
        """The marks of one more probing operator: its own, so no two threads
        write one, and kept here for `unmatched`."""
        var matches = ArcPointer(
            BuildMatches(self._index.num_codes() if self._marks else 0)
        )
        if self._marks:
            self._matches.append(matches)
        return matches^

    def probe(
        self, batch: StructArray, mut matches: BuildMatches, ctx: ExecContext
    ) raises -> Optional[StructArray]:
        """The joined rows of one probe morsel, marking `matches`; nothing
        when the morsel only marks or none of its rows join."""
        if self._marks_only:
            self._index.mark(batch, self._probe_keys, matches, ctx)
            return None
        var result = self._index.probe(
            batch,
            self._probe_keys,
            self._kind,
            self._strictness,
            self._build_side,
            matches,
            ctx,
        )
        if len(result) == 0:
            return None
        return self._relabel(result)

    def unmatched(self, ctx: ExecContext) raises -> StructArray:
        """The build rows the probe side owes, by what every probing
        operator marked: read once all of them have run."""
        var merged = BuildMatches(self._index.num_codes())
        for ref m in self._matches:
            merged.merge(m[])
        var tail = self._index.unmatched(
            merged,
            RecordBatch.empty(self._probe_schema).to_struct_array(),
            self._kind,
            self._strictness,
            self._build_side,
            0,
            self._index.num_build_rows(),
            ctx,
        )
        return self._relabel(tail)

    def _relabel(self, result: StructArray) raises -> StructArray:
        # Re-typed with the *declared* schema: the kernel names its output from
        # the arrays it joined, while `Join.schema()` is what the plan promised
        # its consumers. Handing back the kernel's dtype makes
        # `plan.schema() != plan.execute().schema`.
        #
        # The relabel is positional, so first check that the kernel produced
        # the declared columns in the declared order. Only arity and dtypes are
        # compared: names differ by design (the kernel suffixes a collision
        # `_right`), and an array's struct dtype may widen a required column to
        # nullable.
        ref want = self._schema.fields
        ref got = result.dtype.as_struct().fields
        var matches = len(want) == len(got)
        if matches:
            for i in range(len(want)):
                if want[i].dtype != got[i].dtype:
                    matches = False
                    break
        if not matches:
            raise InternalError(
                t"join: the kernel produced {result.dtype} for a plan that "
                t"declared {len(want)} columns in a different shape"
            )
        return result.select(self._select).with_fields(self._output.fields)


struct JoinOperator(Operator):
    """Equijoin: probes each morsel against a `JoinBuild`, and marks the
    build keys its rows matched in marks of its own."""

    var _build: ArcPointer[JoinBuild]
    var _matches: ArcPointer[BuildMatches]
    var _ctx: ExecContext

    def __init__(
        out self,
        var build: ArcPointer[JoinBuild],
        var matches: ArcPointer[BuildMatches],
        var ctx: ExecContext,
    ):
        self._build = build^
        self._matches = matches^
        self._ctx = ctx^

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        var result = self._build[].probe(
            morsel.batch, self._matches[], self._ctx
        )
        if result:
            return Datum(result.value().copy().to_dyn())
        return None

    def drain(mut self) raises -> Optional[Datum]:
        return None


struct MultisetOperator[M: Multiplicity](Operator):
    """`Intersection` and `Difference`, with or without `ALL`.

    Blocking on both sides: whether a row survives depends on how often it
    occurs on each, and neither count is final until its side is exhausted.
    One `DictionaryEncoder` numbers the distinct rows of both sides — the
    left side's first, as they are pushed, then the right side's at `drain` —
    and `M.copies` turns each code's two counts into the copies it keeps.

    The encoder is what makes NULL equal to itself, which set relations
    require and a join key never allows.
    """

    var _other: List[StructArray]
    """The right side's rows, counted at `drain`."""
    var _all: Bool
    var _schema: Schema
    var _encoder: DictionaryEncoder[KeyHash]
    var _left: List[Int]
    """How many rows of the left side fell into each group."""
    var _right: List[Int]
    var _ctx: ExecContext
    var _emitted: Bool

    def __init__(
        out self,
        var other: List[StructArray],
        all: Bool,
        var schema: Schema,
        var ctx: ExecContext,
    ):
        var types = List[DynType](capacity=len(schema.fields))
        for ref field in schema.fields:
            types.append(field.dtype.copy())
        self._other = other^
        self._all = all
        self._schema = schema^
        self._encoder = DictionaryEncoder[KeyHash](types^, ctx.copy())
        self._left = List[Int]()
        self._right = List[Int]()
        self._ctx = ctx^
        self._emitted = False

    def _count(mut self, batch: StructArray, left: Bool) raises:
        """Encode `batch`'s rows and count each code for one side."""
        var n = len(batch)
        var placed = self._encoder.encode(batch.flatten())
        self._left.resize(len(self._encoder), 0)
        self._right.resize(len(self._encoder), 0)
        var ids = placed.ids.values()
        for i in range(n):
            var g = Int(ids[i])
            if left:
                self._left[g] += 1
            else:
                self._right[g] += 1

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        self._count(morsel.batch, True)
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        var other = self._other^
        self._other = List[StructArray]()
        for ref b in other:
            self._count(b, False)
        var which = Int32Builder(capacity=len(self._left))
        for g in range(len(self._left)):
            for _ in range(
                Self.M.copies(self._left[g], self._right[g], self._all)
            ):
                which.append(Int32(g))
        var indices = which.finish()
        if len(indices) > 0:
            var keys = self._encoder.values()
            var distinct = _struct_of(self._schema, keys^, len(self._left))
            return Datum(
                TakeKernel.dispatch(distinct^.to_dyn(), indices, self._ctx)
            )
        else:
            return None


def _struct_of(
    schema: Schema, var columns: List[DynArray], length: Int
) raises -> StructArray:
    """A struct array over `columns`, typed by `schema`.

    The execution layer works in struct arrays; this is how an operator that
    produced a fresh column list says so. `RecordBatch` appears only at the
    boundary (`finish_batches` out).
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
    return concat(batches, schema, ctx)


struct FileScanOperator[Fmt: Format](Operator):
    """Runs a `FormatScan[Fmt]`: one batch per `drain`, its files read in
    order by `Fmt`'s reader and its columns selected by the scan's schema. Each file
    is opened when the scan reaches it, not here: a `Relation` must not touch
    the filesystem to exist."""

    var _paths: List[String]
    var _current: Int
    var _schema: Schema
    var _names: List[String]
    var _options: StorageOptions
    var _reader: Optional[Self.Fmt.Reader]

    def __init__(
        out self,
        var paths: List[String],
        var schema: Schema,
        var options: StorageOptions = StorageOptions(),
    ):
        self._paths = paths^
        self._current = 0
        self._names = schema.names()
        self._schema = schema^
        self._options = options^
        self._reader = None

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        # A source consumes nothing; the driver never calls this.
        return None

    def drain(mut self) raises -> Optional[Datum]:
        while self._current < len(self._paths):
            if not self._reader:
                self._reader = Self.Fmt.open(
                    self._paths[self._current], self._schema, self._options
                )
            var batch = self._reader.value().read_next_batch()
            if batch:
                return Datum(
                    batch.value().select(self._names).to_struct_array().to_dyn()
                )
            self._reader = None
            self._current += 1
        return None


struct IcebergScanOperator(Operator):
    """Runs an `IcebergScan`: one row group of one data file per `drain`, in
    the scan's schema.

    The snapshot is planned on the first `drain`, not here — reading its
    manifests is I/O, and a `Relation` must not do any to exist. Planning
    skips every data file the pushed predicates prove cannot match, from the
    column bounds and partition values its manifest records; the predicates
    are still evaluated by the `Filter` above, so skipping saves reads and
    never changes an answer."""

    var _table: IcebergTable
    var _snapshot_id: Optional[Int]
    var _schema: Schema
    var _pushed: List[DynValue]
    var _bindings: Bindings
    var _ctx: ExecContext
    var _mapping: Optional[NameMapping]
    var _tasks: Optional[List[FileScanTask]]
    var _next: Int
    var _reader: Optional[IcebergFileReader]

    def __init__(
        out self,
        var table: IcebergTable,
        snapshot_id: Optional[Int],
        var schema: Schema,
        var ctx: ExecContext,
        var pushed: List[DynValue] = [],
        var bindings: Bindings = Bindings(),
    ):
        self._table = table^
        self._snapshot_id = snapshot_id
        self._schema = schema^
        self._pushed = pushed^
        self._bindings = bindings^
        self._ctx = ctx^
        self._mapping = None
        self._tasks = None
        self._next = 0
        self._reader = None

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        # A source consumes nothing; the driver never calls this.
        return None

    def _plan(self) raises -> List[FileScanTask]:
        if not self._snapshot_id:
            return []
        var files = snapshot_files(
            self._table, self._snapshot_id.value(), self._ctx
        )
        if len(self._pushed) == 0:
            return scan_tasks(self._table, files)
        # `Index.from_manifest` describes files of one partition spec, so the
        # files are pruned spec by spec and the verdicts put back in order.
        var keep = List[Bool](length=len(files.data), fill=False)
        var specs = List[Int]()
        for ref entry in files.data:
            if entry.spec_id not in specs:
                specs.append(entry.spec_id)
        for spec_id in specs:
            var positions = List[Int]()
            var group = List[DataFile]()
            for i in range(len(files.data)):
                if files.data[i].spec_id == spec_id:
                    positions.append(i)
                    group.append(files.data[i].data_file.copy())
            var index = Index.from_manifest(
                group, self._schema, self._table.metadata.spec(spec_id)
            )
            for chunk in index.read_plan(self._pushed, self._bindings):
                keep[positions[chunk]] = True
        return scan_tasks(self._table, files, keep^)

    def drain(mut self) raises -> Optional[Datum]:
        if not self._tasks:
            self._mapping = self._table.metadata.name_mapping()
            self._tasks = self._plan()
        while True:
            if self._reader:
                var batch = self._reader.value().next()
                if batch:
                    return Datum(batch.value().to_struct_array().to_dyn())
                self._reader = None
            ref tasks = self._tasks.value()
            if self._next >= len(tasks):
                return None
            self._reader = IcebergFileReader(
                tasks[self._next],
                self._schema,
                self._mapping,
                options=self._table.options,
                ctx=self._ctx,
            )
            self._next += 1


def admitted_bits(mask: DynArray) raises -> Bitmap[mut=False]:
    """The rows a `FILTER` predicate admits: TRUE, and neither FALSE nor NULL.

    `values & validity`, the same expression `kernels.filter.filter` applies
    to its mask. The fused machine masks the operand's validity with it and
    the buffered one compacts, and either is enough because every
    `AggKernel` already skips nulls. A function rather than inline code
    because `intersect_views` needs views with an immutable origin, which a
    borrowed argument has and a local `var` does not. The mask is boolean:
    `reject_non_boolean_filter` refuses anything else at plan time.
    """
    ref m = mask.as_bool()
    return Bitmap.intersect_views(Optional(m.values()), m.validity()).value()


@no_inline
def compact_to_admitted(
    mut column: DynArray,
    mut groups: Groups,
    mut predicate: DynOperator,
    morsel: Morsel,
) raises:
    """Run a buffered aggregate's `FILTER` predicate and drop the rows it
    rejects from both the operand column and the group ids.

    Non-generic and not inlined because `BufferedAggregateOperator.push` is
    instantiated once per kernel the runtime resolver can bind, 163 times in
    `query_streaming_agg`, and anything inlined into it is emitted that many
    times: written inline, these lines cost that gate +7,178,304 bytes. It
    takes the predicate rather than the evaluated mask so the evaluation's
    temporaries stay out of `push` too, which is another +1,793,408.

    The ids are filtered by the same mask so each surviving value keeps its
    group. `num_groups` does not shrink: a group whose rows were all
    rejected still produces a row, with its kernel's answer over no input.
    """
    var rows = len(column)
    var keep = BoolArray(
        length=rows,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=admitted_bits(predicate.push(morsel).value().to_array(rows)),
    )
    if not groups.is_single():
        var ids = (
            FilterKernel.dispatch(groups.ids.copy().to_dyn(), keep)
            .as_int32()
            .copy()
        )
        groups = Groups(ids^, groups.num_groups)
    column = FilterKernel.dispatch(column, keep)


trait FrameEvaluator(Deinitable, Movable):
    """Computes one window function over every frame of a sorted batch.

    What `WindowFunction.to_evaluator` builds — the window counterpart of
    `Operator`. Built for one frame spec; the extents describe the partitions
    and peer groups of the batch it runs over, so one `run` answers the whole
    column, in sorted order.
    """

    def run(
        mut self, sorted: StructArray, extents: WindowExtents, ctx: ExecContext
    ) raises -> DynArray:
        ...


struct CallEvaluator[K: WindowKernel](FrameEvaluator):
    """A window kernel over a sorted batch: evaluate the operand, if the
    kernel reads one, then hand it to `K.compute`."""

    var _kernel: Self.K
    var _argument: Optional[DynOperator]
    """The operand, lowered — absent for a kernel that reads none."""
    var _frame: WindowFrame

    def __init__(
        out self,
        var kernel: Self.K,
        var argument: Optional[DynOperator],
        frame: WindowFrame,
    ):
        self._kernel = kernel^
        self._argument = argument^
        self._frame = frame

    def run(
        mut self, sorted: StructArray, extents: WindowExtents, ctx: ExecContext
    ) raises -> DynArray:
        var n = len(sorted)
        var argument: DynArray
        if self._argument:
            argument = (
                self._argument.value()
                .push(Morsel.ungrouped(sorted.copy()))
                .value()
                .to_array(n)
            )
        else:
            argument = nulls(n, null)
        return self._kernel.compute(extents, argument, self._frame, ctx)


struct ColumnEvaluator[Agg: AggKernel, A: Evaluable](FrameEvaluator):
    """A `Windowable` aggregate over every frame: evaluate the operand to a
    column, mask it by the `FILTER`, answer each frame through `Agg.over`.

    The window counterpart of `BufferedAggregateOperator`, and here for the
    same reason: it needs only an operand that evaluates to a column, which
    both lanes have. A `FILTER` is applied as `CASE WHEN p THEN x END`: a row
    it does not admit folds like a null.
    """

    var _input: Self.A
    var _where: Optional[DynOperator]
    var _bindings: Bindings
    var _frame: WindowFrame

    def __init__(
        out self,
        var input: Self.A,
        var predicate: Optional[DynOperator],
        var bindings: Bindings,
        frame: WindowFrame,
    ):
        self._input = input^
        self._where = predicate^
        self._bindings = bindings^
        self._frame = frame

    def run(
        mut self, sorted: StructArray, extents: WindowExtents, ctx: ExecContext
    ) raises -> DynArray:
        comptime F = downcast[Self.Agg, Windowable]
        var n = len(sorted)
        var operand = self._input.evaluate(sorted, self._bindings).to_array(n)
        if self._where:
            var admitted = self._where.value().push(
                Morsel.ungrouped(sorted.copy())
            )
            operand = CaseWhenKernel.dispatch(
                [admitted.value().to_array(n).as_bool().copy()], [operand^]
            )
        return F.over(
            F.InArray(operand.to_data()), extents, self._frame
        ).to_dyn()


struct PerFrameEvaluator[V: Value](FrameEvaluator):
    """An aggregate evaluated once per distinct frame, through its own
    operator — what an aggregate with no `Windowable.over` evaluates through,
    and what makes every aggregate a window aggregate at once.

    A frame equal to the previous row's is not re-evaluated: under `RANGE`
    that is every row of a peer group, and under `ROWS` every row whose frame
    both partition edges clip. Two partitions never share a non-empty frame,
    so the reuse cannot carry an answer across a partition boundary. Each
    distinct answer is kept once and the column is one gather over them.

    **An empty frame still runs the aggregate**, on a zero-row slice, rather
    than short-circuiting to null: `COUNT` over no rows is 0 and `MIN` over no
    rows is NULL, and that is the aggregate's to decide.

    The batch is narrowed to the columns the aggregate reads before it is
    sliced: `StructArray.slice` is zero-copy but copies the struct's dtype, one
    `Field` per column, so narrowing keeps each frame's cost independent of how
    wide the input is.
    """

    var _aggregate: Self.V
    var _schema: Schema
    var _bindings: Bindings
    var _frame: WindowFrame

    def __init__(
        out self,
        var aggregate: Self.V,
        var schema: Schema,
        var bindings: Bindings,
        frame: WindowFrame,
    ):
        self._aggregate = aggregate^
        self._schema = schema^
        self._bindings = bindings^
        self._frame = frame

    def run(
        mut self, sorted: StructArray, extents: WindowExtents, ctx: ExecContext
    ) raises -> DynArray:
        var n = len(extents)
        var dtype = self._aggregate.dtype(self._schema)
        if n == 0:
            return nulls(0, dtype^)
        var reads = List[Int]()
        for name in self._aggregate.columns():
            # A name missing here is left for `to_operator` to report.
            var i = self._schema.get_field_index(name)
            if i >= 0:
                reads.append(i)
        var input = sorted.select(reads)
        var input_schema = schema(input.dtype.as_struct().fields.copy())
        var answers = List[DynArray]()
        var which = Int32Builder(n)
        var prev_lo = 0
        var prev_hi = 0
        for j in range(n):
            var lo, hi = extents.frame(j, self._frame)
            if j == 0 or lo != prev_lo or hi != prev_hi:
                var op = self._aggregate.to_operator(
                    input_schema, False, self._bindings
                )
                var produced = op.push(
                    Morsel.ungrouped(input.slice(lo, hi - lo))
                )
                if not produced:
                    produced = op.drain()
                if produced:
                    answers.append(produced.value().to_array(1))
                else:
                    answers.append(nulls(1, dtype.copy()))
            which.append(Int32(len(answers) - 1))
            prev_lo = lo
            prev_hi = hi
        return TakeKernel.dispatch(
            DynBuilder.concat(answers), which.finish(), ctx
        )


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
    can do. Hence the operand's bound is `Evaluable` alone, not `Value`.

    The `FILTER` predicate is held as a `DynOperator` although the operand
    is typed: this struct is instantiated once per kernel the runtime
    resolver can bind, so every field held by type is duplicated that many
    times, and a second typed field cost `query_streaming_agg` +2,820,216
    bytes. The operand stays typed to keep its subtree fused.
    """

    var _input: Self.A
    var _bindings: Bindings
    """This execution's parameter values, held by the *operator* rather than
    the node — which is what keeps the plan immutable and lets two executions
    of it bind different values."""

    var _where: Optional[DynOperator]
    """The `FILTER` predicate, lowered; pushed once per morsel."""

    var _state: Self.Agg
    """The accumulator, built at construction from the operand's dtype."""

    var _scatters: Bool
    """Whether the query has `GROUP BY` keys.

    A field and not a parameter, unlike the fused operator's: this arm reads it
    once per morsel to build a `Groups`, not once per row, so specialising on
    it would double the instantiation for a branch that never reaches the inner
    loop — measured at +4.6%.
    """

    var _partial: Bool
    """Whether `drain` answers the state rather than the aggregate."""
    var _emitted: Bool

    def __init__(
        out self,
        var input: Self.A,
        var predicate: Optional[DynOperator],
        var bindings: Bindings,
        scatters: Bool,
        in_dtype: DynType,
        partial: Bool = False,
    ) raises:
        self._input = input^
        self._where = predicate^
        self._bindings = bindings^
        self._state = Self.Agg(in_dtype)
        self._scatters = scatters
        self._partial = partial
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        var n = len(morsel.batch)
        var column = self._input.evaluate(
            morsel.batch, self._bindings
        ).to_array(n)
        var groups = morsel.groups.copy() if self._scatters else Groups.single(
            n
        )
        if self._where:
            # Compaction, not a validity mask: this column was evaluated, so
            # it may carry a sliced batch's offset, which a bitmap built over
            # row 0 cannot be attached to.
            compact_to_admitted(column, groups, self._where.value(), morsel)
        # The one narrowing in this lane, and it is comptime-resolved:
        # `Agg` is a parameter here, so `InArray` is a concrete type and
        # this is a conversion rather than a dispatch.
        self._state.update(groups, Self.Agg.InArray(column.to_data()))
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        # One implicit slot, and an input that produced no morsel at all
        # never grew it. `count_distinct` of nothing is one 0 and `min` of
        # nothing is one NULL — both one row, which is what the stage above
        # builds its output batch from.
        return emit_fold(self._state, 0 if self._scatters else 1, self._partial)


# ---------------------------------------------------------------------------
# Join trees — how a join chain is computed
# ---------------------------------------------------------------------------
@fieldwise_init
struct PlannedJoin(Copyable, Equatable, Movable, Writable):
    """One join of a chain's tree: the two nodes it joins, numbered as in
    `JoinOrder`, and the link it computes — a chain's own, or one joining two
    sets of a multi-join's inputs on the keys their classes share."""

    var left: Int
    var right: Int
    var link: JoinLink

    @staticmethod
    def of(
        link: JoinLink, left: Int, right: Int, build_side: JoinBuildSide
    ) -> Self:
        """`link` joining `left` to `right`, hashing `build_side`."""
        var out = link.copy()
        out.build_side = build_side
        return Self(left, right, out^)

    def __eq__(self, other: Self) -> Bool:
        return (
            self.left == other.left
            and self.right == other.right
            and self.link == other.link
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def computes(self, link: JoinLink) -> Bool:
        """Does this join compute `link`, whatever side it hashes — and hash
        the same side when `link` is `JOIN_ANY`, whose answer depends on
        it?"""
        return (
            self.link.kind == link.kind
            and self.link.strictness == link.strictness
            and (
                link.strictness == JOIN_ALL
                or self.link.build_side == link.build_side
            )
            and self.link.left_keys == link.left_keys
            and self.link.right_keys == link.right_keys
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.link.kind, " ")
        self.link.write_condition(writer)


struct JoinOrder(Copyable, Equatable, Movable, Writable):
    """The tree a join chain is computed by: left-deep as written, or the
    one `JoinOrdering` chose.

    Node `p` below `inputs` is participant `p`; node `inputs + j` is join `j`,
    each over earlier nodes; the root is the last join.
    """

    var inputs: Int
    var joins: List[PlannedJoin]

    def __init__(out self, inputs: Int):
        self.inputs = inputs
        self.joins = List[PlannedJoin]()

    def __eq__(self, other: Self) -> Bool:
        return self.inputs == other.inputs and self.joins == other.joins

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @staticmethod
    def written(inputs: Int, links: List[JoinLink]) -> JoinOrder:
        """`links` over `inputs` participants as written: left-deep, each
        link as it was joined."""
        var out = JoinOrder(inputs)
        var node = 0
        for k in range(len(links)):
            node = out.add(PlannedJoin(node, k + 1, links[k].copy()))
        return out^

    def add(mut self, var join: PlannedJoin) -> Int:
        """Append `join`, answering its node."""
        self.joins.append(join^)
        return self.inputs + len(self.joins) - 1

    def root(self) -> Int:
        return self.inputs + len(self.joins) - 1

    def is_join(self, node: Int) -> Bool:
        return node >= self.inputs

    def join(
        self, node: Int
    ) -> ref[self.joins[node - self.inputs]] PlannedJoin:
        """The join `node` is. Undefined unless `is_join(node)`."""
        return self.joins[node - self.inputs]

    def participants(self) -> List[Set[Int]]:
        """Every node's participants, by node."""
        var out = List[Set[Int]](capacity=self.inputs + len(self.joins))
        for p in range(self.inputs):
            out.append({p})
        for ref j in self.joins:
            out.append(out[j.left] | out[j.right])
        return out^

    def landing(
        self, held: List[Set[Int]], rules: JoinRules, f: JoinFilter
    ) -> Int:
        """The node `f` is evaluated after: the lowest one holding it
        (`JoinRules.holds`) — the nodes that do are the root's path down to
        it."""
        var node = self.root()
        while self.is_join(node):
            ref j = self.join(node)
            if rules.holds(f, held[j.left]):
                node = j.left
            elif rules.holds(f, held[j.right]):
                node = j.right
            else:
                return node
        return node

    def verify(self, chain: JoinChain) raises:
        """Raise unless this tree computes `chain`'s answer: every participant
        joined once; every join between two sets its `JoinRules` may join,
        reading each key from the side holding it; a participant joined by
        its own link joined by that link; and every member of every equality
        class compared by the smallest subtree holding two of them. A planner
        that got any of it wrong would compute different rows; this refuses
        it instead."""
        var used = List[Bool](length=self.inputs + len(self.joins), fill=False)
        for i in range(len(self.joins)):
            ref j = self.joins[i]
            for node in [j.left, j.right]:
                if node < 0 or node >= self.inputs + i or used[node]:
                    raise InternalError(t"join: node {node} joined twice")
                used[node] = True
        var held = self.participants()
        if (
            len(self.joins) != len(chain.links)
            or len(held[self.root()]) != self.inputs
        ):
            raise InternalError("join: a tree not joining every input once")
        var rules = chain.rules()
        var made = JoinClasses()
        for ref j in self.joins:
            ref l = held[j.left]
            ref r = held[j.right]
            for ref k in j.link.left_keys:
                if k.input not in l:
                    raise InternalError(t"join: {k} read from the wrong side")
            for ref k in j.link.right_keys:
                if k.input not in r:
                    raise InternalError(t"join: {k} read from the wrong side")
            var own = rules.attaching(l, r)
            var as_written = (
                own >= 0
                and len(r) == 1
                and own in r
                and j.computes(rules.links[own - 1])
            )
            if not rules.joinable(l, r) or not (
                as_written or (own < 0 and j.link.is_inner())
            ):
                raise InternalError(
                    t"join: {l} and {r} are not joined as the links say"
                )
            if j.link.is_inner():
                made.union_all(j.link.left_keys, j.link.right_keys)
            var s = l | r
            for ref c in rules.classes:
                var inside = c.within(s)
                for k in range(1, len(inside)):
                    if not made.connected(inside[0], inside[k]):
                        raise InternalError(
                            "join: a tree making other equalities"
                        )

    # -- what it costs ---------------------------------------------------------
    def cost(self, chain: JoinChain) raises -> Cost:
        """What computing `chain` by this tree should take: its participants
        and what `JoinPricing.tree` charges for its joins and filters."""
        var pricing = JoinPricing(chain)
        var out = pricing.tree(self)
        for ref input in chain.inputs:
            out = out + input[].cost()
        return out

    # -- rendering -----------------------------------------------------------
    def _write_node[W: Writer](self, node: Int, mut writer: W):
        if not self.is_join(node):
            writer.write("#", node)
            return
        ref j = self.join(node)
        writer.write("(")
        self._write_node(j.left, writer)
        writer.write(" ", j.link.kind, " ")
        self._write_node(j.right, writer)
        writer.write(" ")
        j.link.write_condition(writer)
        writer.write(")")

    def write_to[W: Writer](self, mut writer: W):
        self._write_node(self.root(), writer)
