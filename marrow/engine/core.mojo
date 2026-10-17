# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Operators, and the pipeline one worker pushes its morsels through.

**The engine pushes.** An operator is handed a morsel and answers with what it
produced, if anything:

    push(morsel: Morsel) -> Optional[Datum]
    drain()              -> Optional[Datum]
    done()               -> Bool

That one interface covers streaming and blocking alike, because **blocking
stops being a type distinction and becomes *when you return `Some`***. `Filter`
and `Project` answer from `push` and nothing from `drain`; an aggregate
accumulates through every `push`, answers `None`, and produces its whole result
from `drain`.

`drain` rather than a one-shot `finish`: it is repeatable, so an operator can
emit *several* batches at end of stream — a chunking sort, a fanning join.

**Sources are pulled, by the driver.** A plan lowers into phases, each a source
and one `Pipeline` per worker; the driver reads the source's morsels, `feed`s
each through a worker's pipeline, and drains the pipeline once the source is
spent.

`Pipeline` is one worker's operator chain rather than a single erased
operator. That is what lets `Relation.lower` stay compositional without a
`children()` walk over the plan: a source relation creates the phase, and every
relation above it appends one stage to each worker's pipeline.
"""

from std.memory import ArcPointer
from std.utils import Variant

from ..arrays import DynArray, StructArray
from ..errors import InternalError
from ..kernels.groupby import Groups
from ..scalars import ArrowScalar, DynScalar, NullScalar


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
        return scalar.get[S]().to_array(rows)

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

        Repeatable: a filter answers `None` immediately, an aggregate
        answers its result once, then `None`, and an operator may emit
        *several* batches at end of stream — a chunking sort, a fanning join —
        which a one-shot `finish` could not express at all.
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


# ---------------------------------------------------------------------------
# Pipeline — the assembled pipeline
# ---------------------------------------------------------------------------
struct Pipeline(Movable):
    """One worker's chain of operators.

    Not a second abstraction beside `Operator` — just a `List` of them, so
    `Relation.lower` stays compositional: a source relation creates a phase of
    empty pipelines, and `Filter`, `Project`, `Sort`, `Limit` and `Aggregate`
    each `append` a stage to every worker's. Nothing has to walk the plan,
    which matters because `DynRelation` deliberately exposes no `children()`.

    Appending keeps a plan's stages in one flat list: composing through
    `DynOperator` would nest one pipeline per stage and charge every batch an
    extra trampoline per level.
    """

    var _ops: List[DynOperator]

    def __init__(out self):
        """No stages: a worker's pipeline, fed by its phase's source."""
        self._ops = List[DynOperator]()

    def append(mut self, var op: DynOperator):
        self._ops.append(op^)

    def wrap[F: def(var DynOperator, Int) -> DynOperator](mut self, wrap: F):
        """Replace each stage `op`, the `i`-th, with `wrap(op, i)`: how a
        stage is decorated — observed, for a report — without the pipeline
        knowing with what."""
        var ops = List[DynOperator](capacity=len(self._ops))
        var stage = 0
        while len(self._ops) > 0:
            ops.append(wrap(self._ops.pop(0), stage))
            stage += 1
        self._ops = ops^

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
        mut self, batches: List[StructArray], mut out: List[StructArray]
    ) raises:
        """Push `batches` through every stage, in order, and append to `out`
        what comes out of the last one. Nothing is drained: a blocking stage
        keeps what it absorbed until `drain`."""
        for ref batch in batches:
            self._flow(Morsel.ungrouped(batch.copy()), 0, out)

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

    def drain(mut self, mut out: List[StructArray]) raises:
        """Append to `out` everything the chain still has, once its input is
        spent.

        Walks the stages in order, draining each dry and cascading what it
        yields through the stages above: when stage `i` yields, no later stage
        has seen it, so it flows through `i+1..` before `i+1` is drained — a
        projection over an aggregate returns nothing without this.
        """
        var stage = 0
        while stage < len(self._ops):
            var produced = self._ops[stage].drain()
            if produced:
                self._flow(
                    Morsel.ungrouped(produced.value().struct_array()),
                    stage + 1,
                    out,
                )
                # Early termination: a `Limit` with its rows stops the stages
                # **below** it from producing more — the one thing a push
                # engine must add back that a pull engine got for free. It
                # must not skip the stages **above** it: an aggregate over a
                # `Limit` answers only from `drain`, and an ungrouped one owes
                # one row however few it was pushed.
                var stop = self._first_done()
                if stop >= 0 and stop + 1 > stage:
                    stage = stop + 1
            else:
                stage += 1
