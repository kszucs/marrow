# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The physical layer: the operators each plan node lowers to.

A `Relation` describes a query and is immutable, shareable and rewritable. This
is what that description becomes when it runs — one `Operator` per stage (see
`marrow.engine.core`), owning everything mutable: a grouper's table, an
accumulator's slots, a join's build side.
"""

from std.collections import Set
from std.memory import ArcPointer

from ..errors import InternalError
from ..arrays import BoolArray, DynArray, Int32Array, StructArray
from ..buffers import Bitmap
from std.builtin.rebind import downcast
from ..builders import DynBuilder, Int32Builder, arange, nulls
from ..dtypes import Field, Int32Type, field, null, struct_
from ..kernels.aggregate import AggKernel, Windowable
from ..kernels.conditional import CaseWhenKernel
from ..execution import ExecContext
from ..kernels.concat import concat
from ..kernels.filter import FilterKernel, TakeKernel
from ..kernels.groupby import Groups
from ..kernels.dictionary import DictionaryEncoder
from ..dtypes import DynType
from ..kernels.join import (
    BUILD_LEFT,
    BuildMatches,
    HashJoin,
    JOIN_ALL,
    JoinBuildSide,
    JoinKind,
)
from ..utils import KeyHash
from ..engine.core import Datum, DynOperator, Morsel, Operator, Pipeline
from ..engine.sources import Source
from .logical import Bindings
from .estimates import Cost
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
from ..kernels.sort import SortIndices
from ..kernels.window import WindowExtents, WindowFrame, WindowKernel
from ..schema import Schema, schema
from ..tabular import RecordBatch


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
    join rather than inside it, and a union's inputs, relabelled to its
    names. Positional because two inputs may share a name;
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
# over `DynRelation`. A by-value `DynOperator` spells its whole trampoline
# layout there, past `ld`'s symbol length limit when `libmarrow.so` links; a
# `Pipeline` holds its operators in a `List`, which spells as a pointer.
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
    """Where a combining operator finds the partial states it is pushed;
    read only when combining."""

    @staticmethod
    def final() -> Self:
        return Self(0, 0)

    @staticmethod
    def partial() -> Self:
        return Self(1, 0)

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
    # A stage's partial batch names each fold's struct by the aggregate's
    # output name; inside it, a kernel's state columns are `p0`, `p1`, ...
    # by position, which is how `merge` reads them back.
    var fields = List[Field](capacity=len(columns))
    for i in range(len(columns)):
        fields.append(field(names[i], columns[i].dtype()))
    return StructArray.from_arrays(columns^, fields)


def emit_fold[Agg: AggKernel](mut state: Agg, partial: Bool) raises -> Datum:
    """The aggregate, or with `partial` the state as one struct column — the
    tail every fold operator shares."""
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
    var _emitted: Bool

    def __init__(
        out self, column: Int, grouped: Bool, in_dtype: DynType
    ) raises:
        self._state = Self.Agg(in_dtype, grouped)
        self._column = column
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        # Ungrouped, the stage forwards the one-slot assignment as it is.
        var states = morsel.batch.field(self._column).as_struct().flatten()
        self._state.merge(morsel.groups, states)
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        return emit_fold(self._state, False)


struct ColumnOperator(Operator):
    """A batch's column at a fixed position."""

    var _index: Int

    def __init__(out self, index: Int):
        self._index = index

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        return Datum(morsel.batch.field(self._index))

    def drain(mut self) raises -> Optional[Datum]:
        return None


struct _Folds(Movable):
    """An aggregate stage's folds: each is pushed the same morsel, and all of
    them are drained once, into the stage's one batch."""

    var _folds: List[DynOperator]
    var _schema: Schema
    """The stage's output: keys, then one field per fold."""
    var _partial: Bool
    """Whether the folds answer partial states."""
    var _emitted: Bool

    def __init__(
        out self,
        var folds: List[DynOperator],
        var schema: Schema,
        partial: Bool,
    ):
        self._folds = folds^
        self._schema = schema^
        self._partial = partial
        self._emitted = False

    def push(mut self, morsel: Morsel) raises:
        # Indexed rather than `for ref`: a fold is move-only, and iterating a
        # `List` by reference requires `Copyable`.
        for i in range(len(self._folds)):
            _ = self._folds[i].push(morsel)

    def emit(
        mut self, var columns: List[DynArray], rows: Int
    ) raises -> Optional[Datum]:
        """The stage's one batch, once: the key `columns`, then every fold's
        answer, `rows` long; nothing after."""
        if self._emitted:
            return None
        self._emitted = True
        for i in range(len(self._folds)):
            var col = self._folds[i].drain()
            if not col:
                raise InternalError(t"aggregate: fold {i} answered nothing")
            columns.append(col.value().to_array(rows))
        if self._partial:
            # Partial states are typed by what the folds answered.
            return Datum(
                _partial_struct(columns^, self._schema.names()).to_dyn()
            )
        return Datum(_struct_of(self._schema, columns^, rows).to_dyn())


struct _Slots(Movable, Sized):
    """Which slot each row of a grouped stage lands in: its keys' code — a
    group id is the code of its key — or, once a partial stage's keys have
    stopped repeating, a slot of its own.

    A partial stage whose keys have almost all been new, over enough rows to
    judge it by, reduces little, and the combine groups the same keys again:
    it hands its rows on as they are from then on, each a slot after every
    slot so far, its keys kept after the codes' keys."""

    comptime _JUDGE_ROWS = 4096
    """Rows placed before the stage may judge them too distinct to group."""

    comptime _GROUPING = UInt8(0)
    """Every row is encoded: not a partial stage, or a key is
    dictionary-encoded, so rows' keys would not concatenate with the
    encoder's."""
    comptime _JUDGING = UInt8(1)
    """Rows are encoded while the stage judges whether they repeat."""
    comptime _PASSING = UInt8(2)
    """Rows are handed on as they are."""

    var _encoder: DictionaryEncoder[KeyHash]
    var _mode: UInt8
    var _rows_seen: Int
    """Rows the encoder has placed."""
    var _passed: Int
    """Rows handed on, each a slot after the encoder's codes."""

    def __init__(
        out self,
        var key_types: List[DynType],
        var ctx: ExecContext,
        partial: Bool,
    ):
        var may_pass = partial
        for ref t in key_types:
            may_pass = may_pass and not t.is_dictionary()
        # The encoder gets the caller's context: it is what selects radix
        # placement over the single-table path, and what stripes the key
        # hashing.
        self._encoder = DictionaryEncoder[KeyHash](key_types^, ctx^)
        self._mode = Self._JUDGING if may_pass else Self._GROUPING
        self._rows_seen = 0
        self._passed = 0

    def __len__(self) -> Int:
        """How many slots there are."""
        return len(self._encoder) + self._passed

    def place(mut self, var keys: List[DynArray], n: Int) raises -> Groups:
        """The slots of `n` rows with `keys`."""
        if self._mode == Self._PASSING:
            var first = len(self)
            self._passed += n
            self._encoder.extend_unindexed(keys^)
            return Groups(arange[Int32Type](first, first + n), first + n)
        var placed = self._encoder.encode(keys)
        self._rows_seen += n
        if (
            self._mode == Self._JUDGING
            and self._rows_seen >= Self._JUDGE_ROWS
            and 8 * len(self._encoder) > 7 * self._rows_seen
        ):
            self._mode = Self._PASSING
        return Groups(placed.ids.copy(), len(self._encoder))

    def keys(mut self) raises -> List[DynArray]:
        """Every slot's keys, in slot order."""
        return self._encoder.values()


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
    stage keeps only the slots' key values, which grow with the number of
    *groups* too.

    Group ids are dense and stable across morsels, which is what makes all of
    them work: a state that has already folded batch N keeps its slots when
    batch N+1 introduces new groups (`AggState.reserve` extends them in place),
    and concatenating N morsels' id arrays end to end is a valid assignment
    over the concatenated input rather than N unrelated numberings.

    `HAVING` needs no node of its own: a `FilterOperator` above this stage sees
    the aggregate's *output* batch, which is exactly what the flush cascade in
    `Pipeline.drain` delivers.

    **There is always a key.** A query without one is a single implicit group
    and lowers to `UngroupedAggregateOperator`, which has no slots to place.
    """

    var _keys: List[DynOperator]
    var _slots: _Slots
    var _folds: _Folds

    def __init__(
        out self,
        var keys: List[DynOperator],
        var folds: List[DynOperator],
        var schema: Schema,
        var ctx: ExecContext,
        partial: Bool,
    ):
        # The output schema is keys then aggregates, so the key types are its
        # first `len(keys)` fields.
        var key_types = List[DynType](capacity=len(keys))
        for i in range(len(keys)):
            key_types.append(schema.fields[i].dtype.copy())
        self._keys = keys^
        self._slots = _Slots(key_types^, ctx^, partial)
        self._folds = _Folds(folds^, schema^, partial)

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Assign the rows to slots **once**, then hand the same morsel to
        every fold.

        This is why placement belongs to the operator rather than to each fold:
        N aggregates over one `GROUP BY` evaluate and hash the keys once
        between them, where N folds each owning a grouping would do it N
        times.
        """
        var n = len(morsel.batch)
        var keys = List[DynArray](capacity=len(self._keys))
        for i in range(len(self._keys)):
            keys.append(self._keys[i].push(morsel).value().to_array(n))
        var groups = self._slots.place(keys^, n)
        self._folds.push(Morsel(morsel.batch.copy(), groups^))
        return None

    def drain(mut self) raises -> Optional[Datum]:
        return self._folds.emit(self._slots.keys(), len(self._slots))

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
    it goes to every fold as it is, and there are no slots to build.

    One row even over an input that yields nothing — `sum` of nothing is one
    NULL, not no rows.
    """

    var _folds: _Folds

    def __init__(
        out self,
        var folds: List[DynOperator],
        var schema: Schema,
        partial: Bool,
    ):
        self._folds = _Folds(folds^, schema^, partial)

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        self._folds.push(morsel)
        return None

    def drain(mut self) raises -> Optional[Datum]:
        return self._folds.emit(List[DynArray](), 1)

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


struct _SortKeys(Movable):
    """What a sort orders by: its key operators, their directions, where
    nulls go, and how many of the ordered rows are kept."""

    var _keys: List[DynOperator]
    var _ascending: List[Bool]
    var _nulls_first: Bool
    var _limit: Optional[Int]
    """How many ordered rows the plan above actually needs, or `None` for all.

    Set only by `optimizer.mojo`'s `TopN` rule, which alone knows that nothing
    between the `Limit` and this `Sort` drops rows."""
    var ctx: ExecContext
    """The context its kernels run on."""

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
        self.ctx = ctx^

    @no_inline
    def sorted(mut self, var whole: StructArray) raises -> Datum:
        """`whole` in key order, cut to `limit` rows when there is one.

        Multiple keys are sorted **stably, last key first**: each pass keeps
        the order the previous one established, so the composition orders by
        key 0, ties broken by key 1, and so on."""
        var order = Optional[Int32Array](None)
        for k in range(len(self._keys) - 1, -1, -1):
            var key = (
                self._keys[k]
                .push(Morsel.ungrouped(whole.copy()))
                .value()
                .to_array(len(whole))
            )
            if order:
                key = TakeKernel.dispatch(key, order.value(), self.ctx)
            # **The bound applies to the primary key only.** Every pass but
            # the last must return a *full* permutation for the next one to
            # permute: truncating an earlier pass discards rows the later keys
            # still have to order. `k == 0` is the final, most significant
            # pass.
            var pass_order = SortIndices[nan_largest=True].dispatch(
                key,
                ascending=self._ascending[k],
                nulls_first=self._nulls_first,
                limit=self._limit if k == 0 else None,
                ctx=self.ctx,
            )
            if order:
                # Compose: this pass permutes the previous order, it does not
                # replace it.
                var prev: DynArray = order.value().copy()
                order = (
                    TakeKernel.dispatch(prev, pass_order, self.ctx)
                    .as_int32()
                    .copy()
                )
            else:
                order = pass_order^

        if not order:
            return Datum(whole^.to_dyn())
        return Datum(
            TakeKernel.dispatch(whole.copy().to_dyn(), order.value(), self.ctx)
        )


struct SortOperator(Operator):
    """`ORDER BY` — blocking, because a global order needs every row.

    Buffers each morsel and sorts once at `drain`. That is not a limitation of
    the engine but of the operation: no prefix of the input determines the
    first output row.
    """

    var _by: _SortKeys
    var _batches: List[StructArray]
    var _emitted: Bool

    def __init__(
        out self,
        var keys: List[DynOperator],
        var ascending: List[Bool],
        nulls_first: Bool,
        limit: Optional[Int],
        var ctx: ExecContext,
    ):
        self._by = _SortKeys(keys^, ascending^, nulls_first, limit, ctx^)
        self._batches = List[StructArray]()
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        self._batches.append(morsel.batch.copy())
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted or len(self._batches) == 0:
            return None
        self._emitted = True
        var schema = Schema.from_dtype(self._batches[0].dtype)
        return self._by.sorted(concat(self._batches, schema, self._by.ctx))


struct TopNOperator(Operator):
    """The first `limit` rows of each morsel in key order, as it is pushed:
    the cut a top-N takes of every morsel before the rows that survive are
    sorted again together."""

    var _by: _SortKeys

    def __init__(
        out self,
        var keys: List[DynOperator],
        var ascending: List[Bool],
        nulls_first: Bool,
        limit: Int,
        var ctx: ExecContext,
    ):
        self._by = _SortKeys(keys^, ascending^, nulls_first, limit, ctx^)

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        return self._by.sorted(morsel.batch.copy())

    def drain(mut self) raises -> Optional[Datum]:
        return None


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

        var whole = concat(self._batches, self._input_schema, self._ctx)
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


@fieldwise_init
struct JoinColumns(Movable):
    """The columns of one equijoin: what it joins, what each input holds, and
    what it emits."""

    var joined: Schema
    """What the join promises its consumers: left fields then right, or one
    side alone for the kinds that emit a single side, whichever side is
    built."""
    var build: Schema
    """The build input's own fields: they type the build side when it
    produced no batch."""
    var probe: Schema
    """The probe input's own fields: they type the all-null probe columns
    the unmatched build rows are padded with."""
    var select: List[Int]
    """The positions in `joined` the join emits, as `output`. An empty list
    emits no column, every row."""
    var output: Schema


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
    emitted once, by `owed`, from the build keys every probing operator
    marked. So the probe side is never held.

    Every field is in build/probe terms except `_columns.joined`, the column
    order the kernel's output is relabelled to, and `_build_side`, which says
    which logical side was built. It emits by position, since two inputs of a
    join chain may share a name.
    """

    var _index: HashJoin[KeyHash]
    var _probe_keys: List[Int]
    """Key column indices into the **probe** input's own schema."""
    var _kind: JoinKind
    var _strictness: UInt8
    var _build_side: JoinBuildSide
    """Which logical side was built; the kernel uses it to emit the columns
    in `_columns.joined`'s order."""
    var _columns: JoinColumns
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
        var columns: JoinColumns,
        ctx: ExecContext,
    ) raises:
        var side = concat(build, columns.build, ctx)
        self._index = HashJoin[KeyHash](side, build_keys, ctx.copy())
        self._probe_keys = probe_keys^
        self._kind = kind
        self._strictness = strictness
        self._build_side = build_side
        self._columns = columns^
        self._matches = List[ArcPointer[BuildMatches]]()

    def owes_rows(self) raises -> Bool:
        """Whether build rows are emitted after the probe side, by whether
        they matched: LEFT, FULL, SEMI and ANTI, read against `(build,
        probe)`."""
        return self._build_side.physical(self._kind).needs_left_matches()

    def probe_matches(mut self) raises -> ArcPointer[BuildMatches]:
        """The marks of one more probing operator: its own, so no two threads
        write one, and kept here for `owed`."""
        var matches = ArcPointer(
            self._index.matches(self._kind, self._build_side)
        )
        if self.owes_rows():
            self._matches.append(matches)
        return matches^

    def probe(
        self, batch: StructArray, mut matches: BuildMatches, ctx: ExecContext
    ) raises -> Optional[StructArray]:
        """The joined rows of one probe morsel, marking `matches`; nothing
        when none of its rows join or, for SEMI and ANTI, it only marks."""
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

    def owed(self, ctx: ExecContext) raises -> StructArray:
        """The build rows the probe side owes, by what every probing
        operator marked: read once all of them have run."""
        var merged = self._index.matches(self._kind, self._build_side)
        for ref m in self._matches:
            merged.merge(m[])
        var tail = self._index.owed(
            merged,
            RecordBatch.empty(self._columns.probe).to_struct_array(),
            self._kind,
            self._strictness,
            self._build_side,
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
        ref want = self._columns.joined.fields
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
        return result.select(self._columns.select).with_fields(
            self._columns.output.fields
        )


struct JoinOperator(Operator):
    """Equijoin: probes each morsel against a `JoinBuild`, and marks the
    build keys its rows matched in marks of its own, registered with the
    build when the operator is made."""

    var _build: ArcPointer[JoinBuild]
    var _matches: ArcPointer[BuildMatches]
    var _ctx: ExecContext

    def __init__(
        out self, var build: ArcPointer[JoinBuild], var ctx: ExecContext
    ) raises:
        self._matches = build[].probe_matches()
        self._build = build^
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


struct UnmatchedSource(Source):
    """The build rows a join's probe side owes once it has all run — the
    unmatched ones, or SEMI's matched ones: one morsel."""

    var _build: ArcPointer[JoinBuild]

    def __init__(out self, var build: ArcPointer[JoinBuild]):
        self._build = build^

    def count(self) -> Int:
        return 1

    def rows(self) -> Int:
        return -1

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        var rows = self._build[].owed(ctx)
        if len(rows) == 0:
            return []
        return [rows^]

    def describe(self) -> String:
        return "Unmatched"


struct ResidualJoinOperator(Operator):
    """A semi or anti join whose condition goes beyond its keys: each probe
    row is kept when some build row with equal keys also satisfies every
    residual predicate (semi), or when none does (anti).

    What a correlated `[NOT] EXISTS` comparing its rows to the outer row by
    anything but equality becomes. The candidates are the key matches; each is
    evaluated as a row holding the build columns the residuals read under
    their own names and the probe columns as `Subquery.outer_field(name)`,
    which is what `outer(name)` reads. A residual that is null rejects its
    pair, as `WHERE` does.
    """

    var _join: ArcPointer[HashJoin[KeyHash]]
    var _probe_keys: List[Int]
    var _residuals: List[DynOperator]
    var _build_columns: List[Int]
    """Build column per pair field, in `_fields` order, then…"""
    var _probe_columns: List[Int]
    """…probe column per remaining pair field."""
    var _fields: List[Field]
    var _anti: Bool
    var _ctx: ExecContext

    def __init__(
        out self,
        var join: ArcPointer[HashJoin[KeyHash]],
        var probe_keys: List[Int],
        var residuals: List[DynOperator],
        var build_columns: List[Int],
        var probe_columns: List[Int],
        var fields: List[Field],
        anti: Bool,
        var ctx: ExecContext,
    ):
        self._join = join^
        self._probe_keys = probe_keys^
        self._residuals = residuals^
        self._build_columns = build_columns^
        self._probe_columns = probe_columns^
        self._fields = fields^
        self._anti = anti
        self._ctx = ctx^

    @staticmethod
    def index(
        build: List[StructArray],
        schema: Schema,
        keys: List[Int],
        ctx: ExecContext,
    ) raises -> ArcPointer[HashJoin[KeyHash]]:
        """The build side's batches indexed by `keys`, shared by every
        worker's operator."""
        return ArcPointer(
            HashJoin[KeyHash](concat(build, schema, ctx), keys, ctx.copy())
        )

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        ref batch = morsel.batch
        var index = self._join[].pairs(batch, self._probe_keys, self._ctx)
        if len(index) == 0:
            # No candidate: a semi join keeps nothing, an anti join all of it.
            if self._anti:
                return Datum(batch.copy().to_dyn())
            return None
        var columns = List[DynArray](capacity=len(self._fields))
        for i in self._build_columns:
            columns.append(
                TakeKernel.dispatch(
                    self._join[].build_column(i), index.build, self._ctx
                )
            )
        for i in self._probe_columns:
            columns.append(
                TakeKernel.dispatch(batch.field(i), index.probe, self._ctx)
            )
        var pairs: DynArray = StructArray.from_arrays(columns^, self._fields)
        var probe: DynArray = index.probe.copy()
        # Each residual drops the pairs it does not hold, null as false, so
        # the next sees only the pairs still standing.
        for r in range(len(self._residuals)):
            var produced = self._residuals[r].push(
                Morsel.ungrouped(pairs.as_struct().copy())
            )
            if not produced:
                raise InternalError("join: a residual produced no value")
            var holds = produced.value().to_array(len(pairs))
            pairs = FilterKernel.dispatch(pairs, holds.as_bool(), self._ctx)
            probe = FilterKernel.dispatch(probe, holds.as_bool(), self._ctx)
        # The probe rows some pair stood for are the matched ones.
        var n = len(batch)
        var keep = Bitmap.alloc_zeroed(n)
        if self._anti:
            keep.set_range(0, n, True)
        ref rows = probe.as_int32()
        for k in range(len(rows)):
            keep[Int(rows.unsafe_get(k))] = not self._anti
        var mask = BoolArray(
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=keep^.to_immutable(length=n),
        )
        var out = (
            FilterKernel.dispatch(batch.copy().to_dyn(), mask^, self._ctx)
            .as_struct()
            .copy()
        )
        if len(out) == 0:
            return None
        return Datum(out^.to_dyn())

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
    boundary (`Driver.collect` out).
    """
    return StructArray(
        dtype=struct_(schema.fields.copy()),
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        children=columns^,
    )


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
    """The accumulator, built at construction from the operand's dtype and
    whether the query has `GROUP BY` keys.

    That is a constructor argument and not a parameter, unlike the fused
    operators': specialising this arm on it would double the instantiation for
    a branch that never reaches the inner loop."""

    var _partial: Bool
    """Whether `drain` answers the state rather than the aggregate."""
    var _emitted: Bool

    def __init__(
        out self,
        var input: Self.A,
        var predicate: Optional[DynOperator],
        var bindings: Bindings,
        grouped: Bool,
        in_dtype: DynType,
        partial: Bool,
    ) raises:
        self._input = input^
        self._where = predicate^
        self._bindings = bindings^
        self._state = Self.Agg(in_dtype, grouped)
        self._partial = partial
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        var n = len(morsel.batch)
        var column = self._input.evaluate(
            morsel.batch, self._bindings
        ).to_array(n)
        # Ungrouped, a morsel carries the one-slot assignment already.
        var groups = morsel.groups.copy()
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
        return emit_fold(self._state, self._partial)


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
