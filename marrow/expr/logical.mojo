# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The logical layer: an immutable description of a query.

Paired with `physical.mojo`, which holds what these become when they run, and
with `optimizer.mojo`, which rewrites one plan into a simpler one.

A `Relation` says *what* to compute. It owns nothing that runs, so a plan is
freely copyable, shareable, inspectable and rewritable. `to_operator(ctx)` turns
it into the physical operator that owns the running state.

**A variant for inspection, a trampoline for lowering.** `DynRelation` erases
the thirteen node types behind a `Variant`, so `isa[R]()`/`get[R]()` let an
optimizer rule read a real typed node and construct one — the capability the
previous trampoline-only box lacked, which left its "rules" as four comptime
flags and eight scattered calls inside `to_operator` with no file to read.

Lowering is the exception, and the reason is measured. `_dispatch` resolves the
active member with a `comptime for` over every member, so anything routed
through it is instantiated thirteen times; for `to_operator` that makes
`Sort.to_operator` reach `kernels::sort` and `ParquetScan.to_operator` reach the
Parquet reader and `kernels::cast`, in plans containing neither. It cost
**+348%** of `__text` on `query_streaming`, with `kernels::cast` going from 0 to
694 symbols in the fused gates. So `to_operator` binds a per-type trampoline at
construction and links only what a plan actually uses.

This supersedes the rule that used to head this file — "nothing may name every
node type in one place". The variant does name them. The narrower rule that
survives contact with the measurement is: **a closed type set may be a variant;
what must never go through its ladder is anything that reaches a kernel.**
`schema` and `write_to` stay on it deliberately — one returns a stored field,
the other formats a string.

Nodes carry `traverse(f)`, which applies `f` to their own children and rebuilds
themselves, so the optimizer holds no ladder over node types and a relation
added later needs no change there.
"""

from std.builtin.rebind import downcast
from std.memory import ArcPointer
from std.os import abort
from std.utils import Variant

from ..errors import (
    IndexError,
    InternalError,
    InvalidError,
    KeyError,
    TypeError,
)
from ..arrays import BoolArray, DynArray, StructArray
from ..execution import ExecContext
from ..kernels.join import JoinKind, JOIN_INNER, JoinBuildSide, BUILD_LEFT
from ..kernels.window import (
    FirstValue,
    Lag,
    LastValue,
    Lead,
    NthValue,
    WindowExtents,
    WindowFrame,
    WindowKernel,
)
from ..schema import Schema, schema
from ..tabular import RecordBatch
from ..dtypes import DynType, Field, StringType, field, int64, null
from .bindings import Bindings, ParamSpec, distinct_params
from .estimates import (
    Approx,
    ColumnEstimate,
    Cost,
    DEFAULT_SELECTIVITY,
    Estimate,
)
from .index import Index, keep_every
from .optimizer import RuleSet, optimize
from .`comptime`.leaves import StringParam
from .physical import (
    CallEvaluator,
    Datum,
    EvalOperator,
    Evaluable,
    FrameEvaluator,
    AggregateLowering,
    GroupedAggregateOperator,
    UngroupedAggregateOperator,
    BatchSourceOperator,
    DynOperator,
    LimitOperator,
    Pipeline,
    FilterOperator,
    JoinOperator,
    IpcScanOperator,
    JsonScanOperator,
    ParquetScanOperator,
    ProjectOperator,
    MultisetOperator,
    SortOperator,
    UnionOperator,
    WindowOperator,
)


# ---------------------------------------------------------------------------
# Shape — scalar or columnar
# ---------------------------------------------------------------------------
struct Shape(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """Whether an expression yields one value or one per row.

    A value type rather than a bare `Int`, for the reason `JoinKind` is one:
    `0` and `1` are interchangeable to the compiler and to a reader, and the
    two callers who ask this — `Datum.to_array`, deciding whether to broadcast, and
    the planner, deciding whether a projection needs materialising — would each
    be re-deriving the convention from a comment.
    """

    var _code: UInt8

    comptime scalar = Shape(0)
    """One value for the whole batch. A literal; an aggregate's result."""

    comptime columnar = Shape(1)
    """One value per row."""

    def __init__(out self, code: UInt8):
        self._code = code

    def __eq__(self, other: Self) -> Bool:
        return self._code == other._code

    def __ne__(self, other: Self) -> Bool:
        return self._code != other._code

    def write_to[W: Writer](self, mut writer: W):
        writer.write("scalar" if self == Shape.scalar else "columnar")


# ---------------------------------------------------------------------------
# Analyzable — what a rewriter asks
# ---------------------------------------------------------------------------
trait Value(Copyable, Deinitable, Writable):
    """What every expression is, in both lanes.

    Six members: what it reads, what it is called, what type it produces,
    whether it yields one value or one per row, whether it is an aggregate, and
    how to turn it into something that runs.

    **One trait, not three.** This was `Analyzable & Executable & Writable`, a
    composite alias split that way in reaction to the previous expression
    package's nine-responsibility
    `Value` trait. The reaction overshot: nothing ever bound on `Analyzable`
    alone, and `Executable` was bound alone in exactly one place, for `shape`.
    Two names that only ever appeared composed back together are not two
    abstractions — six members in one trait is the honest count, and it is
    still not nine.

    `dtype` takes a `Schema` because the runtime lane learns its type from one.
    The comptime lane ignores the argument and answers from its own type; that
    asymmetry is the price of one box holding both lanes.

    `to_operator` is the only way to *run* a value — see `physical.mojo`. There
    is deliberately no `evaluate` here. That is not a claim that a node is
    inert: in the comptime lane the node **is** the executable form, and the
    fused `bind`/`lane` machinery lives on the family traits precisely so a
    subtree stays one type and inlines into one loop. What this trait says is
    narrower and true — a node carries no *state*, so nothing outside a lane
    can run one, and two executions of the same plan cannot interfere.
    """

    comptime aggregates: Bool = False
    """Whether this value answers from `Operator.drain` rather than from a
    batch — that is, whether it is an aggregate.

    An aggregate is an ordinary `Value` in every other respect, so nothing
    structural distinguishes it and the relations that cannot accept one had no
    way to say so. All four per-row positions read this through
    `require_per_row` and raise: `Filter`'s predicate, `Project`'s values,
    `Aggregate`'s **keys**, and `Sort`'s keys. Before it,
    `project([col("a").sum()])` reached `ProjectOperator.push`, which called
    `.value()` on the `None` an aggregate answers with and **aborted the
    process** — and `Aggregate` and `Sort` kept aborting that way until the
    check reached them too.

    A defaulted `comptime` rather than a marker trait, because a trait
    constraining nothing documents nothing: every value would still satisfy
    `Value` either way, and only the *answer* differs.
    """

    comptime shape: Shape
    """`Shape.scalar` or `Shape.columnar` — whether this yields one value or
    one per row. Known without running, which is why it lives here and not on
    the operator."""

    def conjuncts(self) -> List[DynValue]:
        """This predicate split on `AND`, or `[self]` if it is not one.

        Decided at the `.filter()` verb like `constant_bool`, and for the same
        reason — the concrete type is visible there and nowhere later. Each
        conjunct is boxed **whole**, so a comptime subtree stays fused: this
        moves the erasure boundary, it never crosses it.

        What splitting buys is that each conjunct prunes and moves
        independently. A compound `AND` node prunes only as well as its weaker
        half, and cannot be pushed below a join at all when one half names the
        left side and the other the right.

        Defaulted to "this is one conjunct", which is always sound.
        """
        return [DynValue(self.copy())]

    def mask(
        self, index: Index, bindings: Bindings = Bindings()
    ) raises -> BoolArray:
        """Which chunks of a source could contain a row this predicate keeps.

        One bit per chunk, computed with the very kernels that filter rows —
        pruning is the same comparison over a different domain, one statistic
        per chunk instead of one value per row.

        **The default keeps everything, and that is the only soundness rule in
        the system.** A node added tomorrow cannot be forgotten by a predicate
        written today, because forgetting it means answering all-true, which is
        always correct. All-true is also the identity for `AND`, so a composite
        needs no special case for an operand that cannot summarise itself.

        A *null* bit means "cannot prove anything" and is read as keep, so an
        absent statistic can never cause a skip.

        `bindings` because a predicate may name a parameter, and a parameter
        that cannot be read is a predicate that cannot prune: the AOT lane's
        whole surface is `col("amount") >= param("min-amount")`.
        """
        return keep_every(index.chunks)

    def constant_bool(self) -> Optional[Bool]:
        """`True`/`False` if this is a constant boolean, else `None`.

        **A trait default, deliberately not a `DynValue` slot.** It is read at
        the `.filter()` verb, where the concrete type is still visible, and the
        answer is stored on the `Filter` node — the same placement argument
        `mask` makes, and for the same reason: a slot on `DynValue`
        is paid for every projection value, every sort key and every aggregate
        input in the program, to serve the one caller that filters.

        Defaulted to `None` — "not known to be constant" — so a node that has
        not been taught costs an optimization and never an answer. Only
        `RuntimeValue` overrides it; a comptime literal could too, but a fused
        predicate that is constant is a program someone wrote by hand.
        """
        return None

    # -- the window functions that read this value --------------------------
    #
    # Trait defaults, so both lanes get them from one definition and neither
    # pays for a window function it never names. Each answers a `WindowCall`,
    # which is not a `Value`: it becomes one only through `.over(...)`, the
    # window it runs in, exactly as in SQL.

    def lag(self, offset: Int = 1) -> WindowCall[Lag, Self]:
        """`LAG(self, offset)` — this value `offset` rows earlier."""
        return WindowCall(Lag(offset), self.copy())

    def lead(self, offset: Int = 1) -> WindowCall[Lead, Self]:
        """`LEAD(self, offset)` — this value `offset` rows later."""
        return WindowCall(Lead(offset), self.copy())

    def first_value(self) -> WindowCall[FirstValue, Self]:
        """`FIRST_VALUE(self)` — this value at the frame's first row."""
        return WindowCall(FirstValue(), self.copy())

    def nth_value(self, n: Int) raises -> WindowCall[NthValue, Self]:
        """`NTH_VALUE(self, n)` — this value at the frame's `n`-th row,
        1-based; null where the frame holds fewer than `n` rows."""
        return WindowCall(NthValue(n), self.copy())

    def last_value(self) -> WindowCall[LastValue, Self]:
        """`LAST_VALUE(self)` — this value at the frame's last row.

        Under the default frame that is the *current* row's peer group's last,
        not the partition's last. See `WindowFrame`.
        """
        return WindowCall(LastValue(), self.copy())

    def references(self, mut into: References):
        """Every column and parameter this expression reads, operands in
        declaration order.

        **Derived for a composite.** A node's operands are its fields that are
        themselves a `Value`, and this walks them by reflection -- so adding a
        node adds no walk, and a composite cannot forget an operand.

        **A leaf must override it**, because a leaf's answer is not in an
        operand: a column names itself, a parameter declares itself, a literal
        reads nothing. The `comptime assert` turns a forgotten override into a
        build error rather than an empty answer, which `ColumnPruning` would
        read as "nothing here needs a column".
        """
        comptime r = reflect[Self]
        comptime assert _operand_count[Self]() > 0, (
            "Value.references: a leaf must override references() to say what"
            " it reads"
        )
        comptime for i in range(r.field_count()):
            comptime if conforms_to(r.field_at[i].T, Value):
                r.field_ref[i](self).references(into)

    def columns(self) -> List[String]:
        """Which columns this expression reads, deduplicated, first-seen
        order."""
        var refs = References()
        self.references(refs)
        return refs.columns.copy()

    def name(self) -> String:
        """This expression's name, or empty when it has none."""
        ...

    def dtype(self, schema: Schema) raises -> DynType:
        """The type this produces, without running anything."""
        ...

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        """The stateful thing that runs this value.

        `schema` describes this value's **input**, and it is what lets a
        name-resolved aggregate pick its kernel here, at plan-build time,
        rather than on the first morsel. That is the difference between the
        runtime lane holding an erased aggregate state and holding a *typed*
        one behind the `DynOperator` box every operator already pays for:
        `dispatch_numeric` hands each arm a concrete `V`, so the arm can
        construct `AggregateOperator[Fold[K, V], RuntimeValue, G]` outright.
        Every relation has its input's schema where it calls this.

        `grouped` picks a fold's placement and is ignored by everything else.
        `bindings` supplies this execution's parameter values — the operator
        carries them and hands them back down to `bind`, where a `NumericParam` reads
        them. That is why a plan holds no parameter state and two executions
        of it cannot interfere.
        """
        ...


# ---------------------------------------------------------------------------
# WindowSpec — PARTITION BY / ORDER BY
# ---------------------------------------------------------------------------
struct WindowSpec(Copyable, Equatable, Movable, Writable):
    """`PARTITION BY ... ORDER BY ...` — the ordering a window value is
    computed in.

    Two window values with equal specs can share one sort, which is what
    `MergeWindows` looks for. **Equality compares renderings**, never the keys
    element by element: comparing erased values structurally is the
    `__eq__` shape that deadlocks the compiler (CLAUDE.md), and a key's
    rendering is already the canonical spelling of what it reads.
    """

    var partition_by: List[DynValue]
    var order_by: List[DynValue]
    var ascending: List[Bool]
    """One direction per `order_by` key."""
    var nulls_first: Bool

    def __init__(
        out self,
        var partition_by: List[DynValue],
        var order_by: List[DynValue],
        var ascending: List[Bool],
        nulls_first: Bool,
    ) raises:
        """`ascending` may be empty, meaning all-ascending. Every key must have
        a value per row: an aggregate or another window value cannot order
        rows it has not been computed for."""
        if len(ascending) == 0:
            for _ in range(len(order_by)):
                ascending.append(True)
        if len(ascending) != len(order_by):
            raise InvalidError(
                t"over: {len(order_by)} order keys but {len(ascending)} "
                t"directions"
            )
        for ref k in partition_by:
            require_per_row(
                k, "over", k.name(), "aggregate first, then window the result"
            )
        for ref k in order_by:
            require_per_row(
                k, "over", k.name(), "aggregate first, then window the result"
            )
        self.partition_by = partition_by^
        self.order_by = order_by^
        self.ascending = ascending^
        self.nulls_first = nulls_first

    def __eq__(self, other: Self) -> Bool:
        return String(self) == String(other)

    def references(self, mut into: References):
        """Every column and parameter the keys read."""
        for ref k in self.partition_by:
            k.references(into)
        for ref k in self.order_by:
            k.references(into)

    def write_to[W: Writer](self, mut writer: W):
        """`partition k order v asc`, then `nulls last` when nulls do not sort
        first, so two specs render alike exactly when they are equal."""
        for i in range(len(self.partition_by)):
            writer.write("partition " if i == 0 else ", ")
            writer.write(self.partition_by[i])
        for i in range(len(self.order_by)):
            writer.write(" order " if i == 0 else ", ")
            writer.write(self.order_by[i])
            writer.write(" asc" if self.ascending[i] else " desc")
        if not self.nulls_first:
            writer.write(" nulls last")


# ---------------------------------------------------------------------------
# DynValue — the box, and the only place the two lanes meet
# ---------------------------------------------------------------------------
struct DynValue(Copyable, Movable, Writable):
    """An expression of either lane, erased.

    **This box is the feature, not overhead.** It is what lets a dynamically
    composed plan hold comptime-fused expressions — measured at 1.46 MB against
    4.91 MB for the same plan with runtime expressions. Removing it would force
    a choice between a fully comptime plan (which instantiates per plan shape
    and no Python frontend can build) and runtime expressions everywhere (which
    is the 4.91 MB configuration).

    Nine function slots — `references`, `name`, `dtype`, `write`,
    `to_operator`, `mask`, `window_spec`, `window_column` and `_drop` — plus
    three constant fields, `shape`, `aggregates` and `windowed`, read once at
    construction because all three are comptime constants. `_drop` is the
    destructor trampoline every erased box here needs; erasure through
    `rebind[ArcPointer[NoneType]]` forgets the pointee's destructor otherwise.
    the previous expression package carried seven and had no `dtype`, computing
    output types by
    evaluating against a zero-row batch instead.

    the previous expression package's two extra slots were `name()`, which
    duplicated what `name` and
    `write` already answered, and `resolve_names` — a *rewrite*, carried by
    every boxed expression in every binary though it is a no-op in the comptime
    lane. Nothing here is a rewrite: parameter values travel *through* an
    execution rather than being substituted into a copy of the plan, so the
    box never has to hand back a re-boxed `DynValue`.

    Deliberately **not** conforming to the traits it erases. A box may hold a
    trait-bound value; it should not be one. `DynValue` exposes the same
    surface as its own API, and nothing in the tree asks it to substitute for a
    typed value in generic code.
    """

    var _boxed: ArcPointer[NoneType]
    var _references: def(ArcPointer[NoneType], mut References) thin
    var _name: def(ArcPointer[NoneType]) thin -> String
    var _dtype: def(ArcPointer[NoneType], Schema) thin raises -> DynType
    var _write: def(ArcPointer[NoneType]) thin -> String
    var _to_operator: def(
        ArcPointer[NoneType], Schema, Bool, Bindings
    ) thin raises -> DynOperator
    var _mask: def(
        ArcPointer[NoneType], Index, Bindings
    ) thin raises -> BoolArray
    """Which chunks of a source this value could keep.

    A sixth slot rather than a second box. `Filter` used to carry its predicate
    twice — once as a `DynValue` to evaluate and once as a `Pruner` to prune —
    two allocations and two trampoline tables for one value, because a `mask`
    slot here was assumed to cost what an extra slot on the old aggregate box
    cost (+3.2 MB). It does not: `Value.mask` defaults to keeping everything,
    so a projection value or a sort key wires a trampoline to `keep_every` and
    drags in nothing. The cone belongs to predicates that actually read an
    index, and those were paying for it anyway."""

    var _window_spec: def(ArcPointer[NoneType]) thin raises -> WindowSpec
    var _window_column: def(
        ArcPointer[NoneType],
        StructArray,
        WindowExtents,
        Schema,
        Bindings,
        ExecContext,
    ) thin raises -> DynArray
    """A window value's spec, and its column over a batch sorted by that spec.

    **Wired by the boxed type**: a `WindowValue` gets its own trampolines and
    every other value one shared stub that raises, so a binary that boxes an
    aggregate for a `GROUP BY` links no window code. Neither signature names
    an operator, because the linker caps a symbol's length and every
    trampoline instantiated over this box spells these slots out."""

    var _shape: Shape
    var _aggregates: Bool
    var _windowed: Bool
    var _drop: def(var ArcPointer[NoneType]) thin
    """Erasure forgets the pointee's destructor; this carries it. See
    `DynOperator._virt_drop` for why the release has to happen at the true
    type, and for the probe that measured it."""

    # -- trampolines --------------------------------------------------------
    # One instantiation per boxed type, wired at construction. There is no
    # registry and nothing names every value type in one place, so a type that
    # is never boxed costs nothing in the binary.

    @staticmethod
    def _mask_tramp[
        V: Value
    ](
        ptr: ArcPointer[NoneType], index: Index, bindings: Bindings
    ) raises -> BoolArray:
        return rebind[ArcPointer[V]](ptr)[].mask(index, bindings)

    @staticmethod
    def _references_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], mut into: References):
        rebind[ArcPointer[V]](ptr)[].references(into)

    @staticmethod
    def _name_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
        return rebind[ArcPointer[V]](ptr)[].name()

    @staticmethod
    def _dtype_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], schema: Schema) raises -> DynType:
        return rebind[ArcPointer[V]](ptr)[].dtype(schema)

    @staticmethod
    def _to_operator_tramp[
        V: Value
    ](
        ptr: ArcPointer[NoneType],
        schema: Schema,
        grouped: Bool,
        bindings: Bindings,
    ) raises -> DynOperator:
        return rebind[ArcPointer[V]](ptr)[].to_operator(
            schema, grouped, bindings
        )

    @staticmethod
    def _write_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
        return String(rebind[ArcPointer[V]](ptr)[])

    @staticmethod
    def _window_spec_tramp[
        V: WindowValue
    ](ptr: ArcPointer[NoneType]) -> WindowSpec:
        return rebind[ArcPointer[V]](ptr)[].spec()

    @staticmethod
    def _window_column_tramp[
        V: WindowValue
    ](
        ptr: ArcPointer[NoneType],
        sorted: StructArray,
        extents: WindowExtents,
        schema: Schema,
        bindings: Bindings,
        ctx: ExecContext,
    ) raises -> DynArray:
        var evaluator = rebind[ArcPointer[V]](ptr)[].to_evaluator(
            schema, bindings
        )
        return evaluator.run(sorted, extents, ctx)

    @staticmethod
    def _not_windowed_spec(ptr: ArcPointer[NoneType]) raises -> WindowSpec:
        raise InvalidError("window: this value has no window; see `.over()`")

    @staticmethod
    def _not_windowed_column(
        ptr: ArcPointer[NoneType],
        sorted: StructArray,
        extents: WindowExtents,
        schema: Schema,
        bindings: Bindings,
        ctx: ExecContext,
    ) raises -> DynArray:
        raise InvalidError("window: this value has no window; see `.over()`")

    @staticmethod
    def _drop_tramp[V: Value](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[V]](ptr)
        _ = ptr^
        _ = typed^

    @implicit
    def __init__[V: Value](out self, value: V):
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._drop = Self._drop_tramp[V]
        self._aggregates = V.aggregates
        self._references = Self._references_tramp[V]
        self._name = Self._name_tramp[V]
        self._dtype = Self._dtype_tramp[V]
        self._write = Self._write_tramp[V]
        self._to_operator = Self._to_operator_tramp[V]
        self._mask = Self._mask_tramp[V]
        self._windowed = conforms_to(V, WindowValue)
        comptime if conforms_to(V, WindowValue):
            self._window_spec = Self._window_spec_tramp[
                downcast[V, WindowValue]
            ]
            self._window_column = Self._window_column_tramp[
                downcast[V, WindowValue]
            ]
        else:
            self._window_spec = Self._not_windowed_spec
            self._window_column = Self._not_windowed_column
        self._shape = V.shape

    def __deinit__(deinit self):
        self._drop(self._boxed^)

    # -- the erased surface -------------------------------------------------

    def mask(
        self, index: Index, bindings: Bindings = Bindings()
    ) raises -> BoolArray:
        """Which chunks of a source could hold a row this value keeps.

        The one indirect call on the pruning path, at the coarsest granularity
        there is: a source calls it once per scan, and everything below is
        monomorphic.
        """
        return self._mask(self._boxed, index, bindings)

    def window_spec(self) raises -> WindowSpec:
        """The boxed window value's spec; raises unless `windowed()`."""
        return self._window_spec(self._boxed)

    def window_column(
        self,
        sorted: StructArray,
        extents: WindowExtents,
        schema: Schema,
        bindings: Bindings,
        ctx: ExecContext,
    ) raises -> DynArray:
        """The boxed window value's column over `sorted`, a batch of `schema`
        already ordered by `window_spec()` and described by `extents` — in
        that sorted order. Raises unless `windowed()`."""
        return self._window_column(
            self._boxed, sorted, extents, schema, bindings, ctx
        )

    def references(self, mut into: References):
        self._references(self._boxed, into)

    def columns(self) -> List[String]:
        var refs = References()
        self.references(refs)
        return refs.columns.copy()

    def name(self) -> String:
        return self._name(self._boxed)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._dtype(self._boxed, schema)

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        """The stateful thing that runs this value.

        The slot an aggregate accumulator would occupy, on the one box that
        holds every value. An aggregate reaches one of its three operators
        through here; an
        elementwise value reaches an `EvalOperator`. The caller cannot tell,
        which is the point.
        """
        return self._to_operator(self._boxed, schema, grouped, bindings)

    def aggregates(self) -> Bool:
        """Whether the boxed value answers from `drain` rather than per batch.

        A field for the same reason `shape` is one: a constant per boxed value,
        so a trampoline would pay an indirect call to read something fixed at
        construction."""
        return self._aggregates

    def windowed(self) -> Bool:
        """Whether the boxed value is a window value — a function over a
        window, which only a `Window` node can compute. A field for the reason
        `aggregates` is one."""
        return self._windowed

    def shape(self) -> Shape:
        """The boxed value's `shape`, read at construction.

        A field rather than a seventh trampoline: it is a constant per boxed
        type, so calling through a pointer to fetch it would pay a call to
        learn something already known.
        """
        return self._shape

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._write(self._boxed))


# ---------------------------------------------------------------------------
# References — what an expression reads from outside itself
# ---------------------------------------------------------------------------
struct References(Movable):
    """What an expression reads from outside itself: columns from the batch,
    parameters from `Bindings`.

    The one collector the structural walk fills, so a new kind of reference is
    a new field here rather than a second walk over every node.
    """

    var columns: List[String]
    """First-seen order, deduplicated -- the `columns()` contract."""

    var params: List[ParamSpec]
    """Every occurrence, in walk order."""

    def __init__(out self):
        self.columns = List[String]()
        self.params = List[ParamSpec]()

    def column(mut self, name: String):
        """Record a column read, once.

        **A linear scan, not a `Set`, and that is measured.** An expression
        reads a handful of columns, so the scan is cheap; a `Set[String]` here
        linked the hash table's growth paths into every fused binary --
        +6,356 bytes of `__text` (+0.44%) on `query_streaming_agg_fused`, whose
        single-operand nodes never linked them before.
        """
        var seen = False
        for ref c in self.columns:
            if c == name:
                seen = True
        if not seen:
            self.columns.append(name.copy())

    def param(mut self, var spec: ParamSpec):
        """Record a parameter read."""
        self.params.append(spec^)

    def extend(mut self, other: References):
        """Record everything `other` recorded, after what is already here."""
        for ref c in other.columns:
            self.column(c)
        for ref p in other.params:
            self.params.append(p.copy())


def _operand_count[T: AnyType]() -> Int:
    """How many of `T`'s fields are operands -- fields that are a `Value`."""
    comptime r = reflect[T]
    var count = 0
    comptime for i in range(r.field_count()):
        comptime if conforms_to(r.field_at[i].T, Value):
            count += 1
    return count


def require_per_row(
    value: DynValue, node: StringSlice, name: StringSlice, remedy: StringSlice
) raises:
    """Refuse a value that has no answer per row in a position evaluated once
    per row: an aggregate, or a window value.

    An aggregate's operator answers `None` to every `push` and yields only at
    `drain`, so a per-row consumer unwraps that `None` and aborts the process
    rather than raising. A window value needs its whole partition, sorted, so
    only a `Window` node can compute one. The per-row positions are `Filter`'s
    predicate, `Project`'s values, `Aggregate`'s **keys**, `Sort`'s keys and a
    window's own keys, and each calls this. `remedy` names the way to say what
    the caller meant for an aggregate, which differs per node; a window value
    always has the same one.
    """
    if value.aggregates():
        raise InvalidError(
            t"{node}: '{name}' is an aggregate, which has no value per row; "
            t"{remedy}"
        )
    if value.windowed():
        raise InvalidError(
            t"{node}: a window function has no value until its partition is "
            t"read; add it with `with_columns` first"
        )


def reject_non_boolean_filter(dtype: DynType) raises:
    """Refuse a `FILTER` predicate that is not boolean. Called at plan time in
    `to_operator`, so the per-morsel path can read the predicate's column as a
    `BoolArray`, which for a non-boolean column would abort the process rather
    than raise.
    """
    if not dtype.is_bool():
        raise TypeError(
            t"filter: an aggregate's FILTER predicate must be boolean, got "
            t"{dtype}"
        )


# ---------------------------------------------------------------------------
# The empty operand slot
# ---------------------------------------------------------------------------
trait Absent(Evaluable, Value):
    """The type of an optional operand slot that holds nothing.

    It declares no members of its own; `is_filled[P]` asks whether a slot's
    type `P` is one. A trait rather than a comparison with `Nothing` because
    `conforms_to` takes a trait and there is no type-equality test.
    """

    pass


struct Nothing(Absent):
    """The empty slot: no fields and no behaviour. Its members exist to
    satisfy the bound and are never reached, because a slot is read only
    under `comptime if is_filled[P]`.
    """

    comptime shape = Shape.scalar

    def __init__(out self):
        pass

    def references(self, mut into: References):
        """Nothing to read. An override rather than the reflected walk, which
        `comptime assert`s that a leaf says what it reads."""
        pass

    def name(self) -> String:
        return String()

    def dtype(self, schema: Schema) raises -> DynType:
        raise InternalError("an empty operand slot has no type")

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        raise InternalError("an empty operand slot cannot run")

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        raise InternalError("an empty operand slot has no value")

    def write_to[W: Writer](self, mut writer: W):
        pass


comptime is_filled[P: AnyType] = not conforms_to(P, Absent)
"""Whether an optional-operand slot of type `P` holds an operand.

The question `Absent` exists to answer, asked in one place. A node that
carries an optional operand and every operator it lowers to each need the
answer for their own `P`, so they name this rather than restating
`conforms_to`: a slot type gaining a second empty representation would
otherwise have to be taught to each of them separately."""


# ---------------------------------------------------------------------------
# FieldRef — an input field, kept as it is
# ---------------------------------------------------------------------------
struct FieldRef(Evaluable, Value):
    """A field of the input schema, evaluated to that column as it is —
    Arrow C++'s `field_ref`.

    What the relational verbs keep a column with: `select`, `drop`, `rename`,
    `with_columns`, `distinct`, and `MergeWindows`. Built from the schema
    rather than written by a caller, so it holds the resolved `Field` —
    dtype, `nullable` and metadata — and belongs to neither lane: reading a
    column needs no kernel, so a kept column links none. That is the
    difference from `col(name)`, whose runtime-lane read would link the
    runtime lane's interpreter into every binary that keeps a column.
    """

    comptime shape = Shape.columnar

    var _field: Field

    def __init__(out self, var field: Field):
        self._field = field^

    def references(self, mut into: References):
        into.column(self._field.name)

    def name(self) -> String:
        return self._field.name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return self._field.dtype.copy()

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        return EvalOperator(self.copy(), bindings.copy())

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(self._field.name).copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._field.name)


# ---------------------------------------------------------------------------
# Window functions and window values
# ---------------------------------------------------------------------------
trait WindowFunction(Copyable, Deinitable, Writable):
    """What can run over a window: an aggregate, or a `WindowCall` to a
    ranking, offset or frame-edge kernel.

    Not a `Value`, and that is the point: a window function has no answer
    until it is given the window it runs in, so `.over(...)` is how it becomes
    one — `Over[Self]`, a window value.

    `to_evaluator` is the window counterpart of `Value.to_operator`: the
    physical thing that computes this function over every frame of a sorted
    batch. The evaluator is a type rather than a box, so a fused aggregate
    stays fused through it.
    """

    comptime Evaluator: FrameEvaluator
    """What `to_evaluator` builds."""

    def dtype(self, schema: Schema) raises -> DynType:
        """The type this produces over an input of `schema`."""
        ...

    def references(self, mut into: References):
        """Every column and parameter this reads."""
        ...

    def to_evaluator(
        self, schema: Schema, bindings: Bindings, frame: WindowFrame
    ) raises -> Self.Evaluator:
        """The evaluator for this function over `frame`, for input batches of
        `schema` and this execution's `bindings`."""
        ...

    def over(
        self,
        var partition_by: List[DynValue] = List[DynValue](),
        var order_by: List[DynValue] = List[DynValue](),
        var ascending: List[Bool] = List[Bool](),
        nulls_first: Bool = True,
        var rows: Optional[Tuple[Int, Int]] = None,
    ) raises -> Over[Self]:
        """`OVER (PARTITION BY ... ORDER BY ...)` — this function over a window.

        `ascending` defaults to all-ascending, sized to `order_by`, so the
        common case names only the keys. `rows` is an explicit `ROWS` frame as
        `(preceding, following)`; without it the default `RANGE` frame
        applies. No keys at all is `OVER ()`: one partition, the whole input.
        """
        var frame = WindowFrame.default()
        if rows:
            ref bounds = rows.value()
            frame = WindowFrame(True, bounds[0], bounds[1])
        return Over(
            self.copy(),
            WindowSpec(partition_by^, order_by^, ascending^, nulls_first),
            frame,
        )


trait WindowValue(Value):
    """A value computed over a window rather than per row — `Over[F]`.

    What `DynValue` wires its window slots for, and the reason a `Window` node
    exists: the node sorts by `spec()`, describes the sorted batch's
    partitions and peer groups once, and hands both to every window value on
    it."""

    comptime Evaluator: FrameEvaluator

    def spec(self) -> WindowSpec:
        """The window this runs over."""
        ...

    def to_evaluator(
        self, schema: Schema, bindings: Bindings
    ) raises -> Self.Evaluator:
        """The evaluator computing this value over a sorted batch."""
        ...


struct WindowCall[K: WindowKernel, A: Value = Nothing](WindowFunction):
    """A call to window kernel `K`, reading operand `A` or none.

    Built by `row_number()` and the other ranking builders, which read no
    operand, and by `Value.lag` and its siblings, which read the value they
    are called on. The operand stays typed, so a fused subtree stays one loop.
    """

    comptime Evaluator = CallEvaluator[Self.K]

    var _kernel: Self.K
    var _argument: Self.A

    def __init__(out self, var kernel: Self.K, var argument: Self.A):
        comptime assert (
            Self.K.reads_argument() == is_filled[Self.A]
        ), "window: a kernel reads an operand exactly when it is given one"
        comptime assert not Self.A.aggregates, (
            "window: an aggregate has no value per row; window the aggregate"
            " itself with .over()"
        )
        comptime assert not conforms_to(Self.A, WindowValue), (
            "window: a window value has no value per row; add it with"
            " with_columns first"
        )
        self._kernel = kernel^
        self._argument = argument^

    def dtype(self, schema: Schema) raises -> DynType:
        comptime if is_filled[Self.A]:
            return Self.K.dtype(self._argument.dtype(schema))
        else:
            return Self.K.dtype(null)

    def references(self, mut into: References):
        self._argument.references(into)

    def to_evaluator(
        self, schema: Schema, bindings: Bindings, frame: WindowFrame
    ) raises -> Self.Evaluator:
        var argument: Optional[DynOperator] = None
        comptime if is_filled[Self.A]:
            argument = self._argument.to_operator(schema, False, bindings)
        return CallEvaluator(self._kernel.copy(), argument^, frame)

    def write_to[W: Writer](self, mut writer: W):
        """`lag(v, 2)`, `ntile(4)`, `row_number()`."""
        writer.write(Self.K.name(), "(")
        var first = True
        comptime if is_filled[Self.A]:
            writer.write(self._argument)
            first = False
        for p in self._kernel.params():
            if not first:
                writer.write(", ")
            writer.write(p)
            first = False
        writer.write(")")


struct Over[F: WindowFunction](WindowValue):
    """Window function `F` over a window — `F OVER (...)`.

    A `Value`, so it is boxed by `DynValue` like any other expression, but one
    only a `Window` node can compute: its answer for a row depends on rows
    that may arrive in a later batch. Every per-row position refuses it
    (`require_per_row`), and `to_operator` raises for the one path that
    check does not cover.
    """

    comptime shape = Shape.columnar
    comptime Evaluator = Self.F.Evaluator

    var _function: Self.F
    var _spec: WindowSpec
    var _frame: WindowFrame

    def __init__(
        out self, var function: Self.F, var spec: WindowSpec, frame: WindowFrame
    ):
        self._function = function^
        self._spec = spec^
        self._frame = frame

    def spec(self) -> WindowSpec:
        return self._spec.copy()

    def to_evaluator(
        self, schema: Schema, bindings: Bindings
    ) raises -> Self.Evaluator:
        return self._function.to_evaluator(schema, bindings, self._frame)

    def name(self) -> String:
        """Empty: a window value is named by the `with_columns` call that
        adds it."""
        return String()

    def dtype(self, schema: Schema) raises -> DynType:
        return self._function.dtype(schema)

    def references(self, mut into: References):
        """The function's reads, then the window keys'."""
        self._function.references(into)
        self._spec.references(into)

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        var rendered = String(self)
        raise InvalidError(
            t"window: '{rendered}' needs its whole partition; add it with "
            t"`with_columns`"
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            self._function, " over(", self._spec, " ", self._frame, ")"
        )


# ---------------------------------------------------------------------------
# The conjunction a source is asked to prove
# ---------------------------------------------------------------------------


trait Relation(Copyable, Deinitable, Movable, Writable):
    """An immutable description of a query."""

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        """This node with `f` applied to each of its children.

        The one method an optimizer needs from a node in order to walk a plan,
        and the reason `optimizer.mojo` contains no ladder over node types: a
        node knows its own children and how to put itself back together, so a
        traversal is `node.traverse(rewrite)` rather than eight `isa` arms that
        have to be extended every time a node is added.

        Defaulted to "no children", which is correct for the two leaves and
        conservative for anything added later — an untraversed node is left
        whole rather than rebuilt wrongly.
        """
        return DynRelation(self.copy())

    def references(self, mut into: References):
        """Every column and parameter this plan's expressions read: the inputs'
        first, then this node's own. Read-only by rule — no node is built and
        nothing is lowered, so walking a plan links no operator."""
        ...

    def schema(self) -> Schema:
        """The columns this relation produces.

        Computed at construction and stored, not derived on demand: a caller
        asks for it once per plan node while building the node above, and a
        `Filter` would otherwise re-derive its input's schema every time.
        """
        ...

    def estimate(self) raises -> Estimate:
        """How many rows this node produces, and what is known about each
        column.

        The default knows nothing, `Estimate()`, which every formula
        propagates as unknown, so a node without an estimate costs an
        optimization and never a wrong answer. Resolved on the variant ladder
        rather than through a trampoline slot: it reaches no kernel, and a
        slot would be paid by every binary that holds a plan.
        """
        return Estimate()

    def cost(self) raises -> Cost:
        """What running this subtree should take; see `estimates.Cost`.

        The default is unknown, not free: a model whose untaught nodes cost
        nothing would pick exactly the plan nobody modelled. Unknown is
        absorbing, so one such node makes the whole plan incomparable.
        """
        return Cost.unknown()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        """The running operator for this description."""
        ...


struct DynRelation(Copyable, Movable, Writable):
    """A `Relation` of any node, erased — a `Variant` for inspection, a
    trampoline for lowering.

    **Inspection is a variant**, the same shape as `DynArray`, `DynScalar` and
    `DynBuilder`: `isa[R]()` is a discriminant compare and `get[R]()` a borrow,
    so `optimizer.mojo`'s rules read a real typed node and can build one.
    Neither instantiates anything per member, so the optimizer's API costs
    nothing in a binary that never optimizes. It also needs no `_drop` slot —
    a variant destroys its member at the true type, where
    `rebind[ArcPointer[NoneType]]` erasure forgot the destructor entirely.

    **Lowering is a trampoline**, and that split is the whole design. Resolving
    `to_operator` through the variant's `comptime for` instantiates it for
    *every* member, so `Sort.to_operator` reaches `kernels::sort` and
    `ParquetScan.to_operator` reaches the Parquet reader and `kernels::cast` —
    in a plan containing neither. Measured at **+348%** of `__text` on
    `query_streaming`, with `kernels::cast` going 0 -> 694 symbols in the fused
    gates. A trampoline binds the single type its caller constructed, so a
    binary links only the operators its plans actually use.

    `schema` and `write_to` stay on the variant ladder deliberately: also
    instantiated thirteen times, but one returns a stored field and the other
    formats a string. Neither reaches a kernel.

    Children sit behind `ArcPointer`: a variant containing a node containing
    that variant by value has no finite size, and the compiler says so —
    *"attempt to resolve a recursive reference to declaration
    'DynRelation.__move_ctor_is_trivial'"*. Same indirection `StructArray` uses
    inside a variant-backed `DynArray`, so copying a plan stays O(1).
    """

    comptime VariantType = Variant[
        EmptyRelation,
        InMemoryTable,
        Filter,
        Project,
        Aggregate,
        Limit,
        Sort,
        Window,
        Join,
        Union,
        Intersection,
        Difference,
        ParquetScan,
        IpcScan,
        JsonScan,
    ]

    var _v: Self.VariantType

    var _virt_to_operator: def(
        Self.VariantType, ExecContext, Bindings
    ) thin raises -> Pipeline
    """Lowering, wired per **constructed** node type. See the struct docstring
    for the 348% this one slot is worth."""

    @staticmethod
    def _to_operator_tramp[
        R: Relation
    ](
        v: Self.VariantType,
        ctx: ExecContext,
        bindings: Bindings,
    ) raises -> Pipeline:
        return v[R].to_operator(ctx, bindings)

    @implicit
    def __init__[R: Relation](out self, var value: R):
        self._v = Self.VariantType(value^)
        self._virt_to_operator = Self._to_operator_tramp[R]

    def __init__(out self, *, copy: Self):
        self._v = Self.VariantType(copy=copy._v)
        self._virt_to_operator = copy._virt_to_operator

    def isa[R: Relation](self) -> Bool:
        """Is this node an `R`? The question every rule opens with."""
        return self._v.isa[R]()

    def get[R: Relation](ref self) -> ref[self._v[R]] R:
        """This node as an `R`, borrowed. Undefined unless `isa[R]()`."""
        return self._v[R]

    def _dispatch[
        R: Movable, //, Func: def[T: Relation](T) raises -> R
    ](self, func: Func) raises -> R:
        """Run `func` on the active member, narrowed to `Relation`.

        Instantiates `func` once per member, which is why `to_operator` does
        **not** come through here. Written out rather than routed through a
        shared helper, for the reason `DynArray._dispatch` records: a narrowing
        closure between caller and ladder is inlined into every arm of every
        instantiation, measured at +662,740 bytes on one gate.
        """
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, Relation):
                if self._v.isa[T]():
                    return func(rebind[downcast[T, Relation]](self._v[T]))
        abort("DynRelation._dispatch: no arm matched")

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        """`traverse` on whichever node this is."""

        def job[T: Relation](node: T) raises {imm} -> DynRelation:
            return node.traverse(f)

        return self._dispatch(job)

    def references(self, mut into: References):
        """`references` on whichever node this is.

        Through `_dispatch`, with each arm collecting into its own `References`
        and one merge afterwards. An inline `isa` ladder writing straight into
        `into` was tried and measured 24,152 bytes against this one's 3,132 on
        `query_cli`: it inlines every node's walk into every arm."""

        def job[T: Relation](node: T) raises {imm} -> References:
            var refs = References()
            node.references(refs)
            return refs^

        try:
            into.extend(self._dispatch(job))
        except:
            abort("DynRelation.references: no arm matched")

    def params(self) -> List[ParamSpec]:
        """Every parameter this plan reads, once per name, in walk order —
        what `QueryCli` turns into a command line."""
        var refs = References()
        self.references(refs)
        return distinct_params(refs.params)

    def schema(self) -> Schema:
        def job[T: Relation](node: T) raises {imm} -> Schema:
            return node.schema()

        try:
            return self._dispatch(job)
        except:
            abort("DynRelation.schema: no arm matched")

    def estimate(self) raises -> Estimate:
        """`estimate` on whichever node this is.

        Raising rather than aborting, unlike `schema` and `references`: an
        estimate reads a predicate through `DynValue.mask`, which reads a
        source's statistics and can fail on a footer this build cannot decode.
        `schema` returns a stored field and cannot.
        """

        def job[T: Relation](node: T) raises {imm} -> Estimate:
            return node.estimate()

        return self._dispatch(job)

    def cost(self) raises -> Cost:
        """`cost` on whichever node this is — the subtree's cost, not this
        node's own, since every node charges its children before itself."""

        def job[T: Relation](node: T) raises {imm} -> Cost:
            return node.cost()

        return self._dispatch(job)

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        return self._virt_to_operator(self._v, ctx, bindings)

    def write_to[W: Writer](self, mut writer: W):
        def job[T: Relation](node: T) raises {imm} -> String:
            return String(node)

        try:
            writer.write(self._dispatch(job))
        except:
            writer.write("<relation>")

    # -- the plan-building API ----------------------------------------------
    #
    # These are the surface a caller actually writes. Every one returns a
    # `DynRelation`, so plans compose left to right —
    # `t.filter(...).aggregate(...).sort_by(...)` — and no caller ever names a
    # node type or wraps anything in `DynRelation` by hand.
    #
    # They live on the box rather than on `Relation` because that is where
    # composition happens: a verb needs a *boxed* input to build its node with,
    # and `self` already is one. Putting them on the trait would make every
    # conformer implement eight methods it does not care about.

    def filter(self, var predicate: DynValue) raises -> DynRelation:
        """Rows where `predicate` is true. Schema-preserving.

        The erased overload. It still prunes — `mask` is a slot on `DynValue`,
        so the box carries it like any other method — but `constant` and
        `conjuncts` are left unanswered, because those are analysis only the
        concrete type can do.
        """
        return Filter(self.copy(), predicate^)

    def filter[V: Value](self, var predicate: V) raises -> DynRelation:
        """Rows where `predicate` is true, **with the predicate analysed**.

        `constant_bool` and `conjuncts` are read here because this is the last
        place the concrete type is visible, and both are decisions a box cannot
        make: whether the predicate folds to a constant, and how it splits on
        `AND` with each conjunct still fused. The two overloads are disjoint —
        `DynValue` does not conform to the traits it erases — which is what
        lets both spellings coexist.
        """
        return Filter(
            self.copy(),
            DynValue(predicate.copy()),
            predicate.constant_bool(),
            predicate.conjuncts(),
        )

    def select(self, names: List[String]) raises -> DynRelation:
        """Keep these columns, in this order.

        Sugar over `project`: the values are `FieldRef`s resolved against the
        input schema, so this needs no dtype from the caller and links no
        lane. A name the input lacks raises here.
        """
        var input_schema = self.schema()
        var values = List[DynValue](capacity=len(names))
        for ref n in names:
            values.append(FieldRef(input_schema.field(name=n).copy()))
        return Project(self.copy(), names.copy(), values^)

    def select(self, *names: String) raises -> DynRelation:
        """`select("a", "b")` — the same verb without the brackets.

        Kept as a second overload rather than replacing the list form: the
        golden corpus's Python twin spells it `select(*names)` and its Mojo
        twin spelled it `select("ts", "label")` until the port rewrote every
        case to a list, so one of the two lanes has to grow the other's
        spelling for the corpus to stay one text. A `VariadicListMem` cannot
        be forwarded to another function's variadic parameter (CLAUDE.md), so
        this copies into a `List` and delegates rather than the list overload
        delegating here.
        """
        var owned = List[String](capacity=len(names))
        for ref n in names:
            owned.append(n.copy())
        return self.select(owned)

    def project(
        self, var names: List[String], var values: List[DynValue]
    ) raises -> DynRelation:
        """`SELECT <values> AS <names>` — new columns over the same rows."""
        return Project(self.copy(), names^, values^)

    def with_columns(
        self, var names: List[String], var values: List[DynValue]
    ) raises -> DynRelation:
        """`SELECT *, <values> AS <names>` — append to the existing columns.

        A per-row value whose name already exists **replaces it in place**
        rather than being appended, which is Polars' `with_columns` rule and
        the only one that keeps the output schema free of duplicates. Position
        is preserved: replacing `qty` leaves `qty` where it was. The per-row
        values are one `Project`; the surviving columns are `FieldRef`s, so
        no caller has to supply their dtypes.

        **Window values come after**, each in a `Window` node of its own,
        stacked in the order given: a window value may read a column a per-row
        value in the same call adds, never the reverse. A window value cannot
        replace a column, because a `Window` node only appends. One node per
        value is the literal reading; `MergeWindows` folds values sharing a
        window into one node, so they share a sort.
        """
        if len(names) != len(values):
            raise InvalidError(
                t"with_columns: {len(names)} names but {len(values)} values"
            )
        # **A name may not appear twice in one call.** Replacing in place
        # keeps the last match, so the first value would be silently
        # discarded, and for a name not already present both would be
        # appended and the second made unreachable by `get_field_index`.
        for i in range(len(names)):
            for j in range(i):
                if names[j] == names[i]:
                    raise InvalidError(
                        t"with_columns: '{names[i]}' is named twice"
                    )
        var input_schema = self.schema()
        var row_names = List[String]()
        var row_values = List[DynValue]()
        for i in range(len(values)):
            if values[i].windowed():
                if input_schema.get_field_index(names[i]) != -1:
                    raise InvalidError(
                        t"with_columns: '{names[i]}' already exists; a "
                        t"window column cannot replace one"
                    )
            else:
                row_names.append(names[i].copy())
                row_values.append(values[i].copy())
        var current = self.copy()
        if len(row_values) > 0:
            var out_names = List[String]()
            var out_values = List[DynValue]()
            for ref f in input_schema.fields:
                var replaced = -1
                for i in range(len(row_names)):
                    if row_names[i] == f.name:
                        replaced = i
                out_names.append(f.name.copy())
                if replaced >= 0:
                    out_values.append(row_values[replaced].copy())
                else:
                    out_values.append(FieldRef(f.copy()))
            for i in range(len(row_names)):
                if input_schema.get_field_index(row_names[i]) == -1:
                    out_names.append(row_names[i].copy())
                    out_values.append(row_values[i].copy())
            current = Project(current^, out_names^, out_values^)
        for i in range(len(values)):
            if values[i].windowed():
                current = Window(
                    current^, [names[i].copy()], [values[i].copy()]
                )
        return current^

    def drop(self, names: List[String]) raises -> DynRelation:
        """`SELECT <everything except names>` — say what goes, not what stays.

        The survivors keep their **input order**, which is what distinguishes
        this from spelling out the complement with `select`: a caller who
        writes the complement by hand has to know the order too, and gets it
        wrong the moment a column is added upstream.

        A name that is not in the schema raises rather than being ignored. A
        typo in a `drop` list is otherwise silent — the column it meant to
        remove survives — and that is the failure mode this verb exists to
        avoid.
        """
        var input_schema = self.schema()
        for ref n in names:
            if input_schema.get_field_index(n) == -1:
                raise KeyError(t"drop: column '{n}' not found in schema")
        var out_names = List[String]()
        var out_values = List[DynValue]()
        for ref f in input_schema.fields:
            var dropped = False
            for ref n in names:
                if n == f.name:
                    dropped = True
            if not dropped:
                out_names.append(f.name.copy())
                out_values.append(FieldRef(f.copy()))
        return Project(self.copy(), out_names^, out_values^)

    def rename(
        self, names: List[String], new_names: List[String]
    ) raises -> DynRelation:
        """Rename `names[i]` to `new_names[i]`, keeping every other column.

        Two parallel lists rather than a mapping because Mojo has no dict
        literal in argument position; `golden/helpers.py` adapts the Python
        frontend's dict spelling to this one for exactly that reason.

        The renamed column keeps its **source `Field`** — dtype, `nullable`
        and metadata — because `Project._output_schema` carries a bare column's
        field over whole. Rebuilding from the dtype alone would silently turn
        `nullable=False` into `True`, which is the divergence that method
        exists to fix.
        """
        if len(names) != len(new_names):
            raise InvalidError(
                t"rename: {len(names)} names but {len(new_names)} new names"
            )
        var input_schema = self.schema()
        for ref n in names:
            if input_schema.get_field_index(n) == -1:
                raise KeyError(t"rename: column '{n}' not found in schema")
        var out_names = List[String]()
        var out_values = List[DynValue]()
        for ref f in input_schema.fields:
            var renamed = -1
            for i in range(len(names)):
                if names[i] == f.name:
                    renamed = i
            if renamed >= 0:
                out_names.append(new_names[renamed].copy())
            else:
                out_names.append(f.name.copy())
            out_values.append(FieldRef(f.copy()))
        return Project(self.copy(), out_names^, out_values^)

    def limit(self, length: Int, offset: Int = 0) raises -> DynRelation:
        """`OFFSET offset LIMIT length`."""
        return Limit(self.copy(), offset, length)

    def sort_by(
        self,
        var keys: List[DynValue],
        var ascending: List[Bool],
        nulls_first: Bool = True,
    ) raises -> DynRelation:
        """`ORDER BY` — a pipeline breaker, so it buffers and sorts at the
        end."""
        return Sort(self.copy(), keys^, ascending^, nulls_first)

    def aggregate(self, var aggs: List[DynValue]) raises -> DynRelation:
        """`SELECT <aggs> ...` — a whole-table aggregate, one implicit group.

        Not a special node: the `Aggregate` a key list builds, with no keys.
        """
        # Its own overload: the key-list one names a key encoder, which a plan
        # that never groups should not link.
        return Aggregate(
            self.copy(),
            List[DynValue](),
            aggs^,
            lowering=UngroupedAggregateOperator.append_to,
        )

    def aggregate(
        self, var aggs: List[DynValue], var keys: List[DynValue]
    ) raises -> DynRelation:
        """`SELECT <keys>, <aggs> ... GROUP BY <keys>`. Aggregates come first
        because they are the part a caller always supplies; an empty `keys` is
        the whole-table aggregate.

        The keys are encoded by `DictionaryEncoder`, which dispatches on each
        key's runtime dtype.
        """
        return Aggregate(self.copy(), keys^, aggs^)

    def join(
        self,
        var right: DynRelation,
        var left_keys: List[Int],
        var right_keys: List[Int],
        kind: JoinKind = JOIN_INNER,
        build_side: JoinBuildSide = BUILD_LEFT,
    ) raises -> DynRelation:
        """Equijoin. The output is `self`'s columns then `right`'s.

        `build_side` names the side to index, `BUILD_LEFT` (`self`) by
        default. It changes the cost and nothing else, and `SelectBuildSide`
        may change it.
        """
        return Join(
            self.copy(),
            right^,
            left_keys^,
            right_keys^,
            kind,
            build_side=build_side,
        )

    def distinct(self) raises -> DynRelation:
        """`SELECT DISTINCT *` — one row per distinct row, NULL equal to
        itself.

        An aggregate keyed by every column with no aggregates, the plan SQL's
        `SELECT DISTINCT` has always built; there is no `Distinct` node
        because it would lower to exactly that `GroupedAggregateOperator`.
        """
        var keys = List[DynValue]()
        for ref f in self.schema().fields:
            keys.append(FieldRef(f.copy()))
        return self.aggregate(List[DynValue](), keys^)

    def union_all(self, var right: DynRelation) raises -> DynRelation:
        """Every row of both sides, matched by position; see `Union`."""
        return Union(self.copy(), right^)

    def union(self, var right: DynRelation) raises -> DynRelation:
        """The distinct rows of both sides: `distinct()` over `union_all`, so
        the dedup is the `Aggregate` it is."""
        return self.union_all(right^).distinct()

    def except_(self, var right: DynRelation) raises -> DynRelation:
        """Distinct rows of `self` that `right` lacks; see `Difference`.
        Trailing underscore because `except` is a keyword."""
        return Difference(self.copy(), right^, False)

    def except_all(self, var right: DynRelation) raises -> DynRelation:
        return Difference(self.copy(), right^, True)

    def intersect(self, var right: DynRelation) raises -> DynRelation:
        """Distinct rows present on both sides; see `Intersection`."""
        return Intersection(self.copy(), right^, False)

    def intersect_all(self, var right: DynRelation) raises -> DynRelation:
        return Intersection(self.copy(), right^, True)

    def optimize[R: RuleSet](self) raises -> DynRelation:
        """This plan, rewritten by `R` until nothing changes.

        Returns an ordinary plan, so the result prints, composes and executes
        like any other and can be diffed against its input. `execute()` alone
        optimizes nothing, and a binary links exactly the rules it names.
        """
        return optimize[R](self)

    def execute(
        self,
        ctx: ExecContext = ExecContext.auto(),
        bindings: Bindings = Bindings(),
    ) raises -> RecordBatch:
        """Run this plan and drain it into one batch."""
        var p = self.to_operator(ctx, bindings)
        # The shim: operators work in struct arrays, the public API hands back
        # a batch. Cheap — children move, schema comes off the struct dtype.
        return RecordBatch.from_struct_array(p.collect(self.schema()))


struct EmptyRelation(Relation, Writable):
    """Zero rows, with a schema. What a plan collapses to when it provably
    returns nothing.

    Exists so a rule never has to answer "no relation". `optimize` returns a
    plan, always; a rewrite that proves a subtree empty replaces it with this
    rather than with an `Optional` that every caller then has to unwrap. It
    also keeps the schema, so everything above it still type-checks and still
    reports the right columns for an empty result.

    Its operator emits nothing and reports `done` immediately, which is what
    lets a `LIMIT 0` stop a scan before it reads a byte.
    """

    var batch: RecordBatch
    """A zero-row batch of the right schema.

    Holding a batch rather than a bare `Schema` means this reuses
    `BatchSourceOperator` unchanged instead of introducing a second kind of
    source that yields nothing — one fewer operator, and the empty case travels
    exactly the code path the non-empty one does."""

    def __init__(out self, var batch: RecordBatch):
        self.batch = batch^

    def references(self, mut into: References):
        pass

    def schema(self) -> Schema:
        return self.batch.schema.copy()

    def estimate(self) raises -> Estimate:
        """Exactly zero rows, and so exactly zero of everything: a rewrite
        proved this subtree empty."""
        return Estimate.empty(self.batch.schema)

    def cost(self) raises -> Cost:
        """Free. Its operator reports `done` before emitting a row, which is
        what lets a `LIMIT 0` stop a scan before it reads a byte."""
        return Cost()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        return Pipeline(BatchSourceOperator(self.batch.to_struct_array()))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Empty(", len(self.batch.schema), " cols)")


struct InMemoryTable(Relation, Writable):
    """A batch already in memory, as a source."""

    var batch: RecordBatch

    def __init__(out self, var batch: RecordBatch):
        self.batch = batch^

    def references(self, mut into: References):
        pass

    def schema(self) -> Schema:
        return self.batch.schema.copy()

    def estimate(self) raises -> Estimate:
        """The exact row and null counts, which the batch stores. Bounds and
        distinct counts stay unknown, because computing them scans the data.
        """
        var cols = List[ColumnEstimate](capacity=self.batch.num_columns())
        for i in range(self.batch.num_columns()):
            ref f = self.batch.schema.fields[i]
            cols.append(
                ColumnEstimate(
                    f.name.copy(),
                    nulls=Approx.exact(self.batch.column(i).null_count()),
                    width=ColumnEstimate.width_of(f.dtype),
                )
            )
        return Estimate(Approx.exact(self.batch.num_rows()), cols^)

    def cost(self) raises -> Cost:
        """The rows it hands on, and the bytes they occupy.

        Charged the same as a scan even though the data is already in memory:
        the two are interchangeable as plan inputs, and pricing an in-memory
        source at zero would make every rewrite that materialises one look
        free.
        """
        var estimate = self.estimate()
        return Cost.source(estimate.rows, estimate.row_width())

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        """The one relation that *creates* a pipeline; every other appends."""
        return Pipeline(BatchSourceOperator(self.batch.to_struct_array()))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("InMemoryTable(", self.batch.num_rows(), " rows)")


struct Filter(Relation, Writable):
    """Rows of `input` where `predicate` is true.

    Schema-preserving: a filter changes which rows survive, never which columns
    exist. That is why it can hold its input's schema rather than computing
    one, and it is also why a filter cannot narrow the columns it compacts —
    knowing what is read downstream is a *physical* property this layer does
    not have.
    """

    var input: ArcPointer[DynRelation]
    var predicate: DynValue

    var conjuncts: List[DynValue]
    """The predicate split on `AND`, decided at the verb.

    Empty when the predicate arrived already boxed, which reads as "not
    split" — the `predicate` field is what actually filters either way, so an
    empty list costs an optimization and never an answer."""

    var constant: Optional[Bool]
    """Whether the predicate is a constant, decided at the verb.

    `EliminateFilter` reads this. `Optional` because the erased overload cannot
    answer, and a `None` costs only an optimization."""

    def __init__(
        out self,
        var input: DynRelation,
        var predicate: DynValue,
        constant: Optional[Bool] = None,
        var conjuncts: List[DynValue] = [],
    ) raises:
        require_per_row(
            predicate,
            "filter",
            predicate.name(),
            "put it in .aggregate() and filter the result (HAVING)",
        )
        self.input = ArcPointer(input^)
        self.predicate = predicate^
        self.constant = constant
        self.conjuncts = conjuncts^

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        return self.with_input(f(self.input[]))

    def with_input(self, var input: DynRelation) raises -> Filter:
        """This filter over a different input, carrying everything else.

        **The only way a rule should move a filter.** A `Filter` holds four
        things — input, predicate, constant, conjuncts — and the last two are
        analysis decided at the verb, where the predicate's concrete type was
        still visible. A rule that rebuilds with
        `Filter(new_input, predicate)` silently drops them,
        which does not fail: the filter still filters, `EliminateFilter` and
        `SplitConjunction` just stop firing. That is exactly what happened when
        `constant` and `conjuncts` were added and six call sites were not
        updated.
        """
        return Filter(
            input^,
            self.predicate.copy(),
            self.constant,
            self.conjuncts.copy(),
        )

    def references(self, mut into: References):
        self.input[].references(into)
        self.predicate.references(into)

    def schema(self) -> Schema:
        return self.input[].schema()

    def estimate(self) raises -> Estimate:
        """The input's rows, reduced by what the predicate can be shown to do.

        Selectivity is `Value.mask` over a one-chunk index built from the
        input's estimate, so it uses the comparisons a scan prunes with. A
        `false` bit proves no row can match, and the answer is exactly zero;
        otherwise it is `DEFAULT_SELECTIVITY`, an estimate that never reaches
        zero. A predicate folded to a constant needs no index, and one that
        cannot be evaluated, such as one naming an unbound parameter, falls
        back to the default.
        """
        var input = self.input[].estimate()
        if self.constant:
            if self.constant.value():
                return input^
            return input.filtered(Approx.exact(0))

        var index = input.to_index()
        var live = keep_every(index.chunks)
        try:
            live = self.predicate.mask(index)
        except:
            pass
        if len(live) == 1 and not live.is_null(0) and not live[0].value():
            return input.filtered(Approx.exact(0))
        return input.filtered(input.rows.scaled(DEFAULT_SELECTIVITY))

    def cost(self) raises -> Cost:
        """Its input's cost, plus one evaluation per row arriving.

        The predicate is charged against the *input's* cardinality and not its
        own output's, which is the point of pushing a filter down: the work is
        what it reads, and the saving is what everything above it no longer
        reads.
        """
        return self.input[].cost() + Cost.per_row(self.input[].estimate().rows)

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        var pipe = self.input[].to_operator(ctx, bindings)
        pipe.append(
            FilterOperator(
                self.predicate.to_operator(
                    self.input[].schema(), False, bindings
                ),
                ctx.copy(),
            )
        )
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Filter(", self.input[], ", ", self.predicate, ")")


struct Project(Relation, Writable):
    """`SELECT <values> AS <names>` — a new set of columns over the same rows.

    This is the node that gives `Analyzable.dtype` and `Analyzable.name` their
    callers: the output schema is one field per value, and a field needs both a
    type and a name.
    """

    var input: ArcPointer[DynRelation]
    var names: List[String]
    var values: List[DynValue]
    var _schema: Schema

    def __init__(
        out self,
        var input: DynRelation,
        var names: List[String],
        var values: List[DynValue],
    ) raises:
        if len(names) != len(values):
            raise InvalidError(
                t"project: {len(names)} names but {len(values)} values"
            )
        for i in range(len(values)):
            require_per_row(
                values[i], "project", names[i], "use .aggregate() instead"
            )
        self._schema = Self._output_schema(input.schema(), names, values)
        self.input = ArcPointer(input^)
        self.names = names^
        self.values = values^

    @staticmethod
    def _output_schema(
        input: Schema, names: List[String], values: List[DynValue]
    ) raises -> Schema:
        """One field per value, computed once at construction.

        A value that is **exactly a bare column** carries its source `Field`
        over whole — dtype, `nullable` and metadata — rather than being
        rebuilt from its dtype alone. Rebuilding loses `nullable`, so
        projecting a column produced a *different* schema for it than
        selecting the same column did; the previous expression package records
        that as a real
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

    def passes_through(self, name: String) -> Bool:
        """Does column `name` reach this projection's input untouched?

        True only when some output is exactly that column: it reads one column,
        that column is the name it is emitted as, and it is not an aggregate.
        A **rename** reads one column too, which is why the name is compared
        and not just the arity — pushing a predicate past a rename would have
        it name a column that does not exist below.
        """
        for i in range(len(self.values)):
            if self.names[i] == name:
                ref v = self.values[i]
                if v.aggregates():
                    return False
                var cols = v.columns()
                return len(cols) == 1 and cols[0] == name and v.name() == name
        return False

    def passes_through_all(self, names: List[String]) -> Bool:
        """Do all of `names` reach the input untouched?"""
        for ref n in names:
            if not self.passes_through(n):
                return False
        return True

    def computes_an_aggregate(self) -> Bool:
        """Does any projected value collapse its input to one row?

        `Project` rejects aggregates at construction, so this answers `False`
        today; it is asked anyway by the rule that moves a `Limit` below a
        projection, because that rewrite is only sound for a row-preserving
        node and should not depend on a constructor check staying in place.
        """
        for ref v in self.values:
            if v.aggregates():
                return True
        return False

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        return Project(f(self.input[]), self.names.copy(), self.values.copy())

    def estimate(self) raises -> Estimate:
        """The input's rows unchanged. A column that `passes_through` keeps
        its summary; any other output, a rename included, is new and keeps
        only the width of its declared dtype, so this and predicate pushdown
        agree on what a rename is.
        """
        var input = self.input[].estimate()
        var cols = List[ColumnEstimate](capacity=len(self._schema.fields))
        for i in range(len(self._schema.fields)):
            ref f = self._schema.fields[i]
            var j = -1
            if self.passes_through(f.name):
                j = input.index_of(f.name)
            if j >= 0:
                cols.append(input.columns[j].copy())
            else:
                cols.append(ColumnEstimate.unknown(f.name.copy(), f.dtype))
        return Estimate(input.rows, cols^)

    def cost(self) raises -> Cost:
        """One evaluation per value per row — a projection of fifty columns is
        not a projection of one, and a cost model that says otherwise cannot
        prefer the narrower of two equivalent plans."""
        return self.input[].cost() + Cost.per_row(
            self.input[].estimate().rows.times(len(self.values))
        )

    def references(self, mut into: References):
        self.input[].references(into)
        for ref v in self.values:
            v.references(into)

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        var pipe = self.input[].to_operator(ctx, bindings)
        var values = List[DynOperator](capacity=len(self.values))
        for ref v in self.values:
            values.append(v.to_operator(self.input[].schema(), False, bindings))
        pipe.append(ProjectOperator(values^, self._schema.copy()))
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Project(", self.input[], ", ")
        for i in range(len(self.names)):
            if i > 0:
                writer.write(", ")
            writer.write(self.names[i], "=", self.values[i])
        writer.write(")")


struct Aggregate(Relation, Writable):
    """`SELECT <keys>, <aggs> FROM input GROUP BY <keys>`.

    The output schema is the key fields followed by the aggregate fields, in
    that order. Everything downstream depends on that ordering — the
    operator reads its key fields back off the front of it, and a `Filter`
    above this node is exactly `HAVING`.

    An empty `keys` is **not** a different node: it is `SELECT sum(x) FROM t`,
    one implicit group. The only thing it changes is which fold each aggregate
    starts, and that is decided here, at plan-build time, because it is known
    here. `Value.to_operator(grouped)` picks one of two loops compiled out of
    one struct, and running the grouped one over a single group measured 14.6x
    worse — a runtime branch could not have made that choice.

    **Whether the keys are encoded is fixed when the node is built**, the way
    `DynRelation` fixes its lowering and `Datum` its broadcast: `lowering`
    appends a `GroupedAggregateOperator`, which encodes the keys with a
    `DictionaryEncoder`, or an `UngroupedAggregateOperator` when there are
    none, so a plan that never groups builds no encoder. Everything else — the
    schema, `HAVING` above it, the optimizer's rules — reads the keys as the
    values they are.
    """

    var input: ArcPointer[DynRelation]
    var keys: List[DynValue]
    var aggs: List[DynValue]
    var _schema: Schema
    var _lowering: AggregateLowering
    """The grouping stage over the lowered keys and folds — see the struct
    docstring."""

    def __init__(
        out self,
        var input: DynRelation,
        var keys: List[DynValue],
        var aggs: List[DynValue],
    ) raises:
        """Keys of any types, encoded by `DictionaryEncoder` — or none, one
        implicit group."""
        if len(keys) == 0:
            self = Self(
                input^,
                keys^,
                aggs^,
                lowering=UngroupedAggregateOperator.append_to,
            )
        else:
            self = Self(
                input^,
                keys^,
                aggs^,
                lowering=GroupedAggregateOperator.append_to,
            )

    def __init__(
        out self,
        var input: DynRelation,
        var keys: List[DynValue],
        var aggs: List[DynValue],
        *,
        lowering: AggregateLowering,
    ) raises:
        """Keys encoded by the operator `lowering` appends."""
        for ref k in keys:
            require_per_row(
                k,
                "aggregate",
                k.name(),
                "group by a column or a per-row expression, not an aggregate",
            )
        for ref a in aggs:
            if a.windowed():
                raise InvalidError(
                    "aggregate: a window function cannot be aggregated; add it"
                    " with `with_columns`, then aggregate the column"
                )
        self._schema = Self._output_schema(input.schema(), keys, aggs)
        self.input = ArcPointer(input^)
        self.keys = keys^
        self.aggs = aggs^
        self._lowering = lowering

    def with_input(self, var input: DynRelation) raises -> Aggregate:
        """This aggregate over `input` instead — keys, aggregates and encoder
        unchanged. How a rewrite moves a node without dropping the encoder its
        verb chose."""
        return Aggregate(
            input^, self.keys.copy(), self.aggs.copy(), lowering=self._lowering
        )

    @staticmethod
    def _output_schema(
        input: Schema, keys: List[DynValue], aggs: List[DynValue]
    ) raises -> Schema:
        """Keys first, then aggregates, computed once at construction.

        A key that is a bare column keeps its own name; anything computed has
        none and is called `key0`, `key1`, … by position. That rule is not
        cosmetic: the previous expression package shipped a defect where one
        lane answered `d` and the
        other `key0` for the same `GROUP BY d`, so one query had two output
        schemas depending on which lane built it.
        """
        var fields = List[Field](capacity=len(keys) + len(aggs))
        for i in range(len(keys)):
            ref k = keys[i]
            var name = k.name()
            if name == "":
                name = "key" + String(i)
            fields.append(field(name^, k.dtype(input)))
        for ref a in aggs:
            fields.append(field(a.name(), a.dtype(input)))
        return schema(fields^)

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        return self.with_input(f(self.input[]))

    def estimate(self) raises -> Estimate:
        """One row per distinct key combination; exactly one when there are no
        keys. The formula and its honesty are in `estimates.grouped`.

        A key is named by `DynValue.name()`, which is the column's own name
        for a bare column and empty for anything computed — so a computed key
        finds no summary and the group count goes unknown. That is the right
        answer: nothing here knows how many distinct values `a + b` takes, and
        `grouped` will not invent one.
        """
        var keys = List[String](capacity=len(self.keys))
        for ref k in self.keys:
            keys.append(k.name())
        return self.input[].estimate().grouped(keys, self._schema)

    def cost(self) raises -> Cost:
        """Hash every input row, then hold one entry per group.

        Two terms because they scale differently and a plan should be able to
        tell them apart: the probe is paid per *input* row and the table is
        paid per *output* group, which is why a high-cardinality group-by is
        expensive in a way a low-cardinality one is not.
        """
        var input = self.input[].estimate()
        var output = self.estimate()
        return (
            self.input[].cost()
            + Cost.hash_probe(input.rows)
            + Cost.hash_build(output.rows, output.row_width())
        )

    def references(self, mut into: References):
        self.input[].references(into)
        for ref k in self.keys:
            k.references(into)
        for ref a in self.aggs:
            a.references(into)

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        var grouped = len(self.keys) > 0
        var folds = List[DynOperator](capacity=len(self.aggs))
        for ref a in self.aggs:
            folds.append(
                a.to_operator(self.input[].schema(), grouped, bindings)
            )
        var pipe = self.input[].to_operator(ctx, bindings)
        var keys = List[DynOperator](capacity=len(self.keys))
        for ref k in self.keys:
            keys.append(k.to_operator(self.input[].schema(), False, bindings))
        self._lowering(pipe, keys^, folds^, self._schema.copy(), ctx.copy())
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Aggregate(", self.input[])
        for ref k in self.keys:
            writer.write(", by=", k)
        for ref a in self.aggs:
            writer.write(", ", a)
        writer.write(")")


struct Limit(Relation, Writable):
    """`OFFSET n LIMIT m` — schema-preserving and streaming.

    Reads no column of its own, so it neither adds nor removes fields. The
    operator it builds reports `done` once it has its rows, which is what stops
    the source: in a push engine nothing downstream can otherwise halt a scan.
    """

    var input: ArcPointer[DynRelation]
    var offset: Int
    var length: Int

    def __init__(out self, var input: DynRelation, offset: Int, length: Int):
        self.input = ArcPointer(input^)
        self.offset = offset
        self.length = length

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        return Limit(f(self.input[]), self.offset, self.length)

    def references(self, mut into: References):
        self.input[].references(into)

    def schema(self) -> Schema:
        return self.input[].schema()

    def estimate(self) raises -> Estimate:
        """`min(length, rows - offset)`: arithmetic on a cardinality rather
        than a guess, so an exact input gives an exact answer.
        """
        return self.input[].estimate().limited(self.offset, self.length)

    def cost(self) raises -> Cost:
        """Charged on what it emits. The input is still charged in full,
        although `LimitOperator` stops its source early: modelling that needs
        the bound pushed down while costing, so the model undervalues a limit
        rather than overvaluing it.
        """
        return self.input[].cost() + Cost.per_row(self.estimate().rows)

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        var pipe = self.input[].to_operator(ctx, bindings)
        pipe.append(LimitOperator(self.offset, self.length))
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Limit(",
            self.input[],
            ", offset=",
            self.offset,
            ", length=",
            self.length,
            ")",
        )


struct Sort(Relation, Writable):
    """`ORDER BY` — schema-preserving, and a pipeline breaker.

    Sorting is blocking by nature: no prefix of the input determines the first
    output row, so the operator buffers every morsel and orders once at
    `drain`. That the engine expresses this with the same methods a filter uses
    is the point of the push interface.
    """

    var input: ArcPointer[DynRelation]
    var keys: List[DynValue]
    var ascending: List[Bool]
    var nulls_first: Bool

    var limit: Optional[Int]
    """The TopN bound — how many ordered rows the consumer actually needs.

    `None` means "order everything", which is what every `Sort` says until the
    `TopN` rule rewrites one. It lives here rather than being discovered at
    execution because only the plan knows what sits above: a `Limit` directly
    above reads as a bound, a `Filter` between them does not — the filter runs
    *after* the sort, so a k-row sort would feed it fewer than k rows and the
    query would silently return too few. `optimizer.mojo` owns that
    distinction."""

    def __init__(
        out self,
        var input: DynRelation,
        var keys: List[DynValue],
        var ascending: List[Bool],
        nulls_first: Bool = True,
        limit: Optional[Int] = None,
    ) raises:
        if len(keys) != len(ascending):
            raise InvalidError(
                t"sort: {len(keys)} keys but {len(ascending)} directions"
            )
        if len(keys) == 0:
            raise InvalidError("sort: needs at least one key")
        for ref k in keys:
            require_per_row(
                k,
                "sort",
                k.name(),
                "aggregate first, then sort the result",
            )
        self.input = ArcPointer(input^)
        self.keys = keys^
        self.ascending = ascending^
        self.nulls_first = nulls_first
        self.limit = limit

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        return Sort(
            f(self.input[]),
            self.keys.copy(),
            self.ascending.copy(),
            self.nulls_first,
            self.limit,
        )

    def references(self, mut into: References):
        self.input[].references(into)
        for ref k in self.keys:
            k.references(into)

    def schema(self) -> Schema:
        return self.input[].schema()

    def estimate(self) raises -> Estimate:
        """The input's estimate, unchanged — except for a `TopN` bound.

        Ordering moves rows and creates none, so every count and every bound
        survives it exactly. `limit` is a `Limit` this node absorbed, so it is
        applied as one rather than as a second rule.
        """
        var input = self.input[].estimate()
        if self.limit:
            return input.limited(0, self.limit.value())
        return input^

    def cost(self) raises -> Cost:
        """`n log n` comparisons over the whole input, all of it held.

        Charged against the **input**, not the `TopN` bound: the bound changes
        how much comes out, and `SortOperator` still has to see every row to
        know which ones they are.
        """
        var input = self.input[].estimate()
        return self.input[].cost() + Cost.sort(input.rows, input.row_width())

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        var pipe = self.input[].to_operator(ctx, bindings)

        # The key operators are built **once**, here, where the plan becomes
        # physical -- not per batch. Anything a key needs to resolve or cache
        # before the first row arrives has a place to live now.
        var input_schema = self.input[].schema()
        var keys = List[DynOperator](capacity=len(self.keys))
        for ref k in self.keys:
            keys.append(k.to_operator(input_schema, False, bindings))

        pipe.append(
            SortOperator(
                keys^,
                self.ascending.copy(),
                self.nulls_first,
                self.limit,
                ctx.copy(),
            )
        )
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Sort(", self.input[])
        if self.limit:
            writer.write(" top ", self.limit.value())
        for i in range(len(self.keys)):
            writer.write(", ", self.keys[i])
            writer.write(" asc" if self.ascending[i] else " desc")
        writer.write(")")


struct Window(Relation, Writable):
    """`OVER (...)` — window columns appended to the input's own rows.

    A node of its own rather than a shape `Project` could carry, because a
    window value is not a per-row value: `Project` evaluates each of its values
    against the batch in front of it, and a window value's answer depends on
    rows that may never share a batch with it. `ProjectOperator` has nowhere
    to put the buffering that needs, and `require_per_row` keeps window values
    out of that position.

    **This node only appends.** The `with_columns` rule that a repeated name
    replaces in place is a *verb's* rule; expressing it here would mean the
    node deciding column order too. So this node's schema is simply the
    input's fields followed by one per value.

    **A node sorts once per distinct window spec among its values**, and its
    values never read each other's output. `with_columns` builds a node per
    value; `MergeWindows` folds a value into the node below it when it reads
    nothing that node adds, which is what lets two values over one window
    share a sort.
    """

    var input: ArcPointer[DynRelation]
    var names: List[String]
    var values: List[DynValue]
    """Window values, each with its own spec, function and frame."""
    var _schema: Schema

    def __init__(
        out self,
        var input: DynRelation,
        var names: List[String],
        var values: List[DynValue],
    ) raises:
        if len(names) != len(values):
            raise InvalidError(
                t"window: {len(names)} names but {len(values)} values"
            )
        if len(values) == 0:
            raise InvalidError("window: needs at least one value")
        for ref v in values:
            if not v.windowed():
                var rendered = String(v)
                raise InvalidError(
                    t"window: '{rendered}' is not a window value; give it a "
                    t"window with `.over()`"
                )
        self._schema = Self._output_schema(input.schema(), names, values)
        self.input = ArcPointer(input^)
        self.names = names^
        self.values = values^

    @staticmethod
    def _output_schema(
        input: Schema, names: List[String], values: List[DynValue]
    ) raises -> Schema:
        """The input's fields, then one per value.

        Every appended field is `nullable`, and that is a property of window
        functions rather than of the argument: `LAG` is null at a partition's
        first row and `LEAD` at its last however non-nullable the column it
        reads. Only the ranking functions are total, and giving them a
        narrower field would make the schema depend on which function was
        named for no gain a reader could use.
        """
        var fields = List[Field](capacity=len(input.fields) + len(values))
        for ref f in input.fields:
            fields.append(f.copy())
        for i in range(len(values)):
            fields.append(field(names[i].copy(), values[i].dtype(input)))
        return schema(fields^)

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        return Window(f(self.input[]), self.names.copy(), self.values.copy())

    def estimate(self) raises -> Estimate:
        """Every input row, plus one unsummarised column per expression.

        A window function appends; it never filters. So the row count and
        every input column's summary carry over untouched, and each appended
        column answers unknown-but-for-its-width — a rank, a lag and a framed
        sum are all values nothing has described.
        """
        return self.input[].estimate().carried(self._schema)

    def cost(self) raises -> Cost:
        """A sort, because that is what it does.

        `WindowOperator` buffers and orders by the partition and order keys
        before it can evaluate anything, so this is priced identically to
        `Sort` — a window is not a cheaper way to order.
        """
        var input = self.input[].estimate()
        return self.input[].cost() + Cost.sort(input.rows, input.row_width())

    def references(self, mut into: References):
        self.input[].references(into)
        for ref v in self.values:
            v.references(into)

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        # **No predicate may reach a scan through this node.** A window
        # function reads its whole partition, so pruning row groups below it
        # would have every rank, row number and running total computed over the
        # wrong population -- a wrong answer, not an error.
        #
        # Nothing here enforces that any more, and nothing needs to:
        # `PushFilterIntoScan` is the only way a predicate reaches a scan, and
        # it descends through `Filter` alone. `Window` is not a `Filter`, so a
        # predicate above one cannot pass. `test_window.mojo` pins it.
        var pipe = self.input[].to_operator(ctx, bindings)
        pipe.append(
            WindowOperator(
                self.values.copy(),
                self.input[].schema(),
                self._schema.copy(),
                bindings.copy(),
                ctx.copy(),
            )
        )
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Window(", self.input[])
        for i in range(len(self.values)):
            writer.write(", ", self.values[i], " as ", self.names[i])
        writer.write(")")


struct Join(Relation, Writable):
    """An equijoin over two sub-plans.

    The first node with two inputs (the set relations are the others), and
    the reason `Pipeline` had to be an `Operator`: the build side is a whole
    plan, and it is handed to the operator as an ordinary boxed stage. Before
    that, a chain of stages was a different kind of thing from a stage, and
    there was nowhere to put a second one.

    `left` and `right` say what the answer is and `build_side` what it costs:
    the output is the left side's columns then the right side's, whichever
    side is indexed.
    """

    var left: ArcPointer[DynRelation]
    var right: ArcPointer[DynRelation]
    var left_keys: List[String]
    var right_keys: List[String]
    """Join keys by **name**, resolved from the caller's indices once, here.

    The public verb still takes indices — `plan.mojo` and every existing caller
    pass them — but a plan node must not *store* them. An index is a position
    in a child's schema, so any rewrite that changes a child silently rebinds
    the join to different columns: projection pushdown narrowing a scan is
    exactly such a rewrite, and it produces a join on the wrong columns with no
    error anywhere. Resolving to names at construction, where both child
    schemas are in hand and correct, makes that unrepresentable.

    `to_operator` resolves back to indices against whatever schema the child
    actually has when the plan runs, which is the point."""
    var kind: JoinKind
    var strictness: UInt8
    var build_side: JoinBuildSide
    """Which input the hash table is built over. Not part of the schema:
    `_output_schema` does not read it, so a rewrite may change it without
    changing what the join returns."""
    var _schema: Schema

    def __init__(
        out self,
        var left: DynRelation,
        var right: DynRelation,
        var left_keys: List[Int],
        var right_keys: List[Int],
        kind: JoinKind = JOIN_INNER,
        strictness: UInt8 = 0,
        build_side: JoinBuildSide = BUILD_LEFT,
    ) raises:
        if len(left_keys) != len(right_keys):
            raise InvalidError(
                t"join: {len(left_keys)} left keys but {len(right_keys)} right "
                t"keys"
            )
        if len(left_keys) == 0:
            raise InvalidError("join: needs at least one key pair")
        self._schema = Self._output_schema(left.schema(), right.schema(), kind)
        self.left = ArcPointer(left^)
        self.right = ArcPointer(right^)
        self.left_keys = Self._names_for(
            self.left[].schema(), left_keys, "left"
        )
        self.right_keys = Self._names_for(
            self.right[].schema(), right_keys, "right"
        )
        self.kind = kind
        self.strictness = strictness
        self.build_side = build_side

    def __init__(
        out self,
        var left: DynRelation,
        var right: DynRelation,
        *,
        var left_names: List[String],
        var right_names: List[String],
        kind: JoinKind = JOIN_INNER,
        strictness: UInt8 = 0,
        build_side: JoinBuildSide = BUILD_LEFT,
    ) raises:
        """By name, for a rewrite putting a join back together.

        The index form is the public verb; this is what `traverse` and
        `optimizer.mojo` use, because a rewrite already holds names and
        converting back to indices only to have them re-resolved would be a
        round trip through the representation this node exists to avoid.

        `build_side` defaults to `BUILD_LEFT`; the rules in `optimizer.mojo`
        pass `j.build_side` through, so a rebuilt join keeps its choice.
        """
        self._schema = Self._output_schema(left.schema(), right.schema(), kind)
        self.left = ArcPointer(left^)
        self.right = ArcPointer(right^)
        self.left_keys = left_names^
        self.right_keys = right_names^
        self.kind = kind
        self.strictness = strictness
        self.build_side = build_side

    @staticmethod
    def _names_for(
        schema: Schema, indices: List[Int], side: String
    ) raises -> List[String]:
        """The column names at `indices`, or a diagnosable error.

        Out-of-range is caught here rather than at execution, where it would
        surface as an opaque kernel failure well after the plan was built.
        """
        var out = List[String](capacity=len(indices))
        for idx in indices:
            if idx < 0 or idx >= len(schema.fields):
                raise IndexError(
                    t"join: {side} key index {idx} out of range for "
                    t"{len(schema.fields)} columns"
                )
            out.append(schema.fields[idx].name.copy())
        return out^

    @staticmethod
    def _indices_for(
        schema: Schema, names: List[String], side: String
    ) raises -> List[Int]:
        """Where `names` live in `schema` now.

        Called at lowering, not construction, so a rewrite that reordered or
        narrowed the child is followed rather than ignored.
        """
        var out = List[Int](capacity=len(names))
        for ref n in names:
            var at = schema.get_field_index(n)
            if at < 0:
                raise KeyError(t"join: {side} key '{n}' is not in the input")
            out.append(at)
        return out^

    @staticmethod
    def _output_schema(
        left: Schema, right: Schema, kind: JoinKind
    ) raises -> Schema:
        """Left fields then right fields, or one side's alone for the
        existence filters, as `JoinKind.emits_left_columns` and
        `emits_right_columns` say. It takes no build side: which side is
        indexed does not change what the join returns.
        """
        var fields = List[Field]()
        if kind.emits_left_columns():
            for ref f in left.fields:
                fields.append(f.copy())
        if kind.emits_right_columns():
            for ref f in right.fields:
                fields.append(f.copy())
        return schema(fields^)

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        """Both sides, which is why this takes a function rather than a single
        child: a join has two inputs."""
        return Join(
            f(self.left[]),
            f(self.right[]),
            left_names=self.left_keys.copy(),
            right_names=self.right_keys.copy(),
            kind=self.kind,
            strictness=self.strictness,
            build_side=self.build_side,
        )

    def with_build_side(self, build_side: JoinBuildSide) raises -> Join:
        """This join, indexing the other input: the same answer at a different
        cost. `schema()` is unchanged, since `_output_schema` does not read the
        build side. On the node so a rule need not rebuild the join field by
        field and risk dropping one.
        """
        return Join(
            self.left[].copy(),
            self.right[].copy(),
            left_names=self.left_keys.copy(),
            right_names=self.right_keys.copy(),
            kind=self.kind,
            strictness=self.strictness,
            build_side=build_side,
        )

    def references(self, mut into: References):
        self.left[].references(into)
        self.right[].references(into)

    def schema(self) -> Schema:
        return self._schema.copy()

    def estimate(self) raises -> Estimate:
        """The containment estimate, computed by `Estimate.joined`. The keys
        are names, so each side's summary is found by name and a narrowed child
        schema cannot misdirect the lookup.
        """
        return Estimate.joined(
            self.left[].estimate(),
            self.right[].estimate(),
            self.left_keys,
            self.right_keys,
            self.kind,
        )

    def cost_with(self, build_side: JoinBuildSide) raises -> Cost:
        """What this join would cost if it indexed `build_side`.

        A hypothetical rather than a reading of `self.build_side`, so
        `SelectBuildSide` can ask for both arrangements from one formula. The
        children cost the same either way; only the build and probe terms
        move, and an unknown on either side makes both answers unknown.
        """
        var left = self.left[].estimate()
        var right = self.right[].estimate()
        var children = self.left[].cost() + self.right[].cost()
        if build_side == BUILD_LEFT:
            return (
                children
                + Cost.hash_build(left.rows, left.row_width())
                + Cost.hash_probe(right.rows)
            )
        else:
            return (
                children
                + Cost.hash_build(right.rows, right.row_width())
                + Cost.hash_probe(left.rows)
            )

    def cost(self) raises -> Cost:
        """What this join costs as written, `cost_with(self.build_side)`, so a
        plan's cost reflects the build side chosen for it.
        """
        return self.cost_with(self.build_side)

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        """The probe side is the pipeline and the build side a stage within it.
        The operator is given build/probe roles, not left and right, and passes
        `build_side` to the kernel to restore the left-then-right order.
        """
        var left_on = Self._indices_for(
            self.left[].schema(), self.left_keys, "left"
        )
        var right_on = Self._indices_for(
            self.right[].schema(), self.right_keys, "right"
        )
        # Two arrangements, spelled out. Picking the roles into locals first
        # and building one `JoinOperator` reads as the tighter code and
        # measured **+3,840 bytes** of `__text` on `query_join`: the branches
        # inline away, while `DynRelation.copy()` on each side does not.
        if self.build_side == BUILD_LEFT:
            var probe = self.right[].to_operator(ctx, bindings)
            probe.append(
                JoinOperator(
                    self.left[].to_operator(ctx, bindings),
                    left_on^,
                    right_on^,
                    self.kind,
                    self.strictness,
                    self.build_side,
                    self._schema.copy(),
                    self.left[].schema(),
                    self.right[].schema(),
                    ctx.copy(),
                )
            )
            return probe^
        else:
            var probe = self.left[].to_operator(ctx, bindings)
            probe.append(
                JoinOperator(
                    self.right[].to_operator(ctx, bindings),
                    right_on^,
                    left_on^,
                    self.kind,
                    self.strictness,
                    self.build_side,
                    self._schema.copy(),
                    self.right[].schema(),
                    self.left[].schema(),
                    ctx.copy(),
                )
            )
            return probe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Join(", self.left[], ", ", self.right[], ", ", self.kind)
        if self.build_side != BUILD_LEFT:
            # Printed only when it is not the default, so a plan that made no
            # physical choice still diffs against the plans in every test that
            # predates the field.
            writer.write(", ", self.build_side)
        writer.write(")")


def _positional_schema(
    verb: StringSlice, left: Schema, right: Schema
) raises -> Schema:
    """The output of a set relation over `left` and `right`.

    **Positional, like SQL.** The inputs must have the same number of columns
    with the same dtypes; the output takes the left side's names, and a field
    is nullable wherever either side's is, since a `UNION ALL` of a required
    column and a nullable one can hold a NULL.
    """
    if len(left.fields) != len(right.fields):
        raise InvalidError(
            t"{verb}: the left side has {len(left.fields)} columns but the "
            t"right side has {len(right.fields)}"
        )
    if len(left.fields) == 0:
        raise InvalidError(t"{verb}: needs at least one column")
    var fields = List[Field](capacity=len(left.fields))
    for i in range(len(left.fields)):
        if left.fields[i].dtype != right.fields[i].dtype:
            raise TypeError(
                t"{verb}: column {i} is {left.fields[i].dtype} on the left but "
                t"{right.fields[i].dtype} on the right"
            )
        var f = left.fields[i].copy()
        f.nullable = f.nullable or right.fields[i].nullable
        fields.append(f^)
    return schema(fields^)


struct Union(Relation, Writable):
    """`left UNION ALL right` — every row of both inputs, matched by position
    (see `_positional_schema`).

    Always a bag: the deduplicating `UNION` is `distinct()` over this node,
    which is what `DynRelation.union` builds, so the dedup is an ordinary
    `Aggregate` to the optimizer. Streaming on both sides, so unlike
    `Intersection` and `Difference` it holds nothing.
    """

    var left: ArcPointer[DynRelation]
    var right: ArcPointer[DynRelation]
    var _schema: Schema

    def __init__(
        out self, var left: DynRelation, var right: DynRelation
    ) raises:
        self._schema = _positional_schema(
            "union", left.schema(), right.schema()
        )
        self.left = ArcPointer(left^)
        self.right = ArcPointer(right^)

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        return Union(f(self.left[]), f(self.right[]))

    def references(self, mut into: References):
        self.left[].references(into)
        self.right[].references(into)

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        """The left side is the pipeline and the right side a stage in it,
        the arrangement `Join` uses for its build side."""
        var pipe = self.left[].to_operator(ctx, bindings)
        pipe.append(
            UnionOperator(
                self.right[].to_operator(ctx, bindings), self._schema.copy()
            )
        )
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Union(", self.left[], ", ", self.right[], ")")


trait Multiplicity:
    """How many copies of a row a `Multiset` relation keeps, from how often it
    occurs on each side. The whole difference between `INTERSECT` and
    `EXCEPT`, so it is the one thing `Multiset` is parameterised on."""

    @staticmethod
    def name() -> String:
        """The relation's name, as a plan prints it."""
        ...

    @staticmethod
    def copies(left: Int, right: Int, all: Bool) -> Int:
        """Copies of a row seen `left` times on the left and `right` times on
        the right; `all` is SQL's `ALL`, keeping multiplicity."""
        ...


struct Intersect(Multiplicity):
    """`INTERSECT [ALL]`: rows on both sides — `min(l, r)` copies with `ALL`,
    one without."""

    @staticmethod
    def name() -> String:
        return "Intersection"

    @staticmethod
    def copies(left: Int, right: Int, all: Bool) -> Int:
        if all:
            return min(left, right)
        else:
            return 1 if left > 0 and right > 0 else 0


struct Except(Multiplicity):
    """`EXCEPT [ALL]`: left rows the right side lacks — `max(l - r, 0)` copies
    with `ALL`; without it, one copy of a row the right side has *not at
    all*, which is not `max(l - r, 0)` capped at one."""

    @staticmethod
    def name() -> String:
        return "Difference"

    @staticmethod
    def copies(left: Int, right: Int, all: Bool) -> Int:
        if all:
            return max(left - right, 0)
        else:
            return 1 if left > 0 and right == 0 else 0


struct Multiset[M: Multiplicity](Relation, Writable):
    """A relation keeping each distinct row of two inputs as many times as
    `M` says — `Intersection` or `Difference`.

    **NULL is equal to itself here**, the opposite of `=` and of a join key:
    `EXCEPT` removes a NULL row the right side also has. That is why this is
    hash grouping, whose keys compare NULLs equal, rather than a semi or anti
    join, whose keys never match a NULL. Grouping is exact
    (`kernels/dictionary.mojo`), as `distinct()` is.

    Blocking on both sides: a row's count on either is final only when that
    side is exhausted.
    """

    var left: ArcPointer[DynRelation]
    var right: ArcPointer[DynRelation]
    var all: Bool
    """SQL's `ALL`: keep multiplicity rather than deduplicating."""
    var _schema: Schema

    def __init__(
        out self, var left: DynRelation, var right: DynRelation, all: Bool
    ) raises:
        self._schema = _positional_schema(
            Self.M.name(), left.schema(), right.schema()
        )
        self.left = ArcPointer(left^)
        self.right = ArcPointer(right^)
        self.all = all

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        return Multiset[Self.M](f(self.left[]), f(self.right[]), self.all)

    def references(self, mut into: References):
        self.left[].references(into)
        self.right[].references(into)

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        var pipe = self.left[].to_operator(ctx, bindings)
        pipe.append(
            MultisetOperator[Self.M](
                self.right[].to_operator(ctx, bindings),
                self.all,
                self._schema.copy(),
                ctx.copy(),
            )
        )
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.M.name(), "(", self.left[], ", ", self.right[])
        if self.all:
            writer.write(", all")
        writer.write(")")


comptime Intersection = Multiset[Intersect]
"""`left INTERSECT [ALL] right`."""

comptime Difference = Multiset[Except]
"""`left EXCEPT [ALL] right`."""


struct _ScanPathRest(Movable):
    """What a `ScanPath` holds beyond one literal path: more paths, or a
    parameter."""

    var more: List[String]
    var param: Optional[StringParam[StringType]]

    def __init__(
        out self,
        var more: List[String],
        var param: Optional[StringParam[StringType]],
    ):
        self.more = more^
        self.param = param^


struct ScanPath(Copyable, Movable, Writable):
    """Where a scan reads from: one or more literal paths, read in order, or a
    `string` parameter resolved per execution.

    A type of its own rather than a second field on the scan, so every place
    that rebuilds a scan carries the parameter with it.
    """

    # Everything but the first path sits behind one `ArcPointer`, measured: a
    # scan is a member of `DynRelation`'s variant, whose copy and destroy code
    # is inlined wherever a plan is copied, in every binary. The parameter
    # inline cost `query_streaming` 19,396 bytes of `__text` and `query_join`
    # 23,904; a second optional pointer for the other paths ~7.7 KB on
    # `query_streaming`; one `List` of every path ~120 KB on `query_cli`.
    var _literal: String
    var _rest: Optional[ArcPointer[_ScanPathRest]]

    def __init__(out self, var path: String):
        self._literal = path^
        self._rest = None

    def __init__(out self, var paths: List[String]):
        self._literal = paths[0].copy() if len(paths) else String()
        self._rest = None
        if len(paths) > 1:
            var more = List[String](capacity=len(paths) - 1)
            for i in range(1, len(paths)):
                more.append(paths[i].copy())
            self._rest = ArcPointer(_ScanPathRest(more^, None))

    def __init__(out self, var param: StringParam[StringType]):
        self._literal = String()
        self._rest = ArcPointer(_ScanPathRest(List[String](), param^))

    def resolve(self, bindings: Bindings) raises -> List[String]:
        """The paths this execution reads, in order."""
        if self._rest and self._rest.value()[].param:
            return [self._rest.value()[].param.value().value(bindings)]
        var out: List[String] = [self._literal.copy()]
        if self._rest:
            out.extend(self._rest.value()[].more.copy())
        return out^

    def references(self, mut into: References):
        if self._rest and self._rest.value()[].param:
            self._rest.value()[].param.value().references(into)

    def write_to[W: Writer](self, mut writer: W):
        if self._rest and self._rest.value()[].param:
            writer.write(self._rest.value()[].param.value())
            return
        writer.write(self._literal)
        if self._rest:
            for ref p in self._rest.value()[].more:
                writer.write(", ", p)


struct ParquetScan(FileScan, Writable):
    """Parquet files as a source, read in order one row group at a time.

    **The schema is the projection.** The scan reads only its own columns out
    of the file, so narrowing a scan's schema *is* how a projection gets pushed
    into it — no separate mechanism, and nothing to keep in sync.

    The schema is supplied rather than read from the file, so building the plan
    touches no I/O: a `Relation` is a description, and a description that has
    to open a file to exist cannot be constructed for a file that is not there
    yet. The operator opens it on first `drain`.
    """

    var path: ScanPath
    var _schema: Schema
    var pruners: List[DynValue]
    """Predicates this scan may use to skip row groups, put here by
    `PushFilterIntoScan`.

    Empty on a plan nobody optimized, which reads every row group — the answer
    the engine gave before pruning existed. **Never consulted for
    correctness:** the `Filter` these came from is still above this scan and
    still evaluates the exact predicate on every row it produces, so an entry
    here costs time and never an answer.
    """

    var statistics: Optional[ArcPointer[Estimate]]
    """What this file's footer says, supplied by a caller that has read it:
    `scan(path, schema).with_statistics(Estimate.from_index(
    Index.from_parquet(file), schema))`. The scan does not open the file
    itself, which would link the Parquet reader into every binary that
    estimates a plan. Behind an `ArcPointer` because a relation node's fields
    are copied wherever a plan is; `ScanPath` measured an inline one at
    23,904 bytes on `query_join`.
    """

    def __init__(
        out self,
        var path: ScanPath,
        var schema: Schema,
        var pruners: List[DynValue] = [],
        var statistics: Optional[ArcPointer[Estimate]] = None,
    ):
        self.path = path^
        self._schema = schema^
        self.pruners = pruners^
        self.statistics = statistics^

    def references(self, mut into: References):
        self.path.references(into)
        for ref p in self.pruners:
            p.references(into)

    def schema(self) -> Schema:
        return self._schema.copy()

    def with_schema(self, var schema: Schema) -> ParquetScan:
        """This scan over a narrower schema, keeping its pruners and
        statistics. On the node so `ColumnPruning` cannot drop either by
        rebuilding the scan from its path and schema. The statistics are kept
        whole: a summary for a column no longer projected is never read.
        """
        return ParquetScan(
            self.path.copy(),
            schema^,
            self.pruners.copy(),
            self.statistics.copy(),
        )

    def with_statistics(self, var statistics: Estimate) -> ParquetScan:
        """This scan, told what its file's footer says. Separate from the
        constructor because building a plan does no I/O and reading a footer
        does.
        """
        return ParquetScan(
            self.path.copy(),
            self._schema.copy(),
            self.pruners.copy(),
            ArcPointer(statistics^),
        )

    def estimate(self) raises -> Estimate:
        """What the footer said, or a column-shaped unknown.

        Unknown is the honest answer for a scan nobody read a footer for, and
        it is what a plan built by `scan(path, schema)` alone gets. It is
        column-shaped rather than empty so that positional readers line up
        with the schema, and so a `Join` above it still concatenates the right
        number of summaries.
        """
        if self.statistics:
            return self.statistics.value()[].copy()
        return Estimate.unknown(self._schema)

    def cost(self) raises -> Cost:
        """The rows it decodes and the bytes they become.

        **Pruning is not subtracted**, though it is the whole reason the
        pruners are on this node: which row groups survive depends on this
        execution's `Bindings`, which a plan does not have — `param` is
        resolved per run, and `ParquetScanOperator.drain` builds the read plan
        with the values it was given. A cost that guessed at it would be a
        different number for the same plan on every run.
        """
        var estimate = self.estimate()
        return Cost.source(estimate.rows, estimate.row_width())

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        return Pipeline(
            ParquetScanOperator(
                self.path.resolve(bindings),
                self._schema.copy(),
                ctx.copy(),
                self.pruners.copy(),
                bindings.copy(),
            )
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("ParquetScan(", self.path, ")")
        if len(self.pruners):
            writer.write(" pruned by ", len(self.pruners))


trait FileScan(Relation):
    """A relation reading files of one format, in order: `ParquetScan`,
    `IpcScan` or `JsonScan`.

    Each is a plan node of its own, run by an operator of its own, so the
    optimizer tells formats apart with `isa`. What they share is that the
    schema is supplied rather than read — building the plan does no I/O — and
    is the projection: `with_schema` is how `ColumnPruning` narrows what any
    of them reads.
    """

    def with_schema(self, var schema: Schema) -> Self:
        """This scan over a narrower schema."""
        ...


struct IpcScan(FileScan, Writable):
    """An Arrow IPC file, one record batch at a time, the schema selecting
    columns by name. IPC stores whole columns per batch, so a narrowed scan
    still reads the others; it has no statistics and prunes nothing."""

    var path: ScanPath
    var _schema: Schema

    def __init__(out self, var path: ScanPath, var schema: Schema):
        self.path = path^
        self._schema = schema^

    def references(self, mut into: References):
        self.path.references(into)

    def schema(self) -> Schema:
        return self._schema.copy()

    def with_schema(self, var schema: Schema) -> Self:
        return Self(self.path.copy(), schema^)

    def estimate(self) raises -> Estimate:
        """Unknown, and column-shaped for the reason `ParquetScan`'s is."""
        return Estimate.unknown(self._schema)

    def cost(self) raises -> Cost:
        var estimate = self.estimate()
        return Cost.source(estimate.rows, estimate.row_width())

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        return Pipeline(
            IpcScanOperator(self.path.resolve(bindings), self._schema.copy())
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("IpcScan(", self.path, ")")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


struct JsonScan(FileScan, Writable):
    """A newline-delimited JSON file, one batch per block; see `marrow.json`.
    Keys outside the schema are skipped, so a narrowed scan reads only the
    columns the plan needs out of rows that carry more. It has no statistics
    and prunes nothing."""

    var path: ScanPath
    var _schema: Schema

    def __init__(out self, var path: ScanPath, var schema: Schema):
        self.path = path^
        self._schema = schema^

    def references(self, mut into: References):
        self.path.references(into)

    def schema(self) -> Schema:
        return self._schema.copy()

    def with_schema(self, var schema: Schema) -> Self:
        return Self(self.path.copy(), schema^)

    def estimate(self) raises -> Estimate:
        """Unknown, and column-shaped for the reason `ParquetScan`'s is."""
        return Estimate.unknown(self._schema)

    def cost(self) raises -> Cost:
        var estimate = self.estimate()
        return Cost.source(estimate.rows, estimate.row_width())

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        return Pipeline(
            JsonScanOperator(self.path.resolve(bindings), self._schema.copy())
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("JsonScan(", self.path, ")")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)
