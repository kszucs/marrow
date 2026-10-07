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
from std.collections import Dict, Set
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
from ..kernels.join import (
    BUILD_LEFT,
    JOIN_ALL,
    JOIN_ANTI,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
    JoinBuildSide,
    JoinKind,
)
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
from ..iceberg.catalog import IcebergTable
from ..tabular import RecordBatch
from ..dtypes import DynType, Field, StringType, field, int64, null
from ..scalars import DynScalar
from .estimates import (
    Approx,
    ColumnEstimate,
    Cost,
    DEFAULT_SELECTIVITY,
    Estimate,
    HISTOGRAM_BUCKETS,
    Selectivity,
    Size,
)
from .index import Index, keep_every
from .analyze import analyze
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
    JoinOrder,
    IpcScanOperator,
    IcebergScanOperator,
    JsonScanOperator,
    ParquetScanOperator,
    ProjectOperator,
    MultisetOperator,
    RenamedPredicate,
    SelectOperator,
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
        return keep_every(index.chunks())

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

    def read_column(self) -> Optional[String]:
        """The column this value merely reads, or `None` when it computes: a
        read names exactly the one column it reads and is not an aggregate —
        an aggregate aliased to its own column computes."""
        if self.aggregates():
            return None
        var cols = self.columns()
        if len(cols) != 1 or cols[0] != self.name():
            return None
        return cols[0].copy()

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
# Parameters — what a plan declares, and the values one execution binds
# ---------------------------------------------------------------------------
comptime Bindings = Dict[String, DynScalar]
"""Parameter values for one execution: a plain name -> scalar map.

A caller writes a dict literal:

    plan.execute(bindings={"min-a": Int64Scalar(4).to_dyn()})

Passed to `to_operator`, not stored on the plan, which is what keeps a plan
immutable and lets two executions use different values without interfering.

Missing names are not an error here — a parameter with a default is satisfied
without one, and `NumericParam` raises naming itself when it has neither.
"""


struct ParamSpec(Copyable, Movable):
    """A parameter a plan declares: what a caller must, or may, bind.

    Holds no `DynScalar`: a `List` of a struct carrying one is the
    List-of-Variant growth defect CLAUDE.md records, and the default is only
    ever *shown* here — the node itself applies it.
    """

    var name: String
    var dtype: DynType
    var help: String
    var default: Optional[String]
    """The default as a command line would spell it; `None` when required."""

    var parse: Optional[def(String) thin raises -> DynScalar]
    """A command-line token as this parameter's scalar, or `None` when the dtype
    has no command-line spelling. Instantiated per dtype a plan names, so a
    binary links only the parsers its parameters need."""

    def __init__(
        out self,
        var name: String,
        var dtype: DynType,
        var help: String,
        var default: Optional[String],
        parse: Optional[def(String) thin raises -> DynScalar],
    ):
        self.name = name^
        self.dtype = dtype^
        self.help = help^
        self.default = default^
        self.parse = parse


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

        The default knows nothing but its columns' widths,
        `Estimate.unknown`, which every formula propagates as unknown, so a
        node without an estimate costs an optimization and never a wrong
        answer. Resolved on the variant ladder
        rather than through a trampoline slot: it reaches no kernel, and a
        slot would be paid by every binary that holds a plan.
        """
        return Estimate.unknown(self.schema())

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
        JoinChain,
        Union,
        Intersection,
        Difference,
        ParquetScan,
        IpcScan,
        JsonScan,
        IcebergScan,
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

    def children(self) raises -> List[DynRelation]:
        """This node's inputs, in the order `with_children` takes them."""
        var children = List[DynRelation]()

        def collect(child: DynRelation) raises {mut children} -> DynRelation:
            children.append(child.copy())
            return child.copy()

        _ = self.traverse(collect)
        return children^

    def with_children(self, children: List[DynRelation]) raises -> DynRelation:
        """This node over `children` in place of its inputs, one for each, in
        the order `children()` gave them."""
        var next = 0

        def put(child: DynRelation) raises {mut next, imm} -> DynRelation:
            next += 1
            return children[next - 1].copy()

        return self.traverse(put)

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
        # The first declaration of a name is the one reported; a later read of
        # another dtype is refused when the value binds, naming the parameter.
        var out = List[ParamSpec]()
        for ref spec in refs.params:
            var seen = False
            for ref kept in out:
                if kept.name == spec.name:
                    seen = True
            if not seen:
                out.append(spec.copy())
        return out^

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

        Over a join chain `MergeProjectIntoJoin` folds it into the chain's own
        output.
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
        """`SELECT <values> AS <names>` — new columns over the same rows.

        Over a join chain `MergeProjectIntoJoin` folds one that only reads
        columns into the chain.
        """
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
        strictness: UInt8 = JOIN_ALL,
        lname: String = "",
        rname: String = "{name}_right",
    ) raises -> DynRelation:
        """Equijoin, as ibis's `Join.join`: on a join chain it adds a join to
        the chain, otherwise it starts one. Keys are positions in `self`'s and
        `right`'s output.

        The output is `self`'s columns then `right`'s, named by ibis's rules:
        an inner key both sides call the same is emitted once, and any other
        clash is renamed by `lname` / `rname`. A clash those leave raises:
        two output columns may not share a name.

        `build_side` names the side to index, the left by default, and changes
        the cost and nothing else — except under `JOIN_ANY`, one match per
        probe row, where it is part of the answer and nothing moves it. Which
        of a probe row's matches `JOIN_ANY` keeps is unspecified.
        """
        return JoinChain.joined(
            self,
            right,
            left_keys,
            right_keys,
            kind,
            strictness,
            build_side,
            lname,
            rname,
        )

    def with_chain(self, var chain: JoinChain) -> DynRelation:
        """This join chain node holding `chain` instead — how a rule rebuilds
        a chain. Undefined unless `isa[JoinChain]()`."""
        # Assigned into a copy rather than boxed afresh: boxing a `JoinChain`
        # wires its lowering, and the join kernels, into every binary that
        # reaches the call, joining or not.
        var out = self.copy()
        out.get[JoinChain]() = chain^
        return out^

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

    def analyze(
        self,
        ctx: ExecContext = ExecContext.auto(),
        bindings: Bindings = Bindings(),
    ) raises -> DynRelation:
        """This plan with statistics on every source — see `analyze.mojo`.

        The one step that reads data before a plan runs, so it is one a caller
        asks for: `plan.analyze().optimize[AllRules]()`.
        """
        return analyze(self, ctx, bindings)

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
    var statistics: Optional[ArcPointer[Estimate]]
    """What `analyze` found in the batch, or `None` until something looked.
    Behind an `ArcPointer` for the reason `ParquetScan.statistics` is."""

    def __init__(
        out self,
        var batch: RecordBatch,
        var statistics: Optional[ArcPointer[Estimate]] = None,
    ):
        self.batch = batch^
        self.statistics = statistics^

    def with_batch(self, var batch: RecordBatch) -> InMemoryTable:
        """This source over a narrower batch, keeping its statistics — on the
        node so `ColumnPruning` cannot drop them by rebuilding it. The
        statistics are kept whole: `estimate` reads only the columns the batch
        still holds."""
        return InMemoryTable(batch^, self.statistics.copy())

    def with_statistics(self, var statistics: Estimate) -> InMemoryTable:
        """This source, told what `analyze` found in it."""
        return InMemoryTable(self.batch.copy(), ArcPointer(statistics^))

    def references(self, mut into: References):
        pass

    def schema(self) -> Schema:
        return self.batch.schema.copy()

    def estimate(self) raises -> Estimate:
        """What `analyze` found, when something asked it to look. Otherwise
        the exact row and null counts, which the batch stores, and nothing a
        scan of the data would have to find — bounds, distinct counts and a
        string's width stay unknown.
        """
        if self.statistics:
            return self.statistics.value()[].select(
                self.batch.schema.names(), self.batch.schema
            )
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
        return Cost.source(estimate.size())

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

        See `verdict_of`: the comparisons a scan prunes with, over the input's
        bounds and over a histogram of the column the predicate reads.
        """
        var input = self.input[].estimate()
        return input.filtered(
            Self.verdict_of(self.predicate, self.constant, input)
        )

    @staticmethod
    def verdict_of(
        predicate: DynValue, constant: Optional[Bool], input: Estimate
    ) raises -> Selectivity:
        """What `predicate` keeps of an input estimated as `input`. Static, so
        a join chain's absorbed predicate, which has no `Filter` around it, is
        judged the same way.

        **A proof first**: `Value.mask` over a one-chunk index built from the
        input's estimate, the comparisons a scan prunes with — a `false` bit
        proves no row matches. **Then a share**, for a predicate reading one
        column with known bounds: the fraction of an equal-width histogram's
        buckets its `mask` keeps (`ColumnEstimate.histogram`), so an equality
        keeps one distinct value's share and a range the share of the range it
        covers. Anything else, and anything that cannot be evaluated — one
        naming an unbound parameter — keeps `DEFAULT_SELECTIVITY`.

        An input already proven empty is kept whole, whatever the predicate
        would say about bounds that no longer describe any row.
        """
        if input.rows == Approx.exact(0):
            return Selectivity.everything()
        if constant:
            if constant.value():
                return Selectivity.everything()
            return Selectivity.nothing()

        if not input.to_index().surviving([predicate.copy()])[0]:
            return Selectivity.nothing()

        var names = predicate.columns()
        if len(names) == 1:
            var column = input.column(names[0])
            if column:
                try:
                    var buckets = column.value().histogram(
                        HISTOGRAM_BUCKETS, input.rows
                    )
                    if buckets.chunks() > 0:
                        var kept = predicate.mask(buckets)
                        var n = 0
                        for i in range(len(kept)):
                            if kept.is_null(i) or kept[i].value():
                                n += 1
                        return Selectivity.share(max(n, 1), len(kept))
                except:
                    pass
        return Selectivity.default()

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
                var read = self.values[i].read_column()
                return Bool(read) and read.value() == name
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
        """The input's rows unchanged. An output that is a bare column read
        keeps that column's summary under its own name, **a rename included**:
        the values are the same ones, whatever they are called. Anything
        computed is new and keeps only the width of its declared dtype.

        Deliberately wider than `passes_through`, which must refuse a rename
        because a predicate pushed past one would name a column that does not
        exist below. An estimate names nothing below, and every input the SQL
        frontend builds is exactly such a rename.
        """
        var sources = List[String](capacity=len(self.values))
        for ref v in self.values:
            # A bare column, as `_output_schema` recognises one.
            if v.name() != "" and len(v.columns()) == 1:
                sources.append(v.name())
            else:
                sources.append(String())
        return self.input[].estimate().select(sources, self._schema)

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
            + Cost.hash_build(output.size())
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
        return self.input[].cost() + Cost.sort(input.size())

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
        return (
            self.input[].estimate().select(self._schema.names(), self._schema)
        )

    def cost(self) raises -> Cost:
        """A sort, because that is what it does.

        `WindowOperator` buffers and orders by the partition and order keys
        before it can evaluate anything, so this is priced identically to
        `Sort` — a window is not a cheaper way to order.
        """
        var input = self.input[].estimate()
        return self.input[].cost() + Cost.sort(input.size())

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


@fieldwise_init
struct JoinRef(Copyable, Equatable, Movable, Writable):
    """A column of one participant of a `JoinChain`: which input, and which of
    its columns.

    By participant rather than by name alone, so two inputs may share a column
    name — a self-join, a natural key — without a key, a filter or an output
    ever meaning the other one.
    """

    var input: Int
    var name: String

    def __eq__(self, other: Self) -> Bool:
        return self.input == other.input and self.name == other.name

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def __lt__(self, other: Self) -> Bool:
        """By participant, then by column name — an order no plan of a chain
        changes."""
        if self.input != other.input:
            return self.input < other.input
        return self.name < other.name

    def key(self) -> String:
        """The name an `Estimate` column for this participant column carries:
        unique across the chain, since two participants may share a name."""
        return String("#", self.input, ".", self.name)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.key())

    @staticmethod
    def keys(refs: List[Self]) -> List[String]:
        """`refs` as the column names an `Estimate` keys them by."""
        var out = List[String](capacity=len(refs))
        for ref r in refs:
            out.append(r.key())
        return out^


struct JoinLink(Copyable, Equatable, Movable, Writable):
    """One `.join` of a chain: how participant `k + 1` joins participants
    `0 … k`, for the chain's `k`-th link.

    `left_keys` read columns of the participants before it, `right_keys`
    columns of its own. The build side is a hint the planner may overrule —
    except under `JOIN_ANY`, which keeps one match per *probe* row and so
    answers differently built from the other side. Which match it keeps is
    unspecified.
    """

    var kind: JoinKind
    var strictness: UInt8
    var build_side: JoinBuildSide
    var left_keys: List[JoinRef]
    var right_keys: List[JoinRef]

    def __init__(
        out self,
        kind: JoinKind,
        strictness: UInt8,
        build_side: JoinBuildSide,
        var left_keys: List[JoinRef],
        var right_keys: List[JoinRef],
    ):
        self.kind = kind
        self.strictness = strictness
        self.build_side = build_side
        self.left_keys = left_keys^
        self.right_keys = right_keys^

    def __eq__(self, other: Self) -> Bool:
        return (
            self.kind == other.kind
            and self.strictness == other.strictness
            and self.build_side == other.build_side
            and self.left_keys == other.left_keys
            and self.right_keys == other.right_keys
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def is_inner(self) -> Bool:
        """An inner `JOIN_ALL` link: associative and commutative with every
        other, so a planner may join its participant in any order."""
        return self.kind == JOIN_INNER and self.strictness == JOIN_ALL

    def attaches(self) -> Bool:
        """Does this link keep, pad or drop the rows before it by what
        matches in its participant, and otherwise leave them as they are?

        A LEFT, SEMI or ANTI join does, and so does a `JOIN_ANY` inner join
        probing from the left. Such a link commutes with an inner join or a
        filter that does not read its participant: `(A ⟕ B) ⋈ C` is
        `(A ⋈ C) ⟕ B` when `C`'s keys do not read `B`.
        """
        return self.passes(True) and not self.passes(False)

    def passes(self, left: Bool) -> Bool:
        """May a filter over the left side — or the right, when `left` is
        false — run below this join and leave its answer as it was?

        An inner `JOIN_ALL` join commutes with a filter over either side. A
        join that keeps, pads or drops one side's rows by what matches on the
        other — LEFT, SEMI or ANTI keep the left, their mirrors the right, a
        `JOIN_ANY` inner join its probe side — commutes with a filter over the
        side it keeps. A filter over a side a join pads would drop the padded
        rows too, and one over the side a `JOIN_ANY` join picks a match from
        would change the match.
        """
        if self.strictness == JOIN_ALL:
            if self.kind == JOIN_INNER:
                return True
            if (
                self.kind == JOIN_LEFT
                or self.kind == JOIN_SEMI
                or self.kind == JOIN_ANTI
            ):
                return left
            if (
                self.kind == JOIN_RIGHT
                or self.kind == JOIN_RIGHT_SEMI
                or self.kind == JOIN_RIGHT_ANTI
            ):
                return not left
            return False
        if self.kind == JOIN_INNER:
            return left == (self.build_side != BUILD_LEFT)
        return False

    def joined(self, left: Estimate, right: Estimate) -> Estimate:
        """`Estimate.joined` of `left` and `right` as this link joins them,
        both keyed by participant column (`JoinRef.key`)."""
        return Estimate.joined(
            left,
            right,
            JoinRef.keys(self.left_keys),
            JoinRef.keys(self.right_keys),
            self.kind,
            self.strictness,
            self.build_side,
        )

    def write_condition[W: Writer](self, mut writer: W):
        """`on <pairs>`, then the build side unless it is the left, and
        `any` under `JOIN_ANY`."""
        writer.write("on ")
        for i in range(len(self.left_keys)):
            if i > 0:
                writer.write(", ")
            writer.write(self.left_keys[i], "=", self.right_keys[i])
        if self.build_side != BUILD_LEFT:
            writer.write(", ", self.build_side)
        if self.strictness != JOIN_ALL:
            writer.write(", any")

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.kind, " #", self.right_keys[0].input, " ")
        self.write_condition(writer)


struct JoinFilter(Copyable, Movable):
    """A predicate a chain evaluates between its joins rather than above
    them — a residual `Filter` absorbed by `PushFilterIntoJoin`.

    `refs` is parallel to `predicate.columns()`: the name the predicate reads
    each column by, and the participant column it means. The predicate never
    moves or changes; only where it is evaluated does — never after the link
    joining participant `bound + 1`, the first one appended after it was
    absorbed, so a join appended later never has a filter moved across it.

    A filter lowers itself through function pointers its constructor wires
    (`stage`, `selection`): only the optimizer builds one, so a binary that
    never optimizes links neither operator.
    """

    var predicate: DynValue
    var refs: List[JoinRef]
    var bound: Int
    var _stage: def(
        DynValue, Schema, List[Int], ExecContext, Bindings
    ) thin raises -> DynOperator
    var _selection: def(List[Int], Schema) thin -> DynOperator

    def __init__(
        out self, var predicate: DynValue, var refs: List[JoinRef], bound: Int
    ):
        self.predicate = predicate^
        self.refs = refs^
        self.bound = bound
        self._stage = Self._stage_of
        self._selection = Self._selection_of

    def with_bounds(self, var refs: List[JoinRef], bound: Int) -> Self:
        """This filter reading `refs` and bounded by `bound` — how a chain
        renumbers a filter when its participants change."""
        var out = self.copy()
        out.refs = refs^
        out.bound = bound
        return out^

    def participants(self) -> Set[Int]:
        """The participants this reads."""
        var out = Set[Int]()
        for ref r in self.refs:
            out.add(r.input)
        return out^

    def verdict(self, estimate: Estimate) raises -> Selectivity:
        """What this filter keeps of `estimate`, whose columns are keyed by
        participant column (`JoinRef.key`)."""
        var names = self.predicate.columns()
        var cols = List[ColumnEstimate](capacity=len(names))
        for i in range(len(names)):
            var c = estimate.column(self.refs[i].key())
            if c:
                cols.append(c.take())
                cols[len(cols) - 1].name = names[i].copy()
        return Filter.verdict_of(
            self.predicate, None, Estimate(estimate.rows, cols^)
        )

    def stage(
        self,
        fields: List[Field],
        indices: List[Int],
        ctx: ExecContext,
        bindings: Bindings,
    ) raises -> DynOperator:
        """The operator evaluating this filter over a batch holding its
        columns at `indices`, typed as `fields` say."""
        var named = List[Field](capacity=len(fields))
        var names = self.predicate.columns()
        for i in range(len(fields)):
            named.append(
                Field(
                    names[i].copy(),
                    fields[i].dtype.copy(),
                    fields[i].nullable,
                    fields[i].metadata.copy(),
                )
            )
        return self._stage(
            self.predicate, schema(named^), indices, ctx, bindings
        )

    def selection(self, indices: List[Int], schema: Schema) -> DynOperator:
        """The operator picking the positions `indices` as `schema` — what a
        chain answers with after the filters landing on its root, which may
        read a column its output drops."""
        return self._selection(indices, schema)

    @staticmethod
    def _selection_of(indices: List[Int], schema: Schema) -> DynOperator:
        return SelectOperator(indices.copy(), schema.copy())

    @staticmethod
    def _stage_of(
        predicate: DynValue,
        view: Schema,
        indices: List[Int],
        ctx: ExecContext,
        bindings: Bindings,
    ) raises -> DynOperator:
        return FilterOperator(
            RenamedPredicate(
                predicate.to_operator(view, False, bindings),
                indices.copy(),
                view.copy(),
            ),
            ctx.copy(),
        )


struct JoinClasses(Copyable, Movable):
    """Equality classes of join keys: a union-find over `JoinRef`s.

    Linear scans rather than a hash map, because the verb that builds a join
    uses this and every binary that joins pays for what it links; a chain has
    tens of keys, not thousands.
    """

    var refs: List[JoinRef]
    var parent: List[Int]

    def __init__(out self):
        self.refs = List[JoinRef]()
        self.parent = List[Int]()

    def index(self, r: JoinRef) -> Int:
        """`r`'s node, or `-1` when no key names it."""
        for i in range(len(self.refs)):
            if self.refs[i] == r:
                return i
        return -1

    def _add(mut self, r: JoinRef) -> Int:
        var i = self.index(r)
        if i >= 0:
            return i
        self.refs.append(r.copy())
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def root(self, var i: Int) -> Int:
        while self.parent[i] != i:
            i = self.parent[i]
        return i

    def union(mut self, a: JoinRef, b: JoinRef):
        var ra = self.root(self._add(a))
        var rb = self.root(self._add(b))
        if ra != rb:
            self.parent[rb] = ra

    def union_all(mut self, left: List[JoinRef], right: List[JoinRef]):
        """Every key pair of one join made equal."""
        for i in range(len(left)):
            self.union(left[i], right[i])

    def connected(self, a: JoinRef, b: JoinRef) -> Bool:
        var i = self.index(a)
        var j = self.index(b)
        if i < 0 or j < 0:
            return False
        return self.root(i) == self.root(j)


@fieldwise_init
struct KeyClass(Copyable, Movable):
    """One equality class of a chain's inner keys: its members in `JoinRef`
    order, and the participants holding one."""

    var members: List[JoinRef]
    var span: Set[Int]

    def within(self, s: Set[Int]) -> List[JoinRef]:
        """The members held by the participants `s`, in order."""
        var out = List[JoinRef]()
        for ref m in self.members:
            if m.input in s:
                out.append(m.copy())
        return out^

    def side(self, s: Set[Int]) -> List[JoinRef]:
        """The members of `s` a join to `s` compares: every one when they all
        sit in one participant — no join has compared those to one another
        yet — and otherwise the first. Undefined unless `span` meets `s`."""
        var out = self.within(s)
        if out[0].input != out[len(out) - 1].input:
            return [out[0].copy()]
        return out^


struct JoinRules(Copyable, Movable):
    """Which join trees compute a chain's answer: the sets of participants a
    tree may join (`valid`, `joinable`), on which keys (`keys`, `link`), and
    where each filter is evaluated (`holds`) — what a planner may choose
    from, and what `JoinOrder.verify` checks a tree against.

    Participant `i > 0` is joined by link `i - 1` in one of three ways:

    - **On inner keys**, by an inner `JOIN_ALL` link: its keys join the
      equality classes (`classes`), and any tree comparing every member of
      every class at least once returns the same bag — a NULL key drops its
      row on every path, and a NaN matches a NaN, the hash join being
      NaN-safe.
    - **Attached**, by a link that keeps, pads or drops the rows before it
      (`JoinLink.attaches`): only by its own link, once the participants its
      keys read are in. It then commutes with every inner join and filter
      over those rows — `(A ⟕ B) ⋈ C` is `(A ⋈ C) ⟕ B` — the rules of
      Moerkotte, Fender and Eich, *On the Correct and Complete Enumeration of
      the Core Search Space*, SIGMOD 2013, for these kinds.
    - **In place**, by any other link — RIGHT, FULL, a right existence join,
      a `JOIN_ANY` probing from the right, or an inner join reading an
      attached participant: only by its own link, onto exactly the
      participants before it, and no set may hold participants from both
      sides of it without holding it too.
    """

    var links: List[JoinLink]
    """Link `i - 1` joins participant `i`."""
    var classes: List[KeyClass]
    """The classes the inner keys make, ascending by their least member, so
    a join's keys come out in an order no plan changes."""
    var spines: List[Set[Int]]
    """Per participant joined by its own link, the participants that must be
    in first: what its keys read when attached, everything before it when in
    place. Empty for one joined on inner keys."""
    var neighbours: List[Set[Int]]
    """Per participant, those it shares a class with, attaches to or is
    attached by — a superset of what joins it, which `joinable` narrows."""

    def __init__(out self, links: List[JoinLink]):
        var n = len(links) + 1
        self.links = links.copy()
        self.classes = List[KeyClass]()
        self.spines = List[Set[Int]](length=n, fill=Set[Int]())
        self.neighbours = List[Set[Int]](length=n, fill=Set[Int]())

        var union = JoinClasses()
        var attached = Set[Int]()
        for i in range(1, n):
            ref link = links[i - 1]
            var reads = Set[Int]()
            for ref r in link.left_keys:
                reads.add(r.input)
            if link.is_inner() and reads.isdisjoint(attached):
                union.union_all(link.left_keys, link.right_keys)
                continue
            if link.attaches():
                self.spines[i] = reads^
                attached.add(i)
            else:
                for j in range(i):
                    self.spines[i].add(j)
            self.neighbours[i] |= self.spines[i]
            for j in self.spines[i]:
                self.neighbours[j].add(i)

        var roots = List[Int]()
        for i in range(len(union.refs)):
            var root = union.root(i)
            var k = -1
            for j in range(len(roots)):
                if roots[j] == root:
                    k = j
            if k < 0:
                k = len(roots)
                roots.append(root)
                self.classes.append(KeyClass(List[JoinRef](), Set[Int]()))
            ref r = union.refs[i]
            ref c = self.classes[k]
            c.span.add(r.input)
            var at = len(c.members)
            while at > 0 and r < c.members[at - 1]:
                at -= 1
            c.members.insert(at, r.copy())
        for i in range(1, len(self.classes)):
            var j = i
            while (
                j > 0
                and self.classes[j].members[0] < self.classes[j - 1].members[0]
            ):
                self.classes.swap_elements(j, j - 1)
                j -= 1
        for ref c in self.classes:
            for ref m in c.members:
                for j in c.span:
                    if j != m.input:
                        self.neighbours[m.input].add(j)

    def participants(self) -> Int:
        return len(self.links) + 1

    def everything(self) -> Set[Int]:
        var out = Set[Int]()
        for i in range(self.participants()):
            out.add(i)
        return out^

    def in_place(self, i: Int) -> Bool:
        """Is participant `i` joined by its own link onto exactly the
        participants before it?"""
        return len(self.spines[i]) > 0 and not self.links[i - 1].attaches()

    # -- which sets a tree may join -------------------------------------------
    def valid(self, s: Set[Int]) -> Bool:
        """Can a tree join exactly the participants `s`? Not one holding a
        participant joined by its own link without what must be in first,
        nor one holding participants from both sides of an in-place link
        without that link's own."""
        if len(s) == 1:
            return True
        for i in s:
            if not self.spines[i] <= s:
                return False
        for i in range(1, self.participants()):
            if self.in_place(i) and i not in s:
                var before = False
                var after = False
                for j in s:
                    if j < i:
                        before = True
                    else:
                        after = True
                if before and after:
                    return False
        return True

    def attaching(self, a: Set[Int], b: Set[Int]) -> Int:
        """The participant joined by its own link when `a` meets `b` — one
        side that participant alone, the other holding what must be in
        first — or `-1`."""
        if len(b) == 1:
            for i in b:
                if len(self.spines[i]) > 0 and self.spines[i] <= a:
                    return i
        if len(a) == 1:
            for i in a:
                if len(self.spines[i]) > 0 and self.spines[i] <= b:
                    return i
        return -1

    def joinable(self, a: Set[Int], b: Set[Int]) -> Bool:
        """May a tree join `a` to `b`: by a participant's own link, or on
        inner keys, between two sets a tree can join into one it can join
        too."""
        if not (self.valid(a) and self.valid(b) and self.valid(a | b)):
            return False
        if self.attaching(a, b) >= 0:
            return True
        for ref c in self.classes:
            if not c.span.isdisjoint(a) and not c.span.isdisjoint(b):
                return True
        return False

    def adjacent(self, s: Set[Int]) -> Set[Int]:
        """The participants outside `s` that share a class with one inside
        it, or are joined to it by a link."""
        var out = Set[Int]()
        for i in s:
            out |= self.neighbours[i]
        return out - s

    def keys(
        self, a: Set[Int], b: Set[Int]
    ) -> Tuple[List[JoinRef], List[JoinRef]]:
        """The key pairs joining `a` to `b`: per class present on both sides,
        `KeyClass.side` of each, each member paired with the other side's
        first — so every member of every class is compared by the smallest
        subtree joining it to another, and the tree returns the bag."""
        var left = List[JoinRef]()
        var right = List[JoinRef]()
        for ref c in self.classes:
            if not c.span.isdisjoint(a) and not c.span.isdisjoint(b):
                var l = c.side(a)
                var r = c.side(b)
                for ref x in l:
                    left.append(x.copy())
                    right.append(r[0].copy())
                for k in range(1, len(r)):
                    left.append(l[0].copy())
                    right.append(r[k].copy())
        return (left^, right^)

    def link(self, a: Set[Int], b: Set[Int]) -> JoinLink:
        """The link joining `a` to `b`: a participant's own, or an inner join
        on `keys`. Undefined unless `joinable(a, b)`."""
        var own = self.attaching(a, b)
        if own >= 0:
            return self.links[own - 1].copy()
        var keys = self.keys(a, b)
        return JoinLink(
            JOIN_INNER, JOIN_ALL, BUILD_LEFT, keys[0].copy(), keys[1].copy()
        )

    # -- filters -------------------------------------------------------------
    def holds(self, f: JoinFilter, s: Set[Int]) -> Bool:
        """Is filter `f` evaluated within every tree over `s`? When `s` holds what it reads, unless a link
        stands between them that the filter may not cross: one over the
        participant it joins alone that does not pass its right side, or one
        at or below the bound, not passing its left side, over participants
        all before it. A filter is evaluated at the lowest node holding it.
        """
        # A filter reading no column, a parameter test constant per
        # execution, goes toward participant 0.
        var need = f.participants()
        if len(need) == 0:
            need.add(0)
        if len(s) == 0 or not need <= s:
            return False
        if len(s) == 1:
            for i in s:
                if i > 0 and not self.links[i - 1].passes(False):
                    return False
        var last = 0
        for i in need:
            last = max(last, i)
        for i in range(1, min(f.bound, len(self.links)) + 1):
            if (
                not self.links[i - 1].passes(True)
                and last < i
                and i not in s
            ):
                return False
        return True


struct JoinPricing(Movable):
    """What joining sets of a chain's participants produces and costs — the
    one definition `JoinChain.estimate`, a search and a tree's
    `JoinOrder.cost` all read."""

    var rules: JoinRules
    var filters: List[JoinFilter]
    var inputs: List[Estimate]
    """Each participant's estimate (`JoinChain.participant_estimates`)."""
    var _sizes: Dict[Set[Int], Size]

    def __init__(out self, chain: JoinChain) raises:
        self.rules = chain.rules()
        self.filters = chain.filters.copy()
        self.inputs = chain.participant_estimates()
        self._sizes = Dict[Set[Int], Size]()

    def size(mut self, s: Set[Int]) raises -> Size:
        """The size of `estimate(s)`, memoised: a search asks for both halves
        of every pair."""
        var hit = self._sizes.get(s)
        if hit:
            return hit.value()
        var out = self.estimate(s).size()
        self._sizes[s.copy()] = out
        return out

    def join(
        mut self, a: Set[Int], b: Set[Int], link: JoinLink
    ) raises -> Cost:
        """Joining `a` to `b` by `link`, hashing its build side, and
        evaluating the filters it is the first to hold over its rows."""
        var cost = Cost.hash_join(
            self.size(a), self.size(b), link.build_side, link.kind
        )
        var s = a | b
        for ref f in self.filters:
            if (
                self.rules.holds(f, s)
                and not self.rules.holds(f, a)
                and not self.rules.holds(f, b)
            ):
                cost = cost + Cost.per_row(self.size(s).rows)
        return cost

    def estimate(self, s: Set[Int]) raises -> Estimate:
        """What joining the participants `s` produces, every filter `s`
        holds applied — unknown when no tree joins `s`.

        They are joined one by one, from the lowest, each time the lowest
        that may join those already in, by `JoinLink.joined`: an order fixed
        by the set, so every tree over `s` is priced on one cardinality and a
        search comparing trees compares only their costs.
        """
        var low = 0
        while low not in s:
            low += 1
        var into: Set[Int] = {low}
        var out = self._filtered(self.inputs[low].copy(), into, Set[Int]())
        while into != s:
            var next = -1
            for i in range(self.rules.participants()):
                if (
                    next < 0
                    and i in s
                    and i not in into
                    and self.rules.joinable(into, {i})
                ):
                    next = i
            if next < 0:
                return Estimate()
            var one: Set[Int] = {next}
            out = self.rules.link(into, one).joined(out, self.inputs[next])
            var before = into.copy()
            into.add(next)
            out = self._filtered(out^, into, before)
        return out^

    def _filtered(
        self, var estimate: Estimate, s: Set[Int], before: Set[Int]
    ) raises -> Estimate:
        """`estimate` of `s` with the filters `s` holds and `before` did not
        applied."""
        var kept = Selectivity.everything()
        for ref f in self.filters:
            if self.rules.holds(f, s) and not self.rules.holds(f, before):
                kept = kept * f.verdict(estimate)
        return estimate.filtered(kept)

    def leaf(mut self, p: Int) raises -> Cost:
        """The filters evaluated on participant `p` alone, over its rows."""
        var one: Set[Int] = {p}
        var out = Cost()
        for ref f in self.filters:
            if self.rules.holds(f, one):
                out = out + Cost.per_row(self.size(one).rows)
        return out

    def tree(mut self, order: JoinOrder) raises -> Cost:
        """What `order`'s joins and filters take, its participants' own work
        aside: each join as it hashes, each filter at the node it lands on."""
        var held = order.participants()
        var out = Cost()
        for p in range(order.inputs):
            out = out + self.leaf(p)
        for ref j in order.joins:
            out = out + self.join(held[j.left], held[j.right], j.link)
        return out


def _templated(template: String, name: String) -> String:
    """ibis's `lname`/`rname`: `{name}` replaced, empty meaning the name."""
    if template == "":
        return name.copy()
    return template.replace("{name}", name)


struct JoinChain(Relation, Writable):
    """Every join, as ibis represents one (`ops.JoinChain`): the participants,
    one link per `.join` saying how its participant joins those before it, and
    the columns the chain answers with — one node.

    **What the answer is and what computes it are separate.** The links say
    what the answer is, and the output is `refs`, a list of participant
    columns parallel to the schema's fields, so it never depends on the order
    joins run in. Which tree runs them is physical (`planned_order`):
    left-deep in link order, unless the optimizer's `JoinOrdering` chose
    another and attached it (`with_order`).

    **Names follow ibis.** An inner key present on both sides under one name is
    emitted once; any other clash is renamed by the join's `lname`/`rname`
    (`{name}_right` by default), and a clash those leave raises at the join.

    Everything resolves by participant, never by name alone: two inputs may
    share a column name, and the chain lowers positionally, so it never has to
    tell them apart by name.
    """

    var inputs: List[ArcPointer[DynRelation]]
    var links: List[JoinLink]
    """Link `k` joins participant `k + 1` to participants `0 … k`."""
    var refs: List[JoinRef]
    """The output: the participant column behind each field of `_schema`."""
    var filters: List[JoinFilter]
    # Shared: inline, the schema makes the chain the variant's largest member
    # and every `DynRelation` move wider, in binaries that never join.
    var _schema: ArcPointer[Schema]
    var _order: ArcPointer[JoinOrder]
    """The tree the chain is computed by: left-deep as written unless the
    optimizer chose another (`with_order`). Shared, like the schema."""

    def __init__(
        out self,
        var inputs: List[ArcPointer[DynRelation]],
        var links: List[JoinLink],
        var refs: List[JoinRef],
        names: List[String],
        var filters: List[JoinFilter],
    ) raises:
        if len(refs) != len(names):
            raise InvalidError(
                t"join: {len(refs)} columns but {len(names)} names"
            )
        if len(links) == 0 or len(links) != len(inputs) - 1:
            raise InvalidError(
                t"join: {len(links)} links for {len(inputs)} inputs"
            )
        var schemas = List[Schema](capacity=len(inputs))
        for ref inp in inputs:
            schemas.append(inp[].schema())
        var fields = List[Field](capacity=len(refs))
        for i in range(len(refs)):
            ref r = refs[i]
            if r.input < 0 or r.input >= len(inputs):
                raise IndexError(t"join: no input {r.input}")
            ref src = schemas[r.input]
            var at = src.get_field_index(r.name)
            if at < 0:
                raise KeyError(
                    t"join: input {r.input} has no column '{r.name}'"
                )
            ref f = src.fields[at]
            fields.append(
                Field(
                    names[i].copy(),
                    f.dtype.copy(),
                    f.nullable,
                    f.metadata.copy(),
                )
            )
        # Link `k` reads participants up to `k` and joins `k + 1`; a filter
        # reads participants up to its bound.
        for k in range(len(links)):
            ref link = links[k]
            if len(link.left_keys) == 0 or len(link.left_keys) != len(
                link.right_keys
            ):
                raise InvalidError(t"join: link {k} has unpaired keys")
            for ref r in link.left_keys:
                if r.input < 0 or r.input > k:
                    raise InvalidError(t"join: link {k} reads {r} on the left")
            for ref r in link.right_keys:
                if r.input != k + 1:
                    raise InvalidError(t"join: link {k} reads {r} on the right")
        for ref f in filters:
            if f.bound < 0 or f.bound >= len(inputs):
                raise InvalidError(t"join: a filter bounded by #{f.bound}")
            for ref r in f.refs:
                if r.input > f.bound:
                    raise InvalidError(
                        t"join: a filter bounded by #{f.bound} reads {r}"
                    )
        self._schema = ArcPointer(schema(fields^))
        self.inputs = inputs^
        self.links = links^
        self.refs = refs^
        self.filters = filters^
        self._order = ArcPointer(
            JoinOrder.written(len(self.inputs), self.links)
        )

    # -- building ------------------------------------------------------------
    @staticmethod
    def _columns_of(relation: DynRelation, input: Int) raises -> List[JoinRef]:
        """`relation`'s columns as participant `input`'s. A participant is
        read by column name, so it may not repeat one."""
        var fields = relation.schema().fields.copy()
        var out = List[JoinRef](capacity=len(fields))
        for i in range(len(fields)):
            for j in range(i):
                if fields[i].name == fields[j].name:
                    raise InvalidError(
                        t"join: input {input} has '{fields[i].name}' twice"
                    )
            out.append(JoinRef(input, fields[i].name.copy()))
        return out^

    @staticmethod
    def joined(
        left: DynRelation,
        right: DynRelation,
        left_keys: List[Int],
        right_keys: List[Int],
        kind: JoinKind,
        strictness: UInt8,
        build_side: JoinBuildSide,
        lname: String,
        rname: String,
    ) raises -> JoinChain:
        """`left` joined to `right` — extending `left` when it is a chain, the
        way ibis's `Join.join` does, with ibis's name rules.

        `right` is one participant, a chain included, as in ibis:
        `MergeJoinChains` splices a nested chain in, so a chain written bushy
        is one the planner can reorder. Doing it here cost every binary that
        joins the splicing code whether or not it ever nests a chain.
        """
        if len(left_keys) != len(right_keys):
            raise InvalidError(
                t"join: {len(left_keys)} left keys but {len(right_keys)} right "
                t"keys"
            )
        if len(left_keys) == 0:
            raise InvalidError("join: needs at least one key pair")

        # The left side as a chain's pieces: its own when it is one,
        # otherwise a first participant outputting its columns.
        var inputs: List[ArcPointer[DynRelation]]
        var links: List[JoinLink]
        var filters: List[JoinFilter]
        var left_refs: List[JoinRef]
        if left.isa[JoinChain]():
            ref c = left.get[JoinChain]()
            inputs = c.inputs.copy()
            links = c.links.copy()
            filters = c.filters.copy()
            left_refs = c.refs.copy()
        else:
            inputs = [ArcPointer(left.copy())]
            links = List[JoinLink]()
            filters = List[JoinFilter]()
            left_refs = Self._columns_of(left, 0)
        var left_schema = left.schema()
        var left_names = left_schema.names()
        var p = len(inputs)
        inputs.append(ArcPointer(right.copy()))
        var right_refs = Self._columns_of(right, p)
        var right_schema = right.schema()
        var right_names = right_schema.names()

        var lk = List[JoinRef](capacity=len(left_keys))
        for idx in left_keys:
            if idx < 0 or idx >= len(left_refs):
                raise IndexError(t"join: left key {idx} out of range")
            lk.append(left_refs[idx].copy())
        var rk = List[JoinRef](capacity=len(right_keys))
        for idx in right_keys:
            if idx < 0 or idx >= len(right_refs):
                raise IndexError(t"join: right key {idx} out of range")
            rk.append(right_refs[idx].copy())
        links.append(JoinLink(kind, strictness, build_side, lk^, rk^))

        # ibis's `disambiguate_fields`, raising where it would leave a clash.
        var names = List[String]()
        var refs = List[JoinRef]()
        if kind == JOIN_SEMI or kind == JOIN_ANTI:
            names = left_names.copy()
            refs = left_refs.copy()
        elif kind == JOIN_RIGHT_SEMI or kind == JOIN_RIGHT_ANTI:
            names = right_names.copy()
            refs = right_refs.copy()
        else:
            # An outer or existence link equates nothing its output can rely
            # on — padded rows are NULL on one side — so only inner ones do.
            var inner = kind == JOIN_INNER
            var classes = JoinClasses()
            for ref l in links:
                if l.kind == JOIN_INNER:
                    classes.union_all(l.left_keys, l.right_keys)
            for i in range(len(left_names)):
                var name = left_names[i].copy()
                var j = right_schema.get_field_index(name)
                if j >= 0 and not (
                    inner and classes.connected(left_refs[i], right_refs[j])
                ):
                    name = _templated(lname, name)
                Self._unclashed(names, name)
                names.append(name^)
                refs.append(left_refs[i].copy())
            for j in range(len(right_names)):
                var name = right_names[j].copy()
                var i = left_schema.get_field_index(name)
                if i >= 0:
                    if inner and classes.connected(left_refs[i], right_refs[j]):
                        continue
                    name = _templated(rname, name)
                Self._unclashed(names, name)
                names.append(name^)
                refs.append(right_refs[j].copy())
        return JoinChain(inputs^, links^, refs^, names, filters^)

    @staticmethod
    def _unclashed(names: List[String], name: String) raises:
        """Raise when `name` is already one of the join's output `names`."""
        if name in names:
            raise InvalidError(
                t"join: two columns would be named '{name}'; tell them apart"
                t" with `lname` or `rname`"
            )

    # -- the output ----------------------------------------------------------
    def names(self) -> List[String]:
        return self._schema[].names()

    def ref_of(self, name: String) raises KeyError -> JoinRef:
        """The participant column behind output `name`."""
        var at = self._schema[].get_field_index(name)
        if at < 0:
            raise KeyError(t"join: no output column '{name}'")
        return self.refs[at].copy()

    def rules(self) -> JoinRules:
        """The trees that compute this chain's answer (`JoinRules`)."""
        return JoinRules(self.links)

    # -- rebuilding ----------------------------------------------------------
    def _rebuilt(
        self,
        var inputs: List[ArcPointer[DynRelation]],
        var refs: List[JoinRef],
        names: List[String],
        var filters: List[JoinFilter],
    ) raises -> JoinChain:
        """This chain's links and order over `inputs`, answering with `refs`
        under `names`, with `filters`."""
        var out = JoinChain(inputs^, self.links.copy(), refs^, names, filters^)
        out._order = self._order
        return out^

    def with_output(
        self, var refs: List[JoinRef], names: List[String]
    ) raises -> JoinChain:
        """This chain answering with the participant columns `refs` under
        `names`, everything else kept."""
        return self._rebuilt(
            self.inputs.copy(), refs^, names, self.filters.copy()
        )

    def with_inputs(
        self, var inputs: List[ArcPointer[DynRelation]]
    ) raises -> JoinChain:
        """This chain over rewritten participants, everything else kept."""
        return self._rebuilt(
            inputs^, self.refs.copy(), self.names(), self.filters.copy()
        )

    def with_filters(self, var filters: List[JoinFilter]) raises -> JoinChain:
        """This chain evaluating `filters` between its joins, everything else
        kept."""
        return self._rebuilt(
            self.inputs.copy(), self.refs.copy(), self.names(), filters^
        )

    def with_order(self, var order: JoinOrder) raises -> JoinChain:
        """This chain computed by the tree `order` — how `JoinOrdering` hands
        a chain the tree it chose. Raises unless `order` computes this chain's
        answer (`JoinOrder.verify`)."""
        order.verify(self)
        var out = self.copy()
        out._order = ArcPointer(order^)
        return out^

    # -- Relation ------------------------------------------------------------
    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        """Every participant — a chain has as many inputs as it joins."""
        var inputs = List[ArcPointer[DynRelation]](capacity=len(self.inputs))
        for ref inp in self.inputs:
            inputs.append(ArcPointer(f(inp[])))
        return self.with_inputs(inputs^)

    def references(self, mut into: References):
        for ref inp in self.inputs:
            inp[].references(into)
        for ref f in self.filters:
            f.predicate.references(into)

    def schema(self) -> Schema:
        return self._schema[].copy()

    def estimate(self) raises -> Estimate:
        """The rows this chain answers with: `JoinPricing.estimate` of every
        participant — so no tree moves it."""
        var pricing = JoinPricing(self)
        var out = pricing.estimate(pricing.rules.everything())
        return out.select(JoinRef.keys(self.refs), self.schema())

    def cost(self) raises -> Cost:
        """What running this chain along its planned tree should take; see
        `JoinOrder.cost`."""
        var pricing = JoinPricing(self)
        var out = pricing.tree(self._order[])
        for ref input in self.inputs:
            out = out + input[].cost()
        return out

    def participant_estimates(self) raises -> List[Estimate]:
        """Each participant's estimate, its columns keyed by participant
        column (`JoinRef.key`), since two participants may share a name."""
        var out = List[Estimate](capacity=len(self.inputs))
        for p in range(len(self.inputs)):
            var est = self.inputs[p][].estimate()
            for ref c in est.columns:
                c.name = JoinRef(p, c.name.copy()).key()
            out.append(est^)
        return out^

    # -- lowering ------------------------------------------------------------
    def planned_order(self) -> JoinOrder:
        """The tree this chain is computed by: the one the optimizer chose,
        or left-deep in link order."""
        return self._order[].copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        """The planned tree, bottom-up: each participant's pipeline and each
        join's `JoinOperator` — probe side the pipeline, build side a stage —
        with the filters landing on it after it, and at the root the chain's
        output picked by position."""
        ref order = self._order[]
        var landings = List[Int](capacity=len(self.filters))
        if len(self.filters) > 0:
            var held = order.participants()
            var rules = self.rules()
            for ref f in self.filters:
                landings.append(order.landing(held, rules, f))
        var cols = List[JoinRef]()
        return self._lower(order.root(), landings, ctx, bindings, cols)

    def _fields(self, cols: List[JoinRef]) raises -> Schema:
        """The participants' own fields behind `cols`, as they are: names may
        repeat, since everything below reads by position."""
        var fields = List[Field](capacity=len(cols))
        for ref c in cols:
            fields.append(
                self.inputs[c.input][].schema().field(name=c.name).copy()
            )
        return schema(fields^)

    @staticmethod
    def _positions(
        cols: List[JoinRef], want: List[JoinRef]
    ) raises -> List[Int]:
        var out = List[Int](capacity=len(want))
        for ref w in want:
            var at = -1
            for i in range(len(cols)):
                if cols[i] == w:
                    at = i
                    break
            if at < 0:
                raise InternalError(t"join: column {w} is not in its input")
            out.append(at)
        return out^

    def _stages(
        self,
        node: Int,
        cols: List[JoinRef],
        landings: List[Int],
        ctx: ExecContext,
        bindings: Bindings,
    ) raises -> List[DynOperator]:
        """The filters landing on `node`, in chain order, each reading its
        columns at their positions in `cols`."""
        var out = List[DynOperator]()
        for i in range(len(self.filters)):
            if landings[i] == node:
                ref f = self.filters[i]
                out.append(
                    f.stage(
                        self._fields(f.refs).fields,
                        Self._positions(cols, f.refs),
                        ctx,
                        bindings,
                    )
                )
        return out^

    def _lower(
        self,
        node: Int,
        landings: List[Int],
        ctx: ExecContext,
        bindings: Bindings,
        mut cols: List[JoinRef],
    ) raises -> Pipeline:
        """`node` of the planned tree as a pipeline, and in `cols` the
        participant column each of its output positions holds — the chain's
        own output at the root."""
        ref order = self._order[]
        if not order.is_join(node):
            var pipe = self.inputs[node][].to_operator(ctx, bindings)
            cols = List[JoinRef]()
            for ref f in self.inputs[node][].schema().fields:
                cols.append(JoinRef(node, f.name.copy()))
            var stages = self._stages(node, cols, landings, ctx, bindings)
            while len(stages) > 0:
                pipe.append(stages.pop(0))
            return pipe^

        ref j = order.join(node)
        var lcols = List[JoinRef]()
        var rcols = List[JoinRef]()
        var left = self._lower(j.left, landings, ctx, bindings, lcols)
        var right = self._lower(j.right, landings, ctx, bindings, rcols)
        var left_on = Self._positions(lcols, j.link.left_keys)
        var right_on = Self._positions(rcols, j.link.right_keys)
        cols = List[JoinRef]()
        if j.link.kind.emits_left_columns():
            cols.extend(lcols.copy())
        if j.link.kind.emits_right_columns():
            cols.extend(rcols.copy())
        var joined = self._fields(cols)
        var stages = self._stages(node, cols, landings, ctx, bindings)

        # The root answers with the chain's output, picked by position: in
        # the join, or after the filters landing on it, which may read a
        # column the output drops.
        var root = node == order.root()
        var pick = Self._positions(cols, self.refs) if root else List[Int]()
        var select: List[Int]
        var output: Schema
        if root and len(stages) == 0:
            select = pick.copy()
            output = self.schema()
        else:
            select = List[Int](capacity=len(cols))
            for i in range(len(cols)):
                select.append(i)
            output = joined.copy()
        if root:
            cols = self.refs.copy()

        # Two arrangements, spelled out: picking the roles into locals and
        # building one `JoinOperator` measured +3,840 bytes on `query_join`.
        var pipe: Pipeline
        if j.link.build_side == BUILD_LEFT:
            right.append(
                JoinOperator(
                    left^,
                    left_on^,
                    right_on^,
                    j.link.kind,
                    j.link.strictness,
                    j.link.build_side,
                    joined^,
                    self._fields(lcols),
                    self._fields(rcols),
                    select^,
                    output^,
                    ctx.copy(),
                )
            )
            pipe = right^
        else:
            left.append(
                JoinOperator(
                    right^,
                    right_on^,
                    left_on^,
                    j.link.kind,
                    j.link.strictness,
                    j.link.build_side,
                    joined^,
                    self._fields(rcols),
                    self._fields(lcols),
                    select^,
                    output^,
                    ctx.copy(),
                )
            )
            pipe = left^
        if len(stages) > 0 and root:
            stages.append(self.filters[0].selection(pick, self.schema()))
        while len(stages) > 0:
            pipe.append(stages.pop(0))
        return pipe^

    # -- rendering -----------------------------------------------------------
    def write_to[W: Writer](self, mut writer: W):
        """`Join(inputs | links | where filters | order tree | output)`: links
        name participants `#i` and keys `#i.column`. The order shows only when
        a cost-based planner chose a tree other than the written one."""
        writer.write("Join(")
        for i in range(len(self.inputs)):
            if i > 0:
                writer.write(", ")
            writer.write(self.inputs[i][])
        writer.write(" | #0")
        for ref l in self.links:
            writer.write(" ", l)
        for i in range(len(self.filters)):
            writer.write(" | where " if i == 0 else " and ")
            writer.write(self.filters[i].predicate)
        if self._order[] != JoinOrder.written(len(self.inputs), self.links):
            writer.write(" | order ", self._order[])
        writer.write(" | ")
        for i in range(len(self.refs)):
            if i > 0:
                writer.write(", ")
            writer.write(self._schema[].fields[i].name)
            if self.refs[i].name != self._schema[].fields[i].name:
                writer.write("=", self.refs[i])
        writer.write(")")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


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
            return self.statistics.value()[].select(
                self._schema.names(), self._schema
            )
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
        return Cost.source(estimate.size())

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
        return Cost.source(estimate.size())

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
        return Cost.source(estimate.size())

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


@fieldwise_init
struct _IcebergSource(Movable):
    """What an `IcebergScan` holds, behind an `ArcPointer`, so copying a plan
    copies one pointer rather than the table's metadata."""

    var table: IcebergTable
    var snapshot_id: Optional[Int]
    """`None` for a table with no snapshot yet, which reads no rows."""
    var schema: Schema
    var pruners: List[DynValue]


struct IcebergScan(FileScan, Writable):
    """One snapshot of an Apache Iceberg table, one row group at a time, in
    the schema it is read with — the snapshot's own, columns matched to data
    files by field id.

    The table's metadata is loaded when the scan is built (`scan_iceberg`);
    its manifests are read when it first runs. `pruners` are predicates
    `PushFilterIntoScan` put here, used to skip data files whose recorded
    bounds and partition values cannot match.
    """

    var _source: ArcPointer[_IcebergSource]

    def __init__(
        out self,
        var table: IcebergTable,
        snapshot_id: Optional[Int],
        var schema: Schema,
        var pruners: List[DynValue] = [],
    ):
        self._source = ArcPointer(
            _IcebergSource(table^, snapshot_id, schema^, pruners^)
        )

    def table(self) -> ref[self._source[].table] IcebergTable:
        return self._source[].table

    def snapshot_id(self) -> Optional[Int]:
        return self._source[].snapshot_id

    def pruners(self) -> ref[self._source[].pruners] List[DynValue]:
        return self._source[].pruners

    def references(self, mut into: References):
        for ref p in self._source[].pruners:
            p.references(into)

    def schema(self) -> Schema:
        return self._source[].schema.copy()

    def with_schema(self, var schema: Schema) -> Self:
        ref s = self._source[]
        return Self(s.table.copy(), s.snapshot_id, schema^, s.pruners.copy())

    def with_pruners(self, var pruners: List[DynValue]) -> Self:
        ref s = self._source[]
        return Self(s.table.copy(), s.snapshot_id, s.schema.copy(), pruners^)

    def estimate(self) raises -> Estimate:
        return Estimate.unknown(self._source[].schema)

    def cost(self) raises -> Cost:
        var estimate = self.estimate()
        return Cost.source(estimate.size())

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
    ) raises -> Pipeline:
        ref s = self._source[]
        return Pipeline(
            IcebergScanOperator(
                s.table.copy(),
                s.snapshot_id,
                s.schema.copy(),
                ctx.copy(),
                s.pruners.copy(),
                bindings.copy(),
            )
        )

    def write_to[W: Writer](self, mut writer: W):
        ref s = self._source[]
        writer.write("IcebergScan(", s.table.root)
        if s.snapshot_id:
            writer.write("@", s.snapshot_id.value())
        writer.write(")")
        if len(s.pruners):
            writer.write(" pruned by ", len(s.pruners))

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)
