"""The logical layer: an immutable description of a query.

Paired with `physical.mojo`, which holds what these become when they run, and
with `optimizer.mojo`, which rewrites one plan into a simpler one.

A `Relation` says *what* to compute. It owns nothing that runs, so a plan is
freely copyable, shareable, inspectable and rewritable. `to_operator(ctx)` turns
it into the physical operator that owns the running state.

**A variant for inspection, a trampoline for lowering.** `DynRelation` erases
the ten node types behind a `Variant`, so `isa[R]()`/`get[R]()` let an
optimizer rule read a real typed node and construct one — the capability the
previous trampoline-only box lacked, which left its "rules" as four comptime
flags and eight scattered calls inside `to_operator` with no file to read.

Lowering is the exception, and the reason is measured. `_dispatch` resolves the
active member with a `comptime for` over every member, so anything routed
through it is instantiated ten times; for `to_operator` that makes
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
from std.collections import Set
from std.memory import ArcPointer
from std.os import abort
from std.utils import Variant

from ..arrays import DynArray
from ..execution import ExecContext
from ..kernels.join import JoinKind, JOIN_INNER
from ..kernels.window import (
    DenseRank,
    Edge,
    FirstValue,
    Lag,
    LastValue,
    Lead,
    Offset,
    Rank,
    RowNumber,
    WindowExtents,
    WindowFunction,
)
from ..schema import Schema, schema
from ..tabular import RecordBatch
from ..dtypes import DynType, Field, field, int64
from .bindings import Bindings
from .pruning import PrunePredicate, Prunable
from .pushdown import Pushdown
from .optimizer import RuleSet, optimize
from .runtime.values import column
from .physical import (
    Datum,
    GroupByOperator,
    BatchSourceOperator,
    DynOperator,
    LimitOperator,
    Pipeline,
    FilterOperator,
    JoinOperator,
    ParquetScanOperator,
    ProjectOperator,
    SortOperator,
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
    `reject_aggregate` and raise: `Filter`'s predicate, `Project`'s values,
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

    def constant_bool(self) -> Optional[Bool]:
        """`True`/`False` if this is a constant boolean, else `None`.

        **A trait default, deliberately not a `DynValue` slot.** It is read at
        the `.filter()` verb, where the concrete type is still visible, and the
        answer is stored on the `Filter` node — the same placement argument
        `PrunePredicate` makes, and for the same reason: a slot on `DynValue`
        is paid for every projection value, every sort key and every aggregate
        input in the program, to serve the one caller that filters.

        Defaulted to `None` — "not known to be constant" — so a node that has
        not been taught costs an optimization and never an answer. Only
        `RuntimeValue` overrides it; a comptime literal could too, but a fused
        predicate that is constant is a program someone wrote by hand.
        """
        return None

    # -- the window surface -------------------------------------------------
    #
    # Five trait **defaults**, so both lanes get them from one definition and
    # neither pays for a window function it never names. They return
    # `WindowFn`/`WindowExpr` — concrete types no conformer overrides, which
    # is what keeps them out of the "trait default whose return type a
    # conformer must change" trap: the hazard is a conformer needing a
    # *different* return type, and here nobody does.
    #
    # `over` is the aggregate entry point and the other four are the
    # non-aggregate ones, which is why `over` is not simply a fifth verb here:
    # `col("v", int64).sum()` is already a `Value`, so it needs a way to say
    # "and evaluate that over a frame", while `lag` has no aggregate to name.

    def lag(self, offset: Int = 1) -> WindowFn:
        """`LAG(self, offset)` — this column read `offset` rows earlier."""
        var boxed: Optional[DynValue] = DynValue(self.copy())
        return WindowFn.of[Lag](boxed^, -offset)

    def lead(self, offset: Int = 1) -> WindowFn:
        """`LEAD(self, offset)` — this column read `offset` rows later."""
        var boxed: Optional[DynValue] = DynValue(self.copy())
        return WindowFn.of[Lead](boxed^, offset)

    def first_value(self) -> WindowFn:
        """`FIRST_VALUE(self)` — this column at the frame's first row."""
        var boxed: Optional[DynValue] = DynValue(self.copy())
        return WindowFn.of[FirstValue](boxed^)

    def last_value(self) -> WindowFn:
        """`LAST_VALUE(self)` — this column at the frame's last row.

        Under the default frame that is the *current* row, not the partition's
        last. See `WindowFrame`.
        """
        var boxed: Optional[DynValue] = DynValue(self.copy())
        return WindowFn.of[LastValue](boxed^)

    def over(
        self,
        var partition_by: List[DynValue] = List[DynValue](),
        var order_by: List[DynValue] = List[DynValue](),
        var ascending: List[Bool] = List[Bool](),
        nulls_first: Bool = True,
        var rows: Optional[Tuple[Int, Int]] = None,
    ) raises -> WindowExpr:
        """`self OVER (...)` — evaluate this aggregate over a frame.

        Raises unless `self` is an aggregate: a per-row value has nothing to
        do with a frame, and `col("v", int64).over(...)` is a mistake worth a
        diagnostic rather than a silent column of copies.
        """
        var boxed: Optional[DynValue] = DynValue(self.copy())
        return WindowFn.aggregate(boxed^).over(
            partition_by^, order_by^, ascending^, nulls_first, rows^
        )

    def columns(self) -> List[String]:
        """Which columns this expression reads, deduplicated, first-seen
        order."""
        ...

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
        carries them and hands them back down to `bind`, where a `Param` reads
        them. That is why a plan holds no parameter state and two executions
        of it cannot interfere.
        """
        ...


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

    Six function slots — `columns`, `name`, `dtype`, `write`, `to_operator`
    and `_drop` — plus two constant fields, `shape` and `aggregates`, read once
    at construction because both are comptime constants. `_drop` is the
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
    var _columns: def(ArcPointer[NoneType]) thin -> List[String]
    var _name: def(ArcPointer[NoneType]) thin -> String
    var _dtype: def(ArcPointer[NoneType], Schema) thin raises -> DynType
    var _write: def(ArcPointer[NoneType]) thin -> String
    var _to_operator: def(
        ArcPointer[NoneType], Schema, Bool, Bindings
    ) thin raises -> DynOperator
    var _shape: Shape
    var _aggregates: Bool
    var _drop: def(var ArcPointer[NoneType]) thin
    """Erasure forgets the pointee's destructor; this carries it. See
    `DynOperator._virt_drop` for why the release has to happen at the true
    type, and for the probe that measured it."""

    # -- trampolines --------------------------------------------------------
    # One instantiation per boxed type, wired at construction. There is no
    # registry and nothing names every value type in one place, so a type that
    # is never boxed costs nothing in the binary.

    @staticmethod
    def _columns_tramp[V: Value](ptr: ArcPointer[NoneType]) -> List[String]:
        return rebind[ArcPointer[V]](ptr)[].columns()

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
        self._columns = Self._columns_tramp[V]
        self._name = Self._name_tramp[V]
        self._dtype = Self._dtype_tramp[V]
        self._write = Self._write_tramp[V]
        self._to_operator = Self._to_operator_tramp[V]
        self._shape = V.shape

    def __deinit__(deinit self):
        self._drop(self._boxed^)

    # -- the erased surface -------------------------------------------------

    def columns(self) -> List[String]:
        return self._columns(self._boxed)

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
# merged — order-preserving union, the shape `columns()` folds with
# ---------------------------------------------------------------------------
# `columns()` folds a node's children into a list that is deduplicated but
# **order-preserving**: first-seen order is part of the contract
# (`test_runtime_columns_are_deduped_in_first_seen_order` asserts it), so a
# plain `Set` cannot answer it on its own.
#
# A `List` carries the order and a `Set` answers membership, which also makes
# this linear — the loop it replaces rescanned the accumulated list once per
# candidate, so a wide expression was quadratic in its own column count.
#
# `bind` and `validity` fold too and are deliberately *not* here: each is
# already a single expression — a tuple and an intersect — so there is nothing
# to extract. They could not be defaulted onto a trait anyway, since a default
# returning `Self.Bound` needs that type to be `ImplicitlyCopyable`, which
# marrow's array types deliberately are not.
def merged(var into: List[String], extra: List[String]) -> List[String]:
    """`into`, followed by whatever in `extra` it does not already contain."""
    var seen = Set[String]()
    for ref n in into:
        seen.add(n.copy())
    for ref n in extra:
        if n not in seen:
            seen.add(n.copy())
            into.append(n.copy())
    return into^


def reject_aggregate(
    value: DynValue, node: StringSlice, name: StringSlice, remedy: StringSlice
) raises:
    """Refuse an aggregate in a position that is evaluated once per row.

    An aggregate's operator answers `None` to every `push` and yields only at
    `drain`, so a per-row consumer unwraps that `None` and aborts the process
    rather than raising. Four node positions are per-row — `Filter`'s
    predicate, `Project`'s values, `Aggregate`'s **keys** (its `aggs` are the
    point) and `Sort`'s keys — and each calls this. `remedy` names the way to
    say what the caller meant, which differs per node.

    Two of the four carried their own copy of this check and two did not, which
    is the shape the guard exists to prevent: `rel.sort_by([col("a",
    int64).sum()], [True])` aborted.
    """
    if value.aggregates():
        raise Error(
            node,
            ": '",
            name,
            "' is an aggregate, which has no value per row; ",
            remedy,
        )


# ---------------------------------------------------------------------------
# Window functions — the description
# ---------------------------------------------------------------------------
struct WindowFrame(Copyable, ImplicitlyCopyable, Movable, Writable):
    """Which rows of the partition an aggregate window function sees.

    Two forms, and the difference between them is the one thing about frames
    that reliably surprises:

    - the **default** — `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`,
      which SQL applies whenever a window has an `ORDER BY` and no explicit
      frame. `RANGE` counts *peers*, so "current row" means the end of the
      current row's peer group, and tied rows all see the same frame.
    - **`ROWS`**, which counts rows, so tied rows see different frames.

    The two agree only when the `ORDER BY` key has no duplicates, which is why
    `window_explicit_rows_frame` exists as a separate golden case from
    `window_partitioned_running_sum`.

    With no `ORDER BY` at all the default frame is the whole partition. That
    falls out here rather than being special-cased: with no order key every row
    is a peer of every other, so the peer group *is* the partition.
    """

    var is_rows: Bool
    """`True` for `ROWS`, `False` for the default `RANGE`."""

    var preceding: Int
    """`ROWS` only: rows before the current one, as a non-positive offset."""

    var following: Int
    """`ROWS` only: rows after the current one, as a non-negative offset."""

    def __init__(out self, is_rows: Bool, preceding: Int, following: Int):
        self.is_rows = is_rows
        self.preceding = preceding
        self.following = following

    @staticmethod
    def default() -> WindowFrame:
        """`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`."""
        return WindowFrame(False, 0, 0)

    def __eq__(self, other: Self) -> Bool:
        return (
            self.is_rows == other.is_rows
            and self.preceding == other.preceding
            and self.following == other.following
        )

    def write_to[W: Writer](self, mut writer: W):
        if self.is_rows:
            writer.write("rows ", self.preceding, "..", self.following)
        else:
            writer.write("range unbounded..current")


struct WindowFn(Copyable, Movable, Writable):
    """A window function before it is told which window to run over.

    Split from `WindowExpr` because the two are written apart: `row_number()`
    and `col("v", int64).lag()` name the function, and `.over(...)` names the
    window. Keeping them one type would mean either a constructor taking
    everything at once — which is not the spelling SQL uses — or a half-built
    `WindowExpr` with a meaningless window.
    """

    var _compute: Optional[
        def(
            WindowExtents, Optional[DynArray], Int, Bool, Int, Int, ExecContext
        ) thin raises -> DynArray
    ]
    """The function itself, as a pointer instantiated where the verb names it.

    **This is what keeps an unnamed window function out of the binary.** A
    tag read by one `if/elif` chain in `WindowOperator` would link all seven
    bodies into any binary using any window; a slot links only the one the
    caller wrote. It is the shape `DynRelation._virt_to_operator` uses, and
    it is safe here for the reason it is safe there: `WindowFn` is not
    self-referential, which is the condition the miscompile in
    `runtime/values.mojo` needed.

    `None` means **the aggregate**, the one open kind — its argument is an
    ordinary aggregate `Value` evaluated over each frame, so it runs through
    that value's own operator rather than through a window kernel."""

    var _name: String
    """How this renders. A stored string, because with the kinds behind a
    pointer there is no tag left to switch on — and rendering a name is not
    worth a second slot."""

    var _ranks: Bool
    """Whether this reads only the ordering, so it takes no argument and
    always answers `int64` and never null. True for the three ranking
    functions."""

    var argument: Optional[DynValue]
    """What the function reads: the shifted column for `lag`/`lead`, the framed
    column for `first_value`/`last_value`, the aggregate itself for the
    aggregate kind, and nothing for the three ranking functions."""

    var offset: Int
    """`lag`/`lead` distance, signed: negative looks back, positive forward."""

    @staticmethod
    def _tramp[
        F: WindowFunction
    ](
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        return F.compute(
            extents, argument, offset, is_rows, preceding, following, ctx
        )

    @staticmethod
    def of[
        F: WindowFunction
    ](var argument: Optional[DynValue], offset: Int = 0) -> Self:
        """A window function backed by `F`.

        Takes only what `F` cannot answer for itself: the operand and the
        offset. Its name and whether it ranks are properties of the function,
        so they come off `F` rather than being repeated at every verb.
        """
        return Self(Self._tramp[F], F.name(), F.ranks(), argument^, offset)

    @staticmethod
    def aggregate(var argument: Optional[DynValue]) -> Self:
        """The open kind: an ordinary aggregate, evaluated over each frame."""
        return Self(None, String("agg"), False, argument^, 0)

    def __init__(
        out self,
        var compute: Optional[
            def(
                WindowExtents,
                Optional[DynArray],
                Int,
                Bool,
                Int,
                Int,
                ExecContext,
            ) thin raises -> DynArray
        ],
        var name: String,
        ranks: Bool,
        var argument: Optional[DynValue],
        offset: Int = 0,
    ):
        self._compute = compute^
        self._name = name^
        self._ranks = ranks
        self.argument = argument^
        self.offset = offset

    def ranks(self) -> Bool:
        """Whether this reads only the ordering — so it takes no argument."""
        return self._ranks

    def is_aggregate(self) -> Bool:
        """Whether this is the open kind, run through its own operator."""
        return not self._compute

    def over(
        self,
        var partition_by: List[DynValue] = List[DynValue](),
        var order_by: List[DynValue] = List[DynValue](),
        var ascending: List[Bool] = List[Bool](),
        nulls_first: Bool = True,
        var rows: Optional[Tuple[Int, Int]] = None,
    ) raises -> WindowExpr:
        """`OVER (PARTITION BY ... ORDER BY ...)` — the window this runs in.

        `ascending` defaults to all-ascending, sized to `order_by`, so the
        common case names only the keys. `rows` supplies an explicit `ROWS`
        frame as `(preceding, following)`; without it the default `RANGE`
        frame applies.
        """
        var dirs = ascending^
        if len(dirs) == 0:
            for _ in range(len(order_by)):
                dirs.append(True)
        if len(dirs) != len(order_by):
            raise Error(
                "over: ",
                len(order_by),
                " order keys but ",
                len(dirs),
                " directions",
            )
        var frame = WindowFrame.default()
        if rows:
            ref bounds = rows.value()
            frame = WindowFrame(True, bounds[0], bounds[1])
        return WindowExpr(
            self.copy(), partition_by^, order_by^, dirs^, nulls_first, frame
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._name, "(")
        if self.argument:
            writer.write(self.argument.value())
        writer.write(")")


struct WindowExpr(Copyable, Movable, Writable):
    """A window function together with the window it runs over.

    **Deliberately not a `Value`**, and that is the entry's point. A `Value`
    answers per row from the batch in front of it; a window function's answer
    depends on rows that may sit in another morsel entirely, so there is no
    honest `to_operator` for one. `Value.aggregates` exists because the
    relations that cannot accept an aggregate had no way to say so — a window
    function is *more* restricted than an aggregate, not less, and making it a
    `Value` would put it in every position that flag exists to keep it out of.

    Not conforming is also what makes `with_columns` unambiguous: `List[
    WindowExpr]` cannot convert to `List[DynValue]`, so the two overloads
    cannot be confused for each other and no caller has to disambiguate.
    """

    var func: WindowFn
    var partition_by: List[DynValue]
    var order_by: List[DynValue]
    var ascending: List[Bool]
    var nulls_first: Bool
    var frame: WindowFrame

    def __init__(
        out self,
        var func: WindowFn,
        var partition_by: List[DynValue],
        var order_by: List[DynValue],
        var ascending: List[Bool],
        nulls_first: Bool,
        frame: WindowFrame,
    ) raises:
        if func.ranks() and func.argument:
            raise Error("over: '", func._name, "' takes no argument")
        if not func.ranks() and not func.argument:
            raise Error("over: '", func._name, "' needs an argument")
        if func.is_aggregate() and not func.argument.value().aggregates():
            raise Error(
                "over: '",
                func.argument.value().name(),
                "' is not an aggregate; only an aggregate takes a frame",
            )
        for ref k in partition_by:
            reject_aggregate(
                k, "over", k.name(), "aggregate first, then window the result"
            )
        for ref k in order_by:
            reject_aggregate(
                k, "over", k.name(), "aggregate first, then window the result"
            )
        self.func = func^
        self.partition_by = partition_by^
        self.order_by = order_by^
        self.ascending = ascending^
        self.nulls_first = nulls_first
        self.frame = frame

    def columns(self) -> List[String]:
        """Every column this reads — function argument and both key lists."""
        var out = List[String]()
        if self.func.argument:
            out = merged(out^, self.func.argument.value().columns())
        for ref k in self.partition_by:
            out = merged(out^, k.columns())
        for ref k in self.order_by:
            out = merged(out^, k.columns())
        return out^

    def dtype(self, schema: Schema) raises -> DynType:
        """The type this produces.

        `int64` for the three ranking functions — they count rows, and nothing
        about the input changes that. Everything else answers with its
        argument's type, which for `aggregate` is the aggregate's own output
        type rather than its input's.
        """
        if self.func.ranks():
            return int64
        return self.func.argument.value().dtype(schema)

    def spec(self) -> String:
        """This expression's *window*, rendered — its identity for grouping.

        Two window expressions sharing a spec can be answered by one sort, and
        `with_columns` stacks a separate `Window` node per distinct spec. A
        rendered string rather than a structural comparison because `DynValue`
        exposes `write` and not equality, and `write` is already the canonical
        rendering of a plan.
        """
        var out = String("p=")
        for ref k in self.partition_by:
            out += String(k) + ","
        out += "|o="
        for i in range(len(self.order_by)):
            out += String(self.order_by[i])
            out += "a" if self.ascending[i] else "d"
            out += ","
        out += "|n=" + String(self.nulls_first)
        return out^

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.func, " over(")
        for i in range(len(self.partition_by)):
            writer.write("partition " if i == 0 else ", ")
            writer.write(self.partition_by[i])
        for i in range(len(self.order_by)):
            writer.write(" order " if i == 0 else ", ")
            writer.write(self.order_by[i])
            writer.write(" asc" if self.ascending[i] else " desc")
        writer.write(" ", self.frame, ")")


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

    def schema(self) -> Schema:
        """The columns this relation produces.

        Computed at construction and stored, not derived on demand: a caller
        asks for it once per plan node while building the node above, and a
        `Filter` would otherwise re-derive its input's schema every time.
        """
        ...

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
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
    instantiated ten times, but one returns a stored field and the other
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
        ParquetScan,
    ]

    var _v: Self.VariantType

    var _virt_to_operator: def(
        Self.VariantType, ExecContext, Bindings, var Pushdown
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
        var pushed: Pushdown,
    ) raises -> Pipeline:
        return v[R].to_operator(ctx, bindings, pushed^)

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

    def schema(self) -> Schema:
        def job[T: Relation](node: T) raises {imm} -> Schema:
            return node.schema()

        try:
            return self._dispatch(job)
        except:
            abort("DynRelation.schema: no arm matched")

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        return self._virt_to_operator(self._v, ctx, bindings, pushed^)

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

        The erased overload: an already-boxed predicate has lost the type
        `prune` needs, so this plan filters exactly and reads every row group.
        """
        return Filter(self.copy(), predicate^)

    def filter[
        V: Value & Prunable
    ](self, var predicate: V) raises -> DynRelation:
        """Rows where `predicate` is true, **with statistics pruning**.

        The concrete type is captured here because this is the last place it is
        visible: `DynValue` erases it, and the box deliberately has no `prune`
        slot. The two overloads are disjoint — `DynValue` does not conform to
        the traits it erases — which is what lets both spellings coexist.
        """
        return Filter(
            self.copy(),
            DynValue(predicate.copy()),
            Optional(PrunePredicate(predicate.copy())),
            predicate.constant_bool(),
            predicate.conjuncts(),
        )

    def select(self, names: List[String]) raises -> DynRelation:
        """Keep these columns, in this order.

        Sugar over `project`: the values are runtime column reads, so this
        needs no dtype from the caller. That is the runtime lane earning its
        keep — a fused `Column[T]` would force `select` to be generic over
        every column's type.
        """
        var values = List[DynValue](capacity=len(names))
        for ref n in names:
            values.append(column(n.copy()))
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

        A name that already exists is **replaced in place** rather than
        appended, which is Polars' `with_columns` rule and the only one that
        keeps the output schema free of duplicates. Position is preserved:
        replacing `qty` leaves `qty` where it was.

        Sugar over `project`, like `select` — the surviving columns are runtime
        column reads, so no caller has to supply their dtypes. That is the same
        reason `select` is not generic: a fused `Column[T]` would make this
        method parametric over every column in the input.
        """
        if len(names) != len(values):
            raise Error(
                "with_columns: ",
                len(names),
                " names but ",
                len(values),
                " values",
            )
        # **A name may not appear twice in one call**, the same rule the
        # window overload enforces. Replacing in place is this overload's
        # documented behaviour, but replacing *twice* has no reading: the
        # loop below keeps the last match, so `values[0]` would be silently
        # discarded, and for a name not already present both copies would be
        # appended and the second made unreachable by `get_field_index`.
        for i in range(len(names)):
            for j in range(i):
                if names[j] == names[i]:
                    raise Error("with_columns: '", names[i], "' is named twice")
        var input_schema = self.schema()
        var out_names = List[String]()
        var out_values = List[DynValue]()
        for ref f in input_schema.fields:
            var replaced = -1
            for i in range(len(names)):
                if names[i] == f.name:
                    replaced = i
            out_names.append(f.name.copy())
            if replaced >= 0:
                out_values.append(values[replaced].copy())
            else:
                out_values.append(column(f.name.copy()))
        for i in range(len(names)):
            if input_schema.get_field_index(names[i]) == -1:
                out_names.append(names[i].copy())
                out_values.append(values[i].copy())
        return Project(self.copy(), out_names^, out_values^)

    def with_columns(
        self, var names: List[String], var exprs: List[WindowExpr]
    ) raises -> DynRelation:
        """`SELECT *, <window functions> AS <names>` — the windowed overload.

        A separate overload rather than a wider element type, because a
        `WindowExpr` deliberately is not a `Value` (see its docstring). That
        makes `List[WindowExpr]` unable to convert to `List[DynValue]`, so the
        two overloads cannot be confused and no caller disambiguates anything.

        **Expressions are grouped by window and stacked.** Each distinct
        `spec()` becomes its own `Window` node, so `rank()` and `dense_rank()`
        over one ordering share a sort while two different orderings get one
        each. Grouping preserves first-seen order, so the output columns come
        out in the order the caller wrote them whenever they share a window —
        which is every case that does not deliberately mix.

        A name that already exists **raises** rather than replacing in place.
        The `DynValue` overload can replace because a `Project` names every
        output column anyway; here the replacement would need a second node
        purely to re-order and rename, and a silently duplicated column name
        is a worse outcome than a diagnostic.
        """
        if len(names) != len(exprs):
            raise Error(
                "with_columns: ",
                len(names),
                " names but ",
                len(exprs),
                " window expressions",
            )
        var input_schema = self.schema()
        for i in range(len(names)):
            ref n = names[i]
            if input_schema.get_field_index(n) != -1:
                raise Error(
                    "with_columns: '",
                    n,
                    "' already exists; a window column cannot replace one",
                )
            # **And against each other**, which the schema check cannot see.
            # Two expressions with different specs become two stacked `Window`
            # nodes below, each built directly rather than through this verb,
            # so the second one's input schema already carries the first one's
            # name and nothing re-checks it. The result is a schema with the
            # column twice, where `get_field_index` answers with the first and
            # the second is unreachable by name -- exactly the silent
            # duplication this overload raises to avoid.
            for j in range(i):
                if names[j] == n:
                    raise Error("with_columns: '", n, "' is named twice")
        var current = self.copy()
        var placed = List[Bool](length=len(exprs), fill=False)
        for i in range(len(exprs)):
            if not placed[i]:
                var group_names = List[String]()
                var group_exprs = List[WindowExpr]()
                for j in range(i, len(exprs)):
                    if not placed[j] and exprs[j].spec() == exprs[i].spec():
                        placed[j] = True
                        group_names.append(names[j].copy())
                        group_exprs.append(exprs[j].copy())
                current = Window(current^, group_names^, group_exprs^)
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
                raise Error("drop: column '", n, "' not found in schema")
        var out_names = List[String]()
        var out_values = List[DynValue]()
        for ref f in input_schema.fields:
            var dropped = False
            for ref n in names:
                if n == f.name:
                    dropped = True
            if not dropped:
                out_names.append(f.name.copy())
                out_values.append(column(f.name.copy()))
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
            raise Error(
                "rename: ",
                len(names),
                " names but ",
                len(new_names),
                " new names",
            )
        var input_schema = self.schema()
        for ref n in names:
            if input_schema.get_field_index(n) == -1:
                raise Error("rename: column '", n, "' not found in schema")
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
            out_values.append(column(f.name.copy()))
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

    def aggregate(
        self,
        var aggs: List[DynValue],
        var keys: List[DynValue] = List[DynValue](),
    ) raises -> DynRelation:
        """`SELECT <keys>, <aggs> ... GROUP BY <keys>`.

        `keys` defaults to empty, which is a whole-table aggregate rather than
        a special node — `sum(x)` with no `GROUP BY` is one implicit group.
        Aggregates come first because they are the part a caller always
        supplies.
        """
        return Aggregate(self.copy(), keys^, aggs^)

    def join(
        self,
        var right: DynRelation,
        var left_keys: List[Int],
        var right_keys: List[Int],
        kind: JoinKind = JOIN_INNER,
    ) raises -> DynRelation:
        """Equijoin. `self` is the build side and `right` streams."""
        return Join(self.copy(), right^, left_keys^, right_keys^, kind)

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

    def schema(self) -> Schema:
        return self.batch.schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        return Pipeline(BatchSourceOperator(self.batch.to_struct_array()))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Empty(", len(self.batch.schema), " cols)")


struct InMemoryTable(Relation, Writable):
    """A batch already in memory, as a source."""

    var batch: RecordBatch

    def __init__(out self, var batch: RecordBatch):
        self.batch = batch^

    def schema(self) -> Schema:
        return self.batch.schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
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

    `EliminateFilter` reads this. Like `pruner`, it is `Optional` because the
    erased overload cannot answer — and like `pruner`, a `None` costs only an
    optimization."""

    var pruner: Optional[PrunePredicate]
    """The predicate again, typed, for statistics pruning — `None` when it
    arrived already boxed.

    Never consulted for correctness: the `DynValue` above is what actually
    filters, and this only ever makes the scan smaller. That is what lets it be
    `Optional` without a soundness caveat — a missing pruner reads every row
    group, which is the answer the engine gave before pruning existed."""

    def __init__(
        out self,
        var input: DynRelation,
        var predicate: DynValue,
        var pruner: Optional[PrunePredicate] = None,
        constant: Optional[Bool] = None,
        var conjuncts: List[DynValue] = [],
    ) raises:
        reject_aggregate(
            predicate,
            "filter",
            predicate.name(),
            "put it in .aggregate() and filter the result (HAVING)",
        )
        self.input = ArcPointer(input^)
        self.predicate = predicate^
        self.pruner = pruner^
        self.constant = constant
        self.conjuncts = conjuncts^

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        return self.with_input(f(self.input[]))

    def with_input(self, var input: DynRelation) raises -> Filter:
        """This filter over a different input, carrying everything else.

        **The only way a rule should move a filter.** A `Filter` holds five
        things — input, predicate, pruner, constant, conjuncts — and the last
        three are analysis decided at the verb, where the predicate's concrete
        type was still visible. A rule that rebuilds with
        `Filter(new_input, predicate, pruner)` silently drops the other two,
        which does not fail: the filter still filters, `EliminateFilter` and
        `SplitConjunction` just stop firing. That is exactly what happened when
        `constant` and `conjuncts` were added and six call sites were not
        updated.
        """
        return Filter(
            input^,
            self.predicate.copy(),
            self.pruner.copy(),
            self.constant,
            self.conjuncts.copy(),
        )

    def schema(self) -> Schema:
        return self.input[].schema()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        var pipe = self.input[].to_operator(
            ctx,
            bindings,
            (pushed.conjoined(self.pruner.value()) if self.pruner else pushed^),
        )
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
            raise Error(
                "project: ", len(names), " names but ", len(values), " values"
            )
        for i in range(len(values)):
            reject_aggregate(
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

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        var pipe = self.input[].to_operator(ctx, bindings, Pushdown())
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
    here. `to_state(grouped)` compiles two loops out of one struct and running
    the grouped one over a single group measured 14.6x worse — a runtime
    branch could not have made that choice.
    """

    var input: ArcPointer[DynRelation]
    var keys: List[DynValue]
    var aggs: List[DynValue]
    var _schema: Schema

    def __init__(
        out self,
        var input: DynRelation,
        var keys: List[DynValue],
        var aggs: List[DynValue],
    ) raises:
        for ref k in keys:
            reject_aggregate(
                k,
                "aggregate",
                k.name(),
                "group by a column or a per-row expression, not an aggregate",
            )
        self._schema = Self._output_schema(input.schema(), keys, aggs)
        self.input = ArcPointer(input^)
        self.keys = keys^
        self.aggs = aggs^

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
        return Aggregate(f(self.input[]), self.keys.copy(), self.aggs.copy())

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        var grouped = len(self.keys) > 0
        var folds = List[DynOperator](capacity=len(self.aggs))
        for ref a in self.aggs:
            folds.append(
                a.to_operator(self.input[].schema(), grouped, bindings)
            )
        var pipe = self.input[].to_operator(ctx, bindings, Pushdown())
        var keys = List[DynOperator](capacity=len(self.keys))
        for ref k in self.keys:
            keys.append(k.to_operator(self.input[].schema(), False, bindings))
        pipe.append(
            GroupByOperator(keys^, folds^, self._schema.copy(), ctx.copy())
        )
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

    def schema(self) -> Schema:
        return self.input[].schema()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        var pipe = self.input[].to_operator(ctx, bindings, Pushdown())
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
            raise Error(
                "sort: ",
                len(keys),
                " keys but ",
                len(ascending),
                " directions",
            )
        if len(keys) == 0:
            raise Error("sort: needs at least one key")
        for ref k in keys:
            reject_aggregate(
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

    def schema(self) -> Schema:
        return self.input[].schema()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        var pipe = self.input[].to_operator(ctx, bindings, pushed^)
        pipe.append(
            SortOperator(
                _to_operators(self.keys, self.input[].schema(), bindings),
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

    A tenth node rather than a shape `Project` could carry, because a window
    function is not a per-row value: `Project` evaluates each of its values
    against the batch in front of it, and a window function's answer depends
    on rows that may never share a batch with it. `ProjectOperator` has
    nowhere to put the buffering that needs, and `reject_aggregate` already
    keeps the one other non-per-row thing out of that position.

    **This node only appends.** The `with_columns` rule that a repeated name
    replaces in place is a *verb's* rule; expressing it here would mean the
    node deciding column order too. So the verb raises on a collision and this
    node's schema is simply the input's fields followed by one per expression.

    **Every expression on one node shares one window spec.** The verb groups
    by `WindowExpr.spec()` and stacks a node per distinct spec, so this node
    always sorts exactly once, and two window functions over different
    orderings cost two sorts rather than one wrong answer.
    """

    var input: ArcPointer[DynRelation]
    var names: List[String]
    var exprs: List[WindowExpr]
    var _schema: Schema

    def __init__(
        out self,
        var input: DynRelation,
        var names: List[String],
        var exprs: List[WindowExpr],
    ) raises:
        if len(names) != len(exprs):
            raise Error(
                "window: ",
                len(names),
                " names but ",
                len(exprs),
                " expressions",
            )
        if len(exprs) == 0:
            raise Error("window: needs at least one expression")
        for ref e in exprs:
            if e.spec() != exprs[0].spec():
                raise Error(
                    "window: '",
                    e,
                    "' and '",
                    exprs[0],
                    "' do not share a window; build one node per window",
                )
        self._schema = Self._output_schema(input.schema(), names, exprs)
        self.input = ArcPointer(input^)
        self.names = names^
        self.exprs = exprs^

    @staticmethod
    def _output_schema(
        input: Schema, names: List[String], exprs: List[WindowExpr]
    ) raises -> Schema:
        """The input's fields, then one per expression.

        Every appended field is `nullable`, and that is a property of window
        functions rather than of the argument: `LAG` is null at a partition's
        first row and `LEAD` at its last however non-nullable the column it
        reads. Only the three ranking functions are total, and giving them a
        narrower field would make the schema depend on which function was
        named for no gain a reader could use.
        """
        var fields = List[Field](capacity=len(input.fields) + len(exprs))
        for ref f in input.fields:
            fields.append(f.copy())
        for i in range(len(exprs)):
            fields.append(field(names[i].copy(), exprs[i].dtype(input)))
        return schema(fields^)

    def traverse[
        F: def(DynRelation) raises -> DynRelation
    ](self, f: F) raises -> DynRelation:
        return Window(f(self.input[]), self.names.copy(), self.exprs.copy())

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        # **The pushdown stops here**, exactly as it does at `Aggregate` and
        # `Limit`. A window function reads its whole partition, so a predicate
        # that reached the scan would prune row groups *before* this operator
        # counts them and every rank, row number and running total would be
        # computed over the wrong population -- a wrong answer, not an error.
        #
        # `Sort` forwards `pushed`, which is safe *for a sort with no bound*:
        # reordering rows never removes any. A `Sort` carrying `TopN`'s limit
        # does drop rows, and the tree guards that case by case rather than in
        # `Sort.to_operator` -- `PushFilterBelowSort` and `RemoveRedundantSort`
        # both bail on `sort.limit`, and `TopN` keeps the `Limit` above, which
        # clears. So the general law is narrower than "sorts may forward"; what
        # matters here is that this node decides *which rows exist*, which puts
        # it with `Aggregate` and `Limit`.
        var pipe = self.input[].to_operator(ctx, bindings, Pushdown())
        pipe.append(
            WindowOperator(
                self.names.copy(),
                self.exprs.copy(),
                self.input[].schema(),
                self._schema.copy(),
                bindings.copy(),
                ctx.copy(),
            )
        )
        return pipe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Window(", self.input[])
        for i in range(len(self.exprs)):
            writer.write(", ", self.exprs[i], " as ", self.names[i])
        writer.write(")")


def _to_operators(
    values: List[DynValue], schema: Schema, bindings: Bindings
) raises -> List[DynOperator]:
    """Turn a list of logical values into the operators that run them.

    Built **once**, when the plan becomes physical — not per batch. That is the
    whole point of the split: anything a value needs to resolve or cache before
    the first row arrives has a place to live now.
    """
    var out = List[DynOperator](capacity=len(values))
    for ref v in values:
        out.append(v.to_operator(schema, False, bindings))
    return out^


struct Join(Relation, Writable):
    """An equijoin over two sub-plans.

    The **only** node with two inputs, and the reason `Pipeline` had to be an
    `Operator`: the build side is a whole plan, and it is handed to the
    operator as an ordinary boxed stage. Before that, a chain of stages was a
    different kind of thing from a stage, and there was nowhere to put a second
    one.

    `left` is the build side and `right` streams. That is the usual convention
    and it is not arbitrary — the build side is materialised and indexed, so it
    should be the smaller one. Choosing it automatically is an optimiser's job
    and this layer does not have one.
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
    var _schema: Schema

    def __init__(
        out self,
        var left: DynRelation,
        var right: DynRelation,
        var left_keys: List[Int],
        var right_keys: List[Int],
        kind: JoinKind = JOIN_INNER,
        strictness: UInt8 = 0,
    ) raises:
        if len(left_keys) != len(right_keys):
            raise Error(
                "join: ",
                len(left_keys),
                " left keys but ",
                len(right_keys),
                " right keys",
            )
        if len(left_keys) == 0:
            raise Error("join: needs at least one key pair")
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

    def __init__(
        out self,
        var left: DynRelation,
        var right: DynRelation,
        *,
        var left_names: List[String],
        var right_names: List[String],
        kind: JoinKind = JOIN_INNER,
        strictness: UInt8 = 0,
    ) raises:
        """By name, for a rewrite putting a join back together.

        The index form is the public verb; this is what `traverse` and
        `optimizer.mojo` use, because a rewrite already holds names and
        converting back to indices only to have them re-resolved would be a
        round trip through the representation this node exists to avoid.
        """
        self._schema = Self._output_schema(left.schema(), right.schema(), kind)
        self.left = ArcPointer(left^)
        self.right = ArcPointer(right^)
        self.left_keys = left_names^
        self.right_keys = right_names^
        self.kind = kind
        self.strictness = strictness

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
                raise Error(
                    "join: ",
                    side,
                    " key index ",
                    idx,
                    " out of range for ",
                    len(schema.fields),
                    " columns",
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
                raise Error(
                    "join: ", side, " key '", n, "' is not in the input"
                )
            out.append(at)
        return out^

    @staticmethod
    def _output_schema(
        left: Schema, right: Schema, kind: JoinKind
    ) raises -> Schema:
        """Left fields then right fields — except for the kinds that emit only
        the left side.

        `SEMI` and `ANTI` answer "which left rows had a match", so the right
        side contributes nothing to the output. `JoinKind.emits_right_columns`
        owns that rule; asking it here keeps the schema and the kernel from
        disagreeing about the shape of the same result.
        """
        var fields = List[Field]()
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
        child: a join is the one node with two inputs."""
        return Join(
            f(self.left[]),
            f(self.right[]),
            left_names=self.left_keys.copy(),
            right_names=self.right_keys.copy(),
            kind=self.kind,
            strictness=self.strictness,
        )

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        """The probe side is the pipeline; the build side is a stage's cargo."""
        var probe = self.right[].to_operator(ctx, bindings, Pushdown())
        probe.append(
            JoinOperator(
                self.left[].to_operator(ctx, bindings, Pushdown()),
                Self._indices_for(self.left[].schema(), self.left_keys, "left"),
                Self._indices_for(
                    self.right[].schema(), self.right_keys, "right"
                ),
                self.kind,
                self.strictness,
                self._schema.copy(),
                self.left[].schema(),
                self.right[].schema(),
                ctx.copy(),
            )
        )
        return probe^

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Join(", self.left[], ", ", self.right[], ", ", self.kind, ")"
        )


struct ParquetScan(Relation, Writable):
    """A Parquet file as a source, read one row group at a time.

    **The schema is the projection.** The scan reads only its own columns out
    of the file, so narrowing a scan's schema *is* how a projection gets pushed
    into it — no separate mechanism, and nothing to keep in sync.

    The schema is supplied rather than read from the file, so building the plan
    touches no I/O: a `Relation` is a description, and a description that has
    to open a file to exist cannot be constructed for a file that is not there
    yet. The operator opens it on first `drain`.
    """

    var path: String
    var _schema: Schema

    def __init__(out self, var path: String, var schema: Schema):
        self.path = path^
        self._schema = schema^

    def schema(self) -> Schema:
        return self._schema.copy()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        return Pipeline(
            ParquetScanOperator(
                self.path.copy(),
                self._schema.copy(),
                pushed^,
                bindings.copy(),
            )
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("ParquetScan(", self.path, ")")
