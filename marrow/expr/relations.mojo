"""Relational plans: a descriptive IR that opens into pull-based operators.

Two layers, cleanly separated:

- ``Relation`` nodes are **pure, immutable descriptions** (``kind``/``schema``/
  ``to_processor``): they hold only their parameters and child relations, no execution
  state. ``DynRelation`` erases them behind an ``ArcPointer``, so copying a plan
  is an O(1) share and the plan is a reusable, inspectable, rewritable template.
- ``Processor`` (``schema``/``pull``) is the executing layer, built by
  ``Relation.to_processor(ctx)``; it owns *all* mutable state (scan offset, built hash
  index, grouper, child operators). ``DynProcessor`` erases it and drives the
  pull loop (``collect``). Operators are single-use and move-only.

``execute(plan, ctx)`` opens the plan into a fresh operator tree and drains it,
so the plan itself is never mutated — run it repeatedly or concurrently.

Concrete nodes / operators: ``InMemoryTable``/``InMemoryTableProcessor``,
``Filter``/``FilterProcessor``, ``Project``/``ProjectProcessor``, ``Aggregate``/``AggregateProcessor``,
``Join``/``JoinProcessor``, ``ParquetScan``/``ParquetScanProcessor``.

Plan-building API
-----------------
Build plans through these, not by constructing nodes: every verb *derives* its
output schema (probing expression dtypes where needed), whereas the node
constructors take one, so a hand-built plan can declare a schema its own
expressions do not produce.

``in_memory_table(batch[, morsel_size])`` / ``parquet_scan(path, schema)`` — leaf sources.
``DynRelation.select(*names)``                   — project columns by name.
``DynRelation.project(names, values)``           — project computed expressions.
``DynRelation.with_columns(names, values)``      — add/replace computed columns,
keeping the rest (polars ``with_columns`` / ibis ``mutate`` semantics).
``DynRelation.drop(names)`` / ``.rename(names, new_names)`` — schema-level
rewrites; both lower to ``Project``, as ``with_columns`` does.
``DynRelation.filter(pred)``                     — filter rows by predicate.
``DynRelation.aggregate(keys, aggs)``            — grouped aggregation
(``HAVING`` is a ``.filter(...)`` on top of it).
``DynRelation.sort(keys, ascending)`` / ``.limit(n[, offset])`` — order and slice
(``.sort(...).limit(k)`` folds into the sort kernel's top-K path).
``DynRelation.join(right, left_on, right_on)``   — hash join.
``DynRelation.execute()``                        — drain to a single RecordBatch.

``project`` and ``aggregate`` each have two overloads, one per expression lane:
one taking runtime ``DynValue``s (names resolved against the input schema
here) and one taking already-bound ``DynValue``s, so a fused comptime plan uses
the same API rather than assembling nodes by hand.

Example
-------
    var plan = in_memory_table(batch).filter(col("x") > lit[Int64Type](0)).select("x")
    var result = plan.execute()
"""

from ..kernels.interval import Interval
from .pruning import PruneStats
from ..arrays import DynArray
from std.builtin.rebind import rebind
from .dynamic import DynValue
from .values import AggExpr, BoxedValue
from std.memory import ArcPointer

from ..dtypes import Field
from ..schema import Schema
from ..tabular import RecordBatch
from .builders import col
from ..execution import ExecContext
from .aggregates import AggFunc
from ..parquet import LeafSet
from .execution import (
    DEFAULT_MORSEL_SIZE,
    DynProcessor,
    InMemoryTableProcessor,
    ParquetScanProcessor,
    FilterProcessor,
    ProjectProcessor,
    AggregateProcessor,
    JoinProcessor,
    SortProcessor,
    LimitProcessor,
)
from ..kernels.join import (
    # JOIN_LEFT/RIGHT/FULL/ANY are unused *here* but re-exported through
    # `marrow.expr`'s `__init__`, which imports them from this module.
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_ALL,
    JOIN_ANY,
    JoinKind,
)


# ---------------------------------------------------------------------------
# Relation trait — the descriptive IR node (pure data; no execution state)
# ---------------------------------------------------------------------------


# Relation kind discriminants — let a plan-building step recognise a node behind
# `DynRelation` (e.g. to push a filter into a `ParquetScan`) without a full RTTI.
comptime RELATION_GENERIC: Int = 0
comptime RELATION_PARQUET_SCAN: Int = 1
comptime RELATION_SORT: Int = 2


trait Relation(Deinitable, Movable):
    """A relational plan node: a pure, immutable description of an operation.

    Nodes hold only their parameters and child relations — no execution state.
    ``to_processor(ctx)`` builds the stateful ``Processor`` that runs (opening children
    recursively), so a plan is a reusable template you can inspect, copy cheaply
    (O(1) — nodes are immutable and shared), and rewrite."""

    def with_predicate(
        self, var predicate: BoxedValue
    ) raises -> Optional[ArcPointer[NoneType]]:
        """This node rebuilt to carry `predicate` as **pruning metadata**, as an
        erased pointer — or `None` when it cannot use one, which is every node
        but a scan.

        Returns the erased data rather than an `DynRelation` on purpose. The
        rebuilt node has the *same concrete type* as this one, so the caller can
        keep its own trampolines and swap only the pointer; returning
        `Optional[DynRelation]` instead puts `DynRelation` inside its own
        trampoline field type and Mojo rejects the struct as recursive.

        This replaces a `downcast` to a concrete `ParquetScan`, which stopped
        being correct once `ParquetScan` gained a comptime `LeafSet`: the
        downcast had to name one parameterisation, so a scan compiled for a
        narrow set was silently rebuilt as the default full-ladder one. A
        downcast cannot see a comptime parameter; a virtual can carry it.
        """
        return None

    def with_projection(
        self, needed: List[String]
    ) raises -> Optional[ArcPointer[NoneType]]:
        """This node rebuilt so its subtree reads only the columns it needs,
        given that its parent reads only `needed` of *this* node's output — or
        `None` to leave the subtree alone, which is the default.

        Erased for the same reason `with_predicate` is: the rebuilt node has the
        *same concrete type* as this one, so `DynRelation` keeps its trampolines
        and swaps only the pointer. Returning `DynRelation` would put it inside
        its own trampoline field type and Mojo rejects the struct as recursive.

        A node implements this by widening `needed` with the columns its **own**
        expressions read — a `Filter`'s predicate, a `Sort`'s keys, an
        `Aggregate`'s keys *and* aggregate inputs — recursing into its input
        through `DynRelation.with_projection`, and rebuilding itself around the
        result. `ParquetScan` is where the recursion terminates and the only
        node whose *own* data changes: its schema is its projection.

        **The output schema of the plan root never changes**, because
        `DynRelation.optimize` seeds `needed` with the root's own columns and a
        schema-passthrough node (`Filter`/`Sort`/`Limit`) can only ever narrow to
        a subset its parent asked for.
        """
        return None

    def children(self) -> List[DynRelation]:
        """This node's child relations, left to right; empty for a leaf.

        The read-only half of plan traversal — what `explain`, a cost model, or
        a test asserting a rewritten scan's schema needs, and what the layer
        lacked. Unlike the rewriting virtuals above it can return `DynRelation`
        directly: a *method* mentioning the erased type is fine, only a **field**
        whose function type mentions it makes the struct recursive."""
        return List[DynRelation]()

    def schema(self) -> Schema:
        ...

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        """Build the physical operator for this node (opening its children)."""
        ...

    def write_to[W: Writer](self, mut writer: W):
        ...

    def kind(self) -> Int:
        """Node discriminant for plan rewrites; generic unless overridden."""
        return RELATION_GENERIC


# ---------------------------------------------------------------------------
# Column-set helpers used by the projection rewrite
# ---------------------------------------------------------------------------


def _add_names(mut acc: List[String], names: List[String]):
    """Append the names not already in `acc`, order-preserving."""
    for ref n in names:
        var seen = False
        for ref a in acc:
            if a == n:
                seen = True
                break
        if not seen:
            acc.append(n.copy())


def _has_name(names: List[String], name: String) -> Bool:
    for ref n in names:
        if n == name:
            return True
    return False


def _schema_names(schema: Schema) -> List[String]:
    var out = List[String](capacity=len(schema.fields))
    for ref f in schema.fields:
        out.append(f.name.copy())
    return out^


def _referenced_by(values: List[BoxedValue]) -> List[String]:
    """The order-preserving deduped union of a list of expressions' columns."""
    var acc = List[String]()
    for ref v in values:
        _add_names(acc, v.referenced_columns())
    return acc^


# ---------------------------------------------------------------------------
# DynRelation — type-erased IR container + plan-building API
# ---------------------------------------------------------------------------


struct DynRelation(ImplicitlyCopyable, Movable, Writable):
    """Type-erased plan node behind an ``ArcPointer``.

    Nodes are immutable descriptions, so copying an ``DynRelation`` is an O(1)
    ``ArcPointer`` share — no deep clone, no reset. ``to_processor(ctx)`` builds the
    operator tree that executes; the plan is never mutated, so it is a reusable
    template and copies never share execution state. Carries the plan-building
    API (``select``/``filter``/``aggregate``/``join``)."""

    var _data: ArcPointer[NoneType]
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_to_processor: def(
        ArcPointer[NoneType], ExecContext
    ) thin raises -> DynProcessor
    var _virt_write_to_string: def(ArcPointer[NoneType]) thin -> String
    var _virt_drop: def(var ArcPointer[NoneType]) thin
    var _virt_kind: def(ArcPointer[NoneType]) thin -> Int
    var _virt_with_predicate: def(
        ArcPointer[NoneType], var BoxedValue
    ) thin raises -> Optional[ArcPointer[NoneType]]
    var _virt_with_projection: def(
        ArcPointer[NoneType], List[String]
    ) thin raises -> Optional[ArcPointer[NoneType]]
    var _virt_children: def(ArcPointer[NoneType]) thin -> List[DynRelation]

    @staticmethod
    def _tramp_kind[T: Relation](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[T]](ptr)[].kind()

    @staticmethod
    def _tramp_with_projection[
        T: Relation
    ](ptr: ArcPointer[NoneType], needed: List[String]) raises -> Optional[
        ArcPointer[NoneType]
    ]:
        return rebind[ArcPointer[T]](ptr)[].with_projection(needed)

    @staticmethod
    def _tramp_children[
        T: Relation
    ](ptr: ArcPointer[NoneType]) -> List[DynRelation]:
        return rebind[ArcPointer[T]](ptr)[].children()

    @staticmethod
    def _tramp_with_predicate[
        T: Relation
    ](ptr: ArcPointer[NoneType], var predicate: BoxedValue) raises -> Optional[
        ArcPointer[NoneType]
    ]:
        return rebind[ArcPointer[T]](ptr)[].with_predicate(predicate^)

    @staticmethod
    def _tramp_schema[T: Relation](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    def _tramp_to_processor[
        T: Relation
    ](ptr: ArcPointer[NoneType], ctx: ExecContext) raises -> DynProcessor:
        return rebind[ArcPointer[T]](ptr)[].to_processor(ctx)

    @staticmethod
    def _tramp_write_to_string[
        T: Relation
    ](ptr: ArcPointer[NoneType]) -> String:
        var s = String()
        rebind[ArcPointer[T]](ptr)[].write_to(s)
        return s^

    @staticmethod
    def _tramp_drop[T: Relation](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    @implicit
    def __init__[T: Relation](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_schema = Self._tramp_schema[T]
        self._virt_to_processor = Self._tramp_to_processor[T]
        self._virt_write_to_string = Self._tramp_write_to_string[T]
        self._virt_drop = Self._tramp_drop[T]
        self._virt_kind = Self._tramp_kind[T]
        self._virt_with_predicate = Self._tramp_with_predicate[T]
        self._virt_with_projection = Self._tramp_with_projection[T]
        self._virt_children = Self._tramp_children[T]

    def __init__(out self, *, copy: Self):
        # O(1) share — nodes are immutable, so aliasing is safe.
        self._data = copy._data
        self._virt_schema = copy._virt_schema
        self._virt_to_processor = copy._virt_to_processor
        self._virt_write_to_string = copy._virt_write_to_string
        self._virt_drop = copy._virt_drop
        self._virt_kind = copy._virt_kind
        self._virt_with_predicate = copy._virt_with_predicate
        self._virt_with_projection = copy._virt_with_projection
        self._virt_children = copy._virt_children

    def __deinit__(deinit self):
        self._virt_drop(self._data^)

    # --- introspection ---

    def schema(self) -> Schema:
        return self._virt_schema(self._data)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._virt_write_to_string(self._data))

    def kind(self) -> Int:
        return self._virt_kind(self._data)

    def downcast[T: Relation](self) -> ArcPointer[T]:
        return rebind[ArcPointer[T]](self._data.copy())

    def children(self) -> List[DynRelation]:
        """This node's children, left to right — the plan is walkable from here.
        """
        return self._virt_children(self._data)

    # --- optimization ---

    def with_projection(self, needed: List[String]) raises -> DynRelation:
        """This plan rewritten so nothing below reads a column no one needs,
        given that the caller reads only `needed` of this node's output.

        Prefer `optimize()`, which seeds `needed` correctly for a root."""
        var data = self._virt_with_projection(self._data, needed)
        if data:
            # Same concrete node type, so the trampolines still apply — copy
            # them and swap in the rebuilt node's data (`with_predicate`'s rule).
            var out = DynRelation(copy=self)
            out._data = data.take()
            return out^
        return DynRelation(copy=self)

    def optimize(self) raises -> DynRelation:
        """This plan with projection pushdown applied, ready to execute.

        Seeds the rewrite with the root's own columns, which is what makes the
        result's schema identical to this one's: a node can only narrow to a
        subset its parent asked for, and the root asks for everything it emits.

        `execute()` calls this, so a plan built through the verbs is optimized
        without the caller doing anything. It is public because a caller that
        drives `to_processor` itself — or a test asserting *what* was pushed —
        needs the rewritten plan in hand."""
        return self.with_projection(_schema_names(self.schema()))

    # --- execution ---

    def to_processor(
        self, ctx: ExecContext = ExecContext.auto()
    ) raises -> DynProcessor:
        """Build the operator tree for this plan; the plan is left untouched."""
        return self._virt_to_processor(self._data, ctx)

    def execute(
        self, ctx: ExecContext = ExecContext.auto()
    ) raises -> RecordBatch:
        """Open this plan into a fresh operator tree and drain it into one
        `RecordBatch`.

        The default is **auto**, not serial: each kernel the plan reaches picks
        serial vs all-cores from its own row-count threshold. It used to be the
        bare `ExecContext()` — `num_threads=1`, forced serial — which made every
        plan-driven query single-threaded no matter what the caller had, and
        made `GroupBy`'s and `HashJoin`'s parallel strategies unreachable from
        the relational API. Pass `ExecContext.serial()` to get the old
        behaviour.

        The plan is optimized first (projection pushdown), which never changes
        the schema or the rows — only how many Parquet column chunks are
        decoded. The plan is a pure description and is never mutated, so it can
        be executed repeatedly and concurrently."""
        var op = self.optimize().to_processor(ctx)
        return op.collect()

    # --- plan-building API ---

    def select(self, *names: String) raises -> DynRelation:
        """Project columns by name, returning a new plan node.

        The variadic spelling, for a call site that writes its columns out:
        ``rel.select("x", "y")``. A caller holding the names in a list — every
        frontend that builds a projection at runtime, the Python bindings
        included — wants the `List[String]` overload below, because a Mojo
        variadic cannot be splatted."""
        var col_names = List[String]()
        for i in range(len(names)):
            col_names.append(names[i])
        return self.select(col_names)

    def select(self, names: List[String]) raises -> DynRelation:
        """Project columns by name — the list spelling of the variadic above.

        Both are pass-through projections and each surviving ``Field`` is
        carried over **whole**: dtype, ``nullable`` and metadata. That is the
        difference from ``project``, which probes each expression's dtype and
        builds a fresh ``Field`` around it — correct for a computed column,
        lossy for a pass-through, since a non-nullable input would come out
        nullable.

        Raises:
            Error: if a name is not in the input schema.
        """
        var schema = self.schema()
        var col_names = List[String]()
        var exprs = List[BoxedValue]()
        var fields = List[Field]()
        for ref name in names:
            var idx = schema.get_field_index(name)
            if idx == -1:
                raise Error("select: column '" + name + "' not found")
            col_names.append(name.copy())
            exprs.append(col(schema.fields[idx].name.copy()))
            fields.append(schema.fields[idx].copy())
        return DynRelation(
            Project(
                input=self,
                names=col_names^,
                values=exprs^,
                schema=Schema(fields=fields^),
            )
        )

    def filter(self, var predicate: BoxedValue) raises -> DynRelation:
        """Filter rows by a boolean predicate. Column references resolve by name
        against the batch schema when the boxed value executes.

        When the input is a `ParquetScan`, the predicate is also pushed into the
        scan as pruning metadata (row-group / page skipping); the `Filter` is
        kept so the rows returned are exactly those that satisfy the predicate.
        """
        var data = self._virt_with_predicate(self._data, predicate.copy())
        if data:
            # Same concrete node type, so the trampolines still apply — copy
            # them and swap in the rebuilt node's data.
            var pushed = DynRelation(copy=self)
            pushed._data = data.take()
            return DynRelation(Filter(input=pushed^, predicate=predicate^))
        return DynRelation(Filter(input=self, predicate=predicate^))

    def aggregate(
        self, keys: List[BoxedValue], aggs: List[AggExpr]
    ) raises -> DynRelation:
        """Grouped aggregation — ``GROUP BY keys`` with one output column per
        aggregate.

            rel.aggregate(
                keys=[col("region")],
                aggs=[col("amount").sum().alias("total"),
                      col("amount").max().alias("biggest")],
            )

        An aggregate is written on the expression it aggregates
        (``col("amount").sum()``), so nothing has to be kept in positional
        correspondence. Keys and aggregate inputs are arbitrary expressions — a
        bare column or a computed one (``col("x") + lit(1)``,
        ``col("ts").date_trunc("month")``, ``case_when(...)``).

        Both expression lanes are accepted, and they mix: ``col("x").sum()`` on
        a ``DynAgg`` carries the function's *name* until this call resolves it
        against the input's dtype, while the same call on a fused node
        (``col("x", int64).sum()``) already names its ``Aggregation`` and
        resolves nothing.

        Output schema: one field per key — named after its source column, or
        ``key<i>`` for a computed key — then one per aggregate, named by
        ``.alias(...)`` or after the function. Leaving ``keys`` empty is
        ``SELECT agg(x), ...`` with no GROUP BY: one implicit group, one row out.

        ``HAVING`` needs no dedicated node: ``rel.aggregate(...).filter(pred)``
        evaluates ``pred`` against the aggregate's output batch, so
        ``...filter(col("n") > lit(100))`` is exactly ``HAVING n > 100``.
        """
        var input_schema = self.schema()
        # Bind key/input expressions to positional form once (names -> indices),
        # so per-morsel eval uses positions directly, and probe each one's output
        # dtype by evaluating it against a 0-row batch of the input schema (the
        # same trick ``project`` uses) — general for computed expressions, where
        # a purely static dtype rule would be incomplete.
        var probe = RecordBatch.empty(input_schema)

        var fields = List[Field]()
        var key_exprs = List[BoxedValue]()
        for i in range(len(keys)):
            var k = keys[i].resolve_names(input_schema)
            # A bare column keeps its own name; anything computed gets a
            # positional one. `bound_column` is the node answering for itself —
            # this used to reach into the interpreter for `kind() == LOAD` and
            # then its payload.
            var pos = k.bound_column(input_schema)
            var name = input_schema.fields[
                pos
            ].name if pos >= 0 else "key" + String(i)
            fields.append(Field(name, k.execute(probe).dtype()))
            key_exprs.append(k^)

        # Resolve each aggregate against the dtype its input turns out to have —
        # the only interpretation step on this path, and the last one: the output
        # dtype is then the aggregation's own (`sum` widens integers to int64;
        # `min`/`max` preserve the input dtype, including a timestamp's unit and
        # timezone; `count` and the distinct counts are int64; `mean` is
        # float64), and an aggregate not defined for the column's type is
        # rejected here rather than at execution.
        var input_exprs = List[BoxedValue]()
        var resolved = List[AggFunc]()
        for i in range(len(aggs)):
            var v = aggs[i].input_for(input_schema)
            resolved.append(aggs[i].resolve(v.execute(probe).dtype()))
            fields.append(Field(aggs[i].out_name, resolved[i].out_dtype.copy()))
            input_exprs.append(v^)

        return DynRelation(
            Aggregate(
                input=self,
                keys=key_exprs^,
                inputs=input_exprs^,
                aggs=resolved^,
                schema=Schema(fields=fields^),
            )
        )

    def aggregate(
        self,
        var keys: List[BoxedValue],
        var inputs: List[BoxedValue],
        var aggs: List[AggFunc],
        names: List[String],
    ) raises -> DynRelation:
        """Grouped aggregation from an already-resolved aggregate spec — the
        fused counterpart of the ``keys``/``aggs`` overload above.

        ``aggs[j]`` carries a *comptime* ``Aggregation`` (built with
        ``AggFunc.of[A]``) and applies to ``inputs[j]``, so there is no function
        name to interpret and nothing is resolved here. ``names`` covers the
        output columns in schema order — one per key, then one per aggregate.

        The output schema is derived, never supplied: key dtypes are probed the
        same way ``project`` probes, and each aggregate's dtype is read off the
        aggregation itself. A caller-written schema would assert the caller's
        own type algebra rather than the plan's."""
        if len(names) != len(keys) + len(aggs):
            raise Error(
                "aggregate: len(names) != len(keys) + len(aggs), got "
                + String(len(names))
                + " for "
                + String(len(keys))
                + " keys and "
                + String(len(aggs))
                + " aggregates"
            )
        if len(inputs) != len(aggs):
            raise Error("aggregate: len(inputs) != len(aggs)")

        var probe = RecordBatch.empty(self.schema())
        var fields = List[Field]()
        for i in range(len(keys)):
            fields.append(Field(names[i], keys[i].execute(probe).dtype()))
        for j in range(len(aggs)):
            fields.append(Field(names[len(keys) + j], aggs[j].out_dtype.copy()))
        return DynRelation(
            Aggregate(
                input=self,
                keys=keys^,
                inputs=inputs^,
                aggs=aggs^,
                schema=Schema(fields=fields^),
            )
        )

    def join(
        self,
        right: DynRelation,
        left_on: List[BoxedValue],
        right_on: List[BoxedValue],
        how: JoinKind = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
    ) raises -> DynRelation:
        """Hash join on equijoin key expressions."""
        if len(left_on) != len(right_on):
            raise Error("join: len(left_on) != len(right_on)")

        var left_schema = self.schema()
        var right_schema = right.schema()

        # Resolve key expressions to positional column indices (each key must be
        # a bare column reference — column_index raises otherwise).
        var left_indices = List[Int]()
        for ref k in left_on:
            var i = k.bound_column(left_schema)
            if i < 0:
                raise Error(
                    "join: key must be a column reference, got a computed"
                    " expression"
                )
            left_indices.append(i)
        var right_indices = List[Int]()
        for ref k in right_on:
            var i = k.bound_column(right_schema)
            if i < 0:
                raise Error(
                    "join: key must be a column reference, got a computed"
                    " expression"
                )
            right_indices.append(i)

        # Output schema: left columns + (suffixed) right columns.
        var fields = List[Field]()
        for ref f in left_schema.fields:
            fields.append(f.copy())
        # One source for the output width — see `JoinKind`.
        if how.emits_right_columns():
            var left_names = List[String]()
            for ref f in left_schema.fields:
                left_names.append(f.name)
            for ref f in right_schema.fields:
                var name = f.name
                var collides = False
                for ref ln in left_names:
                    if ln == name:
                        collides = True
                        break
                if collides:
                    name = name + "_right"
                fields.append(Field(name, f.dtype.copy()))

        return DynRelation(
            Join(
                left=self,
                right=right,
                left_key_indices=left_indices^,
                right_key_indices=right_indices^,
                join_kind=how,
                strictness=strictness,
                schema=Schema(fields=fields^),
            )
        )

    def project(
        self, names: List[String], var values: List[BoxedValue]
    ) raises -> DynRelation:
        """Project arbitrary named expressions (computed columns).

        Unlike ``select`` (column pass-through), each value is an expression
        evaluated per morsel — e.g. ``col("x") + lit(1)``, a literal, or a
        renamed column. The output dtype of each column is the expression's own,
        probed rather than declared (see below).

        Both expression lanes are accepted, exactly as ``filter`` accepts both,
        and — also as in ``filter`` — column references bind by name when the
        boxed value executes rather than being rewritten to positions here. An
        unknown column still fails at plan-build time, because the dtype probe
        below executes the expression.

        There is one list type now: ``DynValue`` is what a runtime frontend
        builds, and the fused nodes convert into it implicitly."""
        if len(names) != len(values):
            raise Error("project: len(names) != len(values)")

        var input_schema = self.schema()
        # Probe each expression's output dtype by evaluating it against a 0-row
        # batch of the input schema — general for computed columns (``x + 1``,
        # literals, CASE) where a purely static dtype rule would be incomplete.
        # It is also the only thing that keeps the declared schema honest: a
        # caller-supplied one asserts the caller's arithmetic, not the plan's.
        var probe = RecordBatch.empty(input_schema)
        var fields = List[Field]()
        for i in range(len(values)):
            fields.append(Field(names[i], values[i].execute(probe).dtype()))
        return DynRelation(
            Project(
                input=self,
                names=names.copy(),
                values=values^,
                schema=Schema(fields=fields^),
            )
        )

    def with_columns(
        self, names: List[String], values: List[BoxedValue]
    ) raises -> DynRelation:
        """Add computed columns, keeping every column already there.

        This is ``project``'s usable half. ``project`` *replaces* the output
        schema, so adding one derived column to a 105-column table means
        re-listing 105 names; this lists only what changes:

            rel.with_columns(names=["total"], values=[col("qty") * col("price")])

        **Append-or-replace, replacement in place.** A name that is not in the
        input schema is appended after the existing columns, in argument order.
        A name that *is* in the input schema overwrites that column **at its
        original position** rather than moving it to the end. That is polars
        `with_columns` and ibis `mutate`, both verified rather than assumed:
        polars keeps `['a', 'b', 'c']` after `with_columns(pl.col('b') * 2)`,
        and ibis builds `ops.Project(self, {**node.fields, **values})`, a dict
        merge, which by Python's insertion-order rule updates an existing key in
        place. Matching them is the whole point — this verb is the most-used one
        in both, so its surprises should be their surprises.

        **Every expression sees the input, never a partially-updated output.**
        The node lowered to is a single ``Project``, and ``ProjectProcessor``
        evaluates all of its values against the same input morsel. So in
        ``with_columns(["b", "c"], [col("a") + lit(1), col("b") * lit(2)])``,
        ``c`` reads the *original* ``b``, not the one computed one slot earlier.
        Again this is polars' and ibis' rule; chain two calls to get sequential
        semantics.

        Pass-through columns are `col(name)` references, exactly as ``select``
        builds them, and keep their input ``Field`` whole — dtype, ``nullable``
        and metadata. A pass-through *is* that field, so copying it is the
        honest answer and strictly more informative than re-probing, which would
        recover the dtype and silently drop the other two. Replaced and appended
        columns get their dtype probed against a 0-row batch the way ``project``
        does, for the reason given there: a declared dtype asserts the caller's
        arithmetic rather than the plan's.

        Raises:
            Error: if the two lists differ in length, or a name is given twice
                — the second would silently win, which is a typo, not an intent.
        """
        if len(names) != len(values):
            raise Error("with_columns: len(names) != len(values)")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if names[i] == names[j]:
                    raise Error(
                        "with_columns: duplicate output column '"
                        + names[i]
                        + "'"
                    )

        var input_schema = self.schema()
        var probe = RecordBatch.empty(input_schema)
        var out_names = List[String]()
        var out_values = List[BoxedValue]()
        var fields = List[Field]()

        # The input schema, in order: kept as a by-name reference, or
        # overwritten in place by its replacement expression.
        for ref f in input_schema.fields:
            var repl = -1
            for i in range(len(names)):
                if names[i] == f.name:
                    repl = i
                    break
            out_names.append(f.name.copy())
            if repl >= 0:
                fields.append(
                    Field(f.name, values[repl].execute(probe).dtype())
                )
                out_values.append(values[repl].copy())
            else:
                fields.append(f.copy())
                out_values.append(col(f.name.copy()))

        # Then the genuinely new names, in argument order.
        for i in range(len(names)):
            if input_schema.get_field_index(names[i]) == -1:
                out_names.append(names[i].copy())
                fields.append(Field(names[i], values[i].execute(probe).dtype()))
                out_values.append(values[i].copy())

        return DynRelation(
            Project(
                input=self,
                names=out_names^,
                values=out_values^,
                schema=Schema(fields=fields^),
            )
        )

    def drop(self, names: List[String]) raises -> DynRelation:
        """Remove the named columns, keeping the rest in their original order.

        The complement of ``select``: say what goes, not what stays. Lowers to
        the same ``Project`` of by-name column references that ``select`` builds
        — this is a schema-level rewrite, so no expression is evaluated and each
        surviving ``Field`` is carried over whole (dtype, ``nullable``,
        metadata).

        Repeating a name is harmless (the column is gone either way), so it is
        not an error; naming a column that does not exist is, because it is
        always a typo or a stale plan and silently dropping nothing would hide
        it. Dropping every column is allowed and yields a 0-column relation, as
        it does in polars.

        Raises:
            Error: if a name is not in the input schema.
        """
        var input_schema = self.schema()
        for ref n in names:
            if input_schema.get_field_index(n) == -1:
                raise Error("drop: column '" + n + "' not found")

        var out_names = List[String]()
        var values = List[BoxedValue]()
        var fields = List[Field]()
        for ref f in input_schema.fields:
            var dropped = False
            for ref n in names:
                if n == f.name:
                    dropped = True
                    break
            if not dropped:
                out_names.append(f.name.copy())
                values.append(col(f.name.copy()))
                fields.append(f.copy())

        return DynRelation(
            Project(
                input=self,
                names=out_names^,
                values=values^,
                schema=Schema(fields=fields^),
            )
        )

    def rename(
        self, names: List[String], new_names: List[String]
    ) raises -> DynRelation:
        """Rename columns, leaving order, dtypes and every other column alone.

            rel.rename(names=["ts"], new_names=["event_time"])

        **Signature.** Two parallel lists, old then new. Three shapes were
        available and this one fits both the file and the problem:

        - ``rename(names, new_names)`` — the shape every other verb here already
          has (``project(names, values)``, ``sort(keys, ascending)``,
          ``aggregate(keys, inputs, aggs, names)``). Mentions only the columns
          that change.
        - ``rename(mapping: Dict[String, String])`` — polars' spelling, and it
          makes "renamed twice" unrepresentable, but it is the only dictionary
          argument in the plan-building API and would read as the odd one out.
        - ``rename_columns(names)`` taking a full-width list — PyArrow's, and
          normally the tie-breaker here. Rejected: it re-lists all 105 columns to
          change one, which is precisely the ergonomic failure ``with_columns``
          exists to fix, and it makes a plain reorder or a dropped column look
          like a rename.

        Like ``drop``, this is a schema-level rewrite over a ``Project`` of
        by-name references: no expression is evaluated, and dtype, ``nullable``
        and metadata pass through untouched — only ``Field.name`` changes.

        Raises:
            Error: if the two lists differ in length; if an old name is not in
                the input schema; if the same column is renamed twice; or if the
                result would contain two columns with the same name (including a
                collision with a column that was not renamed) — a duplicate
                output name makes the relation unusable by name afterwards.
        """
        if len(names) != len(new_names):
            raise Error("rename: len(names) != len(new_names)")

        var input_schema = self.schema()
        for i in range(len(names)):
            if input_schema.get_field_index(names[i]) == -1:
                raise Error("rename: column '" + names[i] + "' not found")
            for j in range(i + 1, len(names)):
                if names[i] == names[j]:
                    raise Error(
                        "rename: column '" + names[i] + "' renamed twice"
                    )

        var out_names = List[String]()
        var values = List[BoxedValue]()
        var fields = List[Field]()
        for ref f in input_schema.fields:
            var out = f.name.copy()
            for i in range(len(names)):
                if names[i] == f.name:
                    out = new_names[i].copy()
                    break
            # Comparing against the names emitted so far catches every colliding
            # pair, since one member of any pair is always the later one.
            for ref prev in out_names:
                if prev == out:
                    raise Error("rename: duplicate output column '" + out + "'")
            values.append(col(f.name.copy()))
            fields.append(
                Field(out, f.dtype.copy(), f.nullable, f.metadata.copy())
            )
            out_names.append(out^)

        return DynRelation(
            Project(
                input=self,
                names=out_names^,
                values=values^,
                schema=Schema(fields=fields^),
            )
        )

    def sort(
        self,
        keys: List[BoxedValue],
        ascending: List[Bool],
        nulls_first: Bool = True,
        stable: Bool = True,
    ) raises -> DynRelation:
        """Sort by one or more key expressions (a pipeline breaker).

        Each key has its own ascending/descending direction; ``nulls_first`` and
        ``stable`` apply to all keys. Keys resolve by name against the input
        schema. The output schema is unchanged from the input."""
        if len(keys) == 0:
            raise Error("sort: keys must not be empty")
        if len(keys) != len(ascending):
            raise Error("sort: len(keys) != len(ascending)")

        var input_schema = self.schema()
        var key_exprs = List[BoxedValue]()
        for ref k in keys:
            key_exprs.append(k.resolve_names(input_schema))
        return DynRelation(
            Sort(
                input=self,
                keys=key_exprs^,
                ascending=ascending.copy(),
                nulls_first=nulls_first,
                stable=stable,
                limit=None,
                schema=input_schema,
            )
        )

    def limit(self, length: Int, offset: Int = 0) raises -> DynRelation:
        """Keep at most ``length`` rows after skipping ``offset`` rows.

        Top-K fast path: when this limits a ``Sort`` with no existing limit and
        ``offset == 0``, the limit is folded into the sort (``Sort(limit=…)``),
        driving the sort kernel's top-K path instead of a full sort. Otherwise a
        streaming ``Limit`` operator is layered on top."""
        if offset == 0 and self.kind() == RELATION_SORT:
            ref s = self.downcast[Sort]()[]
            if not s.limit:
                return DynRelation(
                    Sort(
                        input=s.input.copy(),
                        keys=s.keys.copy(),
                        ascending=s.ascending.copy(),
                        nulls_first=s.nulls_first,
                        stable=s.stable,
                        limit=Optional(length),
                        schema=s.schema(),
                    )
                )
        return DynRelation(Limit(input=self, offset=offset, length=length))


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------


struct InMemoryTable(Relation):
    """Leaf backed by a RecordBatch; opens into a morsel-slicing operator."""

    var batch: RecordBatch
    var morsel_size: Int

    def __init__(
        out self, *, batch: RecordBatch, morsel_size: Int = DEFAULT_MORSEL_SIZE
    ):
        self.batch = RecordBatch(copy=batch)
        self.morsel_size = morsel_size

    def schema(self) -> Schema:
        return Schema(copy=self.batch.schema)

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        return InMemoryTableProcessor(
            batch=RecordBatch(copy=self.batch),  # shares buffers (O(1))
            morsel_size=self.morsel_size,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            t"InMemoryTable(num_rows={self.batch.num_rows()},"
            t" schema={self.batch.schema})"
        )


def in_memory_table(
    batch: RecordBatch, morsel_size: Int = DEFAULT_MORSEL_SIZE
) -> DynRelation:
    """Create a relation backed by an in-memory RecordBatch.

    ``morsel_size`` is the number of rows each ``pull()`` yields. It never
    changes the result — only how many slices the pipeline is driven in — so it
    exists mainly to exercise the streaming boundary from tests."""
    return InMemoryTable(batch=batch, morsel_size=morsel_size)


struct ParquetScan[leaves: LeafSet = LeafSet.all()](Relation):
    """Leaf describing a Parquet file scan with a known schema.

    The schema doubles as the **projection**: the scan reads only its own
    columns out of the file, so narrowing a scan's schema is how a projection is
    pushed into it. The file is read one row group at a time.

    An optional `predicate` is pushed-down pruning metadata: the scan uses it to
    skip row groups (and, later, pages) whose statistics prove no row can match.
    It never changes the rows returned — a `Filter` above the scan applies the
    predicate exactly — so it is safe to carry a partial/over-approximate one.
    """

    var path: String
    var _schema: Schema
    var morsel_size: Int
    var predicate: Optional[BoxedValue]

    def __init__(
        out self,
        *,
        var path: String,
        var schema: Schema,
        morsel_size: Int = DEFAULT_MORSEL_SIZE,
        var predicate: Optional[BoxedValue] = None,
    ):
        self.path = path^
        self._schema = schema^
        self.morsel_size = morsel_size
        self.predicate = predicate^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def with_predicate(
        self, var predicate: BoxedValue
    ) raises -> Optional[ArcPointer[NoneType]]:
        """Accept the predicate as pruning metadata, **keeping `leaves`** —
        which is the point: `Self.leaves` is in scope here and a downcast at the
        call site could not have named it."""
        var pushed = ParquetScan[Self.leaves](
            path=self.path.copy(),
            schema=self.schema(),
            morsel_size=self.morsel_size,
            predicate=Optional(predicate^),
        )
        return Optional(rebind[ArcPointer[NoneType]](ArcPointer(pushed^)))

    def with_projection(
        self, needed: List[String]
    ) raises -> Optional[ArcPointer[NoneType]]:
        """Narrow the schema — and therefore the projection — to `needed`.

        This is the whole point of the rewrite: the scan reads only the columns
        its schema names, so a 105-column file queried for one column stops
        decoding the other 104. Field order is the file's, not `needed`'s, so a
        pushdown never silently reorders a scan's output.

        **A scan never narrows to nothing.** A plan that reads no column at all
        (`COUNT(*)` — `lit(1).count()` references none) still needs the scan to
        report the file's row count, and a zero-column read yields zero-row
        batches, which the streaming loop reads as end-of-file. So the empty case
        keeps one column, and picks the narrowest fixed-width one available
        (`byte_width()` is 0 for the variable-width types, which is exactly the
        set to avoid here) — reading one `uint8` beats reading one `binary`.

        Keeps `leaves` for the reason `with_predicate` does: `Self.leaves` is in
        scope here and a downcast at the call site could not have named it."""
        var fields = List[Field]()
        for ref f in self._schema.fields:
            if _has_name(needed, f.name):
                fields.append(f.copy())
        if len(fields) == 0 and len(self._schema.fields) > 0:
            var best = 0
            var best_width = self._schema.fields[0].dtype.byte_width()
            for i in range(1, len(self._schema.fields)):
                var w = self._schema.fields[i].dtype.byte_width()
                if w > 0 and (best_width == 0 or w < best_width):
                    best = i
                    best_width = w
            fields.append(self._schema.fields[best].copy())
        var pushed = ParquetScan[Self.leaves](
            path=self.path.copy(),
            schema=Schema(fields=fields^),
            morsel_size=self.morsel_size,
            predicate=self.predicate.copy(),
        )
        return Optional(rebind[ArcPointer[NoneType]](ArcPointer(pushed^)))

    def kind(self) -> Int:
        return RELATION_PARQUET_SCAN

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        return ParquetScanProcessor[Self.leaves](
            path=self.path.copy(),
            schema=Schema(copy=self._schema),
            morsel_size=self.morsel_size,
            predicate=self.predicate.copy(),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"ParquetScan({self.path})")


def parquet_scan[
    leaves: LeafSet = LeafSet.all()
](
    path: String, schema: Schema, morsel_size: Int = DEFAULT_MORSEL_SIZE
) -> DynRelation:
    """Create a Parquet file scan with a known schema.

    The schema *is* the projection: only its columns are read out of the file.
    ``morsel_size`` bounds how many rows each ``pull()`` yields (morsels never
    straddle a row-group boundary), and never changes the result."""
    return ParquetScan[leaves](
        path=path, schema=schema, morsel_size=morsel_size
    )


# ---------------------------------------------------------------------------
# Streaming operators
# ---------------------------------------------------------------------------


struct Filter(Relation):
    """Apply a boolean predicate; keep rows where True (schema unchanged)."""

    var input: DynRelation
    var predicate: BoxedValue

    def __init__(
        out self, *, var input: DynRelation, var predicate: BoxedValue
    ):
        self.input = input^
        self.predicate = predicate^

    def schema(self) -> Schema:
        return self.input.schema()

    def children(self) -> List[DynRelation]:
        return [DynRelation(copy=self.input)]

    def with_projection(
        self, needed: List[String]
    ) raises -> Optional[ArcPointer[NoneType]]:
        """The predicate's columns are needed too — they are read here even when
        nothing above reads them."""
        var below = needed.copy()
        _add_names(below, self.predicate.referenced_columns())
        var pushed = Filter(
            input=self.input.with_projection(below),
            predicate=self.predicate.copy(),
        )
        return Optional(rebind[ArcPointer[NoneType]](ArcPointer(pushed^)))

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        return FilterProcessor(
            input=self.input.to_processor(ctx),
            predicate=self.predicate.copy(),
            ctx=ctx.copy(),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Filter(predicate=")
        self.predicate.write_to(writer)
        writer.write(t")")


struct Project(Relation):
    """Evaluate a list of named expressions into output columns."""

    var input: DynRelation
    var names: List[String]
    var values: List[BoxedValue]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: DynRelation,
        var names: List[String],
        var values: List[BoxedValue],
        var schema: Schema,
    ):
        self.input = input^
        self.names = names^
        self.values = values^
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def children(self) -> List[DynRelation]:
        return [DynRelation(copy=self.input)]

    def with_projection(
        self, needed: List[String]
    ) raises -> Optional[ArcPointer[NoneType]]:
        """Drop the output columns nobody reads, then ask the input only for
        what the surviving expressions reference.

        Dropping outputs is what makes the rewrite transitive: without it
        `t.select("a", "b").aggregate(keys=[col("a")], ...)` still reads `b` off
        disk, because the input set would be taken from *every* value here
        rather than the ones that survive. Surviving columns keep this node's own
        order, not `needed`'s, so a pushdown never reorders a projection."""
        var out_names = List[String]()
        var out_values = List[BoxedValue]()
        var fields = List[Field]()
        for i in range(len(self.names)):
            if _has_name(needed, self.names[i]):
                out_names.append(self.names[i].copy())
                out_values.append(self.values[i].copy())
                fields.append(self._schema.fields[i].copy())
        if len(out_names) == 0 and len(self.names) > 0:
            # A zero-column relation carries no row count, so a `COUNT(*)` above
            # would see an empty stream. Keep one column, as the scan does.
            out_names.append(self.names[0].copy())
            out_values.append(self.values[0].copy())
            fields.append(self._schema.fields[0].copy())
        var below = _referenced_by(out_values)
        var pushed = Project(
            input=self.input.with_projection(below),
            names=out_names^,
            values=out_values^,
            schema=Schema(fields=fields^),
        )
        return Optional(rebind[ArcPointer[NoneType]](ArcPointer(pushed^)))

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        return ProjectProcessor(
            input=self.input.to_processor(ctx),
            values=self.values.copy(),
            schema=Schema(copy=self._schema),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Project([")
        for i in range(len(self.names)):
            if i > 0:
                writer.write(t", ")
            writer.write(self.names[i])
            writer.write(t"=")
            self.values[i].write_to(writer)
        writer.write(t"])")


struct Limit(Relation):
    """Row limit/offset — streaming: skip ``offset`` rows, keep ``length``."""

    var input: DynRelation
    var offset: Int
    var length: Int

    def __init__(out self, *, var input: DynRelation, offset: Int, length: Int):
        self.input = input^
        self.offset = offset
        self.length = length

    def schema(self) -> Schema:
        return self.input.schema()

    def children(self) -> List[DynRelation]:
        return [DynRelation(copy=self.input)]

    def with_projection(
        self, needed: List[String]
    ) raises -> Optional[ArcPointer[NoneType]]:
        """Reads no column of its own — `needed` passes straight through."""
        var pushed = Limit(
            input=self.input.with_projection(needed),
            offset=self.offset,
            length=self.length,
        )
        return Optional(rebind[ArcPointer[NoneType]](ArcPointer(pushed^)))

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        return LimitProcessor(
            input=self.input.to_processor(ctx),
            offset=self.offset,
            length=self.length,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Limit(length={self.length}, offset={self.offset})")


# ---------------------------------------------------------------------------
# Blocking operators
# ---------------------------------------------------------------------------


struct Sort(Relation):
    """Sort by key expressions — the descriptive node (keys, order, limit).

    A pipeline breaker; schema is unchanged from the input. An optional ``limit``
    turns it into a top-K (the sort kernel returns only the first ``limit``
    rows), which the plan builder folds in when a ``Limit`` immediately follows a
    ``Sort``."""

    var input: DynRelation
    var keys: List[BoxedValue]
    var ascending: List[Bool]
    var nulls_first: Bool
    var stable: Bool
    var limit: Optional[Int]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: DynRelation,
        var keys: List[BoxedValue],
        var ascending: List[Bool],
        nulls_first: Bool,
        stable: Bool,
        var limit: Optional[Int],
        var schema: Schema,
    ):
        self.input = input^
        self.keys = keys^
        self.ascending = ascending^
        self.nulls_first = nulls_first
        self.stable = stable
        self.limit = limit^
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def kind(self) -> Int:
        return RELATION_SORT

    def children(self) -> List[DynRelation]:
        return [DynRelation(copy=self.input)]

    def with_projection(
        self, needed: List[String]
    ) raises -> Optional[ArcPointer[NoneType]]:
        """The sort keys are read here even when nothing above reads them."""
        var below = needed.copy()
        _add_names(below, _referenced_by(self.keys))
        var input = self.input.with_projection(below)
        # `Sort` stores its schema rather than deriving it, so it has to be
        # re-read off the rewritten input — every other passthrough node here
        # answers `self.input.schema()` and needs no such care.
        var schema = input.schema()
        var pushed = Sort(
            input=input^,
            keys=self.keys.copy(),
            ascending=self.ascending.copy(),
            nulls_first=self.nulls_first,
            stable=self.stable,
            limit=self.limit.copy(),
            schema=schema^,
        )
        return Optional(rebind[ArcPointer[NoneType]](ArcPointer(pushed^)))

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        return SortProcessor(
            input=self.input.to_processor(ctx),
            keys=self.keys.copy(),
            ascending=self.ascending.copy(),
            nulls_first=self.nulls_first,
            stable=self.stable,
            limit=self.limit.copy(),
            schema=Schema(copy=self._schema),
            ctx=ctx.copy(),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Sort(keys=[")
        for i in range(len(self.keys)):
            if i > 0:
                writer.write(", ")
            self.keys[i].write_to(writer)
        writer.write("]")
        if self.limit:
            writer.write(t", limit={self.limit.value()}")
        writer.write(")")


struct Aggregate(Relation):
    """Grouped aggregation — the descriptive node (keys, aggregates, schema).

    ``aggs[i]`` is the aggregate applied to the value expression ``inputs[i]``.
    Each carries a *comptime* ``Aggregation``, so a plan built from fused values
    and ``AggFunc.of[A]`` carries no function-name interpretation at all;
    ``DynRelation.aggregate`` produces the same node from runtime names."""

    var input: DynRelation
    var keys: List[BoxedValue]
    var inputs: List[BoxedValue]
    var aggs: List[AggFunc]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: DynRelation,
        var keys: List[BoxedValue],
        var inputs: List[BoxedValue],
        var aggs: List[AggFunc],
        var schema: Schema,
    ):
        self.input = input^
        self.keys = keys^
        self.inputs = inputs^
        self.aggs = aggs^
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def children(self) -> List[DynRelation]:
        return [DynRelation(copy=self.input)]

    def with_projection(
        self, needed: List[String]
    ) raises -> Optional[ArcPointer[NoneType]]:
        """`needed` is discarded: an aggregate's output columns are not its
        input's, so what the input must supply is the group keys **and** the
        aggregate input expressions — nothing else, however wide the input is.

        Dropping an unread aggregate would be a separate rewrite and is not one:
        `count(*)`'s column set is empty, so an aggregate list is not a safe
        proxy for "reads nothing"."""
        var below = _referenced_by(self.keys)
        _add_names(below, _referenced_by(self.inputs))
        var pushed = Aggregate(
            input=self.input.with_projection(below),
            keys=self.keys.copy(),
            inputs=self.inputs.copy(),
            aggs=self.aggs.copy(),
            schema=Schema(copy=self._schema),
        )
        return Optional(rebind[ArcPointer[NoneType]](ArcPointer(pushed^)))

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        return AggregateProcessor(
            input=self.input.to_processor(ctx),
            keys=self.keys.copy(),
            inputs=self.inputs.copy(),
            aggs=self.aggs.copy(),
            schema=Schema(copy=self._schema),
            ctx=ctx.copy(),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Aggregate(keys=[")
        for i in range(len(self.keys)):
            if i > 0:
                writer.write(", ")
            self.keys[i].write_to(writer)
        writer.write("])")


struct Join(Relation):
    """Equijoin — the descriptive node (key indices, kind, output schema)."""

    var left: DynRelation
    var right: DynRelation
    var left_key_indices: List[Int]
    var right_key_indices: List[Int]
    var join_kind: JoinKind
    var strictness: UInt8
    var _schema: Schema

    def __init__(
        out self,
        *,
        var left: DynRelation,
        var right: DynRelation,
        var left_key_indices: List[Int],
        var right_key_indices: List[Int],
        join_kind: JoinKind,
        strictness: UInt8,
        var schema: Schema,
    ):
        self.left = left^
        self.right = right^
        self.left_key_indices = left_key_indices^
        self.right_key_indices = right_key_indices^
        self.join_kind = join_kind
        self.strictness = strictness
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def children(self) -> List[DynRelation]:
        return [DynRelation(copy=self.left), DynRelation(copy=self.right)]

    # No `with_projection`: the join's key **indices** are positions into its
    # children's schemas, fixed when the node was built, so narrowing a child
    # would silently join on the wrong columns. Pushing a projection through a
    # join means recomputing those indices and the output schema — a separate
    # rewrite with its own correctness conditions, not a wider `needed` set. The
    # inherited default leaves the whole subtree alone, which is the safe answer.

    def to_processor(self, ctx: ExecContext) raises -> DynProcessor:
        return JoinProcessor(
            left=self.left.to_processor(ctx),
            right=self.right.to_processor(ctx),
            left_key_indices=self.left_key_indices.copy(),
            right_key_indices=self.right_key_indices.copy(),
            join_kind=self.join_kind,
            strictness=self.strictness,
            schema=Schema(copy=self._schema),
            ctx=ctx.copy(),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Join(kind={self.join_kind})")
