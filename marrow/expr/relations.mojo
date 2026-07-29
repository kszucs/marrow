"""Relational plans: a descriptive IR that opens into pull-based operators.

Two layers, cleanly separated:

- ``Relation`` nodes are **pure, immutable descriptions** (``kind``/``schema``/
  ``to_processor``): they hold only their parameters and child relations, no execution
  state. ``AnyRelation`` erases them behind an ``ArcPointer``, so copying a plan
  is an O(1) share and the plan is a reusable, inspectable, rewritable template.
- ``Processor`` (``schema``/``pull``) is the executing layer, built by
  ``Relation.to_processor(ctx)``; it owns *all* mutable state (scan offset, built hash
  index, grouper, child operators). ``AnyProcessor`` erases it and drives the
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
``AnyRelation.select(*names)``                   — project columns by name.
``AnyRelation.project(names, values)``           — project computed expressions.
``AnyRelation.filter(pred)``                     — filter rows by predicate.
``AnyRelation.aggregate(keys, aggs)``            — grouped aggregation
(``HAVING`` is a ``.filter(...)`` on top of it).
``AnyRelation.sort(keys, ascending)`` / ``.limit(n[, offset])`` — order and slice
(``.sort(...).limit(k)`` folds into the sort kernel's top-K path).
``AnyRelation.join(right, left_on, right_on)``   — hash join.
``AnyRelation.execute()``                        — drain to a single RecordBatch.

``project`` and ``aggregate`` each have two overloads, one per expression lane:
one taking interpreted ``DynValue``s (names resolved against the input schema
here) and one taking already-bound ``AnyValue``s, so a fused comptime plan uses
the same API rather than assembling nodes by hand.

Example
-------
    var plan = in_memory_table(batch).filter(col("x") > lit[Int64Type](0)).select("x")
    var result = plan.execute()
"""

from std.memory import ArcPointer

from ..dtypes import Field
from ..schema import Schema
from ..tabular import RecordBatch
from .values import AnyValue, AggExpr
from .dynamic import DynValue, col, LOAD
from ..kernels.execution import ExecutionContext
from .aggregates import AggFunc
from ..parquet import LeafSet
from .execution import (
    DEFAULT_MORSEL_SIZE,
    AnyProcessor,
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
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_ALL,
    JOIN_ANY,
)


# ---------------------------------------------------------------------------
# Relation trait — the descriptive IR node (pure data; no execution state)
# ---------------------------------------------------------------------------


# Relation kind discriminants — let a plan-building step recognise a node behind
# `AnyRelation` (e.g. to push a filter into a `ParquetScan`) without a full RTTI.
comptime RELATION_GENERIC: Int = 0
comptime RELATION_PARQUET_SCAN: Int = 1
comptime RELATION_SORT: Int = 2


trait Relation(ImplicitlyDeletable, Movable):
    """A relational plan node: a pure, immutable description of an operation.

    Nodes hold only their parameters and child relations — no execution state.
    ``to_processor(ctx)`` builds the stateful ``Processor`` that runs (opening children
    recursively), so a plan is a reusable template you can inspect, copy cheaply
    (O(1) — nodes are immutable and shared), and rewrite."""

    def with_predicate(
        self, var predicate: AnyValue
    ) raises -> Optional[ArcPointer[NoneType]]:
        """This node rebuilt to carry `predicate` as **pruning metadata**, as an
        erased pointer — or `None` when it cannot use one, which is every node
        but a scan.

        Returns the erased data rather than an `AnyRelation` on purpose. The
        rebuilt node has the *same concrete type* as this one, so the caller can
        keep its own trampolines and swap only the pointer; returning
        `Optional[AnyRelation]` instead puts `AnyRelation` inside its own
        trampoline field type and Mojo rejects the struct as recursive.

        This replaces a `downcast` to a concrete `ParquetScan`, which stopped
        being correct once `ParquetScan` gained a comptime `LeafSet`: the
        downcast had to name one parameterisation, so a scan compiled for a
        narrow set was silently rebuilt as the default full-ladder one. A
        downcast cannot see a comptime parameter; a virtual can carry it.
        """
        return None

    def schema(self) -> Schema:
        ...

    def to_processor(self, ctx: ExecutionContext) raises -> AnyProcessor:
        """Build the physical operator for this node (opening its children)."""
        ...

    def write_to[W: Writer](self, mut writer: W):
        ...

    def kind(self) -> Int:
        """Node discriminant for plan rewrites; generic unless overridden."""
        return RELATION_GENERIC


# ---------------------------------------------------------------------------
# AnyRelation — type-erased IR container + plan-building API
# ---------------------------------------------------------------------------


struct AnyRelation(ImplicitlyCopyable, Movable, Writable):
    """Type-erased plan node behind an ``ArcPointer``.

    Nodes are immutable descriptions, so copying an ``AnyRelation`` is an O(1)
    ``ArcPointer`` share — no deep clone, no reset. ``to_processor(ctx)`` builds the
    operator tree that executes; the plan is never mutated, so it is a reusable
    template and copies never share execution state. Carries the plan-building
    API (``select``/``filter``/``aggregate``/``join``)."""

    var _data: ArcPointer[NoneType]
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_to_processor: def(
        ArcPointer[NoneType], ExecutionContext
    ) thin raises -> AnyProcessor
    var _virt_write_to_string: def(ArcPointer[NoneType]) thin -> String
    var _virt_drop: def(var ArcPointer[NoneType]) thin
    var _virt_kind: def(ArcPointer[NoneType]) thin -> Int
    var _virt_with_predicate: def(
        ArcPointer[NoneType], var AnyValue
    ) thin raises -> Optional[ArcPointer[NoneType]]

    @staticmethod
    def _tramp_kind[T: Relation](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[T]](ptr)[].kind()

    @staticmethod
    def _tramp_with_predicate[
        T: Relation
    ](ptr: ArcPointer[NoneType], var predicate: AnyValue) raises -> Optional[
        ArcPointer[NoneType]
    ]:
        return rebind[ArcPointer[T]](ptr)[].with_predicate(predicate^)

    @staticmethod
    def _tramp_schema[T: Relation](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    def _tramp_to_processor[
        T: Relation
    ](ptr: ArcPointer[NoneType], ctx: ExecutionContext) raises -> AnyProcessor:
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

    def __init__(out self, *, copy: Self):
        # O(1) share — nodes are immutable, so aliasing is safe.
        self._data = copy._data
        self._virt_schema = copy._virt_schema
        self._virt_to_processor = copy._virt_to_processor
        self._virt_write_to_string = copy._virt_write_to_string
        self._virt_drop = copy._virt_drop
        self._virt_kind = copy._virt_kind
        self._virt_with_predicate = copy._virt_with_predicate

    def __del__(deinit self):
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

    # --- execution ---

    def to_processor(
        self, ctx: ExecutionContext = ExecutionContext()
    ) raises -> AnyProcessor:
        """Build the operator tree for this plan; the plan is left untouched."""
        return self._virt_to_processor(self._data, ctx)

    def execute(
        self, ctx: ExecutionContext = ExecutionContext()
    ) raises -> RecordBatch:
        """Open this plan into a fresh operator tree and drain it into one
        `RecordBatch`.

        The plan is a pure description and is never mutated, so it can be
        executed repeatedly and concurrently."""
        var op = self.to_processor(ctx)
        return op.collect()

    # --- plan-building API ---

    def select(self, *names: String) raises -> AnyRelation:
        """Project columns by name, returning a new plan node."""
        var schema = self.schema()
        var col_names = List[String]()
        var exprs = List[AnyValue]()
        var fields = List[Field]()
        for i in range(len(names)):
            var name = names[i]
            var idx = schema.get_field_index(name)
            if idx == -1:
                raise Error("select: column '" + name + "' not found")
            col_names.append(name)
            exprs.append(AnyValue(col(idx)))
            fields.append(schema.fields[idx].copy())
        return AnyRelation(
            Project(
                input=self,
                names=col_names^,
                values=exprs^,
                schema=Schema(fields=fields^),
            )
        )

    def filter(self, var predicate: AnyValue) raises -> AnyRelation:
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
            var pushed = AnyRelation(copy=self)
            pushed._data = data.take()
            return AnyRelation(Filter(input=pushed^, predicate=predicate^))
        return AnyRelation(Filter(input=self, predicate=predicate^))

    def aggregate(
        self, keys: List[DynValue], aggs: List[AggExpr]
    ) raises -> AnyRelation:
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
        a ``DynValue`` carries the function's *name* until this call resolves it
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
        var key_exprs = List[AnyValue]()
        for i in range(len(keys)):
            var k = keys[i].resolve_names(input_schema)
            var name = input_schema.fields[
                Int(k.kind_data())
            ].name if k.kind() == LOAD else "key" + String(i)
            fields.append(Field(name, k.execute(probe).dtype()))
            key_exprs.append(AnyValue(k^))

        # Resolve each aggregate against the dtype its input turns out to have —
        # the only interpretation step on this path, and the last one: the output
        # dtype is then the aggregation's own (`sum` widens integers to int64;
        # `min`/`max` preserve the input dtype, including a timestamp's unit and
        # timezone; `count` and the distinct counts are int64; `mean` is
        # float64), and an aggregate not defined for the column's type is
        # rejected here rather than at execution.
        var input_exprs = List[AnyValue]()
        var resolved = List[AggFunc]()
        for i in range(len(aggs)):
            var v = aggs[i].input_for(input_schema)
            resolved.append(aggs[i].resolve(v.execute(probe).dtype()))
            fields.append(Field(aggs[i].out_name, resolved[i].out_dtype.copy()))
            input_exprs.append(v^)

        return AnyRelation(
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
        var keys: List[AnyValue],
        var inputs: List[AnyValue],
        var aggs: List[AggFunc],
        names: List[String],
    ) raises -> AnyRelation:
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
        return AnyRelation(
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
        right: AnyRelation,
        left_on: List[DynValue],
        right_on: List[DynValue],
        how: UInt8 = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
    ) raises -> AnyRelation:
        """Hash join on equijoin key expressions."""
        if len(left_on) != len(right_on):
            raise Error("join: len(left_on) != len(right_on)")

        var left_schema = self.schema()
        var right_schema = right.schema()

        # Resolve key expressions to positional column indices (each key must be
        # a bare column reference — column_index raises otherwise).
        var left_indices = List[Int]()
        for ref k in left_on:
            left_indices.append(k.column_index(left_schema))
        var right_indices = List[Int]()
        for ref k in right_on:
            right_indices.append(k.column_index(right_schema))

        # Output schema: left columns + (suffixed) right columns.
        var fields = List[Field]()
        for ref f in left_schema.fields:
            fields.append(f.copy())
        if how != JOIN_SEMI and how != JOIN_ANTI:
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

        return AnyRelation(
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
        self, names: List[String], var values: List[AnyValue]
    ) raises -> AnyRelation:
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

        There is deliberately no ``List[DynValue]`` overload: ``AnyValue``
        converts implicitly from ``DynValue``, so a second overload would be
        shadowed by this one at every call site that passes a list literal —
        reachable only by spelling out the conversion, which is not an API."""
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
        return AnyRelation(
            Project(
                input=self,
                names=names.copy(),
                values=values^,
                schema=Schema(fields=fields^),
            )
        )

    def sort(
        self,
        keys: List[DynValue],
        ascending: List[Bool],
        nulls_first: Bool = True,
        stable: Bool = True,
    ) raises -> AnyRelation:
        """Sort by one or more key expressions (a pipeline breaker).

        Each key has its own ascending/descending direction; ``nulls_first`` and
        ``stable`` apply to all keys. Keys resolve by name against the input
        schema. The output schema is unchanged from the input."""
        if len(keys) == 0:
            raise Error("sort: keys must not be empty")
        if len(keys) != len(ascending):
            raise Error("sort: len(keys) != len(ascending)")

        var input_schema = self.schema()
        var key_exprs = List[AnyValue]()
        for ref k in keys:
            key_exprs.append(AnyValue(k.resolve_names(input_schema)))
        return AnyRelation(
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

    def limit(self, length: Int, offset: Int = 0) raises -> AnyRelation:
        """Keep at most ``length`` rows after skipping ``offset`` rows.

        Top-K fast path: when this limits a ``Sort`` with no existing limit and
        ``offset == 0``, the limit is folded into the sort (``Sort(limit=…)``),
        driving the sort kernel's top-K path instead of a full sort. Otherwise a
        streaming ``Limit`` operator is layered on top."""
        if offset == 0 and self.kind() == RELATION_SORT:
            ref s = self.downcast[Sort]()[]
            if not s.limit:
                return AnyRelation(
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
        return AnyRelation(Limit(input=self, offset=offset, length=length))


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

    def to_processor(self, ctx: ExecutionContext) raises -> AnyProcessor:
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
) -> AnyRelation:
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
    var predicate: Optional[AnyValue]

    def __init__(
        out self,
        *,
        var path: String,
        var schema: Schema,
        morsel_size: Int = DEFAULT_MORSEL_SIZE,
        var predicate: Optional[AnyValue] = None,
    ):
        self.path = path^
        self._schema = schema^
        self.morsel_size = morsel_size
        self.predicate = predicate^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def with_predicate(
        self, var predicate: AnyValue
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

    def kind(self) -> Int:
        return RELATION_PARQUET_SCAN

    def to_processor(self, ctx: ExecutionContext) raises -> AnyProcessor:
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
) -> AnyRelation:
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

    var input: AnyRelation
    var predicate: AnyValue

    def __init__(out self, *, var input: AnyRelation, var predicate: AnyValue):
        self.input = input^
        self.predicate = predicate^

    def schema(self) -> Schema:
        return self.input.schema()

    def to_processor(self, ctx: ExecutionContext) raises -> AnyProcessor:
        return FilterProcessor(
            input=self.input.to_processor(ctx), predicate=self.predicate.copy()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Filter(predicate=")
        self.predicate.write_to(writer)
        writer.write(t")")


struct Project(Relation):
    """Evaluate a list of named expressions into output columns."""

    var input: AnyRelation
    var names: List[String]
    var values: List[AnyValue]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: AnyRelation,
        var names: List[String],
        var values: List[AnyValue],
        var schema: Schema,
    ):
        self.input = input^
        self.names = names^
        self.values = values^
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def to_processor(self, ctx: ExecutionContext) raises -> AnyProcessor:
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

    var input: AnyRelation
    var offset: Int
    var length: Int

    def __init__(out self, *, var input: AnyRelation, offset: Int, length: Int):
        self.input = input^
        self.offset = offset
        self.length = length

    def schema(self) -> Schema:
        return self.input.schema()

    def to_processor(self, ctx: ExecutionContext) raises -> AnyProcessor:
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

    var input: AnyRelation
    var keys: List[AnyValue]
    var ascending: List[Bool]
    var nulls_first: Bool
    var stable: Bool
    var limit: Optional[Int]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: AnyRelation,
        var keys: List[AnyValue],
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

    def to_processor(self, ctx: ExecutionContext) raises -> AnyProcessor:
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
    ``AnyRelation.aggregate`` produces the same node from runtime names."""

    var input: AnyRelation
    var keys: List[AnyValue]
    var inputs: List[AnyValue]
    var aggs: List[AggFunc]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: AnyRelation,
        var keys: List[AnyValue],
        var inputs: List[AnyValue],
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

    def to_processor(self, ctx: ExecutionContext) raises -> AnyProcessor:
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

    var left: AnyRelation
    var right: AnyRelation
    var left_key_indices: List[Int]
    var right_key_indices: List[Int]
    var join_kind: UInt8
    var strictness: UInt8
    var _schema: Schema

    def __init__(
        out self,
        *,
        var left: AnyRelation,
        var right: AnyRelation,
        var left_key_indices: List[Int],
        var right_key_indices: List[Int],
        join_kind: UInt8,
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

    def to_processor(self, ctx: ExecutionContext) raises -> AnyProcessor:
        return JoinProcessor(
            left=self.left.to_processor(ctx),
            right=self.right.to_processor(ctx),
            left_key_indices=self.left_key_indices.copy(),
            right_key_indices=self.right_key_indices.copy(),
            join_kind=self.join_kind,
            strictness=self.strictness,
            schema=Schema(copy=self._schema),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Join(kind={self.join_kind})")
