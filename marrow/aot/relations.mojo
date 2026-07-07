"""Named comptime-typed expression nodes for the fully-monomorphized (AOT)
relational layer.

The leaf nodes here — ``NumericColumn[T]`` and ``StringColumn`` — carry only
their ``name`` (a runtime field); the column *type* encodes just the dtype
(which is what drives the SIMD ``core``). A query with many int64 columns
instantiates ``NumericColumn[Int64Type]`` once, not one type per column: the
field name is metadata that never affects the generated compute. The column
position is resolved by name against ``batch.schema`` at execution — a
loop-invariant lookup, hoisted out of the fused loop.

You never spell those leaf nodes by hand. Instead you declare a plain struct
of dtype-tag fields and access columns through a handle:

    struct Orders:
        var a: Int64Type
        var b: Int64Type
        var name: StringType

    var t = Table[Orders]()
    t.a     # NumericColumn[Int64Type]("a")   (numeric)
    t.name  # StringColumn("name")            (string)

``Table[Orders]()`` is a column-access handle whose ``__getattr_param__``
reflects each field's *dtype* on ``Orders`` at compile time
(``reflect[Orders].field[name].T``), then a ``where`` clause picks the numeric
(``NumericColumn``) or string (``StringColumn``) overload — passing the name as
a runtime constructor argument. The struct's fields are plain dtype tags used
*only* for reflection — they are never instantiated, so ``Orders`` needs no
``__init__``.

For a schema-struct-free, polars-style surface, ``col(name, dtype)`` (defined in
``values.mojo``) references a column by name (``col("a", int64)``,
``col("name", string)``) and resolves its position against the batch at
execution. It produces the same ``NumericColumn[T]`` / ``StringColumn`` leaves,
so it composes identically (``Add(col("a", int64), col("b", int64))``,
``Project``/``Filter``).

This file's one ``AnyArray`` erasure boundary (``Project``/``Filter``
assembling heterogeneous columns) stays cheap because it's *closed* —
nothing outside this file constructs an ``AnyArray`` of a dtype a given
query doesn't use, so the compiler can prove and prune the rest of
``filter()``'s/kernels' per-dtype branches. Calling into
``marrow.dyn.executor`` (``Planner``, ``*Processor``) or
``marrow.dyn.values.Expr.eval()`` instead reaches a genuinely *open*
dispatcher built to stay ready for dtypes/node-kinds it can't know ahead of
time, and nothing there can be pruned. Measured in ``benchmarks/binary_size/``:
a ``Project``+``Filter`` plan compiles ~31x smaller (stripped) than the same
query on ``marrow.dyn``'s ``AnyRelation``/``Expr``; a hybrid variant
that only fuses the *predicate* but still calls into the executor is the
same size as the fully runtime one.

See ``docs/aot-relations-design.md`` for the full design.

Usage::

    struct Orders:
        var a: Int32Type
        var b: StringType

    var t = Table[Orders]()
    var result = t.a.execute(batch)  # resolves the "a" column against the batch
"""

from std.reflection import reflect

import marrow.dtypes as dt
from marrow.arrays import AnyArray, StringArray
from marrow.dtypes import Field
from marrow.kernels.filter import filter
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.aot.values import BoolValue, NumericValue, StringValue, Value


# ---------------------------------------------------------------------------
# Named — trait for expression nodes carrying a compile-time field name
# ---------------------------------------------------------------------------


trait Named:
    """Trait for expression nodes with a compile-time column name, used by
    ``Project`` to derive output field names from the expression tree's type
    alone. Declared as a method (not a bare ``comptime name`` alias) because
    converting a ``StringLiteral`` type parameter to ``String`` only resolves
    reliably from inside the concrete node's own method body — accessing
    ``E.name`` from a separate generic function parameterized over
    ``E: Named`` hits compiler limitations (confirmed against the pinned
    toolchain; see ``docs/aot-relations-design.md``).
    """

    def field_name(self) -> String:
        ...


# ---------------------------------------------------------------------------
# Column — base trait for the named leaf column nodes
# ---------------------------------------------------------------------------


trait Column(Named, Value):
    """Base trait for the named leaf column nodes (``NumericColumn`` /
    ``StringColumn``).

    Unifies them behind ``field_name()`` (from ``Named``) and ``to_array()``,
    so ``Project`` can assemble a projection over a heterogeneous column pack
    without dispatching on the numeric-vs-string execution split — each column
    knows how to erase its own fused result to ``AnyArray``.
    """

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        """Execute this column against *batch* and erase the result to
        ``AnyArray`` (the one deliberate erasure boundary in ``Project``)."""
        ...


# ---------------------------------------------------------------------------
# Relation — trait for fully-typed relational plan nodes
# ---------------------------------------------------------------------------


trait Relation(ImplicitlyDeletable, Movable):
    """Trait for nodes in the fully-typed relational layer (``Project``,
    ``Filter``, ...). Mirrors ``marrow.dyn.relations``'s ``Relation``
    trait but for the comptime-typed plan tree — ``execute(batch)`` runs the
    whole plan against a source batch and returns the result directly, with
    no processor/pull-based pipeline (every node here is a single pass).
    """

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        ...


# ---------------------------------------------------------------------------
# NumericColumn — named typed column reference
# ---------------------------------------------------------------------------


struct NumericColumn[T: dt.NumericType](Column, Named, NumericValue):
    """Named typed numeric column reference.

    Carries only its ``name`` (a runtime field); the type parameter is just the
    dtype, which drives the SIMD ``core``. A query with 20 int64 columns
    instantiates ``NumericColumn[Int64Type]`` once, not 20 distinct types — the
    field name is metadata and never affects the generated compute. The column
    position is resolved by name against ``batch.schema`` at execution (a
    loop-invariant lookup, hoisted out of the fused loop).

    You never construct this directly — ``Table[Tbl]()`` and ``col(name, dtype)``
    produce it.
    """

    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var name: String

    def __init__(out self, var name: String):
        self.name = name^

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return (
            batch.columns[batch.schema.get_field_index(self.name)]
            .as_primitive[Self.T]()
            .values()
            .load[W](idx)
        )

    def field_name(self) -> String:
        return self.name.copy()

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        return self.execute(batch).to_any()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Col[", self.name, "]")


# ---------------------------------------------------------------------------
# StringColumn — named typed string column reference, resolved by name
# ---------------------------------------------------------------------------


struct StringColumn(Column, Named, StringValue):
    """Named typed string column reference.

    The string counterpart of ``NumericColumn[T]``, produced by ``Table[Tbl]()``
    for string-typed fields. Carries only its ``name`` (there is no dtype
    parameter — string is a single physical type), so there is exactly one
    ``StringColumn`` type across all string columns; the position is resolved by
    name against ``batch.schema``.
    """

    var name: String

    def __init__(out self, var name: String):
        self.name = name^

    def resolve(self, batch: RecordBatch) -> StringArray:
        return (
            batch.columns[batch.schema.get_field_index(self.name)]
            .as_string()
            .copy()
        )

    def execute(self, batch: RecordBatch) raises -> StringArray:
        return self.resolve(batch)

    def field_name(self) -> String:
        return self.name.copy()

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        return self.execute(batch).to_any()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("StrCol[", self.name, "]")


# ---------------------------------------------------------------------------
# Table — column-access handle over a plain schema struct
# ---------------------------------------------------------------------------


struct Table[T: AnyType](Copyable, Movable):
    """Column-access handle over a plain schema struct — ``Table[Orders]()``.

    ``T`` is any struct whose fields are plain dtype tags (``var a: Int64Type``).
    Attribute access reflects the named field's **dtype** on ``T`` at compile
    time (``reflect[T].field[name].T``, via the ``_dtype`` alias) to pick the
    column type; the position is resolved by name at execution, same as
    ``col(name, dtype)``. So ``Table[Orders]()`` is really just the ``col``
    surface with the dtypes read off a struct instead of spelled per reference.

    A handle is needed because ``T``'s own fields shadow ``__getattr_param__``:
    ``T().a`` would read the ``Int64Type`` field value, not a column node.
    ``T`` itself is never instantiated (only reflected), so it needs no
    ``__init__``.

    The two ``__getattr_param__`` overloads are irreducible — they return
    different node types (``NumericColumn`` vs ``StringColumn``), and Mojo has
    no way to pick a return type by a ``comptime`` condition — so a ``where``
    clause routes numeric fields to one and string fields to the other. The
    ``_dtype`` alias folds to a builtin KGEN attribute, so the constraint solver
    can prove the ``where`` clause during overload selection (a plain ``def``
    lookup would not fold, which is why reflection on a declared struct is the
    only way to do this dispatch from a *name*).
    """

    comptime _dtype[name: StringLiteral] = reflect[Self.T].field[name].T

    def __init__(out self):
        pass

    @always_inline
    def __getattr_param__[
        name: StringLiteral
    ](self) -> NumericColumn[Self._dtype[name]] where conforms_to(
        Self._dtype[name], dt.NumericType
    ):
        return NumericColumn[Self._dtype[name]](String(name))

    @always_inline
    def __getattr_param__[
        name: StringLiteral
    ](self) -> StringColumn where conforms_to(
        Self._dtype[name], dt.StringLikeType
    ):
        return StringColumn(String(name))


# ---------------------------------------------------------------------------
# Project — variadic, fully-typed projection over named expression nodes
# ---------------------------------------------------------------------------


struct Project[*Es: Column](Relation):
    """Fully-typed projection: evaluates a fixed, heterogeneous list of named
    columns and assembles the results into a ``RecordBatch``.

    Each ``Es[i]`` is a ``Column`` (``NumericColumn`` or ``StringColumn``) that
    executes as its own fully-monomorphized, fused kernel. The *only* dynamic
    step is collecting the heterogeneous per-column results into
    ``List[AnyArray]`` / ``RecordBatch`` via ``Column.to_array()`` — inherently
    heterogeneous and O(#columns) — the one deliberate erasure boundary (see
    ``docs/aot-relations-design.md``). Bounding on ``Column`` (rather than the
    broader ``Value``) lets ``execute`` call ``to_array()``/``field_name()``
    uniformly, with no numeric-vs-string branching.

    Construction takes an already-built ``Tuple[*Es]``, not bare variadic
    args (``Project(t.a, t.b)``) — a ``VariadicPack`` captured by one
    function cannot be forwarded to another function's variadic parameter in
    current Mojo (confirmed against the pinned toolchain); only a
    freshly-constructed ``Tuple(a, b, ...)`` at the call site works. Usage::

        var proj = Project(Tuple(t.a, t.b))
        var result = proj.execute(batch)
    """

    var exprs: Tuple[*Self.Es]

    def __init__(out self, var exprs: Tuple[*Self.Es]):
        self.exprs = exprs^

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        var cols = List[AnyArray]()
        var fields = List[Field]()

        comptime for i in range(Self.Es.__len__()):
            ref e = self.exprs[i]
            var arr = e.to_array(batch)
            fields.append(Field(e.field_name(), arr.dtype()))
            cols.append(arr^)

        return RecordBatch(Schema(fields=fields^), cols^)

    def filter[P: BoolValue](var self, var predicate: P) -> Filter[Self, P]:
        """Wrap this projection in a row filter, returning a new plan node."""
        return Filter(self^, predicate^)


# ---------------------------------------------------------------------------
# Filter — row filter over a fully-typed relation, by a fused predicate
# ---------------------------------------------------------------------------


struct Filter[Input: Relation, Pred: BoolValue](Relation):
    """Filter — apply a boolean predicate to a typed relation's rows.

    ``predicate`` is evaluated against the *original* input ``batch`` passed
    to ``execute`` (not against ``input``'s projected output) — the
    predicate's ``NumericColumn`` nodes are typed against the source ``Table``, so
    they must resolve against the batch matching that table's column order,
    exactly like SQL's ``WHERE`` clause can reference columns absent from the
    ``SELECT`` list.

    Usage::

        var plan = Project(Tuple(t.a, t.b)).filter(Gt(t.a, t.b))
        var result = plan.execute(batch)
    """

    var input: Self.Input
    var predicate: Self.Pred

    def __init__(out self, var input: Self.Input, var predicate: Self.Pred):
        self.input = input^
        self.predicate = predicate^

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        var mask = self.predicate.execute(batch).to_any()
        var projected = self.input.execute(batch)
        var cols = List[AnyArray]()
        for i in range(len(projected.columns)):
            cols.append(filter(projected.columns[i].copy(), mask.copy()))
        return RecordBatch(projected.schema.copy(), cols^)


# ---------------------------------------------------------------------------
# execute — free-function entry point, mirrors marrow.dyn.execute(plan)
# ---------------------------------------------------------------------------


def execute[T: Relation](plan: T, batch: RecordBatch) raises -> RecordBatch:
    """Execute a fully-typed relational plan against a batch.

    Equivalent to calling ``plan.execute(batch)`` directly — provided so the
    ``marrow.aot`` / ``marrow.dyn`` packages read the same at the call site
    (``execute(plan)`` on the ``dyn`` side takes an already-bound
    ``AnyRelation``; here ``batch`` is passed alongside since the typed plan
    itself carries no data).
    """
    return plan.execute(batch)


# struct Table[*Ts: datatype]:
#     var columns: Tuple[*Ts]

#     def __init__(out self, var columns: Tuple[*Ts]):
#         self.columns = columns^

#     def __getattr_param__[name: StringLiteral](self) -> Self.Ts[reflect[Self.Ts].field_index[name]()] where conforms_to(
#         reflect[Self.Ts].field[name].T, dt.NumericType
#     ):
#         return self.columns[reflect[Self.Ts].field_index[name]()]

# def table[*Ts](...): return Table[*Ts](Tuple(*Ts)())

# table(field["a"](int64), field["b"](int64), field["name"](string))
# should not be necessary, I mean "a"/"b" the field names could be
# passed entirely at runtime, this way we wouldn't generate
# new code paths for the same type of data just because
# the field name is different
