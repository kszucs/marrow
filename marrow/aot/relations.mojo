"""Named comptime-typed expression nodes for the fully-monomorphized (AOT)
relational layer.

Unlike ``values.mojo``'s ``Column[T]`` (a runtime ``index: Int`` field,
populated however the caller likes), ``Column[Tbl, name, T]`` here has
**zero runtime fields** — ``index`` is a ``comptime`` constant derived by
reflecting ``name``'s field position directly on the enclosing ``Tbl``
struct via ``reflect[Tbl].field_index[name]()``. The whole
name -> position -> type binding is resolved entirely at compile time; there
is no runtime ``Schema`` object involved anywhere in this file. This is what
makes ``t.a`` genuinely fully typed: for ``Orders(Table)`` declaring
``var a: Column[Orders, "a", Int32Type]``, the compiler bakes the constant
``batch.columns[0]`` directly into ``t.a``'s generated code — the position
is never computed from a name at runtime.

``Tbl`` is any struct reflectable via ``std.reflection.reflect`` — typically
the enclosing ``Table``-conforming struct itself, a self-referential
(CRTP-style) parameterization: ``Orders``'s own field ``a`` is typed
``Column[Orders, "a", Int32Type]``, referencing ``Orders`` while ``Orders``
is still being defined. This is legal in Mojo because
``reflect[Tbl].field_index[name]()`` only needs ``Tbl``'s field *names*,
available independent of resolving each field's own type — confirmed
against the pinned toolchain (see ``docs/aot-relations-design.md``).

This file's one ``AnyArray`` erasure boundary (``Project``/``Filter``
assembling heterogeneous columns) stays cheap because it's *closed* —
nothing outside this file constructs an ``AnyArray`` of a dtype a given
query doesn't use, so the compiler can prove and prune the rest of
``filter()``'s/kernels' per-dtype branches. Calling into
``marrow.dyn.executor`` (``Planner``, ``*Processor``) or
``marrow.dyn.values.Expr.eval()`` instead reaches a genuinely *open*
dispatcher built to stay ready for dtypes/node-kinds it can't know ahead of
time, and nothing there can be pruned. Measured in ``benchmarks/binary_size/``:
a ``Project``+``Filter`` plan compiles ~33x smaller (stripped) than the same
query on ``marrow.dyn``'s ``AnyRelation``/``Expr``; a hybrid variant
that only fuses the *predicate* but still calls into the executor is the
same size as the fully runtime one.

See ``docs/aot-relations-design.md`` for the full design.

Usage::

    struct Orders(Table):
        var a: Column[Orders, "a", Int32Type]
        var b: StringColumn[Orders, "b"]

        def __init__(out self):
            self.a = {}
            self.b = {}

    var t = Orders()
    var result = t.a.execute(batch)  # batch.columns[0], baked in at compile time
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
# Table — marker trait for a comptime-declared table schema
# ---------------------------------------------------------------------------


trait Table:
    """Marker for a struct whose fields are ``Column[Self, name, T]`` /
    ``StringColumn[Self, name]`` nodes, giving named, typed, compile-time-
    positioned column access (``t.a``, ``t.b``, ...).
    """

    pass


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
# TypedRelation — trait for fully-typed relational plan nodes
# ---------------------------------------------------------------------------


trait TypedRelation(ImplicitlyDeletable, Movable):
    """Trait for nodes in the fully-typed relational layer (``Project``,
    ``Filter``, ...). Mirrors ``marrow.dyn.relations``'s ``Relation``
    trait but for the comptime-typed plan tree — ``execute(batch)`` runs the
    whole plan against a source batch and returns the result directly, with
    no processor/pull-based pipeline (every node here is a single pass).
    """

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        ...


# ---------------------------------------------------------------------------
# Column — named typed column reference, compile-time-resolved position
# ---------------------------------------------------------------------------


struct Column[Tbl: AnyType, name: StringLiteral, T: dt.NumericType](
    NumericValue, Named
):
    """Named typed column reference whose position is resolved at compile
    time by reflecting ``name`` on ``Tbl``.

    Zero runtime fields — execution (``core[W]``) is otherwise identical to
    ``values.Column[T]``.
    """

    comptime OutType = Self.T
    comptime NativeType = Self.T.native
    comptime index = reflect[Self.Tbl].field_index[Self.name]()

    def __init__(out self):
        pass

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return (
            batch.columns[Self.index]
            .as_primitive[Self.T]()
            .values()
            .load[W](idx)
        )

    def field_name(self) -> String:
        return String(t"{Self.name}")

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Col[{Self.name}]")


# ---------------------------------------------------------------------------
# StringColumn — named typed string column reference, compile-time position
# ---------------------------------------------------------------------------


struct StringColumn[Tbl: AnyType, name: StringLiteral](StringValue, Named):
    """Named typed string column reference whose position is resolved at
    compile time by reflecting ``name`` on ``Tbl``. Mirrors
    ``Column[Tbl, name, T]`` for the string path.
    """

    comptime index = reflect[Self.Tbl].field_index[Self.name]()

    def __init__(out self):
        pass

    def resolve(self, batch: RecordBatch) -> StringArray:
        return batch.columns[Self.index].as_string().copy()

    def execute(self, batch: RecordBatch) raises -> StringArray:
        return self.resolve(batch)

    def field_name(self) -> String:
        return String(t"{Self.name}")

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"StrCol[{Self.name}]")


# ---------------------------------------------------------------------------
# Project — variadic, fully-typed projection over named expression nodes
# ---------------------------------------------------------------------------


def _numeric_col_to_any[
    E: NumericValue
](val: E, batch: RecordBatch) raises -> AnyArray:
    """Execute a NumericValue node and erase the result to AnyArray.

    Routed through a separately-instantiated generic function (rather than
    calling ``val.execute(batch).to_any()`` directly in the ``comptime for``
    body) for the same reason ``schema.mojo``'s ``_construct_default`` is —
    a value whose type comes from indexing a parameter pack only exposes its
    trait-bound methods when the call goes through its own generic
    instantiation.
    """
    return val.execute(batch).to_any()


def _string_col_to_any[
    E: StringValue
](val: E, batch: RecordBatch) raises -> AnyArray:
    """Execute a StringValue node and erase the result to AnyArray."""
    return val.execute(batch).to_any()


def _named_field_name[N: Named](val: N) -> String:
    """Return a Named node's compile-time field name as a String."""
    return val.field_name()


struct Project[*Es: Value](TypedRelation):
    """Fully-typed projection: evaluates a fixed, heterogeneous list of named
    expression nodes and assembles the results into a ``RecordBatch``.

    Each ``Es[i]`` executes as its own fully-monomorphized, fused kernel
    (``NumericValue`` or ``StringValue``, each with independent SIMD fusion).
    The *only* dynamic step is collecting the heterogeneous per-column
    results into ``List[AnyArray]`` / ``RecordBatch`` — inherently
    heterogeneous and O(#columns) — this is the one deliberate erasure
    boundary (see ``docs/aot-relations-design.md``).

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
            comptime E = Self.Es[i]
            ref e = self.exprs[i]
            comptime if conforms_to(E, NumericValue):
                cols.append(_numeric_col_to_any[E](e, batch))
            elif conforms_to(E, StringValue):
                cols.append(_string_col_to_any[E](e, batch))
            else:
                comptime assert (
                    False
                ), "Project: expression must be NumericValue or StringValue"

            comptime assert conforms_to(
                E, Named
            ), "Project: expression must be Named (a bare Column/StringColumn)"
            fields.append(
                Field(_named_field_name[E](e), cols[len(cols) - 1].dtype())
            )

        return RecordBatch(Schema(fields=fields^), cols^)

    def filter[P: BoolValue](var self, var predicate: P) -> Filter[Self, P]:
        """Wrap this projection in a row filter, returning a new plan node.
        """
        return Filter(self^, predicate^)


# ---------------------------------------------------------------------------
# Filter — row filter over a fully-typed relation, by a fused predicate
# ---------------------------------------------------------------------------


struct Filter[Input: TypedRelation, Pred: BoolValue](TypedRelation):
    """Filter — apply a boolean predicate to a typed relation's rows.

    ``predicate`` is evaluated against the *original* input ``batch`` passed
    to ``execute`` (not against ``input``'s projected output) — the
    predicate's ``Column`` nodes are typed against the source ``Table``, so
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


def execute[T: TypedRelation](plan: T, batch: RecordBatch) raises -> RecordBatch:
    """Execute a fully-typed relational plan against a batch.

    Equivalent to calling ``plan.execute(batch)`` directly — provided so the
    ``marrow.aot`` / ``marrow.dyn`` packages read the same at the call site
    (``execute(plan)`` on the ``dyn`` side takes an already-bound
    ``AnyRelation``; here ``batch`` is passed alongside since the typed plan
    itself carries no data).
    """
    return plan.execute(batch)
