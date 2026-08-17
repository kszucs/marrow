"""Python bindings for the runtime expression lane.

Exposes `marrow.expr.dynamic.DynValue` as the Python type ``Expr`` and
``DynAgg`` as ``Agg``. Only the *runtime* lane is bindable: the AOT lane's
nodes are parameterised on comptime dtypes, so there is no single Mojo type a
Python object could hold.

**`DynValue` and `DynAgg` cannot be registered directly.** `add_type[T]`
installs a default `tp_repr` that calls `repr(value)`, which is
`Writable.write_repr_to` — and *that* has a reflection-based default requiring
every field to be `Writable`. `DynValue._eval_fn` is a function pointer:

    format/__init__.mojo:287: constraint failed: Could not derive Writable for
    DynValue - member field `_eval_fn` does not implement Writable

`DynAgg` fails the same way through its `input` field. So this module owns two
one-field boxes, `Expr` and `Agg`, which override `write_repr_to` and thereby
skip the reflection. They are the Python types; `unwrap` / `unwrap_agg` /
`wrap_expr` / `wrap_agg` are the seam any other binding module (the plan
bindings) goes through to reach the `DynValue` inside.

The alternative is a two-line `write_repr_to` on `DynValue`/`DynAgg` in
`marrow/expr/dynamic.mojo`, which would delete both boxes. Worth doing once
that file is free to edit.

**Named methods, not operators.** ``Expr`` exposes ``add`` / ``lt`` / ``and_``
rather than ``__add__`` / ``__lt__`` / ``__and__``, mirroring the stdlib
``operator`` module. Two reasons, in order:

1. The project rule — the Mojo binding stays minimal and strict, the sugar
   lives in pure Python. ``marrow._expr_column.Column`` is the user-facing
   type and it owns the dunders.
2. ``__eq__`` on an expression must return an ``Expr``, not a ``Bool``. Wiring
   that at the C level would make the *binding* object unusable as a dict key
   or in an ``in``-test, for no gain — nothing but ``Column`` ever holds one.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.compute.Expression.html
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from marrow.arrays import DynArray
from marrow.dtypes import DynType
from marrow.expr import DynAgg, DynValue
from marrow.tabular import RecordBatch


# ---------------------------------------------------------------------------
# The two boxes
# ---------------------------------------------------------------------------


struct Expr(Copyable, Movable, Writable):
    """The Python type ``Expr`` — a `DynValue` under an explicit
    `write_repr_to`.

    The box exists only so `add_type`'s default `tp_repr` stops deriving;
    `value` is the whole payload and every method here forwards to it."""

    var value: DynValue

    @implicit
    def __init__(out self, var value: DynValue):
        self.value = value^

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.value.render())

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<marrow.Expr: ", self.value.render(), ">")


struct Agg(Copyable, Movable, Writable):
    """The Python type ``Agg`` — a `DynAgg` under an explicit
    `write_repr_to`."""

    var value: DynAgg

    @implicit
    def __init__(out self, var value: DynAgg):
        self.value = value^

    def write_to[W: Writer](self, mut writer: W):
        self.value.write_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<marrow.Agg: ", String(self.value), ">")


# ---------------------------------------------------------------------------
# The seam — what another binding module uses to cross the box
# ---------------------------------------------------------------------------


def unwrap(py: PythonObject) raises -> DynValue:
    """The `DynValue` inside a Python ``Expr``."""
    return py.downcast_value_ptr[Expr]()[].value.copy()


def unwrap_agg(py: PythonObject) raises -> DynAgg:
    """The `DynAgg` inside a Python ``Agg``."""
    return py.downcast_value_ptr[Agg]()[].value.copy()


def wrap_expr(var value: DynValue) raises -> PythonObject:
    """A Python ``Expr`` holding `value`."""
    var box = Expr(value^)
    return PythonObject(alloc=box^)


def wrap_agg(var value: DynAgg) raises -> PythonObject:
    """A Python ``Agg`` holding `value`."""
    var box = Agg(value^)
    return PythonObject(alloc=box^)


# ---------------------------------------------------------------------------
# Boxing helpers — the three shapes that cover most of `DynValue`'s surface
# ---------------------------------------------------------------------------


def _unary[
    m: def(DynValue) raises thin -> DynValue,
]() -> def(PythonObject) raises thin -> PythonObject:
    """Wrap ``Expr -> Expr``."""

    def wrapper(py_self: PythonObject) raises -> PythonObject:
        var ptr = py_self.downcast_value_ptr[Expr]()
        return wrap_expr(m(ptr[].value))

    return wrapper


def _binary[
    m: def(DynValue, DynValue) raises thin -> DynValue,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    """Wrap ``(Expr, Expr) -> Expr``.

    Strict: the right operand must already be an ``Expr``. Coercing a Python
    scalar is ``Column``'s job — it knows the ``lit()`` rules and this does
    not."""

    def wrapper(
        py_self: PythonObject, other: PythonObject
    ) raises -> PythonObject:
        var a = py_self.downcast_value_ptr[Expr]()
        var b = other.downcast_value_ptr[Expr]()
        return wrap_expr(m(a[].value, b[].value))

    return wrapper


def _reduce[
    m: def(DynValue) raises thin -> DynAgg,
]() -> def(PythonObject) raises thin -> PythonObject:
    """Wrap ``Expr -> Agg``."""

    def wrapper(py_self: PythonObject) raises -> PythonObject:
        var ptr = py_self.downcast_value_ptr[Expr]()
        return wrap_agg(m(ptr[].value))

    return wrapper


# ---------------------------------------------------------------------------
# Methods that carry a payload — spelled out, since the payload type varies
# ---------------------------------------------------------------------------


def _expr_cast(py_self: PythonObject, to: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Expr]()
    return wrap_expr(ptr[].value.cast(DynType(py=to)))


def _expr_isin(
    py_self: PythonObject, value_set: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Expr]()
    return wrap_expr(ptr[].value.isin(DynArray(py=value_set)))


def _expr_like(
    py_self: PythonObject, pattern: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Expr]()
    return wrap_expr(ptr[].value.like(String(py=pattern)))


def _expr_ilike(
    py_self: PythonObject, pattern: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Expr]()
    return wrap_expr(ptr[].value.ilike(String(py=pattern)))


def _expr_date_trunc(
    py_self: PythonObject, unit: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Expr]()
    return wrap_expr(ptr[].value.date_trunc(String(py=unit)))


def _expr_aggregate(
    py_self: PythonObject, func: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Expr]()
    return wrap_agg(ptr[].value.aggregate(String(py=func)))


# ---------------------------------------------------------------------------
# Evaluation and plan analysis
# ---------------------------------------------------------------------------


def _expr_execute(
    py_self: PythonObject, batch: PythonObject
) raises -> PythonObject:
    """Evaluate this expression over one ``RecordBatch`` — the eager escape
    hatch, and what lets a test check that a tree computes rather than merely
    renders."""
    var ptr = py_self.downcast_value_ptr[Expr]()
    return ptr[].value.execute(RecordBatch(py=batch)).to_python_object()


def _expr_render(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Expr]()
    return PythonObject(ptr[].value.render())


def _expr_name(py_self: PythonObject) raises -> PythonObject:
    """This expression's column name, or ``""`` if it is not a bare column."""
    var ptr = py_self.downcast_value_ptr[Expr]()
    return PythonObject(ptr[].value.name())


def _expr_referenced_columns(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Expr]()
    var names = ptr[].value.referenced_columns()
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    for i in range(len(names)):
        _ = out.append(PythonObject(names[i].copy()))
    return out


def _expr_str(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Expr]()
    return PythonObject(ptr[].value.render())


def _expr_repr(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Expr]()
    return PythonObject(repr(ptr[]))


# ---------------------------------------------------------------------------
# Leaves — the module-level constructors
# ---------------------------------------------------------------------------


def expr_column(name: PythonObject) raises -> PythonObject:
    """``col("a")`` — a column reference whose dtype is found on the batch."""
    return wrap_expr(DynValue.column(String(py=name)))


def expr_literal(value: PythonObject) raises -> PythonObject:
    """``lit(3)`` — a constant, given as a **length-1 marrow Array**.

    The array is the conversion, not an implementation detail leaking out:
    `DynScalar` is `ConvertibleToPython` but not `ConvertibleFromPython`, so
    there is no Python-value → `DynScalar` path, whereas ``array([v], t)`` is
    the tree's one well-tested Python → Arrow converter, type inference and
    explicit-dtype override included. ``marrow.lit()`` builds the array."""
    var arr = DynArray(py=value)
    if len(arr) != 1:
        raise Error("literal: expected a length-1 array, got length ", len(arr))
    return wrap_expr(DynValue.literal(arr[0]))


def expr_if_else(
    cond: PythonObject, then_: PythonObject, else_: PythonObject
) raises -> PythonObject:
    """Element-wise conditional."""
    var c = cond.downcast_value_ptr[Expr]()
    var t = then_.downcast_value_ptr[Expr]()
    var e = else_.downcast_value_ptr[Expr]()
    return wrap_expr(DynValue.if_else(c[].value, t[].value, e[].value))


# ---------------------------------------------------------------------------
# Agg — an aggregate applied to a runtime expression
# ---------------------------------------------------------------------------


def _agg_alias(
    py_self: PythonObject, name: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Agg]()
    return wrap_agg(ptr[].value.alias(String(py=name)))


def _agg_function(py_self: PythonObject) raises -> PythonObject:
    """The aggregate function's name — ``"sum"``, ``"mean"``, …"""
    var ptr = py_self.downcast_value_ptr[Agg]()
    return PythonObject(ptr[].value.func.copy())


def _agg_name(py_self: PythonObject) raises -> PythonObject:
    """The output column name: the alias if one was set, else the function."""
    var ptr = py_self.downcast_value_ptr[Agg]()
    if ptr[].value.out_name:
        return PythonObject(ptr[].value.out_name.copy())
    return PythonObject(ptr[].value.func.copy())


def _agg_input(py_self: PythonObject) raises -> PythonObject:
    """The expression being aggregated."""
    var ptr = py_self.downcast_value_ptr[Agg]()
    return wrap_expr(ptr[].value.input.copy())


def _agg_str(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Agg]()
    return PythonObject(String(ptr[].value))


def _agg_repr(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Agg]()
    return PythonObject(repr(ptr[]))


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Register the ``Expr`` and ``Agg`` Python types."""
    ref expr_py = mb.add_type[Expr]("Expr")

    # arithmetic
    _ = (
        expr_py.def_method[_binary[DynValue.__add__]()]("add")
        .def_method[_binary[DynValue.__sub__]()]("sub")
        .def_method[_binary[DynValue.__mul__]()]("mul")
        .def_method[_binary[DynValue.__truediv__]()]("truediv")
        .def_method[_binary[DynValue.__mod__]()]("mod")
        .def_method[_binary[DynValue.__floordiv__]()]("floordiv")
        .def_method[_binary[DynValue.__pow__]()]("pow")
        .def_method[_unary[DynValue.__neg__]()]("neg")
    )

    # comparison — named, never `__eq__`; see the module docstring.
    _ = (
        expr_py.def_method[_binary[DynValue.__lt__]()]("lt")
        .def_method[_binary[DynValue.__le__]()]("le")
        .def_method[_binary[DynValue.__gt__]()]("gt")
        .def_method[_binary[DynValue.__ge__]()]("ge")
        .def_method[_binary[DynValue.__eq__]()]("eq")
        .def_method[_binary[DynValue.__ne__]()]("ne")
    )

    # boolean
    _ = (
        expr_py.def_method[_binary[DynValue.__and__]()]("and_")
        .def_method[_binary[DynValue.__or__]()]("or_")
        .def_method[_binary[DynValue.__xor__]()]("xor")
        .def_method[_unary[DynValue.__invert__]()]("invert")
    )

    # math
    _ = (
        expr_py.def_method[_unary[DynValue.abs]()]("abs")
        .def_method[_unary[DynValue.sign]()]("sign")
        .def_method[_unary[DynValue.floor]()]("floor")
        .def_method[_unary[DynValue.ceil]()]("ceil")
        .def_method[_unary[DynValue.round]()]("round")
        .def_method[_unary[DynValue.sqrt]()]("sqrt")
        .def_method[_unary[DynValue.exp]()]("exp")
        .def_method[_unary[DynValue.ln]()]("ln")
    )

    # string
    _ = (
        expr_py.def_method[_unary[DynValue.upper]()]("upper")
        .def_method[_unary[DynValue.lower]()]("lower")
        .def_method[_unary[DynValue.strip]()]("strip")
        .def_method[_unary[DynValue.lstrip]()]("lstrip")
        .def_method[_unary[DynValue.rstrip]()]("rstrip")
        .def_method[_unary[DynValue.reverse]()]("reverse")
        .def_method[_unary[DynValue.capitalize]()]("capitalize")
        .def_method[_unary[DynValue.length]()]("length")
        .def_method[_binary[DynValue.startswith]()]("startswith")
        .def_method[_binary[DynValue.endswith]()]("endswith")
        .def_method[_binary[DynValue.contains]()]("contains")
        .def_method[_expr_like]("like")
        .def_method[_expr_ilike]("ilike")
    )

    # temporal
    _ = (
        expr_py.def_method[_unary[DynValue.year]()]("year")
        .def_method[_unary[DynValue.month]()]("month")
        .def_method[_unary[DynValue.day]()]("day")
        .def_method[_unary[DynValue.hour]()]("hour")
        .def_method[_unary[DynValue.minute]()]("minute")
        .def_method[_unary[DynValue.second]()]("second")
        .def_method[_unary[DynValue.day_of_week]()]("day_of_week")
        .def_method[_unary[DynValue.quarter]()]("quarter")
        .def_method[_unary[DynValue.day_of_year]()]("day_of_year")
        .def_method[_expr_date_trunc]("date_trunc")
    )

    # conditional / membership / casting
    _ = (
        expr_py.def_method[_binary[DynValue.coalesce]()]("coalesce")
        .def_method[_binary[DynValue.nullif]()]("nullif")
        .def_method[_expr_isin]("isin")
        .def_method[_expr_cast]("cast")
    )
    # TODO(alpha): bind is_null/is_valid/is_nan/fill_null once merged — they do
    # not exist on `DynValue` on this branch (a `NullPredicate` node exists but
    # is reachable only from the fused lane).

    # aggregations
    _ = (
        expr_py.def_method[_reduce[DynValue.sum]()]("sum")
        .def_method[_reduce[DynValue.mean]()]("mean")
        .def_method[_reduce[DynValue.product]()]("product")
        .def_method[_reduce[DynValue.min]()]("min")
        .def_method[_reduce[DynValue.max]()]("max")
        .def_method[_reduce[DynValue.count]()]("count")
        .def_method[_expr_aggregate]("aggregate")
    )

    # evaluation, analysis, representation
    _ = (
        expr_py.def_method[_expr_execute]("execute")
        .def_method[_expr_render]("render")
        .def_method[_expr_name]("name")
        .def_method[_expr_referenced_columns]("referenced_columns")
        .def_method[_expr_str]("__str__")
        .def_method[_expr_repr]("__repr__")
    )

    ref agg_py = mb.add_type[Agg]("Agg")
    _ = (
        agg_py.def_method[_agg_alias]("alias")
        .def_method[_agg_function]("function")
        .def_method[_agg_name]("name")
        .def_method[_agg_input]("input")
        .def_method[_agg_str]("render")
        .def_method[_agg_str]("__str__")
        .def_method[_agg_repr]("__repr__")
    )

    mb.def_function[expr_column]("expr_column")
    mb.def_function[expr_literal]("expr_literal")
    mb.def_function[expr_if_else]("expr_if_else")
