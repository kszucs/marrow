"""Python bindings for the runtime expression lane.

Exposes `marrow.expr.runtime.values.RuntimeValue` as the Python type ``Expr``
and `marrow.expr.runtime.aggregates.RuntimeAggregate` as ``Agg``.

**Only the runtime lane is bindable.** A comptime node's operands are bound on
a family trait and its output dtype is a comptime type, so `Column[Int64Type]`
and `Column[Float64Type]` are different Mojo types and there is no single one a
Python object could hold. That is not a gap in the bindings — it is the lane's
defining property, and the reason `marrow/expr/runtime/` exists.

**The two boxes.** `add_type[T]` installs a default `tp_repr` that calls
`repr(value)`, i.e. `Writable.write_repr_to`, which has a **reflection-based
default that walks every field at comptime**. `RuntimeValue` is recursive
(`List[ArcPointer[Self]]`) and declares only `write_to`; `RuntimeAggregate`
holds one. CLAUDE.md records the hazard: a recursive `Writable` that overrides
only `write_to` inherits a `write_repr_to` whose walk is a monomorphization
cycle. So this module owns two one-field boxes that override `write_repr_to`
and thereby skip the reflection.

*(The earlier bindings needed the same two boxes for a different reason -- the
runtime node then had a `_eval_fn` function-pointer field, which `Writable`
cannot be derived for at all. That field is gone: `evaluate` switches on a tag
because the fn-pointer design was miscompiled. The boxes survive because the
recursion outlived the pointer.)*

**Named methods, not operators.** ``Expr`` exposes ``add`` / ``lt`` / ``and_``
rather than ``__add__`` / ``__lt__`` / ``__and__``. Two reasons, in order:

1. The project rule -- the Mojo binding stays minimal and strict, the sugar
   lives in pure Python. ``marrow._expr.Column`` is the user-facing type and it
   owns the dunders.
2. It would not work anyway. `PythonTypeBuilder.bind` installs exactly four
   slots -- `tp_new`, `tp_init`, `tp_dealloc`, `tp_repr` -- and `def_method`
   fills the type's ``tp_dict``, not a CPython slot. So an ``__add__``
   registered here would never fire for ``+``, and an ``__eq__`` would never
   fire for ``==``.

That second point also decides ``__eq__``'s return type: on an expression it
must answer an ``Expr``, not a ``Bool``, and wiring that at the C level would
make the binding object unusable as a dict key for no gain -- nothing but
``Column`` ever holds one.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.compute.Expression.html
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from marrow.arrays import DynArray
from marrow.dtypes import DynType
from marrow.scalars import DynScalar, Int64Scalar
from marrow.tabular import RecordBatch
from marrow.expr.bindings import Bindings
from marrow.expr.runtime.aggregates import RuntimeAggregate
from marrow.expr.runtime.values import (
    RuntimeValue,
    abs as _abs,
    add as _add,
    and_ as _and,
    array_length as _array_length,
    capitalize as _capitalize,
    case_when as _case_when,
    cast as _cast,
    ceil as _ceil,
    coalesce as _coalesce,
    column as _column,
    contains as _contains,
    date_trunc as _date_trunc,
    day as _day,
    day_of_week as _day_of_week,
    day_of_year as _day_of_year,
    endswith as _endswith,
    eq as _eq,
    exp as _exp,
    fill_null as _fill_null,
    floor as _floor,
    floordiv as _floordiv,
    ge as _ge,
    gt as _gt,
    hour as _hour,
    if_else as _if_else,
    ilike as _ilike,
    is_inf as _is_inf,
    is_nan as _is_nan,
    is_null as _is_null,
    is_valid as _is_valid,
    isin as _isin,
    le as _le,
    length as _length,
    like as _like,
    literal as _literal,
    ln as _ln,
    lower as _lower,
    lstrip as _lstrip,
    lt as _lt,
    minute as _minute,
    mod as _mod,
    month as _month,
    mul as _mul,
    ne as _ne,
    neg as _neg,
    not_ as _not,
    nullif as _nullif,
    or_ as _or,
    pow as _pow,
    quarter as _quarter,
    reverse as _reverse,
    round as _round,
    rstrip as _rstrip,
    second as _second,
    sign as _sign,
    sqrt as _sqrt,
    startswith as _startswith,
    strip as _strip,
    sub as _sub,
    trunc as _trunc,
    truediv as _truediv,
    upper as _upper,
    xor as _xor,
    year as _year,
)


# ---------------------------------------------------------------------------
# The two boxes
# ---------------------------------------------------------------------------


struct Expr(Copyable, Movable, Writable):
    """The Python type ``Expr`` — a `RuntimeValue` under an explicit
    `write_repr_to`.

    The box exists only so `add_type`'s default `tp_repr` stops reflecting over
    a recursive struct; `value` is the whole payload and every method here
    forwards to it."""

    var value: RuntimeValue

    @implicit
    def __init__(out self, var value: RuntimeValue):
        self.value = value^

    def write_to[W: Writer](self, mut writer: W):
        self.value.write_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<marrow.Expr: ", self.value, ">")


struct Agg(Copyable, Movable, Writable):
    """The Python type ``Agg`` — a `RuntimeAggregate` under an explicit
    `write_repr_to`."""

    var value: RuntimeAggregate

    @implicit
    def __init__(out self, var value: RuntimeAggregate):
        self.value = value^

    def write_to[W: Writer](self, mut writer: W):
        self.value.write_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<marrow.Agg: ", self.value, ">")


# ---------------------------------------------------------------------------
# The seam — what another binding module uses to cross the box
# ---------------------------------------------------------------------------


def unwrap(py: PythonObject) raises -> RuntimeValue:
    """The `RuntimeValue` inside a Python ``Expr``."""
    return py.downcast_value_ptr[Expr]()[].value.copy()


def unwrap_agg(py: PythonObject) raises -> RuntimeAggregate:
    """The `RuntimeAggregate` inside a Python ``Agg``."""
    return py.downcast_value_ptr[Agg]()[].value.copy()


def wrap_expr(var value: RuntimeValue) raises -> PythonObject:
    """A Python ``Expr`` holding `value`."""
    var box = Expr(value^)
    return PythonObject(alloc=box^)


def wrap_agg(var value: RuntimeAggregate) raises -> PythonObject:
    """A Python ``Agg`` holding `value`."""
    var box = Agg(value^)
    return PythonObject(alloc=box^)


# ---------------------------------------------------------------------------
# Boxing helpers — the three shapes that cover most of the surface
# ---------------------------------------------------------------------------


def _unary[
    m: def(var RuntimeValue) raises thin -> RuntimeValue,
]() -> def(PythonObject) raises thin -> PythonObject:
    """Wrap ``Expr -> Expr``."""

    def wrapper(py_self: PythonObject) raises -> PythonObject:
        return wrap_expr(m(unwrap(py_self)))

    return wrapper


def _binary[
    m: def(var RuntimeValue, var RuntimeValue) raises thin -> RuntimeValue,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    """Wrap ``(Expr, Expr) -> Expr``.

    Strict: the right operand must already be an ``Expr``. Coercing a Python
    scalar is ``Column``'s job — it knows the ``lit()`` rules and this does
    not."""

    def wrapper(
        py_self: PythonObject, other: PythonObject
    ) raises -> PythonObject:
        return wrap_expr(m(unwrap(py_self), unwrap(other)))

    return wrapper


def _reduce[
    m: def(RuntimeValue) raises thin -> RuntimeAggregate,
]() -> def(PythonObject) raises thin -> PythonObject:
    """Wrap ``Expr -> Agg``."""

    def wrapper(py_self: PythonObject) raises -> PythonObject:
        var ptr = py_self.downcast_value_ptr[Expr]()
        return wrap_agg(m(ptr[].value))

    return wrapper


# ---------------------------------------------------------------------------
# Methods that carry a payload — spelled out, since the payload type varies
# ---------------------------------------------------------------------------


def _expr_cast(
    py_self: PythonObject, to: PythonObject, safe: PythonObject
) raises -> PythonObject:
    return wrap_expr(_cast(unwrap(py_self), DynType(py=to), Bool(py=safe)))


def _expr_isin(
    py_self: PythonObject, value_set: PythonObject
) raises -> PythonObject:
    return wrap_expr(_isin(unwrap(py_self), DynArray(py=value_set)))


def _expr_like(
    py_self: PythonObject, pattern: PythonObject
) raises -> PythonObject:
    return wrap_expr(_like(unwrap(py_self), String(py=pattern)))


def _expr_ilike(
    py_self: PythonObject, pattern: PythonObject
) raises -> PythonObject:
    return wrap_expr(_ilike(unwrap(py_self), String(py=pattern)))


def _expr_date_trunc(
    py_self: PythonObject, unit: PythonObject
) raises -> PythonObject:
    return wrap_expr(_date_trunc(unwrap(py_self), String(py=unit)))


def _expr_aggregate(
    py_self: PythonObject, func: PythonObject
) raises -> PythonObject:
    """Aggregate by name — the one entry point a `(func, column)` pair needs.

    `RuntimeAggregate.__init__` validates the name against its own vocabulary,
    so an unknown aggregate cannot be built from here and the check lives in
    exactly one place."""
    return wrap_agg(RuntimeAggregate(unwrap(py_self), String(py=func)))


# ---------------------------------------------------------------------------
# Evaluation and plan analysis
# ---------------------------------------------------------------------------


def _expr_execute(
    py_self: PythonObject, batch: PythonObject
) raises -> PythonObject:
    """Evaluate this expression over one ``RecordBatch`` — the eager escape
    hatch, and what lets a test check that a tree computes rather than merely
    renders."""
    var b = RecordBatch(py=batch)
    return (
        unwrap(py_self)
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .to_python_object()
    )


def _expr_render(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(String(unwrap(py_self)))


def _expr_name(py_self: PythonObject) raises -> PythonObject:
    """This expression's column name, or ``""`` if it is not a bare column."""
    return PythonObject(unwrap(py_self).name())


def _expr_referenced_columns(py_self: PythonObject) raises -> PythonObject:
    var names = unwrap(py_self).columns()
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    for ref n in names:
        _ = out.append(PythonObject(n.copy()))
    return out


def _expr_str(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(String(unwrap(py_self)))


def _expr_repr(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(repr(py_self.downcast_value_ptr[Expr]()[]))


# ---------------------------------------------------------------------------
# Leaves — the module-level constructors
# ---------------------------------------------------------------------------


def expr_column(name: PythonObject) raises -> PythonObject:
    """``col("a")`` — a column reference whose dtype is found on the batch."""
    return wrap_expr(_column(String(py=name)))


def expr_literal(value: PythonObject) raises -> PythonObject:
    """``lit(3)`` — a constant, given as a **length-1 marrow Array**.

    The array is the conversion, not an implementation detail leaking out:
    `DynScalar` is `ConvertibleToPython` but not `ConvertibleFromPython`, so
    there is no Python-value -> `DynScalar` path, whereas ``array([v], t)`` is
    the tree's one well-tested Python -> Arrow converter, type inference and
    explicit-dtype override included. ``marrow.lit()`` builds the array."""
    var arr = DynArray(py=value)
    if len(arr) != 1:
        raise Error("literal: expected a length-1 array, got length ", len(arr))
    return wrap_expr(_literal(arr[0]))


def expr_if_else(
    cond: PythonObject, then_: PythonObject, else_: PythonObject
) raises -> PythonObject:
    """Element-wise conditional."""
    return wrap_expr(_if_else(unwrap(cond), unwrap(then_), unwrap(else_)))


def expr_coalesce(values: PythonObject) raises -> PythonObject:
    """First non-null across N expressions.

    N-ary rather than a fold of binary nodes, because `CoalesceKernel` is
    already n-ary — folding would materialise one intermediate column per
    extra operand."""
    var out = List[RuntimeValue]()
    for i in range(Int(py=values.__len__())):
        out.append(unwrap(values[i]))
    return wrap_expr(_coalesce(out^))


def expr_case_when(
    conditions: PythonObject, values: PythonObject, else_: PythonObject
) raises -> PythonObject:
    """Multi-branch ``CASE WHEN``. ``else_`` may be ``None``."""
    var conds = List[RuntimeValue]()
    for i in range(Int(py=conditions.__len__())):
        conds.append(unwrap(conditions[i]))
    var vals = List[RuntimeValue]()
    for i in range(Int(py=values.__len__())):
        vals.append(unwrap(values[i]))
    var builtins = Python.import_module("builtins")
    var otherwise = Optional[RuntimeValue](None)
    if not else_.__is__(builtins.None):
        otherwise = unwrap(else_)
    return wrap_expr(_case_when(conds^, vals^, otherwise^))


def expr_count_star() raises -> PythonObject:
    """``COUNT(*)`` — the row count, as an ``Agg`` with no input column.

    A module-level constructor rather than a method, because it is the one
    aggregate that is not *of* an expression: `col("x").count()` counts the
    non-null values of ``x``, and those two differ on every nullable column.

    It needs no new kernel and no new node — `count` counts valid values and a
    literal is valid on every row, so the valid-count of a constant column *is*
    the row count. That is what `builders.count_star` builds in the comptime
    lane, spelled here against `RuntimeValue`.
    """
    return wrap_agg(
        _literal(DynScalar(Int64Scalar(1))).count().alias("count_star")
    )


# ---------------------------------------------------------------------------
# Agg — an aggregate applied to a runtime expression
# ---------------------------------------------------------------------------


def _agg_alias(
    py_self: PythonObject, name: PythonObject
) raises -> PythonObject:
    return wrap_agg(unwrap_agg(py_self).alias(String(py=name)))


def _agg_name(py_self: PythonObject) raises -> PythonObject:
    """The output column name: the alias if one was set, else the function."""
    return PythonObject(unwrap_agg(py_self).name())


def _agg_referenced_columns(py_self: PythonObject) raises -> PythonObject:
    var names = unwrap_agg(py_self).columns()
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    for ref n in names:
        _ = out.append(PythonObject(n.copy()))
    return out


def _agg_str(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(String(unwrap_agg(py_self)))


def _agg_repr(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(repr(py_self.downcast_value_ptr[Agg]()[]))


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Register the ``Expr`` and ``Agg`` Python types."""
    ref expr_py = mb.add_type[Expr]("Expr")

    # arithmetic
    _ = (
        expr_py.def_method[_binary[_add]()]("add")
        .def_method[_binary[_sub]()]("sub")
        .def_method[_binary[_mul]()]("mul")
        .def_method[_binary[_truediv]()]("truediv")
        .def_method[_binary[_floordiv]()]("floordiv")
        .def_method[_binary[_mod]()]("mod")
        .def_method[_binary[_pow]()]("pow")
        .def_method[_unary[_neg]()]("neg")
    )

    # comparison — named, never `__eq__`; see the module docstring.
    _ = (
        expr_py.def_method[_binary[_lt]()]("lt")
        .def_method[_binary[_le]()]("le")
        .def_method[_binary[_gt]()]("gt")
        .def_method[_binary[_ge]()]("ge")
        .def_method[_binary[_eq]()]("eq")
        .def_method[_binary[_ne]()]("ne")
    )

    # boolean
    _ = (
        expr_py.def_method[_binary[_and]()]("and_")
        .def_method[_binary[_or]()]("or_")
        .def_method[_binary[_xor]()]("xor")
        .def_method[_unary[_not]()]("invert")
    )

    # math
    _ = (
        expr_py.def_method[_unary[_abs]()]("abs")
        .def_method[_unary[_sign]()]("sign")
        .def_method[_unary[_floor]()]("floor")
        .def_method[_unary[_ceil]()]("ceil")
        .def_method[_unary[_round]()]("round")
        .def_method[_unary[_trunc]()]("trunc")
        .def_method[_unary[_sqrt]()]("sqrt")
        .def_method[_unary[_exp]()]("exp")
        .def_method[_unary[_ln]()]("ln")
    )

    # string
    _ = (
        expr_py.def_method[_unary[_upper]()]("upper")
        .def_method[_unary[_lower]()]("lower")
        .def_method[_unary[_strip]()]("strip")
        .def_method[_unary[_lstrip]()]("lstrip")
        .def_method[_unary[_rstrip]()]("rstrip")
        .def_method[_unary[_reverse]()]("reverse")
        .def_method[_unary[_capitalize]()]("capitalize")
        .def_method[_unary[_length]()]("length")
        .def_method[_binary[_startswith]()]("startswith")
        .def_method[_binary[_endswith]()]("endswith")
        .def_method[_binary[_contains]()]("contains")
        .def_method[_expr_like]("like")
        .def_method[_expr_ilike]("ilike")
    )

    # temporal
    _ = (
        expr_py.def_method[_unary[_year]()]("year")
        .def_method[_unary[_month]()]("month")
        .def_method[_unary[_day]()]("day")
        .def_method[_unary[_hour]()]("hour")
        .def_method[_unary[_minute]()]("minute")
        .def_method[_unary[_second]()]("second")
        .def_method[_unary[_day_of_week]()]("day_of_week")
        .def_method[_unary[_quarter]()]("quarter")
        .def_method[_unary[_day_of_year]()]("day_of_year")
        .def_method[_expr_date_trunc]("date_trunc")
    )

    # conditional / membership / casting / nested
    _ = (
        expr_py.def_method[_binary[_nullif]()]("nullif")
        .def_method[_binary[_fill_null]()]("fill_null")
        .def_method[_unary[_array_length]()]("array_length")
        .def_method[_expr_isin]("isin")
        .def_method[_expr_cast]("cast")
    )

    # null / value predicates. `is_null` / `is_valid` read the validity bitmap
    # and are never null themselves; `is_nan` / `is_inf` read the values and
    # are null where the input is. Binding all four is what makes ``Expr``
    # closed under the boolean combinators.
    _ = (
        expr_py.def_method[_unary[_is_null]()]("is_null")
        .def_method[_unary[_is_valid]()]("is_valid")
        .def_method[_unary[_is_nan]()]("is_nan")
        .def_method[_unary[_is_inf]()]("is_inf")
    )

    # aggregations
    _ = (
        expr_py.def_method[_reduce[RuntimeValue.sum]()]("sum")
        .def_method[_reduce[RuntimeValue.mean]()]("mean")
        .def_method[_reduce[RuntimeValue.product]()]("product")
        .def_method[_reduce[RuntimeValue.min]()]("min")
        .def_method[_reduce[RuntimeValue.max]()]("max")
        .def_method[_reduce[RuntimeValue.count]()]("count")
        .def_method[_reduce[RuntimeValue.count_distinct]()]("count_distinct")
        .def_method[_reduce[RuntimeValue.approx_count_distinct]()](
            "approx_count_distinct"
        )
        .def_method[_reduce[RuntimeValue.variance]()]("variance")
        .def_method[_reduce[RuntimeValue.var_samp]()]("var_samp")
        .def_method[_reduce[RuntimeValue.stddev]()]("stddev")
        .def_method[_reduce[RuntimeValue.stddev_samp]()]("stddev_samp")
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
        .def_method[_agg_name]("name")
        .def_method[_agg_referenced_columns]("referenced_columns")
        .def_method[_agg_str]("render")
        .def_method[_agg_str]("__str__")
        .def_method[_agg_repr]("__repr__")
    )

    mb.def_function[expr_column]("expr_column")
    mb.def_function[expr_literal]("expr_literal")
    mb.def_function[expr_if_else]("expr_if_else")
    mb.def_function[expr_coalesce]("expr_coalesce")
    mb.def_function[expr_case_when]("expr_case_when")
    mb.def_function[expr_count_star]("expr_count_star")
