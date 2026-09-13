"""Python bindings for the runtime expression lane.

Exposes `marrow.expr.runtime.values.RuntimeValue` as the Python type ``Expr``
and `marrow.expr.runtime.aggregates.RuntimeAggregate` as ``Agg``.

**Only the runtime lane is bindable.** A comptime node's operands are bound on
a family trait and its output dtype is a comptime type, so `NumericColumn[Int64Type]`
and `NumericColumn[Float64Type]` are different Mojo types and there is no single one a
Python object could hold. That is not a gap in the bindings — it is the lane's
defining property, and the reason `marrow/expr/runtime/` exists.

**One entry point, not one per verb.** `RuntimeValue.evaluate` dispatches on a
`String` tag, and 78 of its verbs are constructed by a free function whose
whole body is `RuntimeValue(tag, kids)`. Registering a method apiece restated
that table in a second place, and the two drifted: 27 verbs — the entire SQL
string surface, half the temporal one — existed in Mojo and were unreachable
from Python for no reason anyone had decided. So the binding is `expr_call`,
and `marrow/expr/runtime/values.mojo` owns the one list of what may be called.

The verbs whose construction does real work keep their own entry points, and
they are the ones a table cannot express: `and`/`or`/`not` constant-fold, which
`PropagateEmpty` depends on; `coalesce` and `case_when` are n-ary; `cast`,
`isin`, `like`, `ilike` and `date_trunc` carry typed payloads, and `date_trunc`
parses its unit at construction.

**The two boxes.** `add_type[T]` installs a default `tp_repr` that calls
`repr(value)`, i.e. `Writable.write_repr_to`, which has a **reflection-based
default that walks every field at comptime**. `RuntimeValue` is recursive
(`List[ArcPointer[Self]]`) and declares only `write_to`; `RuntimeAggregate`
holds one. CLAUDE.md records the hazard: a recursive `Writable` that overrides
only `write_to` inherits a `write_repr_to` whose walk is a monomorphization
cycle. So this module owns two one-field boxes that override `write_repr_to`
and thereby skip the reflection.

**Named methods, not operators.** ``Expr`` exposes no dunders. Two reasons, in
order:

1. The project rule -- the Mojo binding stays minimal and strict, the sugar
   lives in pure Python. ``marrow.expr.Column`` is the user-facing type and it
   owns the dunders.
2. It would not work anyway. `PythonTypeBuilder.bind` installs exactly four
   slots -- `tp_new`, `tp_init`, `tp_dealloc`, `tp_repr` -- and `def_method`
   fills the type's ``tp_dict``, not a CPython slot. So an ``__add__``
   registered here would never fire for ``+``, and an ``__eq__`` would never
   fire for ``==``.

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
from marrow.expr.builders import (
    cume_dist as _cume_dist,
    dense_rank as _dense_rank,
    ntile as _ntile,
    percent_rank as _percent_rank,
    rank as _rank,
    row_number as _row_number,
)
from marrow.expr.logical import DynValue, WindowExpr
from marrow.expr.runtime.aggregates import RuntimeAggregate
from marrow.expr.runtime.values import (
    RuntimeValue,
    and_ as _and,
    binary_verbs as _binary_verbs,
    call as _call,
    case_when as _case_when,
    cast as _cast,
    coalesce as _coalesce,
    column as _column,
    date_trunc as _date_trunc,
    if_else as _if_else,
    ilike as _ilike,
    isin as _isin,
    like as _like,
    literal as _literal,
    not_ as _not,
    or_ as _or,
    ternary_verbs as _ternary_verbs,
    unary_verbs as _unary_verbs,
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


struct Window(Copyable, Movable, Writable):
    """The Python type ``Window`` — a `WindowExpr` under an explicit
    `write_repr_to`.

    Boxed for the same reason `Plan` is: `WindowExpr` holds its function as an
    `Optional[fn]` slot -- which is what keeps an unnamed window function out
    of an AOT binary -- and a derived `repr` cannot see through a function
    pointer."""

    var value: WindowExpr

    @implicit
    def __init__(out self, var value: WindowExpr):
        self.value = value^

    def write_to[W: Writer](self, mut writer: W):
        self.value.write_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<marrow.Window: ", self.value, ">")


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


def unwrap_window(py: PythonObject) raises -> WindowExpr:
    """The `WindowExpr` inside a Python ``Window``."""
    return py.downcast_value_ptr[Window]()[].value.copy()


def wrap_window(var value: WindowExpr) raises -> PythonObject:
    """A Python ``Window`` holding `value`."""
    var box = Window(value^)
    return PythonObject(alloc=box^)


def boxed(obj: PythonObject) raises -> DynValue:
    """One expression: a bound ``Expr``, or a ``str`` naming a column.

    Lives here rather than in `plan.mojo` because both modules need it and
    `plan.mojo` already imports this one -- the "a bare string means
    `col(name)`" convention should have exactly one authority."""
    var builtins = Python.import_module("builtins")
    if Bool(py=builtins.isinstance(obj, builtins.str)):
        return DynValue(_column(String(py=obj)))
    return DynValue(unwrap(obj))


def boxed_list(obj: PythonObject) raises -> List[DynValue]:
    """A Python sequence of expressions / column names."""
    var out = List[DynValue]()
    for i in range(Int(py=obj.__len__())):
        out.append(boxed(obj[i]))
    return out^


def bool_list(obj: PythonObject) raises -> List[Bool]:
    var out = List[Bool]()
    for i in range(Int(py=obj.__len__())):
        out.append(Bool(py=obj[i]))
    return out^


def py_str_list(names: List[String]) raises -> PythonObject:
    """A Mojo ``List[String]`` as a Python list."""
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    for ref name in names:
        _ = out.append(PythonObject(name.copy()))
    return out


def _rows(obj: PythonObject) raises -> Optional[Tuple[Int, Int]]:
    """``None``, or a ``(preceding, following)`` pair for an explicit frame."""
    var builtins = Python.import_module("builtins")
    if obj.__is__(builtins.None):
        return None
    if Int(py=obj.__len__()) != 2:
        raise Error("over: rows= expects a (preceding, following) pair")
    return Tuple(Int(py=obj[0]), Int(py=obj[1]))


def _operands(args: PythonObject) raises -> List[RuntimeValue]:
    """A Python sequence of ``Expr`` as `RuntimeValue`s.

    Strict: every element must already be an ``Expr``. Coercing a Python
    scalar is `Column`'s job — it knows the `lit()` rules and this does not."""
    var n = Int(py=args.__len__())
    var out = List[RuntimeValue](capacity=n)
    for i in range(n):
        out.append(unwrap(args[i]))
    return out^


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def expr_call(tag: PythonObject, args: PythonObject) raises -> PythonObject:
    """Build the node `tag` names. Unknown verbs and wrong arities raise here,
    in `values.call`, rather than on the first morsel that evaluates them."""
    return wrap_expr(_call(String(py=tag), _operands(args)))


def expr_verbs() raises -> PythonObject:
    """``{verb: arity}`` for everything `expr_call` accepts.

    The Python layer generates its methods from this rather than restating the
    list, which is the whole point: a verb added to `values.mojo` is reachable
    from Python without touching either this file or `marrow/expr.py`."""
    var builtins = Python.import_module("builtins")
    var out = builtins.dict()
    for ref name in _unary_verbs():
        out[PythonObject(name.copy())] = PythonObject(1)
    for ref name in _binary_verbs():
        out[PythonObject(name.copy())] = PythonObject(2)
    for ref name in _ternary_verbs():
        out[PythonObject(name.copy())] = PythonObject(3)
    return out


def agg_verbs() raises -> PythonObject:
    """The aggregate vocabulary, for the same reason."""
    return py_str_list(RuntimeAggregate.vocabulary())


# ---------------------------------------------------------------------------
# Constructors that do work — the ones a table cannot express
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


def expr_and(left: PythonObject, right: PythonObject) raises -> PythonObject:
    """Kleene AND, folded at construction — `Filter(FALSE)` is what lets
    `PropagateEmpty` collapse a subtree, so the fold is load-bearing."""
    return wrap_expr(_and(unwrap(left), unwrap(right)))


def expr_or(left: PythonObject, right: PythonObject) raises -> PythonObject:
    """Kleene OR, folded at construction."""
    return wrap_expr(_or(unwrap(left), unwrap(right)))


def expr_not(value: PythonObject) raises -> PythonObject:
    """Kleene NOT, folded at construction."""
    return wrap_expr(_not(unwrap(value)))


def expr_cast(
    value: PythonObject, to: PythonObject, safe: PythonObject
) raises -> PythonObject:
    return wrap_expr(_cast(unwrap(value), DynType(py=to), Bool(py=safe)))


def expr_isin(
    value: PythonObject, value_set: PythonObject
) raises -> PythonObject:
    return wrap_expr(_isin(unwrap(value), DynArray(py=value_set)))


def expr_like(
    value: PythonObject, pattern: PythonObject
) raises -> PythonObject:
    return wrap_expr(_like(unwrap(value), String(py=pattern)))


def expr_ilike(
    value: PythonObject, pattern: PythonObject
) raises -> PythonObject:
    return wrap_expr(_ilike(unwrap(value), String(py=pattern)))


def expr_date_trunc(
    value: PythonObject, unit: PythonObject
) raises -> PythonObject:
    """The unit is parsed here, so a bad spelling fails when the plan is built
    rather than on the row that first evaluates it."""
    return wrap_expr(_date_trunc(unwrap(value), String(py=unit)))


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
    return wrap_expr(_coalesce(_operands(values)))


def expr_case_when(
    conditions: PythonObject, values: PythonObject, else_: PythonObject
) raises -> PythonObject:
    """Multi-branch ``CASE WHEN``. ``else_`` may be ``None``."""
    var builtins = Python.import_module("builtins")
    var otherwise = Optional[RuntimeValue](None)
    if not else_.__is__(builtins.None):
        otherwise = unwrap(else_)
    return wrap_expr(
        _case_when(_operands(conditions), _operands(values), otherwise^)
    )


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


def expr_aggregate(
    value: PythonObject, func: PythonObject
) raises -> PythonObject:
    """Aggregate by name — the one entry point every reduction goes through.

    `RuntimeAggregate.__init__` validates the name against its own vocabulary,
    so an unknown aggregate cannot be built from here and the check lives in
    exactly one place. The twelve reductions used to be twelve registered
    methods restating that vocabulary; `agg_verbs` hands Python the list
    instead."""
    return wrap_agg(RuntimeAggregate(unwrap(value), String(py=func)))


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------
#
# The ranking verbs read no column, so they are module functions in both lanes.
# Everything else is a method on what it reads: `lag`/`lead`/`first_value`/
# `last_value`/`nth_value` on an expression, and `over` on an aggregate --
# `Value.over` raises unless its receiver aggregates, because a per-row value
# has nothing to do with a frame.


def window_row_number() raises -> PythonObject:
    return wrap_window(_row_number())


def window_rank() raises -> PythonObject:
    return wrap_window(_rank())


def window_dense_rank() raises -> PythonObject:
    return wrap_window(_dense_rank())


def window_percent_rank() raises -> PythonObject:
    return wrap_window(_percent_rank())


def window_cume_dist() raises -> PythonObject:
    return wrap_window(_cume_dist())


def window_ntile(buckets: PythonObject) raises -> PythonObject:
    return wrap_window(_ntile(Int(py=buckets)))


def _expr_lag(
    py_self: PythonObject, offset: PythonObject
) raises -> PythonObject:
    return wrap_window(unwrap(py_self).lag(Int(py=offset)))


def _expr_lead(
    py_self: PythonObject, offset: PythonObject
) raises -> PythonObject:
    return wrap_window(unwrap(py_self).lead(Int(py=offset)))


def _expr_first_value(py_self: PythonObject) raises -> PythonObject:
    return wrap_window(unwrap(py_self).first_value())


def _expr_last_value(py_self: PythonObject) raises -> PythonObject:
    return wrap_window(unwrap(py_self).last_value())


def _expr_nth_value(
    py_self: PythonObject, n: PythonObject
) raises -> PythonObject:
    return wrap_window(unwrap(py_self).nth_value(Int(py=n)))


def _agg_over(
    py_self: PythonObject,
    partition_by: PythonObject,
    order_by: PythonObject,
    ascending: PythonObject,
    nulls_first: PythonObject,
    rows: PythonObject,
) raises -> PythonObject:
    """`SUM(x) OVER (...)` — the aggregate evaluated over each frame."""
    return wrap_window(
        unwrap_agg(py_self).over(
            boxed_list(partition_by),
            boxed_list(order_by),
            bool_list(ascending),
            Bool(py=nulls_first),
            _rows(rows),
        )
    )


def _window_over(
    py_self: PythonObject,
    partition_by: PythonObject,
    order_by: PythonObject,
    ascending: PythonObject,
    nulls_first: PythonObject,
    rows: PythonObject,
) raises -> PythonObject:
    """The window this function runs in. Returns a copy with it replaced."""
    return wrap_window(
        unwrap_window(py_self).over(
            boxed_list(partition_by),
            boxed_list(order_by),
            bool_list(ascending),
            Bool(py=nulls_first),
            _rows(rows),
        )
    )


def _window_referenced_columns(py_self: PythonObject) raises -> PythonObject:
    return py_str_list(unwrap_window(py_self).columns())


def _window_str(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(String(unwrap_window(py_self)))


def _window_repr(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(repr(py_self.downcast_value_ptr[Window]()[]))


# ---------------------------------------------------------------------------
# Evaluation and analysis — what is genuinely a method on a node
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


def _expr_tag(py_self: PythonObject) raises -> PythonObject:
    """The node's discriminant — what tells a literal from a column."""
    return PythonObject(unwrap(py_self).tag())


def _expr_name(py_self: PythonObject) raises -> PythonObject:
    """This expression's column name, or ``""`` if it is not a bare column."""
    return PythonObject(unwrap(py_self).name())


def _expr_referenced_columns(py_self: PythonObject) raises -> PythonObject:
    return py_str_list(unwrap(py_self).columns())


def _expr_str(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(String(unwrap(py_self)))


def _expr_repr(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(repr(py_self.downcast_value_ptr[Expr]()[]))


def _agg_alias(
    py_self: PythonObject, name: PythonObject
) raises -> PythonObject:
    return wrap_agg(unwrap_agg(py_self).alias(String(py=name)))


def _agg_name(py_self: PythonObject) raises -> PythonObject:
    """The output column name: the alias if one was set, else the function."""
    return PythonObject(unwrap_agg(py_self).name())


def _agg_referenced_columns(py_self: PythonObject) raises -> PythonObject:
    return py_str_list(unwrap_agg(py_self).columns())


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
    _ = (
        expr_py.def_method[_expr_execute]("execute")
        .def_method[_expr_str]("render")
        .def_method[_expr_name]("name")
        .def_method[_expr_tag]("tag")
        .def_method[_expr_lag]("lag")
        .def_method[_expr_lead]("lead")
        .def_method[_expr_first_value]("first_value")
        .def_method[_expr_last_value]("last_value")
        .def_method[_expr_nth_value]("nth_value")
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
        .def_method[_agg_over]("over")
    )

    # Registered last, and with no earlier `ref` still live: `add_type`
    # reallocates the module builder's type list.
    _ = (
        mb.add_type[Window]("Window")
        .def_method[_window_over]("over")
        .def_method[_window_referenced_columns]("referenced_columns")
        .def_method[_window_str]("render")
        .def_method[_window_str]("__str__")
        .def_method[_window_repr]("__repr__")
    )

    mb.def_function[window_row_number]("window_row_number")
    mb.def_function[window_rank]("window_rank")
    mb.def_function[window_dense_rank]("window_dense_rank")
    mb.def_function[window_percent_rank]("window_percent_rank")
    mb.def_function[window_cume_dist]("window_cume_dist")
    mb.def_function[window_ntile]("window_ntile")
    mb.def_function[expr_call]("expr_call")
    mb.def_function[expr_verbs]("expr_verbs")
    mb.def_function[agg_verbs]("agg_verbs")
    mb.def_function[expr_column]("expr_column")
    mb.def_function[expr_literal]("expr_literal")
    mb.def_function[expr_and]("expr_and")
    mb.def_function[expr_or]("expr_or")
    mb.def_function[expr_not]("expr_not")
    mb.def_function[expr_cast]("expr_cast")
    mb.def_function[expr_isin]("expr_isin")
    mb.def_function[expr_like]("expr_like")
    mb.def_function[expr_ilike]("expr_ilike")
    mb.def_function[expr_date_trunc]("expr_date_trunc")
    mb.def_function[expr_if_else]("expr_if_else")
    mb.def_function[expr_coalesce]("expr_coalesce")
    mb.def_function[expr_case_when]("expr_case_when")
    mb.def_function[expr_count_star]("expr_count_star")
    mb.def_function[expr_aggregate]("expr_aggregate")
