"""Python bindings for the relational plan layer.

Exposes `DynRelation` as the Python type ``Plan``: an immutable, inspectable
description of a query that `execute()` opens into a fresh operator tree. The
plan itself is never mutated, so a `Plan` is a reusable template and every verb
returns a new one.

The friendly lazy surface (``marrow.expr.LazyTable``, keyword aggregates,
``order_by`` sugar) lives in pure Python; these entry points stay strict — no
optional arguments, no defaults.

Expression arguments arrive as the bound ``Expr`` / ``Agg`` objects registered
by ``expressions.mojo``. Two marshalling conveniences are deliberate and are
implemented once, in `_boxed` / `_agg` below:

- Anywhere a *column reference* is wanted (select names, sort keys, join keys) a
  plain Python ``str`` is accepted and becomes ``col(name)``. This is the ibis
  spelling (``t.order_by("x")``) and it keeps the common path free of expression
  objects.
- An aggregate may be given as a ``(func, column, out_name)`` triple instead of
  an ``Agg``, which is what the keyword surface ``t.aggregate(total=("sum",
  "amount"))`` marshals into.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from marrow.execution import ExecContext
from marrow.expr.builders import col
from marrow.expr.dynamic import DynValue
from marrow.expr.relations import (
    DynRelation,
    in_memory_table as _in_memory_table,
    parquet_scan as _parquet_scan,
)
from marrow.expr.values import AggExpr, BoxedValue
from expressions import unwrap as _unwrap_expr, unwrap_agg as _unwrap_agg
from marrow.kernels.join import JOIN_ALL, JOIN_ANY, JoinKind
from marrow.parquet import ParquetFile
from marrow.schema import Schema
from marrow.tabular import RecordBatch


# ---------------------------------------------------------------------------
# Marshalling — the one place Python sequences become Mojo expression lists
# ---------------------------------------------------------------------------


def _boxed(obj: PythonObject) raises -> BoxedValue:
    """One expression: a bound ``Expr``, or a ``str`` naming a column."""
    var builtins = Python.import_module("builtins")
    if Bool(py=builtins.isinstance(obj, builtins.str)):
        return BoxedValue(col(String(py=obj)))
    # `Expr` is a one-field box owned by `expressions.mojo` (a bare `DynValue`
    # cannot be registered: deriving `Writable` for it reflects into its
    # function-pointer field). Cross the box through its own accessor.
    return BoxedValue(_unwrap_expr(obj))


def _boxed_list(obj: PythonObject) raises -> List[BoxedValue]:
    """A Python sequence of expressions / column names."""
    var out = List[BoxedValue]()
    for i in range(Int(py=obj.__len__())):
        out.append(_boxed(obj[i]))
    return out^


def _agg(obj: PythonObject) raises -> AggExpr:
    """One aggregate: a bound ``Agg``, or a ``(func, column, out_name)`` triple.
    """
    var builtins = Python.import_module("builtins")
    if Bool(py=builtins.isinstance(obj, builtins.tuple)) or Bool(
        py=builtins.isinstance(obj, builtins.list)
    ):
        var func = String(py=obj[0])
        var input = DynValue.column(String(py=obj[1]))
        return AggExpr(func^, input^).alias(String(py=obj[2]))
    return _unwrap_agg(obj)


def _agg_list(obj: PythonObject) raises -> List[AggExpr]:
    var out = List[AggExpr]()
    for i in range(Int(py=obj.__len__())):
        out.append(_agg(obj[i]))
    return out^


def _string_list(obj: PythonObject) raises -> List[String]:
    var out = List[String]()
    for i in range(Int(py=obj.__len__())):
        out.append(String(py=obj[i]))
    return out^


def _bool_list(obj: PythonObject) raises -> List[Bool]:
    var out = List[Bool]()
    for i in range(Int(py=obj.__len__())):
        out.append(Bool(py=obj[i]))
    return out^


def _strictness(name: String) raises -> UInt8:
    """How many matches a join uses. `JoinKind.parse` owns the *kind* names;
    strictness is still a bare `UInt8`, so its two names are spelled here."""
    if name == "all":
        return JOIN_ALL
    elif name == "any":
        return JOIN_ANY
    else:
        raise Error("join: unknown strictness '", name, "'")


def _scan_schema(schema: PythonObject, path: String) raises -> Schema:
    """The scan's schema — the given one, or the file's own when `None`.

    Reading it here costs footer metadata only, no column data, which is what
    lets `marrow.read_parquet(path)` infer without a separate binding."""
    var builtins = Python.import_module("builtins")
    if schema.__is__(builtins.None):
        return ParquetFile(path).schema()
    return schema.downcast_value_ptr[Schema]()[].copy()


struct Plan(Copyable, Movable, Writable):
    """The registered Python type — a `DynRelation` under a bindable skin.

    `DynRelation` cannot be handed to `add_type` directly. The binding needs a
    `write_repr_to`, `DynRelation` declares only `write_to`, and deriving the
    missing one walks the struct's fields and dies on its trampolines:

        constraint failed: Could not derive Writable for DynRelation —
        member field `_virt_with_predicate` does not implement Writable

    Erasure behind function pointers is exactly what a derived `repr` cannot
    see through, so every `Dyn*` box has this problem. Wrapping keeps the
    binding concern out of `marrow.expr`, which has no Python dependency;
    `DynRelation` growing its own two-line `write_repr_to` would be the smaller
    fix and is the one to make when that file is next open.

    Copying is still O(1) — the wrapper adds a move, the plan is shared."""

    var rel: DynRelation

    @implicit
    def __init__(out self, var rel: DynRelation):
        self.rel = rel^

    def write_to[W: Writer](self, mut writer: W):
        self.rel.write_to(writer)

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("<marrow.Plan: ", self.rel, ">")


def _plan(py_self: PythonObject) raises -> DynRelation:
    return py_self.downcast_value_ptr[Plan]()[].rel.copy()


def _wrap(var rel: DynRelation) raises -> PythonObject:
    """Hand a plan back to Python."""
    return PythonObject(alloc=Plan(rel^))


# ---------------------------------------------------------------------------
# Plan methods
# ---------------------------------------------------------------------------


def _plan_schema(py_self: PythonObject) raises -> PythonObject:
    return _plan(py_self).schema().to_python_object()


def _plan_column_names(py_self: PythonObject) raises -> PythonObject:
    """The output column names.

    The bound `Schema` exposes only `__arrow_c_schema__`, so without this the
    lazy frontend would have to import pyarrow just to read its own column
    names."""
    var builtins = Python.import_module("builtins")
    var names = builtins.list()
    var schema = _plan(py_self).schema()
    for ref f in schema.fields:
        names.append(PythonObject(f.name))
    return names


def _plan_execute(
    py_self: PythonObject, num_threads: PythonObject
) raises -> PythonObject:
    """Run the plan under an `ExecContext` with the caller's worker budget.

    ``num_threads`` is the eager surface's spelling and the eager surface's
    sentinel set (`RecordBatch.group_by(..., num_threads=0)`): 0 auto, 1 serial,
    N forced. Before this argument existed the call was `execute()` with no
    context at all, so `DynRelation.execute`'s `ExecContext()` default made
    every lazy query serial no matter how many cores the machine had — the
    kernels' parallel strategies were unreachable from the query API."""
    return (
        _plan(py_self)
        .execute(ExecContext.parallel(Int(py=num_threads)))
        .to_python_object()
    )


def _plan_select(
    py_self: PythonObject, names: PythonObject
) raises -> PythonObject:
    """Project columns by name.

    Calls `DynRelation.select(List[String])`, the overload added for exactly
    this call site — the other spelling is `*names: String`, and a Mojo
    variadic cannot be splatted from a runtime list. This used to route through
    `project` instead, which is *not* the same node: `project` probes each
    expression's dtype and builds a fresh `Field`, so a non-nullable column came
    out nullable and its metadata was dropped. `select` copies the input field
    whole.
    """
    return _wrap(_plan(py_self).select(_string_list(names)))


def _plan_project(
    py_self: PythonObject, names: PythonObject, values: PythonObject
) raises -> PythonObject:
    return _wrap(
        _plan(py_self).project(_string_list(names), _boxed_list(values))
    )


def _plan_with_columns(
    py_self: PythonObject, names: PythonObject, values: PythonObject
) raises -> PythonObject:
    """Add or replace computed columns, keeping every other column.

    `project`'s usable half — see `DynRelation.with_columns` for the
    append-or-replace rule and why replacement happens in place."""
    return _wrap(
        _plan(py_self).with_columns(_string_list(names), _boxed_list(values))
    )


def _plan_drop(
    py_self: PythonObject, names: PythonObject
) raises -> PythonObject:
    """Remove the named columns, keeping the rest in order."""
    return _wrap(_plan(py_self).drop(_string_list(names)))


def _plan_rename(
    py_self: PythonObject, names: PythonObject, new_names: PythonObject
) raises -> PythonObject:
    """Rename columns — two parallel lists, old then new."""
    return _wrap(
        _plan(py_self).rename(_string_list(names), _string_list(new_names))
    )


def _plan_filter(
    py_self: PythonObject, predicate: PythonObject
) raises -> PythonObject:
    return _wrap(_plan(py_self).filter(_boxed(predicate)))


def _plan_aggregate(
    py_self: PythonObject, keys: PythonObject, aggs: PythonObject
) raises -> PythonObject:
    return _wrap(_plan(py_self).aggregate(_boxed_list(keys), _agg_list(aggs)))


def _plan_sort(
    py_self: PythonObject,
    keys: PythonObject,
    ascending: PythonObject,
    nulls_first: PythonObject,
    stable: PythonObject,
) raises -> PythonObject:
    return _wrap(
        _plan(py_self).sort(
            _boxed_list(keys),
            _bool_list(ascending),
            Bool(py=nulls_first),
            Bool(py=stable),
        )
    )


def _plan_limit(
    py_self: PythonObject, length: PythonObject, offset: PythonObject
) raises -> PythonObject:
    return _wrap(_plan(py_self).limit(Int(py=length), Int(py=offset)))


def _plan_join(
    py_self: PythonObject,
    right: PythonObject,
    left_on: PythonObject,
    right_on: PythonObject,
    how: PythonObject,
    strictness: PythonObject,
) raises -> PythonObject:
    """Hash join. ``how`` uses PyArrow's spelling (`JoinKind.parse` owns the
    name-to-kind mapping); ``strictness`` is "all" or "any"."""
    return _wrap(
        _plan(py_self).join(
            right.downcast_value_ptr[Plan]()[].rel.copy(),
            _boxed_list(left_on),
            _boxed_list(right_on),
            JoinKind.parse(String(py=how)),
            _strictness(String(py=strictness)),
        )
    )


def _plan_str(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(String(_plan(py_self)))


def _plan_repr(py_self: PythonObject) raises -> PythonObject:
    return PythonObject("<marrow.Plan: " + String(_plan(py_self)) + ">")


# ---------------------------------------------------------------------------
# Leaf constructors
# ---------------------------------------------------------------------------


def in_memory_table(
    batch: PythonObject, morsel_size: PythonObject
) raises -> PythonObject:
    """A plan leaf backed by an in-memory RecordBatch."""
    return _wrap(
        _in_memory_table(
            batch.downcast_value_ptr[RecordBatch]()[], Int(py=morsel_size)
        )
    )


def parquet_scan(
    path: PythonObject, schema: PythonObject, morsel_size: PythonObject
) raises -> PythonObject:
    """A plan leaf reading a Parquet file.

    ``schema`` doubles as the projection — only its columns are read. Passing
    ``None`` reads the full schema out of the footer (metadata only, no column
    data), which is what ``marrow.read_parquet(path)`` does.
    """
    var p = String(py=path)
    var sch = _scan_schema(schema, p)
    return _wrap(_parquet_scan(p^, sch^, Int(py=morsel_size)))


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Register the Plan type and the leaf constructors."""
    ref plan_py = mb.add_type[Plan]("Plan")
    _ = (
        plan_py.def_method[_plan_schema]("schema")
        .def_method[_plan_column_names]("column_names")
        .def_method[_plan_execute]("execute")
        .def_method[_plan_select]("select")
        .def_method[_plan_project]("project")
        .def_method[_plan_with_columns]("with_columns")
        .def_method[_plan_drop]("drop")
        .def_method[_plan_rename]("rename")
        .def_method[_plan_filter]("filter")
        .def_method[_plan_aggregate]("aggregate")
        .def_method[_plan_sort]("sort")
        .def_method[_plan_limit]("limit")
        .def_method[_plan_join]("join")
        .def_method[_plan_str]("__str__")
        .def_method[_plan_repr]("__repr__")
    )

    mb.def_function[in_memory_table]("in_memory_table")
    mb.def_function[parquet_scan]("parquet_scan")
