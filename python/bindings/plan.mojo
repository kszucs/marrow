"""Python bindings for the relational plan layer.

Exposes `DynRelation` as the Python type ``Plan``: an immutable, inspectable
description of a query that `execute()` opens into a fresh operator tree. The
plan itself is never mutated, so a `Plan` is a reusable template and every verb
returns a new one.

The friendly lazy surface (``marrow.LazyTable``, keyword aggregates,
``order_by`` sugar) lives in pure Python; these entry points stay strict — no
optional arguments, no defaults.

Expression arguments arrive as the bound ``Expr`` / ``Agg`` objects registered
by ``expressions.mojo``. Two marshalling conveniences are deliberate and are
implemented once, in `_boxed` / `_agg` below:

- Anywhere a *column reference* is wanted (select names, sort keys, join keys)
  a plain Python ``str`` is accepted and becomes ``col(name)``. This is the
  ibis spelling (``t.order_by("x")``) and it keeps the common path free of
  expression objects.
- An aggregate may be given as a ``(func, column, out_name)`` triple instead of
  an ``Agg``, which is what the keyword surface ``t.aggregate(total=("sum",
  "amount"))`` marshals into.

**The verbs mirror `DynRelation`'s, argument for argument**, which is why
`sort` takes parallel key/ascending lists and `join` takes column *indices*:
those are the Mojo signatures, and a binding that reshaped them would be a
second API to keep in step with the first. The reshaping — dicts, keywords,
`on=` shorthand, name-to-index resolution — happens once, in Python.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from marrow.execution import ExecContext
from marrow.expr.bindings import Bindings
from marrow.expr.builders import scan as _scan, table as _table
from marrow.expr.logical import DynRelation, DynValue
from marrow.expr.runtime.aggregates import RuntimeAggregate
from marrow.expr.runtime.values import column as _column
from expressions import unwrap as _unwrap_expr, unwrap_agg as _unwrap_agg
from marrow.kernels.join import JoinKind
from marrow.parquet import ParquetFile
from marrow.schema import Schema
from marrow.tabular import RecordBatch


# ---------------------------------------------------------------------------
# Marshalling — the one place Python sequences become Mojo expression lists
# ---------------------------------------------------------------------------


def _boxed(obj: PythonObject) raises -> DynValue:
    """One expression: a bound ``Expr``, or a ``str`` naming a column."""
    var builtins = Python.import_module("builtins")
    if Bool(py=builtins.isinstance(obj, builtins.str)):
        return DynValue(_column(String(py=obj)))
    # `Expr` is a one-field box owned by `expressions.mojo` -- a bare
    # `RuntimeValue` cannot be registered, since deriving `write_repr_to` for
    # it reflects through its own recursion. Cross the box through its
    # accessor.
    return DynValue(_unwrap_expr(obj))


def _boxed_list(obj: PythonObject) raises -> List[DynValue]:
    """A Python sequence of expressions / column names."""
    var out = List[DynValue]()
    for i in range(Int(py=obj.__len__())):
        out.append(_boxed(obj[i]))
    return out^


def _agg(obj: PythonObject) raises -> DynValue:
    """One aggregate: a bound ``Agg``, or a ``(func, column, out_name)`` triple.

    Both end up as a `DynValue`, because `Aggregate` takes its aggregates in
    the same box as `Project` takes its projections: an aggregate *is* a
    `Value` whose `shape` is scalar, not a separate kind of thing the plan
    layer has to carry in its own list type.
    """
    var builtins = Python.import_module("builtins")
    if Bool(py=builtins.isinstance(obj, builtins.tuple)) or Bool(
        py=builtins.isinstance(obj, builtins.list)
    ):
        var func = String(py=obj[0])
        var input = _column(String(py=obj[1]))
        return DynValue(
            RuntimeAggregate(input^, func^).alias(String(py=obj[2]))
        )
    return DynValue(_unwrap_agg(obj))


def _agg_list(obj: PythonObject) raises -> List[DynValue]:
    var out = List[DynValue]()
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


def _int_list(obj: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(Int(py=obj.__len__())):
        out.append(Int(py=obj[i]))
    return out^


def _scan_schema(schema: PythonObject, path: String) raises -> Schema:
    """The scan's schema — the given one, or the file's own when `None`.

    Reading it here costs footer metadata only, no column data, which is what
    lets `marrow.read_parquet(path)` infer without a separate binding, and what
    keeps `Relation` itself free of the filesystem: a plan node is a
    description and must not touch a file to exist, so the read happens at the
    boundary rather than inside `ParquetScan`."""
    var builtins = Python.import_module("builtins")
    if schema.__is__(builtins.None):
        return ParquetFile(path).schema()
    return schema.downcast_value_ptr[Schema]()[].copy()


struct Plan(Copyable, Movable, Writable):
    """The registered Python type — a `DynRelation` under a bindable skin.

    `DynRelation` cannot be handed to `add_type` directly. The binding installs
    a default `tp_repr` that calls `write_repr_to`, `DynRelation` declares only
    `write_to`, and deriving the missing one walks the struct's fields and dies
    on its trampolines:

        constraint failed: Could not derive Writable for DynRelation —
        member field `_virt_write` does not implement Writable

    Erasure behind function pointers is exactly what a derived `repr` cannot
    see through, so every `Dyn*` box has this problem. Wrapping keeps the
    binding concern out of `marrow.expr`, which has no Python dependency.

    Copying is still O(1) — the wrapper adds a move, the plan is shared behind
    an `ArcPointer`."""

    var rel: DynRelation

    @implicit
    def __init__(out self, var rel: DynRelation):
        self.rel = rel^

    def write_to[W: Writer](self, mut writer: W):
        self.rel.write_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
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
        _ = names.append(PythonObject(f.name.copy()))
    return names


def _plan_execute(
    py_self: PythonObject, num_threads: PythonObject
) raises -> PythonObject:
    """Run the plan under an `ExecContext` with the caller's worker budget.

    ``num_threads`` is the eager surface's spelling and the eager surface's
    sentinel set (`RecordBatch.group_by(..., num_threads=0)`): 0 auto, 1
    serial, N forced. Without this argument the call would be `execute()` with
    no context, and `DynRelation.execute`'s `ExecContext.auto()` default would
    decide for every query — which is right for a Mojo caller who can pass a
    context and wrong for a Python one who then has no way to."""
    return (
        _plan(py_self)
        .execute(ExecContext.parallel(Int(py=num_threads)))
        .to_python_object()
    )


def _plan_select(
    py_self: PythonObject, names: PythonObject
) raises -> PythonObject:
    """Project columns by name.

    Calls `DynRelation.select(List[String])`, the overload that exists for
    exactly this call site — the other spelling is `*names: String`, and a Mojo
    variadic cannot be splatted from a runtime list. Routing through `project`
    instead is *not* the same node: `project` probes each expression's dtype
    and builds a fresh `Field`, so a non-nullable column would come out
    nullable and its metadata would be dropped. `select` copies the input field
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
    """Keep rows where `predicate` is true.

    This reaches `DynRelation.filter(DynValue)`, the **erased** overload, so
    the plan filters exactly and reads every row group. The pruning overload
    takes `V: Value & Prunable` and captures the concrete type, which a
    `PythonObject` has already thrown away by the time it gets here — so
    statistics pruning is not reachable from Python today. `RuntimeValue` does
    conform to `Prunable`, so what is missing is a way to carry the unerased
    value across the boundary, not the pruning itself.
    """
    return _wrap(_plan(py_self).filter(_boxed(predicate)))


def _plan_aggregate(
    py_self: PythonObject, keys: PythonObject, aggs: PythonObject
) raises -> PythonObject:
    """`SELECT <keys>, <aggs> ... GROUP BY <keys>`.

    Argument order is `(keys, aggs)` here and `(aggs, keys)` on `DynRelation`.
    That is not drift: the Mojo verb defaults `keys` to empty so a whole-table
    aggregate needs no key list at all, which forces aggregates first; a
    binding has no defaults, so it takes them in the order a reader expects to
    see them in SQL."""
    return _wrap(_plan(py_self).aggregate(_agg_list(aggs), _boxed_list(keys)))


def _plan_sort(
    py_self: PythonObject,
    keys: PythonObject,
    ascending: PythonObject,
    nulls_first: PythonObject,
) raises -> PythonObject:
    return _wrap(
        _plan(py_self).sort_by(
            _boxed_list(keys),
            _bool_list(ascending),
            Bool(py=nulls_first),
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
) raises -> PythonObject:
    """Hash join on column **indices**.

    `DynRelation.join` takes `List[Int]`, not expressions: the join operator
    hashes whole columns of the input schema, so a key is a position in it
    rather than something to evaluate. Resolving a name to an index needs the
    schema, which `Plan.column_names()` already hands to Python, so the lookup
    happens there and this stays a straight forward. ``how`` uses PyArrow's
    spelling; `JoinKind.parse` owns the name-to-kind mapping."""
    return _wrap(
        _plan(py_self).join(
            right.downcast_value_ptr[Plan]()[].rel.copy(),
            _int_list(left_on),
            _int_list(right_on),
            JoinKind.parse(String(py=how)),
        )
    )


def _plan_str(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(String(_plan(py_self)))


def _plan_repr(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(repr(py_self.downcast_value_ptr[Plan]()[]))


# ---------------------------------------------------------------------------
# Leaf constructors
# ---------------------------------------------------------------------------


def in_memory_table(batch: PythonObject) raises -> PythonObject:
    """A plan leaf backed by an in-memory RecordBatch."""
    return _wrap(_table(RecordBatch(py=batch)))


def parquet_scan(
    path: PythonObject, schema: PythonObject
) raises -> PythonObject:
    """A plan leaf reading a Parquet file.

    ``schema`` doubles as the projection — only its columns are read. Passing
    ``None`` reads the full schema out of the footer (metadata only, no column
    data), which is what ``marrow.read_parquet(path)`` does.
    """
    var p = String(py=path)
    var sch = _scan_schema(schema, p)
    return _wrap(_scan(p^, sch^))


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
