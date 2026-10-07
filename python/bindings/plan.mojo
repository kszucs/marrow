# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

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
from marrow.expr.optimizer import AllRules
from marrow.expr.physical import Pipeline
from marrow.expr.runtime.aggregates import RuntimeAggregate
from marrow.expr.runtime.values import RuntimeValue, column as _column
from expressions import (
    bool_list as _bool_list,
    boxed as _boxed,
    boxed_list as _boxed_list,
    unwrap as _unwrap_expr,
    unwrap_agg as _unwrap_agg,
)
from marrow.kernels.join import JoinKind
from marrow.io import DynSource
from marrow.parquet import ParquetFile
from marrow.expr.builders import scan_json as _scan_json
from marrow.expr.builders import scan_iceberg as _scan_iceberg
from marrow.io.uri import StorageOptions
from marrow.json import open_json
from marrow.datasets import DataFiles, HubDataset, load_dataset as _load_dataset
from marrow.schema import Schema
from marrow.expr.sql import Catalog, sql as _sql
from marrow.tabular import RecordBatch


# ---------------------------------------------------------------------------
# Marshalling — the one place Python sequences become Mojo expression lists
# ---------------------------------------------------------------------------


def _unboxed(obj: PythonObject) raises -> RuntimeValue:
    """One expression at its **concrete** type, not behind `DynValue`.

    `DynRelation.filter` has two overloads and they are not equivalent:
    `filter[V: Value]` captures the type and gives the `Filter` node its
    `constant` and `conjuncts`, which is what `EliminateFilter` and
    `SplitConjunction` read. The erased one cannot answer either -- both are
    decisions a box cannot make -- though it does still prune, since `mask` is
    a slot on `DynValue`.

    The type was never actually lost at this boundary. `Expr` is a one-field
    box holding a `RuntimeValue`, and `downcast_value_ptr` recovers it at that
    type -- so all that was needed was to stop boxing it on the way past."""
    var builtins = Python.import_module("builtins")
    if Bool(py=builtins.isinstance(obj, builtins.str)):
        return _column(String(py=obj))
    return _unwrap_expr(obj)


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
        return ParquetFile(DynSource.open(path)).schema()
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
    append-or-replace rule, why replacement happens in place, and where the
    window values among `values` go."""
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

    Reaches `DynRelation.filter[V: Value & Prunable]` -- the overload that
    keeps the predicate's concrete type.

    Pruning alone no longer needs it: `mask` is a slot on `DynValue`, so a
    boxed predicate prunes as well as a typed one. What only the concrete type
    can answer is `constant_bool` and `conjuncts`, so it is `EliminateFilter`
    and `SplitConjunction` that this buys. See `_unboxed`."""
    return _wrap(_plan(py_self).filter(_unboxed(predicate)))


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


def _plan_distinct(py_self: PythonObject) raises -> PythonObject:
    return _wrap(_plan(py_self).distinct())


def _plan_union_all(
    py_self: PythonObject, other: PythonObject
) raises -> PythonObject:
    return _wrap(_plan(py_self).union_all(_plan(other)))


def _plan_intersect(
    py_self: PythonObject, other: PythonObject, all: PythonObject
) raises -> PythonObject:
    if Bool(py=all):
        return _wrap(_plan(py_self).intersect_all(_plan(other)))
    else:
        return _wrap(_plan(py_self).intersect(_plan(other)))


def _plan_except(
    py_self: PythonObject, other: PythonObject, all: PythonObject
) raises -> PythonObject:
    if Bool(py=all):
        return _wrap(_plan(py_self).except_all(_plan(other)))
    else:
        return _wrap(_plan(py_self).except_(_plan(other)))


def _plan_optimize(py_self: PythonObject) raises -> PythonObject:
    """The plan the rewriter would run — fifteen rules plus column pruning.

    `optimize` takes its rule set as a **comptime** parameter, which fixes it
    at *this module's* compile time rather than the caller's; `AllRules` is
    the only set a Python caller can want, since choosing rules at run time is
    what the comptime parameter exists to avoid. `execute()` alone optimizes
    nothing, so this is opt-in on both sides.

    The result is an ordinary plan: print it, diff it against the input, keep
    composing it, or run it."""
    return _wrap(_plan(py_self).optimize[AllRules]())


def _plan_batches(
    py_self: PythonObject, num_threads: PythonObject
) raises -> PythonObject:
    """The result as its natural batches, rather than concatenated into one.

    `execute()` calls `Pipeline.collect`, which drains the chain and concatenates
    everything into a single `StructArray`. That is the wrong shape for a
    multi-row-group scan: the batch boundaries the engine already produced are
    thrown away and then paid for again in one large allocation. `drain` is
    resumable and answers one batch at a time, so this walks it instead.

    The list is still materialised here. Handing back a live iterator would
    mean registering a Python type holding the `Pipeline`, and an `Operator` is
    `Movable` but not `Copyable` — so that is a separate change, not a
    parameter to this one."""
    var rel = _plan(py_self)
    var pipeline = rel.to_operator(ExecContext.parallel(Int(py=num_threads)))
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    while True:
        var datum = pipeline.drain()
        if not datum:
            break
        _ = out.append(
            RecordBatch.from_struct_array(
                datum.value().struct_array()
            ).to_python_object()
        )
    return out


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


def sql_plan(
    query: PythonObject, names: PythonObject, plans: PythonObject
) raises -> PythonObject:
    """A plan parsed from a SQL string against named tables, each a `Plan`.

    A third leaf constructor beside `in_memory_table` and `parquet_scan`, and
    it belongs with them: what it answers is an ordinary `Plan`, so everything
    downstream — `filter`, `aggregate`, `execute`, the optimizer — applies to a
    parsed query exactly as it does to a built one.

    The catalogue arrives as two parallel sequences rather than a dict because
    `Dict` marshalling would have to decide an iteration order that the Mojo
    side then has to keep; the Python wrapper splits the dict once.
    """
    var catalog = Catalog()
    for i in range(len(names)):
        catalog.add(String(py=names[i]), _plan(plans[i]))
    return _wrap(_sql(String(py=query), catalog^))


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


def json_scan(path: PythonObject, schema: PythonObject) raises -> PythonObject:
    """A plan leaf reading newline-delimited JSON.

    ``schema`` doubles as the projection, as for ``parquet_scan``. Passing
    ``None`` infers it from the file's first block, as
    ``marrow.json.open_json`` does: a key first seen later is skipped, and a
    later value its column's type cannot hold is an error at execution.
    """
    var p = String(py=path)
    var builtins = Python.import_module("builtins")
    var sch: Schema
    if schema.__is__(builtins.None):
        sch = open_json(p).schema.copy()
    else:
        sch = Schema(py=schema)
    return _wrap(_scan_json(p^, sch^))


def load_dataset(
    path: PythonObject,
    name: PythonObject,
    split: PythonObject,
    data_files: PythonObject,
    revision: PythonObject,
) raises -> PythonObject:
    """A plan reading one split of a dataset; see `marrow.datasets`.

    ``data_files`` is ``None`` or a dict of split name to a list of
    patterns; the Python wrapper normalises every other spelling.
    """
    var builtins = Python.import_module("builtins")
    var p = String(py=path)
    var s = String(py=split)
    var r = String(py=revision)
    if data_files.__is__(builtins.None):
        return _wrap(_load_dataset(p, String(py=name), s, r))
    var files = DataFiles()
    for key in data_files:
        var patterns = List[String]()
        for pattern in data_files[key]:
            patterns.append(String(py=pattern))
        files.add(String(py=key), patterns^)
    return _wrap(_load_dataset(p, files, s, r))


def hub_config_names(
    path: PythonObject, revision: PythonObject
) raises -> PythonObject:
    """The config names of a Hub dataset."""
    var hub = HubDataset.fetch(String(py=path), String(py=revision))
    var out = Python.list()
    for ref c in hub.config_names():
        out.append(PythonObject(c))
    return out


def hub_split_names(
    path: PythonObject, name: PythonObject, revision: PythonObject
) raises -> PythonObject:
    """The split names of one config of a Hub dataset."""
    var hub = HubDataset.fetch(String(py=path), String(py=revision))
    var out = Python.list()
    for ref s in hub.config(String(py=name)).data_files.splits():
        out.append(PythonObject(s))
    return out


def iceberg_scan(
    path: PythonObject, snapshot_id: PythonObject, options: PythonObject
) raises -> PythonObject:
    """A plan leaf reading an Iceberg table: its metadata file or a local
    table directory, at `snapshot_id` (`None` for the current snapshot), with
    `options` (a `dict[str, str]`) as its storage options."""
    var builtins = Python.import_module("builtins")
    var snapshot: Optional[Int] = None
    if not snapshot_id.__is__(builtins.None):
        snapshot = Int(py=snapshot_id)
    var kv = Dict[String, String]()
    for item in options.items():
        kv[String(py=item[0])] = String(py=item[1])
    return _wrap(
        _scan_iceberg(String(py=path), snapshot, StorageOptions(kv^))
    )


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
        .def_method[_plan_distinct]("distinct")
        .def_method[_plan_union_all]("union_all")
        .def_method[_plan_intersect]("intersect")
        .def_method[_plan_except]("except_")
        .def_method[_plan_optimize]("optimize")
        .def_method[_plan_batches]("batches")
        .def_method[_plan_str]("__str__")
        .def_method[_plan_repr]("__repr__")
    )

    mb.def_function[in_memory_table]("in_memory_table")
    mb.def_function[parquet_scan]("parquet_scan")
    mb.def_function[json_scan]("json_scan")
    mb.def_function[load_dataset]("load_dataset")
    mb.def_function[hub_config_names]("hub_config_names")
    mb.def_function[hub_split_names]("hub_split_names")
    mb.def_function[iceberg_scan]("iceberg_scan")
    mb.def_function[sql_plan]("sql_plan")
