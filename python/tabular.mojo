"""Python bindings for RecordBatch and Table.

Exposes RecordBatch and Table to Python with APIs matching PyArrow.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html
- https://arrow.apache.org/docs/python/generated/pyarrow.Table.html
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from pontoneer import TypeProtocolBuilder, RichCompareOps, NotImplementedError
from std.collections import OwnedKwargsDict
from marrow.tabular import RecordBatch, Table
from marrow.schema import Schema
from marrow.arrays import AnyArray, ChunkedArray
from marrow.dtypes import Field
from std.memory import ArcPointer
from marrow.c_data import CArrowSchema, CArrowArray, CArrowArrayStream
from marrow.kernels.join import hash_join
from helpers import pymethod, def_display


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_pydict(schema: Schema, columns: List[AnyArray]) raises -> PythonObject:
    """Convert schema + columns to a Python dict mapping names to value lists.
    """
    var builtins = Python.import_module("builtins")
    var result = builtins.dict()
    for i in range(len(columns)):
        var col_obj = columns[i].copy().to_python_object()
        var col_len = Int(col_obj.__len__())
        var values = builtins.list()
        for j in range(col_len):
            values.append(col_obj[j])
        result[schema.fields[i].name.to_python_object()] = values
    return result


def _to_pylist(schema: Schema, columns: List[AnyArray]) raises -> PythonObject:
    """Convert schema + columns to a Python list of row dicts."""
    var builtins = Python.import_module("builtins")
    var n_rows = columns[0].length() if len(columns) > 0 else 0
    var n_cols = len(columns)
    var col_objs = List[PythonObject]()
    var col_names = schema.names()
    for i in range(n_cols):
        col_objs.append(columns[i].copy().to_python_object())
    var result = builtins.list()
    for j in range(n_rows):
        var row = builtins.dict()
        for i in range(n_cols):
            row[col_names[i].to_python_object()] = col_objs[i][j]
        result.append(row)
    return result


def _export_c_array(
    schema: Schema, columns: List[AnyArray]
) raises -> PythonObject:
    """Export schema + columns as Arrow C Data Interface capsule pair."""
    var schema_cap = CArrowSchema.from_schema(schema.fields).to_pycapsule()
    var cols = List[AnyArray]()
    for col in columns:
        cols.append(col.copy())
    var struct_arr: AnyArray = RecordBatch(
        schema=schema, columns=cols^
    ).to_struct_array()
    var array_cap = CArrowArray.from_array(struct_arr).to_pycapsule()
    return Python.tuple(schema_cap, array_cap)


def _build_from_dict(data: PythonObject) raises -> RecordBatch:
    """Build a RecordBatch from a Python dict of {name: array}."""
    var fields = List[Field]()
    var columns = List[AnyArray]()
    for key in data:
        var name = String(py=key)
        var arr = AnyArray(py=data[key])
        fields.append(Field(name=name, dtype=arr.dtype()))
        columns.append(arr^)
    return RecordBatch(schema=Schema(fields=fields^), columns=columns^)


def _build_from_arrays(
    data: PythonObject, names_obj: PythonObject
) raises -> RecordBatch:
    """Build a RecordBatch from a list of arrays + names."""
    var fields = List[Field]()
    var columns = List[AnyArray]()
    var i = 0
    for arr_obj in data:
        var arr = AnyArray(py=arr_obj)
        var name = String(py=names_obj[i])
        fields.append(Field(name=name, dtype=arr.dtype()))
        columns.append(arr^)
        i += 1
    return RecordBatch(schema=Schema(fields=fields^), columns=columns^)


# ---------------------------------------------------------------------------
# RecordBatch: methods that need custom Python ↔ Mojo dispatch
# ---------------------------------------------------------------------------


def _record_batch_schema(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    return ptr[].schema.to_python_object()


def _record_batch_columns(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for i in range(len(ptr[].columns)):
        result.append(ptr[].columns[i].copy().to_python_object())
    return result


def _record_batch_shape(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    return Python.tuple(ptr[].num_rows(), ptr[].num_columns())


def _record_batch_column(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin], key: PythonObject
) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    if builtins.isinstance(key, builtins.int):
        return ptr[].columns[Int(py=key)].copy().to_python_object()
    else:
        var name = String(py=key)
        var idx = ptr[].schema.get_field_index(name)
        if idx == -1:
            raise Error("Column '{}' not found.".format(name))
        return ptr[].columns[idx].copy().to_python_object()


def _record_batch_slice(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin],
    offset: PythonObject,
    length: PythonObject,
) raises -> PythonObject:
    return ptr[].slice(Int(py=offset), Int(py=length)).to_python_object()


def _record_batch_equals(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin], other: PythonObject
) raises -> PythonObject:
    return PythonObject(ptr[] == other.downcast_value_ptr[RecordBatch]()[])


def _record_batch_select(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin], columns: PythonObject
) raises -> PythonObject:
    var n = Int(columns.__len__())
    var builtins = Python.import_module("builtins")
    if n > 0 and builtins.isinstance(columns[0], builtins.int):
        var indices = List[Int]()
        for i in range(n):
            indices.append(Int(py=columns[i]))
        return ptr[].select(indices).to_python_object()
    else:
        var names = List[String]()
        for i in range(n):
            names.append(String(py=columns[i]))
        return ptr[].select(names).to_python_object()


def _record_batch_to_pydict(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    return _to_pydict(ptr[].schema, ptr[].columns)


def _record_batch_to_pylist(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    return _to_pylist(ptr[].schema, ptr[].columns)


def _record_batch_arrow_c_array(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin],
    requested_schema: PythonObject,
) raises -> PythonObject:
    return _export_c_array(ptr[].schema, ptr[].columns)


def _record_batch_arrow_c_schema(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    return CArrowSchema.from_schema(ptr[].schema.fields).to_pycapsule()


def _record_batch_rich_compare(
    first: RecordBatch, second: PythonObject, op: Int
) raises -> Bool:
    if op == RichCompareOps.Py_EQ:
        return first == second.downcast_value_ptr[RecordBatch]()[]
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# RecordBatch constructor
# ---------------------------------------------------------------------------


def record_batch(
    data: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    """Create a RecordBatch from a dict, list+names, or Arrow protocol object.
    """
    # Try converting directly (handles marrow objects and Arrow protocol).
    try:
        return RecordBatch(py=data).to_python_object()
    except:
        pass

    var builtins = Python.import_module("builtins")
    if builtins.isinstance(data, builtins.dict):
        return _build_from_dict(data).to_python_object()

    if opt := kwargs.find("names"):
        return _build_from_arrays(data, opt.value()).to_python_object()

    raise Error(
        "record_batch: expected a dict, or a list of arrays with names= kwarg,"
        " or an object with __arrow_c_record_batch__"
    )


# ---------------------------------------------------------------------------
# Table: methods that need custom Python ↔ Mojo dispatch
# ---------------------------------------------------------------------------


def _table_schema(
    ptr: UnsafePointer[Table, MutAnyOrigin]
) raises -> PythonObject:
    return ptr[].schema.to_python_object()


def _table_columns(
    ptr: UnsafePointer[Table, MutAnyOrigin]
) raises -> PythonObject:
    var rb = ptr[].combine_chunks()
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for i in range(len(rb.columns)):
        result.append(rb.columns[i].copy().to_python_object())
    return result


def _table_shape(
    ptr: UnsafePointer[Table, MutAnyOrigin]
) raises -> PythonObject:
    return Python.tuple(ptr[].num_rows(), ptr[].num_columns())


def _table_column(
    ptr: UnsafePointer[Table, MutAnyOrigin], key: PythonObject
) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var rb = ptr[].combine_chunks()
    # TODO: use try/catch python().int()
    if builtins.isinstance(key, builtins.int):
        return rb.columns[Int(py=key)].copy().to_python_object()
    else:
        var name = String(py=key)
        var idx = ptr[].schema.get_field_index(name)
        if idx == -1:
            raise Error("Column '{}' not found.".format(name))
        return rb.columns[idx].copy().to_python_object()


def _table_equals(
    ptr: UnsafePointer[Table, MutAnyOrigin], other: PythonObject
) raises -> PythonObject:
    return PythonObject(ptr[] == other.downcast_value_ptr[Table]()[])


def _table_to_pydict(
    ptr: UnsafePointer[Table, MutAnyOrigin]
) raises -> PythonObject:
    var rb = ptr[].combine_chunks()
    return _to_pydict(rb.schema, rb.columns)


def _table_to_pylist(
    ptr: UnsafePointer[Table, MutAnyOrigin]
) raises -> PythonObject:
    var rb = ptr[].combine_chunks()
    return _to_pylist(rb.schema, rb.columns)


def _table_arrow_c_stream(
    ptr: UnsafePointer[Table, MutAnyOrigin],
    requested_schema: PythonObject,
) raises -> PythonObject:
    var batches = ptr[].to_batches()
    var fields = List(ptr[].schema.fields)
    return CArrowArrayStream.from_batches(fields^, batches^).to_pycapsule()


def _table_arrow_c_schema(
    ptr: UnsafePointer[Table, MutAnyOrigin]
) raises -> PythonObject:
    return CArrowSchema.from_schema(ptr[].schema.fields).to_pycapsule()


def _table_rich_compare(
    first: Table, second: PythonObject, op: Int
) raises -> Bool:
    if op == RichCompareOps.Py_EQ:
        return first == second.downcast_value_ptr[Table]()[]
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# Table constructor
# ---------------------------------------------------------------------------


def table(
    data: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    """Create a Table from a dict, list+names, or Arrow protocol object."""
    # Try converting directly (handles marrow objects and Arrow protocol).
    try:
        return Table(py=data).to_python_object()
    except:
        pass

    var rb: RecordBatch
    var builtins = Python.import_module("builtins")
    if builtins.isinstance(data, builtins.dict):
        rb = _build_from_dict(data)
    elif opt := kwargs.find("names"):
        rb = _build_from_arrays(data, opt.value())
    else:
        raise Error(
            "table: expected a dict, or a list of arrays with names= kwarg,"
            " or an object with __arrow_c_stream__"
        )
    var schema = rb.schema
    var batch_list = List[RecordBatch]()
    batch_list.append(rb^)
    return Table.from_batches(schema, batch_list).to_python_object()


def _record_batch_join(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin],
    right: PythonObject,
    keys: PythonObject,
    right_keys: PythonObject,
    join_type: PythonObject,
    num_threads: PythonObject,
) raises -> PythonObject:
    """Join two RecordBatches on key column names.

    Args:
        right: Right-side RecordBatch.
        keys: Python list of left key column names (strings).
        right_keys: Python list of right key column names (strings), or None
            to use the same names as ``keys``.
        join_type: Join kind string ("inner", "left", "right", "full",
            "semi", "anti").  Default: "inner".
        num_threads: Worker count for the partition-parallel path.  ``0``
            (default) picks ``num_physical_cores()``; ``1`` forces the
            serial path; ``>=2`` opts into parallel execution.
    """
    from marrow.expr.relations import (
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
    )

    ref left = ptr[]
    ref right_rb = right.downcast_value_ptr[RecordBatch]()[]

    # Resolve left key indices from column names.
    var n_keys = Int(keys.__len__())
    var left_on = List[Int]()
    for i in range(n_keys):
        var name = String(py=keys[i])
        var idx = left.schema.get_field_index(name)
        if idx == -1:
            raise Error("Left key column '{}' not found.".format(name))
        left_on.append(idx)

    # Resolve right key indices.
    var right_on = List[Int]()
    var use_right_keys = Bool(right_keys is not PythonObject(None))
    for i in range(n_keys):
        if use_right_keys:
            var name = String(py=right_keys[i])
            var idx = right_rb.schema.get_field_index(name)
            if idx == -1:
                raise Error("Right key column '{}' not found.".format(name))
            right_on.append(idx)
        else:
            var name = String(py=keys[i])
            var idx = right_rb.schema.get_field_index(name)
            if idx == -1:
                raise Error("Right key column '{}' not found.".format(name))
            right_on.append(idx)

    # Parse join type string.
    var kind_str = String(py=join_type)
    var kind = JOIN_INNER
    if kind_str == "left outer" or kind_str == "left":
        kind = JOIN_LEFT
    elif kind_str == "right outer" or kind_str == "right":
        kind = JOIN_RIGHT
    elif kind_str == "full outer" or kind_str == "full":
        kind = JOIN_FULL
    elif kind_str == "left semi" or kind_str == "semi":
        kind = JOIN_SEMI
    elif kind_str == "left anti" or kind_str == "anti":
        kind = JOIN_ANTI

    var left_sa = left.to_struct_array()
    var right_sa = right_rb.to_struct_array()
    var nt = Int(py=num_threads)
    var result_sa = hash_join(
        left_sa,
        right_sa,
        left_on,
        right_on,
        kind,
        num_threads=nt,
    )

    # Build output schema and RecordBatch from StructArray result.
    var out_fields = List[Field]()
    for ref f in result_sa.dtype.as_struct_type().fields:
        out_fields.append(f.copy())
    return RecordBatch(
        schema=Schema(fields=out_fields^),
        columns=result_sa.children.copy(),
    ).to_python_object()


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add RecordBatch, Table types and constructors to the Python module."""
    ref rb_py = mb.add_type[RecordBatch]("RecordBatch")
    _ = (
        rb_py.def_method[_record_batch_schema]("schema")
        .def_method[_record_batch_columns]("columns")
        .def_method[_record_batch_shape]("shape")
        .def_method[pymethod[RecordBatch.num_rows]()]("num_rows")
        .def_method[pymethod[RecordBatch.num_columns]()]("num_columns")
        .def_method[pymethod[RecordBatch.column_names]()]("column_names")
        .def_method[_record_batch_column]("column")
        .def_method[_record_batch_slice]("slice")
        .def_method[_record_batch_equals]("equals")
        .def_method[_record_batch_equals]("__eq__")
        .def_method[_record_batch_select]("select")
        .def_method[pymethod[RecordBatch.rename_columns]()]("rename_columns")
        .def_method[pymethod[RecordBatch.add_column]()]("add_column")
        .def_method[pymethod[RecordBatch.append_column]()]("append_column")
        .def_method[pymethod[RecordBatch.remove_column]()]("remove_column")
        .def_method[pymethod[RecordBatch.set_column]()]("set_column")
        .def_method[_record_batch_to_pydict]("to_pydict")
        .def_method[_record_batch_to_pylist]("to_pylist")
        .def_method[_record_batch_arrow_c_array]("__arrow_c_array__")
        .def_method[_record_batch_arrow_c_array]("__arrow_c_record_batch__")
        .def_method[_record_batch_arrow_c_schema]("__arrow_c_schema__")
        .def_method[_record_batch_join]("join")
    )
    _ = def_display[RecordBatch](rb_py)
    var rb_tp = TypeProtocolBuilder[RecordBatch](rb_py)
    _ = rb_tp.def_richcompare[_record_batch_rich_compare]()

    mb.def_function[record_batch](
        "record_batch",
        docstring=(
            "record_batch(data, /, *, names=None) -> RecordBatch\n--\n\n"
            "Create a RecordBatch from a dict of arrays, a list of arrays with"
            " names=, or any object implementing __arrow_c_record_batch__."
        ),
    )

    # Table
    ref t_py = mb.add_type[Table]("Table")
    _ = (
        t_py.def_method[_table_schema]("schema")
        .def_method[_table_columns]("columns")
        .def_method[_table_shape]("shape")
        .def_method[pymethod[Table.num_rows]()]("num_rows")
        .def_method[pymethod[Table.num_columns]()]("num_columns")
        .def_method[pymethod[Table.column_names]()]("column_names")
        .def_method[_table_column]("column")
        .def_method[pymethod[Table.to_batches]()]("to_batches")
        .def_method[_table_equals]("equals")
        .def_method[_table_equals]("__eq__")
        .def_method[_table_to_pydict]("to_pydict")
        .def_method[_table_to_pylist]("to_pylist")
        .def_method[_table_arrow_c_stream]("__arrow_c_stream__")
        .def_method[_table_arrow_c_schema]("__arrow_c_schema__")
    )
    _ = def_display[Table](t_py)
    var t_tp = TypeProtocolBuilder[Table](t_py)
    _ = t_tp.def_richcompare[_table_rich_compare]()

    mb.def_function[table](
        "table",
        docstring=(
            "table(data, /, *, names=None) -> Table\n--\n\n"
            "Create a Table from a dict of arrays, a list of arrays with"
            " names=, or any object implementing __arrow_c_stream__."
        ),
    )
