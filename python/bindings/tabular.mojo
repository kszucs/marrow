"""Python bindings for RecordBatch and Table.

Exposes RecordBatch and Table to Python with APIs matching PyArrow.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html
- https://arrow.apache.org/docs/python/generated/pyarrow.Table.html
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from marrow.tabular import RecordBatch, Table
from marrow.execution import ExecContext
from marrow.schema import Schema
from marrow.arrays import DynArray, ChunkedArray
from marrow.dtypes import Field
from std.memory import ArcPointer, UnsafePointer
from std.builtin.type_aliases import MutAnyOrigin
from marrow.c_data import CArrowSchema, CArrowArray, CArrowArrayStream
from marrow.arrays import Int32Array
from helpers import pymethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_pydict(schema: Schema, columns: List[DynArray]) raises -> PythonObject:
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
        result[PythonObject(schema.fields[i].name)] = values
    return result


def _to_pylist(schema: Schema, columns: List[DynArray]) raises -> PythonObject:
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
            row[PythonObject(col_names[i])] = col_objs[i][j]
        result.append(row)
    return result


def _export_c_array(
    schema: Schema, columns: List[DynArray]
) raises -> PythonObject:
    """Export schema + columns as Arrow C Data Interface capsule pair."""
    var schema_cap = CArrowSchema.from_schema(schema).to_pycapsule()
    var cols = List[DynArray]()
    for col in columns:
        cols.append(col.copy())
    var struct_arr: DynArray = RecordBatch(
        schema=schema, columns=cols^
    ).to_struct_array()
    var array_cap = CArrowArray.from_array(struct_arr).to_pycapsule()
    return Python.tuple(schema_cap, array_cap)


def _build_from_dict(data: PythonObject) raises -> RecordBatch:
    """Build a RecordBatch from a Python dict of {name: array}."""
    var fields = List[Field]()
    var columns = List[DynArray]()
    for key in data:
        var name = String(py=key)
        var arr = DynArray(py=data[key])
        fields.append(Field(name=name, dtype=arr.dtype()))
        columns.append(arr^)
    return RecordBatch(schema=Schema(fields=fields^), columns=columns^)


def _build_from_arrays(
    data: PythonObject, names_obj: PythonObject
) raises -> RecordBatch:
    """Build a RecordBatch from a list of arrays + names."""
    var fields = List[Field]()
    var columns = List[DynArray]()
    var i = 0
    for arr_obj in data:
        var arr = DynArray(py=arr_obj)
        var name = String(py=names_obj[i])
        fields.append(Field(name=name, dtype=arr.dtype()))
        columns.append(arr^)
        i += 1
    return RecordBatch(schema=Schema(fields=fields^), columns=columns^)


def _build_from_arrays_with_schema(
    data: PythonObject, schema_obj: PythonObject
) raises -> RecordBatch:
    """Build a RecordBatch from a list of arrays + explicit schema."""
    var schema = Schema(py=schema_obj)
    var columns = List[DynArray]()
    for arr_obj in data:
        columns.append(DynArray(py=arr_obj))
    return RecordBatch(schema=schema^, columns=columns^)


# ---------------------------------------------------------------------------
# RecordBatch: methods that need custom Python ↔ Mojo dispatch
# ---------------------------------------------------------------------------


def _record_batch_schema(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return ptr[].schema.to_python_object()


def _record_batch_columns(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for i in range(len(ptr[].columns)):
        result.append(ptr[].columns[i].copy().to_python_object())
    return result


def _record_batch_column_names(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for name in ptr[].column_names():
        result.append(PythonObject(name))
    return result


def _record_batch_shape(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return Python.tuple(ptr[].num_rows(), ptr[].num_columns())


def _record_batch_column(
    py_self: PythonObject, key: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
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
    py_self: PythonObject,
    offset: PythonObject,
    length: PythonObject,
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return ptr[].slice(Int(py=offset), Int(py=length)).to_python_object()


def _record_batch_equals(
    py_self: PythonObject, other: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return PythonObject(ptr[] == other.downcast_value_ptr[RecordBatch]()[])


def _record_batch_select(
    py_self: PythonObject, columns: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
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


def _record_batch_to_pydict(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return _to_pydict(ptr[].schema, ptr[].columns)


def _record_batch_to_pylist(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return _to_pylist(ptr[].schema, ptr[].columns)


def _record_batch_arrow_c_array(
    py_self: PythonObject,
    requested_schema: PythonObject,
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return _export_c_array(ptr[].schema, ptr[].columns)


def _record_batch_arrow_c_schema(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return CArrowSchema.from_schema(ptr[].schema).to_pycapsule()


# def _record_batch_rich_compare(
#     first: RecordBatch, second: PythonObject, op: Int
# ) raises -> Bool:
#     if op == RichCompareOps.Py_EQ:
#         return first == second.downcast_value_ptr[RecordBatch]()[]
#     raise NotImplementedError()


# ---------------------------------------------------------------------------
# RecordBatch constructor
# ---------------------------------------------------------------------------


def record_batch(
    data: PythonObject, schema: PythonObject, names: PythonObject
) raises -> PythonObject:
    """Create a RecordBatch from a dict, list+names, or Arrow protocol object.
    """
    try:
        return RecordBatch(py=data).to_python_object()
    except:
        pass

    var builtins = Python.import_module("builtins")
    if builtins.isinstance(data, builtins.dict):
        return _build_from_dict(data).to_python_object()

    if not schema.__is__(builtins.None):
        return _build_from_arrays_with_schema(data, schema).to_python_object()

    if not names.__is__(builtins.None):
        return _build_from_arrays(data, names).to_python_object()

    raise Error(
        "record_batch: expected a dict, or a list of arrays with names= or"
        " schema= kwarg, or an object with __arrow_c_record_batch__"
    )


# ---------------------------------------------------------------------------
# Table: methods that need custom Python ↔ Mojo dispatch
# ---------------------------------------------------------------------------


def _table_schema(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    return ptr[].schema.to_python_object()


def _table_columns(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    var rb = ptr[].combine_chunks()
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for i in range(len(rb.columns)):
        result.append(rb.columns[i].copy().to_python_object())
    return result


def _table_column_names(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for name in ptr[].column_names():
        result.append(PythonObject(name))
    return result


def _table_shape(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    return Python.tuple(ptr[].num_rows(), ptr[].num_columns())


def _table_column(
    py_self: PythonObject, key: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
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
    py_self: PythonObject, other: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    return PythonObject(ptr[] == other.downcast_value_ptr[Table]()[])


def _table_to_pydict(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    var rb = ptr[].combine_chunks()
    return _to_pydict(rb.schema, rb.columns)


def _table_to_pylist(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    var rb = ptr[].combine_chunks()
    return _to_pylist(rb.schema, rb.columns)


def _table_arrow_c_stream(
    py_self: PythonObject,
    requested_schema: PythonObject,
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    var batches = ptr[].to_batches()
    return CArrowArrayStream.from_batches(
        ptr[].schema.copy(), batches^
    ).to_pycapsule()


def _table_arrow_c_schema(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    return CArrowSchema.from_schema(ptr[].schema).to_pycapsule()


# def _table_rich_compare(
#     first: Table, second: PythonObject, op: Int
# ) raises -> Bool:
#     if op == RichCompareOps.Py_EQ:
#         return first == second.downcast_value_ptr[Table]()[]
#     raise NotImplementedError()


# ---------------------------------------------------------------------------
# Table constructor
# ---------------------------------------------------------------------------


def table(data: PythonObject, names: PythonObject) raises -> PythonObject:
    """Create a Table from a dict, list+names, or Arrow protocol object."""
    try:
        return Table(py=data).to_python_object()
    except:
        pass

    var rb: RecordBatch
    var builtins = Python.import_module("builtins")
    if builtins.isinstance(data, builtins.dict):
        rb = _build_from_dict(data)
    elif not names.__is__(builtins.None):
        rb = _build_from_arrays(data, names)
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
    py_self: PythonObject,
    right: PythonObject,
    keys: PythonObject,
    right_keys: PythonObject,
    join_type: PythonObject,
    num_threads: PythonObject,
) raises -> PythonObject:
    """Marshal Python arguments and call `RecordBatch.join`.

    The semantics — key resolution, join-kind parsing, result assembly — are on
    the core type. This only converts Python values to Mojo ones.
    """
    ref left = py_self.downcast_value_ptr[RecordBatch]()[]
    ref right_rb = right.downcast_value_ptr[RecordBatch]()[]

    var left_keys = List[String]()
    for i in range(Int(keys.__len__())):
        left_keys.append(String(py=keys[i]))

    var rkeys = List[String]()
    if right_keys is not PythonObject(None):
        for i in range(Int(right_keys.__len__())):
            rkeys.append(String(py=right_keys[i]))

    return left.join(
        right_rb,
        left_keys,
        rkeys,
        String(py=join_type),
        ExecContext.parallel(Int(py=num_threads)),
    ).to_python_object()


def _pylist_str(obj: PythonObject) raises -> List[String]:
    """A Python sequence of strings as a Mojo `List[String]`."""
    var out = List[String]()
    for i in range(Int(obj.__len__())):
        out.append(String(py=obj[i]))
    return out^


def _record_batch_group_by(
    py_self: PythonObject,
    keys: PythonObject,
    values: PythonObject,
    funcs: PythonObject,
    num_threads: PythonObject,
) raises -> PythonObject:
    """Marshal and call `RecordBatch.group_by`."""
    ref rb = py_self.downcast_value_ptr[RecordBatch]()[]
    return rb.group_by(
        _pylist_str(keys),
        _pylist_str(values),
        _pylist_str(funcs),
        ExecContext.parallel(Int(py=num_threads)),
    ).to_python_object()


def _record_batch_aggregate(
    py_self: PythonObject,
    values: PythonObject,
    funcs: PythonObject,
) raises -> PythonObject:
    """Marshal and call `RecordBatch.aggregate`."""
    ref rb = py_self.downcast_value_ptr[RecordBatch]()[]
    return rb.aggregate(
        _pylist_str(values), _pylist_str(funcs)
    ).to_python_object()


def _record_batch_sort_by(
    py_self: PythonObject,
    by: PythonObject,
    null_placement: PythonObject,
    num_threads: PythonObject,
) raises -> PythonObject:
    """Flatten PyArrow's `by` spellings, then call `RecordBatch.sort_by`.

    `by` is a name, or a list whose entries are names or `(name, order)` pairs.
    Unpacking that is genuinely Python-shaped, so it stays here; everything after
    it is on the core type.
    """
    ref rb = py_self.downcast_value_ptr[RecordBatch]()[]
    var nulls_first = True
    if not null_placement.__is__(PythonObject(None)):
        nulls_first = String(py=null_placement) != "at_end"

    var builtins = Python.import_module("builtins")
    var keys = List[String]()
    var ascending = List[Bool]()
    if builtins.isinstance(by, builtins.str):
        keys.append(String(py=by))
        ascending.append(True)
    else:
        for i in range(Int(by.__len__())):
            var entry = by[i]
            if builtins.isinstance(entry, builtins.str):
                keys.append(String(py=entry))
                ascending.append(True)
            else:
                keys.append(String(py=entry[0]))
                ascending.append(String(py=entry[1]) != "descending")

    return rb.sort_by(
        keys,
        ascending,
        nulls_first,
        ExecContext.parallel(Int(py=num_threads)),
    ).to_python_object()


def _record_batch_str(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return PythonObject(String(ptr[]))


def _table_str(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    return PythonObject(String(ptr[]))


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
        .def_method[_record_batch_column_names]("column_names")
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
        .def_method[_record_batch_sort_by]("sort_by")
        .def_method[_record_batch_join]("join")
        .def_method[_record_batch_group_by]("group_by")
        .def_method[_record_batch_aggregate]("aggregate")
    )
    _ = rb_py.def_method[_record_batch_str]("__str__").def_method[
        _record_batch_str
    ]("__repr__")
    # var rb_tp = TypeProtocolBuilder[RecordBatch](rb_py)
    # _ = rb_tp.def_richcompare[_record_batch_rich_compare]()

    mb.def_function[record_batch]("record_batch")

    # Table
    ref t_py = mb.add_type[Table]("Table")
    _ = (
        t_py.def_method[_table_schema]("schema")
        .def_method[_table_columns]("columns")
        .def_method[_table_shape]("shape")
        .def_method[pymethod[Table.num_rows]()]("num_rows")
        .def_method[pymethod[Table.num_columns]()]("num_columns")
        .def_method[_table_column_names]("column_names")
        .def_method[_table_column]("column")
        .def_method[pymethod[Table.to_batches]()]("to_batches")
        .def_method[_table_equals]("equals")
        .def_method[_table_equals]("__eq__")
        .def_method[_table_to_pydict]("to_pydict")
        .def_method[_table_to_pylist]("to_pylist")
        .def_method[_table_arrow_c_stream]("__arrow_c_stream__")
        .def_method[_table_arrow_c_schema]("__arrow_c_schema__")
    )
    _ = t_py.def_method[_table_str]("__str__").def_method[_table_str](
        "__repr__"
    )
    # var t_tp = TypeProtocolBuilder[Table](t_py)
    # _ = t_tp.def_richcompare[_table_rich_compare]()

    mb.def_function[table]("table")
