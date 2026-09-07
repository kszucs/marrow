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
from std.memory import ArcPointer, Pointer
from marrow.c_data import CArrowSchema, CArrowArray, CArrowArrayStream
from marrow.arrays import Int32Array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




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
    """Every column, as the `ChunkedArray`s the table holds."""
    var ptr = py_self.downcast_value_ptr[Table]()
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    for i in range(ptr[].num_columns()):
        _ = out.append(PythonObject(alloc=ptr[].column(i).copy()))
    return out


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
    """The column as the `ChunkedArray` the table holds.

    This used to `combine_chunks()` and hand back a single `Array`, because
    `ChunkedArray` was not a registered type -- so the answer was a copy whose
    shape said nothing about the table's own."""
    var ptr = py_self.downcast_value_ptr[Table]()
    var builtins = Python.import_module("builtins")
    if Bool(py=builtins.isinstance(key, builtins.int)):
        return PythonObject(alloc=ptr[].column(Int(py=key)).copy())
    var name = String(py=key)
    var idx = ptr[].schema.get_field_index(name)
    if idx == -1:
        raise Error("Column '", name, "' not found.")
    return PythonObject(alloc=ptr[].column(idx).copy())


def _table_equals(
    py_self: PythonObject, other: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    return PythonObject(ptr[] == other.downcast_value_ptr[Table]()[])




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


def table_from_batches(batches: PythonObject) raises -> PythonObject:
    """A `Table` whose columns are chunked one chunk per batch.

    The one way to build a multi-chunk column: a `ChunkedArray` reaches Python
    out of a `Table`, and `Table.from_batches` is what puts more than one chunk
    in it. `table()` always answers a single-chunk table because it builds one
    `RecordBatch`."""
    var n = Int(py=batches.__len__())
    if n == 0:
        raise Error("from_batches: needs at least one batch")
    var out = List[RecordBatch](capacity=n)
    for i in range(n):
        out.append(RecordBatch(py=batches[i]))
    var schema = out[0].schema.copy()
    return Table.from_batches(schema, out^).to_python_object()


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


def _record_batch_num_rows(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(py_self.downcast_value_ptr[RecordBatch]()[].num_rows())


def _record_batch_num_columns(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return PythonObject(ptr[].num_columns())


def _record_batch_rename_columns(
    py_self: PythonObject, names: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    var out = List[String]()
    for i in range(Int(py=names.__len__())):
        out.append(String(py=names[i]))
    return ptr[].rename_columns(out).to_python_object()


def _record_batch_add_column(
    py_self: PythonObject,
    index: PythonObject,
    field: PythonObject,
    column: PythonObject,
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return ptr[].add_column(
        Int(py=index), Field(py=field), DynArray(py=column)
    ).to_python_object()


def _record_batch_append_column(
    py_self: PythonObject, field: PythonObject, column: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return ptr[].append_column(
        Field(py=field), DynArray(py=column)
    ).to_python_object()


def _record_batch_remove_column(
    py_self: PythonObject, index: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return ptr[].remove_column(Int(py=index)).to_python_object()


def _record_batch_set_column(
    py_self: PythonObject,
    index: PythonObject,
    field: PythonObject,
    column: PythonObject,
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[RecordBatch]()
    return ptr[].set_column(
        Int(py=index), Field(py=field), DynArray(py=column)
    ).to_python_object()


# ---------------------------------------------------------------------------
# ChunkedArray
# ---------------------------------------------------------------------------
#
# `Table.columns` is a `List[ChunkedArray]` and always has been, but the type
# was never registered -- so `Table.column()` combined the chunks and handed
# back a single `Array`, which is a different object with a different cost and
# says nothing about how the table is actually laid out.


def _chunked_len(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(py_self.downcast_value_ptr[ChunkedArray]()[].length)


def _chunked_type(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[ChunkedArray]()
    return ptr[].dtype.copy().to_python_object()


def _chunked_num_chunks(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[ChunkedArray]()
    return PythonObject(len(ptr[].chunks))


def _chunked_chunk(
    py_self: PythonObject, index: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[ChunkedArray]()
    return ptr[].chunk(Int(py=index)).copy().to_python_object()


def _chunked_chunks(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[ChunkedArray]()
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    for ref chunk in ptr[].chunks:
        _ = out.append(chunk.copy().to_python_object())
    return out


def _chunked_combine_chunks(py_self: PythonObject) raises -> PythonObject:
    """One contiguous `Array`. Copies, since `combine_chunks` consumes."""
    var ptr = py_self.downcast_value_ptr[ChunkedArray]()
    return ptr[].copy().combine_chunks().to_python_object()


def _chunked_str(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(String(py_self.downcast_value_ptr[ChunkedArray]()[]))


def _table_combine_chunks(py_self: PythonObject) raises -> PythonObject:
    """The whole table as one `RecordBatch`."""
    var ptr = py_self.downcast_value_ptr[Table]()
    return ptr[].combine_chunks().to_python_object()


def _table_num_rows(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(py_self.downcast_value_ptr[Table]()[].num_rows())


def _table_num_columns(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(py_self.downcast_value_ptr[Table]()[].num_columns())


def _table_to_batches(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Table]()
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    for ref batch in ptr[].to_batches():
        _ = out.append(batch.copy().to_python_object())
    return out


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add RecordBatch, Table types and constructors to the Python module."""
    # Registered first, and on its own: `add_type` reallocates the module
    # builder's type list, so adding a type while a `ref` from an earlier
    # `add_type` is still live invalidates that reference.
    _ = (
        mb.add_type[ChunkedArray]("ChunkedArray")
        .def_method[_chunked_len]("__len__")
        .def_method[_chunked_type]("type")
        .def_method[_chunked_num_chunks]("num_chunks")
        .def_method[_chunked_chunk]("chunk")
        .def_method[_chunked_chunks]("chunks")
        .def_method[_chunked_combine_chunks]("combine_chunks")
        .def_method[_chunked_str]("__str__")
    )

    ref rb_py = mb.add_type[RecordBatch]("RecordBatch")
    _ = (
        rb_py.def_method[_record_batch_schema]("schema")
        .def_method[_record_batch_columns]("columns")
        .def_method[_record_batch_shape]("shape")
        .def_method[_record_batch_num_rows]("num_rows")
        .def_method[_record_batch_num_columns]("num_columns")
        .def_method[_record_batch_column_names]("column_names")
        .def_method[_record_batch_column]("column")
        .def_method[_record_batch_slice]("slice")
        .def_method[_record_batch_equals]("equals")
        .def_method[_record_batch_equals]("__eq__")
        .def_method[_record_batch_select]("select")
        .def_method[_record_batch_rename_columns]("rename_columns")
        .def_method[_record_batch_add_column]("add_column")
        .def_method[_record_batch_append_column]("append_column")
        .def_method[_record_batch_remove_column]("remove_column")
        .def_method[_record_batch_set_column]("set_column")
        .def_method[_record_batch_arrow_c_array]("__arrow_c_array__")
        .def_method[_record_batch_arrow_c_array]("__arrow_c_record_batch__")
        .def_method[_record_batch_arrow_c_schema]("__arrow_c_schema__")
        .def_method[_record_batch_sort_by]("sort_by")
        .def_method[_record_batch_join]("join")
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
        .def_method[_table_combine_chunks]("combine_chunks")
        .def_method[_table_columns]("columns")
        .def_method[_table_shape]("shape")
        .def_method[_table_num_rows]("num_rows")
        .def_method[_table_num_columns]("num_columns")
        .def_method[_table_column_names]("column_names")
        .def_method[_table_column]("column")
        .def_method[_table_to_batches]("to_batches")
        .def_method[_table_equals]("equals")
        .def_method[_table_equals]("__eq__")
        .def_method[_table_arrow_c_stream]("__arrow_c_stream__")
        .def_method[_table_arrow_c_schema]("__arrow_c_schema__")
    )
    _ = t_py.def_method[_table_str]("__str__").def_method[_table_str](
        "__repr__"
    )
    # var t_tp = TypeProtocolBuilder[Table](t_py)
    # _ = t_tp.def_richcompare[_table_rich_compare]()

    mb.def_function[table]("table")
    mb.def_function[table_from_batches]("table_from_batches")
