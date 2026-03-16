"""Python bindings for RecordBatch.

Exposes RecordBatch to Python with an API matching PyArrow's RecordBatch.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.collections import OwnedKwargsDict
from marrow.tabular import RecordBatch
from marrow.schema import Schema
from marrow.arrays import Array
from marrow.dtypes import Field, struct_
from marrow.c_data import CArrowSchema, CArrowArray
from helpers import pymethod


# ---------------------------------------------------------------------------
# Helper: convert a Python array (marrow or any __arrow_c_array__ object)
# to a Mojo Array via the Arrow C Data Interface.
# ---------------------------------------------------------------------------


fn _py_array_to_mojo(obj: PythonObject) raises -> Array:
    """Convert a Python array to a Mojo Array via __arrow_c_array__."""
    var builtins = Python.import_module("builtins")
    if builtins.hasattr(obj, "__arrow_c_array__"):
        var caps = obj.__arrow_c_array__(Python.none())
        var c_schema = CArrowSchema.from_pycapsule(caps[0])
        var c_array = CArrowArray.from_pycapsule(caps[1])
        return c_array^.to_array(c_schema.to_dtype())
    raise Error("Expected an array supporting __arrow_c_array__")


# ---------------------------------------------------------------------------
# Property / method wrappers that cannot use the generic pymethod helper
# because they return Python containers or need multi-arg dispatch.
# ---------------------------------------------------------------------------


fn _record_batch_schema(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    """Return the RecordBatch schema as a Python Schema object."""
    var schema = ptr[].schema
    return schema.to_python_object()


fn _record_batch_columns(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    """Return the columns as a Python list of typed array objects."""
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for i in range(ptr[].num_columns()):
        result.append(ptr[].columns[i].copy().to_python_object())
    return result


fn _record_batch_shape(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    """Return (num_rows, num_columns) as a Python tuple."""
    return Python.tuple(ptr[].num_rows(), ptr[].num_columns())



fn _record_batch_column(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin], key: PythonObject
) raises -> PythonObject:
    """Return a column by integer index or string name."""
    var builtins = Python.import_module("builtins")
    if builtins.isinstance(key, builtins.int):
        var idx = Int(py=key)
        return ptr[].columns[idx].copy().to_python_object()
    else:
        var name = String(py=key)
        var idx = ptr[].schema.get_field_index(name)
        if idx == -1:
            raise Error("Column '{}' not found.".format(name))
        return ptr[].columns[idx].copy().to_python_object()


fn _record_batch_slice(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin],
    offset: PythonObject,
    length: PythonObject,
) raises -> PythonObject:
    """Return a zero-copy slice from offset with given length."""
    return ptr[].slice(Int(py=offset), Int(py=length)).to_python_object()


fn _record_batch_equals(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin], other: PythonObject
) raises -> PythonObject:
    """Return True if the two RecordBatches have equal schema and columns."""
    var other_rb = other.downcast_value_ptr[RecordBatch]()[].copy()
    return PythonObject(ptr[] == other_rb)



fn _record_batch_select(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin], columns: PythonObject
) raises -> PythonObject:
    """Return a new RecordBatch with a subset of columns (by index or name)."""
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



fn _record_batch_add_column(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin],
    i: PythonObject,
    field: PythonObject,
    column: PythonObject,
) raises -> PythonObject:
    """Return a new RecordBatch with `column` inserted at position `i`."""
    return ptr[].add_column(
        Int(py=i), Field(py=field), _py_array_to_mojo(column)
    ).to_python_object()


fn _record_batch_append_column(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin],
    field: PythonObject,
    column: PythonObject,
) raises -> PythonObject:
    """Return a new RecordBatch with `column` appended at the end."""
    return ptr[].append_column(
        Field(py=field), _py_array_to_mojo(column)
    ).to_python_object()


fn _record_batch_set_column(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin],
    i: PythonObject,
    field: PythonObject,
    column: PythonObject,
) raises -> PythonObject:
    """Return a new RecordBatch with the column at `i` replaced."""
    return ptr[].set_column(
        Int(py=i), Field(py=field), _py_array_to_mojo(column)
    ).to_python_object()


fn _record_batch_to_pydict(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    """Convert to a Python dict mapping column names to lists of values."""
    var builtins = Python.import_module("builtins")
    var result = builtins.dict()
    for i in range(ptr[].num_columns()):
        var name = ptr[].schema.fields[i].name
        var col_obj = ptr[].columns[i].copy().to_python_object()
        var col_len = Int(col_obj.__len__())
        var values = builtins.list()
        for j in range(col_len):
            values.append(col_obj[j])
        result[name.to_python_object()] = values
    return result


fn _record_batch_to_pylist(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    """Convert to a Python list of row dicts."""
    var builtins = Python.import_module("builtins")
    var n_rows = ptr[].num_rows()
    var n_cols = ptr[].num_columns()
    var col_objs = List[PythonObject]()
    var col_names = ptr[].column_names()
    for i in range(n_cols):
        col_objs.append(ptr[].columns[i].copy().to_python_object())
    var result = builtins.list()
    for j in range(n_rows):
        var row = builtins.dict()
        for i in range(n_cols):
            row[col_names[i].to_python_object()] = col_objs[i][j]
        result.append(row)
    return result


fn _record_batch_arrow_c_record_batch(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin],
    requested_schema: PythonObject,
) raises -> PythonObject:
    """Export as Arrow C Data Interface (schema capsule, array capsule) pair."""
    var schema_cap = CArrowSchema.from_schema(ptr[].schema.fields).to_pycapsule()
    var struct_arr: Array = ptr[].to_struct_array()
    var array_cap = CArrowArray.from_array(struct_arr).to_pycapsule()
    return Python.tuple(schema_cap, array_cap)


fn _record_batch_arrow_c_array(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin],
    requested_schema: PythonObject,
) raises -> PythonObject:
    """Export as Arrow C Data Interface (schema capsule, array capsule) pair via __arrow_c_array__."""
    var schema_cap = CArrowSchema.from_schema(ptr[].schema.fields).to_pycapsule()
    var struct_arr: Array = ptr[].to_struct_array()
    var array_cap = CArrowArray.from_array(struct_arr).to_pycapsule()
    return Python.tuple(schema_cap, array_cap)


fn _record_batch_arrow_c_schema(
    ptr: UnsafePointer[RecordBatch, MutAnyOrigin]
) raises -> PythonObject:
    """Export the schema as an Arrow C Data Interface capsule."""
    return CArrowSchema.from_schema(ptr[].schema.fields).to_pycapsule()


# ---------------------------------------------------------------------------
# Constructor: record_batch(data, names=None, schema=None)
# ---------------------------------------------------------------------------


fn record_batch(
    data: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    """Create a RecordBatch from a dict of arrays, a list of arrays + names,
    or any object implementing the __arrow_c_record_batch__ protocol."""
    var builtins = Python.import_module("builtins")

    # Protocol: __arrow_c_record_batch__
    if builtins.hasattr(data, "__arrow_c_record_batch__"):
        var caps = data.__arrow_c_record_batch__()
        var schema = CArrowSchema.from_pycapsule(caps[0]).to_schema()
        var struct_dtype = struct_(schema.fields)
        var c_array = CArrowArray.from_pycapsule(caps[1])
        var struct_arr = c_array^.to_array(struct_dtype)
        var columns = List[Array]()
        for child in struct_arr.children:
            columns.append(child.copy())
        return RecordBatch(
            schema=schema, columns=columns^
        ).to_python_object()

    # Protocol: __arrow_c_array__ returning a struct (e.g. PyArrow RecordBatch)
    if builtins.hasattr(data, "__arrow_c_array__"):
        var caps = data.__arrow_c_array__(Python.none())
        var c_schema = CArrowSchema.from_pycapsule(caps[0])
        var schema = c_schema.to_schema()
        var struct_dtype = struct_(schema.fields)
        var c_array = CArrowArray.from_pycapsule(caps[1])
        var struct_arr = c_array^.to_array(struct_dtype)
        var columns = List[Array]()
        for child in struct_arr.children:
            columns.append(child.copy())
        return RecordBatch(
            schema=schema, columns=columns^
        ).to_python_object()

    # Dict: {column_name: array, ...}
    if builtins.isinstance(data, builtins.dict):
        var fields = List[Field]()
        var columns = List[Array]()
        for key in data:
            var name = String(py=key)
            var arr = _py_array_to_mojo(data[key])
            fields.append(Field(name=name, dtype=arr.dtype.copy()))
            columns.append(arr^)
        return RecordBatch(
            schema=Schema(fields=fields^), columns=columns^
        ).to_python_object()

    # List/tuple of arrays + names= kwarg
    if opt := kwargs.find("names"):
        var names_obj = opt.value()
        var fields = List[Field]()
        var columns = List[Array]()
        var i = 0
        for arr_obj in data:
            var arr = _py_array_to_mojo(arr_obj)
            var name = String(py=names_obj[i])
            fields.append(Field(name=name, dtype=arr.dtype.copy()))
            columns.append(arr^)
            i += 1
        return RecordBatch(
            schema=Schema(fields=fields^), columns=columns^
        ).to_python_object()

    raise Error(
        "record_batch: expected a dict, or a list of arrays with names= kwarg,"
        " or an object with __arrow_c_record_batch__"
    )


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add RecordBatch type and record_batch constructor to the Python module."""
    ref rb_py = mb.add_type[RecordBatch]("RecordBatch")
    _ = (
        rb_py
        .def_method[_record_batch_schema]("schema")
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
        .def_method[_record_batch_add_column]("add_column")
        .def_method[_record_batch_append_column]("append_column")
        .def_method[pymethod[RecordBatch.remove_column]()]("remove_column")
        .def_method[_record_batch_set_column]("set_column")
        .def_method[pymethod[RecordBatch.__str__]()]("__str__")
        .def_method[pymethod[RecordBatch.__str__]()]("__repr__")
        .def_method[_record_batch_to_pydict]("to_pydict")
        .def_method[_record_batch_to_pylist]("to_pylist")
        .def_method[_record_batch_arrow_c_array]("__arrow_c_array__")
        .def_method[_record_batch_arrow_c_record_batch]("__arrow_c_record_batch__")
        .def_method[_record_batch_arrow_c_schema]("__arrow_c_schema__")
    )

    mb.def_function[record_batch](
        "record_batch",
        docstring=(
            "Create a RecordBatch from a dict of arrays, a list of arrays with"
            " names=, or any object implementing __arrow_c_record_batch__."
        ),
    )
