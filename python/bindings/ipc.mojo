"""Python bindings for Arrow IPC file and stream reader/writer."""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.ipc import (
    read_ipc_file as _ipc_read_file,
    read_ipc_stream as _ipc_read_stream,
    read_ipc_file_schema as _ipc_read_file_schema,
    read_ipc_stream_schema as _ipc_read_stream_schema,
    write_ipc_file as _ipc_write_file,
    write_ipc_stream as _ipc_write_stream,
)
from marrow.tabular import RecordBatch


def read_ipc_file(path: PythonObject) raises -> PythonObject:
    var path_str = String(py=path)
    var batches = _ipc_read_file(path_str)
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for batch in batches:
        result.append(batch.copy().to_python_object())
    return result


def write_ipc_file(
    path: PythonObject, batches: PythonObject, schema: PythonObject
) raises -> PythonObject:
    var path_str = String(py=path)
    var rb_list = List[RecordBatch]()
    var builtins = Python.import_module("builtins")
    if not batches.__is__(builtins.None):
        var n = Int(py=batches.__len__())
        for i in range(n):
            rb_list.append(RecordBatch(py=batches[i]))
    if not schema.__is__(builtins.None):
        var schema_rb = RecordBatch(py=schema)
        _ipc_write_file(path_str, schema_rb.schema, rb_list)
    elif len(rb_list) > 0:
        _ipc_write_file(path_str, rb_list)
    else:
        raise Error(
            "write_ipc_file requires 'batches' or 'schema' keyword argument"
        )
    return Python.evaluate("None")


def read_ipc_stream(path: PythonObject) raises -> PythonObject:
    var path_str = String(py=path)
    var batches = _ipc_read_stream(path_str)
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for batch in batches:
        result.append(batch.copy().to_python_object())
    return result


def write_ipc_stream(
    path: PythonObject, batches: PythonObject, schema: PythonObject
) raises -> PythonObject:
    var path_str = String(py=path)
    var rb_list = List[RecordBatch]()
    var builtins = Python.import_module("builtins")
    if not batches.__is__(builtins.None):
        var n = Int(py=batches.__len__())
        for i in range(n):
            rb_list.append(RecordBatch(py=batches[i]))
    if not schema.__is__(builtins.None):
        var schema_rb = RecordBatch(py=schema)
        _ipc_write_stream(path_str, schema_rb.schema, rb_list)
    elif len(rb_list) > 0:
        _ipc_write_stream(path_str, rb_list)
    else:
        raise Error(
            "write_ipc_stream requires 'batches' or 'schema' keyword argument"
        )
    return Python.evaluate("None")


def read_ipc_file_schema(path: PythonObject) raises -> PythonObject:
    var path_str = String(py=path)
    var batch = _ipc_read_file_schema(path_str)
    return batch.copy().to_python_object()


def read_ipc_stream_schema(path: PythonObject) raises -> PythonObject:
    var path_str = String(py=path)
    var batch = _ipc_read_stream_schema(path_str)
    return batch.copy().to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add IPC reader/writer functions to the Python module."""
    mb.def_function[read_ipc_file]("read_ipc_file")
    mb.def_function[write_ipc_file]("write_ipc_file")
    mb.def_function[read_ipc_stream]("read_ipc_stream")
    mb.def_function[read_ipc_file_schema]("read_ipc_file_schema")
    mb.def_function[read_ipc_stream_schema]("read_ipc_stream_schema")
    mb.def_function[write_ipc_stream]("write_ipc_stream")
