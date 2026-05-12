"""Python bindings for Arrow IPC file and stream reader/writer."""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.collections import OwnedKwargsDict
from marrow.ipc import (
    read_ipc_file,
    read_ipc_stream,
    read_ipc_file_schema,
    read_ipc_stream_schema,
    write_ipc_file,
    write_ipc_stream,
)
from marrow.tabular import RecordBatch


def _py_read_ipc_file(
    data: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    var path = String(py=data)
    var batches = read_ipc_file(path)
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for batch in batches:
        result.append(batch.copy().to_python_object())
    return result


def _py_write_ipc_file(
    data: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    var path = String(py=data)
    var batches = List[RecordBatch]()
    if opt_b := kwargs.find("batches"):
        var batches_obj = opt_b.value()
        var n = Int(py=batches_obj.__len__())
        for i in range(n):
            batches.append(RecordBatch(py=batches_obj[i]))
    if opt_s := kwargs.find("schema"):
        var schema_rb = RecordBatch(py=opt_s.value())
        write_ipc_file(path, schema_rb.schema, batches)
    elif len(batches) > 0:
        write_ipc_file(path, batches)
    else:
        raise Error(
            "write_ipc_file requires 'batches' or 'schema' keyword argument"
        )
    return Python.evaluate("None")


def _py_read_ipc_stream(
    data: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    var path = String(py=data)
    var batches = read_ipc_stream(path)
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for batch in batches:
        result.append(batch.copy().to_python_object())
    return result


def _py_write_ipc_stream(
    data: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    var path = String(py=data)
    var batches = List[RecordBatch]()
    if opt_b := kwargs.find("batches"):
        var batches_obj = opt_b.value()
        var n = Int(py=batches_obj.__len__())
        for i in range(n):
            batches.append(RecordBatch(py=batches_obj[i]))
    if opt_s := kwargs.find("schema"):
        var schema_rb = RecordBatch(py=opt_s.value())
        write_ipc_stream(path, schema_rb.schema, batches)
    elif len(batches) > 0:
        write_ipc_stream(path, batches)
    else:
        raise Error(
            "write_ipc_stream requires 'batches' or 'schema' keyword argument"
        )
    return Python.evaluate("None")


def _py_read_ipc_file_schema(
    data: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    var path = String(py=data)
    var batch = read_ipc_file_schema(path)
    return batch.copy().to_python_object()


def _py_read_ipc_stream_schema(
    data: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    var path = String(py=data)
    var batch = read_ipc_stream_schema(path)
    return batch.copy().to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add IPC reader/writer functions to the Python module."""
    mb.def_function[_py_read_ipc_file](
        "read_ipc_file",
        docstring=(
            "read_ipc_file(path) -> list[RecordBatch]\n--\n\nRead an Arrow IPC"
            " file (.arrow) and return a list of RecordBatches."
        ),
    )
    mb.def_function[_py_write_ipc_file](
        "write_ipc_file",
        docstring=(
            "write_ipc_file(path, /, *, batches) -> None\n--\n\n"
            "Write a list of RecordBatches to an Arrow IPC file (.arrow)."
        ),
    )
    mb.def_function[_py_read_ipc_stream](
        "read_ipc_stream",
        docstring=(
            "read_ipc_stream(path) -> list[RecordBatch]\n--\n\n"
            "Read an Arrow IPC stream and return a list of RecordBatches."
        ),
    )
    mb.def_function[_py_read_ipc_file_schema](
        "read_ipc_file_schema",
        docstring=(
            "read_ipc_file_schema(path) -> RecordBatch\n--\n\nRead the schema"
            " from an Arrow IPC file; return a 0-row RecordBatch."
        ),
    )
    mb.def_function[_py_read_ipc_stream_schema](
        "read_ipc_stream_schema",
        docstring=(
            "read_ipc_stream_schema(path) -> RecordBatch\n--\n\nRead the schema"
            " from an Arrow IPC stream; return a 0-row RecordBatch."
        ),
    )
    mb.def_function[_py_write_ipc_stream](
        "write_ipc_stream",
        docstring=(
            "write_ipc_stream(path, /, *, batches) -> None\n--\n\n"
            "Write a list of RecordBatches to an Arrow IPC stream file."
        ),
    )
