# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Python bindings for Arrow IPC file and stream reader/writer."""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.errors import InvalidError
from marrow.formats.ipc import (
    BodyCompression,
    RecordBatchFileReader,
    RecordBatchFileWriter,
    RecordBatchStreamReader,
    RecordBatchStreamWriter,
    read_ipc_file as _ipc_read_file,
    read_ipc_stream as _ipc_read_stream,
    read_ipc_file_schema as _ipc_read_file_schema,
    read_ipc_stream_schema as _ipc_read_stream_schema,
    write_ipc_file as _ipc_write_file,
    write_ipc_stream as _ipc_write_stream,
)
from marrow.io import BufferSource, FileSink
from marrow.schema import Schema
from marrow.tabular import RecordBatch


def read_ipc_file(
    path: PythonObject, native_codecs: PythonObject
) raises -> PythonObject:
    var path_str = String(py=path)
    var batches = _ipc_read_file(path_str, native_codecs=Bool(py=native_codecs))
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for batch in batches:
        result.append(batch.copy())
    return result


def _compression(
    compression: PythonObject,
) raises -> Optional[BodyCompression]:
    """`None`, or a codec by pyarrow's name for it."""
    if compression.__is__(Python.none()):
        return None
    return BodyCompression.from_name(String(py=compression))


def write_ipc_file(
    path: PythonObject,
    batches: PythonObject,
    schema: PythonObject,
    compression: PythonObject,
    native_codecs: PythonObject,
) raises -> PythonObject:
    var path_str = String(py=path)
    var rb_list = List[RecordBatch]()
    var builtins = Python.import_module("builtins")
    if not batches.__is__(builtins.None):
        var n = Int(py=batches.__len__())
        for i in range(n):
            rb_list.append(RecordBatch(py=batches[i]))
    if not schema.__is__(builtins.None):
        _ipc_write_file(
            path_str,
            Schema(py=schema),
            rb_list,
            compression=_compression(compression),
            native_codecs=Bool(py=native_codecs),
        )
    elif len(rb_list) > 0:
        _ipc_write_file(
            path_str,
            rb_list,
            compression=_compression(compression),
            native_codecs=Bool(py=native_codecs),
        )
    else:
        raise InvalidError(
            "write_ipc_file requires 'batches' or 'schema' keyword argument"
        )
    return Python.none()


def read_ipc_stream(
    path: PythonObject, native_codecs: PythonObject
) raises -> PythonObject:
    var path_str = String(py=path)
    var batches = _ipc_read_stream(
        path_str, native_codecs=Bool(py=native_codecs)
    )
    var builtins = Python.import_module("builtins")
    var result = builtins.list()
    for batch in batches:
        result.append(batch.copy())
    return result


def write_ipc_stream(
    path: PythonObject,
    batches: PythonObject,
    schema: PythonObject,
    compression: PythonObject,
    native_codecs: PythonObject,
) raises -> PythonObject:
    var path_str = String(py=path)
    var rb_list = List[RecordBatch]()
    var builtins = Python.import_module("builtins")
    if not batches.__is__(builtins.None):
        var n = Int(py=batches.__len__())
        for i in range(n):
            rb_list.append(RecordBatch(py=batches[i]))
    if not schema.__is__(builtins.None):
        _ipc_write_stream(
            path_str,
            Schema(py=schema),
            rb_list,
            compression=_compression(compression),
            native_codecs=Bool(py=native_codecs),
        )
    elif len(rb_list) > 0:
        _ipc_write_stream(
            path_str,
            rb_list,
            compression=_compression(compression),
            native_codecs=Bool(py=native_codecs),
        )
    else:
        raise InvalidError(
            "write_ipc_stream requires 'batches' or 'schema' keyword argument"
        )
    return Python.none()


def read_ipc_file_schema(path: PythonObject) raises -> PythonObject:
    var path_str = String(py=path)
    var batch = _ipc_read_file_schema(path_str)
    return batch.copy()


def read_ipc_stream_schema(path: PythonObject) raises -> PythonObject:
    var path_str = String(py=path)
    var batch = _ipc_read_stream_schema(path_str)
    return batch.copy()


def _to_dict(metadata: Dict[String, String]) raises -> PythonObject:
    var result = Python.import_module("builtins").dict()
    for entry in metadata.items():
        result[entry.key] = entry.value
    return result


def _from_dict(metadata: PythonObject) raises -> Dict[String, String]:
    var result = Dict[String, String]()
    for key in metadata:
        result[String(py=key)] = String(py=metadata[key])
    return result^


def _batch_and_metadata(
    var batch: RecordBatch, metadata: Dict[String, String]
) raises -> PythonObject:
    var pair = Python.import_module("builtins").list()
    pair.append(batch^)
    pair.append(_to_dict(metadata))
    return Python.import_module("builtins").tuple(pair)


struct IpcFileWriter(Movable, Writable):
    """The registered type behind `marrow.ipc.new_file`."""

    var writer: RecordBatchFileWriter[FileSink]

    def __init__(out self, var writer: RecordBatchFileWriter[FileSink]):
        self.writer = writer^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("IpcFileWriter")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<marrow.IpcFileWriter>")


struct IpcStreamWriter(Movable, Writable):
    """The registered type behind `marrow.ipc.new_stream`."""

    var writer: RecordBatchStreamWriter[FileSink]

    def __init__(out self, var writer: RecordBatchStreamWriter[FileSink]):
        self.writer = writer^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("IpcStreamWriter")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<marrow.IpcStreamWriter>")


struct IpcFileReader(Movable, Writable):
    """The registered type behind `marrow.ipc.open_file`."""

    var reader: RecordBatchFileReader[BufferSource]

    def __init__(out self, var reader: RecordBatchFileReader[BufferSource]):
        self.reader = reader^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("IpcFileReader")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<marrow.IpcFileReader>")


struct IpcStreamReader(Movable, Writable):
    """The registered type behind `marrow.ipc.open_stream`."""

    var reader: RecordBatchStreamReader[BufferSource]

    def __init__(out self, var reader: RecordBatchStreamReader[BufferSource]):
        self.reader = reader^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("IpcStreamReader")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<marrow.IpcStreamReader>")


def ipc_new_file(
    path: PythonObject,
    schema: PythonObject,
    compression: PythonObject,
    native_codecs: PythonObject,
    metadata: PythonObject,
) raises -> PythonObject:
    var writer = RecordBatchFileWriter(
        String(py=path),
        Schema(py=schema),
        _compression(compression),
        Bool(py=native_codecs),
        metadata=_from_dict(metadata),
    )
    return PythonObject(alloc=IpcFileWriter(writer^))


def ipc_new_stream(
    path: PythonObject,
    schema: PythonObject,
    compression: PythonObject,
    native_codecs: PythonObject,
) raises -> PythonObject:
    var writer = RecordBatchStreamWriter(
        String(py=path),
        Schema(py=schema),
        _compression(compression),
        Bool(py=native_codecs),
    )
    return PythonObject(alloc=IpcStreamWriter(writer^))


def ipc_open_file(
    path: PythonObject, native_codecs: PythonObject
) raises -> PythonObject:
    var reader = RecordBatchFileReader(String(py=path), Bool(py=native_codecs))
    return PythonObject(alloc=IpcFileReader(reader^))


def ipc_open_stream(
    path: PythonObject, native_codecs: PythonObject
) raises -> PythonObject:
    var reader = RecordBatchStreamReader(
        String(py=path), Bool(py=native_codecs)
    )
    return PythonObject(alloc=IpcStreamReader(reader^))


def _file_writer_write(
    py_self: PythonObject, batch: PythonObject, custom_metadata: PythonObject
) raises -> PythonObject:
    py_self.downcast_value_ptr[IpcFileWriter]()[].writer.write(
        RecordBatch(py=batch), _from_dict(custom_metadata)
    )
    return Python.none()


def _file_writer_close(py_self: PythonObject) raises -> PythonObject:
    py_self.downcast_value_ptr[IpcFileWriter]()[].writer.close()
    return Python.none()


def _stream_writer_write(
    py_self: PythonObject, batch: PythonObject, custom_metadata: PythonObject
) raises -> PythonObject:
    py_self.downcast_value_ptr[IpcStreamWriter]()[].writer.write(
        RecordBatch(py=batch), _from_dict(custom_metadata)
    )
    return Python.none()


def _stream_writer_close(py_self: PythonObject) raises -> PythonObject:
    py_self.downcast_value_ptr[IpcStreamWriter]()[].writer.close()
    return Python.none()


def _file_reader_schema(py_self: PythonObject) raises -> PythonObject:
    return py_self.downcast_value_ptr[IpcFileReader]()[].reader.schema()


def _file_reader_num_record_batches(
    py_self: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        py_self.downcast_value_ptr[
            IpcFileReader
        ]()[].reader.num_record_batches()
    )


def _file_reader_metadata(py_self: PythonObject) raises -> PythonObject:
    return _to_dict(
        py_self.downcast_value_ptr[IpcFileReader]()[].reader.metadata()
    )


def _file_reader_get_batch(
    py_self: PythonObject, i: PythonObject
) raises -> PythonObject:
    var custom_metadata = Dict[String, String]()
    var batch = py_self.downcast_value_ptr[
        IpcFileReader
    ]()[].reader.read_batch_with_custom_metadata(Int(py=i), custom_metadata)
    return _batch_and_metadata(batch^, custom_metadata)


def _stream_reader_schema(py_self: PythonObject) raises -> PythonObject:
    return py_self.downcast_value_ptr[IpcStreamReader]()[].reader.schema()


def _stream_reader_read_next_batch(
    py_self: PythonObject,
) raises -> PythonObject:
    var custom_metadata = Dict[String, String]()
    var next = py_self.downcast_value_ptr[
        IpcStreamReader
    ]()[].reader.read_next_batch_with_custom_metadata(custom_metadata)
    var result = Python.none()
    if next:
        result = _batch_and_metadata(next.take(), custom_metadata)
    return result


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add IPC reader/writer functions to the Python module."""
    mb.def_function[read_ipc_file]("read_ipc_file")
    mb.def_function[write_ipc_file]("write_ipc_file")
    mb.def_function[read_ipc_stream]("read_ipc_stream")
    mb.def_function[read_ipc_file_schema]("read_ipc_file_schema")
    mb.def_function[read_ipc_stream_schema]("read_ipc_stream_schema")
    mb.def_function[write_ipc_stream]("write_ipc_stream")
    mb.def_function[ipc_new_file]("ipc_new_file")
    mb.def_function[ipc_new_stream]("ipc_new_stream")
    mb.def_function[ipc_open_file]("ipc_open_file")
    mb.def_function[ipc_open_stream]("ipc_open_stream")
    ref file_writer = mb.add_type[IpcFileWriter]("IpcFileWriter")
    _ = file_writer.def_method[_file_writer_write]("write").def_method[
        _file_writer_close
    ]("close")
    ref stream_writer = mb.add_type[IpcStreamWriter]("IpcStreamWriter")
    _ = stream_writer.def_method[_stream_writer_write]("write").def_method[
        _stream_writer_close
    ]("close")
    ref file_reader = mb.add_type[IpcFileReader]("IpcFileReader")
    _ = (
        file_reader.def_method[_file_reader_schema]("schema")
        .def_method[_file_reader_num_record_batches]("num_record_batches")
        .def_method[_file_reader_metadata]("metadata")
        .def_method[_file_reader_get_batch]("get_batch")
    )
    ref stream_reader = mb.add_type[IpcStreamReader]("IpcStreamReader")
    _ = stream_reader.def_method[_stream_reader_schema]("schema").def_method[
        _stream_reader_read_next_batch
    ]("read_next_batch")
