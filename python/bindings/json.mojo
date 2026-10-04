# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Python bindings for the NDJSON reader, `marrow.json`.

Thin marshaling and strict: every argument is required and `None` means
"none". The pyarrow-shaped surface — `ReadOptions`, `ParseOptions`, defaults —
is `python/marrow/json.py`.
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from marrow.io import DynSource
from marrow.schema import Schema
from marrow.json import (
    JsonReader,
    ParseOptions,
    ReadOptions,
    UnexpectedFieldBehavior,
    open_json,
    read_json as _read_json,
    write_json as _write_json,
)
from marrow.tabular import Table


def _options(
    schema: PythonObject, behavior: PythonObject
) raises -> ParseOptions:
    var builtins = Python.import_module("builtins")
    if schema.__is__(builtins.None):
        return ParseOptions()
    return ParseOptions(
        Schema(py=schema), UnexpectedFieldBehavior.parse(String(py=behavior))
    )


struct JsonStreamReader(Movable, Writable):
    """The registered type behind `marrow.json.open_json` — a `JsonReader` in
    a box that spells its own `repr`, so `add_type` does not derive one by
    walking the reader."""

    var reader: JsonReader[DynSource]

    def __init__(out self, var reader: JsonReader[DynSource]):
        self.reader = reader^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("JsonStreamReader")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<marrow.JsonStreamReader>")


def json_read_table(
    path: PythonObject,
    schema: PythonObject,
    behavior: PythonObject,
    block_size: PythonObject,
) raises -> PythonObject:
    return _read_json(
        String(py=path),
        ReadOptions(Int(py=block_size)),
        _options(schema, behavior),
    ).to_python_object()


def json_open(
    path: PythonObject,
    schema: PythonObject,
    behavior: PythonObject,
    block_size: PythonObject,
) raises -> PythonObject:
    var reader = open_json(
        String(py=path),
        ReadOptions(Int(py=block_size)),
        _options(schema, behavior),
    )
    return PythonObject(alloc=JsonStreamReader(reader^))


def _stream_schema(py_self: PythonObject) raises -> PythonObject:
    return (
        py_self.downcast_value_ptr[JsonStreamReader]()[]
        .reader.schema.copy()
        .to_python_object()
    )


def _stream_read_next_batch(py_self: PythonObject) raises -> PythonObject:
    var batch = py_self.downcast_value_ptr[
        JsonStreamReader
    ]()[].reader.read_next_batch()
    if not batch:
        return Python.evaluate("None")
    return batch.take().to_python_object()


def json_write_table(
    table: PythonObject, path: PythonObject
) raises -> PythonObject:
    _write_json(Table(py=table), String(py=path))
    return Python.evaluate("None")


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Register the NDJSON reader — one eager read, one streaming reader — and
    the writer."""
    mb.def_function[json_read_table]("json_read_table")
    mb.def_function[json_open]("json_open")
    mb.def_function[json_write_table]("json_write_table")
    ref stream_py = mb.add_type[JsonStreamReader]("JsonStreamReader")
    _ = stream_py.def_method[_stream_schema]("schema").def_method[
        _stream_read_next_batch
    ]("read_next_batch")
