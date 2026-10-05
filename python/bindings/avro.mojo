# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Python bindings for the Avro object container reader/writer.

Thin marshaling over `marrow.avro.read_avro` / `write_avro`: the friendly API
lives in pure Python; these entry points stay strict.
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from marrow.avro import AvroCodec, read_avro, write_avro
from marrow.tabular import Table


def avro_read_table(
    path: PythonObject, columns: PythonObject
) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    if columns.__is__(builtins.None):
        return read_avro(String(py=path)).to_python_object()
    var cols = List[String]()
    for i in range(Int(py=columns.__len__())):
        cols.append(String(py=columns[i]))
    return read_avro(String(py=path), cols^).to_python_object()


def avro_write_table(
    table: PythonObject, path: PythonObject, codec: PythonObject
) raises -> PythonObject:
    write_avro(
        Table(py=table), String(py=path), AvroCodec.parse(String(py=codec))
    )
    return Python.evaluate("None")


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add the Avro reader/writer functions to the Python module."""
    mb.def_function[avro_read_table]("avro_read_table")
    mb.def_function[avro_write_table]("avro_write_table")
