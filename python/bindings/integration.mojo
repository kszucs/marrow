# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Python bindings for the Arrow integration JSON reader, `marrow.integration`.
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from marrow.c_data import live_allocations
from marrow.integration import IntegrationJson


def read_integration_json(path: PythonObject) raises -> PythonObject:
    """`(schema, batches)`."""
    var parsed = IntegrationJson.read(String(py=path))
    var builtins = Python.import_module("builtins")
    var batches = builtins.list()
    for batch in parsed.batches:
        batches.append(batch.copy().to_python_object())
    return builtins.tuple([parsed.schema.copy().to_python_object(), batches])


def c_data_allocations() raises -> PythonObject:
    """Heap blocks C Data structs still hold; see `live_allocations`."""
    return PythonObject(live_allocations())


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add the integration JSON reader to the Python module."""
    mb.def_function[read_integration_json]("read_integration_json")
    mb.def_function[c_data_allocations]("c_data_allocations")
