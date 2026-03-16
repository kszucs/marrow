"""Python module entry point for marrow."""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from dtypes import add_to_module as add_dtypes
from arrays import add_to_module as add_arrays
from compute import add_to_module as add_compute
from schema import add_to_module as add_schema
from tabular import add_to_module as add_tabular


@export
fn PyInit_marrow() -> PythonObject:
    try:
        var m = PythonModuleBuilder("marrow")
        add_dtypes(m)
        add_arrays(m)
        add_compute(m)
        add_schema(m)
        add_tabular(m)
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))
