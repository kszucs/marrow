"""Python module entry point for marrow."""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from dtypes import add_to_module as add_dtypes
from arrays import add_to_module as add_arrays
from scalars import add_to_module as add_scalars
from compute import add_to_module as add_compute
from schema import add_to_module as add_schema
from tabular import add_to_module as add_tabular
from ipc import add_to_module as add_ipc
from parquet import add_to_module as add_parquet


@export
def PyInit_libmarrow() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("libmarrow")
        add_dtypes(m)
        add_scalars(m)
        add_arrays(m)
        add_compute(m)
        add_schema(m)
        add_tabular(m)
        add_ipc(m)
        add_parquet(m)
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))
