from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.schema import Schema
from marrow.c_data import CArrowSchema
from helpers import pyinit


def _schema_arrow_c_schema(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Schema]()
    return CArrowSchema.from_schema(ptr[]).to_pycapsule()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add Schema type and constructor to the Python module."""
    _ = (
        mb.add_type[Schema]("Schema")
        .def_py_init[pyinit[Schema]]()
        .def_method[_schema_arrow_c_schema]("__arrow_c_schema__")
    )
