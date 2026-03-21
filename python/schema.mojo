from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.schema import Schema
from marrow.dtypes import Field
from marrow.c_data import CArrowSchema
from helpers import marrow_module


def _schema_arrow_c_schema(
    ptr: UnsafePointer[Schema, MutAnyOrigin]
) raises -> PythonObject:
    return CArrowSchema.from_schema(ptr[].fields).to_pycapsule()


def schema(fields_or_schema: PythonObject) raises -> PythonObject:
    """Create a Schema from a list of Fields, a marrow Schema, or any __arrow_c_schema__ object."""
    # Try converting directly (handles marrow Schema and __arrow_c_schema__).
    try:
        return Schema(py=fields_or_schema).to_python_object()
    except:
        pass
    # Fall back to treating as a list of marrow Field objects.
    var fields = List[Field]()
    for f in fields_or_schema:
        fields.append(f.downcast_value_ptr[Field]()[])
    return Schema(fields=fields^).to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add Schema type and constructor to the Python module."""
    ref schema_py = mb.add_type[Schema]("Schema")
    _ = schema_py.def_method[_schema_arrow_c_schema]("__arrow_c_schema__")
        .def_method[marrow_module]("__module__")

    mb.def_function[schema]("schema", docstring="Create an Arrow schema from a list of fields or any Arrow-compatible object.")
