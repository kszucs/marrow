from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.schema import Schema
from marrow.dtypes import Field
from marrow.c_data import CArrowSchema


def _schema_arrow_c_schema(
    ptr: UnsafePointer[Schema, MutAnyOrigin]
) raises -> PythonObject:
    return CArrowSchema.from_schema(ptr[]).to_pycapsule()


def schema(
    fields_or_schema: PythonObject, metadata: PythonObject
) raises -> PythonObject:
    """Create a Schema from a list of Fields, a marrow Schema, or any __arrow_c_schema__ object.
    """
    var s: Schema
    try:
        s = Schema(py=fields_or_schema)
    except:
        var fields = List[Field]()
        for f in fields_or_schema:
            fields.append(f.downcast_value_ptr[Field]()[].copy())
        s = Schema(fields=fields^)
    var builtins = Python.import_module("builtins")
    if not metadata.__is__(builtins.None):
        for key in metadata.keys():
            s.metadata[String(py=key)] = String(py=metadata[key])
    return s.to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add Schema type and constructor to the Python module."""
    ref schema_py = mb.add_type[Schema]("Schema")
    _ = schema_py.def_method[_schema_arrow_c_schema]("__arrow_c_schema__")

    mb.def_function[schema]("schema")
