from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.schema import Schema
from marrow.c_data import CArrowSchema


def _schema_init(
    out self: Schema, args: PythonObject, kwargs: PythonObject
) raises:
    """``Schema(obj)`` — anything `Schema.__init__(py=...)` accepts."""
    self = Schema(py=args[0])


def _schema_arrow_c_schema(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Schema]()
    return CArrowSchema.from_schema(ptr[]).to_pycapsule()


def _schema_len(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(len(py_self.downcast_value_ptr[Schema]()[]))


def _schema_names(py_self: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    for ref name in py_self.downcast_value_ptr[Schema]()[].names():
        _ = out.append(PythonObject(name.copy()))
    return out


def _schema_types(py_self: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    for ref f in py_self.downcast_value_ptr[Schema]()[].fields:
        _ = out.append(f.dtype.copy().to_python_object())
    return out


def _schema_field(
    py_self: PythonObject, key: PythonObject
) raises -> PythonObject:
    """``schema.field(0)`` or ``schema.field("a")`` — index or name."""
    var ptr = py_self.downcast_value_ptr[Schema]()
    var builtins = Python.import_module("builtins")
    if Bool(py=builtins.isinstance(key, builtins.int)):
        return ptr[].field(index=Int(py=key)).copy().to_python_object()
    return ptr[].field(name=String(py=key)).copy().to_python_object()


def _schema_get_field_index(
    py_self: PythonObject, name: PythonObject
) raises -> PythonObject:
    """The field's position, or -1 — the same sentinel Mojo answers with."""
    var ptr = py_self.downcast_value_ptr[Schema]()
    return PythonObject(ptr[].get_field_index(String(py=name)))


def _schema_equals(
    py_self: PythonObject, other: PythonObject
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[Schema]()
    return PythonObject(ptr[] == other.downcast_value_ptr[Schema]()[])


def _schema_str(py_self: PythonObject) raises -> PythonObject:
    return PythonObject(String(py_self.downcast_value_ptr[Schema]()[]))


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add Schema type and constructor to the Python module."""
    _ = (
        mb.add_type[Schema]("Schema")
        .def_py_init[_schema_init]()
        .def_method[_schema_arrow_c_schema]("__arrow_c_schema__")
        .def_method[_schema_len]("__len__")
        .def_method[_schema_names]("names")
        .def_method[_schema_types]("types")
        .def_method[_schema_field]("field")
        .def_method[_schema_get_field_index]("get_field_index")
        .def_method[_schema_equals]("equals")
        .def_method[_schema_str]("__str__")
    )
