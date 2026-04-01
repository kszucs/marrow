"""Python interface for data types."""

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.memory import ArcPointer
import marrow.dtypes as dt
from helpers import marrow_module


def null() raises -> PythonObject:
    """Create a null DataType."""
    return dt.ArrowType(dt.null).to_python_object()


def bool_() raises -> PythonObject:
    """Create a boolean DataType."""
    return dt.ArrowType(dt.bool_).to_python_object()


def int8() raises -> PythonObject:
    """Create an int8 DataType."""
    return dt.ArrowType(dt.int8).to_python_object()


def int16() raises -> PythonObject:
    """Create an int16 DataType."""
    return dt.ArrowType(dt.int16).to_python_object()


def int32() raises -> PythonObject:
    """Create an int32 DataType."""
    return dt.ArrowType(dt.int32).to_python_object()


def int64() raises -> PythonObject:
    """Create an int64 DataType."""
    return dt.ArrowType(dt.int64).to_python_object()


def uint8() raises -> PythonObject:
    """Create a uint8 DataType."""
    return dt.ArrowType(dt.uint8).to_python_object()


def uint16() raises -> PythonObject:
    """Create a uint16 DataType."""
    return dt.ArrowType(dt.uint16).to_python_object()


def uint32() raises -> PythonObject:
    """Create a uint32 DataType."""
    return dt.ArrowType(dt.uint32).to_python_object()


def uint64() raises -> PythonObject:
    """Create a uint64 DataType."""
    return dt.ArrowType(dt.uint64).to_python_object()


def float16() raises -> PythonObject:
    """Create a float16 DataType."""
    return dt.ArrowType(dt.float16).to_python_object()


def float32() raises -> PythonObject:
    """Create a float32 DataType."""
    return dt.ArrowType(dt.float32).to_python_object()


def float64() raises -> PythonObject:
    """Create a float64 DataType."""
    return dt.ArrowType(dt.float64).to_python_object()


def string() raises -> PythonObject:
    """Create a string DataType."""
    return dt.ArrowType(dt.string).to_python_object()


def binary() raises -> PythonObject:
    """Create a binary DataType."""
    return dt.ArrowType(dt.binary).to_python_object()


def field(name: PythonObject, dtype: PythonObject) raises -> PythonObject:
    """Create a Field with the given name and data type."""
    var d = dtype.downcast_value_ptr[dt.ArrowType]()[]
    var f = dt.Field(String(py=name), ArcPointer(d^))
    return f.to_python_object()


def list_(value_type: PythonObject) raises -> PythonObject:
    """Create a list DataType from a value type."""
    var d = value_type.downcast_value_ptr[dt.ArrowType]()[]
    return dt.list_(d^).to_python_object()


def struct_(fields_obj: PythonObject) raises -> PythonObject:
    """Create a struct DataType from a list of Fields."""
    var fields = List[dt.Field]()
    for f in fields_obj:
        fields.append(f.downcast_value_ptr[dt.Field]()[])
    return dt.struct_(fields^).to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add DataType related data to the Python API."""

    _ = mb.add_type[dt.Field]("Field").def_method[marrow_module]("__module__")
    _ = mb.add_type[dt.ArrowType]("DataType").def_method[marrow_module]("__module__")

    mb.def_function[null]("null", docstring="null() -> DataType\n--\n\nCreate a null DataType.")
    mb.def_function[bool_]("bool_", docstring="bool_() -> DataType\n--\n\nCreate a boolean DataType.")
    mb.def_function[int8]("int8", docstring="int8() -> DataType\n--\n\nCreate an int8 DataType.")
    mb.def_function[int16]("int16", docstring="int16() -> DataType\n--\n\nCreate an int16 DataType.")
    mb.def_function[int32]("int32", docstring="int32() -> DataType\n--\n\nCreate an int32 DataType.")
    mb.def_function[int64]("int64", docstring="int64() -> DataType\n--\n\nCreate an int64 DataType.")
    mb.def_function[uint8]("uint8", docstring="uint8() -> DataType\n--\n\nCreate a uint8 DataType.")
    mb.def_function[uint16]("uint16", docstring="uint16() -> DataType\n--\n\nCreate a uint16 DataType.")
    mb.def_function[uint32]("uint32", docstring="uint32() -> DataType\n--\n\nCreate a uint32 DataType.")
    mb.def_function[uint64]("uint64", docstring="uint64() -> DataType\n--\n\nCreate a uint64 DataType.")
    mb.def_function[float16](
        "float16", docstring="float16() -> DataType\n--\n\nCreate a float16 DataType."
    )
    mb.def_function[float32](
        "float32", docstring="float32() -> DataType\n--\n\nCreate a float32 DataType."
    )
    mb.def_function[float64](
        "float64", docstring="float64() -> DataType\n--\n\nCreate a float64 DataType."
    )
    mb.def_function[string]("string", docstring="string() -> DataType\n--\n\nCreate a string DataType.")
    mb.def_function[binary]("binary", docstring="binary() -> DataType\n--\n\nCreate a binary DataType.")
    mb.def_function[field]("field", docstring="field(name: str, dtype: DataType, /) -> Field\n--\n\nCreate a Field with the given name and data type.")
    mb.def_function[list_]("list_", docstring="list_(value_type: DataType, /) -> DataType\n--\n\nCreate a list DataType from a value type.")
    mb.def_function[struct_]("struct", docstring="struct(fields: list[Field], /) -> DataType\n--\n\nCreate a struct DataType from a list of Fields.")
