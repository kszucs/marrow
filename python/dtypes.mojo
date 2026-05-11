"""Python interface for data types."""

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.collections import OwnedKwargsDict
from std.memory import ArcPointer
import marrow.dtypes as dt
from helpers import marrow_module


def _field_name(
    ptr: UnsafePointer[dt.Field, MutAnyOrigin]
) raises -> PythonObject:
    return PythonObject(ptr[].name)


def _field_type(
    ptr: UnsafePointer[dt.Field, MutAnyOrigin]
) raises -> PythonObject:
    return ptr[].dtype.copy().to_python_object()


def null() raises -> PythonObject:
    """Create a null DataType."""
    return dt.null.to_any().to_python_object()


def bool_() raises -> PythonObject:
    """Create a boolean DataType."""
    return dt.bool_.to_any().to_python_object()


def int8() raises -> PythonObject:
    """Create an int8 DataType."""
    return dt.int8.to_any().to_python_object()


def int16() raises -> PythonObject:
    """Create an int16 DataType."""
    return dt.int16.to_any().to_python_object()


def int32() raises -> PythonObject:
    """Create an int32 DataType."""
    return dt.int32.to_any().to_python_object()


def int64() raises -> PythonObject:
    """Create an int64 DataType."""
    return dt.int64.to_any().to_python_object()


def uint8() raises -> PythonObject:
    """Create a uint8 DataType."""
    return dt.uint8.to_any().to_python_object()


def uint16() raises -> PythonObject:
    """Create a uint16 DataType."""
    return dt.uint16.to_any().to_python_object()


def uint32() raises -> PythonObject:
    """Create a uint32 DataType."""
    return dt.uint32.to_any().to_python_object()


def uint64() raises -> PythonObject:
    """Create a uint64 DataType."""
    return dt.uint64.to_any().to_python_object()


def float16() raises -> PythonObject:
    """Create a float16 DataType."""
    return dt.float16.to_any().to_python_object()


def float32() raises -> PythonObject:
    """Create a float32 DataType."""
    return dt.float32.to_any().to_python_object()


def float64() raises -> PythonObject:
    """Create a float64 DataType."""
    return dt.float64.to_any().to_python_object()


def string() raises -> PythonObject:
    """Create a string DataType."""
    return dt.string.to_any().to_python_object()


def binary() raises -> PythonObject:
    """Create a binary DataType."""
    return dt.binary.to_any().to_python_object()


def fixed_size_binary(byte_width: PythonObject) raises -> PythonObject:
    """Create a fixed-size binary DataType."""
    return dt.fixed_size_binary_(Int(py=byte_width)).to_any().to_python_object()


def date32() raises -> PythonObject:
    """Create a date32 DataType."""
    return dt.date32().to_any().to_python_object()


def date64() raises -> PythonObject:
    """Create a date64 DataType."""
    return dt.date64().to_any().to_python_object()


def time32(unit: PythonObject) raises -> PythonObject:
    """Create a time32 DataType."""
    return dt.time32(_parse_time_unit(unit)).to_any().to_python_object()


def time64(unit: PythonObject) raises -> PythonObject:
    """Create a time64 DataType."""
    return dt.time64(_parse_time_unit(unit)).to_any().to_python_object()


def timestamp(
    unit: PythonObject, tz: PythonObject = None
) raises -> PythonObject:
    """Create a timestamp DataType."""
    var tz_str = "" if tz is None else String(py=tz)
    return (
        dt.timestamp(_parse_time_unit(unit), tz_str).to_any().to_python_object()
    )


def duration(unit: PythonObject) raises -> PythonObject:
    """Create a duration DataType."""
    return dt.duration(_parse_time_unit(unit)).to_any().to_python_object()


def _parse_time_unit(unit: PythonObject) raises -> dt.TimeUnit:
    var s = String(py=unit)
    if s == "s":
        return dt.second
    elif s == "ms":
        return dt.millisecond
    elif s == "us":
        return dt.microsecond
    elif s == "ns":
        return dt.nanosecond
    else:
        raise Error("Unknown time unit: " + s)


def field(
    name: PythonObject,
    type: PythonObject,
    nullable: PythonObject = True,
    metadata: PythonObject = None,
) raises -> PythonObject:
    """Create a Field with the given name, data type, and optional nullability.

    Python API: field(name, *, type, nullable=True)
    """
    var m = Dict[String, String]()
    if metadata is not None:
        for item in metadata.items():
            m[String(py=item[0])] = String(py=item[1])

    return dt.Field(
        name=String(py=name),
        dtype=dt.AnyDataType(py=type),
        nullable=Bool(py=nullable),
        metadata=m^,
    ).to_python_object()


def list_(value_type: PythonObject) raises -> PythonObject:
    """Create a list DataType from a value type."""
    var d = dt.AnyDataType(py=value_type)
    return dt.list_(d^).to_any().to_python_object()


def fixed_size_list_(
    value_type: PythonObject, list_size: PythonObject
) raises -> PythonObject:
    """Create a fixed-size list DataType from a value type and list size."""
    var d = dt.AnyDataType(py=value_type)
    return (
        dt.fixed_size_list_(d^, Int(py=list_size)).to_any().to_python_object()
    )


def struct_(fields_obj: PythonObject) raises -> PythonObject:
    """Create a struct DataType from a list of Fields."""
    var fields = List[dt.Field]()
    for f in fields_obj:
        fields.append(dt.Field(py=f))
    return dt.struct_(fields^).to_any().to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add DataType related data to the Python API."""

    _ = (
        mb.add_type[dt.Field]("Field")
        .def_method[marrow_module]("__module__")
        .def_method[_field_name]("name")
        .def_method[_field_type]("type")
    )
    _ = mb.add_type[dt.AnyDataType]("DataType").def_method[marrow_module](
        "__module__"
    )

    mb.def_function[null](
        "null", docstring="null() -> DataType\n--\n\nCreate a null DataType."
    )
    mb.def_function[bool_](
        "bool_",
        docstring="bool_() -> DataType\n--\n\nCreate a boolean DataType.",
    )
    mb.def_function[int8](
        "int8", docstring="int8() -> DataType\n--\n\nCreate an int8 DataType."
    )
    mb.def_function[int16](
        "int16",
        docstring="int16() -> DataType\n--\n\nCreate an int16 DataType.",
    )
    mb.def_function[int32](
        "int32",
        docstring="int32() -> DataType\n--\n\nCreate an int32 DataType.",
    )
    mb.def_function[int64](
        "int64",
        docstring="int64() -> DataType\n--\n\nCreate an int64 DataType.",
    )
    mb.def_function[uint8](
        "uint8", docstring="uint8() -> DataType\n--\n\nCreate a uint8 DataType."
    )
    mb.def_function[uint16](
        "uint16",
        docstring="uint16() -> DataType\n--\n\nCreate a uint16 DataType.",
    )
    mb.def_function[uint32](
        "uint32",
        docstring="uint32() -> DataType\n--\n\nCreate a uint32 DataType.",
    )
    mb.def_function[uint64](
        "uint64",
        docstring="uint64() -> DataType\n--\n\nCreate a uint64 DataType.",
    )
    mb.def_function[float16](
        "float16",
        docstring="float16() -> DataType\n--\n\nCreate a float16 DataType.",
    )
    mb.def_function[float32](
        "float32",
        docstring="float32() -> DataType\n--\n\nCreate a float32 DataType.",
    )
    mb.def_function[float64](
        "float64",
        docstring="float64() -> DataType\n--\n\nCreate a float64 DataType.",
    )
    mb.def_function[string](
        "string",
        docstring="string() -> DataType\n--\n\nCreate a string DataType.",
    )
    mb.def_function[binary](
        "binary",
        docstring="binary() -> DataType\n--\n\nCreate a binary DataType.",
    )
    mb.def_function[fixed_size_binary](
        "fixed_size_binary",
        docstring=(
            "fixed_size_binary(byte_width: int, /) -> DataType\n--\n\nCreate a"
            " fixed-size binary DataType."
        ),
    )
    mb.def_function[field](
        "field",
        docstring=(
            "field(name: str, /, *, type: DataType, nullable: bool = True) ->"
            " Field\n--\n\nCreate a Field with the given name, data type, and"
            " nullability."
        ),
    )
    mb.def_function[list_](
        "list_",
        docstring=(
            "list_(value_type: DataType, /) -> DataType\n--\n\nCreate a list"
            " DataType from a value type."
        ),
    )
    mb.def_function[fixed_size_list_](
        "fixed_size_list_",
        docstring=(
            "fixed_size_list_(value_type: DataType, list_size: int, /) ->"
            " DataType\n--\n\nCreate a fixed-size list DataType from a value"
            " type and list size."
        ),
    )
    mb.def_function[struct_](
        "struct",
        docstring=(
            "struct(fields: list[Field], /) -> DataType\n--\n\nCreate a struct"
            " DataType from a list of Fields."
        ),
    )
    mb.def_function[date32](
        "date32",
        docstring="date32() -> DataType\n--\n\nCreate a date32 DataType.",
    )
    mb.def_function[date64](
        "date64",
        docstring="date64() -> DataType\n--\n\nCreate a date64 DataType.",
    )
    mb.def_function[time32](
        "time32",
        docstring=(
            "time32(unit: str, /) -> DataType\n--\n\nCreate a time32 DataType."
        ),
    )
    mb.def_function[time64](
        "time64",
        docstring=(
            "time64(unit: str, /) -> DataType\n--\n\nCreate a time64 DataType."
        ),
    )
    mb.def_function[timestamp](
        "timestamp",
        docstring=(
            "timestamp(unit: str, tz: str | None = None, /) -> DataType\n--\n\n"
            "Create a timestamp DataType."
        ),
    )
    mb.def_function[duration](
        "duration",
        docstring=(
            "duration(unit: str, /) -> DataType\n--\n\nCreate a duration"
            " DataType."
        ),
    )
