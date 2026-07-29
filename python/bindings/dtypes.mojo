"""Python interface for data types."""

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.memory import ArcPointer
import marrow.dtypes as dt


def _field_name(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[dt.Field]()
    return PythonObject(ptr[].name)


def _field_type(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[dt.Field]()
    return ptr[].dtype.copy().to_python_object()


def null() raises -> PythonObject:
    """Create a null DataType."""
    return dt.null.to_dyn().to_python_object()


def bool_() raises -> PythonObject:
    """Create a boolean DataType."""
    return dt.bool_.to_dyn().to_python_object()


def int8() raises -> PythonObject:
    """Create an int8 DataType."""
    return dt.int8.to_dyn().to_python_object()


def int16() raises -> PythonObject:
    """Create an int16 DataType."""
    return dt.int16.to_dyn().to_python_object()


def int32() raises -> PythonObject:
    """Create an int32 DataType."""
    return dt.int32.to_dyn().to_python_object()


def int64() raises -> PythonObject:
    """Create an int64 DataType."""
    return dt.int64.to_dyn().to_python_object()


def uint8() raises -> PythonObject:
    """Create a uint8 DataType."""
    return dt.uint8.to_dyn().to_python_object()


def uint16() raises -> PythonObject:
    """Create a uint16 DataType."""
    return dt.uint16.to_dyn().to_python_object()


def uint32() raises -> PythonObject:
    """Create a uint32 DataType."""
    return dt.uint32.to_dyn().to_python_object()


def uint64() raises -> PythonObject:
    """Create a uint64 DataType."""
    return dt.uint64.to_dyn().to_python_object()


def float16() raises -> PythonObject:
    """Create a float16 DataType."""
    return dt.float16.to_dyn().to_python_object()


def float32() raises -> PythonObject:
    """Create a float32 DataType."""
    return dt.float32.to_dyn().to_python_object()


def float64() raises -> PythonObject:
    """Create a float64 DataType."""
    return dt.float64.to_dyn().to_python_object()


def string() raises -> PythonObject:
    """Create a string DataType."""
    return dt.string.to_dyn().to_python_object()


def binary() raises -> PythonObject:
    """Create a binary DataType."""
    return dt.binary.to_dyn().to_python_object()


def fixed_size_binary(byte_width: PythonObject) raises -> PythonObject:
    """Create a fixed-size binary DataType."""
    return dt.fixed_size_binary_(Int(py=byte_width)).to_dyn().to_python_object()


def date32() raises -> PythonObject:
    """Create a date32 DataType."""
    return dt.date32().to_dyn().to_python_object()


def date64() raises -> PythonObject:
    """Create a date64 DataType."""
    return dt.date64().to_dyn().to_python_object()


def time32(unit: PythonObject) raises -> PythonObject:
    """Create a time32 DataType."""
    return dt.time32(_parse_time_unit(unit)).to_dyn().to_python_object()


def time64(unit: PythonObject) raises -> PythonObject:
    """Create a time64 DataType."""
    return dt.time64(_parse_time_unit(unit)).to_dyn().to_python_object()


def timestamp(
    unit: PythonObject, tz: PythonObject = None
) raises -> PythonObject:
    """Create a timestamp DataType."""
    var tz_str = "" if tz is None else String(py=tz)
    return (
        dt.timestamp(_parse_time_unit(unit), tz_str).to_dyn().to_python_object()
    )


def duration(unit: PythonObject) raises -> PythonObject:
    """Create a duration DataType."""
    return dt.duration(_parse_time_unit(unit)).to_dyn().to_python_object()


def year_month_interval() raises -> PythonObject:
    """Create a year_month_interval DataType."""
    return dt.year_month_interval().to_dyn().to_python_object()


def day_time_interval() raises -> PythonObject:
    """Create a day_time_interval DataType."""
    return dt.day_time_interval().to_dyn().to_python_object()


def month_day_nano_interval() raises -> PythonObject:
    """Create a month_day_nano_interval DataType."""
    return dt.month_day_nano_interval().to_dyn().to_python_object()


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
        dtype=dt.DynType(py=type),
        nullable=Bool(py=nullable),
        metadata=m^,
    ).to_python_object()


def list_(value_type: PythonObject) raises -> PythonObject:
    """Create a list DataType from a value type."""
    var d = dt.DynType(py=value_type)
    return dt.list_(d^).to_dyn().to_python_object()


def fixed_size_list_(
    value_type: PythonObject, list_size: PythonObject
) raises -> PythonObject:
    """Create a fixed-size list DataType from a value type and list size."""
    var d = dt.DynType(py=value_type)
    return (
        dt.fixed_size_list_(d^, Int(py=list_size)).to_dyn().to_python_object()
    )


def struct_(fields_obj: PythonObject) raises -> PythonObject:
    """Create a struct DataType from a list of Fields."""
    var fields = List[dt.Field]()
    for f in fields_obj:
        fields.append(dt.Field(py=f))
    return dt.struct_(fields^).to_dyn().to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add DataType related data to the Python API."""

    _ = (
        mb.add_type[dt.Field]("Field")
        .def_method[_field_name]("name")
        .def_method[_field_type]("type")
    )
    _ = mb.add_type[dt.DynType]("DataType")

    mb.def_function[null]("null")
    mb.def_function[bool_]("bool_")
    mb.def_function[int8]("int8")
    mb.def_function[int16]("int16")
    mb.def_function[int32]("int32")
    mb.def_function[int64]("int64")
    mb.def_function[uint8]("uint8")
    mb.def_function[uint16]("uint16")
    mb.def_function[uint32]("uint32")
    mb.def_function[uint64]("uint64")
    mb.def_function[float16]("float16")
    mb.def_function[float32]("float32")
    mb.def_function[float64]("float64")
    mb.def_function[string]("string")
    mb.def_function[binary]("binary")
    mb.def_function[fixed_size_binary]("fixed_size_binary")
    mb.def_function[field]("field")
    mb.def_function[list_]("list_")
    mb.def_function[fixed_size_list_]("fixed_size_list_")
    mb.def_function[struct_]("struct")
    mb.def_function[date32]("date32")
    mb.def_function[date64]("date64")
    mb.def_function[time32]("time32")
    mb.def_function[time64]("time64")
    mb.def_function[timestamp]("timestamp")
    mb.def_function[duration]("duration")
    mb.def_function[year_month_interval]("year_month_interval")
    mb.def_function[day_time_interval]("day_time_interval")
    mb.def_function[month_day_nano_interval]("month_day_nano_interval")
