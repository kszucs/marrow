"""Python interface for dtypes."""

from std.python.bindings import PythonModuleBuilder, PythonObject

from marrow import dtypes


fn null() raises -> PythonObject:
    """Create a null DataType."""
    var result = materialize[dtypes.null]()
    return PythonObject(alloc=result^)


fn bool_() raises -> PythonObject:
    """Create a boolean DataType."""
    var result = materialize[dtypes.bool_]()
    return PythonObject(alloc=result^)


fn int8() raises -> PythonObject:
    """Create an int8 DataType."""
    var result = materialize[dtypes.int8]()
    return PythonObject(alloc=result^)


fn int16() raises -> PythonObject:
    """Create an int16 DataType."""
    var result = materialize[dtypes.int16]()
    return PythonObject(alloc=result^)


fn int32() raises -> PythonObject:
    """Create an int32 DataType."""
    var result = materialize[dtypes.int32]()
    return PythonObject(alloc=result^)


fn int64() raises -> PythonObject:
    """Create an int64 DataType."""
    var result = materialize[dtypes.int64]()
    return PythonObject(alloc=result^)


fn uint8() raises -> PythonObject:
    """Create a uint8 DataType."""
    var result = materialize[dtypes.uint8]()
    return PythonObject(alloc=result^)


fn uint16() raises -> PythonObject:
    """Create a uint16 DataType."""
    var result = materialize[dtypes.uint16]()
    return PythonObject(alloc=result^)


fn uint32() raises -> PythonObject:
    """Create a uint32 DataType."""
    var result = materialize[dtypes.uint32]()
    return PythonObject(alloc=result^)


fn uint64() raises -> PythonObject:
    """Create a uint64 DataType."""
    var result = materialize[dtypes.uint64]()
    return PythonObject(alloc=result^)


fn float16() raises -> PythonObject:
    """Create a float16 DataType."""
    var result = materialize[dtypes.float16]()
    return PythonObject(alloc=result^)


fn float32() raises -> PythonObject:
    """Create a float32 DataType."""
    var result = materialize[dtypes.float32]()
    return PythonObject(alloc=result^)


fn float64() raises -> PythonObject:
    """Create a float64 DataType."""
    var result = materialize[dtypes.float64]()
    return PythonObject(alloc=result^)


fn string() raises -> PythonObject:
    """Create a string DataType."""
    var result = materialize[dtypes.string]()
    return PythonObject(alloc=result^)


fn binary() raises -> PythonObject:
    """Create a binary DataType."""
    var result = materialize[dtypes.binary]()
    return PythonObject(alloc=result^)


def add_to_module(mut builder: PythonModuleBuilder) raises:
    """Add DataType related data to the Python API."""

    _ = builder.add_type[dtypes.DataType]("DataType")
    builder.def_function[null]("null", docstring="Create a null DataType.")
    builder.def_function[bool_]("bool_", docstring="Create a boolean DataType.")
    builder.def_function[int8]("int8", docstring="Create an int8 DataType.")
    builder.def_function[int16]("int16", docstring="Create an int16 DataType.")
    builder.def_function[int32]("int32", docstring="Create an int32 DataType.")
    builder.def_function[int64]("int64", docstring="Create an int64 DataType.")
    builder.def_function[uint8]("uint8", docstring="Create a uint8 DataType.")
    builder.def_function[uint16](
        "uint16", docstring="Create a uint16 DataType."
    )
    builder.def_function[uint32](
        "uint32", docstring="Create a uint32 DataType."
    )
    builder.def_function[uint64](
        "uint64", docstring="Create a uint64 DataType."
    )
    builder.def_function[float16](
        "float16", docstring="Create a float16 DataType."
    )
    builder.def_function[float32](
        "float32", docstring="Create a float32 DataType."
    )
    builder.def_function[float64](
        "float64", docstring="Create a float64 DataType."
    )
    builder.def_function[string](
        "string", docstring="Create a string DataType."
    )
    builder.def_function[binary](
        "binary", docstring="Create a binary DataType."
    )
