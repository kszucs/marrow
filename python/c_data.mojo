"""Python-aware Arrow C Data Interface bridge.

This module assembles PyCapsule tuples for the Arrow C Data Interface protocols.
It delegates capsule creation to CArrowSchema.to_pycapsule() and
CArrowArray.to_pycapsule() (defined in marrow.c_data) and handles import from
Python objects for interoperability with any Arrow-compatible Python library.

References:
  https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html
"""
from std.python import Python, PythonObject
from std.python._cpython import CPython
from std.memory import alloc
from marrow.c_data import (
    CArrowArray,
    CArrowSchema,
    ArrowArrayStream,
    CArrowArrayStream,
)
from marrow.arrays import Array
from marrow.dtypes import DataType


# ---------------------------------------------------------------------------
# Export: Mojo Array → Python capsules
# ---------------------------------------------------------------------------


fn array_to_capsule_tuple(arr: Array) raises -> PythonObject:
    """Return (schema_capsule, array_capsule) for __arrow_c_array__ protocol.

    The caller owns the returned Python tuple.  Each capsule owns the
    corresponding heap-allocated C struct; the capsule destructor calls the
    struct's release callback when Python GC collects it.
    """
    var py = Python()
    ref cpy = py.cpython()
    var schema_cap = CArrowSchema.from_dtype(arr.dtype).to_pycapsule().steal_data()
    var array_cap = CArrowArray.from_array(arr).to_pycapsule().steal_data()
    var tup = cpy.PyTuple_New(2)
    _ = cpy.PyTuple_SetItem(tup, 0, schema_cap)
    _ = cpy.PyTuple_SetItem(tup, 1, array_cap)
    return PythonObject(from_owned=tup)


fn schema_to_capsule(dtype: DataType) raises -> PythonObject:
    """Return a schema_capsule for the __arrow_c_schema__ protocol."""
    return CArrowSchema.from_dtype(dtype).to_pycapsule()


# ---------------------------------------------------------------------------
# Import: Python capsules → Mojo Array   (ArrowArrayMove semantics)
#
# After calling PyCapsule_GetPointer, we copy the structs then zero the
# release fields on the source.  When Python GC collects the capsule its
# destructor sees release == NULL and skips the double-free.
# ---------------------------------------------------------------------------


fn array_from_capsule_tuple(capsule_tuple: PythonObject) raises -> Array:
    """Consume a (schema_capsule, array_capsule) tuple, taking ownership.

    The capsule_tuple argument must stay alive for the duration of this call
    so that the capsule Python objects are not collected mid-call.
    """
    var py = Python()
    ref cpy = py.cpython()
    var schema_raw = cpy.PyCapsule_GetPointer(
        capsule_tuple[0]._obj_ptr, "arrow_schema"
    )
    var array_raw = cpy.PyCapsule_GetPointer(
        capsule_tuple[1]._obj_ptr, "arrow_array"
    )
    var c_schema_src = schema_raw.bitcast[CArrowSchema]()
    var c_array_src = array_raw.bitcast[CArrowArray]()
    # Copy the structs (ArrowArrayMove semantics).
    var c_schema = c_schema_src[]
    var c_array = c_array_src[]
    # Zero release on sources to signal ownership transfer.
    UnsafePointer(to=c_schema_src[].release).bitcast[UInt64]()[0] = 0
    UnsafePointer(to=c_array_src[].release).bitcast[UInt64]()[0] = 0
    return c_array^.to_array(c_schema.to_dtype())


# ---------------------------------------------------------------------------
# Stream helper (previously ArrowArrayStream.from_pyarrow)
# ---------------------------------------------------------------------------


fn arrow_array_stream_from_python(
    pyobj: PythonObject, cpython: CPython
) raises -> ArrowArrayStream:
    """Create an ArrowArrayStream from a Python object supporting __arrow_c_stream__."""
    var stream = pyobj.__arrow_c_stream__()
    var ptr = cpython.PyCapsule_GetPointer(
        stream.steal_data(), "arrow_array_stream"
    )
    if not ptr:
        raise Error("Failed to get the arrow array stream pointer")
    return ArrowArrayStream(ptr.bitcast[CArrowArrayStream]())
