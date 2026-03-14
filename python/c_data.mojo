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
