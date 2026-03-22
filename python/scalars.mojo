"""Python bindings for Arrow scalars.

Exposes AnyScalar as a Python type ``Scalar`` with rich comparison,
``as_py()``, ``is_valid()``, ``type()``, ``__str__``, ``__repr__``, and
``__bool__`` support.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.Scalar.html
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from pontoneer import TypeProtocolBuilder, RichCompareOps, NotImplementedError
from marrow.scalars import AnyScalar
from marrow.arrays import AnyArray
from marrow.dtypes import DataType, primitive_dtypes
from helpers import pymethod


# ---------------------------------------------------------------------------
# Helper: extract native Python value from an AnyScalar
# ---------------------------------------------------------------------------


def _as_py(scalar: AnyScalar) raises -> PythonObject:
    """Convert a Mojo AnyScalar to a native Python value (int, float, bool, str, None)."""
    if scalar.is_null():
        return PythonObject(None)
    var dtype = scalar.type()
    comptime for T in primitive_dtypes:
        if dtype == T:
            return PythonObject(scalar.as_primitive[T]().value())
    if dtype.is_string():
        return PythonObject(scalar.as_string().as_string())
    elif dtype.is_list():
        return scalar.as_list().value().to_python_object()
    elif dtype.is_fixed_size_list():
        return scalar.as_fixed_size_list().value().to_python_object()
    elif dtype.is_struct():
        ref s = scalar.as_struct()
        var builtins = Python.import_module("builtins")
        var d = builtins.dict()
        for i in range(s.num_fields()):
            d[dtype.fields[i].name] = _as_py(s.field(i))
        return d
    raise Error("as_py: unsupported dtype")


# ---------------------------------------------------------------------------
# Scalar methods exposed to Python
# ---------------------------------------------------------------------------


def _scalar_as_py(
    ptr: UnsafePointer[AnyScalar, MutAnyOrigin],
) raises -> PythonObject:
    return _as_py(ptr[])


def _scalar_str(
    ptr: UnsafePointer[AnyScalar, MutAnyOrigin],
) raises -> PythonObject:
    return PythonObject(String(ptr[]))


def _scalar_repr(
    ptr: UnsafePointer[AnyScalar, MutAnyOrigin],
) raises -> PythonObject:
    if ptr[].is_null():
        return PythonObject("<marrow.Scalar: null>")
    return PythonObject("<marrow.Scalar: " + String.write(ptr[]) + ">")


def _scalar_bool(
    ptr: UnsafePointer[AnyScalar, MutAnyOrigin],
) raises -> PythonObject:
    """Support bool(scalar) — needed for truthiness checks like ``assert arr[0]``."""
    var py_val = _as_py(ptr[])
    return PythonObject(Bool(py=py_val))


# ---------------------------------------------------------------------------
# Rich comparison — delegates to as_py() for Python-native comparison
# ---------------------------------------------------------------------------


def _scalar_rich_compare(
    first: AnyScalar, second: PythonObject, op: Int,
) raises -> Bool:
    var py_val = _as_py(first)
    var oper = Python.import_module("operator")
    if op == RichCompareOps.Py_EQ:
        return Bool(py=oper.eq(py_val, second))
    elif op == RichCompareOps.Py_NE:
        return Bool(py=oper.ne(py_val, second))
    elif op == RichCompareOps.Py_LT:
        return Bool(py=oper.lt(py_val, second))
    elif op == RichCompareOps.Py_LE:
        return Bool(py=oper.le(py_val, second))
    elif op == RichCompareOps.Py_GT:
        return Bool(py=oper.gt(py_val, second))
    elif op == RichCompareOps.Py_GE:
        return Bool(py=oper.ge(py_val, second))
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Register the Scalar Python type."""
    ref scalar_py = mb.add_type[AnyScalar]("Scalar")
    _ = (
        scalar_py.def_method[_scalar_as_py]("as_py")
        .def_method[pymethod[AnyScalar.is_valid]()]("is_valid")
        .def_method[pymethod[AnyScalar.is_null]()]("is_null")
        .def_method[pymethod[AnyScalar.type]()]("type")
        .def_method[_scalar_str]("__str__")
        .def_method[_scalar_repr]("__repr__")
        .def_method[_scalar_bool]("__bool__")
    )
    var scalar_tp = TypeProtocolBuilder[AnyScalar](scalar_py)
    _ = scalar_tp.def_richcompare[_scalar_rich_compare]()
