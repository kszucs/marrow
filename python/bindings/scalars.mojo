"""Python bindings for Arrow scalars.

Exposes DynScalar as a Python type ``Scalar`` with rich comparison,
``as_py()``, ``is_valid()``, ``type()``, ``__str__``, ``__repr__``, and
``__bool__`` support.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.Scalar.html
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from marrow.scalars import DynScalar
from marrow.arrays import DynArray
from marrow.dtypes import DynType
from helpers import pymethod


# ---------------------------------------------------------------------------
# Scalar methods exposed to Python
# ---------------------------------------------------------------------------


def _scalar_as_py(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[DynScalar]()
    return ptr[].as_py()


# `is_valid` / `is_null` / `type` are wrapped explicitly rather than through
# `pymethod[DynScalar.X]()`. Once `DynScalar` conforms to `ArrowScalar` those
# three are trait members -- declared on the trait and overridden here -- and
# resolving the resulting overload set crashes the compiler
# (`CallParamInf::inferForCall`, `mojo 1.0.0b3`). Naming each call explicitly
# sidesteps the inference. `pymethod` remains fine for non-trait methods.


def _scalar_type(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[DynScalar]()
    return ptr[].type().to_python_object()


def _scalar_is_valid(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[DynScalar]()
    return PythonObject(ptr[].is_valid())


def _scalar_is_null(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[DynScalar]()
    return PythonObject(ptr[].is_null())


def _scalar_str(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[DynScalar]()
    return PythonObject(String(ptr[]))


def _scalar_repr(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[DynScalar]()
    if ptr[].is_null():
        return PythonObject("<marrow.Scalar: null>")
    return PythonObject("<marrow.Scalar: " + String(ptr[]) + ">")


def _scalar_bool(py_self: PythonObject) raises -> PythonObject:
    """Support bool(scalar) — needed for truthiness checks like ``assert arr[0]``.
    """
    var ptr = py_self.downcast_value_ptr[DynScalar]()
    var py_val = ptr[].as_py()
    return PythonObject(Bool(py=py_val))


# ---------------------------------------------------------------------------
# Rich comparison — delegates to as_py() for Python-native comparison
# ---------------------------------------------------------------------------


# def _scalar_rich_compare(
#     first: DynScalar,
#     second: PythonObject,
#     op: Int,
# ) raises -> Bool:
#     var py_val = first.as_py()
#     var oper = Python.import_module("operator")
#     if op == RichCompareOps.Py_EQ:
#         return Bool(py=oper.eq(py_val, second))
#     elif op == RichCompareOps.Py_NE:
#         return Bool(py=oper.ne(py_val, second))
#     elif op == RichCompareOps.Py_LT:
#         return Bool(py=oper.lt(py_val, second))
#     elif op == RichCompareOps.Py_LE:
#         return Bool(py=oper.le(py_val, second))
#     elif op == RichCompareOps.Py_GT:
#         return Bool(py=oper.gt(py_val, second))
#     elif op == RichCompareOps.Py_GE:
#         return Bool(py=oper.ge(py_val, second))
#     raise NotImplementedError()


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Register the Scalar Python type."""
    ref scalar_py = mb.add_type[DynScalar]("Scalar")
    _ = (
        scalar_py.def_method[_scalar_as_py]("as_py")
        .def_method[_scalar_is_valid]("is_valid")
        .def_method[_scalar_is_null]("is_null")
        .def_method[_scalar_type]("type")
        .def_method[_scalar_str]("__str__")
        .def_method[_scalar_repr]("__repr__")
        .def_method[_scalar_bool]("__bool__")
    )
    # var scalar_tp = TypeProtocolBuilder[DynScalar](scalar_py)
    # _ = scalar_tp.def_richcompare[_scalar_rich_compare]()
