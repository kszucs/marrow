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


# `is_valid` / `is_null` / `type` go through `pymethod` like any other zero-arg
# method. They used to be hand-wrapped: while `DynScalar` conformed to
# `ArrowScalar` all three were trait members declared on the trait *and*
# overridden here, and resolving that overload set crashed the compiler
# (`CallParamInf::inferForCall`, `mojo 1.0.0b3`). The conformance is gone, so
# there is one declaration each and inference resolves it.


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
        .def_method[pymethod[DynScalar.is_valid]()]("is_valid")
        .def_method[pymethod[DynScalar.is_null]()]("is_null")
        .def_method[pymethod[DynScalar.type]()]("type")
        .def_method[_scalar_str]("__str__")
        .def_method[_scalar_repr]("__repr__")
        .def_method[_scalar_bool]("__bool__")
    )
    # var scalar_tp = TypeProtocolBuilder[DynScalar](scalar_py)
    # _ = scalar_tp.def_richcompare[_scalar_rich_compare]()
