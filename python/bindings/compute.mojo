"""Free-standing compute functions exposed to Python.

All GPU-capable functions accept an ``ExecContext`` as their last positional argument.
"""

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.arrays import DynArray
from marrow.dtypes import DynType
import marrow.kernels as mk

# ``mk.filter`` collides with the like-named submodule, so the package alias
# resolves to the submodule rather than the function. Import it directly.
from marrow.kernels.filter import filter as _filter_kernel
from marrow.kernels.boolean import IsNullKernel, NotNullKernel
from marrow.execution import ExecContext

from helpers import pyfunction


# ---------------------------------------------------------------------------
# ExecContext factory functions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# sort_indices / sort — composite kernels built from sort_indices + take
# ---------------------------------------------------------------------------


def sort_indices(
    array: DynArray, ascending: Bool, nulls_first: Bool, ctx: ExecContext
) raises -> DynArray:
    return mk.sort_indices(array, ascending, nulls_first, ctx=ctx)


def sort(
    array: DynArray, ascending: Bool, nulls_first: Bool, ctx: ExecContext
) raises -> DynArray:
    var indices = mk.sort_indices(array, ascending, nulls_first, ctx=ctx)
    return mk.take(array, indices, ctx)


def take(
    array: DynArray, indices: DynArray, ctx: ExecContext
) raises -> DynArray:
    return mk.take(array, indices.as_int32().copy(), ctx)


def cast(
    array: DynArray, target: DynType, safe: Bool, ctx: ExecContext
) raises -> DynArray:
    return mk.cast(array, target, safe, ctx)


# ``pykernel`` — wrap a marrow kernel of uniform shape
# ``(DynArray..., ExecContext) -> R`` as a Python-callable. Each overload
# pins the full signature so Mojo can resolve the DynArray runtime overload
# from a kernel reference like ``mk.equal`` even when the kernel has
# additional parametric or concrete-type overloads.


def pykernel[
    func: def(DynArray, ExecContext) raises thin -> Bool,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    return pyfunction[func]()


def pykernel[
    func: def(DynArray, ExecContext) raises thin -> DynArray,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    return pyfunction[func]()


def pykernel[
    func: def(DynArray, DynArray, ExecContext) raises thin -> DynArray,
]() -> def(
    PythonObject, PythonObject, PythonObject
) raises thin -> PythonObject:
    return pyfunction[func]()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    _ = (
        mb.add_type[ExecContext]("ExecContext")
        .def_staticmethod[pyfunction[ExecContext.serial]()]("serial")
        .def_staticmethod[pyfunction[ExecContext.parallel]()]("parallel")
    )
    mb.def_function[pykernel[mk.AddKernel.dispatch]()]("add")
    mb.def_function[pykernel[mk.SubKernel.dispatch]()]("subtract")
    mb.def_function[pykernel[mk.MulKernel.dispatch]()]("multiply")
    mb.def_function[pykernel[mk.DivKernel.dispatch]()]("divide")
    mb.def_function[pykernel[mk.AnyKernel.dispatch]()]("any")
    mb.def_function[pykernel[mk.AllKernel.dispatch]()]("all")
    mb.def_function[pykernel[IsNullKernel.dispatch]()]("is_null")
    mb.def_function[pykernel[NotNullKernel.dispatch]()]("is_valid")
    mb.def_function[pykernel[mk.drop_null]()]("drop_null")
    mb.def_function[pykernel[_filter_kernel]()]("filter")
    mb.def_function[pykernel[mk.EqKernel.dispatch]()]("equal")
    mb.def_function[pykernel[mk.NeKernel.dispatch]()]("not_equal")
    mb.def_function[pykernel[mk.LtKernel.dispatch]()]("less")
    mb.def_function[pykernel[mk.LeKernel.dispatch]()]("less_equal")
    mb.def_function[pykernel[mk.GtKernel.dispatch]()]("greater")
    mb.def_function[pykernel[mk.GeKernel.dispatch]()]("greater_equal")
    mb.def_function[pyfunction[sort_indices]()]("sort_indices")
    mb.def_function[pyfunction[sort]()]("sort")
    mb.def_function[pyfunction[take]()]("take")
    mb.def_function[pyfunction[cast]()]("cast")
