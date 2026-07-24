"""Free-standing compute functions exposed to Python.

All GPU-capable functions accept an ``ExecutionContext`` as their last positional argument.
"""

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.arrays import AnyArray
from marrow.dtypes import AnyDataType
from marrow.scalars import AnyScalar
import marrow.kernels as mk

# ``mk.filter`` collides with the like-named submodule, so the package alias
# resolves to the submodule rather than the function. Import it directly.
from marrow.kernels.filter import filter as _filter_kernel
from marrow.kernels.execution import ExecutionContext
from helpers import pyfunction


# ---------------------------------------------------------------------------
# ExecutionContext factory functions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# sort_indices / sort — composite kernels built from sort_indices + take
# ---------------------------------------------------------------------------


def sort_indices(
    array: AnyArray, ascending: Bool, nulls_first: Bool, ctx: ExecutionContext
) raises -> AnyArray:
    return mk.sort_indices(array, ascending, nulls_first, ctx=ctx)


def sort(
    array: AnyArray, ascending: Bool, nulls_first: Bool, ctx: ExecutionContext
) raises -> AnyArray:
    var indices = mk.sort_indices(array, ascending, nulls_first, ctx=ctx)
    return mk.take(array, indices, ctx)


def take(
    array: AnyArray, indices: AnyArray, ctx: ExecutionContext
) raises -> AnyArray:
    return mk.take(array, indices.as_int32().copy(), ctx)


def cast(
    array: AnyArray, target: AnyDataType, safe: Bool, ctx: ExecutionContext
) raises -> AnyArray:
    return mk.cast(array, target, safe, ctx)


# Thin ``-> AnyScalar`` wrappers so the concrete ``Int64Scalar``-returning
# distinct kernels match the ``pyfunction`` scalar shape.
def count_distinct(array: AnyArray, ctx: ExecutionContext) raises -> AnyScalar:
    return mk.count_distinct(array, ctx)


# ``MeanKernel.reduce`` is a trait method with a defaulted ``ctx``; referencing
# it directly in ``pykernel[...]`` crashes overload inference. Pin the exact
# ``(AnyArray, ExecutionContext) -> AnyScalar`` shape with a thin wrapper.
def mean(array: AnyArray, ctx: ExecutionContext) raises -> AnyScalar:
    return mk.MeanKernel.reduce(array, ctx)


def approx_count_distinct(
    array: AnyArray, ctx: ExecutionContext
) raises -> AnyScalar:
    return mk.approx_count_distinct(array, ctx)


# ``pykernel`` — wrap a marrow kernel of uniform shape
# ``(AnyArray..., ExecutionContext) -> R`` as a Python-callable. Each overload
# pins the full signature so Mojo can resolve the AnyArray runtime overload
# from a kernel reference like ``mk.equal`` even when the kernel has
# additional parametric or concrete-type overloads.


def pykernel[
    func: def(AnyArray, ExecutionContext) raises thin -> Bool,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    return pyfunction[func]()


def pykernel[
    func: def(AnyArray, ExecutionContext) raises thin -> AnyArray,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    return pyfunction[func]()


def pykernel[
    func: def(AnyArray, ExecutionContext) raises thin -> AnyScalar,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    return pyfunction[func]()


def pykernel[
    func: def(AnyArray, AnyArray, ExecutionContext) raises thin -> AnyArray,
]() -> def(
    PythonObject, PythonObject, PythonObject
) raises thin -> PythonObject:
    return pyfunction[func]()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    _ = (
        mb.add_type[ExecutionContext]("ExecutionContext")
        .def_staticmethod[pyfunction[ExecutionContext.serial]()]("serial")
        .def_staticmethod[pyfunction[ExecutionContext.parallel]()]("parallel")
    )
    mb.def_function[pykernel[mk.AddKernel.dispatch]()]("add")
    mb.def_function[pykernel[mk.SubKernel.dispatch]()]("subtract")
    mb.def_function[pykernel[mk.MulKernel.dispatch]()]("multiply")
    mb.def_function[pykernel[mk.DivKernel.dispatch]()]("divide")
    mb.def_function[pykernel[mk.SumKernel.dispatch]()]("sum")
    mb.def_function[pykernel[mk.ProductKernel.dispatch]()]("product")
    mb.def_function[pykernel[mk.MinKernel.dispatch]()]("min")
    mb.def_function[pykernel[mk.MaxKernel.dispatch]()]("max")
    mb.def_function[pykernel[mean]()]("mean")
    mb.def_function[pykernel[mk.AnyKernel.dispatch]()]("any")
    mb.def_function[pykernel[mk.AllKernel.dispatch]()]("all")
    mb.def_function[pykernel[mk.drop_null]()]("drop_null")
    mb.def_function[pykernel[_filter_kernel]()]("filter")
    mb.def_function[pykernel[mk.equal]()]("equal")
    mb.def_function[pykernel[mk.NeKernel.dispatch]()]("not_equal")
    mb.def_function[pykernel[mk.LtKernel.dispatch]()]("less")
    mb.def_function[pykernel[mk.LeKernel.dispatch]()]("less_equal")
    mb.def_function[pykernel[mk.GtKernel.dispatch]()]("greater")
    mb.def_function[pykernel[mk.GeKernel.dispatch]()]("greater_equal")
    mb.def_function[pyfunction[count_distinct]()]("count_distinct")
    mb.def_function[pyfunction[approx_count_distinct]()](
        "approx_count_distinct"
    )
    mb.def_function[pyfunction[sort_indices]()]("sort_indices")
    mb.def_function[pyfunction[sort]()]("sort")
    mb.def_function[pyfunction[take]()]("take")
    mb.def_function[pyfunction[cast]()]("cast")
