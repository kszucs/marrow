"""Free-standing compute functions exposed to Python.

All GPU-capable functions accept an ``ExecutionContext`` as their last positional argument.
"""

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.arrays import DynArray
from marrow.dtypes import DynType
from marrow.scalars import DynScalar
import marrow.kernels as mk

# ``mk.filter`` collides with the like-named submodule, so the package alias
# resolves to the submodule rather than the function. Import it directly.
from marrow.kernels.filter import filter as _filter_kernel
from marrow.kernels.boolean import IsNullKernel, NotNullKernel
from marrow.kernels.execution import ExecutionContext
from marrow.expr.aggregates import (
    Sum,
    Product,
    Mean,
    Min,
    Max,
    CountDistinct,
    ApproxCountDistinct,
)
from marrow.kernels.aggregate import (
    Aggregation,
    AggFunction,
)

from helpers import pyfunction


# ---------------------------------------------------------------------------
# ExecutionContext factory functions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# sort_indices / sort — composite kernels built from sort_indices + take
# ---------------------------------------------------------------------------


def sort_indices(
    array: DynArray, ascending: Bool, nulls_first: Bool, ctx: ExecutionContext
) raises -> DynArray:
    return mk.sort_indices(array, ascending, nulls_first, ctx=ctx)


def sort(
    array: DynArray, ascending: Bool, nulls_first: Bool, ctx: ExecutionContext
) raises -> DynArray:
    var indices = mk.sort_indices(array, ascending, nulls_first, ctx=ctx)
    return mk.take(array, indices, ctx)


def take(
    array: DynArray, indices: DynArray, ctx: ExecutionContext
) raises -> DynArray:
    return mk.take(array, indices.as_int32().copy(), ctx)


def cast(
    array: DynArray, target: DynType, safe: Bool, ctx: ExecutionContext
) raises -> DynArray:
    return mk.cast(array, target, safe, ctx)


# Whole-column aggregates. The column's dtype is a runtime value here, so each
# one resolves its `AggFunction` to the `Aggregation` implementing it (the same
# path a `GROUP BY` plan takes, with one implicit group) and reads the single
# row back out — the thin `(DynArray, ExecutionContext) -> DynScalar` shape
# `pykernel` expects.
def aggregate[
    F: AggFunction
](array: DynArray, ctx: ExecutionContext) raises -> DynScalar:
    var box = List[DynScalar]()

    @parameter
    def run[A: Aggregation]() raises:
        box.append(
            A.whole(A.from_any(array), ctx.resolved_num_threads()).to_dyn()[0]
        )

    F.resolve[run](array.dtype())
    return box[0].copy()


# ``pykernel`` — wrap a marrow kernel of uniform shape
# ``(DynArray..., ExecutionContext) -> R`` as a Python-callable. Each overload
# pins the full signature so Mojo can resolve the DynArray runtime overload
# from a kernel reference like ``mk.equal`` even when the kernel has
# additional parametric or concrete-type overloads.


def pykernel[
    func: def(DynArray, ExecutionContext) raises thin -> Bool,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    return pyfunction[func]()


def pykernel[
    func: def(DynArray, ExecutionContext) raises thin -> DynArray,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    return pyfunction[func]()


def pykernel[
    func: def(DynArray, ExecutionContext) raises thin -> DynScalar,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    return pyfunction[func]()


def pykernel[
    func: def(DynArray, DynArray, ExecutionContext) raises thin -> DynArray,
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
    mb.def_function[pykernel[aggregate[Sum]]()]("sum")
    mb.def_function[pykernel[aggregate[Product]]()]("product")
    mb.def_function[pykernel[aggregate[Min]]()]("min")
    mb.def_function[pykernel[aggregate[Max]]()]("max")
    mb.def_function[pykernel[aggregate[Mean]]()]("mean")
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
    mb.def_function[pykernel[aggregate[CountDistinct]]()]("count_distinct")
    mb.def_function[pykernel[aggregate[ApproxCountDistinct]]()](
        "approx_count_distinct"
    )
    mb.def_function[pyfunction[sort_indices]()]("sort_indices")
    mb.def_function[pyfunction[sort]()]("sort")
    mb.def_function[pyfunction[take]()]("take")
    mb.def_function[pyfunction[cast]()]("cast")
