"""Free-standing compute functions exposed to Python.

All GPU-capable functions accept an ``ExecutionContext`` as their last positional argument.
"""

from std.python import PythonObject, ConvertibleFromPython, ConvertibleToPython
from std.python.bindings import PythonModuleBuilder
from marrow.arrays import AnyArray
from marrow.scalars import AnyScalar
from marrow.kernels.execution import ExecutionContext
from marrow.kernels.aggregate import sum, product, min, max, any, all
from marrow.kernels.arithmetic import add, subtract, multiply, divide
from marrow.kernels.compare import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)
from marrow.kernels.filter import filter, drop_null, take as _take_kernel
from marrow.kernels.sort import sort_indices as _sort_indices_kernel


# ---------------------------------------------------------------------------
# wrap — convert a typed Mojo function to a PythonObject-based callable.
# All type parameters are explicit (no infer-only //) so overloaded kernel
# functions can be disambiguated by specifying the concrete arg/return types.
# ---------------------------------------------------------------------------


def wrap[
    R: ConvertibleToPython,
    func: def() raises thin -> R,
]() -> def() raises thin -> PythonObject:
    def wrapper() raises -> PythonObject:
        return func()

    return wrapper


def wrap[
    A0: ConvertibleFromPython,
    R: ConvertibleToPython,
    func: def(A0) raises thin -> R,
]() -> def(PythonObject) raises thin -> PythonObject:
    def wrapper(arg0: PythonObject) raises -> PythonObject:
        return func(A0(py=arg0))

    return wrapper


def wrap[
    A0: ConvertibleFromPython,
    func: def(A0) raises thin -> Bool,
]() -> def(PythonObject) raises thin -> PythonObject:
    def wrapper(arg0: PythonObject) raises -> PythonObject:
        return func(A0(py=arg0))

    return wrapper


def wrap[
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    R: ConvertibleToPython,
    func: def(A0, A1) raises thin -> R,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    def wrapper(arg0: PythonObject, arg1: PythonObject) raises -> PythonObject:
        return func(A0(py=arg0), A1(py=arg1))

    return wrapper


def wrap[
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    A2: ConvertibleFromPython,
    R: ConvertibleToPython,
    func: def(A0, A1, A2) raises thin -> R,
]() -> def(
    PythonObject, PythonObject, PythonObject
) raises thin -> PythonObject:
    def wrapper(
        arg0: PythonObject, arg1: PythonObject, arg2: PythonObject
    ) raises -> PythonObject:
        return func(A0(py=arg0), A1(py=arg1), A2(py=arg2))

    return wrapper


def wrap[
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    A2: ConvertibleFromPython,
    A3: ConvertibleFromPython,
    R: ConvertibleToPython,
    func: def(A0, A1, A2, A3) raises thin -> R,
]() -> def(
    PythonObject, PythonObject, PythonObject, PythonObject
) raises thin -> PythonObject:
    def wrapper(
        arg0: PythonObject,
        arg1: PythonObject,
        arg2: PythonObject,
        arg3: PythonObject,
    ) raises -> PythonObject:
        return func(A0(py=arg0), A1(py=arg1), A2(py=arg2), A3(py=arg3))

    return wrapper


# ---------------------------------------------------------------------------
# ExecutionContext factory functions
# ---------------------------------------------------------------------------


def _serial_context() raises -> ExecutionContext:
    return ExecutionContext.serial()


def _parallel_context() raises -> ExecutionContext:
    return ExecutionContext.parallel()


# ---------------------------------------------------------------------------
# sort_indices / sort — composite kernels built from sort_indices + take
# ---------------------------------------------------------------------------


def sort_indices(
    array: AnyArray, ascending: Bool, nulls_first: Bool, ctx: ExecutionContext
) raises -> AnyArray:
    return _sort_indices_kernel(array, ascending, nulls_first, ctx=ctx)


def sort(
    array: AnyArray, ascending: Bool, nulls_first: Bool, ctx: ExecutionContext
) raises -> AnyArray:
    var indices = _sort_indices_kernel(array, ascending, nulls_first, ctx=ctx)
    return _take_kernel(array, indices, ctx)


def take(
    array: AnyArray, indices: AnyArray, ctx: ExecutionContext
) raises -> AnyArray:
    return _take_kernel(array, indices.as_int32().copy(), ctx)


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    _ = mb.add_type[ExecutionContext]("ExecutionContext")
    mb.def_function[wrap[ExecutionContext, _serial_context]()](
        "_serial_context"
    )
    mb.def_function[wrap[ExecutionContext, _parallel_context]()](
        "_parallel_context"
    )
    mb.def_function[
        wrap[AnyArray, AnyArray, ExecutionContext, AnyArray, add]()
    ]("add")
    mb.def_function[
        wrap[AnyArray, AnyArray, ExecutionContext, AnyArray, subtract]()
    ]("subtract")
    mb.def_function[
        wrap[AnyArray, AnyArray, ExecutionContext, AnyArray, multiply]()
    ]("multiply")
    mb.def_function[
        wrap[AnyArray, AnyArray, ExecutionContext, AnyArray, divide]()
    ]("divide")
    mb.def_function[wrap[AnyArray, ExecutionContext, AnyScalar, sum]()]("sum")
    mb.def_function[wrap[AnyArray, ExecutionContext, AnyScalar, product]()](
        "product"
    )
    mb.def_function[wrap[AnyArray, ExecutionContext, AnyScalar, min]()]("min")
    mb.def_function[wrap[AnyArray, ExecutionContext, AnyScalar, max]()]("max")
    mb.def_function[wrap[AnyArray, any]()]("any")
    mb.def_function[wrap[AnyArray, all]()]("all")
    mb.def_function[wrap[AnyArray, AnyArray, AnyArray, filter]()]("filter")
    mb.def_function[wrap[AnyArray, AnyArray, drop_null]()]("drop_null")
    mb.def_function[
        wrap[AnyArray, AnyArray, ExecutionContext, AnyArray, equal]()
    ]("equal")
    mb.def_function[
        wrap[AnyArray, AnyArray, ExecutionContext, AnyArray, not_equal]()
    ]("not_equal")
    mb.def_function[
        wrap[AnyArray, AnyArray, ExecutionContext, AnyArray, less]()
    ]("less")
    mb.def_function[
        wrap[AnyArray, AnyArray, ExecutionContext, AnyArray, less_equal]()
    ]("less_equal")
    mb.def_function[
        wrap[AnyArray, AnyArray, ExecutionContext, AnyArray, greater]()
    ]("greater")
    mb.def_function[
        wrap[AnyArray, AnyArray, ExecutionContext, AnyArray, greater_equal]()
    ]("greater_equal")
    mb.def_function[
        wrap[AnyArray, Bool, Bool, ExecutionContext, AnyArray, sort_indices]()
    ]("sort_indices")
    mb.def_function[
        wrap[AnyArray, Bool, Bool, ExecutionContext, AnyArray, sort]()
    ]("sort")
    mb.def_function[wrap[_, _, _, AnyArray, take]()]("take")
