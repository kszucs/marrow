"""Free-standing compute functions exposed to Python.

These use runtime type dispatch via class name to convert PythonObject
back to typed arrays and call the appropriate kernel.

All GPU-capable functions accept an ``ExecutionContext`` as their last positional argument.
Pass ``None`` for CPU execution; pass a ``ma.ExecutionContext()`` instance for GPU execution.
"""

from std.gpu.host import DeviceContext
from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from marrow.arrays import AnyArray
from marrow.scalars import AnyScalar
from marrow.kernels.execution import ExecutionContext
from marrow.kernels.aggregate import (
    sum_ as _sum_agg,
    product as _product_agg,
    min_ as _min_agg,
    max_ as _max_agg,
)
from marrow.kernels.arithmetic import add, sub, mul, div
from marrow.kernels.compare import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)
from marrow.kernels.filter import filter_ as _filter_overloaded, drop_nulls
from helpers import pyfunction


def _ctx_init_gpu(out self: ExecutionContext) raises:
    self = ExecutionContext.gpu(DeviceContext())


def _opt_ctx(device_py: PythonObject) raises -> ExecutionContext:
    """Extract ExecutionContext from a Python ExecutionContext or None."""
    if device_py.__is__(PythonObject(None)):
        return ExecutionContext.serial()
    return ExecutionContext(py=device_py)


# TODO: use explicit AnyArray types in the helper functions below
# otherwise for filter_ at least mojo is unable to resolve the
# right overload
def filter_(array: AnyArray, selection: AnyArray) raises -> AnyArray:
    return _filter_overloaded(array, selection)


def any_(array: AnyArray) raises -> Bool:
    from marrow.kernels.aggregate import any_ as _any

    return _any(array)


def all_(array: AnyArray) raises -> Bool:
    from marrow.kernels.aggregate import all_ as _all

    return _all(array)


def equal_(left: AnyArray, right: AnyArray) raises -> AnyArray:
    return equal(left, right)


# ---------------------------------------------------------------------------
# ExecutionContext-aware wrappers — device is the last PythonObject arg; None → CPU
# ---------------------------------------------------------------------------


def _add(
    a0: PythonObject, a1: PythonObject, a2: PythonObject
) raises -> PythonObject:
    return add(
        AnyArray(py=a0), AnyArray(py=a1), _opt_ctx(a2)
    ).to_python_object()


def _sub(
    a0: PythonObject, a1: PythonObject, a2: PythonObject
) raises -> PythonObject:
    return sub(
        AnyArray(py=a0), AnyArray(py=a1), _opt_ctx(a2)
    ).to_python_object()


def _mul(
    a0: PythonObject, a1: PythonObject, a2: PythonObject
) raises -> PythonObject:
    return mul(
        AnyArray(py=a0), AnyArray(py=a1), _opt_ctx(a2)
    ).to_python_object()


def _div(
    a0: PythonObject, a1: PythonObject, a2: PythonObject
) raises -> PythonObject:
    return div(
        AnyArray(py=a0), AnyArray(py=a1), _opt_ctx(a2)
    ).to_python_object()


def _equal(
    a0: PythonObject, a1: PythonObject, a2: PythonObject
) raises -> PythonObject:
    return equal(
        AnyArray(py=a0), AnyArray(py=a1), _opt_ctx(a2)
    ).to_python_object()


def _not_equal(
    a0: PythonObject, a1: PythonObject, a2: PythonObject
) raises -> PythonObject:
    return not_equal(
        AnyArray(py=a0), AnyArray(py=a1), _opt_ctx(a2)
    ).to_python_object()


def _less(
    a0: PythonObject, a1: PythonObject, a2: PythonObject
) raises -> PythonObject:
    return less(
        AnyArray(py=a0), AnyArray(py=a1), _opt_ctx(a2)
    ).to_python_object()


def _less_equal(
    a0: PythonObject, a1: PythonObject, a2: PythonObject
) raises -> PythonObject:
    return less_equal(
        AnyArray(py=a0), AnyArray(py=a1), _opt_ctx(a2)
    ).to_python_object()


def _greater(
    a0: PythonObject, a1: PythonObject, a2: PythonObject
) raises -> PythonObject:
    return greater(
        AnyArray(py=a0), AnyArray(py=a1), _opt_ctx(a2)
    ).to_python_object()


def _greater_equal(
    a0: PythonObject, a1: PythonObject, a2: PythonObject
) raises -> PythonObject:
    return greater_equal(
        AnyArray(py=a0), AnyArray(py=a1), _opt_ctx(a2)
    ).to_python_object()


def _sum_(a0: PythonObject, a1: PythonObject) raises -> PythonObject:
    return _sum_agg(AnyArray(py=a0), _opt_ctx(a1)).to_python_object()


def _product(a0: PythonObject, a1: PythonObject) raises -> PythonObject:
    return _product_agg(AnyArray(py=a0), _opt_ctx(a1)).to_python_object()


def _min_(a0: PythonObject, a1: PythonObject) raises -> PythonObject:
    return _min_agg(AnyArray(py=a0), _opt_ctx(a1)).to_python_object()


def _max_(a0: PythonObject, a1: PythonObject) raises -> PythonObject:
    return _max_agg(AnyArray(py=a0), _opt_ctx(a1)).to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    _ = mb.add_type[ExecutionContext]("ExecutionContext").def_method[
        _ctx_init_gpu
    ]("__init__")
    mb.def_function[_add](
        "add",
        docstring=(
            "add(left, right, device, /) -> Array\n--\n\nAdd two arrays"
            " element-wise. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
    mb.def_function[_sum_](
        "sum_",
        docstring=(
            "sum_(array, device, /) -> Scalar\n--\n\nSum all valid elements,"
            " skipping nulls. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
    mb.def_function[_product](
        "product",
        docstring=(
            "product(array, device, /) -> Scalar\n--\n\nProduct of all valid"
            " elements, skipping nulls. Pass None for CPU, an ExecutionContext"
            " for GPU."
        ),
    )
    mb.def_function[_min_](
        "min_",
        docstring=(
            "min_(array, device, /) -> Scalar\n--\n\nMinimum of all valid"
            " elements, skipping nulls. Pass None for CPU, an ExecutionContext"
            " for GPU."
        ),
    )
    mb.def_function[_max_](
        "max_",
        docstring=(
            "max_(array, device, /) -> Scalar\n--\n\nMaximum of all valid"
            " elements, skipping nulls. Pass None for CPU, an ExecutionContext"
            " for GPU."
        ),
    )
    mb.def_function[pyfunction[any_]()](
        "any_",
        docstring=(
            "any_(array, /) -> bool\n--\n\nTrue if any valid element is"
            " true, skipping nulls."
        ),
    )
    mb.def_function[pyfunction[all_]()](
        "all_",
        docstring=(
            "all_(array, /) -> bool\n--\n\nTrue if all valid elements are"
            " true, skipping nulls."
        ),
    )
    mb.def_function[_sub](
        "sub",
        docstring=(
            "sub(left, right, device, /) -> Array\n--\n\nSubtract two arrays"
            " element-wise. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
    mb.def_function[_mul](
        "mul",
        docstring=(
            "mul(left, right, device, /) -> Array\n--\n\nMultiply two arrays"
            " element-wise. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
    mb.def_function[_div](
        "div",
        docstring=(
            "div(left, right, device, /) -> Array\n--\n\nDivide two arrays"
            " element-wise. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
    mb.def_function[pyfunction[filter_]()](
        "filter_",
        docstring=(
            "filter_(array, selection, /) -> Array\n--\n\nFilter"
            " an array with a boolean mask."
        ),
    )
    mb.def_function[pyfunction[drop_nulls]()](
        "drop_nulls",
        docstring=(
            "drop_nulls(array, /) -> Array\n--\n\nDrop null values from"
            " an array."
        ),
    )
    mb.def_function[_equal](
        "equal",
        docstring=(
            "equal(left, right, device, /) -> Array\n--\n\nElement-wise"
            " equality. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
    mb.def_function[_not_equal](
        "not_equal",
        docstring=(
            "not_equal(left, right, device, /) -> Array\n--\n\nElement-wise"
            " inequality. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
    mb.def_function[_less](
        "less",
        docstring=(
            "less(left, right, device, /) -> Array\n--\n\nElement-wise"
            " less-than. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
    mb.def_function[_less_equal](
        "less_equal",
        docstring=(
            "less_equal(left, right, device, /) -> Array\n--\n\nElement-wise"
            " less-or-equal. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
    mb.def_function[_greater](
        "greater",
        docstring=(
            "greater(left, right, device, /) -> Array\n--\n\nElement-wise"
            " greater-than. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
    mb.def_function[_greater_equal](
        "greater_equal",
        docstring=(
            "greater_equal(left, right, device, /) -> Array\n--\n\nElement-wise"
            " greater-or-equal. Pass None for CPU, an ExecutionContext for GPU."
        ),
    )
