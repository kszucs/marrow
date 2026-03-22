"""Free-standing compute functions exposed to Python.

These use runtime type dispatch via class name to convert PythonObject
back to typed arrays and call the appropriate kernel.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from marrow.arrays import Array
from marrow.kernels.aggregate import sum_, product, min_, max_, any_, all_
from marrow.kernels.arithmetic import add, sub, mul, div
from marrow.kernels.filter import filter_ as _filter_overloaded, drop_nulls
from helpers import pyfunction


# TODO: use explicit Array types in the helper functions below
# otherwise for filter_ at least mojo is unable to resolve the
# right overload
def filter_(array: Array, selection: Array) raises -> Array:
    return _filter_overloaded(array, selection)


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    mb.def_function[pyfunction[add]()](
        "add", docstring="add(left: Array, right: Array, /) -> Array\n--\n\nAdd two arrays element-wise, propagating nulls."
    )
    mb.def_function[pyfunction[sum_]()](
        "sum_", docstring="sum_(array: Array, /) -> float\n--\n\nSum all valid elements, skipping nulls."
    )
    mb.def_function[pyfunction[product]()](
        "product", docstring="product(array: Array, /) -> float\n--\n\nProduct of all valid elements, skipping nulls."
    )
    mb.def_function[pyfunction[min_]()](
        "min_", docstring="min_(array: Array, /) -> float\n--\n\nMinimum of all valid elements, skipping nulls."
    )
    mb.def_function[pyfunction[max_]()](
        "max_", docstring="max_(array: Array, /) -> float\n--\n\nMaximum of all valid elements, skipping nulls."
    )
    mb.def_function[pyfunction[any_]()](
        "any_", docstring="any_(array: Array, /) -> bool\n--\n\nTrue if any valid element is true, skipping nulls."
    )
    mb.def_function[pyfunction[all_]()](
        "all_", docstring="all_(array: Array, /) -> bool\n--\n\nTrue if all valid elements are true, skipping nulls."
    )
    mb.def_function[pyfunction[sub]()](
        "sub", docstring="sub(left: Array, right: Array, /) -> Array\n--\n\nSubtract two arrays element-wise, propagating nulls."
    )
    mb.def_function[pyfunction[mul]()](
        "mul", docstring="mul(left: Array, right: Array, /) -> Array\n--\n\nMultiply two arrays element-wise, propagating nulls."
    )
    mb.def_function[pyfunction[div]()](
        "div", docstring="div(left: Array, right: Array, /) -> Array\n--\n\nDivide two arrays element-wise, propagating nulls."
    )
    mb.def_function[pyfunction[filter_]()](
        "filter_", docstring="filter_(array: Array, selection: Array, /) -> Array\n--\n\nFilter an array with a boolean mask."
    )
    mb.def_function[pyfunction[drop_nulls]()](
        "drop_nulls", docstring="drop_nulls(array: Array, /) -> Array\n--\n\nDrop null values from an array."
    )
