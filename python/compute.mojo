"""Free-standing compute functions exposed to Python.

These use runtime type dispatch via class name to convert PythonObject
back to typed arrays and call the appropriate kernel.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from marrow.arrays import Array
from marrow.kernels.aggregate import  sum_, product, min_, max_, any_, all_
from marrow.kernels.arithmetic import add, sub, mul, div
from marrow.kernels.filter import filter_ as _filter_overloaded, drop_nulls
from helpers import pyfunction

# TODO: use explicit Array types in the helper functions below
# otherwise for filter_ at least mojo is unable to resolve the
# right overload
def filter_(array: Array, selection: Array) raises -> Array:
    return _filter_overloaded(array, selection)



def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    mb.def_function[pyfunction[add]()]("add", docstring="Add all valid elements.")
    mb.def_function[pyfunction[sum_]()]("sum_", docstring="Sum all valid elements.")
    mb.def_function[pyfunction[product]()]("product", docstring="Product of all valid elements.")
    mb.def_function[pyfunction[min_]()]("min_", docstring="Minimum of all valid elements.")
    mb.def_function[pyfunction[max_]()]("max_", docstring="Maximum of all valid elements.")
    mb.def_function[pyfunction[any_]()]("any_", docstring="True if any valid element is true (bool arrays only).")
    mb.def_function[pyfunction[all_]()]("all_", docstring="True if all valid elements are true (bool arrays only).")
    mb.def_function[pyfunction[sub]()]("sub", docstring="Subtract all valid elements.")
    mb.def_function[pyfunction[mul]()]("mul", docstring="Multiply all valid elements.")
    mb.def_function[pyfunction[div]()]("div", docstring="Divide all valid elements.")
    mb.def_function[pyfunction[filter_]()]("filter_", docstring="Filter an array with a boolean mask.")
    mb.def_function[pyfunction[drop_nulls]()]("drop_nulls", docstring="Drop null values from an array.")


