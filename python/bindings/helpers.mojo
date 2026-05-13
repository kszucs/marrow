"""Generic helpers for Python bindings: pymethod wrappers.

These reduce boilerplate when exposing Mojo methods to Python by
auto-converting arguments via ConvertibleFromPython and return values via
implicit PythonObject construction (ConvertibleToPython or primitive types).
"""

from std.builtin.type_aliases import MutAnyOrigin
from std.memory import UnsafePointer
from std.python import (
    PythonObject,
    Python,
    ConvertibleToPython,
    ConvertibleFromPython,
)

# ---------------------------------------------------------------------------
# pymethod — wrap a Mojo instance method as a Python-callable method
# ---------------------------------------------------------------------------


def pymethod[
    T: AnyType,
    R: ConvertibleToPython,
    //,
    method: def(T) raises thin -> R,
]() -> def(UnsafePointer[T, MutAnyOrigin]) raises thin -> PythonObject:
    """Wrap a zero-arg method returning ConvertibleToPython."""

    def wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        return method(ptr[])

    return wrapper


def pymethod[
    T: AnyType,
    //,
    method: def(T) raises thin -> Int,
]() -> def(UnsafePointer[T, MutAnyOrigin]) raises thin -> PythonObject:
    """Wrap a zero-arg method returning Int."""

    def wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        return method(ptr[])

    return wrapper


def pymethod[
    T: AnyType,
    //,
    method: def(T) raises thin -> Bool,
]() -> def(UnsafePointer[T, MutAnyOrigin]) raises thin -> PythonObject:
    """Wrap a zero-arg method returning Bool."""

    def wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        return method(ptr[])

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: def(T, A0) raises thin -> R,
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject
) raises thin -> PythonObject:
    """Wrap a single-arg method returning ConvertibleToPython."""

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        return method(ptr[], A0(py=arg))

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    //,
    method: def(T, A0) raises thin -> Bool,
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject
) raises thin -> PythonObject:
    """Wrap a single-arg method returning Bool."""

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        return method(ptr[], A0(py=arg))

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: def(T, A0, A1) raises thin -> R,
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
) raises thin -> PythonObject:
    """Wrap a two-arg method returning ConvertibleToPython."""

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin],
        arg0: PythonObject,
        arg1: PythonObject,
    ) raises -> PythonObject:
        return method(ptr[], A0(py=arg0), A1(py=arg1))

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    A2: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: def(T, A0, A1, A2) raises thin -> R,
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject, PythonObject
) raises thin -> PythonObject:
    """Wrap a three-arg method returning ConvertibleToPython."""

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin],
        arg0: PythonObject,
        arg1: PythonObject,
        arg2: PythonObject,
    ) raises -> PythonObject:
        return method(ptr[], A0(py=arg0), A1(py=arg1), A2(py=arg2))

    return wrapper


def pymethod[
    T: AnyType,
    E: ConvertibleToPython & Copyable,
    //,
    method: def(T) raises thin -> List[E],
]() -> def(UnsafePointer[T, MutAnyOrigin]) raises thin -> PythonObject:
    """Wrap a zero-arg method returning List[ConvertibleToPython] as a Python list.
    """

    def wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        for item in method(ptr[]):
            py_list.append(item.copy())
        return py_list

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    E: ConvertibleToPython & Copyable,
    //,
    method: def(T, A0) raises thin -> List[E],
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject
) raises thin -> PythonObject:
    """Wrap a single-arg method returning List[ConvertibleToPython] as a Python list.
    """

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        for item in method(ptr[], A0(py=arg)):
            py_list.append(item.copy())
        return py_list

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    E: ConvertibleToPython & Copyable,
    //,
    method: def(T, A0, A1) raises thin -> List[E],
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
) raises thin -> PythonObject:
    """Wrap a two-arg method returning List[ConvertibleToPython] as a Python list.
    """

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin],
        arg0: PythonObject,
        arg1: PythonObject,
    ) raises -> PythonObject:
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        for item in method(ptr[], A0(py=arg0), A1(py=arg1)):
            py_list.append(item.copy())
        return py_list

    return wrapper


def pymethod[
    T: AnyType,
    E: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: def(T, List[E]) raises thin -> R,
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject
) raises thin -> PythonObject:
    """Wrap a single-arg method taking List[ConvertibleFromPython] returning ConvertibleToPython.
    """

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        var n = Int(arg.__len__())
        var items = List[E]()
        for i in range(n):
            items.append(E(py=arg[i]))
        return method(ptr[], items^)

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    E: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: def(T, A0, List[E]) raises thin -> R,
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
) raises thin -> PythonObject:
    """Wrap a two-arg method where the second arg is List[ConvertibleFromPython].
    """

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin],
        arg0: PythonObject,
        arg1: PythonObject,
    ) raises -> PythonObject:
        var n = Int(arg1.__len__())
        var items = List[E]()
        for i in range(n):
            items.append(E(py=arg1[i]))
        return method(ptr[], A0(py=arg0), items^)

    return wrapper
