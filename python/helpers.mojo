"""Generic helpers for Python bindings: pymethod and pyfunction wrappers.

These reduce boilerplate when exposing Mojo methods and functions to Python
by auto-converting arguments via ConvertibleFromPython and return values via
ConvertibleToPython.
"""

from std.python import (
    PythonObject,
    Python,
    ConvertibleToPython,
    ConvertibleFromPython,
)
from std.python.bindings import PythonTypeBuilder

# ---------------------------------------------------------------------------
# pymethod — wrap a Mojo instance method as a Python-callable method
# ---------------------------------------------------------------------------


def pymethod[
    T: AnyType,
    R: ConvertibleToPython,
    //,
    method: def(T) raises -> R,
]() -> def(UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
    """Wrap a zero-arg method returning ConvertibleToPython."""

    def wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        return method(ptr[]).to_python_object()

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: def(T, A0) raises -> R,
]() -> def(UnsafePointer[T, MutAnyOrigin], PythonObject) raises -> PythonObject:
    """Wrap a single-arg method returning ConvertibleToPython."""

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        return method(ptr[], A0(py=arg)).to_python_object()

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: def(T, A0, A1) raises -> R,
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
) raises -> PythonObject:
    """Wrap a two-arg method returning ConvertibleToPython."""

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin],
        arg0: PythonObject,
        arg1: PythonObject,
    ) raises -> PythonObject:
        return method(ptr[], A0(py=arg0), A1(py=arg1)).to_python_object()

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    A2: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: def(T, A0, A1, A2) raises -> R,
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject, PythonObject
) raises -> PythonObject:
    """Wrap a three-arg method returning ConvertibleToPython."""

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin],
        arg0: PythonObject,
        arg1: PythonObject,
        arg2: PythonObject,
    ) raises -> PythonObject:
        return method(
            ptr[], A0(py=arg0), A1(py=arg1), A2(py=arg2)
        ).to_python_object()

    return wrapper


def pymethod[
    T: AnyType,
    E: ConvertibleToPython & Copyable,
    //,
    method: def(T) raises -> List[E],
]() -> def(UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
    """Wrap a zero-arg method returning List[ConvertibleToPython] as a Python list.
    """

    def wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        for item in method(ptr[]):
            py_list.append(item.copy().to_python_object())
        return py_list

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    E: ConvertibleToPython & Copyable,
    //,
    method: def(T, A0) raises -> List[E],
]() -> def(UnsafePointer[T, MutAnyOrigin], PythonObject) raises -> PythonObject:
    """Wrap a single-arg method returning List[ConvertibleToPython] as a Python list.
    """

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        for item in method(ptr[], A0(py=arg)):
            py_list.append(item.copy().to_python_object())
        return py_list

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    E: ConvertibleToPython & Copyable,
    //,
    method: def(T, A0, A1) raises -> List[E],
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
) raises -> PythonObject:
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
            py_list.append(item.copy().to_python_object())
        return py_list

    return wrapper


def pymethod[
    T: AnyType,
    E: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: def(T, List[E]) raises -> R,
]() -> def(UnsafePointer[T, MutAnyOrigin], PythonObject) raises -> PythonObject:
    """Wrap a single-arg method taking List[ConvertibleFromPython] returning ConvertibleToPython.
    """

    def wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        var n = Int(arg.__len__())
        var items = List[E]()
        for i in range(n):
            items.append(E(py=arg[i]))
        return method(ptr[], items^).to_python_object()

    return wrapper


def pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    E: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: def(T, A0, List[E]) raises -> R,
]() -> def(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
) raises -> PythonObject:
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
        return method(ptr[], A0(py=arg0), items^).to_python_object()

    return wrapper


# ---------------------------------------------------------------------------
# pyfunction — wrap a Mojo free function as a Python-callable function
# ---------------------------------------------------------------------------


def pyfunction[
    A0: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    func: def(A0) raises -> R,
]() -> def(PythonObject) raises -> PythonObject:
    """Wrap a one-arg function returning ConvertibleToPython."""

    def wrapper(arg0: PythonObject) raises -> PythonObject:
        return func(A0(py=arg0)).to_python_object()

    return wrapper


def pyfunction[
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    func: def(A0, A1) raises -> R,
]() -> def(PythonObject, PythonObject) raises -> PythonObject:
    """Wrap a two-arg function returning ConvertibleToPython."""

    def wrapper(arg0: PythonObject, arg1: PythonObject) raises -> PythonObject:
        return func(A0(py=arg0), A1(py=arg1)).to_python_object()

    return wrapper


def marrow_module(obj: PythonObject) raises -> PythonObject:
    """Return the name of the module to implement the __module__ method."""
    return "marrow".to_python_object()


def def_display[
    T: Writable
](mut type_builder: PythonTypeBuilder) -> ref[type_builder] PythonTypeBuilder:
    """Define Python methods that are used to display instances of the type or the type itself.
    """

    def __str__(t: T) -> String:
        return String.write(t)

    return (
        type_builder.def_method[pymethod[__str__]()]("__str__")
        .def_method[pymethod[__str__]()]("__repr__")
        .def_method[marrow_module]("__module__")
    )
