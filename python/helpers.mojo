"""Generic helpers for Python bindings: pymethod and pyfunction wrappers.

These reduce boilerplate when exposing Mojo methods and functions to Python
by auto-converting arguments via ConvertibleFromPython and return values via
ConvertibleToPython.
"""

from std.python import PythonObject, Python, ConvertibleToPython, ConvertibleFromPython


# ---------------------------------------------------------------------------
# pymethod — wrap a Mojo instance method as a Python-callable method
# ---------------------------------------------------------------------------


fn pymethod[
    T: AnyType,
    R: ConvertibleToPython,
    //,
    method: fn(T) raises -> R,
]() -> fn(UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
    """Wrap a zero-arg method returning ConvertibleToPython."""

    fn wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        return method(ptr[]).to_python_object()

    return wrapper


fn pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: fn(T, A0) raises -> R,
]() -> fn(UnsafePointer[T, MutAnyOrigin], PythonObject) raises -> PythonObject:
    """Wrap a single-arg method returning ConvertibleToPython."""

    fn wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        return method(ptr[], A0(py=arg)).to_python_object()

    return wrapper


fn pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: fn(T, A0, A1) raises -> R,
]() -> fn(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
) raises -> PythonObject:
    """Wrap a two-arg method returning ConvertibleToPython."""

    fn wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin],
        arg0: PythonObject,
        arg1: PythonObject,
    ) raises -> PythonObject:
        return method(ptr[], A0(py=arg0), A1(py=arg1)).to_python_object()

    return wrapper


fn pymethod[
    T: AnyType,
    E: ConvertibleToPython & Copyable,
    //,
    method: fn(T) raises -> List[E],
]() -> fn(UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
    """Wrap a zero-arg method returning List[ConvertibleToPython] as a Python list."""

    fn wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        for item in method(ptr[]):
            py_list.append(item.copy().to_python_object())
        return py_list

    return wrapper


fn pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    E: ConvertibleToPython & Copyable,
    //,
    method: fn(T, A0) raises -> List[E],
]() -> fn(UnsafePointer[T, MutAnyOrigin], PythonObject) raises -> PythonObject:
    """Wrap a single-arg method returning List[ConvertibleToPython] as a Python list."""

    fn wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        for item in method(ptr[], A0(py=arg)):
            py_list.append(item.copy().to_python_object())
        return py_list

    return wrapper


fn pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    E: ConvertibleToPython & Copyable,
    //,
    method: fn(T, A0, A1) raises -> List[E],
]() -> fn(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
) raises -> PythonObject:
    """Wrap a two-arg method returning List[ConvertibleToPython] as a Python list."""

    fn wrapper(
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


fn pymethod[
    T: AnyType,
    E: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: fn(T, List[E]) raises -> R,
]() -> fn(UnsafePointer[T, MutAnyOrigin], PythonObject) raises -> PythonObject:
    """Wrap a single-arg method taking List[ConvertibleFromPython] returning ConvertibleToPython."""

    fn wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        var n = Int(arg.__len__())
        var items = List[E]()
        for i in range(n):
            items.append(E(py=arg[i]))
        return method(ptr[], items^).to_python_object()

    return wrapper


fn pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    E: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: fn(T, A0, List[E]) raises -> R,
]() -> fn(
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
) raises -> PythonObject:
    """Wrap a two-arg method where the second arg is List[ConvertibleFromPython]."""

    fn wrapper(
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


fn pyfunction[
    A0: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    func: fn(A0) raises -> R,
]() -> fn(PythonObject) raises -> PythonObject:
    """Wrap a one-arg function returning ConvertibleToPython."""

    fn wrapper(arg0: PythonObject) raises -> PythonObject:
        return func(A0(py=arg0)).to_python_object()

    return wrapper


fn pyfunction[
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    func: fn(A0, A1) raises -> R,
]() -> fn(PythonObject, PythonObject) raises -> PythonObject:
    """Wrap a two-arg function returning ConvertibleToPython."""

    fn wrapper(arg0: PythonObject, arg1: PythonObject) raises -> PythonObject:
        return func(A0(py=arg0), A1(py=arg1)).to_python_object()

    return wrapper
