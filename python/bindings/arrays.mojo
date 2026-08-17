from std.python import (
    PythonObject,
    ConvertibleToPython,
    ConvertibleFromPython,
    Python,
)
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import (
    CPython,
    ExternalFunction,
    PyObjectPtr,
    PyTypeObject,
    PyTypeObjectPtr,
)
from std.ffi import c_char, c_int, c_ssize_t
from std.memory import ArcPointer, Pointer
from std.memory.alloc import unsafe_alloc
from std.utils import Variant
from std.builtin.variadics import Variadic
from std.os import abort
from std.builtin.rebind import downcast
from marrow.c_data import CArrowSchema, CArrowArray
from marrow.execution import ExecContext
from marrow.arrays import (
    DynArray,
    ArrayData,
    ChunkedArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
    Int32Array,
    BoolArray,
    NullArray,
)
from marrow.buffers import Buffer, Bitmap
from marrow.builders import (
    DynBuilder,
    BoolBuilder,
    BinaryBuilder,
    Int32Builder,
    PrimitiveBuilder,
    StringBuilder,
    ListBuilder,
    StructBuilder,
)
from marrow.scalars import DynScalar
import marrow.dtypes as dt

from helpers import pymethod


# int PyBytes_AsStringAndSize(PyObject *obj, char **buffer, Py_ssize_t *length)
comptime _PyBytesAsStringAndSizeFn = ExternalFunction[
    "PyBytes_AsStringAndSize",
    def(
        PyObjectPtr,
        Pointer[OptionalPointer[c_char, ImmutAnyOrigin], MutAnyOrigin],
        Pointer[c_ssize_t, MutAnyOrigin],
    ) thin -> c_int,
]


# ---------------------------------------------------------------------------
# PyHelpers — cached CPython state for hot-path converters
# ---------------------------------------------------------------------------


struct PyHelpers(Copyable, Movable):
    """Caches a Python instance and Py_None pointer for hot-path converters.

    Create once per extend()/append() call so the tight inner loop pays only
    a single Py_None() lookup instead of one per element.  is_none() uses a
    direct pointer comparison — no CPython API call needed per element.
    """

    var py: Python
    var none_ptr: PyObjectPtr
    var _unicode_type: PyTypeObjectPtr
    var _bytes_type: PyTypeObjectPtr
    var _list_type: PyTypeObjectPtr
    var _tuple_type: PyTypeObjectPtr
    var _dict_type: PyTypeObjectPtr
    var _bytes_as_string_and_size_fn: _PyBytesAsStringAndSizeFn.type

    def __init__(out self):
        self.py = Python()
        ref cpy = self.py.cpython()
        self.none_ptr = cpy.Py_None()
        self._unicode_type = (
            cpy.lib.get_symbol[PyTypeObject]("PyUnicode_Type")
            .value()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        self._bytes_type = (
            cpy.lib.get_symbol[PyTypeObject]("PyBytes_Type")
            .value()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        self._list_type = (
            cpy.lib.get_symbol[PyTypeObject]("PyList_Type")
            .value()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        self._tuple_type = (
            cpy.lib.get_symbol[PyTypeObject]("PyTuple_Type")
            .value()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        self._dict_type = cpy.PyDict_Type()
        self._bytes_as_string_and_size_fn = _PyBytesAsStringAndSizeFn.load(
            cpy.lib.borrow()
        )

    def __init__(out self, var other: Self):
        # Python is not Movable/Copyable; re-create it. Cached pointers are process-wide constants.
        self = PyHelpers()

    def __init__(out self, ref other: Self):
        self = PyHelpers()

    @always_inline
    def cpy(mut self) -> ref[ImmutAnyOrigin] CPython:
        return self.py.cpython()

    @always_inline
    def is_none(self, ptr: PyObjectPtr) -> Bool:
        return ptr == self.none_ptr

    @always_inline
    def is_unicode(mut self, ptr: PyObjectPtr) -> Bool:
        return self.cpy().PyObject_TypeCheck(ptr, self._unicode_type) != 0

    @always_inline
    def is_bytes(mut self, ptr: PyObjectPtr) -> Bool:
        return self.cpy().PyObject_TypeCheck(ptr, self._bytes_type) != 0

    @always_inline
    def is_list(mut self, ptr: PyObjectPtr) -> Bool:
        return self.cpy().PyObject_TypeCheck(ptr, self._list_type) != 0

    @always_inline
    def is_tuple(mut self, ptr: PyObjectPtr) -> Bool:
        return self.cpy().PyObject_TypeCheck(ptr, self._tuple_type) != 0

    @always_inline
    def is_dict(mut self, ptr: PyObjectPtr) -> Bool:
        return self.cpy().PyObject_TypeCheck(ptr, self._dict_type) != 0

    @always_inline
    def raise_on_error(mut self) raises:
        """Raise the current Python exception if one is set."""
        if self.cpy().PyErr_Occurred():
            raise self.cpy().unsafe_get_error()

    @always_inline
    def to_scalar[
        dtype: DType
    ](mut self, ptr: PyObjectPtr) raises -> Scalar[dtype]:
        """Convert a Python int, float, or bool object to Scalar[dtype]; raises on type error or overflow.
        """
        comptime if dtype == DType.bool:
            var val = self.cpy().PyLong_AsSsize_t(ptr)
            if val == -1:
                self.raise_on_error()
            # `Bool` is `Intable`, and since Mojo 1.0 that constructor is
            # selected for any integer scalar — it then rejects `bool` as
            # non-integral. Go through the bool-typed conversion instead.
            var flag: Scalar[DType.bool] = val != 0
            return rebind[Scalar[dtype]](flag)
        elif dtype.is_floating_point():
            var val = self.cpy().PyFloat_AsDouble(ptr)
            if val == -1.0:
                self.raise_on_error()
            return Scalar[dtype](val)
        else:
            var val = self.cpy().PyLong_AsSsize_t(ptr)
            if val == -1:
                self.raise_on_error()
            # uint64 is a special case: Scalar[UInt64Type](-1).cast[Int64Type]() == -1 == val, so the
            # roundtrip check below is insufficient for negatives; catch them explicitly.
            comptime if dtype == DType.uint64:
                if val < 0:
                    raise Error("integer value out of range for type")
            var converted = Scalar[dtype](val)
            if Int(converted.cast[DType.int64]()) != val:
                raise Error("integer value out of range for type")
            return converted

    @always_inline
    def to_string_slice(
        mut self, ptr: PyObjectPtr
    ) raises -> StringSlice[ImmutAnyOrigin]:
        """Decode a Python str to a UTF-8 StringSlice; raises on TypeError or encoding error.
        """
        var s = self.cpy().PyUnicode_AsUTF8AndSize(ptr)
        self.raise_on_error()
        return s.value()

    @always_inline
    def to_bytes_slice(
        mut self, ptr: PyObjectPtr
    ) raises -> StringSlice[ImmutAnyOrigin]:
        """Return the raw bytes buffer of a Python bytes object as a StringSlice.
        """
        var data_ptr = OptionalPointer[c_char, ImmutAnyOrigin]()
        var size = c_ssize_t(0)
        # The CPython FFI signature pins `MutAnyOrigin`; the pointers to these
        # locals carry a concrete origin, so cast at the C boundary (the stricter
        # compiler no longer converts implicitly).
        var rc = self._bytes_as_string_and_size_fn(
            ptr,
            Pointer(to=data_ptr).unsafe_origin_cast[MutAnyOrigin](),
            Pointer(to=size).unsafe_origin_cast[MutAnyOrigin](),
        )
        if rc != 0:
            self.raise_on_error()
        return StringSlice[ImmutAnyOrigin](
            unsafe_from_utf8=Span[Byte, ImmutAnyOrigin](
                unsafe_ptr=data_ptr.value().unsafe_bitcast[Byte](),
                length=Int(size),
            )
        )

    @always_inline
    def length(mut self, ptr: PyObjectPtr) -> Int:
        """Return the length of a Python sequence or mapping."""
        return Int(self.cpy().PyObject_Length(ptr))

    @always_inline
    def dict_getitem(
        mut self, dict_ptr: PyObjectPtr, key: PyObjectPtr
    ) raises -> PyObjectPtr:
        """Look up key in a Python dict; returns none_ptr if missing, raises on error.
        """
        var val = self.cpy().PyDict_GetItemWithError(dict_ptr, key)
        if val == PyObjectPtr():
            self.raise_on_error()
            return self.none_ptr
        return val

    @always_inline
    def list_getitem(mut self, list_ptr: PyObjectPtr, i: Int) -> PyObjectPtr:
        """Borrowed reference to list[i]; no bounds checking (mirrors PyList_GetItem).
        """
        return self.cpy().PyList_GetItem(list_ptr, i)

    @always_inline
    def tuple_getitem(mut self, tuple_ptr: PyObjectPtr, i: Int) -> PyObjectPtr:
        """Borrowed reference to tuple[i]; no bounds checking (mirrors PyTuple_GetItem).
        """
        return self.cpy().PyTuple_GetItem(tuple_ptr, i)


# ---------------------------------------------------------------------------
# DynArray helpers for Python __getitem__ and Arrow C Data Interface
# ---------------------------------------------------------------------------


def _any_to_array(arr: DynArray) -> DynArray:
    return arr.copy()


def _any_dtype(arr: DynArray) -> dt.DynType:
    return arr.dtype()


def _any_array_getitem(
    array: DynArray,
    index: Int,
) raises -> PythonObject:
    var n = array.length()
    if index < 0 or index >= n:
        raise Error(t"index {index} out of bounds for length {n}")
    return array[index].to_python_object()


def _any_array_getitem_py(
    py_self: PythonObject,
    index: PythonObject,
) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[DynArray]()
    return _any_array_getitem(ptr[], Int(py=index))


# ---------------------------------------------------------------------------
# PyInferrer — mirrors PyArrow's PyInferrer (inference.cc)
# ---------------------------------------------------------------------------


struct PyInferrer(Copyable, Movable):
    """Infers the Arrow DataType of a Python sequence.

    Mirrors PyArrow's PyInferrer (inference.cc): counts occurrences by Python
    type in a single pass, recursing into list/dict elements via _visit_list and
    _visit_dict, then resolves to a DataType without re-iterating the sequence.
    """

    var none_count: Int
    var bool_count: Int
    var int_count: Int
    var float_count: Int
    var unicode_count: Int
    var bytes_count: Int
    var list_count: Int
    var struct_count: Int

    # Child inferrers — List[PyInferrer] works because PyInferrer declares Copyable
    var _list_child: List[PyInferrer]  # 0 or 1 elements
    var _field_order: List[String]
    var _field_children: List[PyInferrer]  # parallel to _field_order

    var py: PyHelpers

    def __init__(out self) raises:
        self.none_count = 0
        self.bool_count = 0
        self.int_count = 0
        self.float_count = 0
        self.unicode_count = 0
        self.bytes_count = 0
        self.list_count = 0
        self.struct_count = 0
        self._list_child = []
        self._field_order = []
        self._field_children = []
        self.py = PyHelpers()

    # Explicit (empty) destructor so this self-referential struct
    # (`_list_child` / `_field_children` are `List[PyInferrer]`) is
    # Deinitable; fields are still destroyed automatically.
    def __deinit__(deinit self):
        pass

    def visit(mut self, ptr: PyObjectPtr) raises -> Bool:
        """Count one element's Python type, following PyArrow's Visit() order.

        Returns False when the type is fully determined (no further widening possible),
        allowing the caller to stop early. Mirrors PyArrow's keep_going flag.
        """
        ref cpy = self.py.cpy()
        if self.py.is_none(ptr):
            self.none_count += 1
            return True  # null never locks the type
        elif cpy.PyBool_Check(ptr) != 0:  # exact bool check before PyLong_Check
            self.bool_count += 1
            return True  # bool can widen to int64 or float64
        elif cpy.PyFloat_Check(ptr) != 0:  # float before int
            self.float_count += 1
            return False  # float64 is the widest numeric type
        elif cpy.PyLong_Check(ptr) != 0:
            self.int_count += 1
            return True  # int can widen to float64
        elif self.py.is_unicode(ptr):
            self.unicode_count += 1
            return False  # string cannot widen further
        elif self.py.is_bytes(ptr):
            self.bytes_count += 1
            return False  # bytes cannot widen further
        elif self.py.is_dict(ptr):
            self.struct_count += 1
            self._visit_dict(ptr)
            return False  # struct cannot widen further
        elif self.py.is_list(ptr):
            self.list_count += 1
            self._visit_list(ptr)
            return False  # list cannot widen further
        elif self.py.is_tuple(ptr):
            self.list_count += 1
            self._visit_tuple(ptr)
            return False  # list cannot widen further
        else:
            raise Error(
                "cannot include value of type: ",
                PythonObject(from_borrowed=ptr).__class__.__name__,
            )

    def _visit_list(mut self, ptr: PyObjectPtr) raises:
        """Mirrors PyArrow's VisitSequence: recurse into list children."""
        if len(self._list_child) == 0:
            self._list_child.append(PyInferrer())
        var n = self.py.length(ptr)
        for i in range(n):
            if not self._list_child[0].visit(self.py.list_getitem(ptr, i)):
                break

    def _visit_tuple(mut self, ptr: PyObjectPtr) raises:
        """Mirrors PyArrow's VisitSequence for tuples."""
        if len(self._list_child) == 0:
            self._list_child.append(PyInferrer())
        var n = self.py.length(ptr)
        for i in range(n):
            if not self._list_child[0].visit(self.py.tuple_getitem(ptr, i)):
                break

    def _visit_dict(mut self, dict_ptr: PyObjectPtr) raises:
        """Mirrors PyArrow's VisitDict: route each value to its field's child inferrer.
        """
        ref cpy = self.py.cpy()
        var n = self.py.length(dict_ptr)
        var pos = Int(0)
        var key_raw = unsafe_alloc[PyObjectPtr](1)
        var val_raw = unsafe_alloc[PyObjectPtr](1)
        var pos_ptr = Pointer(to=pos)
        for _ in range(n):
            _ = cpy.PyDict_Next(
                dict_ptr,
                pos_ptr.as_unsafe_any_origin(),
                key_raw.as_unsafe_any_origin(),
                val_raw.as_unsafe_any_origin(),
            )
            var name = self.py.to_string_slice(key_raw[])
            var idx = -1
            for i in range(len(self._field_order)):
                if self._field_order[i] == name:
                    idx = i
                    break
            if idx == -1:
                idx = len(self._field_order)
                self._field_order.append(String(name))
                self._field_children.append(PyInferrer())
            _ = self._field_children[idx].visit(val_raw[])
        key_raw.unsafe_free()
        val_raw.unsafe_free()

    def _total_count(self) -> Int:
        return (
            self.none_count
            + self.bool_count
            + self.int_count
            + self.float_count
            + self.unicode_count
            + self.bytes_count
            + self.list_count
            + self.struct_count
        )

    def _get_type(self) raises -> dt.DynType:
        if self.bytes_count > 0:
            if self.bytes_count + self.none_count != self._total_count():
                raise Error("cannot mix bytes and non-bytes values")
            return dt.binary
        if self.list_count > 0:
            if self.list_count + self.none_count != self._total_count():
                raise Error("cannot mix list and non-list values")
            if len(self._list_child) == 0:
                raise Error("cannot infer type: all-null list")
            return dt.list_(self._list_child[0]._get_type())
        if self.struct_count > 0:
            if self.struct_count + self.none_count != self._total_count():
                raise Error("cannot mix dict and non-dict values")
            var fields: List[dt.Field] = []
            for i in range(len(self._field_order)):
                fields.append(
                    dt.Field(
                        self._field_order[i],
                        self._field_children[i]._get_type(),
                        nullable=True,
                    )
                )
            return dt.struct_(fields^)
        if (
            self.unicode_count > 0
            and (self.bool_count + self.int_count + self.float_count) > 0
        ):
            raise Error("cannot mix string and numeric types")
        if self.float_count > 0:
            return dt.float64
        if self.int_count > 0:
            return dt.int64
        if self.bool_count > 0:
            return dt.bool_
        if self.unicode_count > 0:
            return dt.string
        return dt.null  # empty sequence or all-None

    def infer(mut self, obj: PythonObject) raises -> dt.DynType:
        """Visit elements until type is locked, then scan remaining elements for nulls.
        """
        var list_ptr = obj._obj_ptr
        var n = len(obj)
        var stopped_at = n
        for i in range(n):
            var item_ptr = self.py.list_getitem(list_ptr, i)
            if not self.visit(item_ptr):
                stopped_at = i + 1
                break
        # If we stopped early, scan remaining elements for nulls only.
        # This is cheap (single pointer comparison per element) and keeps
        # none_count accurate so the converter can use the fast no-null path.
        if stopped_at < n:
            for i in range(stopped_at, n):
                if self.py.is_none(self.py.list_getitem(list_ptr, i)):
                    self.none_count += 1
                    break  # one null is enough to know has_nulls=True
        return self._get_type()


# ---------------------------------------------------------------------------
# PyConverter trait — interface for typed Python-to-Arrow converters
# ---------------------------------------------------------------------------


trait PyConverter(Deinitable, Movable):
    def append(mut self, value: PyObjectPtr) raises:
        ...

    def extend(mut self, values: PyObjectPtr) raises:
        ...


# ---------------------------------------------------------------------------
# PyAnyConverter — type-erased converter with dynamic dispatch
# ---------------------------------------------------------------------------


struct PyAnyConverter(ImplicitlyCopyable, Movable):
    """Type-erased converter using Variant + comptime-for dispatch.

    Holds `ArcPointer[VariantType]` so copying is O(1) (ref-count bump),
    mirroring `DynBuilder._ptr: ArcPointer[DynBuilder.VariantType]`.
    """

    comptime VariantType = Variant[
        PyBoolConverter,
        PyPrimitiveConverter[dt.Int8Type],
        PyPrimitiveConverter[dt.Int16Type],
        PyPrimitiveConverter[dt.Int32Type],
        PyPrimitiveConverter[dt.Int64Type],
        PyPrimitiveConverter[dt.UInt8Type],
        PyPrimitiveConverter[dt.UInt16Type],
        PyPrimitiveConverter[dt.UInt32Type],
        PyPrimitiveConverter[dt.UInt64Type],
        PyPrimitiveConverter[dt.Float16Type],
        PyPrimitiveConverter[dt.Float32Type],
        PyPrimitiveConverter[dt.Float64Type],
        PyStringConverter,
        PyBinaryConverter,
        PyListConverter,
        PyFixedSizeListConverter,
        PyStructConverter,
    ]

    var _v: ArcPointer[Self.VariantType]

    @implicit
    def __init__[T: PyConverter](out self, var value: T):
        self._v = ArcPointer(Self.VariantType(value^))

    def __init__(out self, *, copy: Self):
        self._v = copy._v.copy()

    # Explicit (empty) destructor so this type is Deinitable despite
    # the `PyStructConverter -> List[PyAnyConverter] -> PyAnyConverter` cycle;
    # the ArcPointer field is still destroyed automatically after the body.
    def __deinit__(deinit self):
        pass

    def __init__(
        out self,
        builder: DynBuilder,
        dtype: dt.DynType,
        has_nulls: Bool = True,
    ) raises:
        if dtype == dt.bool_:
            self = Self(PyBoolConverter(builder, has_nulls))
        elif dtype == dt.int8:
            self = Self(PyPrimitiveConverter[dt.Int8Type](builder, has_nulls))
        elif dtype == dt.int16:
            self = Self(PyPrimitiveConverter[dt.Int16Type](builder, has_nulls))
        elif dtype == dt.int32:
            self = Self(PyPrimitiveConverter[dt.Int32Type](builder, has_nulls))
        elif dtype == dt.int64:
            self = Self(PyPrimitiveConverter[dt.Int64Type](builder, has_nulls))
        elif dtype == dt.uint8:
            self = Self(PyPrimitiveConverter[dt.UInt8Type](builder, has_nulls))
        elif dtype == dt.uint16:
            self = Self(PyPrimitiveConverter[dt.UInt16Type](builder, has_nulls))
        elif dtype == dt.uint32:
            self = Self(PyPrimitiveConverter[dt.UInt32Type](builder, has_nulls))
        elif dtype == dt.uint64:
            self = Self(PyPrimitiveConverter[dt.UInt64Type](builder, has_nulls))
        elif dtype == dt.float16:
            self = Self(
                PyPrimitiveConverter[dt.Float16Type](builder, has_nulls)
            )
        elif dtype == dt.float32:
            self = Self(
                PyPrimitiveConverter[dt.Float32Type](builder, has_nulls)
            )
        elif dtype == dt.float64:
            self = Self(
                PyPrimitiveConverter[dt.Float64Type](builder, has_nulls)
            )
        elif dtype.is_string():
            self = Self(PyStringConverter(builder, has_nulls))
        elif dtype.is_binary():
            self = Self(PyBinaryConverter(builder, has_nulls))
        elif dtype.is_list():
            self = Self(PyListConverter(builder, has_nulls))
        elif dtype.is_fixed_size_list():
            self = Self(PyFixedSizeListConverter(builder, has_nulls))
        elif dtype.is_struct():
            self = Self(PyStructConverter(builder))
        else:
            raise Error("unsupported type: ", dtype)

    # The `isa` ladder is written out here rather than delegated to a shared
    # helper, matching `DynArray._dispatch` and for the same measured reason:
    # interposing a narrowing closure between the caller and the ladder costs a
    # fully inlined copy of the adapter in every arm (+662,740 bytes on one
    # gate, e5509c3). These two were the last callers of the helper that commit
    # deleted, and were missed because CI has not run since 2026-05-11 -- the
    # Python extension has not built since.

    def append(mut self, value: PyObjectPtr) raises:
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, PyConverter):
                if self._v[].isa[T]():
                    ref c = rebind[downcast[T, PyConverter]](self._v[][T])
                    return c.append(value)
        raise Error("PyConverter dispatch: no arm matched")

    def extend(mut self, values: PyObjectPtr) raises:
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, PyConverter):
                if self._v[].isa[T]():
                    ref c = rebind[downcast[T, PyConverter]](self._v[][T])
                    return c.extend(values)
        raise Error("PyConverter dispatch: no arm matched")


# ---------------------------------------------------------------------------
# PyPrimitiveConverter — hot path for numeric/bool types
# ---------------------------------------------------------------------------


struct PyPrimitiveConverter[T: dt.PrimitiveType](PyConverter):
    var _builder: DynBuilder
    var _has_nulls: Bool
    var py: PyHelpers

    def __init__(out self, builder: DynBuilder, has_nulls: Bool = True):
        self._builder = builder
        self._has_nulls = has_nulls
        self.py = PyHelpers()

    def builder(
        ref self,
    ) -> ref[self._builder._ptr[][PrimitiveBuilder[Self.T]]] PrimitiveBuilder[
        Self.T
    ]:
        return self._builder.as_primitive[Self.T]()

    def extend(mut self, values: PyObjectPtr) raises:
        ref b = self.builder()
        var n = self.py.length(values)
        b.reserve(n)
        if self._has_nulls:
            for i in range(n):
                var item = self.py.list_getitem(values, i)
                if self.py.is_none(item):
                    b.unsafe_append_null()
                else:
                    b.unsafe_append(self.py.to_scalar[Self.T.native](item))
        else:
            for i in range(n):
                b.unsafe_append(
                    self.py.to_scalar[Self.T.native](
                        self.py.list_getitem(values, i)
                    )
                )

    def append(mut self, value: PyObjectPtr) raises:
        ref b = self.builder()
        if self.py.is_none(value):
            b.append_null()
        else:
            b.append(self.py.to_scalar[Self.T.native](value))


# ---------------------------------------------------------------------------
# PyBoolConverter — hot path for boolean types
# ---------------------------------------------------------------------------


struct PyBoolConverter(PyConverter):
    var _builder: DynBuilder
    var _has_nulls: Bool
    var py: PyHelpers

    def __init__(out self, builder: DynBuilder, has_nulls: Bool = True):
        self._builder = builder
        self._has_nulls = has_nulls
        self.py = PyHelpers()

    def builder(ref self) -> ref[self._builder._ptr[][BoolBuilder]] BoolBuilder:
        return self._builder.as_bool()

    def extend(mut self, values: PyObjectPtr) raises:
        ref b = self.builder()
        var n = self.py.length(values)
        b.reserve(n)
        if self._has_nulls:
            for i in range(n):
                var item = self.py.list_getitem(values, i)
                if self.py.is_none(item):
                    b.append_null()
                else:
                    b.append(Bool(self.py.to_scalar[DType.bool](item)))
        else:
            for i in range(n):
                b.append(
                    Bool(
                        self.py.to_scalar[DType.bool](
                            self.py.list_getitem(values, i)
                        )
                    )
                )

    def append(mut self, value: PyObjectPtr) raises:
        ref b = self.builder()
        if self.py.is_none(value):
            b.append_null()
        else:
            b.append(Bool(self.py.to_scalar[DType.bool](value)))


# ---------------------------------------------------------------------------
# PyStringConverter — hot path for UTF-8 strings
# ---------------------------------------------------------------------------


struct PyStringConverter(PyConverter):
    var _builder: DynBuilder
    var _has_nulls: Bool
    var py: PyHelpers

    def __init__(out self, builder: DynBuilder, has_nulls: Bool = True):
        self._builder = builder
        self._has_nulls = has_nulls
        self.py = PyHelpers()

    def builder(
        ref self,
    ) -> ref[self._builder._ptr[][StringBuilder]] StringBuilder:
        return self._builder.as_string()

    @always_inline
    def _count_bytes(mut self, values: PyObjectPtr, n: Int) raises -> Int:
        var total = 0
        for i in range(n):
            var item = self.py.list_getitem(values, i)
            if not self.py.is_none(item):
                total += self.py.to_string_slice(item).byte_length()
        return total

    def extend(mut self, values: PyObjectPtr) raises:
        var n = self.py.length(values)
        # `_count_bytes` mutably borrows self, so compute it before binding the
        # interior builder ref (which the call would otherwise invalidate).
        var nbytes = self._count_bytes(values, n)
        ref b = self.builder()
        b.reserve(n)
        b.reserve_bytes(nbytes)

        if self._has_nulls:
            for i in range(n):
                var item = self.py.list_getitem(values, i)
                if self.py.is_none(item):
                    b.unsafe_append_null()
                else:
                    b.unsafe_append(self.py.to_string_slice(item))
        else:
            for i in range(n):
                b.unsafe_append(
                    self.py.to_string_slice(self.py.list_getitem(values, i))
                )

    def append(mut self, value: PyObjectPtr) raises:
        ref b = self.builder()
        if self.py.is_none(value):
            b.append_null()
        else:
            b.append(self.py.to_string_slice(value))


# ---------------------------------------------------------------------------
# PyBinaryConverter — hot path for binary (bytes) types
# ---------------------------------------------------------------------------


struct PyBinaryConverter(PyConverter):
    var _builder: DynBuilder
    var _has_nulls: Bool
    var py: PyHelpers

    def __init__(out self, builder: DynBuilder, has_nulls: Bool = True):
        self._builder = builder
        self._has_nulls = has_nulls
        self.py = PyHelpers()

    def builder(
        ref self,
    ) -> ref[self._builder._ptr[][BinaryBuilder]] BinaryBuilder:
        return self._builder.as_binary()

    @always_inline
    def _count_bytes(mut self, values: PyObjectPtr, n: Int) raises -> Int:
        var total = 0
        for i in range(n):
            var item = self.py.list_getitem(values, i)
            if not self.py.is_none(item):
                total += self.py.to_bytes_slice(item).byte_length()
        return total

    def extend(mut self, values: PyObjectPtr) raises:
        var n = self.py.length(values)
        var nbytes = self._count_bytes(values, n)
        ref b = self.builder()
        b.reserve(n)
        b.reserve_bytes(nbytes)
        if self._has_nulls:
            for i in range(n):
                var item = self.py.list_getitem(values, i)
                if self.py.is_none(item):
                    b.unsafe_append_null()
                else:
                    b.unsafe_append(self.py.to_bytes_slice(item))
        else:
            for i in range(n):
                b.unsafe_append(
                    self.py.to_bytes_slice(self.py.list_getitem(values, i))
                )

    def append(mut self, value: PyObjectPtr) raises:
        ref b = self.builder()
        if self.py.is_none(value):
            b.append_null()
        else:
            b.append(self.py.to_bytes_slice(value))


# ---------------------------------------------------------------------------
# PyListConverter — variable-length list conversion
# ---------------------------------------------------------------------------


struct PyListConverter(PyConverter):
    var _builder: DynBuilder
    var _child: PyAnyConverter
    var _has_nulls: Bool
    var py: PyHelpers

    def __init__(out self, builder: DynBuilder, has_nulls: Bool = True) raises:
        self._builder = builder
        var child_builder = builder.as_list().values()
        var child_dtype = (
            builder.as_list().dtype().as_list().value_type().copy()
        )
        self._child = PyAnyConverter(child_builder, child_dtype, True)
        self._has_nulls = has_nulls
        self.py = PyHelpers()

    def builder(
        ref self,
    ) -> ref[self._builder._ptr[][ListBuilder]] ListBuilder:
        return self._builder.as_list()

    def extend(mut self, values: PyObjectPtr) raises:
        ref b = self.builder()
        var n = self.py.length(values)
        b.reserve(n)
        if self._has_nulls:
            for i in range(n):
                var item = self.py.list_getitem(values, i)
                if self.py.is_none(item):
                    b.unsafe_append_null()
                else:
                    self._child.extend(item)
                    b.unsafe_append_valid()
        else:
            for i in range(n):
                self._child.extend(self.py.list_getitem(values, i))
                b.unsafe_append_valid()

    def append(mut self, value: PyObjectPtr) raises:
        ref b = self.builder()
        if self._has_nulls and self.py.is_none(value):
            b.append_null()
        else:
            self._child.extend(value)
            b.append_valid()


# ---------------------------------------------------------------------------
# PyFixedSizeListConverter — fixed-size list conversion
# ---------------------------------------------------------------------------


struct PyFixedSizeListConverter(PyConverter):
    var _builder: DynBuilder
    var _child: PyAnyConverter
    var _has_nulls: Bool
    var _list_size: Int
    var py: PyHelpers

    def __init__(out self, builder: DynBuilder, has_nulls: Bool = True) raises:
        self._builder = builder
        ref fsl = builder.as_fixed_size_list().dtype().as_fixed_size_list()
        var child_builder = builder.as_fixed_size_list().values()
        var child_dtype = fsl.value_type()
        self._child = PyAnyConverter(child_builder, child_dtype, True)
        self._has_nulls = has_nulls
        self._list_size = fsl.size
        self.py = PyHelpers()

    def extend(mut self, values: PyObjectPtr) raises:
        ref b = self._builder.as_fixed_size_list()
        var n = self.py.length(values)
        b.reserve(n)
        if self._has_nulls:
            for i in range(n):
                var item = self.py.list_getitem(values, i)
                if self.py.is_none(item):
                    for _ in range(self._list_size):
                        self._child.append(self.py.none_ptr)
                    b.unsafe_append_null()
                else:
                    self._child.extend(item)
                    b.unsafe_append_valid()
        else:
            for i in range(n):
                self._child.extend(self.py.list_getitem(values, i))
                b.unsafe_append_valid()

    def append(mut self, value: PyObjectPtr) raises:
        ref b = self._builder.as_fixed_size_list()
        if self._has_nulls and self.py.is_none(value):
            for _ in range(self._list_size):
                self._child.append(self.py.none_ptr)
            b.append_null()
        else:
            self._child.extend(value)
            b.append_valid()


# ---------------------------------------------------------------------------
# PyStructConverter — struct/dict conversion
# ---------------------------------------------------------------------------


struct PyStructConverter(PyConverter):
    var _builder: DynBuilder
    var _children: List[PyAnyConverter]
    var _field_keys: List[PythonObject]
    var py: PyHelpers

    # Explicit (empty) destructor so this type is Deinitable despite the
    # `List[PyAnyConverter]` field (PyAnyConverter is recursive, so not
    # implicitly deletable); fields are still destroyed automatically.
    def __deinit__(deinit self):
        pass

    def __init__(out self, builder: DynBuilder) raises:
        self._builder = builder
        var dtype = builder.as_struct().dtype()
        ref st = dtype.as_struct()
        var n = len(st.fields)
        var children = List[PyAnyConverter](capacity=n)
        var field_keys = List[PythonObject](capacity=n)
        for i in range(n):
            var child_builder = builder.as_struct().field_builder(i)
            var child_dtype = st.fields[i].dtype.copy()
            children.append(PyAnyConverter(child_builder, child_dtype^))
            field_keys.append(PythonObject(st.fields[i].name))
        self._children = children^
        self._field_keys = field_keys^
        self.py = PyHelpers()

    def builder(
        ref self,
    ) -> ref[self._builder._ptr[][StructBuilder]] StructBuilder:
        return self._builder.as_struct()

    def extend(mut self, values: PyObjectPtr) raises:
        var n_fields = len(self._children)
        ref b = self.builder()
        var n = self.py.length(values)
        b.reserve(n)
        for row in range(n):
            var item = self.py.list_getitem(values, row)
            if self.py.is_none(item):
                for i in range(n_fields):
                    self._children[i].append(self.py.none_ptr)
                b.unsafe_append_null()
            else:
                for i in range(n_fields):
                    self._children[i].append(
                        self.py.dict_getitem(item, self._field_keys[i]._obj_ptr)
                    )
                b.unsafe_append_valid()

    def append(mut self, value: PyObjectPtr) raises:
        ref b = self.builder()
        if self.py.is_none(value):
            for i in range(len(self._children)):
                self._children[i].append(self.py.none_ptr)
            b.append_null()
        else:
            for i in range(len(self._children)):
                self._children[i].append(
                    self.py.dict_getitem(value, self._field_keys[i]._obj_ptr)
                )
            b.append_valid()


# ---------------------------------------------------------------------------
# Arrow C Data Interface: __arrow_c_array__ and __arrow_c_schema__ wrappers
#
# Each typed array needs a thin wrapper because def_method requires the exact
# pointer type.
# ---------------------------------------------------------------------------


def arrow_c_array[
    T: Deinitable, //, to_array_fn: def(T) thin -> DynArray
](py_self: PythonObject, requested_schema: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[T]()
    var arr = to_array_fn(ptr[])
    var schema_cap = CArrowSchema.from_dtype(arr.dtype()).to_pycapsule()
    var array_cap = CArrowArray.from_array(arr).to_pycapsule()
    return Python.tuple(schema_cap, array_cap)


def arrow_c_schema[
    T: Deinitable, //, type_fn: def(T) thin -> dt.DynType
](py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[T]()
    return CArrowSchema.from_dtype(type_fn(ptr[])).to_pycapsule()


# ---------------------------------------------------------------------------
# Public Python functions
# ---------------------------------------------------------------------------


def infer_type(obj: PythonObject) raises -> PythonObject:
    var inferrer = PyInferrer()
    return inferrer.infer(obj).to_python_object()


def array(obj: PythonObject, type: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var type_given = not type.__is__(builtins.None)
    if not type_given:
        try:
            return DynArray(py=obj).to_python_object()
        except:
            pass

    var dtype: dt.DynType
    var has_nulls = True
    if type_given:
        dtype = type.downcast_value_ptr[dt.DynType]()[].copy()
    else:
        var inferrer = PyInferrer()
        dtype = inferrer.infer(obj)
        has_nulls = inferrer.none_count > 0

    if dtype.is_null():
        if type_given:
            return NullArray(length=len(obj)).to_dyn().to_python_object()
        raise Error(
            "cannot build array: sequence is empty or all-None"
            " (provide type= explicitly)"
        )

    var builder = DynBuilder(dtype, len(obj))
    var converter = PyAnyConverter(builder, dtype, has_nulls)
    converter.extend(obj._obj_ptr)
    return builder.finish().to_python_object()


def _build_mask_array(
    mask_obj: PythonObject, n: Int
) raises -> Optional[BoolArray]:
    """Build a BoolArray from a Python mask list (True/1=null) or None.

    PyArrow mask convention: True means null, False means valid.
    """
    var builtins = Python.import_module("builtins")
    if mask_obj is builtins.None:
        return None
    var builder = BoolBuilder(n)
    for i in range(n):
        builder.append(Bool(Int(py=mask_obj[i]) != 0))
    return builder.finish()


def _build_offsets_array(offsets_obj: PythonObject) raises -> Int32Array:
    """Build an Int32Array of offsets from a Python list of integers."""
    var n = Int(py=offsets_obj.__len__())
    var builder = Int32Builder(n)
    for i in range(n):
        builder.unsafe_append(Int32(Int(py=offsets_obj[i])))
    return builder.finish()


def list_array_from_arrays(
    offsets: PythonObject, values: PythonObject, mask: PythonObject
) raises -> PythonObject:
    var offsets_arr = _build_offsets_array(offsets)
    var child = DynArray(py=values)
    var n = offsets_arr.length - 1
    var mask_opt = _build_mask_array(mask, n)
    return (
        ListArray.from_arrays(offsets_arr, child^, mask_opt^)
        .to_dyn()
        .to_python_object()
    )


def fixed_size_list_array_from_arrays(
    values: PythonObject, type: PythonObject, mask: PythonObject
) raises -> PythonObject:
    var child = DynArray(py=values)
    var dtype = type.downcast_value_ptr[dt.DynType]()[].copy()
    var list_size = dtype.as_fixed_size_list().size
    var n = child.length() // list_size if list_size > 0 else 0
    var mask_opt = _build_mask_array(mask, n)
    return (
        FixedSizeListArray.from_arrays(child^, list_size, mask_opt^)
        .to_dyn()
        .to_python_object()
    )


def struct_array_from_arrays(
    arrays: PythonObject, fields: PythonObject, mask: PythonObject
) raises -> PythonObject:
    var n_fields = Int(py=arrays.__len__())
    var children = List[DynArray]()
    var fields_list = List[dt.Field]()
    var n = 0
    for i in range(n_fields):
        var arr = DynArray(py=arrays[i])
        if i == 0:
            n = arr.length()
        children.append(arr^)
        fields_list.append(fields[i].downcast_value_ptr[dt.Field]()[].copy())
    var mask_opt = _build_mask_array(mask, n)
    return (
        StructArray.from_arrays(children^, fields_list^, mask_opt^)
        .to_dyn()
        .to_python_object()
    )


def _any_array_str(py_self: PythonObject) raises -> PythonObject:
    var ptr = py_self.downcast_value_ptr[DynArray]()
    return PythonObject(String(ptr[]))


def _array_to_device(self: DynArray, ctx: ExecContext) raises -> DynArray:
    return self.to_device(ctx.device.value())


def _array_to_cpu(self: DynArray, ctx: ExecContext) raises -> DynArray:
    return self.to_cpu(ctx.device.value())


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add array types and constructors to the Python API."""

    # --- Array (type-erased DynArray) ---
    ref array_py = mb.add_type[DynArray]("Array")
    _ = (
        array_py.def_method[pymethod[DynArray.__len__]()]("__len__")
        .def_method[_any_array_getitem_py]("__getitem__")
        .def_method[pymethod[DynArray.null_count]()]("null_count")
        .def_method[pymethod[DynArray.dtype]()]("type")
        .def_method[pymethod[DynArray.is_valid]()]("is_valid")
        .def_method[pymethod[DynArray.slice]()]("slice")
        .def_method[pymethod[_array_to_device]()]("to_device")
        .def_method[pymethod[_array_to_cpu]()]("to_cpu")
        .def_method[arrow_c_array[_any_to_array]]("__arrow_c_array__")
        .def_method[arrow_c_schema[_any_dtype]]("__arrow_c_schema__")
    )
    _ = array_py.def_method[_any_array_str]("__str__").def_method[
        _any_array_str
    ]("__repr__")
    # var array_sp = SequenceProtocolBuilder[DynArray](array_py)
    # _ = array_sp.def_len[DynArray.__len__]().def_getitem[_any_array_getitem]()

    mb.def_function[infer_type]("infer_type")
    mb.def_function[array]("array")
    mb.def_function[list_array_from_arrays]("list_array_from_arrays")
    mb.def_function[fixed_size_list_array_from_arrays](
        "fixed_size_list_array_from_arrays"
    )
    mb.def_function[struct_array_from_arrays]("struct_array_from_arrays")
