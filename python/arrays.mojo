from std.python import (
    PythonObject,
    ConvertibleToPython,
    ConvertibleFromPython,
    Python,
)
from std.python.bindings import PythonModuleBuilder
from std.collections import OwnedKwargsDict
from std.python._cpython import (
    CPython,
    PyObjectPtr,
    PyTypeObject,
    PyTypeObjectPtr,
)
from std.memory import ArcPointer, alloc
from marrow.c_data import CArrowSchema, CArrowArray
from marrow.arrays import (
    AnyArray,
    PrimitiveArray,
    BoolArray,
    Int8Array,
    Int16Array,
    Int32Array,
    Int64Array,
    UInt8Array,
    UInt16Array,
    UInt32Array,
    UInt64Array,
    Float32Array,
    Float64Array,
    StringArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
    ChunkedArray,
)
from marrow.builders import (
    AnyBuilder,
    PrimitiveBuilder,
    StringBuilder,
    ListBuilder,
    StructBuilder,
    make_builder,
)
from marrow.scalars import AnyScalar, ListScalar
from marrow.dtypes import DataType
import marrow.dtypes as dt
from pontoneer import SequenceProtocolBuilder
from helpers import pymethod, def_display


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

    def __init__(out self):
        self.py = Python()
        ref cpy = self.py.cpython()
        self.none_ptr = cpy.Py_None()
        self._unicode_type = cpy.lib.get_symbol[PyTypeObject]("PyUnicode_Type")
        self._bytes_type = cpy.lib.get_symbol[PyTypeObject]("PyBytes_Type")
        self._list_type = cpy.lib.get_symbol[PyTypeObject]("PyList_Type")
        self._tuple_type = cpy.lib.get_symbol[PyTypeObject]("PyTuple_Type")
        self._dict_type = cpy.PyDict_Type()

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
            return Scalar[dtype](val != 0)
        elif dtype.is_floating_point():
            var val = self.cpy().PyFloat_AsDouble(ptr)
            if val == -1.0:
                self.raise_on_error()
            return Scalar[dtype](val)
        else:
            var val = self.cpy().PyLong_AsSsize_t(ptr)
            if val == -1:
                self.raise_on_error()
            # uint64 is a special case: Scalar[uint64](-1).cast[int64]() == -1 == val, so the
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
# Custom wrappers for methods that need special return-type handling
# ---------------------------------------------------------------------------


def _prim_scalar_getitem[T: DataType](
    ptr: UnsafePointer[PrimitiveArray[T], MutAnyOrigin],
    index: Int,
) raises -> PythonObject:
    """Return a Python Scalar for the element at the given index."""
    var n = len(ptr[])
    if index < 0 or index >= n:
        raise Error(t"index {index} out of bounds for length {n}")
    return AnyScalar(ptr[][index]).to_python_object()


def _str_getitem(
    ptr: UnsafePointer[StringArray, MutAnyOrigin],
    index: Int,
) raises -> PythonObject:
    var n = len(ptr[])
    if index < 0 or index >= n:
        raise Error(t"index {index} out of bounds for length {n}")
    return AnyScalar(ptr[][index]).to_python_object()


def _list_getitem(
    ptr: UnsafePointer[ListArray, MutAnyOrigin],
    index: Int,
) raises -> PythonObject:
    var n = len(ptr[])
    if index < 0 or index >= n:
        raise Error(t"index {index} out of bounds for length {n}")
    return AnyScalar(ptr[][index]).to_python_object()


def _fsl_getitem(
    ptr: UnsafePointer[FixedSizeListArray, MutAnyOrigin],
    index: Int,
) raises -> PythonObject:
    var n = len(ptr[])
    if index < 0 or index >= n:
        raise Error(t"index {index} out of bounds for length {n}")
    return AnyScalar(
        ListScalar(value=ptr[].unsafe_get(index), is_valid=ptr[].is_valid(index))
    ).to_python_object()


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
        var key_raw = alloc[PyObjectPtr](1)
        var val_raw = alloc[PyObjectPtr](1)
        var pos_ptr = UnsafePointer(to=pos)
        for _ in range(n):
            _ = cpy.PyDict_Next(dict_ptr, pos_ptr, key_raw, val_raw)
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
        key_raw.free()
        val_raw.free()

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

    def _get_type(self) raises -> dt.DataType:
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
            return dt.struct_(fields)
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

    def infer(mut self, obj: PythonObject) raises -> dt.DataType:
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


trait PyConverter(ImplicitlyDestructible, Movable):
    def append(mut self, value: PyObjectPtr) raises:
        ...

    def extend(mut self, values: PyObjectPtr) raises:
        ...

    def builder(mut self) -> AnyBuilder:
        ...


# ---------------------------------------------------------------------------
# PyAnyConverter — type-erased converter with dynamic dispatch
# ---------------------------------------------------------------------------


struct PyAnyConverter(ImplicitlyCopyable, Movable):
    """Type-erased converter dispatching through def-ptr trampolines."""

    var _data: ArcPointer[NoneType]
    var _virt_append: def(ArcPointer[NoneType], PyObjectPtr) raises
    var _virt_extend: def(ArcPointer[NoneType], PyObjectPtr) raises
    var _virt_builder: def(ArcPointer[NoneType]) -> AnyBuilder
    var _virt_drop: def(var ArcPointer[NoneType])

    @staticmethod
    def _tramp_append[
        T: PyConverter
    ](ptr: ArcPointer[NoneType], value: PyObjectPtr) raises:
        rebind[ArcPointer[T]](ptr)[].append(value)

    @staticmethod
    def _tramp_extend[
        T: PyConverter
    ](ptr: ArcPointer[NoneType], values: PyObjectPtr) raises:
        rebind[ArcPointer[T]](ptr)[].extend(values)

    @staticmethod
    def _tramp_builder[T: PyConverter](ptr: ArcPointer[NoneType]) -> AnyBuilder:
        return rebind[ArcPointer[T]](ptr.copy())[].builder()

    @staticmethod
    def _tramp_drop[T: PyConverter](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    @implicit
    def __init__[T: PyConverter](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_append = Self._tramp_append[T]
        self._virt_extend = Self._tramp_extend[T]
        self._virt_builder = Self._tramp_builder[T]
        self._virt_drop = Self._tramp_drop[T]

    def __copyinit__(out self, read copy: Self):
        self._data = copy._data.copy()
        self._virt_append = copy._virt_append
        self._virt_extend = copy._virt_extend
        self._virt_builder = copy._virt_builder
        self._virt_drop = copy._virt_drop

    def append(mut self, value: PyObjectPtr) raises:
        self._virt_append(self._data, value)

    def extend(mut self, values: PyObjectPtr) raises:
        self._virt_extend(self._data, values)

    def builder(mut self) -> AnyBuilder:
        return self._virt_builder(self._data)

    def __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# PyPrimitiveConverter — hot path for numeric/bool types
# ---------------------------------------------------------------------------


struct PyPrimitiveConverter[T: dt.DataType](PyConverter):
    var _builder: ArcPointer[PrimitiveBuilder[Self.T]]
    var _has_nulls: Bool
    var py: PyHelpers

    def __init__(out self, builder: AnyBuilder, has_nulls: Bool = True):
        self._builder = builder.downcast[PrimitiveBuilder[Self.T]]()
        self._has_nulls = has_nulls
        self.py = PyHelpers()

    def builder(mut self) -> AnyBuilder:
        return self._builder.copy()

    def extend(mut self, values: PyObjectPtr) raises:
        ref b = self._builder[]
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
        ref b = self._builder[]
        if self.py.is_none(value):
            b.append_null()
        else:
            b.append(self.py.to_scalar[Self.T.native](value))


# ---------------------------------------------------------------------------
# PyStringConverter — hot path for UTF-8 strings
# ---------------------------------------------------------------------------


struct PyStringConverter(PyConverter):
    var _builder: ArcPointer[StringBuilder]
    var _has_nulls: Bool
    var py: PyHelpers

    def __init__(out self, builder: AnyBuilder, has_nulls: Bool = True):
        self._builder = builder.downcast[StringBuilder]()
        self._has_nulls = has_nulls
        self.py = PyHelpers()

    def builder(mut self) -> AnyBuilder:
        return self._builder.copy()

    @always_inline
    def _count_bytes(mut self, values: PyObjectPtr, n: Int) raises -> Int:
        var total = 0
        for i in range(n):
            var item = self.py.list_getitem(values, i)
            if not self.py.is_none(item):
                total += len(self.py.to_string_slice(item))
        return total

    def extend(mut self, values: PyObjectPtr) raises:
        ref b = self._builder[]
        var n = self.py.length(values)

        b.reserve(n)
        b.reserve_bytes(self._count_bytes(values, n))

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
        ref b = self._builder[]
        if self.py.is_none(value):
            b.append_null()
        else:
            b.append(self.py.to_string_slice(value))


# ---------------------------------------------------------------------------
# PyListConverter — variable-length list conversion
# ---------------------------------------------------------------------------


struct PyListConverter(PyConverter):
    var _builder: ArcPointer[ListBuilder]
    var _child: PyAnyConverter
    var _has_nulls: Bool
    var py: PyHelpers

    def __init__(out self, builder: AnyBuilder, has_nulls: Bool = True) raises:
        self._builder = builder.downcast[ListBuilder]()
        var child_builder = self._builder[].values()
        self._child = make_converter(child_builder, True)
        self._has_nulls = has_nulls
        self.py = PyHelpers()

    def builder(mut self) -> AnyBuilder:
        return self._builder.copy()

    def extend(mut self, values: PyObjectPtr) raises:
        ref builder = self._builder[]
        var n = self.py.length(values)
        builder.reserve(n)
        if self._has_nulls:
            for i in range(n):
                var item = self.py.list_getitem(values, i)
                if self.py.is_none(item):
                    builder.unsafe_append_null()
                else:
                    self._child.extend(item)
                    builder.unsafe_append_valid()
        else:
            for i in range(n):
                self._child.extend(self.py.list_getitem(values, i))
                builder.unsafe_append_valid()

    def append(mut self, value: PyObjectPtr) raises:
        ref builder = self._builder[]
        if self._has_nulls and self.py.is_none(value):
            builder.append_null()
        else:
            self._child.extend(value)
            builder.append_valid()


# ---------------------------------------------------------------------------
# PyStructConverter — struct/dict conversion
# ---------------------------------------------------------------------------


struct PyStructConverter(PyConverter):
    var _builder: ArcPointer[StructBuilder]
    var _children: List[PyAnyConverter]
    var _field_keys: List[PythonObject]
    var py: PyHelpers

    def __init__(out self, builder: AnyBuilder) raises:
        self._builder = builder.downcast[StructBuilder]()
        var dtype = self._builder[].dtype()
        var n = len(dtype.fields)
        var children = List[PyAnyConverter](capacity=n)
        var field_keys = List[PythonObject](capacity=n)
        for i in range(n):
            var child_builder = self._builder[].field_builder(i)
            children.append(make_converter(child_builder))
            field_keys.append(PythonObject(dtype.fields[i].name))
        self._children = children^
        self._field_keys = field_keys^
        self.py = PyHelpers()

    def builder(mut self) -> AnyBuilder:
        return self._builder.copy()

    def extend(mut self, values: PyObjectPtr) raises:
        var n_fields = len(self._children)
        ref builder = self._builder[]
        var n = self.py.length(values)
        builder.reserve(n)
        for row in range(n):
            var item = self.py.list_getitem(values, row)
            if self.py.is_none(item):
                for i in range(n_fields):
                    self._children[i].append(self.py.none_ptr)
                builder.unsafe_append_null()
            else:
                for i in range(n_fields):
                    self._children[i].append(
                        self.py.dict_getitem(item, self._field_keys[i]._obj_ptr)
                    )
                builder.unsafe_append_valid()

    def append(mut self, value: PyObjectPtr) raises:
        ref builder = self._builder[]
        if self.py.is_none(value):
            for i in range(len(self._children)):
                self._children[i].append(self.py.none_ptr)
            builder.append_null()
        else:
            for i in range(len(self._children)):
                self._children[i].append(
                    self.py.dict_getitem(value, self._field_keys[i]._obj_ptr)
                )
            builder.append_valid()


# ---------------------------------------------------------------------------
# Factory functions for converters
# ---------------------------------------------------------------------------


def make_converter(builder: AnyBuilder, has_nulls: Bool = True) raises -> PyAnyConverter:
    """Create a converter wrapping an existing builder (shared ArcPointer)."""
    dtype = builder.dtype()
    comptime for T in dt.primitive_dtypes:
        if dtype == T:
            return PyPrimitiveConverter[T](builder, has_nulls)
    if dtype.is_string():
        return PyStringConverter(builder, has_nulls)
    elif dtype.is_list():
        return PyListConverter(builder, has_nulls)
    elif dtype.is_struct():
        return PyStructConverter(builder)
    else:
        raise Error("unsupported type: ", dtype)


# ---------------------------------------------------------------------------
# Arrow C Data Interface: __arrow_c_array__ and __arrow_c_schema__ wrappers
#
# Each typed array needs a thin wrapper because def_method requires the exact
# pointer type.
# ---------------------------------------------------------------------------


def arrow_c_array[T: AnyType, //, to_array_fn: def(T) -> AnyArray](
    ptr: UnsafePointer[T, MutAnyOrigin], requested_schema: PythonObject
) raises -> PythonObject:
    var arr = to_array_fn(ptr[])
    var schema_cap = CArrowSchema.from_dtype(arr.dtype()).to_pycapsule()
    var array_cap = CArrowArray.from_array(arr).to_pycapsule()
    return Python.tuple(schema_cap, array_cap)


def arrow_c_schema[T: AnyType, //, type_fn: def(T) -> dt.DataType](
    ptr: UnsafePointer[T, MutAnyOrigin]
) raises -> PythonObject:
    return CArrowSchema.from_dtype(type_fn(ptr[])).to_pycapsule()


# TODO: maybe introduce an AnyArray trait and rename AnyArray struct to AnyArray
def _to_array[D: dt.DataType](arr: PrimitiveArray[D]) -> AnyArray:
    return arr.copy().to_any()


def _str_to_array(arr: StringArray) -> AnyArray:
    return arr.copy().to_any()


def _list_to_array(arr: ListArray) -> AnyArray:
    return arr.copy().to_any()


def _fsl_to_array(arr: FixedSizeListArray) -> AnyArray:
    return arr.copy().to_any()


def _struct_to_array(arr: StructArray) -> AnyArray:
    return arr.copy().to_any()




# ---------------------------------------------------------------------------
# Public Python functions
# ---------------------------------------------------------------------------



def infer_type(obj: PythonObject) raises -> PythonObject:
    var inferrer = PyInferrer()
    return inferrer.infer(obj).to_python_object()


def array(
    obj: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    # Try converting directly (handles marrow arrays and __arrow_c_array__).
    # Skip when type= is given since the user wants explicit type control.
    if not kwargs.find("type"):
        try:
            return AnyArray(py=obj).to_python_object()
        except:
            pass

    # Fall back to building from a Python sequence.
    var dtype: dt.DataType
    var has_nulls = True
    if opt := kwargs.find("type"):
        dtype = opt.value().downcast_value_ptr[dt.DataType]()[]
    else:
        var inferrer = PyInferrer()
        dtype = inferrer.infer(obj)
        has_nulls = inferrer.none_count > 0

    if dtype.is_null():
        raise Error(
            "cannot build array: sequence is empty or all-None"
            " (provide type= explicitly)"
        )

    var builder = make_builder(dtype, len(obj))
    var converter = make_converter(builder, has_nulls)
    converter.extend(obj._obj_ptr)
    return builder.finish().to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add array types and constructors to the Python API."""

    # --- BoolArray ---
    ref bool_array_py = mb.add_type[BoolArray]("BoolArray")
    _ = (
        bool_array_py.def_method[pymethod[BoolArray.null_count]()]("null_count")
        .def_method[pymethod[BoolArray.type]()]("type")
        .def_method[pymethod[BoolArray.is_valid]()]("is_valid")
        .def_method[pymethod[BoolArray.slice]()]("slice")
        .def_method[pymethod[BoolArray.true_count]()]("true_count")
        .def_method[pymethod[BoolArray.false_count]()]("false_count")
        .def_method[arrow_c_array[_to_array[dt.bool_]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[BoolArray.type]]("__arrow_c_schema__")
    )
    _ = def_display[BoolArray](bool_array_py)
    var bool_array_sp = SequenceProtocolBuilder[BoolArray](bool_array_py)
    _ = bool_array_sp.def_len[BoolArray.__len__]().def_getitem[
        _prim_scalar_getitem[dt.bool_]
    ]()

    # --- Numeric PrimitiveArrays ---
    ref int8_array_py = mb.add_type[Int8Array]("Int8Array")
    _ = (
        int8_array_py.def_method[pymethod[Int8Array.null_count]()]("null_count")
        .def_method[pymethod[Int8Array.type]()]("type")
        .def_method[pymethod[Int8Array.is_valid]()]("is_valid")
        .def_method[pymethod[Int8Array.slice]()]("slice")
        .def_method[arrow_c_array[_to_array[dt.int8]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[Int8Array.type]]("__arrow_c_schema__")
    )
    _ = def_display[Int8Array](int8_array_py)
    var int8_array_sp = SequenceProtocolBuilder[Int8Array](int8_array_py)
    _ = int8_array_sp.def_len[Int8Array.__len__]().def_getitem[
        _prim_scalar_getitem[dt.int8]
    ]()

    ref int16_array_py = mb.add_type[Int16Array]("Int16Array")
    _ = (
        int16_array_py.def_method[pymethod[Int16Array.null_count]()](
            "null_count"
        )
        .def_method[pymethod[Int16Array.type]()]("type")
        .def_method[pymethod[Int16Array.is_valid]()]("is_valid")
        .def_method[pymethod[Int16Array.slice]()]("slice")
        .def_method[arrow_c_array[_to_array[dt.int16]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[Int16Array.type]]("__arrow_c_schema__")
    )
    _ = def_display[Int16Array](int16_array_py)
    var int16_array_sp = SequenceProtocolBuilder[Int16Array](int16_array_py)
    _ = int16_array_sp.def_len[Int16Array.__len__]().def_getitem[
        _prim_scalar_getitem[dt.int16]
    ]()

    ref int32_array_py = mb.add_type[Int32Array]("Int32Array")
    _ = (
        int32_array_py.def_method[pymethod[Int32Array.null_count]()](
            "null_count"
        )
        .def_method[pymethod[Int32Array.type]()]("type")
        .def_method[pymethod[Int32Array.is_valid]()]("is_valid")
        .def_method[pymethod[Int32Array.slice]()]("slice")
        .def_method[arrow_c_array[_to_array[dt.int32]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[Int32Array.type]]("__arrow_c_schema__")
    )
    _ = def_display[Int32Array](int32_array_py)
    var int32_array_sp = SequenceProtocolBuilder[Int32Array](int32_array_py)
    _ = int32_array_sp.def_len[Int32Array.__len__]().def_getitem[
        _prim_scalar_getitem[dt.int32]
    ]()

    ref int64_array_py = mb.add_type[Int64Array]("Int64Array")
    _ = (
        int64_array_py.def_method[pymethod[Int64Array.null_count]()](
            "null_count"
        )
        .def_method[pymethod[Int64Array.type]()]("type")
        .def_method[pymethod[Int64Array.is_valid]()]("is_valid")
        .def_method[pymethod[Int64Array.slice]()]("slice")
        .def_method[arrow_c_array[_to_array[dt.int64]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[Int64Array.type]]("__arrow_c_schema__")
    )
    _ = def_display[Int64Array](int64_array_py)
    var int64_array_sp = SequenceProtocolBuilder[Int64Array](int64_array_py)
    _ = int64_array_sp.def_len[Int64Array.__len__]().def_getitem[
        _prim_scalar_getitem[dt.int64]
    ]()

    ref uint8_array_py = mb.add_type[UInt8Array]("UInt8Array")
    _ = (
        uint8_array_py.def_method[pymethod[UInt8Array.null_count]()](
            "null_count"
        )
        .def_method[pymethod[UInt8Array.type]()]("type")
        .def_method[pymethod[UInt8Array.is_valid]()]("is_valid")
        .def_method[pymethod[UInt8Array.slice]()]("slice")
        .def_method[arrow_c_array[_to_array[dt.uint8]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[UInt8Array.type]]("__arrow_c_schema__")
    )
    _ = def_display[UInt8Array](uint8_array_py)
    var uint8_array_sp = SequenceProtocolBuilder[UInt8Array](uint8_array_py)
    _ = uint8_array_sp.def_len[UInt8Array.__len__]().def_getitem[
        _prim_scalar_getitem[dt.uint8]
    ]()

    ref uint16_array_py = mb.add_type[UInt16Array]("UInt16Array")
    _ = (
        uint16_array_py.def_method[pymethod[UInt16Array.null_count]()](
            "null_count"
        )
        .def_method[pymethod[UInt16Array.type]()]("type")
        .def_method[pymethod[UInt16Array.is_valid]()]("is_valid")
        .def_method[pymethod[UInt16Array.slice]()]("slice")
        .def_method[arrow_c_array[_to_array[dt.uint16]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[UInt16Array.type]]("__arrow_c_schema__")
    )
    _ = def_display[UInt16Array](uint16_array_py)
    var uint16_array_sp = SequenceProtocolBuilder[UInt16Array](uint16_array_py)
    _ = uint16_array_sp.def_len[UInt16Array.__len__]().def_getitem[
        _prim_scalar_getitem[dt.uint16]
    ]()

    ref uint32_array_py = mb.add_type[UInt32Array]("UInt32Array")
    _ = (
        uint32_array_py.def_method[pymethod[UInt32Array.null_count]()](
            "null_count"
        )
        .def_method[pymethod[UInt32Array.type]()]("type")
        .def_method[pymethod[UInt32Array.is_valid]()]("is_valid")
        .def_method[pymethod[UInt32Array.slice]()]("slice")
        .def_method[arrow_c_array[_to_array[dt.uint32]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[UInt32Array.type]]("__arrow_c_schema__")
    )
    _ = def_display[UInt32Array](uint32_array_py)
    var uint32_array_sp = SequenceProtocolBuilder[UInt32Array](uint32_array_py)
    _ = uint32_array_sp.def_len[UInt32Array.__len__]().def_getitem[
        _prim_scalar_getitem[dt.uint32]
    ]()

    ref uint64_array_py = mb.add_type[UInt64Array]("UInt64Array")
    _ = (
        uint64_array_py.def_method[pymethod[UInt64Array.null_count]()](
            "null_count"
        )
        .def_method[pymethod[UInt64Array.type]()]("type")
        .def_method[pymethod[UInt64Array.is_valid]()]("is_valid")
        .def_method[pymethod[UInt64Array.slice]()]("slice")
        .def_method[arrow_c_array[_to_array[dt.uint64]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[UInt64Array.type]]("__arrow_c_schema__")
    )
    _ = def_display[UInt64Array](uint64_array_py)
    var uint64_array_sp = SequenceProtocolBuilder[UInt64Array](uint64_array_py)
    _ = uint64_array_sp.def_len[UInt64Array.__len__]().def_getitem[
        _prim_scalar_getitem[dt.uint64]
    ]()

    ref float32_array_py = mb.add_type[Float32Array]("Float32Array")
    _ = (
        float32_array_py.def_method[pymethod[Float32Array.null_count]()](
            "null_count"
        )
        .def_method[pymethod[Float32Array.type]()]("type")
        .def_method[pymethod[Float32Array.is_valid]()]("is_valid")
        .def_method[pymethod[Float32Array.slice]()]("slice")
        .def_method[arrow_c_array[_to_array[dt.float32]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[Float32Array.type]]("__arrow_c_schema__")
    )
    _ = def_display[Float32Array](float32_array_py)
    var float32_array_sp = SequenceProtocolBuilder[Float32Array](
        float32_array_py
    )
    _ = float32_array_sp.def_len[Float32Array.__len__]().def_getitem[
        _prim_scalar_getitem[dt.float32]
    ]()

    ref float64_array_py = mb.add_type[Float64Array]("Float64Array")
    _ = (
        float64_array_py.def_method[pymethod[Float64Array.null_count]()](
            "null_count"
        )
        .def_method[pymethod[Float64Array.type]()]("type")
        .def_method[pymethod[Float64Array.is_valid]()]("is_valid")
        .def_method[pymethod[Float64Array.slice]()]("slice")
        .def_method[arrow_c_array[_to_array[dt.float64]]]("__arrow_c_array__")
        .def_method[arrow_c_schema[Float64Array.type]]("__arrow_c_schema__")
    )
    _ = def_display[Float64Array](float64_array_py)
    var float64_array_sp = SequenceProtocolBuilder[Float64Array](
        float64_array_py
    )
    _ = float64_array_sp.def_len[Float64Array.__len__]().def_getitem[
        _prim_scalar_getitem[dt.float64]
    ]()

    # --- StringArray ---
    ref str_array_py = mb.add_type[StringArray]("StringArray")
    _ = (
        str_array_py.def_method[pymethod[StringArray.null_count]()](
            "null_count"
        )
        .def_method[pymethod[StringArray.type]()]("type")
        .def_method[pymethod[StringArray.is_valid]()]("is_valid")
        .def_method[pymethod[StringArray.slice]()]("slice")
        .def_method[arrow_c_array[_str_to_array]]("__arrow_c_array__")
        .def_method[arrow_c_schema[StringArray.type]]("__arrow_c_schema__")
    )
    _ = def_display[StringArray](str_array_py)
    var str_array_sp = SequenceProtocolBuilder[StringArray](str_array_py)
    _ = str_array_sp.def_len[StringArray.__len__]().def_getitem[_str_getitem]()

    # --- ListArray ---
    ref list_array_py = mb.add_type[ListArray]("ListArray")
    _ = (
        list_array_py.def_method[pymethod[ListArray.null_count]()]("null_count")
        .def_method[pymethod[ListArray.type]()]("type")
        .def_method[pymethod[ListArray.is_valid]()]("is_valid")
        .def_method[pymethod[ListArray.slice]()]("slice")
        .def_method[pymethod[ListArray.flatten]()]("flatten")
        .def_method[pymethod[ListArray.value_lengths]()]("value_lengths")
        .def_method[arrow_c_array[_list_to_array]]("__arrow_c_array__")
        .def_method[arrow_c_schema[ListArray.type]]("__arrow_c_schema__")
    )
    _ = def_display[ListArray](list_array_py)
    var list_array_sp = SequenceProtocolBuilder[ListArray](list_array_py)
    _ = list_array_sp.def_len[ListArray.__len__]().def_getitem[_list_getitem]()

    # --- FixedSizeListArray ---
    ref fsl_array_py = mb.add_type[FixedSizeListArray]("FixedSizeListArray")
    _ = (
        fsl_array_py.def_method[pymethod[FixedSizeListArray.null_count]()](
            "null_count"
        )
        .def_method[pymethod[FixedSizeListArray.type]()]("type")
        .def_method[pymethod[FixedSizeListArray.is_valid]()]("is_valid")
        .def_method[pymethod[FixedSizeListArray.slice]()]("slice")
        .def_method[pymethod[FixedSizeListArray.flatten]()]("flatten")
        .def_method[arrow_c_array[_fsl_to_array]]("__arrow_c_array__")
        .def_method[arrow_c_schema[FixedSizeListArray.type]]("__arrow_c_schema__")
    )
    _ = def_display[FixedSizeListArray](fsl_array_py)
    var fsl_array_sp = SequenceProtocolBuilder[FixedSizeListArray](fsl_array_py)
    _ = fsl_array_sp.def_len[FixedSizeListArray.__len__]().def_getitem[
        _fsl_getitem
    ]()

    # --- StructArray ---
    ref struct_array_py = mb.add_type[StructArray]("StructArray")
    _ = (
        struct_array_py.def_method[pymethod[StructArray.null_count]()](
            "null_count"
        )
        .def_method[pymethod[StructArray.type]()]("type")
        .def_method[pymethod[StructArray.is_valid]()]("is_valid")
        .def_method[pymethod[StructArray.field]()]("field")
        .def_method[arrow_c_array[_struct_to_array]]("__arrow_c_array__")
        .def_method[arrow_c_schema[StructArray.type]]("__arrow_c_schema__")
    )
    _ = def_display[StructArray](struct_array_py)
    var struct_array_sp = SequenceProtocolBuilder[StructArray](struct_array_py)
    _ = struct_array_sp.def_len[StructArray.__len__]()

    mb.def_function[infer_type](
        "infer_type", docstring="infer_type(obj, /) -> DataType\n--\n\nInfer the Arrow type of a Python sequence."
    )
    mb.def_function[array](
        "array", docstring="array(obj, /, *, type=None) -> Array\n--\n\nCreate a marrow array from a Python sequence."
    )
