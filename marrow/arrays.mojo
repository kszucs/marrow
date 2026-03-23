"""Arrow columnar arrays — always immutable.

Every typed array (`PrimitiveArray`, `StringArray`, `ListArray`, `StructArray`)
is immutable.  To *build* an array incrementally, use the corresponding builder
from `marrow.builders` and call `finish()`.

`BoolArray` is an alias for `PrimitiveArray[bool_]`.

Array — the trait
-----------------
`Array` is the trait that all typed arrays implement.  It provides the common
read-only interface: `type()`, `null_count()`, `is_valid()`, and `as_any()`.

AnyArray — the type-erased handle
----------------------------------
`AnyArray` is the type-erased, immutable handle used for storage, exchange
(C Data Interface), and visitor dispatch.  The concrete typed array lives on
the heap behind an `ArcPointer`; copies are O(1) ref-count bumps.

ArrayData — generic flat layout
---------------------------------
`ArrayData` is a plain @fieldwise_init struct (same 7 fields as the old
AnyArray) produced on demand by `as_data()`.  It is used for the C Data
Interface, building nested arrays, and other interop where a flat layout is
required.  It is NOT stored inside AnyArray.
"""

from std.memory import memcpy, ArcPointer
from std.sys import size_of
from std.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from .buffers import Buffer, BufferBuilder
from .bitmap import Bitmap, BitmapBuilder
from .dtypes import *
from .builders import PrimitiveBuilder, StringBuilder
from .scalars import PrimitiveScalar, StringScalar, ListScalar


trait Array(
    ConvertibleToPython,
    Copyable,
    Equatable,
    ImplicitlyDestructible,
    Movable,
    Sized,
    Writable,
):
    """Common interface for all typed Arrow arrays.

    All concrete array types (PrimitiveArray, StringArray, ListArray,
    FixedSizeListArray, StructArray) implement this trait.  AnyArray is
    the type-erased handle that wraps any Array-conforming type.
    """

    def __init__(out self, data: ArrayData) raises:
        ...

    def type(self) -> DataType:
        ...

    def null_count(self) -> Int:
        ...

    def is_valid(self, index: Int) -> Bool:
        ...

    def to_any(deinit self) -> AnyArray:
        ...

    def to_data(self) raises -> ArrayData:
        ...

    def slice(self, offset: Int, length: Int) -> Self:
        ...


struct AnyArray(
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Writable,
):
    """Type-erased, immutable array handle.

    Wraps any `Array`-conforming type on the heap behind an `ArcPointer`.
    Copies are O(1) ref-count bumps + function-pointer copies.
    Runtime dispatch goes through function-pointer trampolines (vtable).

    Use `as_primitive[T]()`, `as_string()`, `as_list()`, etc. to obtain
    typed views.  Use `as_data()` to extract a generic `ArrayData` layout
    for interop (C Data Interface, construction of nested arrays).
    """

    var _data: ArcPointer[NoneType]
    var _virt_length: def(ArcPointer[NoneType]) -> Int
    var _virt_dtype: def(ArcPointer[NoneType]) -> DataType
    var _virt_null_count: def(ArcPointer[NoneType]) -> Int
    var _virt_is_valid: def(ArcPointer[NoneType], Int) -> Bool
    var _virt_to_data: def(ArcPointer[NoneType]) raises -> ArrayData
    var _virt_eq: def(ArcPointer[NoneType], ArcPointer[NoneType]) -> Bool
    var _virt_drop: def(var ArcPointer[NoneType])
    var _virt_slice: def(ArcPointer[NoneType], Int, Int) raises -> ArcPointer[
        NoneType
    ]

    # --- trampolines ---

    @staticmethod
    def _tramp_length[T: Array](ptr: ArcPointer[NoneType]) -> Int:
        return len(rebind[ArcPointer[T]](ptr)[])

    @staticmethod
    def _tramp_dtype[T: Array](ptr: ArcPointer[NoneType]) -> DataType:
        return rebind[ArcPointer[T]](ptr)[].type()

    @staticmethod
    def _tramp_null_count[T: Array](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[T]](ptr)[].null_count()

    @staticmethod
    def _tramp_is_valid[T: Array](ptr: ArcPointer[NoneType], i: Int) -> Bool:
        return rebind[ArcPointer[T]](ptr)[].is_valid(i)

    @staticmethod
    def _tramp_to_data[T: Array](ptr: ArcPointer[NoneType]) raises -> ArrayData:
        return rebind[ArcPointer[T]](ptr)[].to_data()

    @staticmethod
    def _tramp_eq[
        T: Array
    ](ptr: ArcPointer[NoneType], other: ArcPointer[NoneType]) -> Bool:
        return rebind[ArcPointer[T]](ptr)[] == rebind[ArcPointer[T]](other)[]

    @staticmethod
    def _tramp_drop[T: Array](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    @staticmethod
    def _tramp_slice[
        T: Array
    ](ptr: ArcPointer[NoneType], offset: Int, length: Int) raises -> ArcPointer[
        NoneType
    ]:
        # `slice` cannot return `AnyArray` directly because that would make
        # `_virt_slice`'s type `def (...) -> AnyArray`, which creates a
        # recursive struct definition (AnyArray containing a field whose type
        # references AnyArray).  Instead, the trampoline returns a new
        # ArcPointer[NoneType] holding the typed slice.  The caller
        # (AnyArray.slice) then stamps a fresh AnyArray shell — copying the
        # existing vtable (valid because slicing preserves the concrete type)
        # and swapping in the new data pointer.  This heap-allocation may raise
        # on OOM, hence the `raises` on both the trampoline and AnyArray.slice.
        var result = rebind[ArcPointer[T]](ptr)[].slice(offset, length)
        return rebind[ArcPointer[NoneType]](ArcPointer(result^))

    # --- construction ---

    @implicit
    def __init__[T: Array](out self, var array: T):
        var ptr = ArcPointer(array^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_length = Self._tramp_length[T]
        self._virt_dtype = Self._tramp_dtype[T]
        self._virt_null_count = Self._tramp_null_count[T]
        self._virt_is_valid = Self._tramp_is_valid[T]
        self._virt_to_data = Self._tramp_to_data[T]
        self._virt_eq = Self._tramp_eq[T]
        self._virt_drop = Self._tramp_drop[T]
        self._virt_slice = Self._tramp_slice[T]

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._virt_length = copy._virt_length
        self._virt_dtype = copy._virt_dtype
        self._virt_null_count = copy._virt_null_count
        self._virt_is_valid = copy._virt_is_valid
        self._virt_to_data = copy._virt_to_data
        self._virt_eq = copy._virt_eq
        self._virt_drop = copy._virt_drop
        self._virt_slice = copy._virt_slice

    def __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema, CArrowArray

        # Try downcasting from a marrow Python object.
        try:
            comptime for T in primitive_dtypes:
                try:
                    self = AnyArray(
                        py.downcast_value_ptr[PrimitiveArray[T]]()[].copy()
                    )
                    return
                except:
                    pass
            self = AnyArray(py.downcast_value_ptr[StringArray]()[].copy())
            return
        except:
            pass
        try:
            self = AnyArray(py.downcast_value_ptr[ListArray]()[].copy())
            return
        except:
            pass
        try:
            self = AnyArray(
                py.downcast_value_ptr[FixedSizeListArray]()[].copy()
            )
            return
        except:
            pass
        try:
            self = AnyArray(py.downcast_value_ptr[StructArray]()[].copy())
            return
        except:
            pass

        # Fall back to the Arrow C Data Interface for foreign objects.
        var caps: PythonObject
        try:
            caps = py.__arrow_c_array__(Python.none())
        except:
            raise Error(
                "cannot convert Python object of type",
                t" '{py.__class__.__name__}' to AnyArray",
            )
        var c_schema = CArrowSchema.from_pycapsule(caps[0])
        var c_array = CArrowArray.from_pycapsule(caps[1])
        self = c_array^.to_array(c_schema.to_dtype())

    # --- vtable dispatch ---

    def length(self) -> Int:
        return self._virt_length(self._data)

    def dtype(self) -> DataType:
        return self._virt_dtype(self._data)

    def null_count(self) -> Int:
        return self._virt_null_count(self._data)

    def is_valid(self, index: Int) -> Bool:
        return self._virt_is_valid(self._data, index)

    def slice(self, offset: Int, length: Int = -1) raises -> AnyArray:
        """Returns a zero-copy slice starting at offset with the given length.

        Matches PyArrow's Array.slice(offset, length) API.
        """
        var result = AnyArray(copy=self)
        result._data = self._virt_slice(self._data, offset, length)
        return result^

    def to_python_object(var self) raises -> PythonObject:
        """Convert to the corresponding Python typed-array object."""
        var dt = self.dtype()
        comptime for T in primitive_dtypes:
            if dt == T:
                return self.as_primitive[T]().copy().to_python_object()
        if dt.is_string():
            return self.as_string().copy().to_python_object()
        elif dt.is_list():
            return self.as_list().copy().to_python_object()
        elif dt.is_fixed_size_list():
            return self.as_fixed_size_list().copy().to_python_object()
        elif dt.is_struct():
            return self.as_struct().copy().to_python_object()
        raise Error("to_python_object: unsupported dtype")

    # --- typed downcasts ---

    def as_primitive[
        T: DataType
    ](ref self) -> ref[self._data[]] PrimitiveArray[T]:
        return rebind[ArcPointer[PrimitiveArray[T]]](self._data)[]

    def as_bool(ref self) -> ref[self._data[]] BoolArray:
        return self.as_primitive[bool_]()

    def as_int8(ref self) -> ref[self._data[]] PrimitiveArray[int8]:
        return self.as_primitive[int8]()

    def as_int16(ref self) -> ref[self._data[]] PrimitiveArray[int16]:
        return self.as_primitive[int16]()

    def as_int32(ref self) -> ref[self._data[]] PrimitiveArray[int32]:
        return self.as_primitive[int32]()

    def as_int64(ref self) -> ref[self._data[]] PrimitiveArray[int64]:
        return self.as_primitive[int64]()

    def as_uint8(ref self) -> ref[self._data[]] PrimitiveArray[uint8]:
        return self.as_primitive[uint8]()

    def as_uint16(ref self) -> ref[self._data[]] PrimitiveArray[uint16]:
        return self.as_primitive[uint16]()

    def as_uint32(ref self) -> ref[self._data[]] PrimitiveArray[uint32]:
        return self.as_primitive[uint32]()

    def as_uint64(ref self) -> ref[self._data[]] PrimitiveArray[uint64]:
        return self.as_primitive[uint64]()

    def as_float32(ref self) -> ref[self._data[]] PrimitiveArray[float32]:
        return self.as_primitive[float32]()

    def as_float64(ref self) -> ref[self._data[]] PrimitiveArray[float64]:
        return self.as_primitive[float64]()

    def as_string(ref self) -> ref[self._data[]] StringArray:
        return rebind[ArcPointer[StringArray]](self._data)[]

    def as_list(ref self) -> ref[self._data[]] ListArray:
        return rebind[ArcPointer[ListArray]](self._data)[]

    def as_fixed_size_list(ref self) -> ref[self._data[]] FixedSizeListArray:
        return rebind[ArcPointer[FixedSizeListArray]](self._data)[]

    def as_struct(ref self) -> ref[self._data[]] StructArray:
        return rebind[ArcPointer[StructArray]](self._data)[]

    # --- generic layout (interop) ---

    def to_data(self) raises -> ArrayData:
        """Extract a generic ArrayData layout for interop (C Data Interface, etc.).

        Not intended for hot paths — prefer typed downcast methods.
        """
        return self._virt_to_data(self._data)

    @staticmethod
    def from_data(data: ArrayData) raises -> AnyArray:
        """Construct an AnyArray from a generic ArrayData by dispatching on dtype.

        Used by the C Data Interface and other interop paths where a flat
        7-field layout is the natural representation.
        """
        var dt = data.dtype
        comptime for T in primitive_dtypes:
            if dt == T:
                return AnyArray(PrimitiveArray[T](data))
        if dt.is_string() or dt.is_binary():
            return AnyArray(StringArray(data))
        elif dt.is_list():
            return AnyArray(ListArray(data))
        elif dt.is_fixed_size_list():
            return AnyArray(FixedSizeListArray(data))
        elif dt.is_struct():
            return AnyArray(StructArray(data))
        raise Error("from_data: unsupported dtype")

    # --- common operations ---

    def to_any(deinit self) -> AnyArray:
        """Returns this array as AnyArray, transferring ownership."""
        return self^

    def write_to[W: Writer](self, mut writer: W):
        var dt = self.dtype()
        comptime for T in primitive_dtypes:
            if dt == T:
                self.as_primitive[T]().write_to(writer)
                return
        if dt.is_string():
            self.as_string().write_to(writer)
        elif dt.is_list():
            self.as_list().write_to(writer)
        elif dt.is_fixed_size_list():
            self.as_fixed_size_list().write_to(writer)
        elif dt.is_struct():
            self.as_struct().write_to(writer)
        else:
            writer.write("AnyArray(dtype=")
            writer.write(dt)
            writer.write(", length=")
            writer.write(self.length())
            writer.write(")")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def __eq__(self, other: AnyArray) -> Bool:
        return self._virt_eq(self._data, other._data)

    def __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# ArrayData — generic flat layout, produced on demand by as_data()
# ---------------------------------------------------------------------------


@fieldwise_init
struct ArrayData(Copyable, Movable):
    """Generic array layout — the old AnyArray wire format, now a pure DTO.

    Produced by `typed_array.to_data()` or `any_array.to_data()` for use
    in the C Data Interface, construction helpers, and other interop paths.
    Not stored inside AnyArray itself.
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap]
    var buffers: List[Buffer]
    var children: List[ArrayData]


# ---------------------------------------------------------------------------
# PrimitiveArray[T]
# ---------------------------------------------------------------------------


@fieldwise_init
struct PrimitiveArray[T: DataType](
    Array,
    ConvertibleFromPython,
    ConvertibleToPython,
):
    """An immutable Arrow array of fixed-size primitive values (integers, floats, etc.).
    """

    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]

    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap]
    var buffer: Buffer

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    def __init__(out self, data: ArrayData) raises:
        if len(data.buffers) != 1:
            raise Error("PrimitiveArray requires exactly one buffer")
        self = Self(
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            buffer=data.buffers[0],
        )

    def __init__(
        out self, var *values: Self.scalar, __list_literal__: ()
    ) raises:
        """Constructs a primitive array from a list literal [v1, v2, ...].

        Args:
            values: The scalar values to populate the array with.
            __list_literal__: Tells Mojo to use this method for list literal syntax.
        """
        var b = PrimitiveBuilder[Self.T](capacity=len(values))
        for value in values:
            b.unsafe_append(value)
        self = b.finish()

    @always_inline
    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def type(self) -> DataType:
        return Self.T

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array.

        Matches PyArrow's Array.slice(offset, length) API.
        """
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            length=actual_length,
            nulls=self.nulls,
            offset=self.offset + offset,
            bitmap=self.bitmap,
            buffer=self.buffer,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("PrimitiveArray[")
        writer.write(Self.T)
        writer.write("]([")
        for i in range(self.length):
            if i > 0:
                writer.write(", ")
            if i >= 10:
                writer.write("...")
                break
            if self.is_valid(i):
                writer.write(self.unsafe_get(i))
            else:
                writer.write("NULL")
        writer.write("])")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    @always_inline
    def is_valid(self, index: Int) -> Bool:
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(self.offset + index)

    @always_inline
    def unsafe_get(self, index: Int) -> Self.scalar:
        return self.buffer.unsafe_get[Self.T.native](index + self.offset)

    def __getitem__(self, index: Int) raises -> PrimitiveScalar[Self.T]:
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        if not self.is_valid(index):
            return PrimitiveScalar[Self.T].null()
        return PrimitiveScalar[Self.T](self.unsafe_get(index))

    def null_count(self) -> Int:
        return self.nulls

    def true_count(self) raises -> Int:
        """Count True values. Only valid for BoolArray (PrimitiveArray[bool_]).
        """
        comptime assert (
            Self.T == bool_
        ), "true_count is only valid for BoolArray"
        var data_bm = Bitmap(self.buffer, self.offset, self.length)
        if self.nulls > 0:
            return (data_bm & self.bitmap.value()).count_set_bits()
        return data_bm.count_set_bits()

    def false_count(self) raises -> Int:
        """Count False values. Only valid for BoolArray (PrimitiveArray[bool_]).
        """
        comptime assert (
            Self.T == bool_
        ), "false_count is only valid for BoolArray"
        return self.length - self.nulls - self.true_count()

    def to_device(self, ctx: DeviceContext) raises -> PrimitiveArray[Self.T]:
        """Upload array data to the GPU."""
        var bm: Optional[Bitmap] = None
        if self.bitmap:
            bm = Bitmap(
                self.bitmap.value()._buffer.to_device(ctx),
                0,
                self.bitmap.value()._length,
            )
        return PrimitiveArray[Self.T](
            length=self.length,
            nulls=self.nulls,
            offset=0,
            bitmap=bm^,
            buffer=self.buffer.to_device(ctx),
        )

    def to_cpu(self, ctx: DeviceContext) raises -> PrimitiveArray[Self.T]:
        """Download array data from the GPU to owned CPU heap buffers."""
        var bm: Optional[Bitmap] = None
        if self.bitmap:
            bm = Bitmap(
                self.bitmap.value()._buffer.to_cpu(ctx),
                0,
                self.bitmap.value()._length,
            )
        return PrimitiveArray[Self.T](
            length=self.length,
            nulls=self.nulls,
            offset=0,
            bitmap=bm^,
            buffer=self.buffer.to_cpu(ctx),
        )

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def __eq__(self, other: Self) -> Bool:
        """Return True if both arrays have the same length, null pattern, and values.

        Fast path (no nulls, offset=0 on both): full buffer SIMD comparison.
        Slow path (nulls or non-zero offset): element-by-element at valid positions.
        """
        if self.length != other.length:
            return False
        if self.nulls != other.nulls:
            return False
        if self.bitmap.__bool__() != other.bitmap.__bool__():
            return False
        if self.bitmap:
            if self.bitmap.value() != other.bitmap.value():
                return False
        # Fast path: no nulls, no offset — full buffer SIMD comparison.
        if self.nulls == 0 and self.offset == 0 and other.offset == 0:
            return self.buffer == other.buffer
        # Slow path: nulls or offset — compare only valid elements.
        for i in range(self.length):
            if self.is_valid(i):
                if self.unsafe_get(i) != other.unsafe_get(i):
                    return False
        return True

    def to_any(deinit self) -> AnyArray:
        return AnyArray(self^)

    def to_data(self) -> ArrayData:
        """Extract generic array layout for interop."""
        return ArrayData(
            dtype=Self.T,
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[self.buffer],
            children=[],
        )


comptime BoolArray = PrimitiveArray[bool_]
comptime Int8Array = PrimitiveArray[int8]
comptime Int16Array = PrimitiveArray[int16]
comptime Int32Array = PrimitiveArray[int32]
comptime Int64Array = PrimitiveArray[int64]
comptime UInt8Array = PrimitiveArray[uint8]
comptime UInt16Array = PrimitiveArray[uint16]
comptime UInt32Array = PrimitiveArray[uint32]
comptime UInt64Array = PrimitiveArray[uint64]
comptime Float32Array = PrimitiveArray[float32]
comptime Float64Array = PrimitiveArray[float64]


# ---------------------------------------------------------------------------
# StringArray
# ---------------------------------------------------------------------------


@fieldwise_init
struct StringArray(
    Array,
    ConvertibleFromPython,
    ConvertibleToPython,
):
    """An immutable Arrow array of variable-length UTF-8 strings."""

    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap]
    var offsets: Buffer
    var values: Buffer

    def __init__(out self, var *values: String, __list_literal__: ()) raises:
        """Constructs a string array from a list literal ["a", "b", ...].

        Args:
            values: The string values to populate the array with.
            __list_literal__: Tells Mojo to use this method for list literal syntax.
        """
        var b = StringBuilder(capacity=len(values))
        for value in values:
            b.append(value)
        self = b.finish()

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    def __init__(out self, data: ArrayData) raises:
        if len(data.buffers) != 2:
            raise Error("StringArray requires exactly two buffers")
        self = Self(
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            offsets=data.buffers[0],
            values=data.buffers[1],
        )

    def __len__(self) -> Int:
        """Return the number of elements in the array."""
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def null_count(self) -> Int:
        return self.nulls

    def type(self) -> DataType:
        return string

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array.

        Matches PyArrow's Array.slice(offset, length) API.
        """
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            length=actual_length,
            nulls=self.nulls,
            offset=self.offset + offset,
            bitmap=self.bitmap,
            offsets=self.offsets,
            values=self.values,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("StringArray([")
        for i in range(self.length):
            if i > 0:
                writer.write(", ")
            if i >= 10:
                writer.write("...")
                break
            if self.is_valid(i):
                writer.write(self.unsafe_get(UInt(i)))
            else:
                writer.write("NULL")
        writer.write("])")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def is_valid(self, index: Int) -> Bool:
        """Return True if the element at the given index is not null."""
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(self.offset + index)

    def unsafe_get[
        self_origin: Origin[mut=False], //
    ](ref[self_origin] self, index: UInt) -> StringSlice[self_origin]:
        """Return a StringSlice for the element at the given index without bounds checking.
        """
        var offset_idx = Int(index) + self.offset
        var start_offset = self.offsets.unsafe_get[DType.uint32](offset_idx)
        var end_offset = self.offsets.unsafe_get[DType.uint32](offset_idx + 1)
        var length = Int(end_offset) - Int(start_offset)
        var ptr = (
            (self.values.ptr + Int(start_offset))
            .mut_cast[False]()
            .unsafe_origin_cast[self_origin]()
        )
        return StringSlice[self_origin](ptr=ptr, length=length)

    def __getitem__(self, index: Int) raises -> StringScalar:
        """Return a StringScalar for the element at the given index.

        Raises:
            If the index is out of bounds.
        """
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        if not self.is_valid(index):
            return StringScalar.null()
        return StringScalar(String(self.unsafe_get(UInt(index))))

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def __eq__(self, other: Self) -> Bool:
        """Return True if both arrays have the same length, null pattern, and string values.
        """
        if self.length != other.length:
            return False
        if self.nulls != other.nulls:
            return False
        if self.bitmap.__bool__() != other.bitmap.__bool__():
            return False
        if self.bitmap:
            if self.bitmap.value() != other.bitmap.value():
                return False
        for i in range(self.length):
            if self.is_valid(i):
                if self.unsafe_get(UInt(i)) != other.unsafe_get(UInt(i)):
                    return False
        return True

    def to_any(deinit self) -> AnyArray:
        return AnyArray(self^)

    def to_data(self) -> ArrayData:
        """Extract generic array layout for interop."""
        return ArrayData(
            dtype=string,
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[self.offsets, self.values],
            children=[],
        )


# ---------------------------------------------------------------------------
# ListArray
# ---------------------------------------------------------------------------


@fieldwise_init
struct ListArray(
    Array,
    ConvertibleFromPython,
    ConvertibleToPython,
):
    """An immutable Arrow array of variable-length lists (each element is a sub-array).
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap]
    var offsets: Buffer
    var values: AnyArray

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    def __init__(out self, data: ArrayData) raises:
        if len(data.buffers) != 1:
            raise Error("ListArray requires exactly one buffer")
        if len(data.children) != 1:
            raise Error("ListArray requires exactly one child array")
        self = Self(
            dtype=data.dtype,
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            offsets=data.buffers[0],
            values=AnyArray.from_data(data.children[0]),
        )

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def null_count(self) -> Int:
        return self.nulls

    def type(self) -> DataType:
        return self.dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("ListArray([")
        for i in range(self.length):
            if i > 0:
                writer.write(", ")
            if i >= 10:
                writer.write("...")
                break
            if self.is_valid(i):
                try:
                    self.unsafe_get(i).write_to(writer)
                except:
                    writer.write("?")
            else:
                writer.write("NULL")
        writer.write("])")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def is_valid(self, index: Int) -> Bool:
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(self.offset + index)

    def unsafe_get(self, index: Int) raises -> AnyArray:
        """Return a view of the child array for the list at the given index."""
        var start = Int(
            self.offsets.unsafe_get[DType.int32](self.offset + index)
        )
        var end = Int(
            self.offsets.unsafe_get[DType.int32](self.offset + index + 1)
        )
        return self.values.slice(start, end - start)

    def __getitem__(self, index: Int) raises -> ListScalar:
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        return ListScalar(
            value=self.unsafe_get(index), is_valid=self.is_valid(index)
        )

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            dtype=self.dtype,
            length=actual_length,
            nulls=self.nulls,
            offset=self.offset + offset,
            bitmap=self.bitmap,
            offsets=self.offsets,
            values=self.values.copy(),
        )

    def flatten(self) -> AnyArray:
        """Unnest this ListArray, returning the flat child values."""
        return self.values.copy()

    def value_lengths(self) -> PrimitiveArray[int32]:
        """Return an array of list lengths for each element."""
        var buf = BufferBuilder.alloc[DType.int32](self.length)
        for i in range(self.length):
            var start = self.offsets.unsafe_get[DType.int32](self.offset + i)
            var end = self.offsets.unsafe_get[DType.int32](self.offset + i + 1)
            buf.unsafe_set[DType.int32](i, end - start)
        return PrimitiveArray[int32](
            length=self.length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf.finish(),
        )

    def __eq__(self, other: Self) -> Bool:
        """Return True if both arrays have the same dtype, null pattern, and list values.
        """
        if self.dtype != other.dtype:
            return False
        if self.length != other.length:
            return False
        if self.nulls != other.nulls:
            return False
        if self.bitmap.__bool__() != other.bitmap.__bool__():
            return False
        if self.bitmap:
            if self.bitmap.value() != other.bitmap.value():
                return False
        for i in range(self.length):
            if self.is_valid(i):
                try:
                    if self.unsafe_get(i) != other.unsafe_get(i):
                        return False
                except:
                    return False
        return True

    def to_any(deinit self) -> AnyArray:
        return AnyArray(self^)

    def to_data(self) raises -> ArrayData:
        """Extract generic array layout for interop."""
        return ArrayData(
            dtype=self.dtype,
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[self.offsets],
            children=[self.values.to_data()],
        )


# ---------------------------------------------------------------------------
# FixedSizeListArray
# ---------------------------------------------------------------------------


@fieldwise_init
struct FixedSizeListArray(
    Array,
    ConvertibleFromPython,
    ConvertibleToPython,
):
    """An immutable Arrow array of fixed-size lists (each element is a sub-array of the same length).
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap]
    var values: AnyArray

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    def __init__(out self, data: ArrayData) raises:
        if len(data.children) != 1:
            raise Error("FixedSizeListArray requires exactly one child array")
        self = Self(
            dtype=data.dtype,
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            values=AnyArray.from_data(data.children[0]),
        )

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def null_count(self) -> Int:
        return self.nulls

    def type(self) -> DataType:
        return self.dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("FixedSizeListArray([")
        for i in range(self.length):
            if i > 0:
                writer.write(", ")
            if i >= 10:
                writer.write("...")
                break
            if self.is_valid(i):
                try:
                    self.unsafe_get(i).write_to(writer)
                except:
                    writer.write("?")
            else:
                writer.write("NULL")
        writer.write("])")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def is_valid(self, index: Int) -> Bool:
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(self.offset + index)

    def unsafe_get(self, index: Int, out array_data: AnyArray) raises:
        var list_size = self.dtype.size
        var start = (self.offset + index) * list_size
        return self.values.slice(start, list_size)

    def __getitem__(self, index: Int) raises -> AnyArray:
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        return self.unsafe_get(index)

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            dtype=self.dtype,
            length=actual_length,
            nulls=self.nulls,
            offset=self.offset + offset,
            bitmap=self.bitmap,
            values=self.values.copy(),
        )

    def flatten(self) -> AnyArray:
        """Unnest this FixedSizeListArray, returning the flat child values."""
        return self.values.copy()

    def to_device(self, ctx: DeviceContext) raises -> FixedSizeListArray:
        """Upload child values to the GPU."""
        var child_data = self.values.to_data()
        var new_buffers = List[Buffer](capacity=len(child_data.buffers))
        for i in range(len(child_data.buffers)):
            new_buffers.append(child_data.buffers[i].to_device(ctx))
        var child_bm: Optional[Bitmap] = None
        if child_data.bitmap:
            var bv = child_data.bitmap.value()
            child_bm = Bitmap(bv._buffer.to_device(ctx), 0, bv._length)
        var new_child = AnyArray.from_data(
            ArrayData(
                dtype=child_data.dtype,
                length=child_data.length,
                nulls=child_data.nulls,
                offset=child_data.offset,
                bitmap=child_bm^,
                buffers=new_buffers^,
                children=child_data.children.copy(),
            )
        )
        var bm: Optional[Bitmap] = None
        if self.bitmap:
            var bv = self.bitmap.value()
            bm = Bitmap(bv._buffer.to_device(ctx), 0, bv._length)
        return FixedSizeListArray(
            dtype=self.dtype,
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=bm^,
            values=new_child^,
        )

    def __eq__(self, other: Self) -> Bool:
        """Return True if both arrays have the same dtype, null pattern, and element values.
        """
        if self.dtype != other.dtype:
            return False
        if self.length != other.length:
            return False
        if self.nulls != other.nulls:
            return False
        if self.bitmap.__bool__() != other.bitmap.__bool__():
            return False
        if self.bitmap:
            if self.bitmap.value() != other.bitmap.value():
                return False
        for i in range(self.length):
            if self.is_valid(i):
                try:
                    if self.unsafe_get(i) != other.unsafe_get(i):
                        return False
                except:
                    return False
        return True

    def to_any(deinit self) -> AnyArray:
        return AnyArray(self^)

    def to_data(self) raises -> ArrayData:
        """Extract generic array layout for interop."""
        return ArrayData(
            dtype=self.dtype,
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[],
            children=[self.values.to_data()],
        )


# ---------------------------------------------------------------------------
# StructArray
# ---------------------------------------------------------------------------


@fieldwise_init
struct StructArray(
    Array,
    ConvertibleFromPython,
    ConvertibleToPython,
):
    """An immutable Arrow array of structs (each element is a collection of named fields).
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap]
    var children: List[AnyArray]

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    def __init__(out self, data: ArrayData) raises:
        var children = List[AnyArray]()
        for c in data.children:
            children.append(AnyArray.from_data(c))
        self = Self(
            dtype=data.dtype,
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            children=children^,
        )

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def null_count(self) -> Int:
        return self.nulls

    def type(self) -> DataType:
        return self.dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("StructArray({")
        if len(self.children) > 0:
            for i in range(len(self.dtype.fields)):
                if i > 0:
                    writer.write(", ")
                ref field = self.dtype.fields[i]
                writer.write("'")
                writer.write(field.name)
                writer.write("': ")
                self.children[i].write_to(writer)
        writer.write("})")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def is_valid(self, index: Int) -> Bool:
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(self.offset + index)

    def _index_for_field_name(self, name: StringSlice) raises -> Int:
        for idx, ref field in enumerate(self.dtype.fields):
            if field.name == name:
                return idx

        raise Error(t"Field {name} does not exist in this StructArray.")

    def unsafe_get(
        self, name: StringSlice
    ) raises -> ref[self.children[0]] AnyArray:
        """Access the field with the given name in the struct."""
        return self.children[self._index_for_field_name(name)]

    def field(self, index: Int) raises -> AnyArray:
        """Access a child array by field index.

        Matches PyArrow's StructArray.field(index) API.
        """
        if index < 0 or index >= len(self.children):
            raise Error(
                t"field index {index} out of bounds for"
                t" {len(self.children)} fields"
            )
        return self.children[index].copy()

    def field(self, name: StringSlice) raises -> AnyArray:
        """Access a child array by field name.

        Matches PyArrow's StructArray.field(name) API.
        """
        return self.children[self._index_for_field_name(name)].copy()

    def select(self, indices: List[Int]) raises -> Self:
        """Return a new StructArray with only the fields at the given indices.

        O(1) ref-count bumps per selected column — no data copied.
        Matches RecordBatch.select(indices) API.
        """
        var fields = List[Field]()
        var children = List[AnyArray]()
        for idx in indices:
            fields.append(self.dtype.fields[idx].copy())
            children.append(self.children[idx].copy())
        return Self(
            dtype=struct_(fields^),
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap,
            children=children^,
        )

    def flatten(self) -> List[AnyArray]:
        """Return one AnyArray per field.

        Matches PyArrow's StructArray.flatten() API.
        """
        return self.children.copy()

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            dtype=self.dtype,
            length=actual_length,
            nulls=self.nulls,
            offset=self.offset + offset,
            bitmap=self.bitmap,
            children=self.children.copy(),
        )

    def __eq__(self, other: Self) -> Bool:
        """Return True if both arrays have the same dtype, null pattern, and field values.
        """
        if self.dtype != other.dtype:
            return False
        if self.length != other.length:
            return False
        if self.nulls != other.nulls:
            return False
        if self.bitmap.__bool__() != other.bitmap.__bool__():
            return False
        if self.bitmap:
            if self.bitmap.value() != other.bitmap.value():
                return False
        if len(self.children) != len(other.children):
            return False
        for i in range(len(self.children)):
            if self.children[i] != other.children[i]:
                return False
        return True

    def to_any(deinit self) -> AnyArray:
        return AnyArray(self^)

    def to_data(self) raises -> ArrayData:
        """Extract generic array layout for interop."""
        var children = List[ArrayData]()
        for c in self.children:
            children.append(c.to_data())
        return ArrayData(
            dtype=self.dtype,
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[],
            children=children^,
        )


# ---------------------------------------------------------------------------
# ChunkedArray
# ---------------------------------------------------------------------------


struct ChunkedArray(Copyable, Movable, Writable):
    """An array-like composed from a (possibly empty) collection of pyarrow.Arrays.

    [Reference](https://arrow.apache.org/docs/python/generated/pyarrow.ChunkedArray.html#pyarrow-chunkedarray).
    """

    var dtype: DataType
    var length: Int
    var chunks: List[AnyArray]

    def _compute_length(mut self) -> None:
        """Update the length of the array from the length of its chunks."""
        var total_length = 0
        for chunk in self.chunks:
            total_length += chunk.length()
        self.length = total_length

    def __init__(out self, var dtype: DataType, var chunks: List[AnyArray]):
        self.dtype = dtype^
        self.chunks = chunks^
        self.length = 0
        self._compute_length()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("ChunkedArray([")
        for i in range(len(self.chunks)):
            if i > 0:
                writer.write(", ")
            self.chunks[i].write_to(writer)
        writer.write("])")

    def chunk(self, index: Int) -> ref[self.chunks] AnyArray:
        """Returns the chunk at the given index.

        Args:
          index: The desired index.

        Returns:
          A reference to the chunk at the given index.
        """
        return self.chunks[index]

    def combine_chunks(var self) raises -> AnyArray:
        """Combines all chunks into a single array."""
        from .kernels.concat import concat

        if len(self.chunks) == 0:
            return AnyArray.from_data(
                ArrayData(
                    dtype=self.dtype,
                    length=0,
                    nulls=0,
                    offset=0,
                    bitmap=None,
                    buffers=[],
                    children=[],
                )
            )
        return concat(self.chunks)


from .builders import array, nulls, arange
