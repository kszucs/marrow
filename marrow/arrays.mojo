"""Arrow columnar arrays — always immutable.

Every typed array (`PrimitiveArray`, `StringArray`, `ListArray`, `StructArray`)
is immutable.  To *build* an array incrementally, use the corresponding builder
from `marrow.builders` and call `finish()`.

`BoolArray` is an alias for `PrimitiveArray[bool_]`.

Array — the generic container
-----------------------------
`Array` is the low-level, type-erased container used for storage, exchange
(C Data Interface), and visitor dispatch.  It holds immutable bitmaps,
buffers, and child arrays directly (no ArcPointer wrapping — sharing is
handled inside Buffer via its internal ArcPointer[Allocation]).  Typed arrays
convert to/from `Array` via implicit constructors and `as_*()` accessors.
"""

from std.memory import memcpy
from std.sys import size_of
from std.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from .buffers import Buffer, BufferBuilder
from .bitmap import Bitmap, BitmapBuilder
from .dtypes import *
from .builders import PrimitiveBuilder, StringBuilder


@fieldwise_init
struct Array(
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Writable,
):
    """Array is the lower level abstraction directly usable by the library consumer.

    Equivalent with https://github.com/apache/arrow/blob/7184439dea96cd285e6de00e07c5114e4919a465/cpp/src/arrow/array/data.h#L62-L84.

    Array holds immutable bitmap and buffers. Use typed array builders
    (e.g. PrimitiveBuilder[T]) to construct data, then convert to Array
    for storage/exchange.
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var bitmap: Optional[Bitmap]
    var buffers: List[Buffer]
    var children: List[Array]
    var offset: Int

    @implicit
    fn __init__[T: DataType](out self, array: PrimitiveArray[T]):
        self.dtype = T
        self.length = array.length
        self.nulls = array.nulls
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.buffer]
        self.children = []

    @implicit
    fn __init__(out self, array: StringArray):
        self.dtype = string
        self.length = array.length
        self.nulls = array.nulls
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.offsets, array.values]
        self.children = []

    @implicit
    fn __init__(out self, array: ListArray):
        self.dtype = array.dtype
        self.length = array.length
        self.nulls = array.nulls
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.offsets]
        self.children = [array.values.copy()]

    @implicit
    fn __init__(out self, array: FixedSizeListArray):
        self.dtype = array.dtype
        self.length = array.length
        self.nulls = array.nulls
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = []
        self.children = [array.values.copy()]

    @implicit
    fn __init__(out self, array: StructArray):
        self.dtype = array.dtype
        self.length = array.length
        self.nulls = array.nulls
        self.offset = 0
        self.bitmap = array.bitmap
        self.buffers = []
        self.children = array.children.copy()

    fn __init__(out self, *, copy: Self):
        self.dtype = copy.dtype
        self.length = copy.length
        self.nulls = copy.nulls
        self.bitmap = copy.bitmap
        self.buffers = copy.buffers.copy()
        self.children = copy.children.copy()
        self.offset = copy.offset

    fn __init__(out self, *, deinit take: Self):
        self.dtype = take.dtype^
        self.length = take.length
        self.nulls = take.nulls
        self.bitmap = take.bitmap^
        self.buffers = take.buffers^
        self.children = take.children^
        self.offset = take.offset

    fn __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema, CArrowArray

        # Try downcasting from a marrow Python object.
        # downcast_value_ptr safely checks Py_TYPE and raises on mismatch.
        # TODO: would be nice to have a cleaner approach to downcast
        # multiple possible types without so many nested try/catch blocks.
        try:
            comptime for T in primitive_dtypes:
                try:
                    self = Array(py.downcast_value_ptr[PrimitiveArray[T]]()[])
                    return
                except:
                    pass
            self = Array(py.downcast_value_ptr[StringArray]()[])
            return
        except:
            pass
        try:
            self = Array(py.downcast_value_ptr[ListArray]()[])
            return
        except:
            pass
        try:
            self = Array(py.downcast_value_ptr[FixedSizeListArray]()[])
            return
        except:
            pass
        try:
            self = Array(py.downcast_value_ptr[StructArray]()[])
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
                t" '{py.__class__.__name__}' to Array",
            )
        var c_schema = CArrowSchema.from_pycapsule(caps[0])
        var c_array = CArrowArray.from_pycapsule(caps[1])
        self = c_array^.to_array(c_schema.to_dtype())

    fn is_valid(self, index: Int) -> Bool:
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(self.offset + index)

    fn write_to[W: Writer](self, mut writer: W):
        try:
            comptime for T in primitive_dtypes:
                if self.dtype == T:
                    self.as_primitive[T]().write_to(writer)
                    return
            if self.dtype.is_string():
                self.as_string().write_to(writer)
            elif self.dtype.is_list():
                self.as_list().write_to(writer)
            elif self.dtype.is_fixed_size_list():
                self.as_fixed_size_list().write_to(writer)
            elif self.dtype.is_struct():
                self.as_struct().write_to(writer)
            else:
                writer.write("Array(dtype=")
                writer.write(self.dtype)
                writer.write(", length=")
                writer.write(self.length)
                writer.write(")")
        except:
            writer.write("Array(dtype=")
            writer.write(self.dtype)
            writer.write(", length=")
            writer.write(self.length)
            writer.write(")")

    fn write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    fn copy(self) -> Array:
        """Returns an O(1) copy of this array (Arc ref-count bumps only)."""
        return Array(copy=self)

    fn __eq__(self, other: Array) -> Bool:
        """Return True if both arrays have the same dtype, length, and values.

        Dispatches to the typed array's __eq__ for correctness with offsets and nulls.
        """
        if self.dtype != other.dtype:
            return False
        if self.length != other.length:
            return False
        try:
            comptime for T in primitive_dtypes:
                if self.dtype == T:
                    return self.as_primitive[T]() == other.as_primitive[T]()
            if self.dtype.is_string():
                return self.as_string() == other.as_string()
            if self.dtype.is_list():
                return self.as_list() == other.as_list()
            if self.dtype.is_fixed_size_list():
                return self.as_fixed_size_list() == other.as_fixed_size_list()
            if self.dtype.is_struct():
                return self.as_struct() == other.as_struct()
        except:
            return False
        return False

    fn slice(self, offset: Int, length: Int = -1) -> Array:
        """Returns a zero-copy slice starting at offset with the given length.

        Matches PyArrow's Array.slice(offset, length) API.
        """
        var actual_length = length if length >= 0 else self.length - offset
        return Array(
            dtype=self.dtype.copy(),
            length=actual_length,
            nulls=self.nulls,
            bitmap=self.bitmap,
            buffers=self.buffers.copy(),
            children=self.children.copy(),
            offset=self.offset + offset,
        )

    fn to_python_object(var self) raises -> PythonObject:
        comptime for T in primitive_dtypes:
            if self.dtype == T:
                return self.as_primitive[T]().to_python_object()
        if self.dtype.is_string():
            return self.as_string().to_python_object()
        elif self.dtype.is_list():
            return self.as_list().to_python_object()
        elif self.dtype.is_fixed_size_list():
            return self.as_fixed_size_list().to_python_object()
        elif self.dtype.is_struct():
            return self.as_struct().to_python_object()
        else:
            raise Error("unsupported type: ", self.dtype)

    fn as_primitive[T: DataType](self) raises -> PrimitiveArray[T]:
        return PrimitiveArray[T](self)

    fn as_bool(self) raises -> BoolArray:
        return self.as_primitive[bool_]()

    fn as_int8(self) raises -> PrimitiveArray[int8]:
        return PrimitiveArray[int8](self)

    fn as_int16(self) raises -> PrimitiveArray[int16]:
        return PrimitiveArray[int16](self)

    fn as_int32(self) raises -> PrimitiveArray[int32]:
        return PrimitiveArray[int32](self)

    fn as_int64(self) raises -> PrimitiveArray[int64]:
        return PrimitiveArray[int64](self)

    fn as_uint8(self) raises -> PrimitiveArray[uint8]:
        return PrimitiveArray[uint8](self)

    fn as_uint16(self) raises -> PrimitiveArray[uint16]:
        return PrimitiveArray[uint16](self)

    fn as_uint32(self) raises -> PrimitiveArray[uint32]:
        return PrimitiveArray[uint32](self)

    fn as_uint64(self) raises -> PrimitiveArray[uint64]:
        return PrimitiveArray[uint64](self)

    fn as_float32(self) raises -> PrimitiveArray[float32]:
        return PrimitiveArray[float32](self)

    fn as_float64(self) raises -> PrimitiveArray[float64]:
        return PrimitiveArray[float64](self)

    fn as_string(self) raises -> StringArray:
        return StringArray(self)

    fn as_list(self) raises -> ListArray:
        return ListArray(self)

    fn as_fixed_size_list(self) raises -> FixedSizeListArray:
        return FixedSizeListArray(self)

    fn as_struct(self) raises -> StructArray:
        return StructArray(self)


@fieldwise_init
struct PrimitiveArray[T: DataType](
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Sized,
    Writable,
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

    fn __init__(out self, ref data: Array, offset: Int = 0) raises:
        if data.dtype != Self.T:
            # TODO: mojo hangs if we pass data.dtype directly despite that it properly satisfies Writable
            raise Error(
                t"Unexpected dtype '{data.dtype}' instead of '{Self.T}'."
            )
        elif len(data.buffers) != 1:
            raise Error("PrimitiveArray requires exactly one buffer")

        self.length = data.length
        self.nulls = data.nulls
        self.offset = data.offset + offset
        self.bitmap = data.bitmap
        self.buffer = data.buffers[0]

    fn __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    fn __init__(
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
        self = b.finish_typed()

    @always_inline
    fn __len__(self) -> Int:
        return self.length

    fn __str__(self) -> String:
        return String.write(self)

    fn type(self) -> DataType:
        return Self.T

    fn slice(self, offset: Int = 0, length: Int = -1) -> Self:
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

    fn write_to[W: Writer](self, mut writer: W):
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

    fn write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    @always_inline
    fn is_valid(self, index: Int) -> Bool:
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(self.offset + index)

    @always_inline
    fn unsafe_get(self, index: Int) -> Self.scalar:
        return self.buffer.unsafe_get[Self.T.native](index + self.offset)

    fn __getitem__(self, index: Int) raises -> Self.scalar:
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        return self.unsafe_get(index)

    fn null_count(self) -> Int:
        return self.nulls

    fn true_count(self) raises -> Int:
        """Count True values. Only valid for BoolArray (PrimitiveArray[bool_]).
        """
        comptime assert (
            Self.T == bool_
        ), "true_count is only valid for BoolArray"
        var data_bm = Bitmap(self.buffer, self.offset, self.length)
        if self.nulls > 0:
            return (data_bm & self.bitmap.value()).count_set_bits()
        return data_bm.count_set_bits()

    fn false_count(self) raises -> Int:
        """Count False values. Only valid for BoolArray (PrimitiveArray[bool_]).
        """
        comptime assert (
            Self.T == bool_
        ), "false_count is only valid for BoolArray"
        return self.length - self.nulls - self.true_count()

    fn to_device(self, ctx: DeviceContext) raises -> PrimitiveArray[Self.T]:
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

    fn to_cpu(self, ctx: DeviceContext) raises -> PrimitiveArray[Self.T]:
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

    fn to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    fn __eq__(self, other: Self) -> Bool:
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


@fieldwise_init
struct StringArray(
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Sized,
    Writable,
):
    """An immutable Arrow array of variable-length UTF-8 strings."""

    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap]
    var offsets: Buffer
    var values: Buffer

    fn __init__(out self, ref data: Array) raises:
        """Construct a StringArray from a generic Array.

        Raises:
            If the dtype is not string or the buffer count is wrong.
        """
        if data.dtype != string:
            raise Error(t"Unexpected dtype '{data.dtype}' instead of 'string'.")
        elif len(data.buffers) != 2:
            raise Error("StringArray requires exactly two buffers")

        self.length = data.length
        self.nulls = data.nulls
        self.offset = data.offset
        self.bitmap = data.bitmap
        self.offsets = data.buffers[0]
        self.values = data.buffers[1]

    fn __init__(out self, var *values: String, __list_literal__: ()) raises:
        """Constructs a string array from a list literal ["a", "b", ...].

        Args:
            values: The string values to populate the array with.
            __list_literal__: Tells Mojo to use this method for list literal syntax.
        """
        var b = StringBuilder(capacity=len(values))
        for value in values:
            b.append(value)
        self = b.finish_typed()

    fn __len__(self) -> Int:
        """Return the number of elements in the array."""
        return self.length

    fn __str__(self) -> String:
        return String.write(self)

    fn null_count(self) -> Int:
        return self.nulls

    fn type(self) -> DataType:
        return string

    fn slice(self, offset: Int = 0, length: Int = -1) -> Self:
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

    fn write_to[W: Writer](self, mut writer: W):
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

    fn write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    fn is_valid(self, index: Int) -> Bool:
        """Return True if the element at the given index is not null."""
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(self.offset + index)

    fn unsafe_get[
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

    fn __getitem__[
        self_origin: Origin[mut=False], //
    ](ref[self_origin] self, index: Int) raises -> StringSlice[self_origin]:
        """Return a StringSlice for the element at the given index.

        Raises:
            If the index is out of bounds.
        """
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        return self.unsafe_get(UInt(index))

    fn to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    fn __eq__(self, other: Self) -> Bool:
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

    fn __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()


@fieldwise_init
struct ListArray(
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Sized,
    Writable,
):
    """An immutable Arrow array of variable-length lists (each element is a sub-array).
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap]
    var offsets: Buffer
    var values: Array

    fn __init__(out self, ref data: Array) raises:
        if not data.dtype.is_list():
            raise Error(t"Unexpected dtype '{data.dtype}' instead of 'list'.")
        elif len(data.buffers) != 1:
            raise Error("ListArray requires exactly one buffer")
        elif len(data.children) != 1:
            raise Error("ListArray requires exactly one child array")

        self.dtype = data.dtype
        self.length = data.length
        self.nulls = data.nulls
        self.offset = data.offset
        self.bitmap = data.bitmap
        self.offsets = data.buffers[0]
        self.values = data.children[0].copy()

    fn __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    fn to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    fn __len__(self) -> Int:
        return self.length

    fn __str__(self) -> String:
        return String.write(self)

    fn null_count(self) -> Int:
        return self.nulls

    fn type(self) -> DataType:
        return self.dtype

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("ListArray([")
        for i in range(self.length):
            if i > 0:
                writer.write(", ")
            if i >= 10:
                writer.write("...")
                break
            if self.is_valid(i):
                self.unsafe_get(i).write_to(writer)
            else:
                writer.write("NULL")
        writer.write("])")

    fn write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    fn is_valid(self, index: Int) -> Bool:
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(self.offset + index)

    fn unsafe_get(self, index: Int) -> Array:
        """Return a view of the child array for the list at the given index."""
        var start = Int(
            self.offsets.unsafe_get[DType.int32](self.offset + index)
        )
        var end = Int(
            self.offsets.unsafe_get[DType.int32](self.offset + index + 1)
        )
        var result = Array(copy=self.values)
        result.offset = start
        result.length = end - start
        result.nulls = 0
        return result^

    fn __getitem__(self, index: Int) raises -> Array:
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        return self.unsafe_get(index)

    fn slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            dtype=self.dtype,
            length=actual_length,
            nulls=self.nulls,
            offset=self.offset + offset,
            bitmap=self.bitmap,
            offsets=self.offsets,
            values=Array(copy=self.values),
        )

    fn flatten(self) -> Array:
        """Unnest this ListArray, returning the flat child values."""
        return Array(copy=self.values)

    fn value_lengths(self) -> PrimitiveArray[int32]:
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

    fn __eq__(self, other: Self) -> Bool:
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
                if self.unsafe_get(i) != other.unsafe_get(i):
                    return False
        return True


@fieldwise_init
struct FixedSizeListArray(
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Sized,
    Writable,
):
    """An immutable Arrow array of fixed-size lists (each element is a sub-array of the same length).
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap]
    var values: Array

    fn __init__(out self, ref data: Array) raises:
        if not data.dtype.is_fixed_size_list():
            raise Error(
                t"Unexpected dtype '{data.dtype}' instead of 'fixed_size_list'."
            )
        elif len(data.buffers) != 0:
            raise Error("FixedSizeListArray requires zero buffers")
        elif len(data.children) != 1:
            raise Error("FixedSizeListArray requires exactly one child array")

        self.dtype = data.dtype
        self.length = data.length
        self.nulls = data.nulls
        self.offset = data.offset
        self.bitmap = data.bitmap
        self.values = data.children[0].copy()

    fn __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    fn to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    fn __len__(self) -> Int:
        return self.length

    fn __str__(self) -> String:
        return String.write(self)

    fn null_count(self) -> Int:
        return self.nulls

    fn type(self) -> DataType:
        return self.dtype

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("FixedSizeListArray([")
        for i in range(self.length):
            if i > 0:
                writer.write(", ")
            if i >= 10:
                writer.write("...")
                break
            if self.is_valid(i):
                self.unsafe_get(i).write_to(writer)
            else:
                writer.write("NULL")
        writer.write("])")

    fn write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    fn is_valid(self, index: Int) -> Bool:
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(self.offset + index)

    fn unsafe_get(self, index: Int, out array_data: Array):
        var list_size = self.dtype.size
        var start = (self.offset + index) * list_size
        return Array(
            dtype=self.values.dtype,
            length=list_size,
            # TODO: calculate nullcount
            nulls=0,
            bitmap=self.values.bitmap,  # Optional[Bitmap] — shared ref-counted copy
            buffers=self.values.buffers.copy(),
            children=self.values.children.copy(),
            offset=start,
        )

    fn __getitem__(self, index: Int) raises -> Array:
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        return self.unsafe_get(index)

    fn slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            dtype=self.dtype,
            length=actual_length,
            nulls=self.nulls,
            offset=self.offset + offset,
            bitmap=self.bitmap,
            values=Array(copy=self.values),
        )

    fn flatten(self) -> Array:
        """Unnest this FixedSizeListArray, returning the flat child values."""
        return Array(copy=self.values)

    fn to_device(self, ctx: DeviceContext) raises -> FixedSizeListArray:
        """Upload child values to the GPU."""
        var new_buffers = List[Buffer](capacity=len(self.values.buffers))
        for i in range(len(self.values.buffers)):
            new_buffers.append(self.values.buffers[i].to_device(ctx))
        var child_bm: Optional[Bitmap] = None
        if self.values.bitmap:
            var bv = self.values.bitmap.value()
            child_bm = Bitmap(bv._buffer.to_device(ctx), 0, bv._length)
        var new_child = Array(
            dtype=self.values.dtype,
            length=self.values.length,
            nulls=self.values.nulls,
            bitmap=child_bm^,
            buffers=new_buffers^,
            children=[],
            offset=self.values.offset,
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

    fn __eq__(self, other: Self) -> Bool:
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
                if self.unsafe_get(i) != other.unsafe_get(i):
                    return False
        return True


@fieldwise_init
struct StructArray(
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Sized,
    Writable,
):
    """An immutable Arrow array of structs (each element is a collection of named fields).
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var bitmap: Optional[Bitmap]
    var children: List[Array]

    fn __init__(out self, ref data: Array):
        self.dtype = data.dtype
        self.length = data.length
        self.nulls = data.nulls
        self.bitmap = data.bitmap
        self.children = data.children.copy()

    fn __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    fn to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    fn __len__(self) -> Int:
        return self.length

    fn __str__(self) -> String:
        return String.write(self)

    fn null_count(self) -> Int:
        return self.nulls

    fn type(self) -> DataType:
        return self.dtype

    fn write_to[W: Writer](self, mut writer: W):
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

    fn write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    fn is_valid(self, index: Int) -> Bool:
        if not self.bitmap:
            return True
        return self.bitmap.value().is_valid(index)

    fn _index_for_field_name(self, name: StringSlice) raises -> Int:
        for idx, ref field in enumerate(self.dtype.fields):
            if field.name == name:
                return idx

        raise Error(t"Field {name} does not exist in this StructArray.")

    fn unsafe_get(
        self, name: StringSlice
    ) raises -> ref[self.children[0]] Array:
        """Access the field with the given name in the struct."""
        return self.children[self._index_for_field_name(name)]

    fn field(self, index: Int) raises -> Array:
        """Access a child array by field index.

        Matches PyArrow's StructArray.field(index) API.
        """
        if index < 0 or index >= len(self.children):
            raise Error(
                t"field index {index} out of bounds for"
                t" {len(self.children)} fields"
            )
        return self.children[index].copy()

    fn field(self, name: StringSlice) raises -> Array:
        """Access a child array by field name.

        Matches PyArrow's StructArray.field(name) API.
        """
        return self.children[self._index_for_field_name(name)].copy()

    fn flatten(self) -> List[Array]:
        """Return one Array per field.

        Matches PyArrow's StructArray.flatten() API.
        """
        return self.children.copy()

    fn __eq__(self, other: Self) -> Bool:
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


struct ChunkedArray(Copyable, Movable, Writable):
    """An array-like composed from a (possibly empty) collection of pyarrow.Arrays.

    [Reference](https://arrow.apache.org/docs/python/generated/pyarrow.ChunkedArray.html#pyarrow-chunkedarray).
    """

    var dtype: DataType
    var length: Int
    var chunks: List[Array]

    fn _compute_length(mut self) -> None:
        """Update the length of the array from the length of its chunks."""
        var total_length = 0
        for chunk in self.chunks:
            total_length += chunk.length
        self.length = total_length

    fn __init__(out self, var dtype: DataType, var chunks: List[Array]):
        self.dtype = dtype^
        self.chunks = chunks^
        self.length = 0
        self._compute_length()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("ChunkedArray([")
        for i in range(len(self.chunks)):
            if i > 0:
                writer.write(", ")
            self.chunks[i].write_to(writer)
        writer.write("])")

    fn chunk(self, index: Int) -> ref[self.chunks] Array:
        """Returns the chunk at the given index.

        Args:
          index: The desired index.

        Returns:
          A reference to the chunk at the given index.
        """
        return self.chunks[index]

    fn combine_chunks(var self) raises -> Array:
        """Combines all chunks into a single array."""
        from .kernels.concat import concat

        if len(self.chunks) == 0:
            return Array(
                dtype=self.dtype,
                length=0,
                nulls=0,
                bitmap=None,
                buffers=[],
                children=[],
                offset=0,
            )
        return concat(self.chunks)


from .builders import array, nulls, arange
