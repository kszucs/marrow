"""Arrow columnar arrays — always immutable.

Every typed array (`PrimitiveArray`, `StringArray`, `ListArray`, `StructArray`)
is immutable.  To *build* an array incrementally, use the corresponding builder
from `marrow.builders` and call `finish()`.

`BoolArray` is a dedicated bit-packed boolean array type.

Array — the trait
-----------------
`Array` is the trait that all typed arrays implement.  It provides the common
read-only interface: `type()`, `null_count()`, `is_valid()`, and `to_any()`.

AnyArray — the type-erased handle
----------------------------------
`AnyArray` is the type-erased, immutable handle backed by an inline `Variant`.
Copies are O(1) — all typed arrays hold their data behind ref-counted `Buffer` /
`Bitmap` handles, so copying the variant is just a few ref-count bumps.

Runtime dispatch goes through `_dispatch`, which iterates the variant members at
compile time and selects the active type via `isa[T]()`.  No unsafe `rebind`
casts or function-pointer trampolines are used.

Use `as_primitive[T]()`, `as_bool()`, `as_string()`, `as_list()`, etc.
to obtain typed references (zero-cost borrows).  Use `to_data()` to extract
a generic `ArrayData` layout for interop (C Data Interface, nested arrays).

ArrayData — generic flat layout
---------------------------------
`ArrayData` is a plain @fieldwise_init struct produced on demand by `to_data()`.
It is used for the C Data Interface, building nested arrays, and other interop
paths.  It is NOT stored inside AnyArray.
"""

from std.bit import pop_count
from std.memory import memcpy, ArcPointer, OwnedPointer
from std.sys import size_of
from std.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from std.utils import Variant
from std.builtin.variadics import Variadic
from std.os import abort
from .buffers import Buffer, Bitmap
from .views import BufferView, BitmapView
from .dtypes import *
from .builders import PrimitiveBuilder, StringBuilder
from .scalars import (
    AnyScalar,
    BoolScalar,
    PrimitiveScalar,
    StringScalar,
    ListScalar,
    StructScalar,
    Scalar as ScalarTrait,
)


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

    def type(self) -> AnyDataType:
        ...

    def null_count(self) -> Int:
        ...

    def is_valid(self, index: Int) -> Bool:
        ...

    def to_any(deinit self) -> AnyArray:
        ...

    def to_data(self) raises -> ArrayData:
        ...

    comptime ScalarType: ScalarTrait

    def slice(self, offset: Int, length: Int) -> Self:
        ...

    def __getitem__(self, index: Int) raises -> Self.ScalarType:
        ...


# ---------------------------------------------------------------------------
# ArrayData — generic flat layout, produced on demand by to_data()
# ---------------------------------------------------------------------------


@fieldwise_init
struct ArrayData(Copyable, Movable):
    """Generic array layout — the old AnyArray wire format, now a pure DTO.

    Produced by `typed_array.to_data()` or `any_array.to_data()` for use
    in the C Data Interface, construction helpers, and other interop paths.
    Not stored inside AnyArray itself.
    """

    var dtype: AnyDataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var buffers: List[Buffer[mut=False]]
    var children: List[ArrayData]


# ---------------------------------------------------------------------------
# BoolArray
# ---------------------------------------------------------------------------


@fieldwise_init
struct BoolArray(
    Array,
    ConvertibleFromPython,
    ConvertibleToPython,
):
    """Immutable array of boolean values, packed as bits in a Bitmap buffer.

    Null values are represented by a separate validity bitmap (if any), not
    by a special bit pattern in the data buffer.  This allows for efficient
    boolean operations using bitwise logic, without needing to check for nulls.
    """

    comptime ScalarType = BoolScalar

    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var buffer: Bitmap[mut=False]

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    def __init__(out self, data: ArrayData) raises:
        if len(data.buffers) != 1:
            raise Error("BoolArray requires exactly one buffer")
        self = Self(
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            buffer=Bitmap(data.buffers[0]),
        )

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def type(self) -> AnyDataType:
        return bool_

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
        writer.write("BoolArray([")
        for i in range(self.length):
            if i > 0:
                writer.write(", ")
            if i >= 10:
                writer.write("...")
                break
            if self.is_valid(i):
                writer.write(
                    "True" if self.values().test(self.offset + i) else "False"
                )
            else:
                writer.write("NULL")
        writer.write("])")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def null_count(self) -> Int:
        return self.nulls

    def is_valid(self, index: Int) -> Bool:
        if not self.bitmap:
            return True
        return self.bitmap.value().test(self.offset + index)

    def __getitem__(self, index: Int) -> BoolScalar:
        var valid = self.is_valid(index)
        if not valid:
            return BoolScalar(is_valid=False)
        return BoolScalar(self.values().test(self.offset + index))

    def values(self) -> BitmapView[origin_of(self.buffer)]:
        """Non-owning bit-level view of the values buffer."""
        return self.buffer.view(self.offset, self.length)

    def validity(
        ref self,
    ) -> Optional[BitmapView[origin_of(self.bitmap._value)]]:
        """Validity bitmap view, or None if all values are valid."""
        if not self.bitmap:
            return None
        return self.bitmap.value().view(self.offset, self.length)

    def to_device(self, ctx: DeviceContext) raises -> BoolArray:
        """Upload array data to the GPU."""
        var bm: Optional[Bitmap[]] = None
        if self.bitmap:
            bm = self.bitmap.value().to_device(ctx)
        return BoolArray(
            length=self.length,
            nulls=self.nulls,
            offset=0,
            bitmap=bm^,
            buffer=self.buffer.to_device(ctx),
        )

    def to_cpu(self, ctx: DeviceContext) raises -> BoolArray:
        """Download array data from the GPU to owned CPU heap buffers."""
        var bm: Optional[Bitmap[]] = None
        if self.bitmap:
            bm = self.bitmap.value().to_cpu(ctx)
        return BoolArray(
            length=self.length,
            nulls=self.nulls,
            offset=0,
            bitmap=bm^,
            buffer=self.buffer.to_cpu(ctx),
        )

    def to_any(deinit self) -> AnyArray:
        return AnyArray(self^)

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def to_data(self) raises -> ArrayData:
        return ArrayData(
            dtype=bool_,
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[self.buffer._buffer],
            children=[],
        )

    def __eq__(self, other: Self) -> Bool:
        """Return True if both arrays have the same length, null pattern, and values.
        """
        if self.length != other.length or self.nulls != other.nulls:
            return False
        for i in range(self.length):
            var lv = self.is_valid(i)
            var rv = other.is_valid(i)
            if lv != rv:
                return False
            if lv and self[i] != other[i]:
                return False
        return True

    def __ne__(self, other: Self) -> Bool:
        return not Self.__eq__(self, other)


# ---------------------------------------------------------------------------
# PrimitiveArray[T]
# ---------------------------------------------------------------------------


# TODO: add conditional conformance where: T.is_primitive()
@fieldwise_init
struct PrimitiveArray[T: PrimitiveType](
    Array,
    ConvertibleFromPython,
    ConvertibleToPython,
):
    """An immutable Arrow array of fixed-size primitive values (integers, floats, etc.).
    """

    comptime ScalarType = PrimitiveScalar[Self.T]

    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]

    # TODO: make these protected to discourage direct access
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var buffer: Buffer[mut=False]

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

    def type(self) -> AnyDataType:
        return Self.T()

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
        writer.write(self.type())
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
        return self.bitmap.value().test(self.offset + index)

    @always_inline
    def unsafe_get(self, index: Int) -> Self.scalar:
        return self.buffer.unsafe_get[Self.T.native](index + self.offset)

    # --- View accessors ---

    def values(
        self,
    ) -> BufferView[Self.T.native, origin_of(self.buffer)]:
        """Non-owning typed view of this array's data values (offset baked in).

        For bool arrays, returns a BitmapView instead — use
        ``values()`` in that case.
        """
        comptime assert (
            Self.T.native != DType.bool
        ), "use values() for bool arrays"
        return self.buffer.view[Self.T.native](self.offset, self.length)

    def validity(
        ref self,
    ) -> Optional[BitmapView[origin_of(self.bitmap._value)]]:
        """Validity bitmap view, or None if all values are valid."""
        if not self.bitmap:
            return None
        return self.bitmap.value().view(self.offset, self.length)

    def __getitem__(self, index: Int) raises -> PrimitiveScalar[Self.T]:
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        if not self.is_valid(index):
            return PrimitiveScalar[Self.T].null()
        return PrimitiveScalar[Self.T](self.unsafe_get(index))

    def null_count(self) -> Int:
        return self.nulls

    def to_device(self, ctx: DeviceContext) raises -> PrimitiveArray[Self.T]:
        """Upload array data to the GPU."""
        var bm: Optional[Bitmap[]] = None
        if self.bitmap:
            bm = self.bitmap.value().to_device(ctx)
        return PrimitiveArray[Self.T](
            length=self.length,
            nulls=self.nulls,
            offset=0,
            bitmap=bm^,
            buffer=self.buffer.to_device(ctx),
        )

    def to_cpu(self, ctx: DeviceContext) raises -> PrimitiveArray[Self.T]:
        """Download array data from the GPU to owned CPU heap buffers."""
        var bm: Optional[Bitmap[]] = None
        if self.bitmap:
            bm = self.bitmap.value().to_cpu(ctx)
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
            if not (self.bitmap.value() == other.bitmap.value()):
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
            dtype=self.type(),
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[self.buffer],
            children=[],
        )


# BoolArray is a distinct struct (not comptime PrimitiveArray[BoolType])
comptime Int8Array = PrimitiveArray[Int8Type]
comptime Int16Array = PrimitiveArray[Int16Type]
comptime Int32Array = PrimitiveArray[Int32Type]
comptime Int64Array = PrimitiveArray[Int64Type]
comptime UInt8Array = PrimitiveArray[UInt8Type]
comptime UInt16Array = PrimitiveArray[UInt16Type]
comptime UInt32Array = PrimitiveArray[UInt32Type]
comptime UInt64Array = PrimitiveArray[UInt64Type]
comptime Float16Array = PrimitiveArray[Float16Type]
comptime Float32Array = PrimitiveArray[Float32Type]
comptime Float64Array = PrimitiveArray[Float64Type]


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

    comptime ScalarType = StringScalar

    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var offsets: Buffer[mut=False]
    var values: Buffer[mut=False]

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

    def type(self) -> AnyDataType:
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
        return self.bitmap.value().test(self.offset + index)

    def unsafe_get(
        ref self, index: UInt
    ) -> StringSlice[origin_of(self.values)]:
        """Return a StringSlice for the element at the given index without bounds checking.
        """
        var offset_idx = Int(index) + self.offset
        var start_offset = self.offsets.unsafe_get[DType.uint32](offset_idx)
        var end_offset = self.offsets.unsafe_get[DType.uint32](offset_idx + 1)
        var length = end_offset - start_offset
        return self.values.slice(
            Int(start_offset), Int(length)
        ).to_string_slice()

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
            if not (self.bitmap.value() == other.bitmap.value()):
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


struct ListArray(
    Array,
    ConvertibleFromPython,
    ConvertibleToPython,
):
    """An immutable Arrow array of variable-length lists (each element is a sub-array).
    """

    comptime ScalarType = ListScalar

    var dtype: AnyDataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var offsets: Buffer[mut=False]
    var child: OwnedPointer[AnyArray]

    def __init__(
        out self,
        *,
        dtype: AnyDataType,
        length: Int,
        nulls: Int,
        offset: Int,
        bitmap: Optional[Bitmap[mut=False]],
        offsets: Buffer[mut=False],
        var values: AnyArray,
    ):
        self.dtype = dtype.copy()
        self.length = length
        self.nulls = nulls
        self.offset = offset
        self.bitmap = bitmap
        self.offsets = offsets
        self.child = OwnedPointer(values^)

    def __init__(out self, *, copy: Self):
        self.dtype = copy.dtype.copy()
        self.length = copy.length
        self.nulls = copy.nulls
        self.offset = copy.offset
        self.bitmap = copy.bitmap
        self.offsets = copy.offsets
        self.child = OwnedPointer(copy.child[].copy())

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    def __init__(out self, data: ArrayData) raises:
        if len(data.buffers) != 1:
            raise Error("ListArray requires exactly one buffer")
        if len(data.children) != 1:
            raise Error("ListArray requires exactly one child array")
        self = Self(
            dtype=data.dtype.copy(),
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            offsets=data.buffers[0],
            values=AnyArray.from_data(data.children[0]),
        )

    def values(ref self) -> ref[self.child[]] AnyArray:
        """The child array containing the list elements."""
        return self.child[]

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def null_count(self) -> Int:
        return self.nulls

    def type(self) -> AnyDataType:
        return self.dtype.copy()

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
        return self.bitmap.value().test(self.offset + index)

    def unsafe_get(self, index: Int) raises -> AnyArray:
        """Return a view of the child array for the list at the given index."""
        var start = Int(
            self.offsets.unsafe_get[DType.int32](self.offset + index)
        )
        var end = Int(
            self.offsets.unsafe_get[DType.int32](self.offset + index + 1)
        )
        return self.values().slice(start, end - start)

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
            dtype=self.dtype.copy(),
            length=actual_length,
            nulls=self.nulls,
            offset=self.offset + offset,
            bitmap=self.bitmap,
            offsets=self.offsets,
            values=self.child[].copy(),
        )

    def flatten(self) -> AnyArray:
        """Unnest this ListArray, returning the flat child values."""
        return self.child[].copy()

    def value_lengths(self) -> PrimitiveArray[Int32Type]:
        """Return an array of list lengths for each element."""
        var buf = Buffer.alloc_zeroed[DType.int32](self.length)
        for i in range(self.length):
            var start = self.offsets.unsafe_get[DType.int32](self.offset + i)
            var end = self.offsets.unsafe_get[DType.int32](self.offset + i + 1)
            buf.unsafe_set[DType.int32](i, end - start)
        return PrimitiveArray[Int32Type](
            length=self.length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf^.to_immutable(),
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
            if not (self.bitmap.value() == other.bitmap.value()):
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
            dtype=self.dtype.copy(),
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[self.offsets],
            children=[self.values().to_data()],
        )


# ---------------------------------------------------------------------------
# FixedSizeListArray
# ---------------------------------------------------------------------------


struct FixedSizeListArray(
    Array,
    ConvertibleFromPython,
    ConvertibleToPython,
):
    """An immutable Arrow array of fixed-size lists (each element is a sub-array of the same length).
    """

    comptime ScalarType = ListScalar

    var dtype: AnyDataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var child: OwnedPointer[AnyArray]

    def __init__(
        out self,
        *,
        dtype: AnyDataType,
        length: Int,
        nulls: Int,
        offset: Int,
        bitmap: Optional[Bitmap[mut=False]],
        var values: AnyArray,
    ):
        self.dtype = dtype.copy()
        self.length = length
        self.nulls = nulls
        self.offset = offset
        self.bitmap = bitmap
        self.child = OwnedPointer(values^)

    def __init__(out self, *, copy: Self):
        self.dtype = copy.dtype.copy()
        self.length = copy.length
        self.nulls = copy.nulls
        self.offset = copy.offset
        self.bitmap = copy.bitmap
        self.child = OwnedPointer(copy.child[].copy())

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    def __init__(out self, data: ArrayData) raises:
        if len(data.children) != 1:
            raise Error("FixedSizeListArray requires exactly one child array")
        self = Self(
            dtype=data.dtype.copy(),
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            values=AnyArray.from_data(data.children[0]),
        )

    def values(ref self) -> ref[self.child[]] AnyArray:
        """The child array containing all list elements (length * list_size elements).
        """
        return self.child[]

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def null_count(self) -> Int:
        return self.nulls

    def type(self) -> AnyDataType:
        return self.dtype.copy()

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
        return self.bitmap.value().test(self.offset + index)

    def unsafe_get(self, index: Int, out array_data: AnyArray) raises:
        var list_size = self.dtype.as_fixed_size_list_type().size
        var start = (self.offset + index) * list_size
        return self.values().slice(start, list_size)

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
            dtype=self.dtype.copy(),
            length=actual_length,
            nulls=self.nulls,
            offset=self.offset + offset,
            bitmap=self.bitmap,
            values=self.child[].copy(),
        )

    def flatten(self) -> AnyArray:
        """Unnest this FixedSizeListArray, returning the flat child values."""
        return self.child[].copy()

    def to_device(self, ctx: DeviceContext) raises -> FixedSizeListArray:
        """Upload child values to the GPU."""
        var child_data = self.values().to_data()
        var new_buffers = List[Buffer[]](capacity=len(child_data.buffers))
        for i in range(len(child_data.buffers)):
            new_buffers.append(child_data.buffers[i].to_device(ctx))
        var child_bm: Optional[Bitmap[]] = None
        if child_data.bitmap:
            child_bm = child_data.bitmap.value().to_device(ctx)
        var new_child = AnyArray.from_data(
            ArrayData(
                dtype=child_data.dtype.copy(),
                length=child_data.length,
                nulls=child_data.nulls,
                offset=child_data.offset,
                bitmap=child_bm^,
                buffers=new_buffers^,
                children=child_data.children.copy(),
            )
        )
        var bm: Optional[Bitmap[]] = None
        if self.bitmap:
            bm = self.bitmap.value().to_device(ctx)
        return FixedSizeListArray(
            dtype=self.dtype.copy(),
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
            if not (self.bitmap.value() == other.bitmap.value()):
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
            dtype=self.dtype.copy(),
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[],
            children=[self.values().to_data()],
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

    comptime ScalarType = StructScalar

    var dtype: AnyDataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var children: List[AnyArray]

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    def __init__(out self, data: ArrayData) raises:
        var children = List[AnyArray]()
        for c in data.children:
            children.append(AnyArray.from_data(c))
        self = Self(
            dtype=data.dtype.copy(),
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

    def type(self) -> AnyDataType:
        return self.dtype.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("StructArray({")
        if len(self.children) > 0:
            var st = self.dtype.as_struct_type()
            for i in range(len(st.fields)):
                if i > 0:
                    writer.write(", ")
                ref field = st.fields[i]
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
        return self.bitmap.value().test(self.offset + index)

    def _index_for_field_name(self, name: StringSlice) raises -> Int:
        var fields = self.dtype.as_struct_type().fields.copy()
        for idx, ref field in enumerate(fields):
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

    def __getitem__(self, index: Int) raises -> StructScalar:
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        if not self.is_valid(index):
            return StructScalar.null(self.dtype.copy())
        var fields = List[AnyScalar]()
        for ref child in self.children:
            fields.append(child[index])
        return StructScalar(
            dtype=self.dtype.copy(), value=fields^, is_valid=True
        )

    def select(self, indices: List[Int]) raises -> Self:
        """Return a new StructArray with only the fields at the given indices.

        O(1) ref-count bumps per selected column — no data copied.
        Matches RecordBatch.select(indices) API.
        """
        var fields = List[Field]()
        var children = List[AnyArray]()
        var st = self.dtype.as_struct_type()
        for idx in indices:
            fields.append(st.fields[idx].copy())
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
            dtype=self.dtype.copy(),
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
            if not (self.bitmap.value() == other.bitmap.value()):
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
            dtype=self.dtype.copy(),
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

    var dtype: AnyDataType
    var length: Int
    var chunks: List[AnyArray]

    def _compute_length(mut self) -> None:
        """Update the length of the array from the length of its chunks."""
        var total_length = 0
        for chunk in self.chunks:
            total_length += chunk.length()
        self.length = total_length

    def __init__(out self, dtype: AnyDataType, var chunks: List[AnyArray]):
        self.dtype = dtype.copy()
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
                    dtype=self.dtype.copy(),
                    length=0,
                    nulls=0,
                    offset=0,
                    bitmap=None,
                    buffers=[],
                    children=[],
                )
            )
        return concat(self.chunks)


# ---------------------------------------------------------------------------
# AnyArray — Variant-based type-erased array handle
# ---------------------------------------------------------------------------


struct AnyArray(
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Sized,
    Writable,
):
    """Type-erased, immutable array handle backed by an inline Variant.

    Wraps any `Array`-conforming type.  Copies are O(1) — typed arrays
    hold their data behind ref-counted `Buffer` / `Bitmap` handles, so
    copying the variant bumps a few ref-counts and copies some small ints.

    Runtime dispatch goes through `_dispatch`, which iterates the variant
    members at compile time and selects the active type via `isa[T]()`.
    No unsafe `rebind` casts or function-pointer trampolines are used.

    Use `as_primitive[T]()`, `as_bool()`, `as_string()`, `as_list()`, etc.
    to obtain typed references (zero-cost borrows from the variant storage).
    Use `to_data()` to extract a generic `ArrayData` for interop.
    """

    comptime VariantType = Variant[
        BoolArray,
        Int8Array,
        Int16Array,
        Int32Array,
        Int64Array,
        UInt8Array,
        UInt16Array,
        UInt32Array,
        UInt64Array,
        Float16Array,
        Float32Array,
        Float64Array,
        StringArray,
        ListArray,
        FixedSizeListArray,
        StructArray,
    ]

    var _v: Self.VariantType

    # --- construction ---

    @implicit
    def __init__[T: Array](out self, var array: T):
        self._v = Self.VariantType(array^)

    def __init__(out self, *, copy: Self):
        self._v = Self.VariantType(copy=copy._v)

    def __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema, CArrowArray

        # Fast path: marrow arrays are now exposed as a single AnyArray Python type.
        try:
            self = py.downcast_value_ptr[AnyArray]()[].copy()
            return
        except:
            pass
        # Fall back to the Arrow C Data Interface for foreign objects.
        try:
            var caps = py.__arrow_c_array__(Python.none())
            var c_schema = CArrowSchema.from_pycapsule(caps[0])
            var c_array = CArrowArray.from_pycapsule(caps[1])
            self = c_array^.to_array(c_schema.to_dtype())
        except:
            raise Error(
                "cannot convert Python object of type",
                t" '{py.__class__.__name__}' to AnyArray",
            )

    # --- dispatch-based methods ---

    def length(self) -> Int:
        @parameter
        def f[T: Array](a: T) -> Int:
            return len(a)

        return variant_dispatch[Array, func=f](self._v)

    def dtype(self) -> AnyDataType:
        @parameter
        def f[T: Array](a: T) -> AnyDataType:
            return a.type()

        return variant_dispatch[Array, func=f](self._v)

    def null_count(self) -> Int:
        @parameter
        def f[T: Array](a: T) -> Int:
            return a.null_count()

        return variant_dispatch[Array, func=f](self._v)

    def is_valid(self, index: Int) -> Bool:
        @parameter
        def f[T: Array](a: T) -> Bool:
            return a.is_valid(index)

        return variant_dispatch[Array, func=f](self._v)

    def slice(self, offset: Int, length: Int = -1) raises -> AnyArray:
        """Returns a zero-copy slice starting at offset with the given length.

        Matches PyArrow's Array.slice(offset, length) API.
        """

        @parameter
        def f[T: Array](a: T) -> AnyArray:
            var actual_length = length if length >= 0 else len(a) - offset
            return a.slice(offset, actual_length)

        return variant_dispatch[Array, func=f](self._v)

    def to_data(self) raises -> ArrayData:
        """Extract a generic ArrayData layout for interop (C Data Interface, etc.).

        Not intended for hot paths — prefer typed downcast methods.
        """

        @parameter
        def f[T: Array](a: T) raises -> ArrayData:
            return a.to_data()

        return variant_dispatch_raises[Array, func=f](self._v)

    def to_any(deinit self) -> AnyArray:
        """Returns this array as AnyArray, transferring ownership."""
        return self^

    def write_to[W: Writer](self, mut writer: W):
        @parameter
        def f[T: Array](a: T):
            a.write_to(writer)

        variant_dispatch[Array, func=f](self._v)

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def __eq__(self, other: AnyArray) -> Bool:
        return self._v == other._v

    def to_python_object(var self) raises -> PythonObject:
        """Convert to a Python Array object (type-erased)."""
        return PythonObject(alloc=self^)

    def __len__(self) -> Int:
        return self.length()

    def __getitem__(self, index: Int) raises -> AnyScalar:
        """Return the element at index as a type-erased AnyScalar."""
        if index < 0 or index >= self.length():
            raise Error(
                t"index {index} out of bounds for length {self.length()}"
            )

        @parameter
        def f[T: Array](a: T) raises -> AnyScalar:
            return a[index].to_any()

        return variant_dispatch_raises[Array, func=f](self._v)

    # --- typed downcasts (zero-cost reference borrows) ---

    def as_primitive[
        T: PrimitiveType
    ](ref self) -> ref[self._v] PrimitiveArray[T]:
        return self._v[PrimitiveArray[T]]

    def as_bool(ref self) -> ref[self._v] BoolArray:
        return self._v[BoolArray]

    def as_int8(ref self) -> ref[self._v] Int8Array:
        return self._v[Int8Array]

    def as_int16(ref self) -> ref[self._v] Int16Array:
        return self._v[Int16Array]

    def as_int32(ref self) -> ref[self._v] Int32Array:
        return self._v[Int32Array]

    def as_int64(ref self) -> ref[self._v] Int64Array:
        return self._v[Int64Array]

    def as_uint8(ref self) -> ref[self._v] UInt8Array:
        return self._v[UInt8Array]

    def as_uint16(ref self) -> ref[self._v] UInt16Array:
        return self._v[UInt16Array]

    def as_uint32(ref self) -> ref[self._v] UInt32Array:
        return self._v[UInt32Array]

    def as_uint64(ref self) -> ref[self._v] UInt64Array:
        return self._v[UInt64Array]

    def as_float16(ref self) -> ref[self._v] Float16Array:
        return self._v[Float16Array]

    def as_float32(ref self) -> ref[self._v] Float32Array:
        return self._v[Float32Array]

    def as_float64(ref self) -> ref[self._v] Float64Array:
        return self._v[Float64Array]

    def as_string(ref self) -> ref[self._v] StringArray:
        return self._v[StringArray]

    def as_list(ref self) -> ref[self._v] ListArray:
        return self._v[ListArray]

    def as_fixed_size_list(ref self) -> ref[self._v] FixedSizeListArray:
        return self._v[FixedSizeListArray]

    def as_struct(ref self) -> ref[self._v] StructArray:
        return self._v[StructArray]

    # --- factory from generic layout ---

    @staticmethod
    def from_data(data: ArrayData) raises -> AnyArray:
        """Construct an AnyArray from a generic ArrayData by dispatching on dtype.

        Used by the C Data Interface and other interop paths where a flat
        7-field layout is the natural representation.
        """
        var dt = data.dtype.copy()
        if dt == bool_:
            return AnyArray(BoolArray(data))
        elif dt == int8:
            return AnyArray(Int8Array(data))
        elif dt == int16:
            return AnyArray(Int16Array(data))
        elif dt == int32:
            return AnyArray(Int32Array(data))
        elif dt == int64:
            return AnyArray(Int64Array(data))
        elif dt == uint8:
            return AnyArray(UInt8Array(data))
        elif dt == uint16:
            return AnyArray(UInt16Array(data))
        elif dt == uint32:
            return AnyArray(UInt32Array(data))
        elif dt == uint64:
            return AnyArray(UInt64Array(data))
        elif dt == float16:
            return AnyArray(Float16Array(data))
        elif dt == float32:
            return AnyArray(Float32Array(data))
        elif dt == float64:
            return AnyArray(Float64Array(data))
        if dt.is_string() or dt.is_binary():
            return AnyArray(StringArray(data))
        elif dt.is_list():
            return AnyArray(ListArray(data))
        elif dt.is_fixed_size_list():
            return AnyArray(FixedSizeListArray(data))
        elif dt.is_struct():
            return AnyArray(StructArray(data))
        raise Error("from_data: unsupported dtype")
