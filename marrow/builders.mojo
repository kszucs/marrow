"""Array builders for constructing Arrow arrays incrementally.

`Builder` is the trait that all typed builders implement.  `AnyBuilder` is
the type-erased container that dispatches to the concrete builder at runtime
via function-pointer trampolines.

Typed builders (`PrimitiveBuilder[T]`, `StringBuilder`, `ListBuilder`,
`FixedSizeListBuilder`, `StructBuilder`) each own their data directly and
conform to the `Builder` trait.

`AnyBuilder` wraps any `Builder`-conforming type on the heap behind an
`ArcPointer`, so copies are O(1) ref-count bumps.  Composite builders
(`ListBuilder`, `StructBuilder`) hold `AnyBuilder` children for nesting.

Example
-------
    var b = PrimitiveBuilder[int64](capacity=1024)
    b.append(42)
    b.append_null()
    var arr = b.finish_typed()  # PrimitiveArray[int64]

    # Typed builders implicitly convert to AnyBuilder
    var child = PrimitiveBuilder[float32](capacity=64)
    var list_b = ListBuilder(child^, capacity=10)
"""

from std.memory import memcpy, ArcPointer
from std.sys import size_of
from .buffers import Buffer, BufferBuilder
from .bitmap import Bitmap, BitmapBuilder
from .dtypes import *
from .arrays import (
    Array,
    BoolArray,
    PrimitiveArray,
    StringArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
)


# ---------------------------------------------------------------------------
# Builder trait — the interface every typed builder must implement
# ---------------------------------------------------------------------------


trait Builder(ImplicitlyDestructible, Movable):
    fn length(self) -> Int:
        ...

    fn null_count(self) -> Int:
        ...

    fn dtype(self) -> DataType:
        ...

    fn reserve(mut self, additional: Int) raises:
        ...

    fn append_null(mut self) raises:
        ...

    fn finish(mut self) raises -> Array:
        ...

    fn reset(mut self):
        ...


# ---------------------------------------------------------------------------
# AnyBuilder — type-erased builder with dynamic dispatch
# ---------------------------------------------------------------------------


struct AnyBuilder(ImplicitlyCopyable, Movable):
    """Type-erased builder container.

    Wraps any `Builder`-conforming type and dispatches through function
    pointers.  The inner value lives on the heap behind an `ArcPointer`,
    so copies are O(1) ref-count bumps.
    """

    var _data: ArcPointer[NoneType]
    var _virt_length: fn(ArcPointer[NoneType]) -> Int
    var _virt_null_count: fn(ArcPointer[NoneType]) -> Int
    var _virt_dtype: fn(ArcPointer[NoneType]) -> DataType
    var _virt_reserve: fn(ArcPointer[NoneType], Int) raises
    var _virt_append_null: fn(ArcPointer[NoneType]) raises
    var _virt_finish: fn(ArcPointer[NoneType]) raises -> Array
    var _virt_reset: fn(ArcPointer[NoneType])
    var _virt_drop: fn(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    fn _tramp_length[T: Builder](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[T]](ptr)[].length()

    @staticmethod
    fn _tramp_null_count[T: Builder](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[T]](ptr)[].null_count()

    @staticmethod
    fn _tramp_dtype[T: Builder](ptr: ArcPointer[NoneType]) -> DataType:
        return rebind[ArcPointer[T]](ptr)[].dtype()

    @staticmethod
    fn _tramp_reserve[
        T: Builder
    ](ptr: ArcPointer[NoneType], additional: Int) raises:
        rebind[ArcPointer[T]](ptr)[].reserve(additional)

    @staticmethod
    fn _tramp_append_null[T: Builder](ptr: ArcPointer[NoneType]) raises:
        rebind[ArcPointer[T]](ptr)[].append_null()

    @staticmethod
    fn _tramp_finish[
        T: Builder
    ](ptr: ArcPointer[NoneType],) raises -> Array:
        return rebind[ArcPointer[T]](ptr)[].finish()

    @staticmethod
    fn _tramp_reset[T: Builder](ptr: ArcPointer[NoneType]):
        rebind[ArcPointer[T]](ptr)[].reset()

    @staticmethod
    fn _tramp_drop[T: Builder](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- public API ---

    @implicit
    fn __init__[T: Builder](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_length = Self._tramp_length[T]
        self._virt_null_count = Self._tramp_null_count[T]
        self._virt_dtype = Self._tramp_dtype[T]
        self._virt_reserve = Self._tramp_reserve[T]
        self._virt_append_null = Self._tramp_append_null[T]
        self._virt_finish = Self._tramp_finish[T]
        self._virt_reset = Self._tramp_reset[T]
        self._virt_drop = Self._tramp_drop[T]

    fn __copyinit__(out self, read copy: Self):
        self._data = copy._data.copy()
        self._virt_length = copy._virt_length
        self._virt_null_count = copy._virt_null_count
        self._virt_dtype = copy._virt_dtype
        self._virt_reserve = copy._virt_reserve
        self._virt_append_null = copy._virt_append_null
        self._virt_finish = copy._virt_finish
        self._virt_reset = copy._virt_reset
        self._virt_drop = copy._virt_drop

    fn length(self) -> Int:
        return self._virt_length(self._data)

    fn null_count(self) -> Int:
        return self._virt_null_count(self._data)

    fn dtype(self) -> DataType:
        return self._virt_dtype(self._data)

    fn reserve(mut self, additional: Int) raises:
        self._virt_reserve(self._data, additional)

    fn append_null(mut self) raises:
        self._virt_append_null(self._data)

    fn finish(mut self) raises -> Array:
        return self._virt_finish(self._data)

    fn reset(mut self):
        self._virt_reset(self._data)

    fn downcast[T: Builder](self) -> ArcPointer[T]:
        return rebind[ArcPointer[T]](self._data.copy())

    @always_inline
    fn as_primitive[T: DataType](self) -> ArcPointer[PrimitiveBuilder[T]]:
        return self.downcast[PrimitiveBuilder[T]]()

    @always_inline
    fn as_string(self) -> ArcPointer[StringBuilder]:
        return self.downcast[StringBuilder]()

    @always_inline
    fn as_list(self) -> ArcPointer[ListBuilder]:
        return self.downcast[ListBuilder]()

    @always_inline
    fn as_fixed_size_list(self) -> ArcPointer[FixedSizeListBuilder]:
        return self.downcast[FixedSizeListBuilder]()

    @always_inline
    fn as_struct(self) -> ArcPointer[StructBuilder]:
        return self.downcast[StructBuilder]()

    fn child(self, index: Int) -> AnyBuilder:
        """Access child builder by index (for composite types)."""
        var dt = self.dtype()
        if dt.is_list() or dt.is_fixed_size_list():
            return self.as_list()[].values()
        else:
            return self.as_struct()[].child(index)

    fn __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# PrimitiveBuilder
# ---------------------------------------------------------------------------


struct PrimitiveBuilder[T: DataType](Builder, Sized):
    """Builder for fixed-size primitive arrays (integers, floats)."""

    comptime scalar = Scalar[Self.T.native]

    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: BitmapBuilder
    var _buffer: BufferBuilder

    fn __init__(out self, capacity: Int = 0):
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = BitmapBuilder.alloc(capacity)
        self._buffer = BufferBuilder.alloc[Self.T.native](capacity)

    fn __len__(self) -> Int:
        return self._length

    fn length(self) -> Int:
        return self._length

    fn null_count(self) -> Int:
        return self._null_count

    fn dtype(self) -> DataType:
        return Self.T

    fn append(mut self, value: Self.scalar) raises:
        self.reserve(1)
        self.unsafe_append(value)

    @always_inline
    fn unsafe_append(mut self, value: Self.scalar):
        """Append without bounds checking. Caller must ensure capacity."""
        self._bitmap.set_bit(self._length, True)
        self._buffer.unsafe_set[Self.T.native](self._length, value)
        self._length += 1

    fn append(mut self, value: Bool) raises:
        comptime assert (
            Self.T == bool_
        ), "append(Bool) only supported for PrimitiveBuilder[bool_]"
        self.append(Self.scalar(value))

    @always_inline
    fn append_null(mut self) raises:
        self.reserve(1)
        self.unsafe_append_null()

    @always_inline
    fn unsafe_append_null(mut self):
        """Append null without bounds checking. Caller must ensure capacity."""
        self._bitmap.set_bit(self._length, False)
        self._null_count += 1
        self._length += 1

    fn extend(mut self, values: List[Self.scalar]) raises:
        self.reserve(len(values))
        for value in values:
            self.unsafe_append(value)

    fn extend(mut self, values: List[Self.scalar], valid: List[Bool]) raises:
        for i in range(len(values)):
            if valid[i]:
                self.append(values[i])
            else:
                self.append_null()

    fn reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._buffer.resize[Self.T.native](new_cap)
            self._capacity = new_cap

    fn finish_typed(mut self) raises -> PrimitiveArray[Self.T]:
        # trim over-allocated buffer capacity down to actual length
        self._buffer.resize[Self.T.native](self._length)
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap] = None
        if null_count != 0:
            bm = self._bitmap.finish(self._length)
        # freeze the value buffer into an immutable Buffer
        var values = self._buffer.finish()
        # construct the immutable result array
        var result = PrimitiveArray[Self.T](
            length=self._length,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            buffer=values^,
        )
        # reset builder state for potential reuse
        self.reset()
        return result^

    fn finish(mut self) raises -> Array:
        return self.finish_typed()

    fn reset(mut self):
        self._length = 0
        self._capacity = 0
        self._null_count = 0


# ---------------------------------------------------------------------------
# StringBuilder
# ---------------------------------------------------------------------------


struct StringBuilder(Builder, Sized):
    """Builder for variable-length UTF-8 string arrays.

    _offsets — uint32 offsets
    _values  — utf-8 byte data (grown on demand)
    """

    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: BitmapBuilder
    var _offsets: BufferBuilder
    var _values: BufferBuilder

    fn __init__(out self, capacity: Int = 0, bytes_capacity: Int = 0):
        var offsets = BufferBuilder.alloc[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = BitmapBuilder.alloc(capacity)
        self._offsets = offsets^
        self._values = BufferBuilder.alloc[DType.uint8](bytes_capacity)

    fn __len__(self) -> Int:
        return self._length

    fn length(self) -> Int:
        return self._length

    fn null_count(self) -> Int:
        return self._null_count

    fn dtype(self) -> DataType:
        return string

    fn append(mut self, value: String) raises:
        self.append(StringSlice(value))

    fn append[origin: Origin](mut self, s: StringSlice[origin]) raises:
        self.reserve(1)
        self.reserve_bytes(len(s))
        self.unsafe_append(s)

    fn append_null(mut self) raises:
        self.reserve(1)
        var index = self._length
        var last_offset = self._offsets.ptr.bitcast[UInt32]()[index]
        self._bitmap.set_bit(index, False)
        self._null_count += 1
        self._length += 1
        self._offsets.unsafe_set[DType.uint32](index + 1, last_offset)

    fn extend(mut self, values: List[String], valid: List[Bool]) raises:
        for i in range(len(values)):
            if valid[i]:
                self.append(values[i])
            else:
                self.append_null()

    fn reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._offsets.resize[DType.uint32](new_cap + 1)
            self._capacity = new_cap

    fn reserve_bytes(mut self, additional: Int) raises:
        """Pre-allocate space in the byte data buffer."""
        var needed = self._values.size + additional
        self._values.resize[DType.uint8](needed)

    @always_inline
    fn unsafe_append[origin: Origin](mut self, s: StringSlice[origin]):
        """Append string bytes without capacity checks. Caller must ensure capacity.
        """
        var length = len(s)
        var index = self._length
        var last_offset = self._offsets.ptr.bitcast[UInt32]()[index]
        var next_offset = last_offset + UInt32(length)
        self._bitmap.set_bit(index, True)
        self._offsets.unsafe_set[DType.uint32](index + 1, next_offset)
        memcpy(
            dest=self._values.ptr + Int(last_offset),
            src=s.unsafe_ptr(),
            count=length,
        )
        self._length += 1

    @always_inline
    fn unsafe_append_null(mut self):
        """Append null without capacity checks. Caller must ensure capacity."""
        var index = self._length
        var last_offset = self._offsets.ptr.bitcast[UInt32]()[index]
        self._bitmap.set_bit(index, False)
        self._null_count += 1
        self._offsets.unsafe_set[DType.uint32](index + 1, last_offset)
        self._length += 1

    fn finish_typed(mut self) raises -> StringArray:
        # trim offsets buffer to length+1 entries (one past the last string)
        self._offsets.resize[DType.uint32](self._length + 1)
        # trim byte data buffer to the actual number of bytes written
        var used = Int(self._offsets.ptr.bitcast[UInt32]()[self._length])
        self._values.resize[DType.uint8](used)
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap] = None
        if null_count != 0:
            bm = self._bitmap.finish(self._length)
        # freeze offsets and byte data buffers into immutable Buffers
        var offsets = self._offsets.finish()
        var values = self._values.finish()
        # construct the immutable result array
        var result = StringArray(
            length=self._length,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            offsets=offsets^,
            values=values^,
        )
        # reset builder state for potential reuse
        self.reset()
        return result^

    fn finish(mut self) raises -> Array:
        return self.finish_typed()

    fn reset(mut self):
        self._length = 0
        self._capacity = 0
        self._null_count = 0


# ---------------------------------------------------------------------------
# ListBuilder
# ---------------------------------------------------------------------------


struct ListBuilder(Builder, Sized):
    """Builder for variable-length list arrays.

    _offsets — uint32 offsets
    _child   — child element builder (AnyBuilder)
    """

    var _dtype: DataType
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: BitmapBuilder
    var _offsets: BufferBuilder
    var _child: AnyBuilder

    fn __init__(out self, var child: AnyBuilder, capacity: Int = 0):
        var offsets = BufferBuilder.alloc[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        var child_dtype = child.dtype().copy()
        self._dtype = list_(child_dtype^)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = BitmapBuilder.alloc(capacity)
        self._offsets = offsets^
        self._child = child^

    fn __len__(self) -> Int:
        return self._length

    fn length(self) -> Int:
        return self._length

    fn null_count(self) -> Int:
        return self._null_count

    fn dtype(self) -> DataType:
        return self._dtype

    fn values(self) -> AnyBuilder:
        return self._child

    fn append_null(mut self) raises:
        self.reserve(1)
        self.unsafe_append_null()

    fn append_valid(mut self) raises:
        self.reserve(1)
        self.unsafe_append_valid()

    @always_inline
    fn unsafe_append_null(mut self):
        """Append null without capacity check. Caller must ensure capacity."""
        self._bitmap.set_bit(self._length, False)
        self._null_count += 1
        var child_length = self._child.length()
        self._offsets.unsafe_set[DType.uint32](
            self._length + 1, UInt32(child_length)
        )
        self._length += 1

    @always_inline
    fn unsafe_append_valid(mut self):
        """Append valid without capacity check. Caller must ensure capacity."""
        self._bitmap.set_bit(self._length, True)
        var child_length = self._child.length()
        self._offsets.unsafe_set[DType.uint32](
            self._length + 1, UInt32(child_length)
        )
        self._length += 1

    fn reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._offsets.resize[DType.uint32](new_cap + 1)
            self._capacity = new_cap

    fn finish_typed(mut self) raises -> ListArray:
        # trim offsets buffer to length+1 entries (one past the last list)
        self._offsets.resize[DType.uint32](self._length + 1)
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap] = None
        if null_count != 0:
            bm = self._bitmap.finish(self._length)
        # freeze offsets buffer and recursively finish the child builder
        var offsets = self._offsets.finish()
        var values = self._child.finish()
        # construct the immutable result array
        var result = ListArray(
            dtype=self._dtype,
            length=self._length,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            offsets=offsets^,
            values=values^,
        )
        # reset builder state for potential reuse
        self.reset()
        return result^

    fn finish(mut self) raises -> Array:
        return self.finish_typed()

    fn reset(mut self):
        self._length = 0
        self._capacity = 0
        self._null_count = 0


# ---------------------------------------------------------------------------
# FixedSizeListBuilder
# ---------------------------------------------------------------------------


struct FixedSizeListBuilder(Builder, Sized):
    """Builder for fixed-size list arrays.

    _child — child element builder (AnyBuilder)
    """

    var _dtype: DataType
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: BitmapBuilder
    var _child: AnyBuilder

    fn __init__(
        out self, var child: AnyBuilder, list_size: Int, capacity: Int = 0
    ):
        var child_dtype = child.dtype().copy()
        self._dtype = fixed_size_list_(child_dtype^, list_size)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = BitmapBuilder.alloc(capacity)
        self._child = child^

    fn __len__(self) -> Int:
        return self._length

    fn length(self) -> Int:
        return self._length

    fn null_count(self) -> Int:
        return self._null_count

    fn dtype(self) -> DataType:
        return self._dtype

    fn values(self) -> AnyBuilder:
        return self._child

    fn append_null(mut self) raises:
        self.reserve(1)
        self.unsafe_append_null()

    fn append_valid(mut self) raises:
        self.reserve(1)
        self.unsafe_append_valid()

    @always_inline
    fn unsafe_append_null(mut self):
        """Append null without capacity check. Caller must ensure capacity."""
        self._bitmap.set_bit(self._length, False)
        self._null_count += 1
        self._length += 1

    @always_inline
    fn unsafe_append_valid(mut self):
        """Append valid without capacity check. Caller must ensure capacity."""
        self._bitmap.set_bit(self._length, True)
        self._length += 1

    fn reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._capacity = new_cap

    fn finish_typed(mut self) raises -> FixedSizeListArray:
        # no offset buffer to trim — child length is implicit (length * list_size)
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap] = None
        if null_count != 0:
            bm = self._bitmap.finish(self._length)
        # recursively finish the child builder
        var values = self._child.finish()
        # construct the immutable result array
        var result = FixedSizeListArray(
            dtype=self._dtype,
            length=self._length,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            values=values^,
        )
        # reset builder state for potential reuse
        self.reset()
        return result^

    fn finish(mut self) raises -> Array:
        return self.finish_typed()

    fn reset(mut self):
        self._length = 0
        self._capacity = 0
        self._null_count = 0


# ---------------------------------------------------------------------------
# StructBuilder
# ---------------------------------------------------------------------------


struct StructBuilder(Builder, Sized):
    """Builder for struct arrays.

    _children[i] — field builder for field i (AnyBuilder)
    """

    var _dtype: DataType
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: BitmapBuilder
    var _children: List[AnyBuilder]

    fn __init__(
        out self,
        var fields: List[Field],
        var field_builders: List[AnyBuilder],
        capacity: Int = 0,
    ):
        self._dtype = struct_(fields)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = BitmapBuilder.alloc(capacity)
        self._children = field_builders^

    fn __len__(self) -> Int:
        return self._length

    fn length(self) -> Int:
        return self._length

    fn null_count(self) -> Int:
        return self._null_count

    fn dtype(self) -> DataType:
        return self._dtype

    fn child(self, index: Int) -> AnyBuilder:
        return self._children[index]

    fn append_null(mut self) raises:
        if self._length >= self._capacity:
            var new_cap = max(self._capacity * 2, self._length + 1)
            self._bitmap.resize(new_cap)
            self._capacity = new_cap
        self.unsafe_append_null()

    fn append_valid(mut self) raises:
        if self._length >= self._capacity:
            var new_cap = max(self._capacity * 2, self._length + 1)
            self._bitmap.resize(new_cap)
            self._capacity = new_cap
        self.unsafe_append_valid()

    @always_inline
    fn unsafe_append_null(mut self):
        """Append null without capacity check. Caller must ensure capacity."""
        self._bitmap.set_bit(self._length, False)
        self._null_count += 1
        self._length += 1

    @always_inline
    fn unsafe_append_valid(mut self):
        """Append valid without capacity check. Caller must ensure capacity."""
        self._bitmap.set_bit(self._length, True)
        self._length += 1

    fn reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._capacity = new_cap
        for ref child in self._children:
            child.reserve(additional)

    fn finish_typed(mut self) raises -> StructArray:
        # no data buffers to trim — struct layout is encoded in child arrays
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap] = None
        if null_count != 0:
            bm = self._bitmap.finish(self._length)
        # recursively finish each field builder into a frozen child array
        var frozen_children = List[Array]()
        for ref child in self._children:
            frozen_children.append(child.finish())
        # construct the immutable result array
        var result = Array(
            dtype=self._dtype.copy(),
            length=self._length,
            nulls=null_count,
            bitmap=bm^,
            buffers=List[Buffer](),
            children=frozen_children^,
            offset=0,
        )
        # reset builder state for potential reuse
        self.reset()
        return StructArray(data=result^)

    fn finish(mut self) raises -> Array:
        return self.finish_typed()

    fn reset(mut self):
        self._length = 0
        self._capacity = 0
        self._null_count = 0


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

comptime BoolBuilder = PrimitiveBuilder[bool_]
comptime Int8Builder = PrimitiveBuilder[int8]
comptime Int16Builder = PrimitiveBuilder[int16]
comptime Int32Builder = PrimitiveBuilder[int32]
comptime Int64Builder = PrimitiveBuilder[int64]
comptime UInt8Builder = PrimitiveBuilder[uint8]
comptime UInt16Builder = PrimitiveBuilder[uint16]
comptime UInt32Builder = PrimitiveBuilder[uint32]
comptime UInt64Builder = PrimitiveBuilder[uint64]
comptime Float32Builder = PrimitiveBuilder[float32]
comptime Float64Builder = PrimitiveBuilder[float64]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


fn make_builder(dtype: DataType, capacity: Int = 0) raises -> AnyBuilder:
    """Create the right builder tree for any dtype."""
    comptime for T in primitive_dtypes:
        if dtype == T:
            return PrimitiveBuilder[T](capacity)
    if dtype.is_string():
        return StringBuilder(capacity)
    elif dtype.is_list():
        var child = make_builder(dtype.fields[0].dtype)
        return ListBuilder(child^, capacity)
    elif dtype.is_fixed_size_list():
        var child = make_builder(dtype.fields[0].dtype)
        return FixedSizeListBuilder(child^, dtype.size, capacity)
    elif dtype.is_struct():
        var children = List[AnyBuilder]()
        for i in range(len(dtype.fields)):
            children.append(make_builder(dtype.fields[i].dtype))
        return StructBuilder(dtype.fields.copy(), children^, capacity)
    else:
        raise Error("unsupported type: {}".format(dtype))


fn array[T: DataType]() raises -> PrimitiveArray[T]:
    """Create an empty primitive array."""
    var b = PrimitiveBuilder[T](0)
    return b.finish_typed()


fn array[T: DataType](values: List[Optional[Int]]) raises -> PrimitiveArray[T]:
    """Create a primitive array from optional ints (`None` → null)."""
    var b = PrimitiveBuilder[T](len(values))
    for value in values:
        if value:
            b.append(Scalar[T.native](value.value()))
        else:
            b.append_null()
    return b.finish_typed()


fn array(values: List[Optional[Bool]]) raises -> BoolArray:
    """Create a boolean array from optional bools (`None` → null)."""
    var b = BoolBuilder(len(values))
    for value in values:
        if value:
            b.append(Scalar[bool_.native](value.value()))
        else:
            b.append_null()
    return b.finish_typed()


fn nulls[T: DataType](size: Int) raises -> PrimitiveArray[T]:
    """Create a primitive array of `size` null values."""
    var b = PrimitiveBuilder[T](capacity=size)
    b._length = size
    b._null_count = size
    return b.finish_typed()


fn arange[T: DataType](start: Int, end: Int) raises -> PrimitiveArray[T]:
    """Create a numeric array with values [start, end)."""
    comptime assert T.is_numeric(), "arange() only supports numeric DataTypes"
    var b = PrimitiveBuilder[T](end - start)
    for i in range(start, end):
        b.append(Scalar[T.native](i))
    return b.finish_typed()
