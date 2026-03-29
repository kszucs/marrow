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
    var arr = b.finish()  # PrimitiveArray[int64]

    # Typed builders implicitly convert to AnyBuilder
    var child = PrimitiveBuilder[float32](capacity=64)
    var list_b = ListBuilder(child^, capacity=10)
"""

from std.memory import memcpy, ArcPointer
from std.sys import size_of
from .buffers import Buffer, Bitmap
from .views import BitmapView
from .dtypes import *
from .arrays import (
    Array,
    AnyArray,
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
    comptime ArrayType: Array

    def length(self) -> Int:
        ...

    def null_count(self) -> Int:
        ...

    def dtype(self) -> DataType:
        ...

    def reserve(mut self, additional: Int) raises:
        ...

    def append_null(mut self) raises:
        ...

    def extend(mut self, arr: AnyArray) raises:
        ...

    def finish(
        mut self, *, shrink_to_fit: Bool = True
    ) raises -> Self.ArrayType:
        ...

    def reset(mut self):
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
    var _virt_length: def(ArcPointer[NoneType]) -> Int
    var _virt_null_count: def(ArcPointer[NoneType]) -> Int
    var _virt_dtype: def(ArcPointer[NoneType]) -> DataType
    var _virt_reserve: def(ArcPointer[NoneType], Int) raises
    var _virt_append_null: def(ArcPointer[NoneType]) raises
    var _virt_extend: def(ArcPointer[NoneType], AnyArray) raises
    var _virt_finish: def(ArcPointer[NoneType]) raises -> AnyArray
    var _virt_reset: def(ArcPointer[NoneType])
    var _virt_drop: def(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    def _tramp_length[T: Builder](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[T]](ptr)[].length()

    @staticmethod
    def _tramp_null_count[T: Builder](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[T]](ptr)[].null_count()

    @staticmethod
    def _tramp_dtype[T: Builder](ptr: ArcPointer[NoneType]) -> DataType:
        return rebind[ArcPointer[T]](ptr)[].dtype()

    @staticmethod
    def _tramp_reserve[
        T: Builder
    ](ptr: ArcPointer[NoneType], additional: Int) raises:
        rebind[ArcPointer[T]](ptr)[].reserve(additional)

    @staticmethod
    def _tramp_append_null[T: Builder](ptr: ArcPointer[NoneType]) raises:
        rebind[ArcPointer[T]](ptr)[].append_null()

    @staticmethod
    def _tramp_extend[
        T: Builder
    ](ptr: ArcPointer[NoneType], arr: AnyArray) raises:
        rebind[ArcPointer[T]](ptr)[].extend(arr)

    @staticmethod
    def _tramp_finish[T: Builder](ptr: ArcPointer[NoneType]) raises -> AnyArray:
        return rebind[ArcPointer[T]](ptr)[].finish().to_any()

    @staticmethod
    def _tramp_reset[T: Builder](ptr: ArcPointer[NoneType]):
        rebind[ArcPointer[T]](ptr)[].reset()

    @staticmethod
    def _tramp_drop[T: Builder](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- public API ---

    @implicit
    def __init__[T: Builder](out self, var value: T):
        self = Self(ArcPointer(value^))

    @implicit
    def __init__[T: Builder](out self, ptr: ArcPointer[T]):
        self._data = rebind[ArcPointer[NoneType]](ptr.copy())
        self._virt_length = Self._tramp_length[T]
        self._virt_null_count = Self._tramp_null_count[T]
        self._virt_dtype = Self._tramp_dtype[T]
        self._virt_reserve = Self._tramp_reserve[T]
        self._virt_append_null = Self._tramp_append_null[T]
        self._virt_extend = Self._tramp_extend[T]
        self._virt_finish = Self._tramp_finish[T]
        self._virt_reset = Self._tramp_reset[T]
        self._virt_drop = Self._tramp_drop[T]

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._virt_length = copy._virt_length
        self._virt_null_count = copy._virt_null_count
        self._virt_dtype = copy._virt_dtype
        self._virt_reserve = copy._virt_reserve
        self._virt_append_null = copy._virt_append_null
        self._virt_extend = copy._virt_extend
        self._virt_finish = copy._virt_finish
        self._virt_reset = copy._virt_reset
        self._virt_drop = copy._virt_drop

    def length(self) -> Int:
        return self._virt_length(self._data)

    def null_count(self) -> Int:
        return self._virt_null_count(self._data)

    def dtype(self) -> DataType:
        return self._virt_dtype(self._data)

    def reserve(mut self, additional: Int) raises:
        self._virt_reserve(self._data, additional)

    def append_null(mut self) raises:
        self._virt_append_null(self._data)

    def extend(mut self, arr: AnyArray) raises:
        self._virt_extend(self._data, arr)

    def finish(mut self) raises -> AnyArray:
        return self._virt_finish(self._data)

    def reset(mut self):
        self._virt_reset(self._data)

    @always_inline
    def as_primitive[
        T: DataType
    ](ref self) -> ref[self._data[]] PrimitiveBuilder[T]:
        return rebind[ArcPointer[PrimitiveBuilder[T]]](self._data)[]

    @always_inline
    def as_string(ref self) -> ref[self._data[]] StringBuilder:
        return rebind[ArcPointer[StringBuilder]](self._data)[]

    @always_inline
    def as_list(ref self) -> ref[self._data[]] ListBuilder:
        return rebind[ArcPointer[ListBuilder]](self._data)[]

    @always_inline
    def as_fixed_size_list(ref self) -> ref[self._data[]] FixedSizeListBuilder:
        return rebind[ArcPointer[FixedSizeListBuilder]](self._data)[]

    @always_inline
    def as_struct(ref self) -> ref[self._data[]] StructBuilder:
        return rebind[ArcPointer[StructBuilder]](self._data)[]

    @always_inline
    def downcast[T: Builder](self) -> ArcPointer[T]:
        return rebind[ArcPointer[T]](self._data.copy())

    def child(self, index: Int) -> AnyBuilder:
        """Access child builder by index (for composite types)."""
        var dt = self.dtype()
        if dt.is_list() or dt.is_fixed_size_list():
            return self.as_list().values()
        else:
            return self.as_struct().field_builder(index)

    def __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# PrimitiveBuilder
# ---------------------------------------------------------------------------


struct PrimitiveBuilder[T: DataType](Builder, Sized):
    """Builder for fixed-size primitive arrays (integers, floats)."""

    comptime ArrayType = PrimitiveArray[Self.T]

    comptime ScalarType = Scalar[Self.T.native]

    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _buffer: Buffer[mut=True]

    def __init__(out self, capacity: Int = 0, *, zeroed: Bool = True):
        """Create a builder with the given initial capacity.

        Args:
            capacity: Initial element capacity.
            zeroed: If True (default), zero-fill the data buffer. Pass
                False when every element will be written via
                ``unsafe_append`` — avoids wasted memset.
        """
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        # TODO: always allocate uninit since finish() will trim the
        # buffer to the actual length
        if zeroed:
            self._buffer = Buffer.alloc_zeroed[Self.T.native](capacity)
        else:
            self._buffer = Buffer.alloc_uninit[Self.T.native](capacity)

    def __len__(self) -> Int:
        return self._length

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DataType:
        return Self.T

    def append(mut self, value: Self.ScalarType) raises:
        self.reserve(1)
        self.unsafe_append(value)

    @always_inline
    def unsafe_append(mut self, value: Self.ScalarType):
        """Append without bounds checking. Caller must ensure capacity."""
        self._bitmap.set(self._length)
        self._buffer.unsafe_set[Self.T.native](self._length, value)
        self._length += 1

    def append(mut self, value: Bool) raises:
        self.append(Self.ScalarType(value))

    @always_inline
    def append_null(mut self) raises:
        self.reserve(1)
        self.unsafe_append_null()

    @always_inline
    def unsafe_append_null(mut self):
        """Append null without bounds checking. Caller must ensure capacity."""
        self._bitmap.clear(self._length)
        self._null_count += 1
        self._length += 1

    def extend(mut self, values: List[Self.ScalarType]) raises:
        self.reserve(len(values))
        for value in values:
            self.unsafe_append(value)

    def extend(
        mut self, values: List[Self.ScalarType], valid: List[Bool]
    ) raises:
        for i in range(len(values)):
            if valid[i]:
                self.append(values[i])
            else:
                self.append_null()

    def extend(mut self, arr: AnyArray) raises:
        self.extend(arr.as_primitive[Self.T]())

    def extend(mut self, arr: PrimitiveArray[Self.T]) raises:
        """Bulk-append all elements from an existing PrimitiveArray."""
        var n = arr.length
        self.reserve(n)
        if arr.nulls == 0:
            self._bitmap.set_range(self._length, n, True)
        else:
            self._null_count += arr.nulls
            if arr.bitmap:
                var bm = arr.bitmap.value()
                self._bitmap.extend(bm.view(arr.offset, n), self._length, n)
            else:
                self._bitmap.set_range(self._length, n, True)

        self._buffer.extend(arr.values(), self._length, n)
        self._length += n

    def reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._buffer.resize[Self.T.native](new_cap)
            self._capacity = new_cap

    def finish(
        mut self, *, shrink_to_fit: Bool = True
    ) raises -> PrimitiveArray[Self.T]:
        """Finish the builder, optionally skipping the shrink-to-fit realloc."""
        if shrink_to_fit:
            self._buffer.resize[Self.T.native](self._length)
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap[]] = None
        if null_count != 0:
            bm = self._bitmap^.to_immutable(length=self._length)
            self._bitmap = Bitmap.alloc_zeroed(0)
        # freeze the value buffer into an immutable Buffer
        var values = self._buffer^.to_immutable()
        self._buffer = Buffer.alloc_zeroed[Self.T.native](0)
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

    def reset(mut self):
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

    comptime ArrayType = StringArray

    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _offsets: Buffer[mut=True]
    var _values: Buffer[mut=True]

    def __init__(out self, capacity: Int = 0, bytes_capacity: Int = 0):
        var offsets = Buffer.alloc_zeroed[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        self._offsets = offsets^
        self._values = Buffer.alloc_zeroed[DType.uint8](bytes_capacity)

    def __len__(self) -> Int:
        return self._length

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DataType:
        return string

    def append(mut self, value: String) raises:
        self.append(StringSlice(value))

    def append[origin: Origin](mut self, s: StringSlice[origin]) raises:
        self.reserve(1)
        self.reserve_bytes(len(s))
        self.unsafe_append(s)

    def append_null(mut self) raises:
        self.reserve(1)
        var index = self._length
        var last_offset = self._offsets.unsafe_get[DType.uint32](index)
        self._bitmap.clear(index)
        self._null_count += 1
        self._length += 1
        self._offsets.unsafe_set[DType.uint32](index + 1, last_offset)

    def extend(mut self, values: List[String], valid: List[Bool]) raises:
        for i in range(len(values)):
            if valid[i]:
                self.append(values[i])
            else:
                self.append_null()

    def extend(mut self, arr: AnyArray) raises:
        self.extend(arr.as_string())

    def extend(mut self, arr: StringArray) raises:
        """Bulk-append all elements from an existing StringArray."""
        var n = arr.length
        var chunk_start = Int(arr.offsets.unsafe_get[DType.uint32](arr.offset))
        var chunk_end = Int(
            arr.offsets.unsafe_get[DType.uint32](arr.offset + n)
        )
        var chunk_bytes = chunk_end - chunk_start
        self.reserve(n)
        self.reserve_bytes(chunk_bytes)
        if arr.nulls == 0:
            self._bitmap.set_range(self._length, n, True)
        else:
            self._null_count += arr.nulls
            if arr.bitmap:
                var bm = arr.bitmap.value()
                self._bitmap.extend(bm.view(arr.offset, n), self._length, n)
            else:
                self._bitmap.set_range(self._length, n, True)
        var cur_bytes = Int(self._offsets.unsafe_get[DType.uint32](self._length))
        for i in range(n):
            var orig = Int(arr.offsets.unsafe_get[DType.uint32](arr.offset + i))
            self._offsets.unsafe_set[DType.uint32](
                self._length + i, UInt32(cur_bytes + orig - chunk_start)
            )
        self._offsets.unsafe_set[DType.uint32](
            self._length + n, UInt32(cur_bytes + chunk_bytes)
        )
        memcpy(
            dest=self._values.view[DType.uint8](cur_bytes).unsafe_ptr(),
            src=arr.values.view[DType.uint8](chunk_start).unsafe_ptr(),
            count=chunk_bytes,
        )
        self._length += n

    def reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._offsets.resize[DType.uint32](new_cap + 1)
            self._capacity = new_cap

    def reserve_bytes(mut self, additional: Int) raises:
        """Pre-allocate space in the byte data buffer."""
        var needed = len(self._values) + additional
        self._values.resize[DType.uint8](needed)

    @always_inline
    def unsafe_append[origin: Origin](mut self, s: StringSlice[origin]):
        """Append string bytes without capacity checks. Caller must ensure capacity.
        """
        var length = len(s)
        var index = self._length
        var last_offset = self._offsets.unsafe_get[DType.uint32](index)
        var next_offset = last_offset + UInt32(length)
        self._bitmap.set(index)
        self._offsets.unsafe_set[DType.uint32](index + 1, next_offset)
        memcpy(
            dest=self._values.view[DType.uint8](Int(last_offset)).unsafe_ptr(),
            src=s.unsafe_ptr(),
            count=length,
        )
        self._length += 1

    @always_inline
    def unsafe_append_null(mut self):
        """Append null without capacity checks. Caller must ensure capacity."""
        var index = self._length
        var last_offset = self._offsets.unsafe_get[DType.uint32](index)
        self._bitmap.clear(index)
        self._null_count += 1
        self._offsets.unsafe_set[DType.uint32](index + 1, last_offset)
        self._length += 1

    def finish(mut self, *, shrink_to_fit: Bool = True) raises -> StringArray:
        if shrink_to_fit:
            self._offsets.resize[DType.uint32](self._length + 1)
            var used = Int(self._offsets.unsafe_get[DType.uint32](self._length))
            self._values.resize[DType.uint8](used)
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap[]] = None
        if null_count != 0:
            bm = self._bitmap^.to_immutable(length=self._length)
            self._bitmap = Bitmap.alloc_zeroed(0)
        # freeze offsets and byte data buffers into immutable Buffers
        var offsets = self._offsets^.to_immutable()
        self._offsets = Buffer.alloc_zeroed[DType.uint32](0)
        var values = self._values^.to_immutable()
        self._values = Buffer.alloc_zeroed(0)
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

    def reset(mut self):
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

    comptime ArrayType = ListArray

    var _dtype: DataType
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _offsets: Buffer[mut=True]
    var _child: AnyBuilder

    def __init__(out self, var child: AnyBuilder, capacity: Int = 0):
        var offsets = Buffer.alloc_zeroed[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        var child_dtype = child.dtype().copy()
        self._dtype = list_(child_dtype^)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        self._offsets = offsets^
        self._child = child^

    def __len__(self) -> Int:
        return self._length

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DataType:
        return self._dtype

    def values(self) -> AnyBuilder:
        return self._child

    def append_null(mut self) raises:
        self.reserve(1)
        self.unsafe_append_null()

    def append_valid(mut self) raises:
        self.reserve(1)
        self.unsafe_append_valid()

    @always_inline
    def unsafe_append_null(mut self):
        """Append null without capacity check. Caller must ensure capacity."""
        self._bitmap.clear(self._length)
        self._null_count += 1
        var child_length = self._child.length()
        self._offsets.unsafe_set[DType.uint32](
            self._length + 1, UInt32(child_length)
        )
        self._length += 1

    @always_inline
    def unsafe_append_valid(mut self):
        """Append valid without capacity check. Caller must ensure capacity."""
        self._bitmap.set(self._length)
        var child_length = self._child.length()
        self._offsets.unsafe_set[DType.uint32](
            self._length + 1, UInt32(child_length)
        )
        self._length += 1

    def extend(mut self, arr: AnyArray) raises:
        self.extend(arr.as_list())

    def extend(mut self, arr: ListArray) raises:
        """Bulk-append all elements from an existing ListArray."""
        var n = arr.length
        self.reserve(n)
        if arr.nulls == 0:
            self._bitmap.set_range(self._length, n, True)
        else:
            self._null_count += arr.nulls
            if arr.bitmap:
                var bm = arr.bitmap.value()
                self._bitmap.extend(bm.view(arr.offset, n), self._length, n)
            else:
                self._bitmap.set_range(self._length, n, True)
        var child_start = Int(arr.offsets.unsafe_get[DType.int32](arr.offset))
        var child_end = Int(arr.offsets.unsafe_get[DType.int32](arr.offset + n))
        var cur_child_len = self._child.length()
        for i in range(n):
            var orig = Int(arr.offsets.unsafe_get[DType.int32](arr.offset + i))
            self._offsets.unsafe_set[DType.uint32](
                self._length + i,
                UInt32(cur_child_len + orig - child_start),
            )
        self._offsets.unsafe_set[DType.uint32](
            self._length + n,
            UInt32(cur_child_len + child_end - child_start),
        )
        var child_slice = arr.values.slice(child_start, child_end - child_start)
        self._child.extend(child_slice)
        self._length += n

    def reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._offsets.resize[DType.uint32](new_cap + 1)
            self._capacity = new_cap

    def finish(mut self, *, shrink_to_fit: Bool = True) raises -> ListArray:
        if shrink_to_fit:
            self._offsets.resize[DType.uint32](self._length + 1)
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap[]] = None
        if null_count != 0:
            bm = self._bitmap^.to_immutable(length=self._length)
            self._bitmap = Bitmap.alloc_zeroed(0)
        # freeze offsets buffer and recursively finish the child builder
        var offsets = self._offsets^.to_immutable()
        self._offsets = Buffer.alloc_zeroed[DType.uint32](0)
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

    def reset(mut self):
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

    comptime ArrayType = FixedSizeListArray

    var _dtype: DataType
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _child: AnyBuilder

    def __init__(
        out self, var child: AnyBuilder, list_size: Int, capacity: Int = 0
    ):
        var child_dtype = child.dtype().copy()
        self._dtype = fixed_size_list_(child_dtype^, list_size)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        self._child = child^

    def __len__(self) -> Int:
        return self._length

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DataType:
        return self._dtype

    def values(self) -> AnyBuilder:
        return self._child

    def append_null(mut self) raises:
        self.reserve(1)
        self.unsafe_append_null()

    def append_valid(mut self) raises:
        self.reserve(1)
        self.unsafe_append_valid()

    @always_inline
    def unsafe_append_null(mut self):
        """Append null without capacity check. Caller must ensure capacity."""
        self._bitmap.clear(self._length)
        self._null_count += 1
        self._length += 1

    @always_inline
    def unsafe_append_valid(mut self):
        """Append valid without capacity check. Caller must ensure capacity."""
        self._bitmap.set(self._length)
        self._length += 1

    def extend(mut self, arr: AnyArray) raises:
        self.extend(arr.as_fixed_size_list())

    def extend(mut self, arr: FixedSizeListArray) raises:
        """Bulk-append all elements from an existing FixedSizeListArray."""
        var n = arr.length
        self.reserve(n)
        if arr.nulls == 0:
            self._bitmap.set_range(self._length, n, True)
        else:
            self._null_count += arr.nulls
            if arr.bitmap:
                var bm = arr.bitmap.value()
                self._bitmap.extend(bm.view(arr.offset, n), self._length, n)
            else:
                self._bitmap.set_range(self._length, n, True)
        var list_size = arr.dtype.size
        var child_slice = arr.values.slice(
            arr.offset * list_size, n * list_size
        )
        self._child.extend(child_slice)
        self._length += n

    def reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._capacity = new_cap

    def finish(
        mut self, *, shrink_to_fit: Bool = True
    ) raises -> FixedSizeListArray:
        # no offset buffer to trim — child length is implicit (length * list_size)
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap[]] = None
        if null_count != 0:
            bm = self._bitmap^.to_immutable(length=self._length)
            self._bitmap = Bitmap.alloc_zeroed(0)
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

    def reset(mut self):
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

    comptime ArrayType = StructArray

    var _dtype: DataType
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _children: List[AnyBuilder]

    def __init__(out self, var fields: List[Field], capacity: Int = 0) raises:
        var children = List[AnyBuilder](capacity=len(fields))
        for i in range(len(fields)):
            children.append(make_builder(fields[i].dtype))
        self._dtype = struct_(fields)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        self._children = children^

    def __len__(self) -> Int:
        return self._length

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DataType:
        return self._dtype

    def field_builder(ref self, index: Int) -> ref[self._children] AnyBuilder:
        return self._children[index]

    def append_null(mut self) raises:
        if self._length >= self._capacity:
            var new_cap = max(self._capacity * 2, self._length + 1)
            self._bitmap.resize(new_cap)
            self._capacity = new_cap
        self.unsafe_append_null()

    def append_valid(mut self) raises:
        if self._length >= self._capacity:
            var new_cap = max(self._capacity * 2, self._length + 1)
            self._bitmap.resize(new_cap)
            self._capacity = new_cap
        self.unsafe_append_valid()

    @always_inline
    def unsafe_append_null(mut self):
        """Append null without capacity check. Caller must ensure capacity."""
        self._bitmap.clear(self._length)
        self._null_count += 1
        self._length += 1

    @always_inline
    def unsafe_append_valid(mut self):
        """Append valid without capacity check. Caller must ensure capacity."""
        self._bitmap.set(self._length)
        self._length += 1

    # TODO
    def extend(mut self, arr: AnyArray) raises:
        self.extend(arr.as_struct())

    def extend(mut self, arr: StructArray) raises:
        """Bulk-append all elements from an existing StructArray."""
        var n = arr.length
        self.reserve(n)
        if arr.nulls == 0:
            self._bitmap.set_range(self._length, n, True)
        else:
            self._null_count += arr.nulls
            if arr.bitmap:
                var bm = arr.bitmap.value()
                self._bitmap.extend(bm, self._length, n)
            else:
                self._bitmap.set_range(self._length, n, True)
        for f in range(len(arr.children)):
            self._children[f].extend(arr.children[f].copy())
        self._length += n

    def reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._capacity = new_cap
        for ref child in self._children:
            child.reserve(additional)

    def finish(mut self, *, shrink_to_fit: Bool = True) raises -> StructArray:
        # no data buffers to trim — struct layout is encoded in child arrays
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap[]] = None
        if null_count != 0:
            bm = self._bitmap^.to_immutable(length=self._length)
            self._bitmap = Bitmap.alloc_zeroed(0)
        # recursively finish each field builder into a frozen child array
        var frozen_children = List[AnyArray](capacity=len(self._children))
        for ref child in self._children:
            frozen_children.append(child.finish())
        # construct the immutable result array
        var result = StructArray(
            dtype=self._dtype.copy(),
            length=self._length,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            children=frozen_children^,
        )
        # reset builder state for potential reuse
        self.reset()
        return result^

    def reset(mut self):
        self._length = 0
        self._capacity = 0
        self._null_count = 0


# ---------------------------------------------------------------------------
# BoolBuilder — bit-packed boolean array builder
# ---------------------------------------------------------------------------


struct BoolBuilder(Builder, Sized):
    """Builder for bit-packed BoolArray values."""

    comptime ArrayType = BoolArray

    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _buffer: Bitmap[mut=True]

    def __init__(out self, capacity: Int = 0):
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        self._buffer = Bitmap.alloc_zeroed(capacity)

    def __len__(self) -> Int:
        return self._length

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DataType:
        return bool_

    def reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._buffer.resize(new_cap)
            self._capacity = new_cap

    def append(mut self, value: Bool) raises:
        self.reserve(1)
        self._bitmap.set(self._length)
        if value:
            self._buffer.set(self._length)
        else:
            self._buffer.clear(self._length)
        self._length += 1

    def append_null(mut self) raises:
        self.reserve(1)
        self._bitmap.clear(self._length)
        self._buffer.clear(self._length)
        self._null_count += 1
        self._length += 1

    def extend(mut self, arr: AnyArray) raises:
        self.extend(arr.as_bool())

    def extend(mut self, b: BoolArray) raises:
        self.reserve(b.length)
        self._buffer.extend(b.values(), self._length, b.length)
        if b.nulls != 0:
            if b.bitmap:
                self._bitmap.extend(b.validity(), self._length, b.length)
            self._null_count += b.nulls
        else:
            self._bitmap.set_range(self._length, b.length, True)
        self._length += b.length

    def finish(mut self, *, shrink_to_fit: Bool = True) raises -> BoolArray:
        var n = self._length
        var null_count = self._null_count
        var bm: Optional[Bitmap[]] = None
        if null_count != 0:
            bm = self._bitmap^.to_immutable(length=n)
            self._bitmap = Bitmap.alloc_zeroed(0)
        var data = self._buffer^.to_immutable(length=n)
        self._buffer = Bitmap.alloc_zeroed(0)
        var result = BoolArray(
            length=n,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            buffer=data^,
        )
        self._length = 0
        self._null_count = 0
        return result^

    def reset(mut self):
        self._length = 0
        self._null_count = 0


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
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


def make_builder(dtype: DataType, capacity: Int = 0) raises -> AnyBuilder:
    """Create the right builder tree for any dtype."""
    if dtype == bool_:
        return BoolBuilder(capacity)
    comptime for T in numeric_dtypes:
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
        return StructBuilder(dtype.fields.copy(), capacity)
    else:
        raise Error("unsupported type: ", dtype)


def array[T: DataType]() raises -> PrimitiveArray[T]:
    """Create an empty primitive array."""
    var b = PrimitiveBuilder[T](0)
    return b.finish()


def array[T: DataType](values: List[Optional[Int]]) raises -> PrimitiveArray[T]:
    """Create a primitive array from optional ints (`None` → null)."""
    # comptime assert T.is_integer(), "array() with int values only supported for integer DataTypes"
    var b = PrimitiveBuilder[T](len(values))
    for value in values:
        if value:
            b.append(Scalar[T.native](value.value()))
        else:
            b.append_null()
    return b.finish()


def array[
    T: DataType
](values: List[Optional[Float64]]) raises -> PrimitiveArray[T]:
    """Create a primitive array from optional ints (`None` → null)."""
    # comptime assert T.is_integer(), "array() with int values only supported for integer DataTypes"
    var b = PrimitiveBuilder[T](len(values))
    for value in values:
        if value:
            b.append(Scalar[T.native](value.value()))
        else:
            b.append_null()
    return b.finish()


def array(values: List[Optional[Bool]]) raises -> BoolArray:
    """Create a boolean array from optional bools (`None` → null)."""
    var b = BoolBuilder(len(values))
    for value in values:
        if value:
            b.append(Bool(value.value()))
        else:
            b.append_null()

    return b.finish()


def array(values: List[String]) raises -> StringArray:
    """Create a string array from a list of strings."""
    var b = StringBuilder(len(values))
    for i in range(len(values)):
        b.append(values[i])
    return b.finish()


def nulls[T: DataType](size: Int) raises -> PrimitiveArray[T]:
    """Create a primitive array of `size` null values."""
    var b = PrimitiveBuilder[T](capacity=size)
    b._length = size
    b._null_count = size
    return b.finish()


def arange[T: DataType](start: Int, end: Int) raises -> PrimitiveArray[T]:
    """Create a numeric array with values [start, end)."""
    comptime assert T.is_numeric(), "arange() only supports numeric DataTypes"
    var b = PrimitiveBuilder[T](end - start)
    for i in range(start, end):
        b.append(Scalar[T.native](i))
    return b.finish()
