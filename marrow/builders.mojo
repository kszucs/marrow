"""Array builders for constructing Arrow arrays incrementally.

`Builder` is the trait that all typed builders implement.  `DynBuilder` is
the type-erased container that dispatches to the concrete builder at runtime
via function-pointer trampolines.

Typed builders (`PrimitiveBuilder[T]`, `BinaryBuilder`, `ListBuilder`,
`FixedSizeListBuilder`, `StructBuilder`) each own their data directly and
conform to the `Builder` trait.

`DynBuilder` wraps any `Builder`-conforming type on the heap behind an
`ArcPointer`, so copies are O(1) ref-count bumps.  Composite builders
(`ListBuilder`, `StructBuilder`) hold `DynBuilder` children for nesting.

Example
-------
    var b = Int64Builder(capacity=1024)
    b.append(42)
    b.append_null()
    var arr = b.finish()  # Int64Array

    # Typed builders implicitly convert to DynBuilder
    var child = Float32Builder(capacity=64)
    var list_b = ListBuilder(child^, capacity=10)
"""

from std.memory import ArcPointer
from std.utils import Variant

from .buffers import Buffer, Bitmap

from .utils import variant_dispatch, variant_dispatch_raises
from .dtypes import (
    DynType,
    BinaryLikeType,
    BinaryType,
    Date32Type,
    Date64Type,
    DayTimeIntervalType,
    Decimal128Type,
    Decimal256Type,
    Decimal32Type,
    Decimal64Type,
    DurationType,
    Field,
    FixedSizeBinaryType,
    Float16Type,
    Float32Type,
    Float64Type,
    Int16Type,
    Int32Type,
    Int64Type,
    Int8Type,
    LargeBinaryType,
    LargeListType,
    LargeStringType,
    ListLikeType,
    ListType,
    MapType,
    MonthDayNanoIntervalType,
    NumericType,
    PrimitiveType,
    StringType,
    Time32Type,
    Time64Type,
    TimestampType,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    UInt8Type,
    YearMonthIntervalType,
    bool_,
    dictionary,
    fixed_size_list_,
    float16,
    float32,
    float64,
    int16,
    int32,
    int64,
    int8,
    large_list_,
    list_,
    null,
    struct_,
    uint16,
    uint32,
    uint64,
    uint8,
)
from .arrays import (
    Array,
    DynArray,
    NullArray,
    BoolArray,
    PrimitiveArray,
    BinaryLikeArray,
    StringArray,
    ListLikeArray,
    MapArray,
    FixedSizeListArray,
    FixedSizeBinaryArray,
    StructArray,
    DictionaryArray,
)


# ---------------------------------------------------------------------------
# Builder trait — the interface every typed builder must implement
# ---------------------------------------------------------------------------


trait Builder(ImplicitlyDeletable, Movable, Sized):
    comptime ArrayType: Array

    def __len__(self) -> Int:
        return self.length()

    def length(self) -> Int:
        ...

    def null_count(self) -> Int:
        ...

    def dtype(self) -> DynType:
        ...

    def reserve(mut self, additional: Int) raises:
        ...

    def append_null(mut self) raises:
        ...

    def extend(mut self, arr: DynArray) raises:
        ...

    def finish(
        mut self, *, shrink_to_fit: Bool = True
    ) raises -> Self.ArrayType:
        ...

    def reset(mut self) raises:
        """Discard accumulated state and start over.

        `raises` for the same reason `Array.slice` does: the erased
        implementation dispatches over a variant and an uncovered member falls
        through. Typed builders stay non-raising, and Mojo accepts a
        non-raising body against a raising requirement, so their call sites are
        unaffected.
        """
        ...


# ---------------------------------------------------------------------------
# DynBuilder — type-erased builder with dynamic dispatch
# ---------------------------------------------------------------------------


struct DynBuilder(Builder, ImplicitlyCopyable, Movable):
    """Type-erased builder container.

    Wraps any `Builder`-conforming type in a Variant on the heap behind an
    `ArcPointer`. Copies are O(1) ref-count bumps (shared-mutation semantics).
    Dispatch goes through `_dispatch` / `_dispatch_mut`, which iterate the
    Variant members at compile time and select the active type via `isa[T]()`.
    No unsafe `rebind` casts or function-pointer trampolines are used.
    """

    comptime VariantType = Variant[
        NullBuilder,
        BoolBuilder,
        Int8Builder,
        Int16Builder,
        Int32Builder,
        Int64Builder,
        UInt8Builder,
        UInt16Builder,
        UInt32Builder,
        UInt64Builder,
        Float16Builder,
        Float32Builder,
        Float64Builder,
        Date32Builder,
        Date64Builder,
        Time32Builder,
        Time64Builder,
        DurationBuilder,
        TimestampBuilder,
        YearMonthIntervalBuilder,
        DayTimeIntervalBuilder,
        MonthDayNanoIntervalBuilder,
        Decimal32Builder,
        Decimal64Builder,
        Decimal128Builder,
        Decimal256Builder,
        BinaryBuilder,
        LargeBinaryBuilder,
        StringBuilder,
        LargeStringBuilder,
        ListBuilder,
        LargeListBuilder,
        MapBuilder,
        FixedSizeListBuilder,
        FixedSizeBinaryBuilder,
        StructBuilder,
        DictionaryBuilder,
    ]

    var _ptr: ArcPointer[Self.VariantType]

    # --- construction ---

    @implicit
    def __init__[T: Builder](out self, var value: T):
        self._ptr = ArcPointer(Self.VariantType(value^))

    def __init__(out self, *, copy: Self):
        self._ptr = copy._ptr.copy()

    # Explicit (empty) destructor so this type is ImplicitlyDeletable despite
    # the `StructBuilder -> List[DynBuilder] -> DynBuilder` reference cycle; the
    # ArcPointer field is still destroyed automatically after the body runs.
    def __del__(deinit self):
        pass

    def __init__(out self, dtype: DynType, capacity: Int = 0) raises:
        if dtype.is_null():
            self = NullBuilder(capacity)
        elif dtype == bool_:
            self = BoolBuilder(capacity)
        elif dtype == int8:
            self = Int8Builder(capacity)
        elif dtype == int16:
            self = Int16Builder(capacity)
        elif dtype == int32:
            self = Int32Builder(capacity)
        elif dtype == int64:
            self = Int64Builder(capacity)
        elif dtype == uint8:
            self = UInt8Builder(capacity)
        elif dtype == uint16:
            self = UInt16Builder(capacity)
        elif dtype == uint32:
            self = UInt32Builder(capacity)
        elif dtype == uint64:
            self = UInt64Builder(capacity)
        elif dtype == float16:
            self = Float16Builder(capacity)
        elif dtype == float32:
            self = Float32Builder(capacity)
        elif dtype == float64:
            self = Float64Builder(capacity)
        elif dtype.is_string():
            self = StringBuilder(capacity)
        elif dtype.is_binary():
            self = BinaryBuilder(capacity)
        elif dtype.is_large_string():
            self = LargeStringBuilder(capacity)
        elif dtype.is_large_binary():
            self = LargeBinaryBuilder(capacity)
        elif dtype.is_list():
            var child = DynBuilder(dtype.as_list().value_type())
            self = ListBuilder(child^, capacity)
        elif dtype.is_large_list():
            var child = DynBuilder(dtype.as_large_list().value_type())
            self = LargeListBuilder(child^, capacity)
        elif dtype.is_fixed_size_list():
            ref fsl = dtype.as_fixed_size_list()
            var child = DynBuilder(fsl.value_type())
            self = FixedSizeListBuilder(child^, fsl.size, capacity)
        elif dtype.is_fixed_size_binary():
            self = FixedSizeBinaryBuilder(
                dtype.as_fixed_size_binary().byte_width, capacity
            )
        elif dtype.is_date32():
            self = Date32Builder(Date32Type(), capacity)
        elif dtype.is_date64():
            self = Date64Builder(Date64Type(), capacity)
        elif dtype.is_time32():
            self = Time32Builder(dtype.as_time32(), capacity)
        elif dtype.is_time64():
            self = Time64Builder(dtype.as_time64(), capacity)
        elif dtype.is_timestamp():
            self = TimestampBuilder(dtype.as_timestamp(), capacity)
        elif dtype.is_duration():
            self = DurationBuilder(dtype.as_duration(), capacity)
        elif dtype.is_year_month_interval():
            self = YearMonthIntervalBuilder(YearMonthIntervalType(), capacity)
        elif dtype.is_day_time_interval():
            self = DayTimeIntervalBuilder(DayTimeIntervalType(), capacity)
        elif dtype.is_month_day_nano_interval():
            self = MonthDayNanoIntervalBuilder(
                MonthDayNanoIntervalType(), capacity
            )
        elif dtype.is_decimal32():
            self = Decimal32Builder(dtype.as_decimal32(), capacity)
        elif dtype.is_decimal64():
            self = Decimal64Builder(dtype.as_decimal64(), capacity)
        elif dtype.is_decimal128():
            self = Decimal128Builder(dtype.as_decimal128(), capacity)
        elif dtype.is_decimal256():
            self = Decimal256Builder(dtype.as_decimal256(), capacity)
        elif dtype.is_map():
            self = MapBuilder(dtype.as_map(), capacity)
        elif dtype.is_struct():
            self = StructBuilder(dtype.as_struct().fields.copy(), capacity)
        elif dtype.is_dictionary():
            ref dt = dtype.as_dictionary()
            var idx_builder = DynBuilder(dt.index_type(), capacity)
            self = DictionaryBuilder(
                idx_builder^, array(dt.value_type()), dt.ordered
            )
        else:
            raise Error("unsupported type: ", dtype)

    def length(self) -> Int:
        @parameter
        def f[T: Builder](b: T) -> Int:
            return b.length()

        return variant_dispatch[Builder, func=f](self._ptr[])

    def null_count(self) -> Int:
        @parameter
        def f[T: Builder](b: T) -> Int:
            return b.null_count()

        return variant_dispatch[Builder, func=f](self._ptr[])

    def dtype(self) -> DynType:
        @parameter
        def f[T: Builder](b: T) -> DynType:
            return b.dtype()

        return variant_dispatch[Builder, func=f](self._ptr[])

    def reserve(mut self, additional: Int) raises:
        @parameter
        def f[T: Builder](mut b: T) raises:
            b.reserve(additional)

        variant_dispatch_raises[Builder, func=f](self._ptr[])

    def append_null(mut self) raises:
        @parameter
        def f[T: Builder](mut b: T) raises:
            b.append_null()

        variant_dispatch_raises[Builder, func=f](self._ptr[])

    def extend(mut self, arr: DynArray) raises:
        @parameter
        def f[T: Builder](mut b: T) raises:
            b.extend(arr)

        variant_dispatch_raises[Builder, func=f](self._ptr[])

    comptime ArrayType = DynArray
    """`Builder`'s companion-array member. This is what `DynArray: Array`
    unblocked: the trait requires `ArrayType: Array`, and until the erased array
    conformed there was nothing for the erased builder to name."""

    def finish(mut self, *, shrink_to_fit: Bool = True) raises -> DynArray:
        @parameter
        def f[T: Builder](mut b: T) raises -> DynArray:
            return b.finish(shrink_to_fit=shrink_to_fit).to_dyn()

        return variant_dispatch_raises[Builder, func=f](self._ptr[])

    def reset(mut self) raises:
        @parameter
        def f[T: Builder](mut b: T) raises:
            b.reset()

        variant_dispatch_raises[Builder, func=f](self._ptr[])

    # --- typed downcasts (zero-cost reference borrows) ---

    def as_type[T: Builder](ref self) -> ref[self._ptr[][T]] T:
        """This builder as the concrete `T` it holds — a borrow, no copy."""
        debug_assert(
            self._ptr[].isa[T](), "_as: wrong type, holds ", self.dtype()
        )
        return self._ptr[][T]

    def as_primitive[
        T: PrimitiveType
    ](ref self) -> ref[self._ptr[][PrimitiveBuilder[T]]] PrimitiveBuilder[T]:
        return self.as_type[PrimitiveBuilder[T]]()

    def as_null(ref self) -> ref[self._ptr[][NullBuilder]] NullBuilder:
        return self.as_type[NullBuilder]()

    def as_bool(ref self) -> ref[self._ptr[][BoolBuilder]] BoolBuilder:
        return self.as_type[BoolBuilder]()

    def as_int8(ref self) -> ref[self._ptr[][Int8Builder]] Int8Builder:
        return self.as_type[Int8Builder]()

    def as_int16(ref self) -> ref[self._ptr[][Int16Builder]] Int16Builder:
        return self.as_type[Int16Builder]()

    def as_int32(ref self) -> ref[self._ptr[][Int32Builder]] Int32Builder:
        return self.as_type[Int32Builder]()

    def as_int64(ref self) -> ref[self._ptr[][Int64Builder]] Int64Builder:
        return self.as_type[Int64Builder]()

    def as_uint8(ref self) -> ref[self._ptr[][UInt8Builder]] UInt8Builder:
        return self.as_type[UInt8Builder]()

    def as_uint16(ref self) -> ref[self._ptr[][UInt16Builder]] UInt16Builder:
        return self.as_type[UInt16Builder]()

    def as_uint32(ref self) -> ref[self._ptr[][UInt32Builder]] UInt32Builder:
        return self.as_type[UInt32Builder]()

    def as_uint64(ref self) -> ref[self._ptr[][UInt64Builder]] UInt64Builder:
        return self.as_type[UInt64Builder]()

    def as_float16(ref self) -> ref[self._ptr[][Float16Builder]] Float16Builder:
        return self.as_type[Float16Builder]()

    def as_float32(ref self) -> ref[self._ptr[][Float32Builder]] Float32Builder:
        return self.as_type[Float32Builder]()

    def as_float64(ref self) -> ref[self._ptr[][Float64Builder]] Float64Builder:
        return self.as_type[Float64Builder]()

    def as_string(ref self) -> ref[self._ptr[][StringBuilder]] StringBuilder:
        return self.as_type[StringBuilder]()

    def as_binary(ref self) -> ref[self._ptr[][BinaryBuilder]] BinaryBuilder:
        return self.as_type[BinaryBuilder]()

    def as_large_string(
        ref self,
    ) -> ref[self._ptr[][LargeStringBuilder]] LargeStringBuilder:
        return self.as_type[LargeStringBuilder]()

    def as_large_binary(
        ref self,
    ) -> ref[self._ptr[][LargeBinaryBuilder]] LargeBinaryBuilder:
        return self.as_type[LargeBinaryBuilder]()

    def as_list(ref self) -> ref[self._ptr[][ListBuilder]] ListBuilder:
        return self.as_type[ListBuilder]()

    def as_large_list(
        ref self,
    ) -> ref[self._ptr[][LargeListBuilder]] LargeListBuilder:
        return self.as_type[LargeListBuilder]()

    def as_fixed_size_list(
        ref self,
    ) -> ref[self._ptr[][FixedSizeListBuilder]] FixedSizeListBuilder:
        return self.as_type[FixedSizeListBuilder]()

    def as_fixed_size_binary(
        ref self,
    ) -> ref[self._ptr[][FixedSizeBinaryBuilder]] FixedSizeBinaryBuilder:
        return self.as_type[FixedSizeBinaryBuilder]()

    def as_date32(ref self) -> ref[self._ptr[][Date32Builder]] Date32Builder:
        return self.as_type[Date32Builder]()

    def as_date64(ref self) -> ref[self._ptr[][Date64Builder]] Date64Builder:
        return self.as_type[Date64Builder]()

    def as_time32(ref self) -> ref[self._ptr[][Time32Builder]] Time32Builder:
        return self.as_type[Time32Builder]()

    def as_time64(ref self) -> ref[self._ptr[][Time64Builder]] Time64Builder:
        return self.as_type[Time64Builder]()

    def as_duration(
        ref self,
    ) -> ref[self._ptr[][DurationBuilder]] DurationBuilder:
        return self.as_type[DurationBuilder]()

    def as_timestamp(
        ref self,
    ) -> ref[self._ptr[][TimestampBuilder]] TimestampBuilder:
        return self.as_type[TimestampBuilder]()

    def as_year_month_interval(
        ref self,
    ) -> ref[self._ptr[][YearMonthIntervalBuilder]] YearMonthIntervalBuilder:
        return self.as_type[YearMonthIntervalBuilder]()

    def as_day_time_interval(
        ref self,
    ) -> ref[self._ptr[][DayTimeIntervalBuilder]] DayTimeIntervalBuilder:
        return self.as_type[DayTimeIntervalBuilder]()

    def as_month_day_nano_interval(
        ref self,
    ) -> ref[
        self._ptr[][MonthDayNanoIntervalBuilder]
    ] MonthDayNanoIntervalBuilder:
        return self.as_type[MonthDayNanoIntervalBuilder]()

    def as_decimal32(
        ref self,
    ) -> ref[self._ptr[][Decimal32Builder]] Decimal32Builder:
        return self.as_type[Decimal32Builder]()

    def as_decimal64(
        ref self,
    ) -> ref[self._ptr[][Decimal64Builder]] Decimal64Builder:
        return self.as_type[Decimal64Builder]()

    def as_decimal128(
        ref self,
    ) -> ref[self._ptr[][Decimal128Builder]] Decimal128Builder:
        return self.as_type[Decimal128Builder]()

    def as_decimal256(
        ref self,
    ) -> ref[self._ptr[][Decimal256Builder]] Decimal256Builder:
        return self.as_type[Decimal256Builder]()

    def as_struct(ref self) -> ref[self._ptr[][StructBuilder]] StructBuilder:
        return self.as_type[StructBuilder]()

    def as_dictionary(
        ref self,
    ) -> ref[self._ptr[][DictionaryBuilder]] DictionaryBuilder:
        return self.as_type[DictionaryBuilder]()


# ---------------------------------------------------------------------------
# PrimitiveBuilder
# ---------------------------------------------------------------------------


struct PrimitiveBuilder[T: PrimitiveType](Builder):
    """Builder for fixed-size primitive arrays (integers, floats, temporal, decimal).
    """

    comptime ArrayType = PrimitiveArray[Self.T]
    comptime ScalarType = Scalar[Self.T.native]

    var _dtype: Self.T
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _buffer: Buffer[mut=True]

    def __init__[
        NT: NumericType
    ](
        out self: PrimitiveBuilder[NT],
        capacity: Int = 0,
        *,
        zeroed: Bool = True,
    ):
        """Create a builder for a numeric type without an explicit dtype instance.

        Only available for NumericType. For temporal and decimal types, use
        the overload that accepts an explicit dtype.

        Args:
            capacity: Initial element capacity.
            zeroed: If True (default), zero-fill the data buffer. Pass
                False when every element will be written via
                ``unsafe_append`` — avoids wasted memset.
        """
        self = PrimitiveBuilder[NT](NT(), capacity, zeroed=zeroed)

    def __init__(
        out self, dtype: Self.T, capacity: Int = 0, *, zeroed: Bool = True
    ):
        """Create a builder with an explicit dtype (required for temporal and decimal).

        Args:
            dtype: The concrete type instance (carries unit/tz or precision/scale).
            capacity: Initial element capacity.
            zeroed: If True (default), zero-fill the data buffer.
        """
        self._dtype = dtype.copy()
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        if zeroed:
            self._buffer = Buffer.alloc_zeroed[Self.T.native](capacity)
        else:
            self._buffer = Buffer.alloc_uninit[Self.T.native](capacity)

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def unsafe_get(self, index: Int) -> Scalar[Self.T.native]:
        """Read element at index without bounds checking."""
        return self._buffer.unsafe_get[Self.T.native](index)

    def unsafe_set(mut self, index: Int, value: Scalar[Self.T.native]):
        """Write element at index without bounds checking."""
        self._buffer.unsafe_set[Self.T.native](index, value)

    def set_length(mut self, n: Int):
        """Commit the builder length after direct bulk population."""
        self._length = n

    def dtype(self) -> Self.T:
        return self._dtype.copy()

    def append(mut self, value: Optional[Self.ScalarType]) raises:
        """Append a value, treating None as null."""
        if value:
            self.append(value.value())
        else:
            self.append_null()

    def append(mut self, value: Self.ScalarType) raises:
        self.reserve(1)
        self.unsafe_append(value)

    @always_inline
    def unsafe_append(mut self, value: Self.ScalarType):
        """Append without bounds checking. Caller must ensure capacity."""
        self._bitmap.set(self._length)
        self._buffer.unsafe_set[Self.T.native](self._length, value)
        self._length += 1

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

    def append_nulls(mut self, count: Int) raises:
        """Append a run of `count` nulls — one bitmap clear per element, and the
        null count kept in step, which is what a caller writing the run itself
        would have had to remember."""
        self.reserve(count)
        self._bitmap.set_range(self._length, count, False)
        self._null_count += count
        self._length += count

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

    def extend(mut self, arr: DynArray) raises:
        self.extend(arr.as_primitive[Self.T]())

    def extend(mut self, arr: PrimitiveArray[Self.T]) raises:
        """Bulk-append all elements from an existing PrimitiveArray."""
        var n = arr.length
        self.reserve(n)
        if arr.null_count() == 0:
            self._bitmap.set_range(self._length, n, True)
        else:
            self._null_count += arr.null_count()
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
            dtype=self._dtype.copy(),
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
# BinaryBuilder
# ---------------------------------------------------------------------------


struct BinaryLikeBuilder[T: BinaryLikeType](Builder):
    """Builder for variable-length UTF-8 string or binary arrays."""

    comptime ArrayType = BinaryLikeArray[Self.T]

    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _offsets: Buffer[mut=True]
    var _values: Buffer[mut=True]

    def __init__(out self, capacity: Int = 0, bytes_capacity: Int = 0):
        var offsets = Buffer.alloc_zeroed[Self.T.offset](capacity + 1)
        offsets.unsafe_set[Self.T.offset](0, 0)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        self._offsets = offsets^
        self._values = Buffer.alloc_zeroed[DType.uint8](bytes_capacity)

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DynType:
        return Self.T().to_dyn()

    def append(mut self, value: String) raises:
        self.append(StringSlice(value))

    def append[origin: Origin](mut self, s: StringSlice[origin]) raises:
        self.reserve(1)
        self.reserve_bytes(s.byte_length())
        self.unsafe_append(s)

    def append_null(mut self) raises:
        self.reserve(1)
        var index = self._length
        var last_offset = self._offsets.unsafe_get[Self.T.offset](index)
        self._bitmap.clear(index)
        self._null_count += 1
        self._length += 1
        self._offsets.unsafe_set[Self.T.offset](index + 1, last_offset)

    def extend(mut self, values: List[String], valid: List[Bool]) raises:
        for i in range(len(values)):
            if valid[i]:
                self.append(values[i])
            else:
                self.append_null()

    def extend(mut self, arr: DynArray) raises:
        comptime if Self.T.offset == DType.int32:
            self.extend(arr.as_string())
        else:
            self.extend(arr.as_large_string())

    def extend[U: BinaryLikeType](mut self, arr: BinaryLikeArray[U]) raises:
        """Bulk-append all elements from an existing BinaryLikeArray."""
        var n = arr.length
        var chunk_start = Int(arr.offsets.unsafe_get[U.offset](arr.offset))
        var chunk_end = Int(arr.offsets.unsafe_get[U.offset](arr.offset + n))
        var chunk_bytes = chunk_end - chunk_start
        self.reserve(n)
        self.reserve_bytes(chunk_bytes)
        if arr.null_count() == 0:
            self._bitmap.set_range(self._length, n, True)
        else:
            self._null_count += arr.null_count()
            if arr.bitmap:
                var bm = arr.bitmap.value()
                self._bitmap.extend(bm.view(arr.offset, n), self._length, n)
            else:
                self._bitmap.set_range(self._length, n, True)
        var cur_bytes = Int(
            self._offsets.unsafe_get[Self.T.offset](self._length)
        )
        for i in range(n):
            var orig = Int(arr.offsets.unsafe_get[U.offset](arr.offset + i))
            self._offsets.unsafe_set[Self.T.offset](
                self._length + i,
                Scalar[Self.T.offset](cur_bytes + orig - chunk_start),
            )
        self._offsets.unsafe_set[Self.T.offset](
            self._length + n,
            Scalar[Self.T.offset](cur_bytes + chunk_bytes),
        )
        self._values.view[DType.uint8](cur_bytes).copy_from(
            arr.values.view[DType.uint8](chunk_start), chunk_bytes
        )
        self._length += n

    def reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._offsets.resize[Self.T.offset](new_cap + 1)
            self._capacity = new_cap

    def reserve_bytes(mut self, additional: Int) raises:
        """Pre-allocate space in the byte data buffer."""
        var used = Int(self._offsets.unsafe_get[Self.T.offset](self._length))
        var needed = used + additional
        if needed > len(self._values):
            var new_cap = max(len(self._values) * 2, needed)
            self._values.resize[DType.uint8](new_cap)

    @always_inline
    def unsafe_append[origin: Origin](mut self, s: StringSlice[origin]):
        """Append string bytes without capacity checks. Caller must ensure capacity.
        """
        var length = s.byte_length()
        var index = self._length
        var last_offset = self._offsets.unsafe_get[Self.T.offset](index)
        var next_offset = last_offset + Scalar[Self.T.offset](length)
        self._bitmap.set(index)
        self._offsets.unsafe_set[Self.T.offset](index + 1, next_offset)
        self._values.view[DType.uint8](Int(last_offset)).copy_from(s)
        self._length += 1

    @always_inline
    def unsafe_append_null(mut self):
        """Append null without capacity checks. Caller must ensure capacity."""
        var index = self._length
        var last_offset = self._offsets.unsafe_get[Self.T.offset](index)
        self._bitmap.clear(index)
        self._null_count += 1
        self._offsets.unsafe_set[Self.T.offset](index + 1, last_offset)
        self._length += 1

    def finish(
        mut self, *, shrink_to_fit: Bool = True
    ) raises -> BinaryLikeArray[Self.T]:
        if shrink_to_fit:
            self._offsets.resize[Self.T.offset](self._length + 1)
            var used = Int(
                self._offsets.unsafe_get[Self.T.offset](self._length)
            )
            self._values.resize[DType.uint8](used)
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap[]] = None
        if null_count != 0:
            bm = self._bitmap^.to_immutable(length=self._length)
            self._bitmap = Bitmap.alloc_zeroed(0)
        # freeze offsets and byte data buffers into immutable Buffers
        var offsets = self._offsets^.to_immutable()
        self._offsets = Buffer.alloc_zeroed[Self.T.offset](0)
        var values = self._values^.to_immutable()
        self._values = Buffer.alloc_zeroed(0)
        # construct the immutable result array
        var result = BinaryLikeArray[Self.T](
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


comptime BinaryBuilder = BinaryLikeBuilder[BinaryType]
comptime LargeBinaryBuilder = BinaryLikeBuilder[LargeBinaryType]
comptime StringBuilder = BinaryLikeBuilder[StringType]
comptime LargeStringBuilder = BinaryLikeBuilder[LargeStringType]


# ---------------------------------------------------------------------------
# ListBuilder
# ---------------------------------------------------------------------------


struct ListLikeBuilder[T: ListLikeType](Builder):
    """Builder for variable-length list arrays."""

    comptime ArrayType = ListLikeArray[Self.T]

    var _dtype: DynType
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _offsets: Buffer[mut=True]
    var _child: DynBuilder

    def __init__(out self, var child: DynBuilder, capacity: Int = 0):
        var offsets = Buffer.alloc_zeroed[Self.T.offset](capacity + 1)
        offsets.unsafe_set[Self.T.offset](0, 0)
        var child_dtype = child.dtype().copy()
        var list_dtype: DynType
        comptime if Self.T.offset == DType.int32:
            list_dtype = list_(child_dtype^)
        else:
            list_dtype = large_list_(child_dtype^)
        self._dtype = list_dtype^
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        self._offsets = offsets^
        self._child = child^

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DynType:
        return self._dtype.copy()

    def values(self) -> DynBuilder:
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
        self._offsets.unsafe_set[Self.T.offset](
            self._length + 1, Scalar[Self.T.offset](child_length)
        )
        self._length += 1

    @always_inline
    def unsafe_append_valid(mut self):
        """Append valid without capacity check. Caller must ensure capacity."""
        self._bitmap.set(self._length)
        var child_length = self._child.length()
        self._offsets.unsafe_set[Self.T.offset](
            self._length + 1, Scalar[Self.T.offset](child_length)
        )
        self._length += 1

    def extend(mut self, arr: DynArray) raises:
        comptime if Self.T.offset == DType.int32:
            self.extend(arr.as_list())
        else:
            self.extend(arr.as_large_list())

    def extend[U: ListLikeType](mut self, arr: ListLikeArray[U]) raises:
        """Bulk-append all elements from an existing ListLikeArray."""
        var n = arr.length
        self.reserve(n)
        if arr.null_count() == 0:
            self._bitmap.set_range(self._length, n, True)
        else:
            self._null_count += arr.null_count()
            if arr.bitmap:
                var bm = arr.bitmap.value()
                self._bitmap.extend(bm.view(arr.offset, n), self._length, n)
            else:
                self._bitmap.set_range(self._length, n, True)
        var child_start = Int(arr.offsets.unsafe_get[U.offset](arr.offset))
        var child_end = Int(arr.offsets.unsafe_get[U.offset](arr.offset + n))
        var cur_child_len = self._child.length()
        for i in range(n):
            var orig = Int(arr.offsets.unsafe_get[U.offset](arr.offset + i))
            self._offsets.unsafe_set[Self.T.offset](
                self._length + i,
                Scalar[Self.T.offset](cur_child_len + orig - child_start),
            )
        self._offsets.unsafe_set[Self.T.offset](
            self._length + n,
            Scalar[Self.T.offset](cur_child_len + child_end - child_start),
        )
        var child_slice = arr.values().slice(
            child_start, child_end - child_start
        )
        self._child.extend(child_slice)
        self._length += n

    def reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._offsets.resize[Self.T.offset](new_cap + 1)
            self._capacity = new_cap

    def finish(
        mut self, *, shrink_to_fit: Bool = True
    ) raises -> ListLikeArray[Self.T]:
        if shrink_to_fit:
            self._offsets.resize[Self.T.offset](self._length + 1)
        # only materialise the validity bitmap when there are nulls
        var null_count = self._null_count
        var bm: Optional[Bitmap[]] = None
        if null_count != 0:
            bm = self._bitmap^.to_immutable(length=self._length)
            self._bitmap = Bitmap.alloc_zeroed(0)
        # freeze offsets buffer and recursively finish the child builder
        var offsets = self._offsets^.to_immutable()
        self._offsets = Buffer.alloc_zeroed[Self.T.offset](0)
        var values = self._child.finish()
        # construct the immutable result array
        var result = ListLikeArray[Self.T](
            dtype=self._dtype.copy(),
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


comptime ListBuilder = ListLikeBuilder[ListType]
comptime LargeListBuilder = ListLikeBuilder[LargeListType]


# ---------------------------------------------------------------------------
# MapBuilder
# ---------------------------------------------------------------------------


struct MapBuilder(Builder):
    """Builder for map arrays. A map is physically a list of a non-nullable
    ``entries`` struct of (key, value), so this is a thin composition over a
    `ListBuilder` whose child is that struct — the map dtype is applied only at
    the `dtype()`/`finish()` boundary, and incoming `MapArray`s are retagged as
    lists for `extend`. Append entries through `entries()` (the struct builder)
    and mark map boundaries with `append_valid`/`append_null`."""

    comptime ArrayType = MapArray

    var _inner: ListBuilder
    var _dtype: DynType

    def __init__(out self, dtype: MapType, capacity: Int = 0) raises:
        var entries = StructBuilder(
            [dtype.key_field(), dtype.item_field()], capacity
        )
        self._inner = ListBuilder(entries^, capacity)
        self._dtype = MapType(copy=dtype)

    def length(self) -> Int:
        return self._inner.length()

    def null_count(self) -> Int:
        return self._inner.null_count()

    def dtype(self) -> DynType:
        return self._dtype.copy()

    def entries(self) -> DynBuilder:
        """The (key, value) entries struct builder — append map entries here,
        then call `append_valid`/`append_null` to close each map."""
        return self._inner.values()

    def reserve(mut self, additional: Int) raises:
        self._inner.reserve(additional)

    def append_null(mut self) raises:
        self._inner.append_null()

    def append_valid(mut self) raises:
        self._inner.append_valid()

    def extend(mut self, arr: DynArray) raises:
        # Feed the map to the inner ListBuilder as a plain list of its entries.
        self._inner.extend(arr.as_map().to_list())

    def finish(mut self, *, shrink_to_fit: Bool = True) raises -> MapArray:
        return self._inner.finish(shrink_to_fit=shrink_to_fit).to_map(
            self._dtype.as_map().keys_sorted
        )

    def reset(mut self):
        self._inner.reset()


# ---------------------------------------------------------------------------
# FixedSizeListBuilder
# ---------------------------------------------------------------------------


struct FixedSizeListBuilder(Builder):
    """Builder for fixed-size list arrays.

    _child — child element builder (DynBuilder)
    """

    comptime ArrayType = FixedSizeListArray

    var _dtype: DynType
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _child: DynBuilder

    def __init__(
        out self, var child: DynBuilder, list_size: Int, capacity: Int = 0
    ):
        var child_dtype = child.dtype().copy()
        self._dtype = fixed_size_list_(child_dtype^, list_size)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        self._child = child^

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DynType:
        return self._dtype.copy()

    def values(self) -> DynBuilder:
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

    def extend(mut self, arr: DynArray) raises:
        self.extend(arr.as_fixed_size_list())

    def extend(mut self, arr: FixedSizeListArray) raises:
        """Bulk-append all elements from an existing FixedSizeListArray."""
        var n = arr.length
        self.reserve(n)
        if arr.null_count() == 0:
            self._bitmap.set_range(self._length, n, True)
        else:
            self._null_count += arr.null_count()
            if arr.bitmap:
                var bm = arr.bitmap.value()
                self._bitmap.extend(bm.view(arr.offset, n), self._length, n)
            else:
                self._bitmap.set_range(self._length, n, True)
        var list_size = arr.dtype.as_fixed_size_list().size
        var child_slice = arr.values().slice(
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
            dtype=self._dtype.copy(),
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


struct StructBuilder(Builder):
    """Builder for struct arrays.

    _children[i] — field builder for field i (DynBuilder)
    """

    comptime ArrayType = StructArray

    var _dtype: DynType
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _children: List[DynBuilder]

    def __init__(out self, var fields: List[Field], capacity: Int = 0) raises:
        var children = List[DynBuilder](capacity=len(fields))
        for i in range(len(fields)):
            children.append(DynBuilder(fields[i].dtype))
        self._dtype = struct_(fields^)
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        self._children = children^

    # Explicit (empty) destructor so this struct's `_children: List[DynBuilder]`
    # (which cycles back through DynBuilder's variant) is ImplicitlyDeletable;
    # fields are still destroyed automatically after the body runs.
    def __del__(deinit self):
        pass

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DynType:
        return self._dtype.copy()

    def field_builder(
        ref self, index: Int
    ) -> ref[self._children[index]] DynBuilder:
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
    def extend(mut self, arr: DynArray) raises:
        self.extend(arr.as_struct())

    def extend(mut self, arr: StructArray) raises:
        """Bulk-append all elements from an existing StructArray (honouring the
        source's logical offset for both validity and children)."""
        var n = arr.length
        self.reserve(n)
        if arr.null_count() == 0:
            self._bitmap.set_range(self._length, n, True)
        else:
            self._null_count += arr.null_count()
            if arr.bitmap:
                var bm = arr.bitmap.value()
                self._bitmap.extend(bm.view(arr.offset, n), self._length, n)
            else:
                self._bitmap.set_range(self._length, n, True)
        for f in range(len(arr.children)):
            self._children[f].extend(arr.children[f].slice(arr.offset, n))
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
        var frozen_children = List[DynArray](capacity=len(self._children))
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
# DictionaryBuilder — builds dictionary-encoded arrays
# ---------------------------------------------------------------------------


struct DictionaryBuilder(Builder):
    """Builder for dictionary-encoded arrays.

    Wraps an indices builder (for any integer type) and a fixed dictionary
    values array. Call ``append(index)`` to add index values; call ``finish()``
    to produce a ``DictionaryArray``.

    Equivalent to PyArrow's pattern of maintaining a pre-built dictionary and
    appending integer indices.
    """

    comptime ArrayType = DictionaryArray

    var _dtype: DynType
    var _indices: DynBuilder
    var _values: DynArray

    def __init__(
        out self,
        var indices_builder: DynBuilder,
        var values: DynArray,
        ordered: Bool = False,
    ) raises:
        self._dtype = dictionary(
            indices_builder.dtype(), values.dtype(), ordered
        )
        self._indices = indices_builder^
        self._values = values^

    def length(self) -> Int:
        return self._indices.length()

    def null_count(self) -> Int:
        return self._indices.null_count()

    def dtype(self) -> DynType:
        return self._dtype.copy()

    def reserve(mut self, additional: Int) raises:
        self._indices.reserve(additional)

    def append_null(mut self) raises:
        self._indices.append_null()

    def append(mut self, index: Int) raises:
        """Append an integer index into the dictionary."""
        ref index_type = self._dtype.as_dictionary().index_type()
        if index_type.is_int8():
            self._indices.as_int8().append(Int8(index))
        elif index_type.is_int16():
            self._indices.as_int16().append(Int16(index))
        elif index_type.is_int32():
            self._indices.as_int32().append(Int32(index))
        elif index_type.is_int64():
            self._indices.as_int64().append(Int64(index))
        elif index_type.is_uint8():
            self._indices.as_uint8().append(UInt8(index))
        elif index_type.is_uint16():
            self._indices.as_uint16().append(UInt16(index))
        elif index_type.is_uint32():
            self._indices.as_uint32().append(UInt32(index))
        elif index_type.is_uint64():
            self._indices.as_uint64().append(UInt64(index))
        else:
            raise Error(
                "DictionaryBuilder.append: unexpected index type: ",
                index_type,
            )

    def extend(mut self, arr: DynArray) raises:
        if not arr.dtype().is_dictionary():
            raise Error(
                "DictionaryBuilder.extend: expected DictionaryArray, got: ",
                arr.dtype(),
            )
        self._indices.extend(arr.as_dictionary().indices())

    def finish(
        mut self, *, shrink_to_fit: Bool = True
    ) raises -> DictionaryArray:
        var indices = self._indices.finish()
        # `ordered` is carried on `_dtype`, not derivable from the arrays —
        # `from_arrays` defaults it to False, which silently dropped the flag
        # the builder was constructed with.
        return DictionaryArray.from_arrays(
            indices^,
            self._values.copy(),
            ordered=self._dtype.as_dictionary().ordered,
        )

    def reset(mut self):
        try:
            self._indices.reset()
        except:
            pass


# ---------------------------------------------------------------------------
# BoolBuilder — bit-packed boolean array builder
# ---------------------------------------------------------------------------


struct NullBuilder(Builder):
    """Builder for NullArray — a length-only counter, no buffers."""

    comptime ArrayType = NullArray

    var _length: Int

    def __init__(out self, capacity: Int = 0):
        self._length = 0

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._length

    def dtype(self) -> DynType:
        return null

    def reserve(mut self, additional: Int) raises:
        pass

    def append_null(mut self) raises:
        self._length += 1

    def extend(mut self, arr: DynArray) raises:
        if not arr.dtype().is_null():
            raise Error("NullBuilder.extend: expected a null array")
        self._length += len(arr)

    def finish(mut self, *, shrink_to_fit: Bool = True) raises -> NullArray:
        var n = self._length
        self._length = 0
        return NullArray(length=n)

    def reset(mut self):
        self._length = 0


struct FixedSizeBinaryBuilder(Builder):
    """Builder for FixedSizeBinaryArray — fixed-width binary values."""

    comptime ArrayType = FixedSizeBinaryArray

    var _byte_width: Int
    var _length: Int
    var _capacity: Int
    var _null_count: Int
    var _bitmap: Bitmap[mut=True]
    var _buffer: Buffer[mut=True]

    def __init__(out self, byte_width: Int, capacity: Int = 0):
        self._byte_width = byte_width
        self._length = 0
        self._capacity = capacity
        self._null_count = 0
        self._bitmap = Bitmap.alloc_zeroed(capacity)
        self._buffer = Buffer.alloc_zeroed[DType.uint8](capacity * byte_width)

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DynType:
        return FixedSizeBinaryType(self._byte_width).to_dyn()

    def reserve(mut self, additional: Int) raises:
        var needed = self._length + additional
        if needed > self._capacity:
            var new_cap = max(self._capacity * 2, needed)
            self._bitmap.resize(new_cap)
            self._buffer.resize[DType.uint8](new_cap * self._byte_width)
            self._capacity = new_cap

    def append(mut self, bytes: Span[UInt8, _]) raises:
        if len(bytes) != self._byte_width:
            raise Error(
                "FixedSizeBinaryBuilder.append: expected ",
                self._byte_width,
                " bytes, got ",
                len(bytes),
            )
        self.reserve(1)
        var index = self._length
        self._bitmap.set(index)
        var start = index * self._byte_width
        for k in range(self._byte_width):
            self._buffer.unsafe_set[DType.uint8](start + k, bytes[k])
        self._length += 1

    def append_null(mut self) raises:
        self.reserve(1)
        var index = self._length
        self._bitmap.clear(index)
        # Zero-pad the slot so the data buffer never has uninit bytes.
        var start = index * self._byte_width
        for k in range(self._byte_width):
            self._buffer.unsafe_set[DType.uint8](start + k, UInt8(0))
        self._null_count += 1
        self._length += 1

    def extend(mut self, arr: DynArray) raises:
        if not arr.dtype().is_fixed_size_binary():
            raise Error(
                "FixedSizeBinaryBuilder.extend: expected fixed_size_binary"
                " array"
            )
        self.extend(arr.as_fixed_size_binary())

    def extend(mut self, b: FixedSizeBinaryArray) raises:
        if b.byte_width != self._byte_width:
            raise Error(
                "FixedSizeBinaryBuilder.extend: byte_width mismatch ",
                b.byte_width,
                " vs ",
                self._byte_width,
            )
        self.reserve(b.length)
        var dst_start = self._length * self._byte_width
        var src_start = b.offset * self._byte_width
        var n_bytes = b.length * self._byte_width
        for i in range(n_bytes):
            self._buffer.unsafe_set[DType.uint8](
                dst_start + i, b.buffer.unsafe_get[DType.uint8](src_start + i)
            )
        if b.null_count() != 0:
            if b.bitmap:
                var bm = b.bitmap.value()
                self._bitmap.extend(
                    bm.view(b.offset, b.length), self._length, b.length
                )
            self._null_count += b.null_count()
        else:
            self._bitmap.set_range(self._length, b.length, True)
        self._length += b.length

    def finish(
        mut self, *, shrink_to_fit: Bool = True
    ) raises -> FixedSizeBinaryArray:
        if shrink_to_fit:
            self._buffer.resize[DType.uint8](self._length * self._byte_width)
        var null_count = self._null_count
        var bm: Optional[Bitmap[]] = None
        if null_count != 0:
            bm = self._bitmap^.to_immutable(length=self._length)
            self._bitmap = Bitmap.alloc_zeroed(0)
        var values = self._buffer^.to_immutable()
        self._buffer = Buffer.alloc_zeroed[DType.uint8](0)
        var result = FixedSizeBinaryArray(
            length=self._length,
            nulls=null_count,
            offset=0,
            byte_width=self._byte_width,
            bitmap=bm^,
            buffer=values^,
        )
        self.reset()
        return result^

    def reset(mut self):
        self._length = 0
        self._capacity = 0
        self._null_count = 0


struct BoolBuilder(Builder):
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

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> DynType:
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

    def extend(mut self, arr: DynArray) raises:
        self.extend(arr.as_bool())

    def extend(mut self, b: BoolArray) raises:
        self.reserve(b.length)
        self._buffer.extend(b.values(), self._length, b.length)
        if b.null_count() != 0:
            if b.bitmap:
                self._bitmap.extend(
                    b.validity().value(), self._length, b.length
                )
            self._null_count += b.null_count()
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
comptime Int8Builder = PrimitiveBuilder[Int8Type]
comptime Int16Builder = PrimitiveBuilder[Int16Type]
comptime Int32Builder = PrimitiveBuilder[Int32Type]
comptime Int64Builder = PrimitiveBuilder[Int64Type]
comptime UInt8Builder = PrimitiveBuilder[UInt8Type]
comptime UInt16Builder = PrimitiveBuilder[UInt16Type]
comptime UInt32Builder = PrimitiveBuilder[UInt32Type]
comptime UInt64Builder = PrimitiveBuilder[UInt64Type]
comptime Float16Builder = PrimitiveBuilder[Float16Type]
comptime Float32Builder = PrimitiveBuilder[Float32Type]
comptime Float64Builder = PrimitiveBuilder[Float64Type]

comptime Date32Builder = PrimitiveBuilder[Date32Type]
comptime Date64Builder = PrimitiveBuilder[Date64Type]
comptime Time32Builder = PrimitiveBuilder[Time32Type]
comptime Time64Builder = PrimitiveBuilder[Time64Type]
comptime DurationBuilder = PrimitiveBuilder[DurationType]
comptime TimestampBuilder = PrimitiveBuilder[TimestampType]

comptime YearMonthIntervalBuilder = PrimitiveBuilder[YearMonthIntervalType]
comptime DayTimeIntervalBuilder = PrimitiveBuilder[DayTimeIntervalType]
comptime MonthDayNanoIntervalBuilder = PrimitiveBuilder[
    MonthDayNanoIntervalType
]

comptime Decimal32Builder = PrimitiveBuilder[Decimal32Type]
comptime Decimal64Builder = PrimitiveBuilder[Decimal64Type]
comptime Decimal128Builder = PrimitiveBuilder[Decimal128Type]
comptime Decimal256Builder = PrimitiveBuilder[Decimal256Type]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def array[
    T: NumericType
](values: List[Scalar[T.native]], type: T) raises -> PrimitiveArray[T]:
    """Create a primitive array from native scalars (mirrors ``pa.array``)."""
    var b = PrimitiveBuilder[T](len(values))
    for i in range(len(values)):
        b.unsafe_append(values[i])
    return b.finish()


def array[
    T: NumericType
](values: List[Optional[Scalar[T.native]]], type: T) raises -> PrimitiveArray[
    T
]:
    """Create a primitive array from optional native scalars (None → null)."""
    var b = PrimitiveBuilder[T](len(values))
    for i in range(len(values)):
        var v = values[i]
        if v:
            b.unsafe_append(v.value())
        else:
            b.append_null()
    return b.finish()


def array[T: NumericType](type: T) raises -> PrimitiveArray[T]:
    """Create an empty primitive array of the given type."""
    var b = PrimitiveBuilder[T](0)
    return b.finish()


def array[
    T: NumericType
](values: List[Optional[Int]], type: T) raises -> PrimitiveArray[T]:
    """Create a primitive array from integer literals, with optional nulls.

    Accepts list literals like ``[1, None, 3]`` or ``[10, 20, 30]`` and
    converts each element to ``Scalar[T.native]``.
    """
    var b = PrimitiveBuilder[T](len(values))
    for i in range(len(values)):
        var v = values[i]
        if v:
            b.unsafe_append(Scalar[T.native](v.value()))
        else:
            b.append_null()
    return b.finish()


def array[
    T: NumericType
](values: List[Optional[Float64]], type: T) raises -> PrimitiveArray[T]:
    """Create a primitive array from float literals, with optional nulls.

    Accepts list literals like ``[1.5, None, 3.14]`` or ``[1.0, 2.0]`` and
    converts each element to ``Scalar[T.native]``.
    """
    var b = PrimitiveBuilder[T](len(values))
    for i in range(len(values)):
        var v = values[i]
        if v:
            b.unsafe_append(Scalar[T.native](v.value()))
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


def nulls[T: NumericType](size: Int, type: T) raises -> PrimitiveArray[T]:
    """Create a primitive array of `size` null values.

    Mirrors ``pa.nulls(size, type=pa.int64())``.
    """
    var b = PrimitiveBuilder[T](capacity=size)
    b.append_nulls(size)
    return b.finish()


def arange[T: NumericType](start: Int, end: Int) raises -> PrimitiveArray[T]:
    """Create a numeric array with values [start, end)."""
    comptime assert (
        T.native != DType.bool
    ), "arange() only supports numeric types"
    var b = PrimitiveBuilder[T](T(), end - start)
    for i in range(start, end):
        b.append(Scalar[T.native](i))
    return b.finish()


def array(dtype: DynType) raises -> DynArray:
    """Create a zero-length empty array for the given dtype."""
    var b = DynBuilder(dtype)
    return b.finish()
