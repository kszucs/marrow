"""Arrow columnar arrays — always immutable.

Every typed array (`PrimitiveArray`, `BinaryArray`, `ListArray`, `StructArray`)
is immutable.  To *build* an array incrementally, use the corresponding builder
from `marrow.builders` and call `finish()`.

`BoolArray` is a dedicated bit-packed boolean array type.

Array — the trait
-----------------
`Array` is the trait that all typed arrays implement.  It provides the common
read-only interface: `type()`, `null_count()`, `is_valid()`, and `to_dyn()`.

DynArray — the type-erased handle
----------------------------------
`DynArray` is the type-erased, immutable handle backed by an inline `Variant`.
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
paths.  It is NOT stored inside DynArray.
"""


from std.memory import OwnedPointer

from std.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from std.utils import Variant

from .buffers import Buffer, Bitmap
from .views import BufferView, BitmapView
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
    IntegerType,
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
    field,
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
from .builders import DynBuilder, PrimitiveBuilder, BinaryLikeBuilder
from .scalars import (
    DynScalar,
    NullScalar,
    BoolScalar,
    FixedSizeBinaryScalar,
    PrimitiveScalar,
    StringScalar,
    ListScalar,
    StructScalar,
    DictionaryScalar,
    ArrowScalar,
)


trait Array(
    Copyable,
    Equatable,
    ImplicitlyDeletable,
    Movable,
    Sized,
    Writable,
):
    """Common interface for all typed Arrow arrays.

    All concrete array types (PrimitiveArray, BinaryArray, ListArray,
    FixedSizeListArray, StructArray) implement this trait.  DynArray is
    the type-erased handle that wraps any Array-conforming type.
    """

    comptime ScalarType: ArrowScalar

    def __init__(out self, data: ArrayData) raises:
        ...

    def type(self) -> DynType:
        ...

    def null_count(self) -> Int:
        ...

    def is_valid(self, index: Int) -> Bool:
        ...

    def is_null(self, index: Int) -> Bool:
        return not self.is_valid(index)

    def to_dyn(deinit self) -> DynArray:
        return DynArray(self^)

    def to_device(self, ctx: DeviceContext) raises -> Self:
        raise Error("to_device: not supported for this array type")

    def to_cpu(self, ctx: DeviceContext) raises -> Self:
        raise Error("to_cpu: not supported for this array type")

    def to_data(self) raises -> ArrayData:
        ...

    def slice(self, offset: Int, length: Int) raises -> Self:
        """A zero-copy slice.

        `raises` because the erased implementation can fail: `DynArray.slice`
        dispatches over its variant, and an uncovered member falls through. Every
        *typed* implementation is non-raising and stays that way -- Mojo accepts
        a non-raising body against a raising requirement, so typed call sites are
        unaffected. Only generic code holding `[T: Array]` has to propagate.
        """
        ...

    def __getitem__(self, index: Int) raises -> Self.ScalarType:
        ...


def _sliced_null_count(
    parent_nulls: Int,
    parent_length: Int,
    bitmap: Optional[Bitmap[mut=False]],
    offset: Int,
    length: Int,
) -> Int:
    """The null count of a sub-range, derived from the parent's.

    Computed eagerly at `slice()` rather than deferred behind a "not computed"
    sentinel, because every lazy design needs somewhere to cache the answer:
    Arrow C++ uses a `mutable std::atomic<int64_t>` (`kUnknownNullCount = -1`,
    resolved in `GetNullCount`), polars a `RelaxedCell<u64>` inside `Bitmap`
    (`u64::MAX` = unknown). Mojo has no interior mutability here and
    `null_count()` is an immutable method, so a sentinel would re-count on every
    call instead of once — strictly worse than counting once here.

    arrow-rs makes the same choice for the same reason, keeping the count as an
    invariant of the validity buffer rather than a cache. Polars, which slices
    hardest, is lazy *inside* the bitmap but still forces the count at the array
    boundary on every slice, so that a null-free slice can drop its bitmap
    entirely and give downstream kernels an all-valid fast path.

    Measured: the sentinel cost +12,980 bytes of `__text` on the AOT gate
    against +8,452 for this, because both `slice()` and `null_count()` are
    dispatched over a 37-member variant and the sentinel puts branching code in
    both.

    Not implemented, and worth having if slicing ever shows up in a profile:
    polars counts only the *discarded* head and tail and subtracts, when the
    slice keeps at least ~80% of the parent.
    """
    # Two exact answers without scanning, both of which every reference
    # implementation carries: a null-free parent has a null-free child, and an
    # all-null parent has an all-null child. On a streaming morsel path over
    # NOT NULL columns the first arm carries essentially all the traffic.
    if parent_nulls == 0 or not bitmap:
        return 0
    if parent_nulls == parent_length:
        return length
    return bitmap.value().view(offset, length).unset_count()


# ---------------------------------------------------------------------------
# ArrayData — generic flat layout, produced on demand by to_data()
# ---------------------------------------------------------------------------


@fieldwise_init
struct ArrayData(Copyable, Movable):
    """Generic array layout — the old DynArray wire format, now a pure DTO.

    Produced by `typed_array.to_data()` or `any_array.to_data()` for use
    in the C Data Interface, construction helpers, and other interop paths.
    Not stored inside DynArray itself.
    """

    var dtype: DynType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var buffers: List[Buffer[mut=False]]
    var children: List[ArrayData]

    # Explicit (empty) destructor so this self-referential struct
    # (`children: List[ArrayData]`) is ImplicitlyDeletable; fields are still
    # destroyed automatically after the body runs.
    def validity(
        ref self,
    ) -> Optional[BitmapView[origin_of(self.bitmap._value)]]:
        """Validity bitmap view, or None if all values are valid."""
        if not self.bitmap:
            return None
        return self.bitmap.value().view(self.offset, self.length)

    def owned_validity(self) raises -> Optional[Bitmap[mut=False]]:
        """Validity as an *offset-0 owned* bitmap (None = all valid).

        `validity()` hands out a view into this array's own buffer at its own
        offset; a consumer that bakes the validity into a differently-offset
        result needs its own copy, and this is that copy.
        """
        var v = self.validity()
        if v:
            return v.value().to_owned()
        else:
            return None

    def __del__(deinit self):
        pass


# ---------------------------------------------------------------------------
# BoolArray
# ---------------------------------------------------------------------------


@fieldwise_init
struct NullArray(Array):
    """Immutable array of nulls — Arrow's `Null` type.

    Holds nothing but a length: every element is null.  The Arrow spec
    prescribes zero body buffers (no validity, no data) and `null_count`
    equal to `length`.
    """

    comptime ScalarType = NullScalar

    var length: Int

    def __init__(out self, data: ArrayData) raises:
        self.length = data.length

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def type(self) -> DynType:
        return null

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        var actual_length = length if length >= 0 else self.length - offset
        return Self(length=actual_length)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("NullArray(", self.length, ")")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def null_count(self) -> Int:
        return self.length

    def is_valid(self, index: Int) -> Bool:
        return False

    def __getitem__(self, index: Int) -> NullScalar:
        return NullScalar()

    def to_data(self) raises -> ArrayData:
        return ArrayData(
            dtype=null,
            length=self.length,
            nulls=self.length,
            offset=0,
            bitmap=None,
            buffers=List[Buffer[mut=False]](),
            children=List[ArrayData](),
        )

    def __eq__(self, other: Self) -> Bool:
        return self.length == other.length


@fieldwise_init
struct BoolArray(Array):
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

    @staticmethod
    def empty() raises -> BoolArray:
        """A zero-length bool array."""
        return BoolArray(
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=Bitmap.alloc_zeroed(0).to_immutable(),
        )

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

    def type(self) -> DynType:
        return bool_

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array.

        Matches PyArrow's Array.slice(offset, length) API.
        """
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            length=actual_length,
            # the parent's count says nothing about a sub-range; count this
            # window. A parent with no nulls has a child with none, so the
            # common case never touches the bitmap.
            nulls=_sliced_null_count(
                self.nulls,
                self.length,
                self.bitmap,
                self.offset + offset,
                length,
            ),
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
                # `values()` is already offset-applied — adding `self.offset`
                # again reads `2*offset + i`. `is_valid` above goes through the
                # raw `bitmap`, which is *not* offset-applied, hence the
                # asymmetry between these two lines.
                writer.write("True" if self.values().test(i) else "False")
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
        # `values()` is offset-applied; `self.bitmap` (used by `is_valid`) is not.
        return BoolScalar(self.values().test(index))

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
            nulls=self.null_count(),
            # see PrimitiveArray.to_device: the whole bit-packed buffer moves,
            # so the offset stays meaningful and must be carried over
            offset=self.offset,
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
            nulls=self.null_count(),
            # see PrimitiveArray.to_device: the whole bit-packed buffer moves,
            # so the offset stays meaningful and must be carried over
            offset=self.offset,
            bitmap=bm^,
            buffer=self.buffer.to_cpu(ctx),
        )

    def to_data(self) raises -> ArrayData:
        return ArrayData(
            dtype=bool_,
            length=self.length,
            nulls=self.null_count(),
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[self.buffer._buffer],
            children=[],
        )

    def __eq__(self, other: Self) -> Bool:
        """Return True if both arrays have the same length, null pattern, and values.
        """
        if self.length != other.length or self.null_count() != other.null_count():
            return False
        for i in range(self.length):
            var lv = self.is_valid(i)
            var rv = other.is_valid(i)
            if lv != rv:
                return False
            if lv and self[i] != other[i]:
                return False
        return True


# ---------------------------------------------------------------------------
# PrimitiveArray[T]
# ---------------------------------------------------------------------------


# TODO: add conditional conformance where: T.is_primitive()
struct PrimitiveArray[T: PrimitiveType](Array):
    """An immutable Arrow array of fixed-size primitive values (integers, floats, etc.).
    """

    comptime ScalarType = PrimitiveScalar[Self.T]

    comptime scalar = Scalar[Self.T.native]

    # TODO: make these protected to discourage direct access
    var dtype: Self.T
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var buffer: Buffer[mut=False]

    @staticmethod
    def empty(dtype: Self.T) raises -> Self:
        """A zero-length array of `dtype`."""
        return Self(
            dtype=dtype,
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=Buffer.alloc_zeroed[Self.T.native](0).to_immutable(),
        )

    def __init__(
        out self,
        dtype: Self.T,
        *,
        length: Int,
        nulls: Int,
        offset: Int,
        bitmap: Optional[Bitmap[mut=False]],
        buffer: Buffer[mut=False],
    ):
        self.dtype = dtype
        self.length = length
        self.nulls = nulls
        self.offset = offset
        self.bitmap = bitmap
        self.buffer = buffer

    def __init__[
        DT: NumericType
    ](
        out self: PrimitiveArray[DT],
        *,
        length: Int,
        nulls: Int,
        offset: Int,
        bitmap: Optional[Bitmap[mut=False]],
        buffer: Buffer[mut=False],
    ):
        self = PrimitiveArray[DT](
            DT(),
            length=length,
            nulls=nulls,
            offset=offset,
            bitmap=bitmap,
            buffer=buffer,
        )

    def __init__(out self, data: ArrayData) raises:
        if len(data.buffers) != 1:
            raise Error("PrimitiveArray requires exactly one buffer")
        self = Self(
            dtype=data.dtype.as_type[Self.T]().copy(),
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            buffer=data.buffers[0],
        )

    def __init__[
        DT: NumericType
    ](
        out self: PrimitiveArray[DT],
        var *values: Scalar[DT.native],
        __list_literal__: NoneType,
    ) raises:
        """Constructs a primitive array from a list literal [v1, v2, ...].

        Only valid for numeric types (NumericType). Temporal and decimal types
        require an explicit builder with a dtype instance.

        Args:
            values: The scalar values to populate the array with.
            __list_literal__: Tells Mojo to use this method for list literal syntax.
        """
        var b = PrimitiveBuilder[DT](capacity=len(values))
        for value in values:
            b.unsafe_append(value)
        self = b.finish()

    @always_inline
    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def type(self) -> DynType:
        return self.dtype.copy().to_dyn()

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array.

        Matches PyArrow's Array.slice(offset, length) API.
        """
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            dtype=self.dtype.copy(),
            length=actual_length,
            # the parent's count says nothing about a sub-range; count this
            # window. A parent with no nulls has a child with none, so the
            # common case never touches the bitmap.
            nulls=_sliced_null_count(
                self.nulls,
                self.length,
                self.bitmap,
                self.offset + offset,
                length,
            ),
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
            return PrimitiveScalar[Self.T](None, self.dtype)
        return PrimitiveScalar[Self.T](self.unsafe_get(index), self.dtype)

    def null_count(self) -> Int:
        return self.nulls

    def to_device(self, ctx: DeviceContext) raises -> PrimitiveArray[Self.T]:
        """Upload array data to the GPU."""
        var bm: Optional[Bitmap[]] = None
        if self.bitmap:
            bm = self.bitmap.value().to_device(ctx)
        return PrimitiveArray[Self.T](
            dtype=self.dtype.copy(),
            length=self.length,
            nulls=self.null_count(),
            # the whole values buffer is transferred, so the offset still
            # addresses the same elements; zeroing it silently turned a slice
            # into the parent's first `length` elements
            offset=self.offset,
            bitmap=bm^,
            buffer=self.buffer.to_device(ctx),
        )

    def to_cpu(self, ctx: DeviceContext) raises -> PrimitiveArray[Self.T]:
        """Download array data from the GPU to owned CPU heap buffers."""
        var bm: Optional[Bitmap[]] = None
        if self.bitmap:
            bm = self.bitmap.value().to_cpu(ctx)
        return PrimitiveArray[Self.T](
            dtype=self.dtype.copy(),
            length=self.length,
            nulls=self.null_count(),
            # the whole values buffer is transferred, so the offset still
            # addresses the same elements; zeroing it silently turned a slice
            # into the parent's first `length` elements
            offset=self.offset,
            bitmap=bm^,
            buffer=self.buffer.to_cpu(ctx),
        )

    def __eq__(self, other: Self) -> Bool:
        """Return True if both arrays have the same length, null pattern, and values.

        Fast path (no nulls, offset=0 on both): full buffer SIMD comparison.
        Slow path (nulls or non-zero offset): element-by-element at valid positions.
        """
        if self.length != other.length:
            return False
        if self.null_count() != other.null_count():
            return False
        if self.bitmap.__bool__() != other.bitmap.__bool__():
            return False
        if self.bitmap:
            if not (self.bitmap.value() == other.bitmap.value()):
                return False
        # Compare only the valid length elements (buffer may be over-allocated
        # in filtered output, so full Buffer.__eq__ would read uninitialized bytes).
        for i in range(self.length):
            if self.is_valid(i):
                if self.unsafe_get(i) != other.unsafe_get(i):
                    return False
        return True

    def to_data(self) -> ArrayData:
        """Extract generic array layout for interop."""
        return ArrayData(
            dtype=self.dtype.copy().to_dyn(),
            length=self.length,
            nulls=self.null_count(),
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
# BinaryArray
# ---------------------------------------------------------------------------


@fieldwise_init
struct BinaryLikeArray[T: BinaryLikeType](Array):
    """An immutable Arrow array of variable-length bytes (binary or string).

    The semantic type (binary, large_binary, string, large_string) is carried
    by the type parameter T; T.offset determines the physical offset DType.
    """

    comptime ScalarType = StringScalar

    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var offsets: Buffer[mut=False]
    var values: Buffer[mut=False]

    @staticmethod
    def empty() raises -> Self:
        """A zero-length binary-like array."""
        return Self(
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            offsets=Buffer.alloc_zeroed[Self.T.offset](1).to_immutable(),
            values=Buffer.alloc_zeroed[DType.uint8](0).to_immutable(),
        )

    def __init__(
        out self, var *values: String, __list_literal__: NoneType
    ) raises:
        """Constructs a string array from a list literal ["a", "b", ...].

        Args:
            values: The string values to populate the array with.
            __list_literal__: Tells Mojo to use this method for list literal syntax.
        """
        var b = BinaryLikeBuilder[Self.T](capacity=len(values))
        for value in values:
            b.append(value)
        self = b.finish()

    def __init__(out self, data: ArrayData) raises:
        if len(data.buffers) != 2:
            raise Error("BinaryArray requires exactly two buffers")
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

    def type(self) -> DynType:
        return Self.T().to_dyn()

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array.

        Matches PyArrow's Array.slice(offset, length) API.
        """
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            length=actual_length,
            # the parent's count says nothing about a sub-range; count this
            # window. A parent with no nulls has a child with none, so the
            # common case never touches the bitmap.
            nulls=_sliced_null_count(
                self.nulls,
                self.length,
                self.bitmap,
                self.offset + offset,
                length,
            ),
            offset=self.offset + offset,
            bitmap=self.bitmap,
            offsets=self.offsets,
            values=self.values,
        )

    def write_to[W: Writer](self, mut writer: W):
        var dtype = Self.T().to_dyn()
        if dtype.is_string():
            writer.write("StringArray([")
        elif dtype.is_large_string():
            writer.write("LargeStringArray([")
        elif dtype.is_large_binary():
            writer.write("LargeBinaryArray([")
        else:
            writer.write("BinaryArray([")
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
        var start_offset = self.offsets.unsafe_get[Self.T.offset](offset_idx)
        var end_offset = self.offsets.unsafe_get[Self.T.offset](offset_idx + 1)
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

    def __eq__(self, other: Self) -> Bool:
        """Return True if both arrays have the same length, null pattern, and string values.
        """
        if self.length != other.length:
            return False
        if self.null_count() != other.null_count():
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

    def to_data(self) -> ArrayData:
        """Extract generic array layout for interop."""
        return ArrayData(
            dtype=Self.T().to_dyn(),
            length=self.length,
            nulls=self.null_count(),
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[self.offsets, self.values],
            children=[],
        )


comptime BinaryArray = BinaryLikeArray[BinaryType]
comptime LargeBinaryArray = BinaryLikeArray[LargeBinaryType]
comptime StringArray = BinaryLikeArray[StringType]
comptime LargeStringArray = BinaryLikeArray[LargeStringType]


# ---------------------------------------------------------------------------
# ListArray / LargeListArray
# ---------------------------------------------------------------------------


struct ListLikeArray[T: ListLikeType](Array):
    """An immutable Arrow array of variable-length lists (each element is a sub-array).
    """

    comptime ScalarType = ListScalar

    var dtype: DynType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var offsets: Buffer[mut=False]
    var child: OwnedPointer[DynArray]

    def validity(
        ref self,
    ) -> Optional[BitmapView[origin_of(self.bitmap._value)]]:
        """Validity bitmap view, or None if all values are valid."""
        if not self.bitmap:
            return None
        return self.bitmap.value().view(self.offset, self.length)

    def __init__(
        out self,
        *,
        dtype: DynType,
        length: Int,
        nulls: Int,
        offset: Int,
        bitmap: Optional[Bitmap[mut=False]],
        offsets: Buffer[mut=False],
        var values: DynArray,
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
            values=DynArray.from_data(data.children[0]),
        )

    def values(ref self) -> ref[self.child[]] DynArray:
        """The child array containing the list elements."""
        return self.child[]

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def null_count(self) -> Int:
        return self.nulls

    def type(self) -> DynType:
        return self.dtype.copy()

    def write_to[W: Writer](self, mut writer: W):
        if self.dtype.is_map():
            writer.write("MapArray([")
        elif self.dtype.is_large_list():
            writer.write("LargeListArray([")
        else:
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

    def child_range(self, index: Int) -> Tuple[Int, Int]:
        """The `[start, end)` range in the child values array for element
        `index` — the offsets pair, adjusted for this array's own offset."""
        var start = Int(
            self.offsets.unsafe_get[Self.T.offset](self.offset + index)
        )
        var end = Int(
            self.offsets.unsafe_get[Self.T.offset](self.offset + index + 1)
        )
        return (start, end)

    def unsafe_get(self, index: Int) raises -> DynArray:
        """Return a view of the child array for the list at the given index."""
        var start, end = self.child_range(index)
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
            # the parent's count says nothing about a sub-range; count this
            # window. A parent with no nulls has a child with none, so the
            # common case never touches the bitmap.
            nulls=_sliced_null_count(
                self.nulls,
                self.length,
                self.bitmap,
                self.offset + offset,
                length,
            ),
            offset=self.offset + offset,
            bitmap=self.bitmap,
            offsets=self.offsets,
            values=self.child[].copy(),
        )

    def flatten(self) -> DynArray:
        """Unnest this ListArray, returning the flat child values."""
        return self.child[].copy()

    def to_map(self, keys_sorted: Bool = False) raises -> MapArray:
        """Retag this list of (key, value) entries structs as a `MapArray` — same
        physical layout (offsets, validity, and the entries child are shared),
        only the dtype tag changes: the child struct dtype becomes the map's
        entries field (its key/value field names are preserved). `keys_sorted` is
        caller-supplied (Parquet carries no such flag). The single point where a
        list becomes a map — inverse of `MapArray.to_list`."""
        var map_dtype: DynType = MapType(
            field("entries", self.values().dtype(), nullable=False), keys_sorted
        )
        return MapArray(
            dtype=map_dtype,
            length=self.length,
            nulls=self.null_count(),
            offset=self.offset,
            bitmap=self.bitmap,
            offsets=self.offsets,
            values=self.values().copy(),
        )

    def to_list(self) -> ListArray:
        """Retag this map as a plain list of its entries struct — the inverse of
        `ListArray.to_map`, same shared layout. Lets list-oriented machinery
        (builders, concat) operate on a map without knowing it is one."""
        return ListArray(
            dtype=list_(self.values().dtype()),
            length=self.length,
            nulls=self.null_count(),
            offset=self.offset,
            bitmap=self.bitmap,
            offsets=self.offsets,
            values=self.values().copy(),
        )

    def value_lengths(self) -> Int32Array:
        """Return an array of list lengths for each element."""
        var buf = Buffer.alloc_zeroed[DType.int32](self.length)
        for i in range(self.length):
            var start = self.offsets.unsafe_get[Self.T.offset](self.offset + i)
            var end = self.offsets.unsafe_get[Self.T.offset](
                self.offset + i + 1
            )
            buf.unsafe_set[DType.int32](i, Int32(end - start))
        return Int32Array(
            dtype=Int32Type(),
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
        if self.null_count() != other.null_count():
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

    @staticmethod
    def from_arrays[
        O: IntegerType
    ](
        offsets: PrimitiveArray[O],
        var values: DynArray,
        var mask: Optional[BoolArray] = None,
    ) -> Self:
        """Construct a ListArray from offsets, values, and optional null mask.

        Matches PyArrow's ListArray.from_arrays(offsets, values, mask=None, type=None) API.
        mask uses PyArrow convention: True=null, False=valid. dtype is derived from values.
        """
        var n = offsets.length - 1
        var null_count = 0
        var bitmap: Optional[Bitmap[mut=False]] = None
        if m := mask^:
            var bm = Bitmap[mut=True].alloc_zeroed(n)
            for i in range(n):
                if m.value().values().test(i):
                    null_count += 1
                else:
                    bm.set(i)
            bitmap = bm^.to_immutable(length=n)
        var list_dtype: DynType
        comptime if Self.T.offset == DType.int32:
            list_dtype = list_(values.dtype())
        else:
            list_dtype = large_list_(values.dtype())
        return Self(
            dtype=list_dtype^,
            length=n,
            nulls=null_count,
            offset=0,
            bitmap=bitmap^,
            offsets=offsets.buffer,
            values=values^,
        )

    @staticmethod
    def from_arrays(
        offsets: Int32Array,
        var keys: DynArray,
        var items: DynArray,
        keys_sorted: Bool = False,
        var mask: Optional[BoolArray] = None,
    ) raises -> MapArray:
        """Construct a MapArray from int32 offsets (length n+1), key/item child
        arrays, and an optional null mask (PyArrow convention: True = null).

        Matches PyArrow's ``MapArray.from_arrays(offsets, keys, items)``. The
        entries struct is built non-nullable with a required key, then the
        offsets fold it into a map (`ListArray.from_arrays(...).to_map()`)."""
        var entry_fields = [
            field("key", keys.dtype(), nullable=False),
            field("value", items.dtype(), nullable=True),
        ]
        var entries: DynArray = StructArray.from_arrays(
            [keys^, items^], entry_fields, None
        )
        return ListArray.from_arrays(offsets, entries^, mask^).to_map(
            keys_sorted
        )

    def to_data(self) raises -> ArrayData:
        """Extract generic array layout for interop."""
        return ArrayData(
            dtype=self.dtype.copy(),
            length=self.length,
            nulls=self.null_count(),
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[self.offsets],
            children=[self.values().to_data()],
        )


comptime ListArray = ListLikeArray[ListType]
comptime LargeListArray = ListLikeArray[LargeListType]
comptime MapArray = ListLikeArray[MapType]


# ---------------------------------------------------------------------------
# FixedSizeListArray
# ---------------------------------------------------------------------------


struct FixedSizeListArray(Array):
    """An immutable Arrow array of fixed-size lists (each element is a sub-array of the same length).
    """

    comptime ScalarType = ListScalar

    var dtype: DynType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var child: OwnedPointer[DynArray]

    def validity(
        ref self,
    ) -> Optional[BitmapView[origin_of(self.bitmap._value)]]:
        """Validity bitmap view, or None if all values are valid."""
        if not self.bitmap:
            return None
        return self.bitmap.value().view(self.offset, self.length)

    def __init__(
        out self,
        *,
        dtype: DynType,
        length: Int,
        nulls: Int,
        offset: Int,
        bitmap: Optional[Bitmap[mut=False]],
        var values: DynArray,
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

    def __init__(out self, data: ArrayData) raises:
        if len(data.children) != 1:
            raise Error("FixedSizeListArray requires exactly one child array")
        self = Self(
            dtype=data.dtype.copy(),
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            values=DynArray.from_data(data.children[0]),
        )

    def values(ref self) -> ref[self.child[]] DynArray:
        """The child array containing all list elements (length * list_size elements).
        """
        return self.child[]

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def null_count(self) -> Int:
        return self.nulls

    def type(self) -> DynType:
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

    def unsafe_get(self, index: Int, out array_data: DynArray) raises:
        var list_size = self.dtype.as_fixed_size_list().size
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
            # the parent's count says nothing about a sub-range; count this
            # window. A parent with no nulls has a child with none, so the
            # common case never touches the bitmap.
            nulls=_sliced_null_count(
                self.nulls,
                self.length,
                self.bitmap,
                self.offset + offset,
                length,
            ),
            offset=self.offset + offset,
            bitmap=self.bitmap,
            values=self.child[].copy(),
        )

    def flatten(self) -> DynArray:
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
        var new_child = DynArray.from_data(
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
            nulls=self.null_count(),
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
        if self.null_count() != other.null_count():
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

    @staticmethod
    def from_arrays(
        var values: DynArray,
        list_size: Int,
        var mask: Optional[BoolArray] = None,
    ) -> Self:
        """Construct a FixedSizeListArray from a flat child array and fixed list size.

        Matches PyArrow's FixedSizeListArray.from_arrays(values, type=None, mask=None) API.
        mask uses PyArrow convention: True=null, False=valid. dtype is derived from values.
        """
        var n = values.length() // list_size if list_size > 0 else 0
        var null_count = 0
        var bitmap: Optional[Bitmap[mut=False]] = None
        if m := mask^:
            var bm = Bitmap[mut=True].alloc_zeroed(n)
            for i in range(n):
                if m.value().values().test(i):
                    null_count += 1
                else:
                    bm.set(i)
            bitmap = bm^.to_immutable(length=n)
        return Self(
            dtype=fixed_size_list_(values.dtype(), list_size),
            length=n,
            nulls=null_count,
            offset=0,
            bitmap=bitmap^,
            values=values^,
        )

    def to_data(self) raises -> ArrayData:
        """Extract generic array layout for interop."""
        return ArrayData(
            dtype=self.dtype.copy(),
            length=self.length,
            nulls=self.null_count(),
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[],
            children=[self.values().to_data()],
        )


# ---------------------------------------------------------------------------
# FixedSizeBinaryArray
# ---------------------------------------------------------------------------


@fieldwise_init
struct FixedSizeBinaryArray(Array):
    """An immutable Arrow array of fixed-width binary values.

    Layout: a single contiguous data buffer of `length * byte_width` bytes,
    plus an optional validity bitmap.  Each element occupies exactly
    `byte_width` bytes — no offset buffer.
    """

    comptime ScalarType = FixedSizeBinaryScalar

    var length: Int
    var nulls: Int
    var offset: Int
    var byte_width: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var buffer: Buffer[mut=False]

    def validity(
        ref self,
    ) -> Optional[BitmapView[origin_of(self.bitmap._value)]]:
        """Validity bitmap view, or None if all values are valid."""
        if not self.bitmap:
            return None
        return self.bitmap.value().view(self.offset, self.length)

    def __init__(out self, data: ArrayData) raises:
        if not data.dtype.is_fixed_size_binary():
            raise Error("FixedSizeBinaryArray requires fixed_size_binary dtype")
        if len(data.buffers) != 1:
            raise Error("FixedSizeBinaryArray requires exactly one buffer")
        self = Self(
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            byte_width=data.dtype.as_fixed_size_binary().byte_width,
            bitmap=data.bitmap,
            buffer=data.buffers[0],
        )

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def type(self) -> DynType:
        return FixedSizeBinaryType(self.byte_width).to_dyn()

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            length=actual_length,
            # the parent's count says nothing about a sub-range; count this
            # window. A parent with no nulls has a child with none, so the
            # common case never touches the bitmap.
            nulls=_sliced_null_count(
                self.nulls,
                self.length,
                self.bitmap,
                self.offset + offset,
                length,
            ),
            offset=self.offset + offset,
            byte_width=self.byte_width,
            bitmap=self.bitmap,
            buffer=self.buffer,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("FixedSizeBinaryArray([")
        for i in range(self.length):
            if i > 0:
                writer.write(", ")
            if i >= 10:
                writer.write("...")
                break
            if self.is_valid(i):
                writer.write("<", self.byte_width, " bytes>")
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

    def __getitem__(self, index: Int) raises -> FixedSizeBinaryScalar:
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        if not self.is_valid(index):
            return FixedSizeBinaryScalar.null(self.byte_width)
        var bytes = List[UInt8](capacity=self.byte_width)
        var start = (self.offset + index) * self.byte_width
        for i in range(self.byte_width):
            bytes.append(self.buffer.unsafe_get[DType.uint8](start + i))
        return FixedSizeBinaryScalar(bytes^, self.byte_width)

    def to_data(self) raises -> ArrayData:
        return ArrayData(
            dtype=FixedSizeBinaryType(self.byte_width).to_dyn(),
            length=self.length,
            nulls=self.null_count(),
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[self.buffer],
            children=[],
        )

    def __eq__(self, other: Self) -> Bool:
        if self.byte_width != other.byte_width:
            return False
        if self.length != other.length:
            return False
        if self.null_count() != other.null_count():
            return False
        if self.bitmap.__bool__() != other.bitmap.__bool__():
            return False
        if self.bitmap:
            if not (self.bitmap.value() == other.bitmap.value()):
                return False
        for i in range(self.length):
            if self.is_valid(i):
                var ls = (self.offset + i) * self.byte_width
                var rs = (other.offset + i) * self.byte_width
                for k in range(self.byte_width):
                    var lv = self.buffer.unsafe_get[DType.uint8](ls + k)
                    var rv = other.buffer.unsafe_get[DType.uint8](rs + k)
                    if lv != rv:
                        return False
        return True


comptime Date32Array = PrimitiveArray[Date32Type]
comptime Date64Array = PrimitiveArray[Date64Type]
comptime Time32Array = PrimitiveArray[Time32Type]
comptime Time64Array = PrimitiveArray[Time64Type]
comptime DurationArray = PrimitiveArray[DurationType]
comptime TimestampArray = PrimitiveArray[TimestampType]

comptime YearMonthIntervalArray = PrimitiveArray[YearMonthIntervalType]
comptime DayTimeIntervalArray = PrimitiveArray[DayTimeIntervalType]
comptime MonthDayNanoIntervalArray = PrimitiveArray[MonthDayNanoIntervalType]

comptime Decimal32Array = PrimitiveArray[Decimal32Type]
comptime Decimal64Array = PrimitiveArray[Decimal64Type]
comptime Decimal128Array = PrimitiveArray[Decimal128Type]
comptime Decimal256Array = PrimitiveArray[Decimal256Type]


# ---------------------------------------------------------------------------
# StructArray
# ---------------------------------------------------------------------------


@fieldwise_init
struct StructArray(Array):
    """An immutable Arrow array of structs (each element is a collection of named fields).
    """

    comptime ScalarType = StructScalar

    var dtype: DynType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var children: List[DynArray]

    def validity(
        ref self,
    ) -> Optional[BitmapView[origin_of(self.bitmap._value)]]:
        """Validity bitmap view, or None if all values are valid."""
        if not self.bitmap:
            return None
        return self.bitmap.value().view(self.offset, self.length)

    def __init__(out self, data: ArrayData) raises:
        var children = List[DynArray]()
        for c in data.children:
            children.append(DynArray.from_data(c))
        self = Self(
            dtype=data.dtype.copy(),
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            children=children^,
        )

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String.write(self)

    def null_count(self) -> Int:
        return self.nulls

    def type(self) -> DynType:
        return self.dtype.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("StructArray({")
        if len(self.children) > 0:
            ref st = self.dtype.as_struct()
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
        var fields = self.dtype.as_struct().fields.copy()
        for idx, ref field in enumerate(fields):
            if field.name == name:
                return idx

        raise Error(t"Field {name} does not exist in this StructArray.")

    def unsafe_get(
        self, name: StringSlice
    ) raises -> ref[self.children[0]] DynArray:
        """Access the field with the given name in the struct."""
        return self.children[self._index_for_field_name(name)]

    def field(self, index: Int) raises -> DynArray:
        """Access a child array by field index.

        Matches PyArrow's StructArray.field(index) API.
        """
        if index < 0 or index >= len(self.children):
            raise Error(
                t"field index {index} out of bounds for"
                t" {len(self.children)} fields"
            )
        return self.children[index].copy()

    def field(self, name: StringSlice) raises -> DynArray:
        """Access a child array by field name.

        Matches PyArrow's StructArray.field(name) API.
        """
        return self.children[self._index_for_field_name(name)].copy()

    def __getitem__(self, index: Int) raises -> StructScalar:
        if index < 0 or index >= self.length:
            raise Error(t"index {index} out of bounds for length {self.length}")
        if not self.is_valid(index):
            return StructScalar.null(self.dtype.copy())
        # Pre-allocate to avoid reallocation: when List[DynScalar] grows it
        # moves existing elements, and Mojo's Variant __moveinit__ resets the
        # discriminant to 0 (the first type), corrupting already-stored scalars.
        var fields = List[DynScalar](capacity=len(self.children))
        for i in range(len(self.children)):
            fields.append(self.children[i][self.offset + index])
        return StructScalar(
            dtype=self.dtype.copy(), value=fields^, is_valid=True
        )

    def select(self, indices: List[Int]) raises -> Self:
        """Return a new StructArray with only the fields at the given indices.

        O(1) ref-count bumps per selected column — no data copied.
        Matches RecordBatch.select(indices) API.
        """
        var fields = List[Field]()
        var children = List[DynArray]()
        ref st = self.dtype.as_struct()
        for idx in indices:
            fields.append(st.fields[idx].copy())
            children.append(self.children[idx].copy())
        return Self(
            dtype=struct_(fields^),
            length=self.length,
            nulls=self.null_count(),
            offset=self.offset,
            bitmap=self.bitmap,
            children=children^,
        )

    def flatten(self) -> List[DynArray]:
        """Return one DynArray per field.

        Matches PyArrow's StructArray.flatten() API.
        """
        return self.children.copy()

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            dtype=self.dtype.copy(),
            length=actual_length,
            # the parent's count says nothing about a sub-range; count this
            # window. A parent with no nulls has a child with none, so the
            # common case never touches the bitmap.
            nulls=_sliced_null_count(
                self.nulls,
                self.length,
                self.bitmap,
                self.offset + offset,
                length,
            ),
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
        if self.null_count() != other.null_count():
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

    @staticmethod
    def from_arrays(
        var children: List[DynArray],
        fields: List[Field],
        var mask: Optional[BoolArray] = None,
    ) -> Self:
        """Construct a StructArray from child arrays and field descriptors.

        Matches PyArrow's StructArray.from_arrays(arrays, names=None, fields=None, mask=None) API.
        mask uses PyArrow convention: True=null, False=valid.
        """
        var n = children[0].length() if len(children) > 0 else 0
        var null_count = 0
        var bitmap: Optional[Bitmap[mut=False]] = None
        if m := mask^:
            var bm = Bitmap[mut=True].alloc_zeroed(n)
            for i in range(n):
                if m.value().values().test(i):
                    null_count += 1
                else:
                    bm.set(i)
            bitmap = bm^.to_immutable(length=n)
        return Self(
            dtype=struct_(fields.copy()),
            length=n,
            nulls=null_count,
            offset=0,
            bitmap=bitmap^,
            children=children^,
        )

    def to_data(self) raises -> ArrayData:
        """Extract generic array layout for interop."""
        var children = List[ArrayData]()
        for c in self.children:
            children.append(c.to_data())
        return ArrayData(
            dtype=self.dtype.copy(),
            length=self.length,
            nulls=self.null_count(),
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=[],
            children=children^,
        )


# ---------------------------------------------------------------------------
# DictionaryArray
# ---------------------------------------------------------------------------


struct DictionaryArray(Array):
    """An Arrow dictionary-encoded array.

    Stores integer indices into a separate dictionary (values) array.
    Equivalent to PyArrow's ``pyarrow.DictionaryArray``.

    Memory layout matches the Arrow spec: the raw indices buffer is stored in
    ``buffers[0]``; the dictionary values array is stored in ``children[0]``
    when round-tripping through ``ArrayData`` / C Data Interface.
    """

    comptime ScalarType = DictionaryScalar

    var _dtype: DynType
    var _length: Int
    var _nulls: Int
    var _offset: Int  # extra logical offset into _indices (on top of _indices' own offset)
    var _indices: OwnedPointer[DynArray]
    var _values: OwnedPointer[DynArray]

    def __init__(
        out self,
        *,
        dtype: DynType,
        length: Int,
        nulls: Int,
        offset: Int,
        var indices: DynArray,
        var values: DynArray,
    ):
        self._dtype = dtype.copy()
        self._length = length
        self._nulls = nulls
        self._offset = offset
        self._indices = OwnedPointer(indices^)
        self._values = OwnedPointer(values^)

    def __init__(out self, *, copy: Self):
        self._dtype = copy._dtype.copy()
        self._length = copy._length
        self._nulls = copy._nulls
        self._offset = copy._offset
        self._indices = OwnedPointer(copy._indices[].copy())
        self._values = OwnedPointer(copy._values[].copy())

    def __init__(out self, data: ArrayData) raises:
        ref dt = data.dtype.as_dictionary()
        var indices_data = ArrayData(
            dtype=dt.index_type().copy(),
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            buffers=data.buffers.copy(),
            children=[],
        )
        self._dtype = data.dtype.copy()
        self._length = data.length
        self._nulls = data.nulls
        self._offset = 0  # offset is now embedded in the reconstructed _indices
        self._indices = OwnedPointer(DynArray.from_data(indices_data))
        self._values = OwnedPointer(DynArray.from_data(data.children[0]))

    @staticmethod
    def from_arrays(
        var indices: DynArray, var values: DynArray, ordered: Bool = False
    ) raises -> Self:
        """Construct from existing indices and dictionary arrays.

        Matches PyArrow's ``DictionaryArray.from_arrays(indices, dictionary)`` API.
        """
        if not indices.dtype().is_integer():
            raise Error(
                "DictionaryArray: indices must have an integer dtype, got: ",
                indices.dtype(),
            )
        var n = indices.length()
        return Self(
            dtype=dictionary(indices.dtype(), values.dtype(), ordered),
            length=n,
            nulls=indices.null_count(),
            offset=0,
            indices=indices^,
            values=values^,
        )

    def __len__(self) -> Int:
        return self._length

    def __str__(self) -> String:
        return String.write(self)

    def type(self) -> DynType:
        return self._dtype.copy()

    def null_count(self) -> Int:
        return self._nulls

    def is_valid(self, index: Int) -> Bool:
        return self._indices[].is_valid(self._offset + index)

    def indices(self) raises -> DynArray:
        """The logical index array (the dictionary's `_offset` applied). Matches
        PyArrow's DictionaryArray.indices."""
        return self._indices[].slice(self._offset, self._length)

    def dictionary(self) -> DynArray:
        """Return the dictionary (values) array. Matches PyArrow's DictionaryArray.dictionary.
        """
        return self._values[].copy()

    def __getitem__(self, index: Int) raises -> DictionaryScalar:
        if index < 0 or index >= self._length:
            raise Error(
                t"index {index} out of bounds for length {self._length}"
            )
        var adj = self._offset + index
        if not self._indices[].is_valid(adj):
            return DictionaryScalar.null(self._dtype.copy())
        var idx_scalar = self._indices[][adj]
        ref index_type = self._dtype.as_dictionary().index_type()
        var dict_idx: Int
        if index_type.is_int8():
            dict_idx = Int(idx_scalar.as_int8().value())
        elif index_type.is_int16():
            dict_idx = Int(idx_scalar.as_int16().value())
        elif index_type.is_int32():
            dict_idx = Int(idx_scalar.as_int32().value())
        elif index_type.is_int64():
            dict_idx = Int(idx_scalar.as_int64().value())
        elif index_type.is_uint8():
            dict_idx = Int(idx_scalar.as_uint8().value())
        elif index_type.is_uint16():
            dict_idx = Int(idx_scalar.as_uint16().value())
        elif index_type.is_uint32():
            dict_idx = Int(idx_scalar.as_uint32().value())
        elif index_type.is_uint64():
            dict_idx = Int(idx_scalar.as_uint64().value())
        else:
            raise Error("DictionaryArray: unexpected index type: ", index_type)
        var decoded = self._values[][dict_idx]
        return DictionaryScalar(
            dtype=self._dtype.copy(), index=dict_idx, decoded=decoded^
        )

    def slice(self, offset: Int, length: Int) -> Self:
        """Zero-copy slice: adjusts logical offset, shares indices and values.
        """
        return Self(
            dtype=self._dtype.copy(),
            length=length,
            nulls=self._nulls,
            offset=self._offset + offset,
            indices=self._indices[].copy(),
            values=self._values[].copy(),
        )

    def to_dyn(deinit self) -> DynArray:
        return DynArray(self^)

    def to_data(self) raises -> ArrayData:
        """Extract generic ArrayData layout for C Data Interface interop.

        The dictionary values array is stored as ``children[0]``.
        """
        var indices_data = self._indices[].to_data()
        var values_data = self._values[].to_data()
        return ArrayData(
            dtype=self._dtype.copy(),
            length=self._length,
            nulls=self._nulls,
            offset=self._offset + indices_data.offset,
            bitmap=indices_data.bitmap,
            buffers=indices_data.buffers.copy(),
            children=[values_data^],
        )

    def __eq__(self, other: Self) -> Bool:
        return (
            self._dtype == other._dtype
            and self._length == other._length
            and self._indices[] == other._indices[]
            and self._values[] == other._values[]
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "DictionaryArray(values=",
            self._values[],
            ", indices=",
            self._indices[],
            ")",
        )

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# ChunkedArray
# ---------------------------------------------------------------------------


struct ChunkedArray(Copyable, Movable, Writable):
    """An array-like composed from a (possibly empty) collection of pyarrow.Arrays.

    [Reference](https://arrow.apache.org/docs/python/generated/pyarrow.ChunkedArray.html#pyarrow-chunkedarray).
    """

    var dtype: DynType
    var length: Int
    var chunks: List[DynArray]

    def _compute_length(mut self) -> None:
        """Update the length of the array from the length of its chunks."""
        var total_length = 0
        for chunk in self.chunks:
            total_length += chunk.length()
        self.length = total_length

    def __init__(out self, dtype: DynType, var chunks: List[DynArray]):
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

    def chunk(self, index: Int) -> ref[self.chunks[index]] DynArray:
        """Returns the chunk at the given index.

        Args:
          index: The desired index.

        Returns:
          A reference to the chunk at the given index.
        """
        return self.chunks[index]

    def combine_chunks(var self) raises -> DynArray:
        """Combines all chunks into a single array."""
        from .kernels.concat import concat
        from .builders import DynBuilder

        if len(self.chunks) == 0:
            # An empty ArrayData with no buffers is not a valid array for most
            # dtypes (a primitive needs its data buffer, a string its offsets,
            # etc.), so build a properly-structured empty array of the dtype.
            var builder = DynBuilder(self.dtype)
            return builder.finish()
        if len(self.chunks) == 1:
            return self.chunks[0].copy()
        return concat(self.chunks)


# ---------------------------------------------------------------------------
# DynArray — Variant-based type-erased array handle
# ---------------------------------------------------------------------------


struct DynArray(
    Array,
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
        NullArray,
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
        Date32Array,
        Date64Array,
        Time32Array,
        Time64Array,
        DurationArray,
        TimestampArray,
        YearMonthIntervalArray,
        DayTimeIntervalArray,
        MonthDayNanoIntervalArray,
        Decimal32Array,
        Decimal64Array,
        Decimal128Array,
        Decimal256Array,
        BinaryArray,
        LargeBinaryArray,
        StringArray,
        LargeStringArray,
        ListArray,
        LargeListArray,
        FixedSizeListArray,
        FixedSizeBinaryArray,
        StructArray,
        MapArray,
        DictionaryArray,
    ]

    var _v: Self.VariantType

    # --- construction ---

    @implicit
    def __init__[T: Array](out self, var array: T):
        self._v = Self.VariantType(array^)

    def __init__(out self, data: ArrayData) raises:
        """`Array`'s constructor-from-`ArrayData`, delegating to `from_data`.

        The static factory stays the implementation and the primary spelling;
        this exists so the trait is satisfied without moving 30-odd call sites.
        No ambiguity with the `@implicit [T: Array]` constructor above:
        `ArrayData` is only `Copyable, Movable` and does not conform to `Array`.
        """
        self = DynArray.from_data(data)

    def __init__(out self, *, copy: Self):
        self._v = Self.VariantType(copy=copy._v)

    # Explicit (empty) destructor so this type is ImplicitlyDeletable despite
    # the `StructArray -> List[DynArray] -> DynArray` reference cycle; the
    # variant field is still destroyed automatically after the body runs.
    def __del__(deinit self):
        pass

    def __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema, CArrowArray

        # Fast path: marrow arrays are now exposed as a single DynArray Python type.
        try:
            self = py.downcast_value_ptr[DynArray]()[].copy()
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
                t" '{py.__class__.__name__}' to DynArray",
            )

    # --- dispatch-based methods ---

    def length(self) -> Int:
        @parameter
        def f[T: Array](a: T) -> Int:
            return len(a)

        return variant_dispatch[Array, func=f](self._v)

    comptime ScalarType = DynScalar
    """`Array`'s companion-scalar member. `DynScalar` conforms to `ArrowScalar`,
    which is what lets this conform to `Array` in turn — the two erasures line
    up rather than each needing its own escape hatch."""

    def type(self) -> DynType:
        """`Array`'s spelling of `dtype()`.

        Both names exist because both are load-bearing: `Array` requires
        `type()`, while ~200 call sites and the `Builder` trait say `dtype()`.
        Renaming either would be a large diff for no behaviour, so this is a
        one-line forward and `dtype()` remains the one with the implementation.
        """
        return self.dtype()

    def dtype(self) -> DynType:
        @parameter
        def f[T: Array](a: T) -> DynType:
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

    def is_null(self, index: Int) -> Bool:
        return not self.is_valid(index)

    def slice(self, offset: Int, length: Int = -1) raises -> DynArray:
        """Returns a zero-copy slice starting at offset with the given length.

        Matches PyArrow's Array.slice(offset, length) API.
        """

        @parameter
        def f[T: Array](a: T) raises -> DynArray:
            var actual_length = length if length >= 0 else len(a) - offset
            return a.slice(offset, actual_length)

        return variant_dispatch_raises[Array, func=f](self._v)

    def view(self, var dtype: DynType) raises -> DynArray:
        """Reinterpret this array's buffers under a same-layout `dtype`.

        Zero-copy: only the logical type is replaced, the buffers are shared.
        Mirrors `pyarrow.Array.view(target_type)` / arrow-rs
        `ArrayData::with_data_type`. Used to relabel a column produced in its
        physical storage type (an integer-backed temporal build, say) to the
        Arrow type it logically is.

        The caller is responsible for the layouts actually matching — this does
        not validate, exactly as PyArrow's does not.
        """
        var d = self.to_data()
        d.dtype = dtype^
        return DynArray.from_data(d^)

    def to_data(self) raises -> ArrayData:
        """Extract a generic ArrayData layout for interop (C Data Interface, etc.).

        Not intended for hot paths — prefer typed downcast methods.
        """

        @parameter
        def f[T: Array](a: T) raises -> ArrayData:
            return a.to_data()

        return variant_dispatch_raises[Array, func=f](self._v)

    def to_device(self, ctx: DeviceContext) raises -> DynArray:
        """Upload this array to the GPU device."""

        @parameter
        def f[T: Array](a: T) raises -> DynArray:
            return a.to_device(ctx)

        return variant_dispatch_raises[Array, func=f](self._v)

    def to_cpu(self, ctx: DeviceContext) raises -> DynArray:
        """Download this array from the GPU device to CPU memory."""

        @parameter
        def f[T: Array](a: T) raises -> DynArray:
            return a.to_cpu(ctx)

        return variant_dispatch_raises[Array, func=f](self._v)

    def to_dyn(deinit self) -> DynArray:
        """Returns this array as DynArray, transferring ownership."""
        return self^

    def write_to[W: Writer](self, mut writer: W):
        @parameter
        def f[T: Array](a: T):
            a.write_to(writer)

        variant_dispatch[Array, func=f](self._v)

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def __eq__(self, other: DynArray) -> Bool:
        return self._v == other._v

    def to_python_object(var self) raises -> PythonObject:
        """Convert to a Python Array object (type-erased)."""
        return PythonObject(alloc=self^)

    def __len__(self) -> Int:
        return self.length()

    def __getitem__(self, index: Int) raises -> DynScalar:
        """Return the element at index as a type-erased DynScalar."""
        if index < 0 or index >= self.length():
            raise Error(
                t"index {index} out of bounds for length {self.length()}"
            )

        @parameter
        def f[T: Array](a: T) raises -> DynScalar:
            return a[index].to_dyn()

        return variant_dispatch_raises[Array, func=f](self._v)

    # --- typed downcasts (zero-cost reference borrows) ---

    def as_type[T: Array](ref self) -> ref[self._v[T]] T:
        """This array as the concrete `T` it holds — a borrow, no copy.

        The named accessors below are one-liners over it; generic code
        (a fused node reading a typed slot) needs the parameterized form."""
        debug_assert(
            self._v.isa[T](), "as_type: wrong type, holds ", self.dtype()
        )
        return self._v[T]

    def as_primitive[
        T: PrimitiveType
    ](ref self) -> ref[self._v[PrimitiveArray[T]]] PrimitiveArray[T]:
        return self.as_type[PrimitiveArray[T]]()

    def as_binary_like[
        T: BinaryLikeType
    ](ref self) -> ref[self._v[BinaryLikeArray[T]]] BinaryLikeArray[T]:
        return self.as_type[BinaryLikeArray[T]]()

    def as_null(ref self) -> ref[self._v[NullArray]] NullArray:
        return self.as_type[NullArray]()

    def as_bool(ref self) -> ref[self._v[BoolArray]] BoolArray:
        return self.as_type[BoolArray]()

    def as_int8(ref self) -> ref[self._v[Int8Array]] Int8Array:
        return self.as_type[Int8Array]()

    def as_int16(ref self) -> ref[self._v[Int16Array]] Int16Array:
        return self.as_type[Int16Array]()

    def as_int32(ref self) -> ref[self._v[Int32Array]] Int32Array:
        return self.as_type[Int32Array]()

    def as_int64(ref self) -> ref[self._v[Int64Array]] Int64Array:
        return self.as_type[Int64Array]()

    def as_uint8(ref self) -> ref[self._v[UInt8Array]] UInt8Array:
        return self.as_type[UInt8Array]()

    def as_uint16(ref self) -> ref[self._v[UInt16Array]] UInt16Array:
        return self.as_type[UInt16Array]()

    def as_uint32(ref self) -> ref[self._v[UInt32Array]] UInt32Array:
        return self.as_type[UInt32Array]()

    def as_uint64(ref self) -> ref[self._v[UInt64Array]] UInt64Array:
        return self.as_type[UInt64Array]()

    def as_float16(ref self) -> ref[self._v[Float16Array]] Float16Array:
        return self.as_type[Float16Array]()

    def as_float32(ref self) -> ref[self._v[Float32Array]] Float32Array:
        return self.as_type[Float32Array]()

    def as_float64(ref self) -> ref[self._v[Float64Array]] Float64Array:
        return self.as_type[Float64Array]()

    def as_string(ref self) -> ref[self._v[StringArray]] StringArray:
        return self.as_type[StringArray]()

    def as_binary(ref self) -> ref[self._v[BinaryArray]] BinaryArray:
        return self.as_type[BinaryArray]()

    def as_large_string(
        ref self,
    ) -> ref[self._v[LargeStringArray]] LargeStringArray:
        return self.as_type[LargeStringArray]()

    def as_large_binary(
        ref self,
    ) -> ref[self._v[LargeBinaryArray]] LargeBinaryArray:
        return self.as_type[LargeBinaryArray]()

    def as_list(ref self) -> ref[self._v[ListArray]] ListArray:
        return self.as_type[ListArray]()

    def as_list_like[
        T: ListLikeType
    ](ref self) -> ref[self._v[ListLikeArray[T]]] ListLikeArray[T]:
        return self.as_type[ListLikeArray[T]]()

    def as_large_list(ref self) -> ref[self._v[LargeListArray]] LargeListArray:
        return self.as_type[LargeListArray]()

    def as_fixed_size_list(
        ref self,
    ) -> ref[self._v[FixedSizeListArray]] FixedSizeListArray:
        return self.as_type[FixedSizeListArray]()

    def as_fixed_size_binary(
        ref self,
    ) -> ref[self._v[FixedSizeBinaryArray]] FixedSizeBinaryArray:
        return self.as_type[FixedSizeBinaryArray]()

    def as_date32(ref self) -> ref[self._v[Date32Array]] Date32Array:
        return self.as_type[Date32Array]()

    def as_date64(ref self) -> ref[self._v[Date64Array]] Date64Array:
        return self.as_type[Date64Array]()

    def as_time32(ref self) -> ref[self._v[Time32Array]] Time32Array:
        return self.as_type[Time32Array]()

    def as_time64(ref self) -> ref[self._v[Time64Array]] Time64Array:
        return self.as_type[Time64Array]()

    def as_year_month_interval(
        ref self,
    ) -> ref[self._v[YearMonthIntervalArray]] YearMonthIntervalArray:
        return self.as_type[YearMonthIntervalArray]()

    def as_day_time_interval(
        ref self,
    ) -> ref[self._v[DayTimeIntervalArray]] DayTimeIntervalArray:
        return self.as_type[DayTimeIntervalArray]()

    def as_month_day_nano_interval(
        ref self,
    ) -> ref[self._v[MonthDayNanoIntervalArray]] MonthDayNanoIntervalArray:
        return self.as_type[MonthDayNanoIntervalArray]()

    def as_duration(ref self) -> ref[self._v[DurationArray]] DurationArray:
        return self.as_type[DurationArray]()

    def as_timestamp(ref self) -> ref[self._v[TimestampArray]] TimestampArray:
        return self.as_type[TimestampArray]()

    def as_decimal32(ref self) -> ref[self._v[Decimal32Array]] Decimal32Array:
        return self.as_type[Decimal32Array]()

    def as_decimal64(ref self) -> ref[self._v[Decimal64Array]] Decimal64Array:
        return self.as_type[Decimal64Array]()

    def as_decimal128(
        ref self,
    ) -> ref[self._v[Decimal128Array]] Decimal128Array:
        return self.as_type[Decimal128Array]()

    def as_decimal256(
        ref self,
    ) -> ref[self._v[Decimal256Array]] Decimal256Array:
        return self.as_type[Decimal256Array]()

    def as_struct(ref self) -> ref[self._v[StructArray]] StructArray:
        return self.as_type[StructArray]()

    def as_map(ref self) -> ref[self._v[MapArray]] MapArray:
        return self.as_type[MapArray]()

    def as_dictionary(
        ref self,
    ) -> ref[self._v[DictionaryArray]] DictionaryArray:
        return self.as_type[DictionaryArray]()

    # --- factory from generic layout ---

    @staticmethod
    def from_data(data: ArrayData) raises -> DynArray:
        """Construct an DynArray from a generic ArrayData by dispatching on dtype.

        Used by the C Data Interface and other interop paths where a flat
        7-field layout is the natural representation.
        """
        var dt = data.dtype.copy()
        if dt.is_null():
            return NullArray(data)
        elif dt == bool_:
            return BoolArray(data)
        elif dt == int8:
            return Int8Array(data)
        elif dt == int16:
            return Int16Array(data)
        elif dt == int32:
            return Int32Array(data)
        elif dt == int64:
            return Int64Array(data)
        elif dt == uint8:
            return UInt8Array(data)
        elif dt == uint16:
            return UInt16Array(data)
        elif dt == uint32:
            return UInt32Array(data)
        elif dt == uint64:
            return UInt64Array(data)
        elif dt == float16:
            return Float16Array(data)
        elif dt == float32:
            return Float32Array(data)
        elif dt == float64:
            return Float64Array(data)
        if dt.is_string():
            return StringArray(data)
        elif dt.is_binary():
            return BinaryArray(data)
        elif dt.is_large_string():
            return LargeStringArray(data)
        elif dt.is_large_binary():
            return LargeBinaryArray(data)
        elif dt.is_list():
            return ListArray(data)
        elif dt.is_large_list():
            return LargeListArray(data)
        elif dt.is_fixed_size_list():
            return FixedSizeListArray(data)
        elif dt.is_fixed_size_binary():
            return FixedSizeBinaryArray(data)
        elif dt.is_date32():
            return Date32Array(data)
        elif dt.is_date64():
            return Date64Array(data)
        elif dt.is_time32():
            return Time32Array(data)
        elif dt.is_time64():
            return Time64Array(data)
        elif dt.is_timestamp():
            return TimestampArray(data)
        elif dt.is_duration():
            return DurationArray(data)
        elif dt.is_year_month_interval():
            return YearMonthIntervalArray(data)
        elif dt.is_day_time_interval():
            return DayTimeIntervalArray(data)
        elif dt.is_month_day_nano_interval():
            return MonthDayNanoIntervalArray(data)
        elif dt.is_decimal32():
            return Decimal32Array(data)
        elif dt.is_decimal64():
            return Decimal64Array(data)
        elif dt.is_decimal128():
            return Decimal128Array(data)
        elif dt.is_decimal256():
            return Decimal256Array(data)
        elif dt.is_struct():
            return StructArray(data)
        elif dt.is_map():
            return MapArray(data)
        elif dt.is_dictionary():
            return DictionaryArray(data)
        else:
            raise Error("from_data: unsupported dtype")
