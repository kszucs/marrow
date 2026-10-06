# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

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


from std.math import ceildiv
from std.memory import OwnedPointer
from std.bit import byte_swap

from max.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from std.utils import Variant

from .errors import (
    IndexError,
    InternalError,
    InvalidError,
    KeyError,
    NotImplementedError,
    TypeError,
    debug_assert_lazy,
)
from .buffers import Buffer, Bitmap
from .views import BufferView, BitmapView
from std.builtin.rebind import downcast
from std.os import abort
from .dtypes import (
    ArrayLayout,
    DynType,
    BinaryLikeType,
    StringLikeType,
    BinaryViewLikeType,
    StringViewLikeType,
    BinaryViewType,
    StringViewType,
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
from .builders import (
    DynBuilder,
    PrimitiveBuilder,
    BinaryLikeBuilder,
    BinaryViewLikeBuilder,
)
from .scalars import (
    DynScalar,
    NullScalar,
    BoolScalar,
    FixedSizeBinaryScalar,
    PrimitiveScalar,
    BinaryLikeScalar,
    BinaryViewLikeScalar,
    ListScalar,
    StructScalar,
    DictionaryScalar,
    ArrowScalar,
)


trait Array(
    Copyable,
    Deinitable,
    Equatable,
    Movable,
    Sized,
    Writable,
):
    """Common interface for all typed Arrow arrays.

    All concrete array types (PrimitiveArray, BinaryArray, ListArray,
    FixedSizeListArray, StructArray) implement this trait.  DynArray is
    the type-erased handle that wraps any Array-conforming type.

    Equality (`Equatable`) is **structural**, not logical: two arrays are
    equal when they have the same dtype, the same length, the same null
    pattern, and the same window -- including `offset` -- over equal
    underlying data, all the way down through any children. It is
    deliberately not a contents/value comparison: `a.slice(1, 2)` and a
    freshly built array holding the same two values are not equal, because
    they do not share a layout. `PrimitiveArray`, `BoolArray`,
    `BinaryLikeArray`, `FixedSizeBinaryArray`, `ListLikeArray`,
    `FixedSizeListArray`, `StructArray` and `ArrayData` all follow this rule,
    and `DynArray.__eq__` routes through it via `to_data()`. Element-wise,
    value-level comparison -- where `a.slice(1, 2) == b` can be True by
    contents alone -- is `EqKernel`'s job, not this trait's.

    `DictionaryArray` is the one deliberate exception: its typed `__eq__`
    decodes both sides and compares values, so that two encodings of the same
    column against differently ordered dictionaries compare equal. That
    exception does not reach `DynArray`, whose `__eq__` always goes through
    `to_data()` regardless of the concrete type underneath -- see
    `DynArray.__eq__` and `DictionaryArray.__eq__` for what that costs.
    """

    comptime ScalarType: ArrowScalar

    def __init__(out self, data: ArrayData) raises:
        ...

    def type(self) -> DynType:
        ...

    def null_count(self) -> Int:
        """How many elements are null.

        Always known in O(1): a stored count, recounted when an array is
        sliced, never resolved lazily. A null-free array need not carry a
        validity bitmap at all.
        """
        ...

    def is_valid(self, index: Int) -> Bool:
        ...

    def is_null(self, index: Int) -> Bool:
        return not self.is_valid(index)

    def to_dyn(deinit self) -> DynArray:
        return DynArray(self^)

    def to_device(self, ctx: DeviceContext) raises -> Self:
        raise NotImplementedError(
            "to_device: not supported for this array type"
        )

    def to_cpu(self, ctx: DeviceContext) raises -> Self:
        raise NotImplementedError("to_cpu: not supported for this array type")

    def to_data(self) raises -> ArrayData:
        ...

    def slice(self, offset: Int, length: Int) -> Self:
        """A zero-copy slice.

        Non-raising: every typed implementation is total. It was `raises` for as
        long as `DynArray` conformed to this trait, because the erased
        implementation dispatches over a variant and an uncovered member falls
        through -- one box's failure mode widening the contract for all nine
        typed arrays, at a recorded cost of +13,428 bytes of `__text`.
        `DynArray.slice` keeps its own `raises`; it is no longer implementing
        this.
        """
        ...

    def __getitem__(self, index: Int) raises -> Self.ScalarType:
        ...


# ---------------------------------------------------------------------------
# ArrayData — generic flat layout, produced on demand by to_data()
# ---------------------------------------------------------------------------


struct ArrayData(Copyable, Equatable, Movable):
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

    def __init__(
        out self,
        var dtype: DynType,
        length: Int,
        nulls: Int,
        offset: Int,
        var bitmap: Optional[Bitmap[mut=False]],
        var buffers: List[Buffer[mut=False]],
        var children: List[ArrayData],
    ):
        """Build a layout, checking it against what its dtype describes.

        Written out rather than `@fieldwise_init` so the buffer-count invariant
        cannot be bypassed: all ~39 construction sites go through here.

        `debug_assert` rather than `raise` because `to_data()` is non-raising on
        several array types, and making it raise would cascade a `raises` through
        the whole array API for a check about *our* correctness. It is live under
        `-D ASSERT=all`, which is how every test in this repo runs.

        Foreign data additionally gets the raising `validate()` -- see the C Data
        Interface importer. There a wrong count reads past the end of somebody
        else's allocation, so it must be checked in release builds too.
        """

        # A view layout has a variable number of data buffers after its views,
        # so its count is a minimum; every other layout's is exact. A closure,
        # so release builds never compute the layout.
        def fits() {imm} -> Bool:
            var layout = dtype.layout()
            var n = len(buffers)
            return n == layout.num_buffers() or (
                layout.kind == ArrayLayout.VIEW and n > layout.num_buffers()
            )

        debug_assert(fits, "ArrayData: buffer count does not match dtype")
        self.dtype = dtype^
        self.length = length
        self.nulls = nulls
        self.offset = offset
        self.bitmap = bitmap^
        self.buffers = buffers^
        self.children = children^

    def __eq__(self, other: Self) -> Bool:
        """Structural equality of the layout's fields.

        **Low-level on purpose.** This compares what the layout *is* — dtype,
        length, null count, offset, validity, buffers, children — not what it
        decodes to. Two layouts holding the same values at different offsets,
        or against differently ordered dictionaries, are not equal here.
        Logical, value-level comparison is `EqKernel`'s job.

        Hand-written rather than derived, for a reason that is not style:
        `children` is a `List[ArrayData]`, so the derived `__eq__` recurses
        through `List.__eq__`, which is `always_inline` — and the compiler
        rejects a recursive call to an always-inline function. A self-recursive
        type cannot have a derived comparison.

        No element loop, and that is what makes it compile at all. Comparing a
        nested array element by element materialises a `DynArray` per element,
        which makes `DynArray.__eq__` and the nested arrays' `__eq__` mutually
        recursive at instantiation; the elaborator never resolves that, parking
        at 0% CPU with no diagnostic, and it kept every case in
        `marrow/tests/test_arrays.mojo` from compiling. Measured: `_v == _v`,
        `_dispatch` narrowing and `@no_inline` all still deadlock — only a path
        that never names a typed `__eq__` works.
        """
        if self.dtype != other.dtype:
            return False
        if self.length != other.length:
            return False
        if self.nulls != other.nulls:
            return False
        if self.offset != other.offset:
            return False
        if Bool(self.bitmap) != Bool(other.bitmap):
            return False
        if self.bitmap:
            if not (self.bitmap.value() == other.bitmap.value()):
                return False
        if len(self.buffers) != len(other.buffers):
            return False
        for i in range(len(self.buffers)):
            if not (self.buffers[i] == other.buffers[i]):
                return False
        if len(self.children) != len(other.children):
            return False
        for i in range(len(self.children)):
            if not (self.children[i] == other.children[i]):
                return False
        return True

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def validate(self, full: Bool = False) raises InvalidError:
        """Raise unless this array is safe to read: `validate_node` on every
        node of it, as pyarrow's `Array.validate`."""
        self.validate_node(full)
        for child in self.children:
            child.validate(full)

    def validate_node(self, full: Bool = False) raises InvalidError:
        """Raise unless this node is safe to read: it has the buffers and
        children its dtype's `layout()` describes, each long enough for
        `offset + length` values, and any nulls come with a validity bitmap.
        With `full`, also the checks that read every value: offsets never
        decrease, and every valid dictionary index names a value.

        This node only: a reader validates each child as it builds it, so
        `validate` would repeat the work below. Without `full` it is O(1). A
        view array's views are checked by its typed constructor.
        """
        var layout = self.dtype.layout()
        var want = layout.num_buffers()
        var n = len(self.buffers)
        if n < want or (n > want and layout.kind != ArrayLayout.VIEW):
            raise InvalidError(
                t"ArrayData: {self.dtype} owns {want} data buffer(s), got {n}"
            )
        if len(self.children) != layout.num_children:
            raise InvalidError(
                t"ArrayData: {self.dtype} needs {layout.num_children}"
                t" children, got {len(self.children)}"
            )
        if (
            self.length < 0
            or self.offset < 0
            or self.offset >= Int.MAX - self.length
        ):
            raise InvalidError(
                t"ArrayData: length {self.length} at offset {self.offset}"
            )
        if layout.width < 0:
            raise InvalidError(t"ArrayData: {self.dtype} has a negative width")
        # Sizes are compared by division, so a length read from a file cannot
        # overflow a product into a pass.
        var end = self.offset + self.length
        if layout.kind != ArrayLayout.NULL:
            if self.nulls < 0 or self.nulls > self.length:
                raise InvalidError(
                    t"ArrayData: {self.nulls} nulls in {self.length} values"
                )
            if self.bitmap:
                if self.bitmap.value().byte_count() < ceildiv(end, 8):
                    raise InvalidError(
                        t"ArrayData: the validity bitmap is shorter than"
                        t" {end} bits"
                    )
            elif self.nulls > 0:
                raise InvalidError(
                    t"ArrayData: {self.nulls} nulls but no validity bitmap"
                )

        if layout.kind == ArrayLayout.BITMAP:
            self._expect(ceildiv(end, 8), 1)
        elif (
            layout.kind == ArrayLayout.BINARY or layout.kind == ArrayLayout.LIST
        ):
            # An empty array may omit its offsets.
            if self.length > 0 or len(self.buffers[0]) > 0:
                self._expect(end + 1, layout.width)
                var limit = (
                    len(self.buffers[1]) if layout.kind
                    == ArrayLayout.BINARY else self.children[0].length
                )
                var first = self._offset(0, layout.width)
                var last = self._offset(self.length, layout.width)
                if first < 0 or last < first or last > limit:
                    raise InvalidError(
                        t"ArrayData: {self.dtype} offsets run from {first} to"
                        t" {last}, outside 0..{limit}"
                    )
        elif layout.kind == ArrayLayout.NESTED:
            for i in range(len(self.children)):
                var held = self.children[i].length
                if layout.width > 0 and end > held // layout.width:
                    raise InvalidError(
                        t"ArrayData: {self.dtype} child {i} is shorter than its"
                        t" parent: {held} values, fewer than {end} x"
                        t" {layout.width}"
                    )
        elif layout.kind != ArrayLayout.NULL:
            self._expect(end, layout.width)
        if full:
            self._validate_values()

    def _validate_values(self) raises InvalidError:
        """The checks of `validate_node(full=True)` that read every value."""
        var layout = self.dtype.layout()
        if layout.kind == ArrayLayout.BINARY or layout.kind == ArrayLayout.LIST:
            var bad = (
                self._first_decrease[DType.int32]() if layout.width
                == 4 else self._first_decrease[DType.int64]()
            )
            if bad >= 0:
                raise InvalidError(
                    t"ArrayData: {self.dtype} offset {bad} decreases"
                )
        elif layout.kind == ArrayLayout.DICTIONARY:
            var n_values = self.children[0].length

            # The index type decides signedness, which the layout does not.
            def first_bad[T: IntegerType](d: T) raises {imm} -> Int:
                var indices = self.buffers[0].view[T.native](self.offset)
                var valid = self.validity()
                for i in range(self.length):
                    var index = Int(indices[i])
                    if (index < 0 or index >= n_values) and (
                        not valid or valid.value().test(i)
                    ):
                        return i
                return -1

            var bad: Int
            try:
                bad = (
                    self.dtype.as_dictionary()
                    .index_type()
                    .dispatch_integer(first_bad)
                )
            except:
                raise InvalidError(
                    t"ArrayData: {self.dtype} has no integer indices"
                )
            if bad >= 0:
                raise InvalidError(
                    t"ArrayData: dictionary index at {bad} is outside"
                    t" 0..{n_values}"
                )

    def _expect(self, count: Int, width: Int) raises InvalidError:
        """Raise unless buffer 0 holds `count` values of `width` bytes."""
        if width > 0 and count > len(self.buffers[0]) // width:
            raise InvalidError(
                t"ArrayData: {self.dtype} needs {count} values of {width}"
                t" bytes, got {len(self.buffers[0])} bytes"
            )

    def _offset(self, i: Int, width: Int) -> Int:
        """Offset `i`, read from buffer 0's `width`-byte offsets."""
        if width == 4:
            return Int(self.buffers[0].view[DType.int32](self.offset)[i])
        return Int(self.buffers[0].view[DType.int64](self.offset)[i])

    def _first_decrease[T: DType](self) -> Int:
        """The first position whose offset is below the one before, or -1."""
        var offsets = self.buffers[0].view[T](self.offset)
        for i in range(self.length):
            if offsets[i + 1] < offsets[i]:
                return i + 1
        return -1

    # Explicit (empty) destructor so this self-referential struct
    # (`children: List[ArrayData]`) is Deinitable; fields are still
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

        An array with no bitmap and every row null — the `null` dtype, which
        the spec gives no validity buffer — answers an all-clear bitmap, not
        `None`: the missing bitmap there means "all null", not "all valid".
        """
        var v = self.validity()
        if v:
            return v.value().to_owned()
        elif self.length > 0 and self.nulls == self.length:
            return Bitmap.alloc_zeroed(self.length).to_immutable()
        else:
            return None

    def __deinit__(deinit self):
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
        return String(self)

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
            raise InvalidError("BoolArray requires exactly one buffer")
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
        return String(self)

    def type(self) -> DynType:
        return bool_

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array.

        Matches PyArrow's Array.slice(offset, length) API.
        """
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            length=actual_length,
            nulls=0 if self.nulls
            == 0 else self.bitmap.value()
            .view(self.offset + offset, actual_length)
            .unset_count(),
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
        if self.length != other.length:
            return False
        # A nonzero null count implies a bitmap, so once the counts agree the
        # two views can be compared directly. Note the converse does not hold:
        # a slice that excludes every null still carries its parent's bitmap.
        if self.null_count() != other.null_count():
            return False
        if self.null_count() != 0:
            var sv = self.validity()
            var ov = other.validity()
            if not sv or not ov:
                return False
            if not (sv.value() == ov.value()):
                return False
        # Structural, matching `ArrayData.__eq__`: equal arrays share the same
        # layout, not merely the same logical contents. `self[i]` below goes
        # through `is_valid`/`values()`, both offset-applied, which is what
        # made this loop offset-invariant -- `a.slice(1, 2) == a.slice(2, 2)`
        # could answer True whenever the two windows held the same bits.
        if self.offset != other.offset:
            return False
        for i in range(self.length):
            if self.is_valid(i) and self[i] != other[i]:
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
            raise InvalidError("PrimitiveArray requires exactly one buffer")
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
        return String(self)

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
            nulls=0 if self.nulls
            == 0 else self.bitmap.value()
            .view(self.offset + offset, actual_length)
            .unset_count(),
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
            raise IndexError(
                t"index {index} out of bounds for length {self.length}"
            )
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
        """Return True if both arrays have the same dtype, length, null
        pattern, offset, and values.

        Element-by-element at valid positions, not `Buffer.__eq__` on the
        backing buffer: a buffer may be over-allocated (e.g. filtered output),
        so a whole-buffer comparison would read uninitialized bytes past the
        logical end.
        """
        if self.length != other.length:
            return False
        # A nonzero null count implies a bitmap, so once the counts agree the
        # two views can be compared directly. Note the converse does not hold:
        # a slice that excludes every null still carries its parent's bitmap.
        if self.null_count() != other.null_count():
            return False
        if self.null_count() != 0:
            var sv = self.validity()
            var ov = other.validity()
            if not sv or not ov:
                return False
            if not (sv.value() == ov.value()):
                return False
        # Structural, matching `ArrayData.__eq__`: equal arrays share the same
        # layout, not merely the same logical contents. `unsafe_get` below is
        # offset-applied, which is what made this loop offset-invariant --
        # `a.slice(1, 2) == a.slice(2, 2)` could answer True whenever the two
        # windows held the same values.
        if self.offset != other.offset:
            return False
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


trait BytesArray(Array):
    """An array whose elements are byte strings, in either layout: offsets
    (`BinaryLikeArray`) or views (`BinaryViewLikeArray`).

    The one thing a kernel that only *reads* elements needs, so hashing,
    equality, sorting and the string kernels are written once over both
    layouts. A kernel that exploits a layout -- offsets arithmetic, gathering
    views -- keeps a typed overload instead. The text/bytes distinction stays
    on the dtype, where the kernels' runtime guards already check it.
    """

    def unsafe_get(ref self, index: UInt) -> StringSlice[origin_of(self)]:
        """The element at `index`, borrowed; no bounds or validity check."""
        ...

    @staticmethod
    def from_values(values: List[Optional[String]]) raises -> Self:
        """An array of `values`, `None` as null, in this layout."""
        ...

    def validity(ref self) -> Optional[BitmapView[origin_of(self)]]:
        """Validity bitmap view, offset-applied, or None if all values are
        valid -- what `Bitmap.intersect_views` combines."""
        ...

    def sort_key(self, index: Int) -> UInt64:
        """The element's first eight bytes, zero-padded, big-endian: two keys
        compare as the elements' bytes do, and equal keys decide nothing. A
        sort reads it once per row and settles most comparisons on it."""
        ...


@fieldwise_init
struct BinaryLikeArray[T: BinaryLikeType](BytesArray):
    """An immutable Arrow array of variable-length bytes (binary or string).

    The semantic type (binary, large_binary, string, large_string) is carried
    by the type parameter T; T.offset determines the physical offset DType.
    """

    comptime ScalarType = BinaryLikeScalar[Self.T]

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

    @staticmethod
    def from_values(values: List[Optional[String]]) raises -> Self:
        var b = BinaryLikeBuilder[Self.T](capacity=len(values))
        for v in values:
            if v:
                b.append(v.value())
            else:
                b.append_null()
        return b.finish()

    def __init__(out self, data: ArrayData) raises:
        if len(data.buffers) != 2:
            raise InvalidError("BinaryArray requires exactly two buffers")
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
        return String(self)

    def null_count(self) -> Int:
        return self.nulls

    def validity(ref self) -> Optional[BitmapView[origin_of(self)]]:
        """Validity bitmap view, or None if all values are valid.

        Offset-applied, like every sibling's. This type was the one array
        without it, which is why the string predicate kernels reached for the
        raw `.bitmap` and shifted their nulls on sliced inputs (Q2.3).
        """
        if not self.bitmap:
            return None
        return rebind[BitmapView[origin_of(self)]](
            self.bitmap.value().view(self.offset, self.length)
        )

    def sort_key(self, index: Int) -> UInt64:
        var bytes = self.unsafe_get(UInt(index)).as_bytes()
        var key = UInt64(0)
        for k in range(8):
            key <<= 8
            if k < len(bytes):
                key |= UInt64(bytes[k])
        return key

    def type(self) -> DynType:
        return Self.T().to_dyn()

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array.

        Matches PyArrow's Array.slice(offset, length) API.
        """
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            length=actual_length,
            nulls=0 if self.nulls
            == 0 else self.bitmap.value()
            .view(self.offset + offset, actual_length)
            .unset_count(),
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

    def unsafe_get(ref self, index: UInt) -> StringSlice[origin_of(self)]:
        """Return a StringSlice for the element at the given index without bounds checking.
        """
        var offset_idx = Int(index) + self.offset
        var start_offset = self.offsets.unsafe_get[Self.T.offset](offset_idx)
        var end_offset = self.offsets.unsafe_get[Self.T.offset](offset_idx + 1)
        var length = end_offset - start_offset
        return rebind[StringSlice[origin_of(self)]](
            self.values.slice(Int(start_offset), Int(length)).to_string_slice()
        )

    def __getitem__(self, index: Int) raises -> BinaryLikeScalar[Self.T]:
        """Return the scalar for the element at the given index.

        Raises:
            If the index is out of bounds.
        """
        if index < 0 or index >= self.length:
            raise IndexError(
                t"index {index} out of bounds for length {self.length}"
            )
        if not self.is_valid(index):
            return BinaryLikeScalar[Self.T].null()
        return BinaryLikeScalar[Self.T](String(self.unsafe_get(UInt(index))))

    def __eq__(self, other: Self) -> Bool:
        """Return True if both arrays have the same length, null pattern, and string values.
        """
        if self.length != other.length:
            return False
        # A nonzero null count implies a bitmap, so once the counts agree the
        # two views can be compared directly. Note the converse does not hold:
        # a slice that excludes every null still carries its parent's bitmap.
        if self.null_count() != other.null_count():
            return False
        if self.null_count() != 0:
            var sv = self.validity()
            var ov = other.validity()
            if not sv or not ov:
                return False
            if not (sv.value() == ov.value()):
                return False
        # Structural, matching `ArrayData.__eq__`: equal arrays share the same
        # layout, not merely the same logical contents. `unsafe_get` below is
        # offset-applied, which is what made this loop offset-invariant --
        # `a.slice(1, 2) == a.slice(2, 2)` could answer True whenever the two
        # windows held the same strings.
        if self.offset != other.offset:
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
# BinaryViewArray
# ---------------------------------------------------------------------------


@fieldwise_init
struct BinaryViewLikeArray[T: BinaryViewLikeType](BytesArray):
    """An immutable Arrow array of variable-length bytes in the *view* layout
    (binary_view or string_view).

    Each element is a 16-byte view in `views`, read as four `int32` words:

        word 0      length
        words 1-3   length <= 12: the bytes themselves, zero-padded
                    length  > 12: 4-byte prefix, buffer index, byte offset

    so a value of up to 12 bytes needs no data buffer at all, and a longer one
    points into any of the variadic `buffers`. Slicing moves only `offset`
    over the views; the data buffers are shared as they are, which is what lets
    `take` and `filter` gather views without copying a byte.

    The words are read as `int32`s rather than one 16-byte load: a buffer
    imported over the C Data Interface is only guaranteed 8-byte alignment.
    """

    comptime ScalarType = BinaryViewLikeScalar[Self.T]

    comptime VIEW_SIZE = 16
    """Bytes per view."""
    comptime INLINE_SIZE = 12
    """The longest value stored inside its view."""

    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var views: Buffer[mut=False]
    var buffers: List[Buffer[mut=False]]

    @staticmethod
    def empty() -> Self:
        """A zero-length view array."""
        return Self(
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            views=Buffer.alloc_zeroed[DType.uint8](0).to_immutable(),
            buffers=[],
        )

    def __init__(
        out self, var *values: String, __list_literal__: NoneType
    ) raises:
        """Constructs a view array from a list literal ["a", "b", ...].

        Args:
            values: The string values to populate the array with.
            __list_literal__: Tells Mojo to use this method for list literal syntax.
        """
        var b = BinaryViewLikeBuilder[Self.T](capacity=len(values))
        for value in values:
            b.append(value)
        self = b.finish()

    @staticmethod
    def from_values(values: List[Optional[String]]) raises -> Self:
        var b = BinaryViewLikeBuilder[Self.T](capacity=len(values))
        for v in values:
            if v:
                b.append(v.value())
            else:
                b.append_null()
        return b.finish()

    def __init__(out self, data: ArrayData) raises:
        """Adopt a layout: `buffers[0]` holds the views, the rest are the
        variadic data buffers. Every valid view is checked against the buffer
        it points into — see `validate` — and a null view that points outside
        them is zeroed, see `_zero_wild_null_views`."""
        self = Self(unsafe_from_data=data)
        self.validate()
        self._zero_wild_null_views()

    def __init__(out self, *, unsafe_from_data: ArrayData) raises:
        """Adopt a layout without checking a view: for one marrow built
        itself, such as `to_data()`'s, where the scan `validate` makes would
        only re-prove what the builder guaranteed. Raises only when there is
        no views buffer at all."""
        ref data = unsafe_from_data
        if len(data.buffers) < 1:
            raise InvalidError("BinaryViewArray requires a views buffer")
        var buffers = List[Buffer[mut=False]](capacity=len(data.buffers) - 1)
        for i in range(1, len(data.buffers)):
            buffers.append(data.buffers[i])
        self = Self(
            length=data.length,
            nulls=data.nulls,
            offset=data.offset,
            bitmap=data.bitmap,
            views=data.buffers[0],
            buffers=buffers^,
        )

    def _wild(self, index: Int) -> Bool:
        """Whether the view at `index` points outside the data buffers."""
        var n = self.view_length(index)
        if n < 0:
            return True
        elif n <= Self.INLINE_SIZE:
            return False
        else:
            var b = Int(self.view_buffer_index(index))
            var start = Int(self.view_offset(index))
            return (
                b < 0
                or b >= len(self.buffers)
                or start < 0
                or start + n > len(self.buffers[b])
            )

    def _zero_wild_null_views(mut self) raises:
        """Zero every null view that points outside the data buffers.

        The format leaves a null's view undefined and `validate` exempts it,
        as Arrow C++ and arrow-rs do -- but the fused string loops read every
        row, nulls included, and let validity sort it out afterwards. A
        foreign null view claiming bytes that do not exist would be read out
        of bounds there. Zeroing it makes it an empty inline value.

        The views buffer is copied only when such a view exists; marrow's own
        builder zeroes null views, so arrays it produced never pay for it.
        """
        var wild = False
        if self.nulls > 0:
            for i in range(self.length):
                if not self.is_valid(i) and self._wild(i):
                    wild = True
                    break
        if wild:
            var views = Buffer.alloc_uninit[DType.uint8](len(self.views))
            views.view[DType.uint8]().copy_from(
                self.views.view[DType.uint8](), len(self.views)
            )
            for i in range(self.length):
                if not self.is_valid(i) and self._wild(i):
                    var word = (self.offset + i) * 4
                    for k in range(4):
                        views.unsafe_set[DType.int32](word + k, 0)
            self.views = views^.to_immutable()

    def validate(self) raises:
        """Raise unless every view in the window fits what it points at.

        A view names a buffer and a byte range inside it, so a malformed one
        is a read past the end of somebody else's allocation. This is the
        check Arrow C++'s `ValidateFull` and arrow-rs's `ArrayData::validate`
        both make for this layout; nulls are exempt, as they are there.
        """
        if len(self.views) < (self.offset + self.length) * Self.VIEW_SIZE:
            raise InvalidError("BinaryViewArray: views buffer is too short")
        var extents = self.buffer_extents()
        for b in range(len(self.buffers)):
            if extents[b] > len(self.buffers[b]):
                raise InvalidError(
                    t"BinaryViewArray: a view overruns data buffer {b}"
                )

    def is_sparse(self) -> Bool:
        """Whether the data buffers hold more than twice the bytes the valid
        views in this window reach -- a slice, or a filter, take or decode
        result, keeping most of its source alive.

        The rule `BinaryViewLikeBuilder.extend` decides by, as DataFusion's
        batch coalescer does: adopt a dense array's buffers, copy a sparse
        one's values."""
        var held = 0
        for b in self.buffers:
            held += len(b)
        var reached = 0
        for i in range(self.length):
            if self.is_valid(i):
                var n = self.view_length(i)
                if n > Self.INLINE_SIZE:
                    reached += n
        return reached * 2 < held

    def compact(self) raises -> Self:
        """The same values in buffers that are not `is_sparse()`: this array,
        its buffers shared, when they already are not; otherwise the values
        copied into data buffers of their own, holding only the bytes the
        views reach.

        What an array becomes wherever it outlives the source it was cut from
        -- a query result, an IPC body -- so a filter or slice result does not
        keep its source's pages alive."""
        if self.is_sparse():
            # `extend` copies a sparse array's values rather than adopting
            # its buffers -- the one place that loop is written.
            var b = BinaryViewLikeBuilder[Self.T](self.length)
            b.extend(self)
            return b.finish()
        else:
            return self.copy()

    def buffer_extents(self) raises -> List[Int]:
        """How far into each data buffer the valid views in this window reach
        -- the byte length a consumer needs of it.

        Raises if a view has a negative length or offset, or names a buffer
        that does not exist. `validate` compares the extents with the
        buffers; the C Data exporter reports them as the variadic sizes.
        """
        var extents = List[Int](length=len(self.buffers), fill=0)
        for i in range(self.length):
            if self.is_valid(i):
                var n = self.view_length(i)
                if n < 0:
                    raise InvalidError(
                        t"BinaryViewArray: negative length at {i}"
                    )
                if n > Self.INLINE_SIZE:
                    var b = Int(self.view_buffer_index(i))
                    var start = Int(self.view_offset(i))
                    if b < 0 or b >= len(self.buffers) or start < 0:
                        raise InvalidError(
                            t"BinaryViewArray: view {i} points outside its"
                            t" {len(self.buffers)} data buffer(s)"
                        )
                    extents[b] = max(extents[b], start + n)
        return extents^

    def __len__(self) -> Int:
        """Return the number of elements in the array."""
        return self.length

    def __str__(self) -> String:
        return String(self)

    def null_count(self) -> Int:
        return self.nulls

    def validity(ref self) -> Optional[BitmapView[origin_of(self)]]:
        """Validity bitmap view, or None if all values are valid."""
        if not self.bitmap:
            return None
        return rebind[BitmapView[origin_of(self)]](
            self.bitmap.value().view(self.offset, self.length)
        )

    def type(self) -> DynType:
        return Self.T().to_dyn()

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array.

        Matches PyArrow's Array.slice(offset, length) API.
        """
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            length=actual_length,
            nulls=0 if self.nulls
            == 0 else self.bitmap.value()
            .view(self.offset + offset, actual_length)
            .unset_count(),
            offset=self.offset + offset,
            bitmap=self.bitmap,
            views=self.views,
            buffers=self.buffers.copy(),
        )

    def write_to[W: Writer](self, mut writer: W):
        comptime if conforms_to(Self.T, StringViewLikeType):
            writer.write("StringViewArray([")
        else:
            writer.write("BinaryViewArray([")
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

    @always_inline
    def _word(self, index: Int, k: Int) -> Int32:
        """Word `k` (0-3) of the view at `index`, offset applied."""
        return self.views.unsafe_get[DType.int32]((self.offset + index) * 4 + k)

    @always_inline
    def view_length(self, index: Int) -> Int:
        """Byte length of the element at `index`, read from its view alone."""
        return Int(self._word(index, 0))

    @always_inline
    def view_prefix(self, index: Int) -> UInt32:
        """The first four bytes of the element at `index`, zero-padded, as the
        view stores them — inline or out of line alike, which is what makes it
        a comparison fast path."""
        return UInt32(self._word(index, 1))

    def sort_key(self, index: Int) -> UInt64:
        # The first four bytes are in the view for every value; the next four
        # are too for an inline one, and at the value's offset+4 otherwise.
        var n = self.view_length(index)
        var key = UInt64(0)
        if n > 0:
            var high = UInt64(byte_swap(UInt32(self._word(index, 1))))
            var low = UInt64(0)
            if n <= Self.INLINE_SIZE:
                low = UInt64(byte_swap(UInt32(self._word(index, 2))))
            else:
                ref data = self.buffers[Int(self.view_buffer_index(index))]
                var at = Int(self.view_offset(index)) + 4
                for k in range(4):
                    low = (low << 8) | UInt64(
                        data.unsafe_get[DType.uint8](at + k)
                    )
            key = (high << 32) | low
            # Bytes past a short value are masked rather than trusted to be
            # zero: a foreign producer need not pad an inline view.
            if n < 8:
                key &= ~((UInt64(1) << UInt64(8 * (8 - n))) - 1)
        return key

    @always_inline
    def view_buffer_index(self, index: Int) -> Int32:
        """Which data buffer an out-of-line element's bytes live in.
        Meaningless for an element of `INLINE_SIZE` bytes or fewer."""
        return self._word(index, 2)

    @always_inline
    def view_offset(self, index: Int) -> Int32:
        """Where in its data buffer an out-of-line element's bytes start.
        Meaningless for an element of `INLINE_SIZE` bytes or fewer."""
        return self._word(index, 3)

    def unsafe_get(ref self, index: UInt) -> StringSlice[origin_of(self)]:
        """Return a StringSlice for the element at the given index without
        bounds checking.

        The bytes live either inside the view or in one of the data buffers;
        the two borrows have different origins, both inside `self`, so each is
        rebound to `origin_of(self)`.
        """
        # The whole view in one load: length, prefix, buffer, offset.
        var j = Int(index) + self.offset
        var view = self.views.view[DType.int32]().load[4](j * 4)
        var n = Int(view[0])
        if n <= Self.INLINE_SIZE:
            return rebind[StringSlice[origin_of(self)]](
                self.views.slice(j * Self.VIEW_SIZE + 4, n).to_string_slice()
            )
        else:
            return rebind[StringSlice[origin_of(self)]](
                self.buffers.unsafe_get(Int(view[2]))
                .slice(Int(view[3]), n)
                .to_string_slice()
            )

    def __getitem__(self, index: Int) raises -> BinaryViewLikeScalar[Self.T]:
        """Return the scalar for the element at the given index.

        Raises:
            If the index is out of bounds.
        """
        if index < 0 or index >= self.length:
            raise IndexError(
                t"index {index} out of bounds for length {self.length}"
            )
        if not self.is_valid(index):
            return BinaryViewLikeScalar[Self.T].null()
        return BinaryViewLikeScalar[Self.T](
            String(self.unsafe_get(UInt(index)))
        )

    def __eq__(self, other: Self) -> Bool:
        """Structural equality: same length, null pattern, window and values.

        Compared by value rather than by view, because two builders can lay
        the same strings out across different buffers. Length and prefix are
        compared first, from the views alone -- the prefix only once it is
        four real bytes, since a foreign producer need not zero the padding
        of a shorter inline value.
        """
        if self.length != other.length:
            return False
        if self.null_count() != other.null_count():
            return False
        if self.null_count() != 0:
            var sv = self.validity()
            var ov = other.validity()
            if not sv or not ov:
                return False
            if not (sv.value() == ov.value()):
                return False
        if self.offset != other.offset:
            return False
        for i in range(self.length):
            if self.is_valid(i):
                if self.view_length(i) != other.view_length(i):
                    return False
                if self.view_length(i) >= 4 and self.view_prefix(
                    i
                ) != other.view_prefix(i):
                    return False
                if self.unsafe_get(UInt(i)) != other.unsafe_get(UInt(i)):
                    return False
        return True

    def to_data(self) -> ArrayData:
        """Extract generic array layout for interop: the views, then every
        data buffer."""
        var buffers = List[Buffer[mut=False]](capacity=1 + len(self.buffers))
        buffers.append(self.views)
        for b in self.buffers:
            buffers.append(b)
        return ArrayData(
            dtype=Self.T().to_dyn(),
            length=self.length,
            nulls=self.null_count(),
            offset=self.offset,
            bitmap=self.bitmap,
            buffers=buffers^,
            children=[],
        )


comptime BinaryViewArray = BinaryViewLikeArray[BinaryViewType]
comptime StringViewArray = BinaryViewLikeArray[StringViewType]


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
            raise InvalidError("ListArray requires exactly one buffer")
        if len(data.children) != 1:
            raise InvalidError("ListArray requires exactly one child array")
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

    def child_slice(self) -> DynArray:
        """Just the child elements this array's own range covers --
        `[offsets[offset], offsets[offset + length])`.

        `values()` hands back the *whole* child: slicing a list moves its own
        offset and leaves the child untouched, so anything that walks leaf
        elements positionally rather than through `child_range` has to narrow
        it first.
        """
        var start = Int(self.offsets.unsafe_get[Self.T.offset](self.offset))
        var end = Int(
            self.offsets.unsafe_get[Self.T.offset](self.offset + self.length)
        )
        return self.child[].slice(start, end - start)

    def __len__(self) -> Int:
        return self.length

    def __str__(self) -> String:
        return String(self)

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
            raise IndexError(
                t"index {index} out of bounds for length {self.length}"
            )
        return ListScalar(
            dtype=self.dtype,
            value=self.unsafe_get(index),
            is_valid=self.is_valid(index),
        )

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            dtype=self.dtype.copy(),
            length=actual_length,
            nulls=0 if self.nulls
            == 0 else self.bitmap.value()
            .view(self.offset + offset, actual_length)
            .unset_count(),
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
        # A nonzero null count implies a bitmap, so once the counts agree the
        # two views can be compared directly. Note the converse does not hold:
        # a slice that excludes every null still carries its parent's bitmap.
        if self.null_count() != other.null_count():
            return False
        if self.null_count() != 0:
            var sv = self.validity()
            var ov = other.validity()
            if not sv or not ov:
                return False
            if not (sv.value() == ov.value()):
                return False
        # Fields, not elements -- materialising a `DynArray` per element is
        # what made this method and `DynArray.__eq__` mutually recursive and
        # deadlocked the compiler.
        #
        # The comparison is structural, matching `ArrayData.__eq__`: equal
        # arrays have the same layout, not merely the same logical contents.
        # So `offset` and the offsets buffer are fields like any other, and
        # skipping them is what let `[[1, 2], [3, 4]]` and `[[1, 2, 3], [4]]`
        # compare equal -- same dtype, same length, same null count, same
        # child, different partition of it.
        if self.offset != other.offset:
            return False
        for k in range(self.length + 1):
            if self.offsets.unsafe_get[Self.T.offset](
                self.offset + k
            ) != other.offsets.unsafe_get[Self.T.offset](other.offset + k):
                return False
        return self.values() == other.values()

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
        var m = mask^
        if m:
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
        var entry_fields: List[Field] = [
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
            raise InvalidError(
                "FixedSizeListArray requires exactly one child array"
            )
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
        return String(self)

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
            raise IndexError(
                t"index {index} out of bounds for length {self.length}"
            )
        return ListScalar(
            dtype=self.dtype,
            value=self.unsafe_get(index),
            is_valid=self.is_valid(index),
        )

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            dtype=self.dtype.copy(),
            length=actual_length,
            nulls=0 if self.nulls
            == 0 else self.bitmap.value()
            .view(self.offset + offset, actual_length)
            .unset_count(),
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
        # A nonzero null count implies a bitmap, so once the counts agree the
        # two views can be compared directly. Note the converse does not hold:
        # a slice that excludes every null still carries its parent's bitmap.
        if self.null_count() != other.null_count():
            return False
        if self.null_count() != 0:
            var sv = self.validity()
            var ov = other.validity()
            if not sv or not ov:
                return False
            if not (sv.value() == ov.value()):
                return False
        # Fields, not elements -- materialising a `DynArray` per element is
        # what made this method and `DynArray.__eq__` mutually recursive and
        # deadlocked the compiler.
        #
        # Structural, matching `ArrayData.__eq__`. There is no offsets buffer
        # here, so `offset` alone distinguishes two windows of one child:
        # without it `fsl.slice(0, 1)` and `fsl.slice(1, 1)` compared equal,
        # since slicing moves the offset and leaves the child untouched.
        if self.offset != other.offset:
            return False
        return self.values() == other.values()

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
        var m = mask^
        if m:
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
            raise TypeError(
                "FixedSizeBinaryArray requires fixed_size_binary dtype"
            )
        if len(data.buffers) != 1:
            raise InvalidError(
                "FixedSizeBinaryArray requires exactly one buffer"
            )
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
        return String(self)

    def type(self) -> DynType:
        return FixedSizeBinaryType(self.byte_width).to_dyn()

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            length=actual_length,
            nulls=0 if self.nulls
            == 0 else self.bitmap.value()
            .view(self.offset + offset, actual_length)
            .unset_count(),
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
            raise IndexError(
                t"index {index} out of bounds for length {self.length}"
            )
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
        # A nonzero null count implies a bitmap, so once the counts agree the
        # two views can be compared directly. Note the converse does not hold:
        # a slice that excludes every null still carries its parent's bitmap.
        if self.null_count() != other.null_count():
            return False
        if self.null_count() != 0:
            var sv = self.validity()
            var ov = other.validity()
            if not sv or not ov:
                return False
            if not (sv.value() == ov.value()):
                return False
        # Structural, matching `ArrayData.__eq__`: equal arrays share the same
        # layout, not merely the same logical contents. The byte offsets below
        # are each array's own -- `self.offset` / `other.offset` -- which is
        # what made this loop offset-invariant -- `a.slice(1, 2) ==
        # a.slice(2, 2)` could answer True whenever the two windows held the
        # same bytes.
        if self.offset != other.offset:
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
        return String(self)

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

        raise KeyError(t"Field {name} does not exist in this StructArray.")

    def unsafe_get(
        self, name: StringSlice
    ) raises -> ref[self.children[0]] DynArray:
        """Access the field with the given name in the struct."""
        return self.children[self._index_for_field_name(name)]

    def field(self, index: Int) raises -> DynArray:
        """Access a child array by field index, **with this struct's slice
        applied**.

        Matches PyArrow's StructArray.field(index) API, which also carries the
        parent's offset and length down to the child.

        The slice is not optional bookkeeping. `slice()` is zero-copy: it moves
        `self.offset`/`self.length` and shares `children` untouched, so a child
        read raw is the *parent's* full column, not the slice's. Returning it
        that way was wrong for every consumer of a sliced struct, and the two
        expression lanes failed differently on it:

        - the runtime lane materialises the child and `Datum.to_array` catches
          the length mismatch -- `expected a column of 4 rows, got 7` from
          `limit(4).filter(...)`, which is how this was found;
        - the comptime lane reads elements `[0, len(batch))` of the child, and
          the child's own offset is 0, so it silently read the **wrong window**
          whenever this struct's offset was non-zero -- `limit(n, offset=k)`,
          or any morsel where `LimitOperator` skipped rows. With offset 0 it
          was right by coincidence.

        `kernels/sort.mojo` reads its keys the same way and had the same
        exposure. Slicing is O(1) ref-count bumps, and on an unsliced struct
        (offset 0, full length) it is a no-op, so nothing pays for the fix.
        """
        if index < 0 or index >= len(self.children):
            raise IndexError(
                t"field index {index} out of bounds for {len(self.children)} "
                t"fields"
            )
        return self.children[index].slice(self.offset, self.length)

    def field(self, name: StringSlice) raises -> DynArray:
        """Access a child array by field name, with this struct's slice applied.

        Matches PyArrow's StructArray.field(name) API. See the index overload
        for why the slice has to travel with it.
        """
        return self.children[self._index_for_field_name(name)].slice(
            self.offset, self.length
        )

    def __getitem__(self, index: Int) raises -> StructScalar:
        if index < 0 or index >= self.length:
            raise IndexError(
                t"index {index} out of bounds for length {self.length}"
            )
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

    def flatten(self) raises -> List[DynArray]:
        """One `DynArray` per field, each narrowed to this array's window.

        Matches PyArrow's `StructArray.flatten()`, which applies the parent's
        offset — the raw `children` of a sliced struct (a morsel a `Limit`
        cut) start at its parent's first row, not its own. Like `field`, the
        parent's validity is not merged in.
        """
        var out = List[DynArray](capacity=len(self.children))
        for i in range(len(self.children)):
            out.append(self.field(i))
        return out^

    def slice(self, offset: Int = 0, length: Int = -1) -> Self:
        """Zero-copy slice of this array."""
        var actual_length = length if length >= 0 else self.length - offset
        return Self(
            dtype=self.dtype.copy(),
            length=actual_length,
            nulls=0 if self.nulls
            == 0 else self.bitmap.value()
            .view(self.offset + offset, actual_length)
            .unset_count(),
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
        # A nonzero null count implies a bitmap, so once the counts agree the
        # two views can be compared directly. Note the converse does not hold:
        # a slice that excludes every null still carries its parent's bitmap.
        if self.null_count() != other.null_count():
            return False
        if self.null_count() != 0:
            var sv = self.validity()
            var ov = other.validity()
            if not sv or not ov:
                return False
            if not (sv.value() == ov.value()):
                return False
        if len(self.children) != len(other.children):
            return False
        # Fields, not elements, and structural -- matching `ArrayData.__eq__`.
        # Slicing a struct moves `offset` and leaves the children whole, so
        # without comparing `offset` two disjoint windows of one array --
        # `sa.slice(0, 2)` and `sa.slice(2, 2)` -- compared equal.
        if self.offset != other.offset:
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
        var m = mask^
        if m:
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
            raise TypeError(
                t"DictionaryArray: indices must have an integer dtype, got: "
                t"{indices.dtype()}"
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
        return String(self)

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

    def _index_at(self, index: Int) raises -> Int:
        """Where logical row `index` points into `_values`.

        Split out of `__getitem__` so `__eq__` can decode a position without
        also materialising the value there -- see that method for why touching
        an erased element is the thing to avoid.
        """
        var idx_scalar = self._indices[][self._offset + index]
        ref index_type = self._dtype.as_dictionary().index_type()
        if index_type.is_int8():
            return Int(idx_scalar.as_int8().value())
        elif index_type.is_int16():
            return Int(idx_scalar.as_int16().value())
        elif index_type.is_int32():
            return Int(idx_scalar.as_int32().value())
        elif index_type.is_int64():
            return Int(idx_scalar.as_int64().value())
        elif index_type.is_uint8():
            return Int(idx_scalar.as_uint8().value())
        elif index_type.is_uint16():
            return Int(idx_scalar.as_uint16().value())
        elif index_type.is_uint32():
            return Int(idx_scalar.as_uint32().value())
        elif index_type.is_uint64():
            return Int(idx_scalar.as_uint64().value())
        else:
            raise TypeError(
                t"DictionaryArray: unexpected index type: {index_type}"
            )

    def __getitem__(self, index: Int) raises -> DictionaryScalar:
        if index < 0 or index >= self._length:
            raise IndexError(
                t"index {index} out of bounds for length {self._length}"
            )
        if not self._indices[].is_valid(self._offset + index):
            return DictionaryScalar.null(self._dtype.copy())
        var dict_idx = self._index_at(index)
        var decoded = self._values[][dict_idx]
        return DictionaryScalar(
            dtype=self._dtype.copy(), index=dict_idx, decoded=decoded^
        )

    def slice(self, offset: Int, length: Int) -> Self:
        """Zero-copy slice: adjusts logical offset, shares indices and values.

        The null count is recounted for the sub-range. A dictionary's validity
        is its indices', so the count comes from slicing those.
        """
        var start = self._offset + offset
        return Self(
            dtype=self._dtype.copy(),
            length=length,
            nulls=self._indices[].slice(start, length).null_count(),
            offset=start,
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
        """Return True if both arrays decode to the same values, null for null.

        Compares decoded values, not the encoding: two arrays holding the same
        column against differently ordered dictionaries are equal.

        **The comparison is typed, and that is not a performance choice.** This
        loop used to read `self[i].value() != other[i].value()`, comparing two
        `DynScalar`s -- and comparing *elements* of an erased container is the
        thing `ArrayData.__eq__`'s docstring warns about two thousand lines up:
        the elaborator parks at 0% CPU with no diagnostic. It did not fire from
        `marrow/tests/test_arrays.mojo`, which is why it survived; reached from
        `test_ipc.mojo` it deadlocked `mojo build` -- 6m40s of wall clock for
        11.9s of user CPU -- on Linux (both architectures), while macOS
        compiled it. Resolving the values dtype to a concrete array type first
        means no typed `__eq__` is ever named through an erased one, and the
        same case then compiles in 26s.
        """
        if self._dtype != other._dtype:
            return False
        if self._length != other._length:
            return False
        if self.null_count() != other.null_count():
            return False
        try:
            return self._decoded_equals(other)
        except:
            return False

    def _decoded_equals(self, other: Self) raises -> Bool:
        """Row-by-row decoded comparison, against a concrete values type."""
        ref value_type = self._dtype.as_dictionary().value_type()

        def rows[A: Array](lhs: A, rhs: A) raises {imm} -> Bool:
            for i in range(self._length):
                var lv = self.is_valid(i)
                if lv != other.is_valid(i):
                    return False
                if lv:
                    if lhs[self._index_at(i)] != rhs[other._index_at(i)]:
                        return False
            return True

        def on_bytes[A: BytesArray](lhs: A) raises {imm} -> Bool:
            return rows(lhs, other._values[].as_type[A]())

        def on_primitive[T: PrimitiveType](t: T) raises {imm} -> Bool:
            return rows(
                self._values[].as_type[PrimitiveArray[T]](),
                other._values[].as_type[PrimitiveArray[T]](),
            )

        # `is_bool` and `is_fixed_size_binary` are named on their own because
        # neither belongs to a `dispatch_*` family -- bool is bit-packed rather
        # than fixed-byte-width, so `is_primitive` deliberately excludes it, and
        # `FixedSizeBinaryArray` takes no type parameter. Both are flat, so both
        # can be compared row by row like the rest.
        if value_type.is_bool():
            return rows(self._values[].as_bool(), other._values[].as_bool())
        elif value_type.is_fixed_size_binary():
            return rows(
                self._values[].as_fixed_size_binary(),
                other._values[].as_fixed_size_binary(),
            )
        elif (
            value_type.is_binary_like()
            or value_type.is_string_view()
            or value_type.is_binary_view()
        ):
            return self._values[].dispatch_bytes(on_bytes)
        elif value_type.is_primitive():
            return value_type.dispatch_primitive(on_primitive)
        else:
            # A dictionary over a nested type. Decoding one of those needs the
            # element comparison this method exists to avoid, so answer on the
            # representation instead: equal encodings are equal columns, and a
            # permuted one reads as unequal rather than as a hang. Arrow allows
            # it; nothing in marrow builds one.
            return (
                self._values[] == other._values[]
                and self._indices[] == other._indices[]
                and self._offset == other._offset
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

    def append(mut self, var chunk: DynArray):
        """Add ``chunk``, of this array's ``dtype``, at the end."""
        debug_assert(chunk.dtype() == self.dtype, "chunk of another dtype")
        self.length += len(chunk)
        self.chunks.append(chunk^)

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
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Sized,
    Writable,
):
    """Type-erased, immutable array handle backed by an inline Variant.

    **Does not conform to `Array`.** The surface is the same, but it is this
    struct's own API rather than a trait implementation: every `[T: Array]`
    bound in the tree lives inside the `_dispatch` closures below, so the
    conformance had no consumer — while forcing `Array.slice` to be `raises`
    for all nine typed arrays. It was added in `8334bf0` for a lane
    unification that `7d57398` then abandoned.

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
        BinaryViewArray,
        StringViewArray,
    ]

    var _v: Self.VariantType

    def _dispatch[
        R: Movable, //, Func: def[T: Array](T) -> R
    ](self, func: Func) -> R:
        """Run `func` on the active variant member, narrowed to `Array`.

        The one narrowing adapter for this type. `Array` is named concretely
        because a closure type cannot be generic over its own trait bound, and
        the `isa` ladder is written out here rather than delegated to a shared
        helper: interposing a narrowing closure between the caller and the
        ladder costs a fully inlined copy of the adapter in *every* arm.
        Routing the four boxes through one generic `variant_dispatch` helper
        measured **+662,740 bytes** on `query_streaming_agg_fused` — 31.9% of
        `__text`. Duplicating five lines of `comptime for` per box is the price.
        """

        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, Array):
                if self._v.isa[T]():
                    return func(rebind[downcast[T, Array]](self._v[T]))
        abort("DynArray._dispatch: no arm matched")

    def _dispatch[
        R: Movable, //, Func: def[T: Array](T) raises -> R
    ](self, func: Func) raises -> R:
        """Raising counterpart of `_dispatch`."""

        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, Array):
                if self._v.isa[T]():
                    return func(rebind[downcast[T, Array]](self._v[T]))
        raise InternalError("DynArray._dispatch: no arm matched")

    def dispatch_bytes[
        R: Movable, //, Func: def[A: BytesArray](A) raises -> R
    ](self, func: Func) raises -> R:
        """Run `func` on this array as a `BytesArray` -- a binary or string
        array in either layout, offsets or views.

        For the consumers that only read elements: one arm instead of an
        offsets arm and a views arm at every call site. The same local `isa`
        ladder as `_dispatch`, narrowed to `BytesArray`; any other array raises
        a `TypeError`.
        """

        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, BytesArray):
                if self._v.isa[T]():
                    return func(rebind[downcast[T, BytesArray]](self._v[T]))
        raise TypeError(
            t"dispatch_bytes: expected a binary or string array, got"
            t" {self.dtype()}"
        )

    # --- construction ---

    @implicit
    def __init__[T: Array](out self, var array: T):
        self._v = Self.VariantType(array^)

    def __init__(out self, *, copy: Self):
        self._v = Self.VariantType(copy=copy._v)

    # Explicit (empty) destructor so this type is Deinitable despite
    # the `StructArray -> List[DynArray] -> DynArray` reference cycle; the
    # variant field is still destroyed automatically after the body runs.
    def __deinit__(deinit self):
        pass

    def __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema, CArrowArray

        # Fast path: marrow arrays are now exposed as a single DynArray Python type.
        try:
            self = py.downcast_value_ptr[DynArray]()[].copy()
            return
        except:
            pass
        # Fall back to the Arrow C Data Interface for foreign objects. Only a
        # missing protocol is a TypeError: a failing producer or a malformed
        # export raises its own kind.
        var export: PythonObject
        try:
            export = py.__arrow_c_array__
        except:
            raise TypeError(
                t"cannot convert Python object of type '"
                t"{py.__class__.__name__}' to DynArray"
            )
        var caps = export(Python.none())
        var c_schema = CArrowSchema.from_pycapsule(caps[0])
        var c_array = CArrowArray.from_pycapsule(caps[1])
        self = c_array^.to_array(c_schema.to_dtype())

    # --- dispatch-based methods ---

    def length(self) -> Int:
        def f[T: Array](a: T) {imm} -> Int:
            return len(a)

        return self._dispatch(f)

    def dtype(self) -> DynType:
        def f[T: Array](a: T) {imm} -> DynType:
            return a.type()

        return self._dispatch(f)

    def null_count(self) -> Int:
        def f[T: Array](a: T) {imm} -> Int:
            return a.null_count()

        return self._dispatch(f)

    def is_valid(self, index: Int) -> Bool:
        def f[T: Array](a: T) {imm} -> Bool:
            return a.is_valid(index)

        return self._dispatch(f)

    def is_null(self, index: Int) -> Bool:
        return not self.is_valid(index)

    def slice(self, offset: Int, length: Int = -1) -> DynArray:
        """Returns a zero-copy slice starting at offset with the given length.

        Matches PyArrow's Array.slice(offset, length) API.
        """

        def f[T: Array](a: T) {imm} -> DynArray:
            var actual_length = length if length >= 0 else len(a) - offset
            return a.slice(offset, actual_length)

        return self._dispatch(f)

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

        def f[T: Array](a: T) raises {imm} -> ArrayData:
            return a.to_data()

        return self._dispatch(f)

    def to_device(self, ctx: DeviceContext) raises -> DynArray:
        """Upload this array to the GPU device."""

        def f[T: Array](a: T) raises {imm} -> DynArray:
            return a.to_device(ctx)

        return self._dispatch(f)

    def to_cpu(self, ctx: DeviceContext) raises -> DynArray:
        """Download this array from the GPU device to CPU memory."""

        def f[T: Array](a: T) raises {imm} -> DynArray:
            return a.to_cpu(ctx)

        return self._dispatch(f)

    def write_to[W: Writer](self, mut writer: W):
        def f[T: Array](a: T) {mut writer, imm}:
            a.write_to(writer)

        self._dispatch(f)

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def __eq__(self, other: DynArray) -> Bool:
        """Compare the two arrays' fields, via their flat layout.

        Deliberately **not** `self._v == other._v`. `Variant.__eq__` resolves
        the active member on both sides and so dispatches into every typed
        `__eq__`, including the nested ones — which hold `DynArray` fields and
        come straight back here. That cycle deadlocks the compiler (see
        `ArrayData.__eq__`). Going through `to_data()` reaches the same fields
        without ever naming a typed `__eq__`, so the recursion is this method
        calling itself and nothing else.

        One conversion per array, not per element."""
        try:
            return self.to_data() == other.to_data()
        except:
            return False

    def __ne__(self, other: DynArray) -> Bool:
        return not (self == other)

    def to_python_object(var self) raises -> PythonObject:
        """Convert to a Python Array object (type-erased)."""
        return PythonObject(alloc=self^)

    def __len__(self) -> Int:
        return self.length()

    def __getitem__(self, index: Int) raises -> DynScalar:
        """Return the element at index as a type-erased DynScalar."""
        if index < 0 or index >= self.length():
            raise IndexError(
                t"index {index} out of bounds for length {self.length()}"
            )

        def f[T: Array](a: T) raises {imm} -> DynScalar:
            return a[index].to_dyn()

        return self._dispatch(f)

    # --- typed downcasts (zero-cost reference borrows) ---

    def as_type[T: Array](ref self) -> ref[self._v[T]] T:
        """This array as the concrete `T` it holds — a borrow, no copy.

        The named accessors below are one-liners over it; generic code
        (a fused node reading a typed slot) needs the parameterized form.

        **A wrong `T` kills the process, and only says so in a debug build.**
        The `debug_assert` — the one thing here that names the held dtype — is
        compiled out under release, so a mismatch falls through to
        `Variant.__getitem__` and aborts with a bare `get: wrong variant type`
        naming this line and nothing else. Every array type shares this
        accessor, so that message identifies neither the type held nor the type
        asked for; from a pool worker it also prints once per
        thread. Diagnosing one costs an afternoon.

        So: **only call this where the type has actually been proven** — inside
        a `dispatch_*` arm, or off a `comptime` type the caller owns. Deriving
        `T` from a property that merely correlates with the type is what caused
        that incident: `BinaryLikeBuilder.extend` picked `StringType` vs
        `LargeStringType` off the *builder's* offset width, which says nothing
        about whether the *source* holds text or bytes, and a `binary` column
        aborted every group-by that crossed into the thread-local path."""

        def held() {imm} -> DynType:
            return self.dtype()

        debug_assert_lazy(self._v.isa[T](), "as_type: wrong type, holds ", held)
        return self._v[T]

    def as_primitive[
        T: PrimitiveType
    ](ref self) -> ref[self._v[PrimitiveArray[T]]] PrimitiveArray[T]:
        return self.as_type[PrimitiveArray[T]]()

    def as_binary_like[
        T: BinaryLikeType
    ](ref self) -> ref[self._v[BinaryLikeArray[T]]] BinaryLikeArray[T]:
        return self.as_type[BinaryLikeArray[T]]()

    def as_binary_view_like[
        T: BinaryViewLikeType
    ](ref self) -> ref[self._v[BinaryViewLikeArray[T]]] BinaryViewLikeArray[T]:
        return self.as_type[BinaryViewLikeArray[T]]()

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

    def as_string_view(
        ref self,
    ) -> ref[self._v[StringViewArray]] StringViewArray:
        return self.as_type[StringViewArray]()

    def as_binary_view(
        ref self,
    ) -> ref[self._v[BinaryViewArray]] BinaryViewArray:
        return self.as_type[BinaryViewArray]()

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
        elif dt.is_string_view():
            return StringViewArray(data)
        elif dt.is_binary_view():
            return BinaryViewArray(data)
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
            raise NotImplementedError("from_data: unsupported dtype")


def dispatch_array[
    R: Movable, //, Func: def[A: Array]() raises -> R
](in_dtype: DynType, func: Func) raises -> R:
    """Resolve a runtime dtype to the **array type** that holds it, and run
        `func` at that type.

        The tenth member of the `dispatch_*` family, and the only free function in
        it: the other nine are methods on `DynType`, which narrow a dtype to a
        *dtype* parameter, while this one narrows it to an *array* type -- so it
        belongs beside `DynArray` rather than on `DynType`.

        It was `kernels.aggregate.dispatch_agg_array`, which named neither of the
        two things it does: it takes no array, and knows nothing about aggregates.
        It was also the only free `dispatch_*` in the kernels package, and read as
        a sibling of `expr.runtime.resolve_aggregate` (then `dispatch_agg`), a
        completely different operation in a different layer.

        The counterpart of the `dispatch` static every value kernel exposes
        (`HashKernel.dispatch`, `kernels/hashing.mojo`). Those
        narrow an erased *array* and run immediately, because they are stateless.
        An aggregate is a state machine that must exist before its first morsel, so
        this narrows a *dtype* instead and hands back whatever the caller builds at
        that type — an accumulator, or an operator holding one.

    Its motivating callers are `ValidCount[A]` and `DistinctCount[A]`,
        which it is what lets be typed at all. Both are dtype-generic in what they *compute* — a cardinality is an
        int64 whatever was counted — but not in how they *read*: a valid count
        wants a validity bitmap and a distinct count wants to hash values, and both
        are faster knowing the array type than walking an erased one per row.
        Without this they had to declare `InArray = DynArray`, which made
        `AggKernel.InArray` mean "some column-ish thing" and put an unchecked
        reinterpret at every call site.

        Four arms rather than one ladder over every layout: each existing family
        dispatcher already knows its own array, and `dispatch_primitive` spans
        numeric, temporal, interval and decimal. A layout outside them raises here,
        at plan time, rather than at the first morsel.
    """
    if in_dtype.is_bool():
        return func[BoolArray]()
    elif in_dtype.is_string() or in_dtype.is_large_string():

        def stringly[T: StringLikeType](d: T) raises {imm func} -> R:
            return func[BinaryLikeArray[T]]()

        return in_dtype.dispatch_stringlike(stringly)
    elif in_dtype.is_string_view():

        def viewly[T: BinaryViewLikeType](d: T) raises {imm func} -> R:
            return func[BinaryViewLikeArray[T]]()

        return in_dtype.dispatch_binaryview(viewly)
    elif in_dtype.is_primitive():

        def primitive[T: PrimitiveType](d: T) raises {imm func} -> R:
            return func[PrimitiveArray[T]]()

        return in_dtype.dispatch_primitive(primitive)
    else:
        raise TypeError(t"no array type for {in_dtype} columns")
