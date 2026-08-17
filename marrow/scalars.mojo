"""Arrow scalar types — single-value containers.

Following Arrow C++'s design: scalars hold native values directly, not
length-1 arrays.

Typed scalars:
  PrimitiveScalar[T]  — holds Scalar[T.native] (built-in) + Bool validity
  StringScalar        — holds String value + Bool validity
  ListScalar          — holds DynArray (child values) + Bool validity
  StructScalar        — holds List[DynScalar] (one per field) + DataType + Bool validity
  DictionaryScalar    — holds integer index + decoded DynScalar value + DataType + Bool validity

Type-erased container:
  DynScalar          — wraps any typed scalar via @implicit conversion;
                       backed by an inline Variant, dispatched at runtime.

ArrowScalar trait:
  Common interface implemented by all four typed scalars.
  (Named `ArrowScalar`, not `Scalar`, to avoid shadowing the builtin
  `Scalar[dtype]` alias — the bare name is ambiguous along the
  `arrays` <-> `dtypes` circular imports.)
"""

from std.utils import Variant

from std.python import Python, PythonObject
from std.python.conversions import ConvertibleToPython
from std.builtin.rebind import downcast
from std.memory import OwnedPointer

from .arrays import (
    BoolArray,
    PrimitiveArray,
    StringArray,
    DynArray,
)
from .builders import BoolBuilder, PrimitiveBuilder, StringBuilder
from std.os import abort
from .dtypes import (
    DynType,
    PrimitiveType,
    StringLikeType,
    Date32Type,
    Date64Type,
    DayTimeIntervalType,
    Decimal128Type,
    Decimal256Type,
    Decimal32Type,
    Decimal64Type,
    DurationType,
    FixedSizeBinaryType,
    Float16Type,
    Float32Type,
    Float64Type,
    Int16Type,
    Int32Type,
    Int64Type,
    Int8Type,
    MonthDayNanoIntervalType,
    NumericType,
    PrimitiveType,
    Time32Type,
    Time64Type,
    TimestampType,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    UInt8Type,
    YearMonthIntervalType,
    bool_,
    field,
    list_,
    null,
    string,
)

# ---------------------------------------------------------------------------
# ArrowScalar trait
# ---------------------------------------------------------------------------


trait ArrowScalar(Copyable, Deinitable, Equatable, Movable, Writable):
    """Common interface for all typed Arrow scalars."""

    def type(self) -> DynType:
        ...

    def is_valid(self) -> Bool:
        ...

    def is_null(self) -> Bool:
        return not self.is_valid()

    def to_dyn(deinit self) -> DynScalar:
        return DynScalar(self^)


struct NullScalar(ArrowScalar):
    """A single null value — Arrow's `Null` type holds nothing but null."""

    def __init__(out self):
        pass

    @staticmethod
    def null() -> Self:
        return Self()

    def type(self) -> DynType:
        return null

    def is_valid(self) -> Bool:
        return False

    def write_to[W: Writer](self, mut writer: W):
        writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("null")


struct BoolScalar(ArrowScalar):
    """A single boolean value: holds a Bool + validity flag."""

    var _value: Bool
    var _is_valid: Bool

    @implicit
    def __init__(out self, value: Bool):
        self._value = value
        self._is_valid = True

    def __init__(out self, *, is_valid: Bool):
        self._value = False
        self._is_valid = is_valid

    @staticmethod
    def null() -> Self:
        return Self(is_valid=False)

    def repeat(self, times: Int) raises -> BoolArray:
        """Broadcast this scalar into an array of length `times`."""
        var builder = BoolBuilder(times)
        if self._is_valid:
            for _ in range(times):
                builder.append(self._value)
        else:
            for _ in range(times):
                builder.append_null()
        return builder.finish()

    def type(self) -> DynType:
        return bool_

    def is_valid(self) -> Bool:
        return self._is_valid

    def value(self) -> Bool:
        """Get the underlying boolean value. Undefined if null."""
        return self._value


# ---------------------------------------------------------------------------
# PrimitiveScalar[T]
# ---------------------------------------------------------------------------


struct PrimitiveScalar[T: PrimitiveType](ArrowScalar):
    """A single primitive value: holds a native Mojo scalar + type info + validity flag.

    `_dtype: T` carries runtime type information — zero-sized for NumericType,
    but holds unit/timezone for TemporalType and precision/scale for DecimalType.
    """

    comptime NativeScalar = Scalar[Self.T.native]

    var _value: Self.NativeScalar
    var _dtype: Self.T
    var _is_valid: Bool

    def __init__(
        out self, value: Self.NativeScalar
    ) where conforms_to(Self.T, Defaultable):
        comptime DT = downcast[Self.T, Defaultable]()
        self._dtype = DT.__init__()
        self._value = value
        self._is_valid = True

    def __init__(
        out self, value: Optional[Self.NativeScalar]
    ) where conforms_to(Self.T, Defaultable):
        comptime DT = downcast[Self.T, Defaultable]()
        self._dtype = DT.__init__()
        if value:
            self._value = value.value()
            self._is_valid = True
        else:
            self._value = Self.NativeScalar(0)
            self._is_valid = False

    def __init__(out self, value: Optional[Self.NativeScalar], dtype: Self.T):
        if value:
            self._value = value.value()
            self._is_valid = True
        else:
            self._value = Self.NativeScalar(0)
            self._is_valid = False
        self._dtype = dtype.copy()

    def type(self) -> DynType:
        return self._dtype.copy().to_dyn()

    def is_valid(self) -> Bool:
        return self._is_valid

    def value(self) -> Self.NativeScalar:
        """Get the underlying native value. Undefined if null."""
        return self._value

    def repeat(self, times: Int) raises -> PrimitiveArray[Self.T]:
        """Broadcast this scalar into an array of length `times`."""
        var builder = PrimitiveBuilder[Self.T](self._dtype.copy(), times)
        if self._is_valid:
            for _ in range(times):
                builder.unsafe_append(self._value)
        else:
            for _ in range(times):
                builder.unsafe_append_null()
        return builder.finish()

    def write_to[W: Writer](self, mut writer: W):
        if self._is_valid:
            writer.write(self._value)
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# PrimitiveScalar aliases
# ---------------------------------------------------------------------------

comptime Int8Scalar = PrimitiveScalar[Int8Type]
comptime Int16Scalar = PrimitiveScalar[Int16Type]
comptime Int32Scalar = PrimitiveScalar[Int32Type]
comptime Int64Scalar = PrimitiveScalar[Int64Type]
comptime UInt8Scalar = PrimitiveScalar[UInt8Type]
comptime UInt16Scalar = PrimitiveScalar[UInt16Type]
comptime UInt32Scalar = PrimitiveScalar[UInt32Type]
comptime UInt64Scalar = PrimitiveScalar[UInt64Type]
comptime Float16Scalar = PrimitiveScalar[Float16Type]
comptime Float32Scalar = PrimitiveScalar[Float32Type]
comptime Float64Scalar = PrimitiveScalar[Float64Type]

comptime Date32Scalar = PrimitiveScalar[Date32Type]
comptime Date64Scalar = PrimitiveScalar[Date64Type]
comptime Time32Scalar = PrimitiveScalar[Time32Type]
comptime Time64Scalar = PrimitiveScalar[Time64Type]
comptime DurationScalar = PrimitiveScalar[DurationType]
comptime TimestampScalar = PrimitiveScalar[TimestampType]

comptime YearMonthIntervalScalar = PrimitiveScalar[YearMonthIntervalType]
comptime DayTimeIntervalScalar = PrimitiveScalar[DayTimeIntervalType]
comptime MonthDayNanoIntervalScalar = PrimitiveScalar[MonthDayNanoIntervalType]

comptime Decimal32Scalar = PrimitiveScalar[Decimal32Type]
comptime Decimal64Scalar = PrimitiveScalar[Decimal64Type]
comptime Decimal128Scalar = PrimitiveScalar[Decimal128Type]
comptime Decimal256Scalar = PrimitiveScalar[Decimal256Type]


# ---------------------------------------------------------------------------
# StringScalar
# ---------------------------------------------------------------------------


struct StringScalar(ArrowScalar):
    """A single string value: holds a String + validity flag."""

    var _value: String
    var _is_valid: Bool

    @implicit
    def __init__(out self, value: String):
        self._value = value
        self._is_valid = True

    def __init__(out self, *, is_valid: Bool):
        self._value = String()
        self._is_valid = is_valid

    @staticmethod
    def null() -> Self:
        return Self(is_valid=False)

    def type(self) -> DynType:
        return string

    def is_valid(self) -> Bool:
        return self._is_valid

    def repeat(self, times: Int) raises -> StringArray:
        """Broadcast this scalar into an array of length `times`.

        The numeric and bool scalars have had this; string did not, so
        `DynScalar.repeat` could not support it and the erased `like` path — which
        materialises its constant pattern per morsel — failed at run time."""
        var builder = StringBuilder(capacity=times)
        if self._is_valid:
            for _ in range(times):
                builder.append(self._value)
        else:
            for _ in range(times):
                builder.append_null()
        return builder.finish()

    def to_string(self) -> String:
        """Get the value as an owned String."""
        return self._value

    def write_to[W: Writer](self, mut writer: W):
        if self._is_valid:
            writer.write(self._value)
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        if self._is_valid:
            writer.write('"')
            writer.write(self._value)
            writer.write('"')
        else:
            writer.write("null")


# ---------------------------------------------------------------------------
# FixedSizeBinaryScalar
# ---------------------------------------------------------------------------


struct FixedSizeBinaryScalar(ArrowScalar):
    """A single fixed-size-binary value: holds `byte_width` bytes + validity flag.
    """

    var _value: List[UInt8]
    var _is_valid: Bool
    var _byte_width: Int

    def __init__(out self, var value: List[UInt8], byte_width: Int):
        self._value = value^
        self._is_valid = True
        self._byte_width = byte_width

    def __init__(out self, *, byte_width: Int, is_valid: Bool):
        self._value = List[UInt8]()
        self._is_valid = is_valid
        self._byte_width = byte_width

    @staticmethod
    def null(byte_width: Int) -> Self:
        return Self(byte_width=byte_width, is_valid=False)

    def type(self) -> DynType:
        return FixedSizeBinaryType(self._byte_width).to_dyn()

    def value(ref self) -> ref[self._value] List[UInt8]:
        """The `byte_width` value bytes (empty for a null scalar)."""
        return self._value

    def is_valid(self) -> Bool:
        return self._is_valid

    def write_to[W: Writer](self, mut writer: W):
        if self._is_valid:
            writer.write("b'")
            for i in range(len(self._value)):
                writer.write(self._value[i])
            writer.write("'")
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# ListScalar
# ---------------------------------------------------------------------------


struct ListScalar(ArrowScalar):
    """A single list value: holds an DynArray of child elements + validity flag.
    """

    var _value: OwnedPointer[DynArray]
    var _is_valid: Bool

    def __init__(out self, *, var value: DynArray, is_valid: Bool):
        self._value = OwnedPointer(value^)
        self._is_valid = is_valid

    def __init__(out self, *, copy: Self):
        self._value = OwnedPointer(copy._value[].copy())
        self._is_valid = copy._is_valid

    def type(self) -> DynType:
        return list_(self._value[].dtype())

    def is_valid(self) -> Bool:
        return self._is_valid

    def value(self) -> DynArray:
        """Get the child elements array."""
        return self._value[].copy()

    def __eq__(self, other: Self) -> Bool:
        return (
            self._is_valid == other._is_valid
            and self._value[] == other._value[]
        )

    def write_to[W: Writer](self, mut writer: W):
        if self._is_valid:
            self._value[].write_to(writer)
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# StructScalar
# ---------------------------------------------------------------------------


struct StructScalar(ArrowScalar):
    """A single struct value: holds one DynScalar per field + validity flag."""

    var _dtype: DynType
    var _value: List[DynScalar]
    var _is_valid: Bool

    def __init__(
        out self,
        *,
        dtype: DynType,
        var value: List[DynScalar],
        is_valid: Bool,
    ):
        self._dtype = dtype.copy()
        self._value = value^
        self._is_valid = is_valid

    @staticmethod
    def null(dtype: DynType) -> Self:
        return Self(dtype=dtype, value=List[DynScalar](), is_valid=False)

    def type(self) -> DynType:
        return self._dtype.copy()

    def is_valid(self) -> Bool:
        return self._is_valid

    def num_fields(self) -> Int:
        return len(self._value)

    def field(self, index: Int) -> DynScalar:
        """Return the i-th field as an DynScalar."""
        debug_assert(
            index >= 0 and index < len(self._value), "field index out of bounds"
        )
        return self._value[index].copy()

    def write_to[W: Writer](self, mut writer: W):
        if self._is_valid:
            writer.write("{")
            for i in range(len(self._value)):
                if i > 0:
                    writer.write(", ")
                self._value[i].write_to(writer)
            writer.write("}")
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# DictionaryScalar
# ---------------------------------------------------------------------------


struct DictionaryScalar(ArrowScalar):
    """A single dictionary-encoded value: holds the integer index + decoded value.

    Equivalent to PyArrow's ``pyarrow.DictionaryScalar``.
    """

    var _dtype: DynType
    var _index: Int  # integer index into the dictionary; -1 when null
    var _decoded: OwnedPointer[
        DynScalar
    ]  # decoded (looked-up) value; NullScalar when invalid

    def __init__(
        out self,
        *,
        dtype: DynType,
        index: Int,
        var decoded: DynScalar,
    ):
        self._dtype = dtype.copy()
        self._index = index
        self._decoded = OwnedPointer(decoded^)

    def __init__(out self, *, copy: Self):
        self._dtype = copy._dtype.copy()
        self._index = copy._index
        self._decoded = OwnedPointer(copy._decoded[].copy())

    @staticmethod
    def null(dtype: DynType) -> Self:
        return Self(dtype=dtype, index=-1, decoded=NullScalar())

    def type(self) -> DynType:
        return self._dtype.copy()

    def is_valid(self) -> Bool:
        return not self._decoded[].is_null()

    def index(self) -> Int:
        """The integer index into the dictionary. -1 when null."""
        return self._index

    def value(self) -> DynScalar:
        """The decoded dictionary value. Matches PyArrow's DictionaryScalar.as_py().
        """
        return self._decoded[].copy()

    def __eq__(self, other: Self) -> Bool:
        return (
            self._dtype == other._dtype
            and self._index == other._index
            and self._decoded[] == other._decoded[]
        )

    def write_to[W: Writer](self, mut writer: W):
        self._decoded[].write_to(writer)

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# DynScalar — type-erased scalar container
# ---------------------------------------------------------------------------


struct DynScalar(ConvertibleToPython, Copyable, Equatable, Movable, Writable):
    """Type-erased scalar container backed by a Variant.

    Wraps any typed scalar inline in a discriminated union.
    Runtime dispatch goes through the `_dispatch` helper.

    **Does not conform to `ArrowScalar`.** It exposes the same surface as its
    own API. The conformance existed to satisfy `DynArray.ScalarType`, which in
    turn existed to satisfy `DynBuilder.ArrayType` — a closed loop with no
    consumer outside it, added in `8334bf0` for a lane unification that
    `7d57398` then abandoned. Note `type()` keeps its name here: that is the
    spelling all nine *typed* scalars use, and renaming only the box would
    create the divergence this removal is undoing.
    """

    comptime VariantType = Variant[
        NullScalar,
        BoolScalar,
        Int8Scalar,
        Int16Scalar,
        Int32Scalar,
        Int64Scalar,
        UInt8Scalar,
        UInt16Scalar,
        UInt32Scalar,
        UInt64Scalar,
        Float16Scalar,
        Float32Scalar,
        Float64Scalar,
        Date32Scalar,
        Date64Scalar,
        Time32Scalar,
        Time64Scalar,
        DurationScalar,
        TimestampScalar,
        YearMonthIntervalScalar,
        DayTimeIntervalScalar,
        MonthDayNanoIntervalScalar,
        Decimal32Scalar,
        Decimal64Scalar,
        Decimal128Scalar,
        Decimal256Scalar,
        StringScalar,
        FixedSizeBinaryScalar,
        ListScalar,
        StructScalar,
        DictionaryScalar,
    ]

    var _v: Self.VariantType

    def _dispatch[
        R: Movable, //, Func: def[T: ArrowScalar](T) -> R
    ](self, func: Func) -> R:
        """Run `func` on the active variant member, narrowed to `ArrowScalar`.

        The one narrowing adapter for this type. `ArrowScalar` is named concretely
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
            comptime if conforms_to(T, ArrowScalar):
                if self._v.isa[T]():
                    return func(rebind[downcast[T, ArrowScalar]](self._v[T]))
        abort("DynScalar._dispatch: no arm matched")

    def _dispatch[
        R: Movable, //, Func: def[T: ArrowScalar](T) raises -> R
    ](self, func: Func) raises -> R:
        """Raising counterpart of `_dispatch`."""

        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, ArrowScalar):
                if self._v.isa[T]():
                    return func(rebind[downcast[T, ArrowScalar]](self._v[T]))
        raise Error("DynScalar._dispatch: no arm matched")

    # --- construction ---

    @implicit
    def __init__[T: ArrowScalar](out self, var typed: T):
        self._v = Self.VariantType(typed^)

    def __init__(out self, *, copy: Self):
        self._v = Self.VariantType(copy=copy._v)

    # Explicit (empty) destructor so this type is Deinitable despite
    # the `StructScalar -> List[DynScalar] -> DynScalar` reference cycle; the
    # variant field is still destroyed automatically after the body runs.
    def __deinit__(deinit self):
        pass

    # --- dispatch-based methods ---

    def type(self) -> DynType:
        def f[T: ArrowScalar](t: T) {imm} -> DynType:
            return t.type()

        return self._dispatch(f)

    def is_valid(self) -> Bool:
        def f[T: ArrowScalar](t: T) {imm} -> Bool:
            return t.is_valid()

        return self._dispatch(f)

    def repeat(self, times: Int) raises -> DynArray:
        """Broadcast this scalar into an array of length `times`.

        A twelve-arm dtype ladder before, which is why it silently lacked string
        — the erased `like` path materialises its pattern through here and hit
        "unsupported dtype string".

        Dispatches over **numeric**, not `PrimitiveType`, and that is a measured
        choice rather than a timid one: `repeat` builds a whole array per
        instantiation, so widening it to every primitive type (adding temporal,
        interval and the four decimals) cost **34,052 bytes** on
        `query_streaming` — 83% of that commit's growth — to support scalars
        nothing repeats today. The families that are covered are enumerated by
        `dispatch_*`, so none of *them* can be silently dropped; a temporal or
        decimal scalar raises, which is the same answer the ladder gave and is
        loud rather than wrong."""
        var dt = self.type()
        if dt == bool_:
            return self.as_bool().repeat(times).to_dyn()
        elif dt.is_string_like():

            def stringlike[T: StringLikeType](d: T) raises {imm} -> DynArray:
                return self.as_string().repeat(times).to_dyn()

            return dt.dispatch_stringlike(stringlike)
        elif dt.is_numeric():

            def numeric[T: NumericType](d: T) raises {imm} -> DynArray:
                return self.as_primitive[T]().repeat(times).to_dyn()

            return dt.dispatch_numeric(numeric)
        else:
            raise Error(t"DynScalar.repeat: unsupported dtype {dt}")

    def is_null(self) -> Bool:
        return not self.is_valid()

    # --- typed downcasts ---

    def as_type[T: ArrowScalar](ref self) -> ref[self._v[T]] T:
        """This scalar as the concrete `T` it holds — a borrow, no copy."""
        debug_assert(
            self._v.isa[T](), "as_type: wrong type, holds ", self.type()
        )
        return self._v[T]

    def as_null(ref self) -> ref[self._v[NullScalar]] NullScalar:
        return self.as_type[NullScalar]()

    def as_bool(ref self) -> ref[self._v[BoolScalar]] BoolScalar:
        return self.as_type[BoolScalar]()

    def as_primitive[
        T: PrimitiveType
    ](ref self) -> ref[self._v[PrimitiveScalar[T]]] PrimitiveScalar[T]:
        return self.as_type[PrimitiveScalar[T]]()

    def as_int8(ref self) -> ref[self._v[Int8Scalar]] Int8Scalar:
        return self.as_type[Int8Scalar]()

    def as_int16(ref self) -> ref[self._v[Int16Scalar]] Int16Scalar:
        return self.as_type[Int16Scalar]()

    def as_int32(ref self) -> ref[self._v[Int32Scalar]] Int32Scalar:
        return self.as_type[Int32Scalar]()

    def as_int64(ref self) -> ref[self._v[Int64Scalar]] Int64Scalar:
        return self.as_type[Int64Scalar]()

    def as_uint8(ref self) -> ref[self._v[UInt8Scalar]] UInt8Scalar:
        return self.as_type[UInt8Scalar]()

    def as_uint16(ref self) -> ref[self._v[UInt16Scalar]] UInt16Scalar:
        return self.as_type[UInt16Scalar]()

    def as_uint32(ref self) -> ref[self._v[UInt32Scalar]] UInt32Scalar:
        return self.as_type[UInt32Scalar]()

    def as_uint64(ref self) -> ref[self._v[UInt64Scalar]] UInt64Scalar:
        return self.as_type[UInt64Scalar]()

    def as_float16(ref self) -> ref[self._v[Float16Scalar]] Float16Scalar:
        return self.as_type[Float16Scalar]()

    def as_float32(ref self) -> ref[self._v[Float32Scalar]] Float32Scalar:
        return self.as_type[Float32Scalar]()

    def as_float64(ref self) -> ref[self._v[Float64Scalar]] Float64Scalar:
        return self.as_type[Float64Scalar]()

    def as_string(ref self) -> ref[self._v[StringScalar]] StringScalar:
        return self.as_type[StringScalar]()

    def as_fixed_size_binary(
        ref self,
    ) -> ref[self._v[FixedSizeBinaryScalar]] FixedSizeBinaryScalar:
        return self.as_type[FixedSizeBinaryScalar]()

    def as_date32(ref self) -> ref[self._v[Date32Scalar]] Date32Scalar:
        return self.as_type[Date32Scalar]()

    def as_date64(ref self) -> ref[self._v[Date64Scalar]] Date64Scalar:
        return self.as_type[Date64Scalar]()

    def as_time32(ref self) -> ref[self._v[Time32Scalar]] Time32Scalar:
        return self.as_type[Time32Scalar]()

    def as_time64(ref self) -> ref[self._v[Time64Scalar]] Time64Scalar:
        return self.as_type[Time64Scalar]()

    def as_duration(ref self) -> ref[self._v[DurationScalar]] DurationScalar:
        return self.as_type[DurationScalar]()

    def as_timestamp(ref self) -> ref[self._v[TimestampScalar]] TimestampScalar:
        return self.as_type[TimestampScalar]()

    def as_year_month_interval(
        ref self,
    ) -> ref[self._v[YearMonthIntervalScalar]] YearMonthIntervalScalar:
        return self.as_type[YearMonthIntervalScalar]()

    def as_day_time_interval(
        ref self,
    ) -> ref[self._v[DayTimeIntervalScalar]] DayTimeIntervalScalar:
        return self.as_type[DayTimeIntervalScalar]()

    def as_month_day_nano_interval(
        ref self,
    ) -> ref[self._v[MonthDayNanoIntervalScalar]] MonthDayNanoIntervalScalar:
        return self.as_type[MonthDayNanoIntervalScalar]()

    def as_decimal32(ref self) -> ref[self._v[Decimal32Scalar]] Decimal32Scalar:
        return self.as_type[Decimal32Scalar]()

    def as_decimal64(ref self) -> ref[self._v[Decimal64Scalar]] Decimal64Scalar:
        return self.as_type[Decimal64Scalar]()

    def as_decimal128(
        ref self,
    ) -> ref[self._v[Decimal128Scalar]] Decimal128Scalar:
        return self.as_type[Decimal128Scalar]()

    def as_decimal256(
        ref self,
    ) -> ref[self._v[Decimal256Scalar]] Decimal256Scalar:
        return self.as_type[Decimal256Scalar]()

    def as_list(ref self) -> ref[self._v[ListScalar]] ListScalar:
        return self.as_type[ListScalar]()

    def as_fixed_size_list(ref self) -> ref[self._v[ListScalar]] ListScalar:
        return self.as_type[ListScalar]()

    def as_struct(ref self) -> ref[self._v[StructScalar]] StructScalar:
        return self.as_type[StructScalar]()

    def as_dictionary(
        ref self,
    ) -> ref[self._v[DictionaryScalar]] DictionaryScalar:
        return self.as_type[DictionaryScalar]()

    def __eq__(self, other: Self) -> Bool:
        return self._v == other._v

    def write_to[W: Writer](self, mut writer: W):
        def f[T: ArrowScalar](t: T) {mut writer, imm}:
            t.write_to(writer)

        self._dispatch(f)

    def write_repr_to[W: Writer](self, mut writer: W):
        def f[T: ArrowScalar](t: T) {mut writer, imm}:
            t.write_repr_to(writer)

        self._dispatch(f)

    def to_python_object(var self) raises -> PythonObject:
        """Convert to a Python Scalar wrapper object."""
        return PythonObject(alloc=self^)

    def as_py(self) raises -> PythonObject:
        """This scalar as a native Python value — `int`, `float`, `bool`, `str`,
        `list`, `dict`, or `None`. Matches `pyarrow.Scalar.as_py`.

        Lived in `python/bindings/scalars.mojo` as a 50-line `_as_py` ladder over
        every dtype. Converting a *core* type to a Python value is core work: the
        bindings should say `scalar.as_py()`, not re-derive the dtype mapping.

        Dispatches rather than laddering, so a new numeric or string-like dtype
        cannot be silently omitted — which the ladder had no protection against.
        """
        if self.is_null():
            return PythonObject(None)
        var dt = self.type()
        if dt.is_bool():
            return PythonObject(self.as_bool().value())
        elif dt.is_numeric() or dt.is_interval():

            def numeric[T: PrimitiveType](d: T) raises {imm} -> PythonObject:
                return PythonObject(self.as_primitive[T]().value())

            return dt.dispatch_primitive(numeric)
        elif dt.is_string_like():
            return PythonObject(self.as_string().to_string())
        elif dt.is_list():
            return self.as_list().value().to_python_object()
        elif dt.is_fixed_size_list():
            return self.as_fixed_size_list().value().to_python_object()
        elif dt.is_struct():
            ref st = self.as_struct()
            var builtins = Python.import_module("builtins")
            var d = builtins.dict()
            for i in range(st.num_fields()):
                d[dt.as_struct().fields[i].name] = st.field(i).as_py()
            return d
        else:
            raise Error(t"as_py: unsupported dtype {dt}")


# ---------------------------------------------------------------------------
# scalar() factory
# ---------------------------------------------------------------------------


def scalar[T: NumericType](value: Scalar[T.native]) -> PrimitiveScalar[T]:
    """Create a typed primitive scalar from a native Mojo scalar."""
    return PrimitiveScalar[T](value)


def scalar(value: Bool) -> BoolScalar:
    """Create a boolean scalar."""
    return BoolScalar(value)
