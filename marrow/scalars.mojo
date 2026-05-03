"""Arrow scalar types — single-value containers.

Following Arrow C++'s design: scalars hold native values directly, not
length-1 arrays.

Typed scalars:
  PrimitiveScalar[T] — holds _Scalar[T.native] (built-in Scalar) + Bool validity
  StringScalar       — holds String value + Bool validity
  ListScalar         — holds AnyArray (child values) + Bool validity
  StructScalar       — holds List[AnyArray] (one per field) + DataType + Bool validity

Type-erased container:
  AnyScalar          — wraps any typed scalar via @implicit conversion;
                       backed by a length-1 AnyArray for uniform storage.

Scalar trait:
  Common interface implemented by all four typed scalars.
"""

from std.utils import Variant
from std.os import abort
from std.python import PythonObject
from std.python.conversions import ConvertibleToPython
from std.builtin.rebind import downcast

from .arrays import (
    PrimitiveArray,
    StringArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
    AnyArray,
)
from .builders import PrimitiveBuilder, StringBuilder
from .dtypes import (
    AnyDataType,
    PrimitiveType,
    NumericType,
    IntegerType,
    FloatingType,
    DecimalType,
    Field,
    variant_dispatch,
    BoolType,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    Float16Type,
    Float32Type,
    Float64Type,
    NullType,
    FixedSizeBinaryType,
    Date32Type,
    Date64Type,
    Time32Type,
    Time64Type,
    DurationType,
    TimestampType,
    Decimal32Type,
    Decimal64Type,
    Decimal128Type,
    Decimal256Type,
    TimeUnit,
    bool_,
    null,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    list_,
    string,
)

# Alias the built-in Scalar[DType] to avoid shadowing by the local Scalar trait.
from std.builtin.simd import Scalar as _Scalar



# ---------------------------------------------------------------------------
# Scalar trait
# ---------------------------------------------------------------------------


trait Scalar(Copyable, Equatable, Movable, Writable):
    """Common interface for all typed Arrow scalars."""

    def type(self) -> AnyDataType:
        ...

    def is_valid(self) -> Bool:
        ...

    def is_null(self) -> Bool:
        return not self.is_valid()

    def to_any(deinit self) -> AnyScalar:
        return AnyScalar(self^)


struct NullScalar(Scalar):
    """A single null value — Arrow's `Null` type holds nothing but null."""

    def __init__(out self):
        pass

    @staticmethod
    def null() -> Self:
        return Self()

    def type(self) -> AnyDataType:
        return null

    def is_valid(self) -> Bool:
        return False

    def write_to[W: Writer](self, mut writer: W):
        writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("null")


struct BoolScalar(Scalar):
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

    def type(self) -> AnyDataType:
        return bool_

    def is_valid(self) -> Bool:
        return self._is_valid

    def value(self) -> Bool:
        """Get the underlying boolean value. Undefined if null."""
        return self._value


# ---------------------------------------------------------------------------
# PrimitiveScalar[T]
# ---------------------------------------------------------------------------


struct PrimitiveScalar[T: PrimitiveType](Scalar):
    """A single primitive value: holds a native Mojo scalar + type info + validity flag.

    `_dtype: T` carries runtime type information — zero-sized for NumericType,
    but holds unit/timezone for TemporalType and precision/scale for DecimalType.
    """
    comptime NativeScalar = _Scalar[Self.T.native]

    var _value: Self.NativeScalar
    var _dtype: Self.T
    var _is_valid: Bool

    def __init__(out self, value: Self.NativeScalar) where conforms_to(Self.T, Defaultable):
        comptime DT = downcast[Self.T, Defaultable]()
        self._dtype = DT.__init__()
        self._value = value
        self._is_valid = True

    def __init__(out self, value: Optional[Self.NativeScalar]) where conforms_to(Self.T, Defaultable):
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

    def type(self) -> AnyDataType:
        return self._dtype.copy().to_any()

    def is_valid(self) -> Bool:
        return self._is_valid

    def value(self) -> Self.NativeScalar:
        """Get the underlying native value. Undefined if null."""
        return self._value

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

comptime Date32Scalar    = PrimitiveScalar[Date32Type]
comptime Date64Scalar    = PrimitiveScalar[Date64Type]
comptime Time32Scalar    = PrimitiveScalar[Time32Type]
comptime Time64Scalar    = PrimitiveScalar[Time64Type]
comptime DurationScalar  = PrimitiveScalar[DurationType]
comptime TimestampScalar = PrimitiveScalar[TimestampType]

comptime Decimal32Scalar  = PrimitiveScalar[Decimal32Type]
comptime Decimal64Scalar  = PrimitiveScalar[Decimal64Type]
comptime Decimal128Scalar = PrimitiveScalar[Decimal128Type]
comptime Decimal256Scalar = PrimitiveScalar[Decimal256Type]


# ---------------------------------------------------------------------------
# StringScalar
# ---------------------------------------------------------------------------


struct StringScalar(Scalar):
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

    def type(self) -> AnyDataType:
        return string

    def is_valid(self) -> Bool:
        return self._is_valid

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


struct FixedSizeBinaryScalar(Scalar):
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

    def type(self) -> AnyDataType:
        return FixedSizeBinaryType(self._byte_width).to_any()

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


struct ListScalar(Scalar):
    """A single list value: holds an AnyArray of child elements + validity flag.
    """

    var _value: AnyArray
    var _is_valid: Bool

    def __init__(out self, *, value: AnyArray, is_valid: Bool):
        self._value = value.copy()
        self._is_valid = is_valid

    def type(self) -> AnyDataType:
        return list_(self._value.dtype())

    def is_valid(self) -> Bool:
        return self._is_valid

    def value(self) -> AnyArray:
        """Get the child elements array."""
        return self._value.copy()

    def write_to[W: Writer](self, mut writer: W):
        if self._is_valid:
            self._value.write_to(writer)
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# StructScalar
# ---------------------------------------------------------------------------


struct StructScalar(Scalar):
    """A single struct value: holds one AnyScalar per field + validity flag."""

    var _dtype: AnyDataType
    var _value: List[AnyScalar]
    var _is_valid: Bool

    def __init__(
        out self,
        *,
        dtype: AnyDataType,
        value: List[AnyScalar],
        is_valid: Bool,
    ):
        self._dtype = dtype.copy()
        self._value = value.copy()
        self._is_valid = is_valid

    @staticmethod
    def null(dtype: AnyDataType) -> Self:
        return Self(dtype=dtype, value=List[AnyScalar](), is_valid=False)

    def type(self) -> AnyDataType:
        return self._dtype.copy()

    def is_valid(self) -> Bool:
        return self._is_valid

    def num_fields(self) -> Int:
        return len(self._value)

    def field(self, index: Int) -> ref[self._value] AnyScalar:
        """Return the i-th field as an AnyScalar."""
        debug_assert(index >= 0 and index < len(self._value), "field index out of bounds")
        return self._value[index]

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
# AnyScalar — type-erased scalar container
# ---------------------------------------------------------------------------


struct AnyScalar(ConvertibleToPython, Copyable, Equatable, Movable, Writable):
    """Type-erased scalar container backed by a Variant.

    Wraps any typed scalar inline in a discriminated union.
    Runtime dispatch goes through the `_dispatch` helper.
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
        Decimal32Scalar,
        Decimal64Scalar,
        Decimal128Scalar,
        Decimal256Scalar,
        StringScalar,
        FixedSizeBinaryScalar,
        ListScalar,
        StructScalar,
    ]

    var _v: Self.VariantType

    # --- construction ---

    @implicit
    def __init__[T: Scalar](out self, var typed: T):
        self._v = Self.VariantType(typed^)

    def __init__(out self, *, copy: Self):
        self._v = Self.VariantType(copy=copy._v)

    # --- dispatch-based methods ---

    def type(self) -> AnyDataType:
        @parameter
        def f[T: Scalar](t: T) -> AnyDataType:
            return t.type()

        return variant_dispatch[Scalar, func=f](self._v)

    def is_valid(self) -> Bool:
        @parameter
        def f[T: Scalar](t: T) -> Bool:
            return t.is_valid()

        return variant_dispatch[Scalar, func=f](self._v)

    def is_null(self) -> Bool:
        return not self.is_valid()

    # --- typed downcasts ---

    def _as[T: Scalar](ref self) -> ref[self._v] T:
        debug_assert(self._v.isa[T](), "_as: wrong type, holds ", self.type())
        return self._v[T]

    def as_null(ref self) -> ref[self._v] NullScalar:
        return self._as[NullScalar]()

    def as_bool(ref self) -> ref[self._v] BoolScalar:
        return self._as[BoolScalar]()

    def as_primitive[T: PrimitiveType](ref self) -> ref[self._v] PrimitiveScalar[T]:
        return self._as[PrimitiveScalar[T]]()

    def as_int8(ref self) -> ref[self._v] Int8Scalar:
        return self._as[Int8Scalar]()

    def as_int16(ref self) -> ref[self._v] Int16Scalar:
        return self._as[Int16Scalar]()

    def as_int32(ref self) -> ref[self._v] Int32Scalar:
        return self._as[Int32Scalar]()

    def as_int64(ref self) -> ref[self._v] Int64Scalar:
        return self._as[Int64Scalar]()

    def as_uint8(ref self) -> ref[self._v] UInt8Scalar:
        return self._as[UInt8Scalar]()

    def as_uint16(ref self) -> ref[self._v] UInt16Scalar:
        return self._as[UInt16Scalar]()

    def as_uint32(ref self) -> ref[self._v] UInt32Scalar:
        return self._as[UInt32Scalar]()

    def as_uint64(ref self) -> ref[self._v] UInt64Scalar:
        return self._as[UInt64Scalar]()

    def as_float16(ref self) -> ref[self._v] Float16Scalar:
        return self._as[Float16Scalar]()

    def as_float32(ref self) -> ref[self._v] Float32Scalar:
        return self._as[Float32Scalar]()

    def as_float64(ref self) -> ref[self._v] Float64Scalar:
        return self._as[Float64Scalar]()

    def as_string(ref self) -> ref[self._v] StringScalar:
        return self._as[StringScalar]()

    def as_fixed_size_binary(ref self) -> ref[self._v] FixedSizeBinaryScalar:
        return self._as[FixedSizeBinaryScalar]()

    def as_date32(ref self) -> ref[self._v] Date32Scalar:
        return self._as[Date32Scalar]()

    def as_date64(ref self) -> ref[self._v] Date64Scalar:
        return self._as[Date64Scalar]()

    def as_time32(ref self) -> ref[self._v] Time32Scalar:
        return self._as[Time32Scalar]()

    def as_time64(ref self) -> ref[self._v] Time64Scalar:
        return self._as[Time64Scalar]()

    def as_duration(ref self) -> ref[self._v] DurationScalar:
        return self._as[DurationScalar]()

    def as_timestamp(ref self) -> ref[self._v] TimestampScalar:
        return self._as[TimestampScalar]()

    def as_decimal32(ref self) -> ref[self._v] Decimal32Scalar:
        return self._as[Decimal32Scalar]()

    def as_decimal64(ref self) -> ref[self._v] Decimal64Scalar:
        return self._as[Decimal64Scalar]()

    def as_decimal128(ref self) -> ref[self._v] Decimal128Scalar:
        return self._as[Decimal128Scalar]()

    def as_decimal256(ref self) -> ref[self._v] Decimal256Scalar:
        return self._as[Decimal256Scalar]()

    def as_list(ref self) -> ref[self._v] ListScalar:
        return self._as[ListScalar]()

    def as_fixed_size_list(ref self) -> ref[self._v] ListScalar:
        return self._as[ListScalar]()

    def as_struct(ref self) -> ref[self._v] StructScalar:
        return self._as[StructScalar]()

    def __eq__(self, other: Self) -> Bool:
        return self._v == other._v

    def write_to[W: Writer](self, mut writer: W):
        @parameter
        def f[T: Scalar](t: T):
            t.write_to(writer)

        variant_dispatch[Scalar, func=f](self._v)

    def write_repr_to[W: Writer](self, mut writer: W):
        @parameter
        def f[T: Scalar](t: T):
            t.write_repr_to(writer)

        variant_dispatch[Scalar, func=f](self._v)

    def to_python_object(var self) raises -> PythonObject:
        """Convert to a Python Scalar wrapper object."""
        return PythonObject(alloc=self^)


# ---------------------------------------------------------------------------
# scalar() factory
# ---------------------------------------------------------------------------


def scalar[T: NumericType](value: _Scalar[T.native]) -> PrimitiveScalar[T]:
    """Create a typed primitive scalar from a native Mojo scalar."""
    return PrimitiveScalar[T](value)


def scalar(value: Bool) -> BoolScalar:
    """Create a boolean scalar."""
    return BoolScalar(value)
