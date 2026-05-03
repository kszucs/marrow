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


trait Scalar(Copyable, Movable, Writable):
    """Common interface for all typed Arrow scalars."""

    def type(self) -> AnyDataType:
        ...

    def is_valid(self) -> Bool:
        ...

    def is_null(self) -> Bool:
        ...

    def to_any(deinit self) -> AnyScalar:
        ...


struct NullScalar(Copyable, Equatable, Movable, Scalar, Writable):
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

    def is_null(self) -> Bool:
        return True

    def to_any(deinit self) -> AnyScalar:
        return self^

    def __eq__(self, other: Self) -> Bool:
        return True

    def __ne__(self, other: Self) -> Bool:
        return False

    def write_to[W: Writer](self, mut writer: W):
        writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("null")


# TODO(kszucs): base Scalar already inherits from copyable/movable/writable, so
# we don't need to repeat those in each struct definition. We can just have the
# struct definitions inherit from Scalar and then add Equatable where needed.
struct BoolScalar(Copyable, Equatable, Movable, Scalar, Writable):
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

    def is_null(self) -> Bool:
        return not self._is_valid

    def value(self) -> Bool:
        """Get the underlying boolean value. Undefined if null."""
        return self._value

    def to_any(deinit self) -> AnyScalar:
        return self^


# ---------------------------------------------------------------------------
# PrimitiveScalar[T]
# ---------------------------------------------------------------------------


struct PrimitiveScalar[T: PrimitiveType](
    Copyable, Equatable, Movable, Scalar, Writable
):
    """A single primitive value: holds a native Mojo scalar + type info + validity flag.

    `_dtype: T` carries runtime type information — zero-sized for NumericType,
    but holds unit/timezone for TemporalType and precision/scale for DecimalType.
    """

    comptime NativeScalar = _Scalar[Self.T.native]

    var _value: Self.NativeScalar
    var _dtype: Self.T
    var _is_valid: Bool


    def __init__[NT: NumericType](out self: PrimitiveScalar[NT], value: _Scalar[NT.native]):
        self._value = value
        self._is_valid = True
        self._dtype = NT()

    def __init__[NT: NumericType](out self: PrimitiveScalar[NT], none: NoneType):
        self._value = _Scalar[NT.native](0)
        self._is_valid = False
        self._dtype = NT()

    def __init__[NT: NumericType](out self: PrimitiveScalar[NT], value: Optional[_Scalar[NT.native]]):
        self = PrimitiveScalar[NT](value, NT())

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

    def is_null(self) -> Bool:
        return not self._is_valid

    def value(self) -> Self.NativeScalar:
        """Get the underlying native value. Undefined if null."""
        return self._value

    def to_any(deinit self) -> AnyScalar:
        return self^

    def __eq__(self, other: Self) -> Bool:
        if self.is_null() and other.is_null():
            return True
        if self.is_null() or other.is_null():
            return False
        return self._dtype == other._dtype and self._value == other._value

    # TODO(kszucs): shouldn't define __ne__ since there is a default impl
    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    # def __bool__(self) -> Bool:
    #     if self._is_valid:
    #         return Bool(self._value)
    #     return False

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


struct StringScalar(Copyable, Equatable, Movable, Scalar, Writable):
    """A single string value: holds a String + validity flag."""

    var _value: String
    var _is_valid: Bool

    @implicit
    def __init__(out self, value: String) raises:
        self._value = value
        self._is_valid = True

    def __init__(out self, *, is_valid: Bool) raises:
        self._value = String()
        self._is_valid = is_valid

    @staticmethod
    def null() raises -> Self:
        return Self(is_valid=False)

    def type(self) -> AnyDataType:
        return string

    def is_valid(self) -> Bool:
        return self._is_valid

    def is_null(self) -> Bool:
        return not self._is_valid

    def to_string(self) -> String:
        """Get the value as an owned String."""
        return self._value

    def to_any(deinit self) -> AnyScalar:
        return self^

    def __eq__(self, other: Self) -> Bool:
        if self.is_null() and other.is_null():
            return True
        if self.is_null() or other.is_null():
            return False
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

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


struct FixedSizeBinaryScalar(Copyable, Equatable, Movable, Scalar, Writable):
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

    def is_null(self) -> Bool:
        return not self._is_valid

    def to_any(deinit self) -> AnyScalar:
        return self^

    def __eq__(self, other: Self) -> Bool:
        if self.is_null() and other.is_null():
            return True
        if self.is_null() or other.is_null():
            return False
        if self._byte_width != other._byte_width:
            return False
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

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


struct ListScalar(Copyable, Movable, Scalar, Writable):
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

    def is_null(self) -> Bool:
        return not self._is_valid

    def value(self) -> AnyArray:
        """Get the child elements array."""
        return self._value.copy()

    def to_any(deinit self) -> AnyScalar:
        return self^

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


struct StructScalar(Copyable, Movable, Scalar, Writable):
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

    def is_null(self) -> Bool:
        return not self._is_valid

    def num_fields(self) -> Int:
        return len(self._value)

    def field(self, index: Int) -> AnyScalar:
        """Return the i-th field as an AnyScalar."""
        return self._value[index].copy()

    def to_any(deinit self) -> AnyScalar:
        return self^

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


struct AnyScalar(ConvertibleToPython, Copyable, Movable, Writable):
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

    # TODO(kszucs): should remove references just like we do in arrays.mojo
    def as_null(self) -> NullScalar:
        debug_assert(self._v.isa[NullScalar](), "expected null scalar but holds ", self.type())
        return self._v[NullScalar].copy()

    def as_bool(self) -> BoolScalar:
        debug_assert(self._v.isa[BoolScalar](), "expected bool scalar but holds ", self.type())
        return self._v[BoolScalar].copy()

    def as_primitive[T: PrimitiveType](self) -> PrimitiveScalar[T]:
        debug_assert(
            self._v.isa[PrimitiveScalar[T]](), "as_primitive: wrong type, holds ", self.type()
        )
        return self._v[PrimitiveScalar[T]].copy()

    def as_int8(self) -> Int8Scalar:
        return self.as_primitive[Int8Type]()

    def as_int16(self) -> Int16Scalar:
        return self.as_primitive[Int16Type]()

    def as_int32(self) -> Int32Scalar:
        return self.as_primitive[Int32Type]()

    def as_int64(self) -> Int64Scalar:
        return self.as_primitive[Int64Type]()

    def as_uint8(self) -> UInt8Scalar:
        return self.as_primitive[UInt8Type]()

    def as_uint16(self) -> UInt16Scalar:
        return self.as_primitive[UInt16Type]()

    def as_uint32(self) -> UInt32Scalar:
        return self.as_primitive[UInt32Type]()

    def as_uint64(self) -> UInt64Scalar:
        return self.as_primitive[UInt64Type]()

    def as_float16(self) -> Float16Scalar:
        return self.as_primitive[Float16Type]()

    def as_float32(self) -> Float32Scalar:
        return self.as_primitive[Float32Type]()

    def as_float64(self) -> Float64Scalar:
        return self.as_primitive[Float64Type]()

    def as_string(self) -> StringScalar:
        debug_assert(self._v.isa[StringScalar](), "expected string scalar but holds ", self.type())
        return self._v[StringScalar].copy()

    def as_fixed_size_binary(self) -> FixedSizeBinaryScalar:
        debug_assert(
            self._v.isa[FixedSizeBinaryScalar](),
            "expected fixed_size_binary scalar but holds ", self.type(),
        )
        return self._v[FixedSizeBinaryScalar].copy()

    def as_date32(self) -> Date32Scalar:
        return self.as_primitive[Date32Type]()

    def as_date64(self) -> Date64Scalar:
        return self.as_primitive[Date64Type]()

    def as_time32(self) -> Time32Scalar:
        return self.as_primitive[Time32Type]()

    def as_time64(self) -> Time64Scalar:
        return self.as_primitive[Time64Type]()

    def as_duration(self) -> DurationScalar:
        return self.as_primitive[DurationType]()

    def as_timestamp(self) -> TimestampScalar:
        return self.as_primitive[TimestampType]()

    def as_decimal32(self) -> Decimal32Scalar:
        return self.as_primitive[Decimal32Type]()

    def as_decimal64(self) -> Decimal64Scalar:
        return self.as_primitive[Decimal64Type]()

    def as_decimal128(self) -> Decimal128Scalar:
        return self.as_primitive[Decimal128Type]()

    def as_decimal256(self) -> Decimal256Scalar:
        return self.as_primitive[Decimal256Type]()

    def as_list(self) -> ListScalar:
        debug_assert(self._v.isa[ListScalar](), "expected list scalar but holds ", self.type())
        return self._v[ListScalar].copy()

    def as_fixed_size_list(self) -> ListScalar:
        debug_assert(self._v.isa[ListScalar](), "expected fixed_size_list scalar but holds ", self.type())
        return self._v[ListScalar].copy()

    def as_struct(self) -> StructScalar:
        debug_assert(self._v.isa[StructScalar](), "expected struct scalar but holds ", self.type())
        return self._v[StructScalar].copy()

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
