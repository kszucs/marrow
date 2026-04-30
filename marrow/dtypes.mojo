"""Arrow data type system — Variant-based implementation.

`DataType` is the trait that all concrete Arrow type structs implement.
`PrimitiveType` is a sub-trait that adds a `comptime native: DType` field,
enabling primitive-typed generics to use `T.native` as a compile-time type
parameter (e.g. `Buffer[T.native]`, `Scalar[T.native]`).

`AnyDataType` is the type-erased runtime container backed by a `Variant` — no
heap allocation, no vtable, direct member access.

Concrete zero-size type structs (one per Arrow type):
    NullType, BoolType,
    Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    Float16Type, Float32Type, Float64Type,
    BinaryType, StringType,
    ListType, FixedSizeListType, StructType

Comptime singletons (same names as before):
    null, bool_, int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64, binary, string

"""

from std.utils import Variant
from std.builtin.rebind import downcast, trait_downcast
from std.sys import size_of, bit_width_of
from std.os import abort
from std.memory import ArcPointer, OwnedPointer
from std.python import PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from std.sys.compile import codegen_unreachable

from .utils import _always_true, variant_dispatch, variant_dispatch_raises


# ---------------------------------------------------------------------------
# DataType trait and PrimitiveType sub-trait
# ---------------------------------------------------------------------------


trait DataType(Copyable, Equatable, ImplicitlyDestructible, Movable, Writable):
    def to_any(deinit self) -> AnyDataType:
        ...


trait PrimitiveType(DataType, Defaultable, TrivialRegisterPassable):
    comptime native: DType

    def __init__(out self):
        ...

    def byte_width(self) -> Int:
        return size_of[Self.native]()

    def bit_width(self) -> Int:
        return bit_width_of[Self.native]()


trait TemporalType(DataType):
    """Trait for Arrow temporal types (date32, date64, time32, time64, timestamp, duration).

    Extends DataType with a compile-time native DType for physical storage
    (int32 or int64). Not a sub-trait of PrimitiveType.
    """

    comptime native: DType


# ---------------------------------------------------------------------------
# Concrete zero-size Arrow type structs
# ---------------------------------------------------------------------------


struct NullType(DataType, Defaultable, TrivialRegisterPassable):
    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("null")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct BoolType(DataType, Defaultable, TrivialRegisterPassable):
    comptime native: DType = DType.bool

    def __init__(out self):
        pass

    def byte_width(self) -> Int:
        return 0

    def bit_width(self) -> Int:
        return 1

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("bool")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct Int8Type(PrimitiveType):
    comptime native: DType = DType.int8

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("int8")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct Int16Type(PrimitiveType):
    comptime native: DType = DType.int16

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("int16")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct Int32Type(PrimitiveType):
    comptime native: DType = DType.int32

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("int32")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct Int64Type(PrimitiveType):
    comptime native: DType = DType.int64

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("int64")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct UInt8Type(PrimitiveType):
    comptime native: DType = DType.uint8

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("uint8")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct UInt16Type(PrimitiveType):
    comptime native: DType = DType.uint16

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("uint16")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct UInt32Type(PrimitiveType):
    comptime native: DType = DType.uint32

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("uint32")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct UInt64Type(PrimitiveType):
    comptime native: DType = DType.uint64

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("uint64")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct Float32Type(PrimitiveType):
    comptime native: DType = DType.float32

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("float32")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct Float64Type(PrimitiveType):
    comptime native: DType = DType.float64

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("float64")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct Float16Type(PrimitiveType):
    comptime native: DType = DType.float16

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("float16")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct BinaryType(DataType, Defaultable, TrivialRegisterPassable):
    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("binary")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct StringType(DataType, Defaultable, TrivialRegisterPassable):
    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("string")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct FixedSizeBinaryType(DataType, TrivialRegisterPassable):
    """Fixed-size binary type — every element is exactly `byte_width` bytes."""

    var byte_width: Int

    def __init__(out self, byte_width: Int):
        self.byte_width = byte_width

    def __eq__(self, other: Self) -> Bool:
        return self.byte_width == other.byte_width

    def write_to[W: Writer](self, mut writer: W):
        writer.write("fixed_size_binary[", self.byte_width, "]")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


# ---------------------------------------------------------------------------
# Temporal types — date, time, timestamp, duration
# ---------------------------------------------------------------------------


struct TimeUnit(Copyable, Equatable, Movable, TrivialRegisterPassable, Writable):
    """Unit for time-based Arrow types (SECOND=0, MILLISECOND=1, MICROSECOND=2, NANOSECOND=3)."""

    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def to_string(self) -> String:
        if self.value == 0:
            return "s"
        elif self.value == 1:
            return "ms"
        elif self.value == 2:
            return "us"
        else:
            return "ns"

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.to_string())


comptime second = TimeUnit(0)
comptime millisecond = TimeUnit(1)
comptime microsecond = TimeUnit(2)
comptime nanosecond = TimeUnit(3)


struct Date32Type(TemporalType, Defaultable, TrivialRegisterPassable):
    """Date32 — days since Unix epoch (int32)."""

    comptime native: DType = DType.int32

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("date32")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct Date64Type(TemporalType, Defaultable, TrivialRegisterPassable):
    """Date64 — milliseconds since Unix epoch (int64)."""

    comptime native: DType = DType.int64

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def write_to[W: Writer](self, mut writer: W):
        writer.write("date64")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct Time32Type(TemporalType, TrivialRegisterPassable):
    """Time32 — seconds or milliseconds since midnight (int32)."""

    comptime native: DType = DType.int32

    var unit: TimeUnit

    def __init__(out self, unit: TimeUnit):
        self.unit = unit

    def __eq__(self, other: Self) -> Bool:
        return self.unit == other.unit

    def write_to[W: Writer](self, mut writer: W):
        writer.write("time32[", self.unit, "]")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct Time64Type(TemporalType, TrivialRegisterPassable):
    """Time64 — microseconds or nanoseconds since midnight (int64)."""

    comptime native: DType = DType.int64

    var unit: TimeUnit

    def __init__(out self, unit: TimeUnit):
        self.unit = unit

    def __eq__(self, other: Self) -> Bool:
        return self.unit == other.unit

    def write_to[W: Writer](self, mut writer: W):
        writer.write("time64[", self.unit, "]")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


struct TimestampType(TemporalType):
    """Timestamp — int64 elapsed units since Unix epoch, with optional timezone."""

    comptime native: DType = DType.int64

    var unit: TimeUnit
    var timezone: String

    def __init__(out self, unit: TimeUnit, timezone: String = ""):
        self.unit = unit
        self.timezone = timezone

    def __init__(out self, *, copy: Self):
        self.unit = copy.unit
        self.timezone = copy.timezone

    def __eq__(self, other: Self) -> Bool:
        return self.unit == other.unit and self.timezone == other.timezone

    def write_to[W: Writer](self, mut writer: W):
        writer.write("timestamp[", self.unit, "]")
        if self.timezone:
            writer.write("[tz=", self.timezone, "]")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self^)


struct DurationType(TemporalType, TrivialRegisterPassable):
    """Duration — elapsed int64 units, no epoch reference."""

    comptime native: DType = DType.int64

    var unit: TimeUnit

    def __init__(out self, unit: TimeUnit):
        self.unit = unit

    def __eq__(self, other: Self) -> Bool:
        return self.unit == other.unit

    def write_to[W: Writer](self, mut writer: W):
        writer.write("duration[", self.unit, "]")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self)


# ---------------------------------------------------------------------------
# Field and nested compound types
# ---------------------------------------------------------------------------


struct Field(
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Writable,
):
    var name: String
    var dtype: AnyDataType
    var nullable: Bool
    var metadata: Dict[String, String]

    def __init__(
        out self, name: String, var dtype: AnyDataType, nullable: Bool = True,var  metadata: Dict[String, String] = {}
    ):
        self.name = name
        self.dtype = dtype^
        self.nullable = nullable
        self.metadata = metadata^

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.dtype = copy.dtype.copy()
        self.nullable = copy.nullable
        self.metadata = copy.metadata.copy()

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Field]()[].copy()

    def __eq__(self, other: Self) -> Bool:
        return (
            self.name == other.name
            and self.dtype == other.dtype
            and self.nullable == other.nullable
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, ": ", self.dtype)

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write(
            "Field(name=", self.name, ", nullable=", self.nullable, ")"
        )

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)


struct ListType(DataType):
    var item: OwnedPointer[Field]

    def __init__(out self, var item: Field):
        self.item = OwnedPointer(item^)

    def __init__(out self, *, copy: Self):
        self.item = OwnedPointer(copy.item[].copy())

    def __eq__(self, other: Self) -> Bool:
        return self.item[] == other.item[]

    def value_field(ref self) -> ref[self.item] Field:
        return self.item[]

    def value_type(ref self) -> ref[self.item[].dtype] AnyDataType:
        return self.item[].dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("list<", self.item[].dtype, ">")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self^)


struct FixedSizeListType(DataType):
    var item: OwnedPointer[Field]
    var size: Int

    def __init__(out self, var item: Field, size: Int):
        self.item = OwnedPointer(item^)
        self.size = size

    def __init__(out self, *, copy: Self):
        self.item = OwnedPointer(copy.item[].copy())
        self.size = copy.size

    def __eq__(self, other: Self) -> Bool:
        return self.item[] == other.item[] and self.size == other.size

    def value_field(ref self) -> ref[self.item] Field:
        return self.item[]

    def value_type(self) -> AnyDataType:
        return self.item[].dtype.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("fixed_size_list<", self.item[], ">")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self^)


struct StructType(DataType):
    var fields: List[Field]

    def __init__(out self, var fields: List[Field]):
        self.fields = fields^

    def __init__(out self, *, copy: Self):
        self.fields = copy.fields.copy()

    def __eq__(self, other: Self) -> Bool:
        return self.fields == other.fields

    def write_to[W: Writer](self, mut writer: W):
        writer.write("struct<")
        for i in range(len(self.fields)):
            if i > 0:
                writer.write(", ")
            writer.write(self.fields[i])
        writer.write(">")

    def to_any(deinit self) -> AnyDataType:
        return AnyDataType(self^)


# ---------------------------------------------------------------------------
# AnyDataType — Variant-based type-erased handle
# ---------------------------------------------------------------------------


struct AnyDataType(
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    Equatable,
    Movable,
    Writable,
):
    comptime VariantType = Variant[
        NullType,
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
        BinaryType,
        StringType,
        ListType,
        FixedSizeListType,
        FixedSizeBinaryType,
        StructType,
        Date32Type,
        Date64Type,
        Time32Type,
        Time64Type,
        TimestampType,
        DurationType,
    ]

    var _v: Self.VariantType

    @implicit
    def __init__[T: DataType](out self, var value: T):
        self._v = Self.VariantType(value^)

    def __init__(out self, *, copy: Self):
        self._v = Self.VariantType(copy=copy._v)

    def __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema

        # Try downcasting from a marrow Python object.
        try:
            self = py.downcast_value_ptr[Self]()[].copy()
            return
        except:
            pass
        # Fall back to the Arrow C Schema Interface for foreign objects.
        var capsule: PythonObject
        try:
            capsule = py.__arrow_c_schema__()
        except:
            raise Error("cannot convert Python object to AnyDataType")
        self = CArrowSchema.from_pycapsule(capsule).to_dtype()

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def byte_width(self) raises -> Int:
        if not self.is_primitive():
            raise Error("byte_width is only defined for primitive types")

        comptime IsPrimitive[T: Movable] = conforms_to(T, PrimitiveType)

        @parameter
        def f[T: PrimitiveType](t: T) -> Int:
            return t.byte_width()

        return variant_dispatch[PrimitiveType, predicate=IsPrimitive, func=f](
            self._v
        )

    # --- convenience predicates ---

    def is_bool(self) -> Bool:
        return self._v.isa[BoolType]()

    def is_signed_integer(self) -> Bool:
        return (
            self._v.isa[Int8Type]()
            or self._v.isa[Int16Type]()
            or self._v.isa[Int32Type]()
            or self._v.isa[Int64Type]()
        )

    def is_unsigned_integer(self) -> Bool:
        return (
            self._v.isa[UInt8Type]()
            or self._v.isa[UInt16Type]()
            or self._v.isa[UInt32Type]()
            or self._v.isa[UInt64Type]()
        )

    def is_integer(self) -> Bool:
        return self.is_signed_integer() or self.is_unsigned_integer()

    def is_floating_point(self) -> Bool:
        return (
            self._v.isa[Float16Type]()
            or self._v.isa[Float32Type]()
            or self._v.isa[Float64Type]()
        )

    def is_numeric(self) -> Bool:
        return self.is_integer() or self.is_floating_point()

    def is_primitive(self) -> Bool:
        return self.is_bool() or self.is_numeric()

    def is_string(self) -> Bool:
        return self._v.isa[StringType]()

    def is_null(self) -> Bool:
        return self._v.isa[NullType]()

    def is_binary(self) -> Bool:
        return self._v.isa[BinaryType]()

    def is_list(self) -> Bool:
        return self._v.isa[ListType]()

    def is_fixed_size_list(self) -> Bool:
        return self._v.isa[FixedSizeListType]()

    def is_fixed_size_binary(self) -> Bool:
        return self._v.isa[FixedSizeBinaryType]()

    def is_struct(self) -> Bool:
        return self._v.isa[StructType]()

    def is_date32(self) -> Bool:
        return self._v.isa[Date32Type]()

    def is_date64(self) -> Bool:
        return self._v.isa[Date64Type]()

    def is_time32(self) -> Bool:
        return self._v.isa[Time32Type]()

    def is_time64(self) -> Bool:
        return self._v.isa[Time64Type]()

    def is_timestamp(self) -> Bool:
        return self._v.isa[TimestampType]()

    def is_duration(self) -> Bool:
        return self._v.isa[DurationType]()

    def is_temporal(self) -> Bool:
        return (
            self.is_date32()
            or self.is_date64()
            or self.is_time32()
            or self.is_time64()
            or self.is_timestamp()
            or self.is_duration()
        )

    def temporal_bit_width(self) -> Int:
        """Physical bit width of the temporal type's integer storage."""
        if self.is_date32() or self.is_time32():
            return 32
        else:
            return 64

    def is_fixed_size(self) -> Bool:
        return self.is_primitive()

    def write_to[W: Writer](self, mut writer: W):
        @parameter
        def f[T: DataType](t: T):
            t.write_to(writer)

        variant_dispatch[DataType, func=f](self._v)

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def __eq__(self, other: Self) -> Bool:
        return self._v == other._v

    def __is__(self, other: Self) -> Bool:
        return self == other

    # --- structural layout (Arrow IPC buffer model) ---

    def n_data_buffers(self) -> Int:
        """Number of flat data buffers for this type (excludes validity bitmap)."""
        if self.is_string() or self.is_binary():
            return 2
        elif (
            self.is_bool()
            or self.is_primitive()
            or self.is_list()
            or self.is_fixed_size_binary()
            or self.is_temporal()
        ):
            return 1
        else:
            return 0

    def child_dtypes(self) -> List[AnyDataType]:
        """Ordered list of child types (for list, fixed-size-list, and struct)."""
        var result = List[AnyDataType]()
        if self.is_list():
            result.append(self.as_list_type().value_type().copy())
        elif self.is_fixed_size_list():
            result.append(self.as_fixed_size_list_type().value_type())
        elif self.is_struct():
            var st = self.as_struct_type()
            for i in range(len(st.fields)):
                result.append(st.fields[i].dtype.copy())
        return result^

    # --- compound type accessors ---

    def as_list_type(self) -> ListType:
        """For list types, returns the inner ListType."""
        return ListType(copy=self._v[ListType])

    def as_fixed_size_list_type(self) -> FixedSizeListType:
        """For fixed-size list types, returns the inner FixedSizeListType."""
        return FixedSizeListType(copy=self._v[FixedSizeListType])

    def as_struct_type(self) -> StructType:
        """For struct types, returns the inner StructType."""
        return StructType(copy=self._v[StructType])

    def as_fixed_size_binary_type(self) -> FixedSizeBinaryType:
        """For fixed-size binary types, returns the inner FixedSizeBinaryType."""
        return self._v[FixedSizeBinaryType]

    def as_time32_type(self) -> Time32Type:
        return self._v[Time32Type]

    def as_time64_type(self) -> Time64Type:
        return self._v[Time64Type]

    def as_timestamp_type(self) -> TimestampType:
        return TimestampType(copy=self._v[TimestampType])

    def as_duration_type(self) -> DurationType:
        return self._v[DurationType]


# ---------------------------------------------------------------------------
# Field constructor and factory functions
# ---------------------------------------------------------------------------


def field(name: String, var dtype: AnyDataType, nullable: Bool = True) -> Field:
    """Construct a Field. Equivalent to PyArrow's ``pa.field()``."""
    return Field(name, dtype^, nullable)


def list_(var value_type: AnyDataType) -> ListType:
    """Construct a list type. Equivalent to PyArrow's ``pa.list_()``."""
    return ListType(field("item", value_type^))


def fixed_size_list_(
    var value_type: AnyDataType, size: Int
) -> FixedSizeListType:
    """Construct a fixed-size list type. Equivalent to PyArrow's ``pa.list_()`` with list_size.
    """
    return FixedSizeListType(field("item", value_type^), size)


def fixed_size_binary_(byte_width: Int) -> FixedSizeBinaryType:
    """Construct a fixed-size binary type. Equivalent to PyArrow's ``pa.binary(byte_width)``.
    """
    return FixedSizeBinaryType(byte_width)


def date32() -> Date32Type:
    """Construct a date32 type. Equivalent to PyArrow's ``pa.date32()``."""
    return Date32Type()


def date64() -> Date64Type:
    """Construct a date64 type. Equivalent to PyArrow's ``pa.date64()``."""
    return Date64Type()


def time32(unit: TimeUnit) -> Time32Type:
    """Construct a time32 type. Equivalent to PyArrow's ``pa.time32(unit)``."""
    return Time32Type(unit)


def time64(unit: TimeUnit) -> Time64Type:
    """Construct a time64 type. Equivalent to PyArrow's ``pa.time64(unit)``."""
    return Time64Type(unit)


def timestamp(unit: TimeUnit, timezone: String = "") -> TimestampType:
    """Construct a timestamp type. Equivalent to PyArrow's ``pa.timestamp(unit, tz)``."""
    return TimestampType(unit, timezone)


def duration(unit: TimeUnit) -> DurationType:
    """Construct a duration type. Equivalent to PyArrow's ``pa.duration(unit)``."""
    return DurationType(unit)


def struct_(var fields: List[Field]) -> StructType:
    """Construct a struct type from a list of fields."""
    return StructType(fields^)


def struct_(var *fields: Field) -> StructType:
    """Construct a struct type from variadic fields."""
    var list = List[Field]()
    for field in fields:
        list.append(field.copy())
    return StructType(list^)


# ---------------------------------------------------------------------------
# Comptime singletons
# ---------------------------------------------------------------------------


comptime null = NullType()
comptime bool_ = BoolType()
comptime int8 = Int8Type()
comptime int16 = Int16Type()
comptime int32 = Int32Type()
comptime int64 = Int64Type()
comptime uint8 = UInt8Type()
comptime uint16 = UInt16Type()
comptime uint32 = UInt32Type()
comptime uint64 = UInt64Type()
comptime float16 = Float16Type()
comptime float32 = Float32Type()
comptime float64 = Float64Type()
comptime binary = BinaryType()
comptime string = StringType()
