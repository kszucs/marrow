"""Arrow data type system — Variant-based implementation.

`DataType` is the trait that all concrete Arrow type structs implement.
`PrimitiveType` is a sub-trait for all fixed-width, buffer-backed types. It
provides `comptime native: DType` — the physical storage type used for buffer
reads and SIMD operations.

Trait hierarchy:
    DataType
    └── PrimitiveType(DataType)        comptime native: DType
        ├── NumericType(PrimitiveType) + TrivialRegisterPassable, Defaultable
        │   ├── IntegerType(NumericType)
        │   └── FloatingType(NumericType)
        ├── TemporalType(PrimitiveType)
        ├── IntervalType(PrimitiveType)
        └── DecimalType(PrimitiveType)

`DynType` is the type-erased runtime container backed by a `Variant` — no
heap allocation, no vtable, direct member access.

Concrete zero-size type structs (one per Arrow type):
    NullType, BoolType,
    Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    Float16Type, Float32Type, Float64Type,
    BinaryLikeType (trait), StringLikeType (trait), BinaryType, LargeBinaryType, StringType, LargeStringType,
    ListType, FixedSizeListType, FixedSizeBinaryType, StructType, DictionaryType,
    Date32Type, Date64Type, Time32Type, Time64Type, TimestampType, DurationType,
    YearMonthIntervalType, DayTimeIntervalType, MonthDayNanoIntervalType,
    Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type

Comptime singletons (same names as before):
    null, bool_, int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64, binary, string

"""

from std.utils import Variant

from std.sys import size_of, bit_width_of

from std.memory import OwnedPointer
from std.python import PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython


from .utils import variant_dispatch, variant_dispatch_raises


# ---------------------------------------------------------------------------
# DataType trait and PrimitiveType sub-trait
# ---------------------------------------------------------------------------


trait DataType(Copyable, Equatable, ImplicitlyDeletable, Movable, Writable):
    """A concrete Arrow type.

    Deliberately minimal: five inherited traits and one defaulted method, and
    **no associated types**. It used to name companion `ScalarType`/`ArrayType`
    members, which forced this module to import `arrays` and `scalars` — both of
    which import it back. That cycle broke the build three times in one day and
    the members were removed in `63b93aa`; nothing consumed them. A dtype's
    companion types are reached from the *array* side instead.

    Because the trait requires so little, `DynType` can satisfy it — the erased
    dtype is a **peer** of the concrete ones, not a supertype, and generic code
    bound on `DataType` accepts either."""

    def to_dyn(deinit self) -> DynType:
        return DynType(self^)


trait PrimitiveType(DataType, ImplicitlyCopyable):
    """Base trait for all fixed-width, buffer-backed Arrow types.

    Provides `comptime native: DType` — the physical storage type for buffer
    reads and SIMD operations.
    """

    comptime native: DType

    def byte_width(self) -> Int:
        return size_of[Self.native]()

    def bit_width(self) -> Int:
        return bit_width_of[Self.native]()


trait NumericType(Defaultable, PrimitiveType):
    """Integers, unsigned integers, and floats — zero-sized register-passable markers.
    """

    pass


trait IntegerType(NumericType):
    """Signed and unsigned integer types."""

    pass


trait BinaryLikeType(DataType, Defaultable, ImplicitlyCopyable):
    """Variable-width binary-like types: binary, large_binary, string, large_string.

    Provides `comptime offset: DType` — the physical integer type of the
    offset buffer (int32 for standard variants, int64 for large variants).
    """

    comptime offset: DType


trait StringLikeType(BinaryLikeType):
    """Sub-trait of BinaryLikeType for UTF-8 text types (string, large_string).

    Kernels that require valid UTF-8 (e.g. to_lowercase, regex) constrain on
    StringLikeType; byte-level operations constrain on BinaryLikeType and
    accept all four variants.
    """

    pass


trait ListLikeType(DataType):
    """Variable-length list types (list, large_list, map).

    Provides `comptime offset: DType` — the physical integer type of the
    offset buffer (int32 for standard list, int64 for large_list).
    """

    comptime offset: DType


trait FloatingType(NumericType):
    """Floating-point types (float16, float32, float64)."""

    pass


trait TemporalType(PrimitiveType):
    """Date, time, duration, and timestamp types.

    Physical storage is int32 or int64. Logical type carries a time unit and
    (for timestamps) an optional timezone — runtime values in `array.dtype`.
    """

    pass


trait IntervalType(Defaultable, PrimitiveType):
    """Arrow interval types (year-month, day-time, month-day-nano).

    Physical storage is int32, int64, or int128 depending on the variant.
    """

    pass


trait DecimalType(PrimitiveType):
    """Fixed-point decimal types backed by int32, int64, int128, or int256.

    Logical type carries precision and scale — runtime values in `array.dtype`.
    Traits cannot require `var` fields, so reading `precision`/`scale` off an
    erased dtype still needs a `DynType` ladder; `dispatch_decimal` covers the
    cases that only need `T.native`.
    """

    pass


# ---------------------------------------------------------------------------
# Concrete zero-size Arrow type structs
# ---------------------------------------------------------------------------


struct NullType(DataType, ImplicitlyCopyable):
    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("null")


struct BoolType(DataType, ImplicitlyCopyable):
    comptime native: DType = DType.bool

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("bool")


struct _IntegerType[T: DType](IntegerType):
    comptime native = Self.T

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.T)


struct _FloatingType[T: DType](FloatingType):
    comptime native = Self.T

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.T)


struct _DecimalType[T: DType](DecimalType):
    comptime native = Self.T

    var precision: Int
    var scale: Int

    def __init__(out self, precision: Int, scale: Int = 0):
        self.precision = precision
        self.scale = scale

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "decimal",
            bit_width_of[Self.native](),
            "[",
            self.precision,
            ", ",
            self.scale,
            "]",
        )


comptime Int8Type = _IntegerType[DType.int8]
comptime Int16Type = _IntegerType[DType.int16]
comptime Int32Type = _IntegerType[DType.int32]
comptime Int64Type = _IntegerType[DType.int64]
comptime UInt8Type = _IntegerType[DType.uint8]
comptime UInt16Type = _IntegerType[DType.uint16]
comptime UInt32Type = _IntegerType[DType.uint32]
comptime UInt64Type = _IntegerType[DType.uint64]

comptime Float16Type = _FloatingType[DType.float16]
comptime Float32Type = _FloatingType[DType.float32]
comptime Float64Type = _FloatingType[DType.float64]

comptime Decimal32Type = _DecimalType[DType.int32]
comptime Decimal64Type = _DecimalType[DType.int64]
comptime Decimal128Type = _DecimalType[DType.int128]
comptime Decimal256Type = _DecimalType[DType.int256]


struct BinaryType(BinaryLikeType):
    comptime offset: DType = DType.int32

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("binary")


struct LargeBinaryType(BinaryLikeType):
    comptime offset: DType = DType.int64

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("large_binary")


struct StringType(StringLikeType):
    comptime offset: DType = DType.int32

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("string")


struct LargeStringType(StringLikeType):
    comptime offset: DType = DType.int64

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("large_string")


struct FixedSizeBinaryType(DataType, ImplicitlyCopyable):
    """Fixed-size binary type — every element is exactly `byte_width` bytes."""

    var byte_width: Int

    def __init__(out self, byte_width: Int):
        self.byte_width = byte_width

    def write_to[W: Writer](self, mut writer: W):
        writer.write("fixed_size_binary[", self.byte_width, "]")


# ---------------------------------------------------------------------------
# Temporal types — date, time, timestamp, duration
# ---------------------------------------------------------------------------


struct TimeUnit(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Unit for time-based Arrow types (SECOND=0, MILLISECOND=1, MICROSECOND=2, NANOSECOND=3).
    """

    var value: Int

    def __init__(out self, value: Int):
        self.value = value

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


struct Date32Type(TemporalType):
    """Date32 — days since Unix epoch (int32)."""

    comptime native: DType = DType.int32

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("date32")


struct Date64Type(TemporalType):
    """Date64 — milliseconds since Unix epoch (int64)."""

    comptime native: DType = DType.int64

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("date64")


struct Time32Type(TemporalType):
    """Time32 — seconds or milliseconds since midnight (int32)."""

    comptime native: DType = DType.int32

    var unit: TimeUnit

    def __init__(out self, unit: TimeUnit):
        self.unit = unit

    def write_to[W: Writer](self, mut writer: W):
        writer.write("time32[", self.unit, "]")


struct Time64Type(TemporalType):
    """Time64 — microseconds or nanoseconds since midnight (int64)."""

    comptime native: DType = DType.int64

    var unit: TimeUnit

    def __init__(out self, unit: TimeUnit):
        self.unit = unit

    def write_to[W: Writer](self, mut writer: W):
        writer.write("time64[", self.unit, "]")


struct TimestampType(TemporalType):
    """Timestamp — int64 elapsed units since Unix epoch, with optional timezone.
    """

    comptime native: DType = DType.int64

    var unit: TimeUnit
    var timezone: String

    def __init__(out self, unit: TimeUnit, timezone: String = ""):
        self.unit = unit
        self.timezone = timezone

    def __init__(out self, *, copy: Self):
        self.unit = copy.unit
        self.timezone = copy.timezone

    def write_to[W: Writer](self, mut writer: W):
        writer.write("timestamp[", self.unit, "]")
        if self.timezone:
            writer.write("[tz=", self.timezone, "]")


struct DurationType(TemporalType):
    """Duration — elapsed int64 units, no epoch reference."""

    comptime native: DType = DType.int64

    var unit: TimeUnit

    def __init__(out self, unit: TimeUnit):
        self.unit = unit

    def write_to[W: Writer](self, mut writer: W):
        writer.write("duration[", self.unit, "]")


struct YearMonthIntervalType(IntervalType):
    """Year/month interval — number of months as int32."""

    comptime native: DType = DType.int32

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("month_interval")


struct DayTimeIntervalType(IntervalType):
    """Day/time interval — {days: int32, milliseconds: int32} packed into int64.
    """

    comptime native: DType = DType.int64

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("day_time_interval")


struct MonthDayNanoIntervalType(IntervalType):
    """Month/day/nanosecond interval — {months: int32, days: int32, nanos: int64} in 16 bytes.
    """

    comptime native: DType = DType.int128

    def __init__(out self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write("month_day_nano_interval")


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
    var dtype: DynType
    var nullable: Bool
    var metadata: Dict[String, String]

    def __init__(
        out self,
        name: String,
        var dtype: DynType,
        nullable: Bool = True,
        var metadata: Dict[String, String] = {},
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
            and self.metadata == other.metadata
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, ": ", self.dtype)

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write(
            "Field(name=", self.name, ", nullable=", self.nullable, ")"
        )

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)


struct ListType(DataType, ListLikeType):
    comptime offset: DType = DType.int32

    var item: OwnedPointer[Field]

    def __init__(out self, var item: Field):
        self.item = OwnedPointer(item^)

    def __init__(out self, *, copy: Self):
        self.item = OwnedPointer(copy.item[].copy())

    def __eq__(self, other: Self) -> Bool:
        return self.item[] == other.item[]

    def value_field(ref self) -> ref[self.item[]] Field:
        return self.item[]

    def value_type(ref self) -> ref[self.item[].dtype] DynType:
        return self.item[].dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("list<", self.item[].dtype, ">")


struct LargeListType(DataType, ListLikeType):
    comptime offset: DType = DType.int64

    var item: OwnedPointer[Field]

    def __init__(out self, var item: Field):
        self.item = OwnedPointer(item^)

    def __init__(out self, *, copy: Self):
        self.item = OwnedPointer(copy.item[].copy())

    def __eq__(self, other: Self) -> Bool:
        return self.item[] == other.item[]

    def value_field(ref self) -> ref[self.item[]] Field:
        return self.item[]

    def value_type(ref self) -> ref[self.item[].dtype] DynType:
        return self.item[].dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("large_list<", self.item[].dtype, ">")


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

    def value_field(ref self) -> ref[self.item[]] Field:
        return self.item[]

    def value_type(self) -> DynType:
        return self.item[].dtype.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("fixed_size_list<", self.item[], ">")


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


struct MapType(DataType, ListLikeType):
    """Arrow map type — an ordered sequence of (key, value) entries per element.

    Physically a list of a non-nullable 2-field ``entries`` struct (a required
    ``key`` and a ``value``); the int32 offsets let `MapArray` reuse
    `ListLikeArray[MapType]`. Following arrow-rs (`DataType::Map(field, sorted)`)
    and Arrow C++ (`MapType::value_field()`), the entries struct is stored as a
    single `Field` — preserving the key/value field names, nullability, and
    metadata — with `key_field()`/`item_field()` derived from it. `keys_sorted`
    is advisory. Equivalent to PyArrow's ``pa.map_(key_type, item_type,
    keys_sorted)``. Accessors return by value so a map-dtype temporary can be
    read safely.
    """

    comptime offset: DType = DType.int32

    var entries: OwnedPointer[Field]
    var keys_sorted: Bool

    def __init__(out self, var entries: Field, keys_sorted: Bool = False):
        """From the entries struct field directly (preserves its field names).
        """
        self.entries = OwnedPointer(entries^)
        self.keys_sorted = keys_sorted

    def __init__(
        out self,
        var key_type: DynType,
        var value_type: DynType,
        value_nullable: Bool = True,
        keys_sorted: Bool = False,
    ):
        """From key/value types, with the canonical ``entries``/``key``/``value``
        field names and a non-nullable key."""
        var entries = field(
            "entries",
            struct_(
                field("key", key_type^, nullable=False),
                field("value", value_type^, nullable=value_nullable),
            ),
            nullable=False,
        )
        self.entries = OwnedPointer(entries^)
        self.keys_sorted = keys_sorted

    def __init__(out self, *, copy: Self):
        self.entries = OwnedPointer(copy.entries[].copy())
        self.keys_sorted = copy.keys_sorted

    def __eq__(self, other: Self) -> Bool:
        return (
            self.entries[] == other.entries[]
            and self.keys_sorted == other.keys_sorted
        )

    def entries_field(self) -> Field:
        """The list-element field — the non-nullable ``entries`` struct of
        (key, value). This is the single ``+m`` child in the C Data Interface and
        the parquet ``key_value`` group."""
        return self.entries[].copy()

    def key_field(self) -> Field:
        return self.entries[].dtype.as_struct().fields[0].copy()

    def item_field(self) -> Field:
        return self.entries[].dtype.as_struct().fields[1].copy()

    def key_type(self) -> DynType:
        return self.entries[].dtype.as_struct().fields[0].dtype.copy()

    def item_type(self) -> DynType:
        return self.entries[].dtype.as_struct().fields[1].dtype.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("map<", self.key_type(), ", ", self.item_type(), ">")


struct DictionaryType(DataType):
    """Dictionary-encoded type — indices into a separate dictionary array.

    Equivalent to PyArrow's ``pa.dictionary(index_type, value_type, ordered)``.
    The index type must be an integer type (int8/16/32/64, uint8/16/32/64).
    The value type (the dictionary) can be any Arrow type.
    """

    var _index_type: OwnedPointer[DynType]
    var _value_type: OwnedPointer[DynType]
    var ordered: Bool

    def __init__(
        out self,
        var index_type: DynType,
        var value_type: DynType,
        ordered: Bool = False,
    ) raises:
        if not index_type.is_integer():
            raise Error(
                "DictionaryType: index_type must be an integer type, got: ",
                index_type,
            )
        self._index_type = OwnedPointer(index_type^)
        self._value_type = OwnedPointer(value_type^)
        self.ordered = ordered

    def __init__(out self, *, copy: Self):
        self._index_type = OwnedPointer(copy._index_type[].copy())
        self._value_type = OwnedPointer(copy._value_type[].copy())
        self.ordered = copy.ordered

    # Explicit (empty) destructor: `OwnedPointer[DynType]` is not implicitly
    # deletable, so the compiler cannot synthesize one. Fields are still
    # destroyed automatically after the body runs.
    def __del__(deinit self):
        pass

    def index_type(ref self) -> ref[self._index_type[]] DynType:
        return self._index_type[]

    def value_type(ref self) -> ref[self._value_type[]] DynType:
        return self._value_type[]

    def __eq__(self, other: Self) -> Bool:
        return (
            self._index_type[] == other._index_type[]
            and self._value_type[] == other._value_type[]
            and self.ordered == other.ordered
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "dictionary<values=",
            self._value_type[],
            ", indices=",
            self._index_type[],
            ", ordered=",
            Int(self.ordered),
            ">",
        )


# ---------------------------------------------------------------------------
# DynType — Variant-based type-erased handle
# ---------------------------------------------------------------------------


struct DynType(
    ConvertibleFromPython,
    ConvertibleToPython,
    Copyable,
    DataType,
    Equatable,
    Movable,
    Writable,
):
    comptime offset = DType.int32
    """Placeholder, never read — see `native`.

    Required by `BinaryLikeType`/`ListLikeType`, which `StringValue.OutType` is
    bound on. An erased dtype does not know its offset width; the erased arm
    resolves it at run time via `dispatch_binarylike`/`dispatch_listlike`."""

    # `to_dyn` is overridden below rather than inherited, and that override is
    # load-bearing: the trait's default body is `DynType(self^)`, which for
    # `Self = DynType` would be `DynType(DynType)` — and `DynType` is
    # deliberately not a member of its own variant.

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
        LargeBinaryType,
        StringType,
        LargeStringType,
        ListType,
        LargeListType,
        FixedSizeListType,
        FixedSizeBinaryType,
        StructType,
        MapType,
        DictionaryType,
        Date32Type,
        Date64Type,
        Time32Type,
        Time64Type,
        TimestampType,
        DurationType,
        YearMonthIntervalType,
        DayTimeIntervalType,
        MonthDayNanoIntervalType,
        Decimal32Type,
        Decimal64Type,
        Decimal128Type,
        Decimal256Type,
    ]

    var _v: Self.VariantType

    @implicit
    def __init__[T: DataType](out self, var value: T):
        self._v = Self.VariantType(value^)

    def dispatch_primitive[
        R: AnyType,
        //,
        func: def[T: PrimitiveType](T) raises capturing[_] -> R,
    ](self) raises -> R:
        """Resolve any runtime fixed-width dtype to its comptime type and run `func`.

        The widest family — numeric, temporal, interval, and decimal. Prefer a
        narrower member where one fits: this one instantiates `func` for every
        fixed-width type in the variant.
        """
        return variant_dispatch_raises[PrimitiveType, func=func](self._v)

    def dispatch_numeric[
        R: AnyType,
        //,
        func: def[T: NumericType](T) raises capturing[_] -> R,
    ](self) raises -> R:
        """Resolve a runtime numeric dtype to its comptime type and run `func`.
        """
        return variant_dispatch_raises[NumericType, func=func](self._v)

    def dispatch_integer[
        R: AnyType,
        //,
        func: def[T: IntegerType](T) raises capturing[_] -> R,
    ](self) raises -> R:
        """Resolve a runtime integer dtype to its comptime type and run `func`.
        """
        return variant_dispatch_raises[IntegerType, func=func](self._v)

    def dispatch_floating[
        R: AnyType,
        //,
        func: def[T: FloatingType](T) raises capturing[_] -> R,
    ](self) raises -> R:
        """Resolve a runtime floating dtype to its comptime type and run `func`.
        """
        return variant_dispatch_raises[FloatingType, func=func](self._v)

    def dispatch_temporal[
        R: AnyType,
        //,
        func: def[T: TemporalType](T) raises capturing[_] -> R,
    ](self) raises -> R:
        """Resolve a runtime date/time/timestamp/duration dtype to its comptime type
            and run `func`.

            Only needed when the *logical* type matters (unit, timezone). Kernels that
            are value- or order-preserving should instead reinterpret the column through
            the typed leaf directly — every kernel leaf is bound on
        `PrimitiveType`, which the temporal types satisfy.
        """
        return variant_dispatch_raises[TemporalType, func=func](self._v)

    def dispatch_decimal[
        R: AnyType,
        //,
        func: def[T: DecimalType](T) raises capturing[_] -> R,
    ](self) raises -> R:
        """Resolve a runtime decimal dtype to its comptime type and run `func`.

        Only needed when the *logical* type matters (precision, scale) or when
        the backing integer width drives the code — `T.native`. A kernel that
        merely reads value bytes should go through `dispatch_primitive`, which
        the decimal types satisfy.
        """
        return variant_dispatch_raises[DecimalType, func=func](self._v)

    def dispatch_stringlike[
        R: AnyType,
        //,
        func: def[T: StringLikeType](T) raises capturing[_] -> R,
    ](self) raises -> R:
        """Resolve a runtime string-like dtype to its comptime type and run `func`.
        """
        return variant_dispatch_raises[StringLikeType, func=func](self._v)

    def dispatch_binarylike[
        R: AnyType,
        //,
        func: def[T: BinaryLikeType](T) raises capturing[_] -> R,
    ](self) raises -> R:
        """Resolve a runtime binary-like dtype to its comptime type and run `func`.
        """
        return variant_dispatch_raises[BinaryLikeType, func=func](self._v)

    def dispatch_listlike[
        R: AnyType,
        //,
        func: def[T: ListLikeType](T) raises capturing[_] -> R,
    ](self) raises -> R:
        """Resolve a runtime list/large_list/map dtype to its comptime type and run
        `func`."""
        return variant_dispatch_raises[ListLikeType, func=func](self._v)

    def to_dyn(deinit self) -> DynType:
        return self^

    def __init__(out self):
        """The Arrow `null` type — the identity `Defaultable` requires.

        `NumericType` extends `Defaultable`, so conforming needs this. `null` is
        the honest default for an erased dtype that has not been told what it
        holds: it is a real Arrow type meaning "no value", not a numeric width
        picked arbitrarily."""
        self._v = Self.VariantType(NullType())

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
            raise Error("cannot convert Python object to DynType")
        self = CArrowSchema.from_pycapsule(capsule).to_dtype()

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def byte_width(self) -> Int:
        """Physical byte width per element, or **0** if this is not a fixed-width
        type.

        This *overrides* `PrimitiveType.byte_width`, and the override is
        load-bearing: the inherited default is `size_of[Self.native]()`, and
        `DynType.native` is the `bool` placeholder, so without this every erased
        dtype would report width 1. It has to be **non-raising** to override at
        all — a `raises` signature does not match the trait's, so the default
        silently won instead and `DynType(int16).byte_width()` returned 1.
        `test_byte_width` is what caught that.

        Returning 0 rather than raising is the cost of the override. Every caller
        already establishes the type first and treats an unexpected width as an
        error — `TemporalCast.dispatch` and `DateTruncKernel.apply` both reject
        anything that is not 4 or 8 — so a 0 surfaces loudly there rather than
        being mistaken for a real width."""
        if not self.is_primitive():
            return 0

        @parameter
        def f[T: PrimitiveType](t: T) -> Int:
            return t.byte_width()

        return variant_dispatch[PrimitiveType, func=f](self._v)

    # --- convenience predicates ---

    def is_bool(self) -> Bool:
        return self._v.isa[BoolType]()

    def is_int8(self) -> Bool:
        return self._v.isa[Int8Type]()

    def is_int16(self) -> Bool:
        return self._v.isa[Int16Type]()

    def is_int32(self) -> Bool:
        return self._v.isa[Int32Type]()

    def is_int64(self) -> Bool:
        return self._v.isa[Int64Type]()

    def is_uint8(self) -> Bool:
        return self._v.isa[UInt8Type]()

    def is_uint16(self) -> Bool:
        return self._v.isa[UInt16Type]()

    def is_uint32(self) -> Bool:
        return self._v.isa[UInt32Type]()

    def is_uint64(self) -> Bool:
        return self._v.isa[UInt64Type]()

    def is_float16(self) -> Bool:
        return self._v.isa[Float16Type]()

    def is_float32(self) -> Bool:
        return self._v.isa[Float32Type]()

    def is_float64(self) -> Bool:
        return self._v.isa[Float64Type]()

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
        """True for the fixed-**byte**-width, buffer-backed types: numeric,
        temporal, interval, decimal.

        Exactly the set that conforms to the `PrimitiveType` trait, which is
        what this predicate is for — every caller uses it to guard a
        `variant_dispatch[PrimitiveType]`.

        **Deliberately narrower than PyArrow and Arrow C++**, which both count
        bool (`pa.types.is_primitive(pa.bool_())` is True; Arrow C++'s
        `is_primitive` lists `Type::BOOL` first). Neither has a `PrimitiveType`
        trait to stay consistent with; marrow does, and `BoolType` cannot
        conform to it because bool is bit-packed rather than fixed-byte-width.
        Answering True here made `byte_width()` abort. For PyArrow parity write
        `is_bool() or is_primitive()` — which is what `ipc.mojo` already does.
        There is deliberately no general fixed-width predicate: the one that
        existed (`is_fixed_size`) covered neither `fixed_size_binary` nor
        `fixed_size_list`, had no production caller, and was removed rather
        than left lying. Arrow C++ spells the real thing `is_fixed_width` =
        `is_primitive || is_dictionary || is_fixed_size_binary`; add that,
        against that definition, if something ever needs it.
        """
        return (
            self.is_numeric()
            or self.is_temporal()
            or self.is_interval()
            or self.is_decimal()
        )

    def is_string(self) -> Bool:
        return self._v.isa[StringType]()

    def is_large_string(self) -> Bool:
        return self._v.isa[LargeStringType]()

    def num_buffers(self) -> Int:
        """How many **data** buffers a value of this type owns.

        Validity is not counted. Arrow C++ and arrow-rs both put the validity
        bitmap at `buffers[0]`, but marrow's `ArrayData` carries `bitmap` as its
        own field — so these counts are one lower than the references' and there
        is no bitmap buffer kind to model.

        This is the one genuinely structural fact the codecs kept re-deriving:
        `ipc.mojo`'s buffer walk was a two-branch ladder asking exactly this, and
        `c_data`'s importer hard-codes the same numbers as `_need_buffers`
        arguments. `arrays.mojo` already enforced these counts in each typed
        constructor without ever saying them anywhere a codec could ask.

        Deliberately just a count. Both references also carry a per-buffer
        `BufferSpec` (kind plus byte width) — and neither drives its IPC or C-ABI
        codec from it; they use it for `ArrayData` validation. If a caller ever
        needs the widths, add them then; see the A3 card for why the wider
        version buys less than it looks.
        """
        if self.is_null() or self.is_struct() or self.is_fixed_size_list():
            # Nothing of their own: a struct and a fixed-size list are entirely
            # described by their children plus validity.
            return 0
        elif self.is_binary_like():
            return 2  # offsets + data
        elif self.is_list_like():
            return 1  # offsets; the values are the child
        elif self.is_dictionary():
            # The array holds indices; the values are a separate dictionary.
            # arrow-rs resolves this identically -- `layout(key_type)`.
            return self.as_dictionary().index_type().num_buffers()
        else:
            # bool, numeric, temporal, interval, decimal, fixed_size_binary:
            # one values buffer, bit-packed for bool and byte-addressed
            # otherwise.
            return 1

    def is_null(self) -> Bool:
        return self._v.isa[NullType]()

    def is_binary(self) -> Bool:
        return self._v.isa[BinaryType]()

    def is_large_binary(self) -> Bool:
        return self._v.isa[LargeBinaryType]()

    def is_string_like(self) -> Bool:
        """A UTF-8 string of either offset width.

        The narrower half of `is_binary_like`: the kernels that require *text*
        rather than any variable-width payload guard with this, so their
        diagnostic can name the family before `dispatch_stringlike` reduces the
        dtype to a comptime type.
        """
        return self.is_string() or self.is_large_string()

    def is_binary_like(self) -> Bool:
        return (
            self.is_binary() or self.is_large_binary() or self.is_string_like()
        )

    def is_list(self) -> Bool:
        return self._v.isa[ListType]()

    def is_large_list(self) -> Bool:
        return self._v.isa[LargeListType]()

    def is_list_like(self) -> Bool:
        # Map included: `MapType` conforms to `ListLikeType` and
        # `dispatch_listlike` walks it, so a predicate that excluded it forced
        # every caller to write `is_list_like() or is_map()` by hand.
        return self.is_list() or self.is_large_list() or self.is_map()

    def is_fixed_size_list(self) -> Bool:
        return self._v.isa[FixedSizeListType]()

    def is_fixed_size_binary(self) -> Bool:
        return self._v.isa[FixedSizeBinaryType]()

    def is_struct(self) -> Bool:
        return self._v.isa[StructType]()

    def is_map(self) -> Bool:
        return self._v.isa[MapType]()

    def is_dictionary(self) -> Bool:
        return self._v.isa[DictionaryType]()

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

    def is_year_month_interval(self) -> Bool:
        return self._v.isa[YearMonthIntervalType]()

    def is_day_time_interval(self) -> Bool:
        return self._v.isa[DayTimeIntervalType]()

    def is_month_day_nano_interval(self) -> Bool:
        return self._v.isa[MonthDayNanoIntervalType]()

    def is_interval(self) -> Bool:
        return (
            self.is_year_month_interval()
            or self.is_day_time_interval()
            or self.is_month_day_nano_interval()
        )

    def is_decimal32(self) -> Bool:
        return self._v.isa[Decimal32Type]()

    def is_decimal64(self) -> Bool:
        return self._v.isa[Decimal64Type]()

    def is_decimal128(self) -> Bool:
        return self._v.isa[Decimal128Type]()

    def is_decimal256(self) -> Bool:
        return self._v.isa[Decimal256Type]()

    def is_decimal(self) -> Bool:
        return (
            self.is_decimal32()
            or self.is_decimal64()
            or self.is_decimal128()
            or self.is_decimal256()
        )


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

    # --- compound type accessors ---

    def as_type[T: DataType](ref self) -> ref[self._v[T]] T:
        """This dtype as the concrete `T` it holds — a borrow, no copy.

        The named accessors below are one-liners over it; generic code (a
        `PrimitiveArray[T]` reading its own dtype) needs the parameterized form.
        """
        debug_assert(self._v.isa[T](), "as_type: wrong type, holds ", self)
        return self._v[T]

    def as_list(ref self) -> ref[self._v[ListType]] ListType:
        """For list types, returns the inner ListType."""
        return self.as_type[ListType]()

    def as_large_list(ref self) -> ref[self._v[LargeListType]] LargeListType:
        """For large_list types, returns the inner LargeListType."""
        return self.as_type[LargeListType]()

    def as_fixed_size_list(
        ref self,
    ) -> ref[self._v[FixedSizeListType]] FixedSizeListType:
        """For fixed-size list types, returns the inner FixedSizeListType."""
        return self.as_type[FixedSizeListType]()

    def as_struct(ref self) -> ref[self._v[StructType]] StructType:
        """For struct types, returns the inner StructType."""
        return self.as_type[StructType]()

    def as_map(ref self) -> ref[self._v[MapType]] MapType:
        """For map types, returns the inner MapType."""
        return self.as_type[MapType]()

    def as_dictionary(ref self) -> ref[self._v[DictionaryType]] DictionaryType:
        """For dictionary types, returns the inner DictionaryType."""
        return self.as_type[DictionaryType]()

    def as_fixed_size_binary(
        ref self,
    ) -> ref[self._v[FixedSizeBinaryType]] FixedSizeBinaryType:
        """For fixed-size binary types, returns the inner FixedSizeBinaryType.
        """
        return self.as_type[FixedSizeBinaryType]()

    def as_time32(ref self) -> ref[self._v[Time32Type]] Time32Type:
        return self.as_type[Time32Type]()

    def as_time64(ref self) -> ref[self._v[Time64Type]] Time64Type:
        return self.as_type[Time64Type]()

    def as_timestamp(ref self) -> ref[self._v[TimestampType]] TimestampType:
        return self.as_type[TimestampType]()

    def as_duration(ref self) -> ref[self._v[DurationType]] DurationType:
        return self.as_type[DurationType]()

    def as_year_month_interval(
        ref self,
    ) -> ref[self._v[YearMonthIntervalType]] YearMonthIntervalType:
        return self.as_type[YearMonthIntervalType]()

    def as_day_time_interval(
        ref self,
    ) -> ref[self._v[DayTimeIntervalType]] DayTimeIntervalType:
        return self.as_type[DayTimeIntervalType]()

    def as_month_day_nano_interval(
        ref self,
    ) -> ref[self._v[MonthDayNanoIntervalType]] MonthDayNanoIntervalType:
        return self.as_type[MonthDayNanoIntervalType]()

    def as_decimal32(ref self) -> ref[self._v[Decimal32Type]] Decimal32Type:
        return self.as_type[Decimal32Type]()

    def as_decimal64(ref self) -> ref[self._v[Decimal64Type]] Decimal64Type:
        return self.as_type[Decimal64Type]()

    def as_decimal128(ref self) -> ref[self._v[Decimal128Type]] Decimal128Type:
        return self.as_type[Decimal128Type]()

    def as_decimal256(ref self) -> ref[self._v[Decimal256Type]] Decimal256Type:
        return self.as_type[Decimal256Type]()


# ---------------------------------------------------------------------------
# Field constructor and factory functions
# ---------------------------------------------------------------------------


def field(name: String, var dtype: DynType, nullable: Bool = True) -> Field:
    """Construct a Field. Equivalent to PyArrow's ``pa.field()``."""
    return Field(name, dtype^, nullable)


def list_(var value_type: DynType) -> ListType:
    """Construct a list type. Equivalent to PyArrow's ``pa.list_()``."""
    return ListType(field("item", value_type^))


def large_list_(var value_type: DynType) -> LargeListType:
    """Construct a large_list type. Equivalent to PyArrow's ``pa.large_list()``.
    """
    return LargeListType(field("item", value_type^))


def fixed_size_list_(var value_type: DynType, size: Int) -> FixedSizeListType:
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
    """Construct a timestamp type. Equivalent to PyArrow's ``pa.timestamp(unit, tz)``.
    """
    return TimestampType(unit, timezone)


def duration(unit: TimeUnit) -> DurationType:
    """Construct a duration type. Equivalent to PyArrow's ``pa.duration(unit)``.
    """
    return DurationType(unit)


def year_month_interval() -> YearMonthIntervalType:
    """Construct a year-month interval type. Equivalent to PyArrow's ``pa.month_interval()``.
    """
    return YearMonthIntervalType()


def day_time_interval() -> DayTimeIntervalType:
    """Construct a day-time interval type. Equivalent to PyArrow's ``pa.day_time_interval()``.
    """
    return DayTimeIntervalType()


def month_day_nano_interval() -> MonthDayNanoIntervalType:
    """Construct a month-day-nanosecond interval type. Equivalent to PyArrow's ``pa.month_day_nano_interval()``.
    """
    return MonthDayNanoIntervalType()


def struct_(var fields: List[Field]) -> StructType:
    """Construct a struct type from a list of fields."""
    return StructType(fields^)


def dictionary(
    var index_type: DynType,
    var value_type: DynType,
    ordered: Bool = False,
) raises -> DictionaryType:
    """Construct a dictionary type. Equivalent to PyArrow's ``pa.dictionary()``.

    The index type must be an integer type (int8/16/32/64, uint8/16/32/64).
    """
    return DictionaryType(index_type^, value_type^, ordered)


def struct_(var *fields: Field) -> StructType:
    """Construct a struct type from variadic fields."""
    var list = List[Field]()
    for field in fields:
        list.append(field.copy())
    return StructType(list^)


def map_(
    var key_type: DynType,
    var item_type: DynType,
    keys_sorted: Bool = False,
) -> MapType:
    """Construct a map type. Equivalent to PyArrow's ``pa.map_(key_type,
    item_type, keys_sorted)``. The key is non-nullable; the value is nullable.
    """
    return MapType(
        key_type^, item_type^, value_nullable=True, keys_sorted=keys_sorted
    )


def decimal32(precision: Int, scale: Int = 0) -> Decimal32Type:
    """Construct a decimal32 type. Equivalent to PyArrow's ``pa.decimal32(precision, scale)``.
    """
    return Decimal32Type(precision, scale)


def decimal64(precision: Int, scale: Int = 0) -> Decimal64Type:
    """Construct a decimal64 type. Equivalent to PyArrow's ``pa.decimal64(precision, scale)``.
    """
    return Decimal64Type(precision, scale)


def decimal128(precision: Int, scale: Int = 0) -> Decimal128Type:
    """Construct a decimal128 type. Equivalent to PyArrow's ``pa.decimal128(precision, scale)``.
    """
    return Decimal128Type(precision, scale)


def decimal256(precision: Int, scale: Int = 0) -> Decimal256Type:
    """Construct a decimal256 type. Equivalent to PyArrow's ``pa.decimal256(precision, scale)``.
    """
    return Decimal256Type(precision, scale)


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
comptime large_binary = LargeBinaryType()
comptime string = StringType()
comptime large_string = LargeStringType()
