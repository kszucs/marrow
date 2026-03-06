from std.io.write import Writable, Writer
from std.sys import size_of

# The following enum codes are copied from the C++ implementation of Arrow

comptime NA: UInt8 = 0
"""A NULL type having no physical storage."""

comptime BOOL: UInt8 = 1
"""Boolean as 1 bit, LSB bit-packed ordering."""

comptime UINT8: UInt8 = 2
"""Unsigned 8-bit little-endian integer."""

comptime INT8: UInt8 = 3
"""Signed 8-bit little-endian integer."""

comptime UINT16: UInt8 = 4
"""Unsigned 16-bit little-endian integer."""

comptime INT16: UInt8 = 5
"""Signed 16-bit little-endian integer."""

comptime UINT32: UInt8 = 6
"""Unsigned 32-bit little-endian integer."""

comptime INT32: UInt8 = 7
"""Signed 32-bit little-endian integer."""

comptime UINT64: UInt8 = 8
"""Unsigned 64-bit little-endian integer."""

comptime INT64: UInt8 = 9
"""Signed 64-bit little-endian integer."""

comptime FLOAT16: UInt8 = 10
"""2-byte floating point value."""

comptime FLOAT32: UInt8 = 11
"""4-byte floating point value."""

comptime FLOAT64: UInt8 = 12
"""8-byte floating point value."""

comptime STRING: UInt8 = 13
"""UTF8 variable-length string as List<Char>."""

comptime BINARY: UInt8 = 14
"""Variable-length bytes (no guarantee of UTF8-ness)."""

comptime FIXED_SIZE_BINARY: UInt8 = 15
"""Fixed-size binary. Each value occupies the same number of bytes."""

comptime DATE32: UInt8 = 16
"""Type int32_t days since the UNIX epoch."""

comptime DATE64: UInt8 = 17
"""Type int64_t milliseconds since the UNIX epoch."""

comptime TIMESTAMP: UInt8 = 18
"""Exact timestamp encoded with int64 since UNIX epoch. Default unit millisecond."""

comptime TIME32: UInt8 = 19
"""Time as signed 32-bit integer, representing either seconds or milliseconds since midnight."""

comptime TIME64: UInt8 = 20
"""Time as signed 64-bit integer, representing either microseconds or nanoseconds since midnight."""

comptime INTERVAL_MONTHS: UInt8 = 21
"""YEAR_MONTH interval in SQL style."""

comptime INTERVAL_DAY_TIME: UInt8 = 22
"""DAY_TIME interval in SQL style."""

comptime DECIMAL128: UInt8 = 23
"""Precision- and scale-based decimal type with 128 bits."""

comptime DECIMAL: UInt8 = DECIMAL128
"""Defined for backward-compatibility."""

comptime DECIMAL256: UInt8 = 24
"""Precision- and scale-based decimal type with 256 bits."""

comptime LIST: UInt8 = 25
"""A list of some logical data type."""

comptime STRUCT: UInt8 = 26
"""Struct of logical types."""

comptime SPARSE_UNION: UInt8 = 27
"""Sparse unions of logical types."""

comptime DENSE_UNION: UInt8 = 28
"""Dense unions of logical types."""

comptime DICTIONARY: UInt8 = 29
"""Dictionary-encoded type, also called "categorical" or "factor"
in other programming languages. Holds the dictionary value
type but not the dictionary itself, which is part of the
ArrayData struct."""

comptime MAP: UInt8 = 30
"""Map, a repeated struct logical type."""

comptime EXTENSION: UInt8 = 31
"""Custom data type, implemented by user."""

comptime FIXED_SIZE_LIST: UInt8 = 32
"""Fixed size list of some logical type."""

comptime DURATION: UInt8 = 33
"""Measure of elapsed time in either seconds, milliseconds, microseconds or nanoseconds."""

comptime LARGE_STRING: UInt8 = 34
"""Like STRING, but with 64-bit offsets."""

comptime LARGE_BINARY: UInt8 = 35
"""Like BINARY, but with 64-bit offsets."""

comptime LARGE_LIST: UInt8 = 36
"""Like LIST, but with 64-bit offsets."""

comptime INTERVAL_MONTH_DAY_NANO: UInt8 = 37
"""Calendar interval type with three fields."""

comptime RUN_END_ENCODED: UInt8 = 38
"""Run-end encoded data."""

comptime STRING_VIEW: UInt8 = 39
"""String (UTF8) view type with 4-byte prefix and inline small string optimization."""

comptime BINARY_VIEW: UInt8 = 40
"""Bytes view type with 4-byte prefix and inline small string optimization."""

comptime LIST_VIEW: UInt8 = 41
"""A list of some logical data type represented by offset and size."""

comptime LARGE_LIST_VIEW: UInt8 = 42
"""Like LIST_VIEW, but with 64-bit offsets and sizes."""


struct Field(Copyable, Equatable, Writable):
    var name: String
    var dtype: DataType
    var nullable: Bool

    fn __init__(
        out self, name: String, var dtype: DataType, nullable: Bool = False
    ):
        self.name = name
        self.dtype = dtype^
        self.nullable = nullable

    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this Field to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write(t'Field(name="{self.name}", dtype={self.dtype}, nullable={self.nullable}, )')


struct DataType(Copyable, Equatable, Writable):
    var code: UInt8
    var native: DType
    var fields: List[Field]

    fn __init__(out self, *, code: UInt8):
        self.code = code
        self.native = DType.invalid
        self.fields = []

    fn __init__(out self, native: DType):
        if native == DType.bool:
            self.code = BOOL
        elif native == DType.int8:
            self.code = INT8
        elif native == DType.int16:
            self.code = INT16
        elif native == DType.int32:
            self.code = INT32
        elif native == DType.int64:
            self.code = INT64
        elif native == DType.uint8:
            self.code = UINT8
        elif native == DType.uint16:
            self.code = UINT16
        elif native == DType.uint32:
            self.code = UINT32
        elif native == DType.uint64:
            self.code = UINT64
        elif native == DType.float32:
            self.code = FLOAT32
        elif native == DType.float64:
            self.code = FLOAT64
        else:
            self.code = NA
        self.native = native
        self.fields = []

    fn __init__(out self, *, code: UInt8, native: DType):
        self.code = code
        self.native = native
        self.fields = []

    fn __init__(out self, *, code: UInt8, fields: List[Field]):
        self.code = code
        self.native = DType.invalid
        self.fields = fields.copy()

    fn __copyinit__(out self, copy: Self):
        self.code = copy.code
        self.native = copy.native
        self.fields = copy.fields.copy()

    fn __moveinit__(out self, deinit take: Self):
        self.code = take.code
        self.native = take.native
        self.fields = take.fields^

    fn __is__(self, other: DataType) -> Bool:
        return self == other

    fn __eq__(self, other: DataType) -> Bool:
        if self.code != other.code:
            return False
        if len(self.fields) != len(other.fields):
            return False
        for i in range(len(self.fields)):
            if self.fields[i] != other.fields[i]:
                return False
        return True

    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this DataType to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        name: String
        if self.code == NA:
            name ="null"
        elif self.code == BOOL:
            name ="bool"
        elif self.code == UINT8:
            name ="uint8"
        elif self.code == INT8:
            name ="int8"
        elif self.code == INT16:
            name ="int16"
        elif self.code == INT32:
            name ="int32"
        elif self.code == INT64:
            name ="int64"
        elif self.code == LIST:
            name ="list"
        elif self.code == STRUCT:
            name ="struct"
        else:
            name ="unknown " + String(self.code)

        writer.write(t"DataType(code={name})")

    fn write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


    fn is_bool(self) -> Bool:
        return self.code == BOOL

    fn bitwidth(self) -> UInt8:
        if self.code == BOOL:
            return 1
        elif self.code == INT8:
            return 8
        elif self.code == INT16:
            return 16
        elif self.code == INT32:
            return 32
        elif self.code == INT64:
            return 64
        elif self.code == UINT8:
            return 8
        elif self.code == UINT16:
            return 16
        elif self.code == UINT32:
            return 32
        elif self.code == UINT64:
            return 64
        elif self.code == FLOAT32:
            return 32
        elif self.code == FLOAT64:
            return 64
        else:
            return 0

    @always_inline
    fn is_boolean(self) -> Bool:
        return self.code == BOOL

    @always_inline
    fn is_fixed_size(self) -> Bool:
        return self.bitwidth() > 0

    @always_inline
    fn is_integer(self) -> Bool:
        return self.code in [
            INT8,
            INT16,
            INT32,
            INT64,
            UINT8,
            UINT16,
            UINT32,
            UINT64,
        ]

    @always_inline
    fn is_signed_integer(self) -> Bool:
        return self.code in [INT8, INT16, INT32, INT64]

    @always_inline
    fn is_unsigned_integer(self) -> Bool:
        return self.code in [
            UINT8,
            UINT16,
            UINT32,
            UINT64,
        ]

    @always_inline
    fn is_floating_point(self) -> Bool:
        return self.code in [FLOAT32, FLOAT64]

    @always_inline
    fn is_numeric(self) -> Bool:
        return self.is_integer() or self.is_floating_point()

    @always_inline
    fn is_string(self) -> Bool:
        return self.code == STRING

    @always_inline
    fn is_list(self) -> Bool:
        return self.code == LIST

    @always_inline
    fn is_struct(self) -> Bool:
        return self.code == STRUCT


fn list_(var value_type: DataType) -> DataType:
    return DataType(code=LIST, fields=[Field("value", value_type^)])


fn struct_(fields: List[Field]) -> DataType:
    return DataType(code=STRUCT, fields=fields)


fn struct_(var *fields: Field) -> DataType:
    return DataType(code=STRUCT, fields=List(elements=fields^))


comptime null = DataType(code=NA)
comptime bool_ = DataType(code=BOOL, native=DType.bool)
comptime int8 = DataType(code=INT8, native=DType.int8)
comptime int16 = DataType(code=INT16, native=DType.int16)
comptime int32 = DataType(code=INT32, native=DType.int32)
comptime int64 = DataType(code=INT64, native=DType.int64)
comptime uint8 = DataType(code=UINT8, native=DType.uint8)
comptime uint16 = DataType(code=UINT16, native=DType.uint16)
comptime uint32 = DataType(code=UINT32, native=DType.uint32)
comptime uint64 = DataType(code=UINT64, native=DType.uint64)
comptime float16 = DataType(code=FLOAT16, native=DType.float16)
comptime float32 = DataType(code=FLOAT32, native=DType.float32)
comptime float64 = DataType(code=FLOAT64, native=DType.float64)
comptime string = DataType(code=STRING)
comptime binary = DataType(code=BINARY)

comptime all_numeric_dtypes = [
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
]


fn dynamic_size_of(dtype: DType) -> Int:
    """Get size of a dtype by dispatching to compile-time size_of."""
    if dtype == DType.bool:
        return size_of[DType.bool]()
    elif dtype == DType.int8:
        return size_of[DType.int8]()
    elif dtype == DType.int16:
        return size_of[DType.int16]()
    elif dtype == DType.int32:
        return size_of[DType.int32]()
    elif dtype == DType.int64:
        return size_of[DType.int64]()
    elif dtype == DType.uint8:
        return size_of[DType.uint8]()
    elif dtype == DType.uint16:
        return size_of[DType.uint16]()
    elif dtype == DType.uint32:
        return size_of[DType.uint32]()
    elif dtype == DType.uint64:
        return size_of[DType.uint64]()
    elif dtype == DType.float32:
        return size_of[DType.float32]()
    elif dtype == DType.float64:
        return size_of[DType.float64]()
    debug_assert(False, "Can't get the size of ", dtype)
    return 0
