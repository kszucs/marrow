from std.python import PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython

# The following enum codes are copied from the C++ implementation of Arrow

# A NULL type having no physical storage
comptime NA: UInt8 = 0

# Boolean as 1 bit, LSB bit-packed ordering
comptime BOOL: UInt8 = 1

# Unsigned 8-bit little-endian integer
comptime UINT8: UInt8 = 2

# Signed 8-bit little-endian integer
comptime INT8: UInt8 = 3

# Unsigned 16-bit little-endian integer
comptime UINT16: UInt8 = 4

# Signed 16-bit little-endian integer
comptime INT16: UInt8 = 5

# Unsigned 32-bit little-endian integer
comptime UINT32: UInt8 = 6

# Signed 32-bit little-endian integer
comptime INT32: UInt8 = 7

# Unsigned 64-bit little-endian integer
comptime UINT64: UInt8 = 8

# Signed 64-bit little-endian integer
comptime INT64: UInt8 = 9

# 2-byte floating point value
comptime FLOAT16: UInt8 = 10

# 4-byte floating point value
comptime FLOAT32: UInt8 = 11

# 8-byte floating point value
comptime FLOAT64: UInt8 = 12

# UTF8 variable-length string as List<Char>
comptime STRING: UInt8 = 13

# Variable-length bytes (no guarantee of UTF8-ness)
comptime BINARY: UInt8 = 14

# Fixed-size binary. Each value occupies the same number of bytes
comptime FIXED_SIZE_BINARY: UInt8 = 15

# int32_t days since the UNIX epoch
comptime DATE32: UInt8 = 16

# int64_t milliseconds since the UNIX epoch
comptime DATE64: UInt8 = 17

# Exact timestamp encoded with int64 since UNIX epoch
# Default unit millisecond
comptime TIMESTAMP: UInt8 = 18

# Time as signed 32-bit integer, representing either seconds or
# milliseconds since midnight
comptime TIME32: UInt8 = 19

# Time as signed 64-bit integer, representing either microseconds or
# nanoseconds since midnight
comptime TIME64: UInt8 = 20

# YEAR_MONTH interval in SQL style
comptime INTERVAL_MONTHS: UInt8 = 21

# DAY_TIME interval in SQL style
comptime INTERVAL_DAY_TIME: UInt8 = 22

# Precision- and scale-based decimal type with 128 bits.
comptime DECIMAL128: UInt8 = 23

# Defined for backward-compatibility.
comptime DECIMAL: UInt8 = DECIMAL128

# Precision- and scale-based decimal type with 256 bits.
comptime DECIMAL256: UInt8 = 24

# A list of some logical data type
comptime LIST: UInt8 = 25

# Struct of logical types
comptime STRUCT: UInt8 = 26

# Sparse unions of logical types
comptime SPARSE_UNION: UInt8 = 27

# Dense unions of logical types
comptime DENSE_UNION: UInt8 = 28

# Dictionary-encoded type, also called "categorical" or "factor"
# in other programming languages. Holds the dictionary value
# type but not the dictionary itself, which is part of the
# Array struct
comptime DICTIONARY: UInt8 = 29

# Map, a repeated struct logical type
comptime MAP: UInt8 = 30

# Custom data type, implemented by user
comptime EXTENSION: UInt8 = 31

# Fixed size list of some logical type
comptime FIXED_SIZE_LIST: UInt8 = 32

# Measure of elapsed time in either seconds, milliseconds, microseconds
# or nanoseconds.
comptime DURATION: UInt8 = 33

# Like STRING, but with 64-bit offsets
comptime LARGE_STRING: UInt8 = 34

# Like BINARY, but with 64-bit offsets
comptime LARGE_BINARY: UInt8 = 35

# Like LIST, but with 64-bit offsets
comptime LARGE_LIST: UInt8 = 36

# Calendar interval type with three fields.
comptime INTERVAL_MONTH_DAY_NANO: UInt8 = 37

# Run-end encoded data.
comptime RUN_END_ENCODED: UInt8 = 38

# String (UTF8) view type with 4-byte prefix and inline small string
# optimization
comptime STRING_VIEW: UInt8 = 39

# Bytes view type with 4-byte prefix and inline small string optimization
comptime BINARY_VIEW: UInt8 = 40

# A list of some logical data type represented by offset and size.
comptime LIST_VIEW: UInt8 = 41

# Like LIST_VIEW, but with 64-bit offsets and sizes
comptime LARGE_LIST_VIEW: UInt8 = 42


struct Field(
    ConvertibleFromPython,
    ConvertibleToPython,
    Equatable,
    ImplicitlyCopyable,
    Movable,
    Writable,
):
    var name: String
    var dtype: DataType
    var nullable: Bool

    def __init__(
        out self, name: String, var dtype: DataType, nullable: Bool = False
    ):
        self.name = name
        self.dtype = dtype^
        self.nullable = nullable

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Field]()[]

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


def field(name: String, var dtype: DataType, nullable: Bool = True) -> Field:
    """Construct a Field from a name and data type.

    Equivalent to PyArrow's `pa.field()`.

    Example:
        field("x", int32)
        field("y", float64, nullable=False)
    """
    return Field(name, dtype^, nullable)


struct DataType(
    ConvertibleFromPython,
    ConvertibleToPython,
    Equatable,
    ImplicitlyCopyable,
    Movable,
    Writable,
):
    var code: UInt8
    var native: DType
    var fields: List[Field]
    var size: Int

    def __init__(out self, *, code: UInt8):
        self.code = code
        self.native = DType.invalid
        self.fields = []
        self.size = 0

    def __init__(out self, native: DType):
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
        self.size = 0

    def __init__(out self, *, code: UInt8, native: DType):
        self.code = code
        self.native = native
        self.fields = []
        self.size = 0

    def __init__(out self, *, code: UInt8, fields: List[Field]):
        self.code = code
        self.native = DType.invalid
        self.fields = fields.copy()
        self.size = 0

    def __init__(out self, *, copy: Self):
        self.code = copy.code
        self.native = copy.native
        self.fields = copy.fields.copy()
        self.size = copy.size

    def __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema

        # Try downcasting from a marrow Python object.
        try:
            self = py.downcast_value_ptr[DataType]()[]
            return
        except:
            pass
        # Fall back to the Arrow C Schema Interface for foreign objects.
        var capsule: PythonObject
        try:
            capsule = py.__arrow_c_schema__()
        except:
            raise Error("cannot convert Python object to DataType")
        self = CArrowSchema.from_pycapsule(capsule).to_dtype()

    def __eq__(self, other: Self) -> Bool:
        if self.code != other.code or self.size != other.size:
            return False
        return self.fields == other.fields

    def __is__(self, other: DataType) -> Bool:
        return self == other

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def write_to[W: Writer](self, mut writer: W):
        if self.code == NA:
            writer.write("null")
        elif self.code == BOOL:
            writer.write("bool")
        elif self.code == INT8:
            writer.write("int8")
        elif self.code == INT16:
            writer.write("int16")
        elif self.code == INT32:
            writer.write("int32")
        elif self.code == INT64:
            writer.write("int64")
        elif self.code == UINT8:
            writer.write("uint8")
        elif self.code == UINT16:
            writer.write("uint16")
        elif self.code == UINT32:
            writer.write("uint32")
        elif self.code == UINT64:
            writer.write("uint64")
        elif self.code == FLOAT16:
            writer.write("float16")
        elif self.code == FLOAT32:
            writer.write("float32")
        elif self.code == FLOAT64:
            writer.write("float64")
        elif self.code == STRING:
            writer.write("string")
        elif self.code == BINARY:
            writer.write("binary")
        elif self.code == LIST:
            writer.write("list<")
            writer.write(self.fields[0].dtype)
            writer.write(">")
        elif self.code == FIXED_SIZE_LIST:
            writer.write("fixed_size_list<")
            writer.write(self.fields[0].dtype)
            writer.write(">")
        elif self.code == STRUCT:
            writer.write("struct<")
            for i in range(len(self.fields)):
                if i > 0:
                    writer.write(", ")
                writer.write(self.fields[i].name, ": ", self.fields[i].dtype)
            writer.write(">")
        else:
            writer.write("unknown ", self.code)

    def bit_width(self) -> UInt8:
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

    def byte_width(self) -> Int:
        return Int(self.bit_width()) // 8

    @always_inline
    def is_bool(self) -> Bool:
        return self.code == BOOL

    @always_inline
    def is_fixed_size(self) -> Bool:
        return self.bit_width() > 0

    @always_inline
    def is_integer(self) -> Bool:
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
    def is_signed_integer(self) -> Bool:
        return self.code in [INT8, INT16, INT32, INT64]

    @always_inline
    def is_unsigned_integer(self) -> Bool:
        return self.code in [
            UINT8,
            UINT16,
            UINT32,
            UINT64,
        ]

    @always_inline
    def is_floating_point(self) -> Bool:
        return self.code in [FLOAT32, FLOAT64]

    @always_inline
    def is_numeric(self) -> Bool:
        return self.is_integer() or self.is_floating_point()

    @always_inline
    def is_primitive(self) -> Bool:
        return self.is_numeric() or self.is_bool()

    @always_inline
    def is_string(self) -> Bool:
        return self.code == STRING

    @always_inline
    def is_list(self) -> Bool:
        return self.code == LIST

    @always_inline
    def is_fixed_size_list(self) -> Bool:
        return self.code == FIXED_SIZE_LIST

    @always_inline
    def is_null(self) -> Bool:
        return self.code == NA

    @always_inline
    def is_binary(self) -> Bool:
        return self.code == BINARY

    @always_inline
    def is_struct(self) -> Bool:
        return self.code == STRUCT


def list_(var value_type: DataType) -> DataType:
    return DataType(code=LIST, fields=[Field("item", value_type^, True)])


def fixed_size_list_(var value_type: DataType, size: Int) -> DataType:
    var dt = DataType(
        code=FIXED_SIZE_LIST, fields=[Field("item", value_type^, True)]
    )
    dt.size = size
    return dt^


def struct_(fields: List[Field]) -> DataType:
    return DataType(code=STRUCT, fields=fields)


def struct_(var *fields: Field) -> DataType:
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

comptime numeric_dtypes = [
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
comptime primitive_dtypes = [bool_] + numeric_dtypes
