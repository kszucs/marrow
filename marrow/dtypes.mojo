"""Arrow data type system — Variant-based implementation.

`DataType` is the trait that all concrete Arrow type structs implement.
`PrimitiveType` is a sub-trait that adds a `comptime native: DType` field,
enabling primitive-typed generics to use `T.native` as a compile-time type
parameter (e.g. `Buffer[T.native]`, `Scalar[T.native]`).

`AnyType` is the type-erased runtime container backed by a `Variant` — no
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

Comptime type tuples (replacing the old `*_dtypes` lists):
    primitive_types, numeric_types, integer_types, float_types,
    signed_integer_types, unsigned_integer_types
"""

from std.utils import Variant
from std.sys import size_of
from std.memory import ArcPointer
from std.python import PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython

# ---------------------------------------------------------------------------
# DataType trait and PrimitiveType sub-trait
# ---------------------------------------------------------------------------


trait DataType(Equatable, ImplicitlyDestructible, Movable, Writable):
    def byte_width(self) -> Int: ...
    def bit_width(self) -> UInt8: ...
    def to_any(deinit self) -> AnyType: ...


trait PrimitiveType(DataType):
    comptime native: DType

    def bit_width(self) -> UInt8:
        return UInt8(size_of[Self.native]() * 8)


# ---------------------------------------------------------------------------
# Concrete zero-size Arrow type structs
# ---------------------------------------------------------------------------


struct NullType(DataType, ImplicitlyCopyable):
    def __init__(out self): pass
    def byte_width(self) -> Int: return 0
    def bit_width(self) -> UInt8: return UInt8(0)
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("null")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct BoolType(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.bool

    def __init__(out self): pass
    def byte_width(self) -> Int: return 0
    def bit_width(self) -> UInt8: return UInt8(1)
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("bool")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct Int8Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.int8

    def __init__(out self): pass
    def byte_width(self) -> Int: return 1
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("int8")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct Int16Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.int16

    def __init__(out self): pass
    def byte_width(self) -> Int: return 2
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("int16")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct Int32Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.int32

    def __init__(out self): pass
    def byte_width(self) -> Int: return 4
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("int32")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct Int64Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.int64

    def __init__(out self): pass
    def byte_width(self) -> Int: return 8
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("int64")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct UInt8Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.uint8

    def __init__(out self): pass
    def byte_width(self) -> Int: return 1
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("uint8")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct UInt16Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.uint16

    def __init__(out self): pass
    def byte_width(self) -> Int: return 2
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("uint16")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct UInt32Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.uint32

    def __init__(out self): pass
    def byte_width(self) -> Int: return 4
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("uint32")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct UInt64Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.uint64

    def __init__(out self): pass
    def byte_width(self) -> Int: return 8
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("uint64")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct Float32Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.float32

    def __init__(out self): pass
    def byte_width(self) -> Int: return 4
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("float32")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct Float64Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.float64

    def __init__(out self): pass
    def byte_width(self) -> Int: return 8
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("float64")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct Float16Type(PrimitiveType, ImplicitlyCopyable):
    comptime native: DType = DType.float16

    def __init__(out self): pass
    def byte_width(self) -> Int: return 2
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("float16")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct BinaryType(DataType, ImplicitlyCopyable):
    def __init__(out self): pass
    def byte_width(self) -> Int: return 0
    def bit_width(self) -> UInt8: return UInt8(0)
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("binary")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct StringType(DataType, ImplicitlyCopyable):
    def __init__(out self): pass
    def byte_width(self) -> Int: return 0
    def bit_width(self) -> UInt8: return UInt8(0)
    def __eq__(self, other: Self) -> Bool: return True
    def write_to[W: Writer](self, mut writer: W): writer.write("string")
    def to_any(deinit self) -> AnyType: return AnyType(self^)


# ---------------------------------------------------------------------------
# Field and nested compound types
# (use ArcPointer[AnyType] for child dtype to break the circular layout)
# ---------------------------------------------------------------------------


struct Field(
    ConvertibleFromPython,
    ConvertibleToPython,
    Equatable,
    ImplicitlyCopyable,
    Movable,
    Writable,
):
    var name: String
    var dtype: ArcPointer[AnyType]
    var nullable: Bool

    def __init__(
        out self, name: String, dtype: ArcPointer[AnyType], nullable: Bool = True
    ):
        self.name = name
        self.dtype = dtype
        self.nullable = nullable

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Field]()[]

    def __eq__(self, other: Self) -> Bool:
        return (
            self.name == other.name
            and self.dtype[] == other.dtype[]
            and self.nullable == other.nullable
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, ": ", self.dtype[])

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write(
            "Field(name=", self.name, ", nullable=", self.nullable, ")"
        )

    def to_python_object(var self) -> PythonObject:
        return PythonObject(alloc=self^)


struct ListType(DataType, ImplicitlyCopyable):
    var item: ArcPointer[AnyType]

    def __init__(out self, item: ArcPointer[AnyType]):
        self.item = item

    def byte_width(self) -> Int: return 0
    def bit_width(self) -> UInt8: return UInt8(0)

    def __eq__(self, other: Self) -> Bool:
        return self.item[] == other.item[]

    def write_to[W: Writer](self, mut writer: W):
        writer.write("list<", self.item[], ">")

    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct FixedSizeListType(DataType, Copyable):
    var item: Field
    var size: Int

    def __init__(out self, var item: Field, size: Int):
        self.item = item^
        self.size = size

    def byte_width(self) -> Int: return 0
    def bit_width(self) -> UInt8: return UInt8(0)

    def __eq__(self, other: Self) -> Bool:
        return self.item == other.item and self.size == other.size

    def write_to[W: Writer](self, mut writer: W):
        writer.write("fixed_size_list<", self.item, ">")

    def to_any(deinit self) -> AnyType: return AnyType(self^)


struct StructType(DataType, Copyable):
    var fields: List[Field]

    def __init__(out self, var fields: List[Field]):
        self.fields = fields^

    def __init__(out self, *, copy: Self):
        self.fields = copy.fields.copy()

    def byte_width(self) -> Int: return 0
    def bit_width(self) -> UInt8: return UInt8(0)

    def __eq__(self, other: Self) -> Bool:
        return self.fields == other.fields

    def write_to[W: Writer](self, mut writer: W):
        writer.write("struct<")
        for i in range(len(self.fields)):
            if i > 0:
                writer.write(", ")
            writer.write(self.fields[i])
        writer.write(">")

    def to_any(deinit self) -> AnyType: return AnyType(self^)


# ---------------------------------------------------------------------------
# AnyType — Variant-based type-erased handle
# ---------------------------------------------------------------------------

comptime _AnyTypeV = Variant[
    NullType, BoolType,
    Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    Float16Type, Float32Type, Float64Type,
    BinaryType, StringType,
    ListType, FixedSizeListType, StructType,
]


struct AnyType(
    Copyable,
    ConvertibleFromPython,
    ConvertibleToPython,
    Equatable,
    ImplicitlyCopyable,
    Movable,
    Writable,
):
    var _v: _AnyTypeV

    @implicit
    def __init__[T: DataType](out self, var value: T):
        self._v = _AnyTypeV(value^)

    def __init__(out self, *, copy: Self):
        self._v = _AnyTypeV(copy=copy._v)

    def __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema

        # Try downcasting from a marrow Python object.
        try:
            self = py.downcast_value_ptr[Self]()[]
            return
        except:
            pass
        # Fall back to the Arrow C Schema Interface for foreign objects.
        var capsule: PythonObject
        try:
            capsule = py.__arrow_c_schema__()
        except:
            raise Error("cannot convert Python object to AnyType")
        self = CArrowSchema.from_pycapsule(capsule).to_dtype()

    def to_python_object(var self) -> PythonObject:
        return PythonObject(alloc=self^)

    # --- generic type dispatch ---

    # R: Movable and should be infer only
    def _dispatch[
        R: Copyable,
        func: def[T: DataType](T) capturing[_] -> R,
    ](self) -> R:
        if self._v.isa[NullType](): return func[NullType](self._v[NullType])
        if self._v.isa[BoolType](): return func[BoolType](self._v[BoolType])
        if self._v.isa[Int8Type](): return func[Int8Type](self._v[Int8Type])
        if self._v.isa[Int16Type](): return func[Int16Type](self._v[Int16Type])
        if self._v.isa[Int32Type](): return func[Int32Type](self._v[Int32Type])
        if self._v.isa[Int64Type](): return func[Int64Type](self._v[Int64Type])
        if self._v.isa[UInt8Type](): return func[UInt8Type](self._v[UInt8Type])
        if self._v.isa[UInt16Type](): return func[UInt16Type](self._v[UInt16Type])
        if self._v.isa[UInt32Type](): return func[UInt32Type](self._v[UInt32Type])
        if self._v.isa[UInt64Type](): return func[UInt64Type](self._v[UInt64Type])
        if self._v.isa[Float16Type](): return func[Float16Type](self._v[Float16Type])
        if self._v.isa[Float32Type](): return func[Float32Type](self._v[Float32Type])
        if self._v.isa[Float64Type](): return func[Float64Type](self._v[Float64Type])
        if self._v.isa[BinaryType](): return func[BinaryType](self._v[BinaryType])
        if self._v.isa[StringType](): return func[StringType](self._v[StringType])
        if self._v.isa[ListType](): return func[ListType](self._v[ListType])
        if self._v.isa[FixedSizeListType](): return func[FixedSizeListType](self._v[FixedSizeListType])
        return func[StructType](self._v[StructType])
        # TODO: raise otherwise

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

    def is_struct(self) -> Bool:
        return self._v.isa[StructType]()

    def is_fixed_size(self) -> Bool:
        return self.bit_width() > 0

    def bit_width(self) -> UInt8:
        @parameter
        def f[T: DataType](t: T) -> UInt8: return t.bit_width()
        return self._dispatch[UInt8, f]()

    def byte_width(self) -> Int:
        @parameter
        def f[T: DataType](t: T) -> Int: return t.byte_width()
        return self._dispatch[Int, f]()

    def write_to[W: Writer](self, mut writer: W):
        @parameter
        def f[T: DataType](t: T) -> NoneType:
            t.write_to(writer)
            return NoneType()
        _ = self._dispatch[NoneType, f]()

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def __eq__(self, other: Self) -> Bool:
        return self._v == other._v

    def __is__(self, other: Self) -> Bool:
        return self == other

    # --- compound type accessors ---

    def as_list_type(ref self) -> ref[self] ListType:
        """For list types, returns a reference to the inner ListType."""
        return self._v[ListType]

    def as_fixed_size_list_type(ref self) -> ref[self] FixedSizeListType:
        """For fixed-size list types, returns a reference to the inner FixedSizeListType."""
        return self._v[FixedSizeListType]

    def as_struct_type(ref self) -> ref[self] StructType:
        """For struct types, returns a reference to the inner StructType."""
        return self._v[StructType]


# ---------------------------------------------------------------------------
# Field constructor and factory functions
# ---------------------------------------------------------------------------


def field(name: String, var dtype: AnyType, nullable: Bool = True) -> Field:
    """Construct a Field. Equivalent to PyArrow's ``pa.field()``."""
    return Field(name, ArcPointer(dtype^), nullable)


def list_(var value_type: AnyType) -> AnyType:
    """Construct a list type. Equivalent to PyArrow's ``pa.list_()``."""
    return ListType(ArcPointer(value_type^))


def fixed_size_list_(var value_type: AnyType, size: Int) -> AnyType:
    """Construct a fixed-size list type. Equivalent to PyArrow's ``pa.list_()`` with list_size."""
    return FixedSizeListType(field("item", value_type^), size)


def struct_(var fields: List[Field]) -> AnyType:
    """Construct a struct type from a list of fields."""
    return StructType(fields^)


def struct_(var *fields: Field) -> AnyType:
    """Construct a struct type from variadic fields."""
    return StructType(List(elements=fields^))


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

comptime signed_integer_types = (Int8Type, Int16Type, Int32Type, Int64Type)
comptime unsigned_integer_types = (UInt8Type, UInt16Type, UInt32Type, UInt64Type)
comptime integer_types = (Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type)
comptime float_types = (Float16Type, Float32Type, Float64Type)
comptime numeric_types = (
    Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    Float16Type, Float32Type, Float64Type,
)
comptime primitive_types = (
    BoolType,
    Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    Float16Type, Float32Type, Float64Type,
)
