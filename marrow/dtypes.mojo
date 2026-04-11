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

    def __init__(
        out self, name: String, var dtype: AnyDataType, nullable: Bool = True
    ):
        self.name = name
        self.dtype = dtype^
        self.nullable = nullable

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.dtype = copy.dtype.copy()
        self.nullable = copy.nullable

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
        StructType,
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

        comptime IsPrimitive[T: Movable & ImplicitlyDestructible] = conforms_to(T, PrimitiveType)

        @parameter
        def f[T: PrimitiveType](t: T) -> Int:
            return t.byte_width()

        return variant_dispatch[PrimitiveType, predicate=IsPrimitive, func=f](self._v)

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
