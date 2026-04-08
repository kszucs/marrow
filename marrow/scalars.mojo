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
from std.builtin.variadics import Variadic
from std.builtin.rebind import downcast
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
    ArrowType,
    PrimitiveType,
    Field,
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
    bool_,
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

    def type(self) -> ArrowType:
        ...

    def is_valid(self) -> Bool:
        ...

    def is_null(self) -> Bool:
        ...

    def to_any(deinit self) -> AnyScalar:
        ...


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

    def type(self) -> ArrowType:
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
    Boolable, Copyable, Equatable, Movable, Scalar, Writable
):
    """A single primitive value: holds a native Mojo scalar + validity flag."""

    var _value: _Scalar[Self.T.native]
    var _is_valid: Bool

    @implicit
    def __init__(out self, value: _Scalar[Self.T.native]):
        self._value = value
        self._is_valid = True

    @implicit
    def __init__(out self, value: IntLiteral):
        self._value = _Scalar[Self.T.native](value)
        self._is_valid = True

    @implicit
    def __init__(out self, value: FloatLiteral):
        self._value = _Scalar[Self.T.native](value)
        self._is_valid = True

    def __init__(out self, *, is_valid: Bool):
        self._value = 0
        self._is_valid = is_valid

    @staticmethod
    def null() -> Self:
        return Self(is_valid=False)

    def type(self) -> ArrowType:
        return Self.T()

    def is_valid(self) -> Bool:
        return self._is_valid

    def is_null(self) -> Bool:
        return not self._is_valid

    def value(self) -> _Scalar[Self.T.native]:
        """Get the underlying native value. Undefined if null."""
        return self._value

    def to_array(self) raises -> PrimitiveArray[Self.T]:
        """Build a length-1 PrimitiveArray from this scalar."""
        var b = PrimitiveBuilder[Self.T](1)
        if self._is_valid:
            b.append(self._value)
        else:
            b.append_null()
        return b.finish()

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

    def __bool__(self) -> Bool:
        if self._is_valid:
            return Bool(self._value)
        return False

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

    def type(self) -> ArrowType:
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

    def type(self) -> ArrowType:
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

    var _dtype: ArrowType
    var _value: List[AnyScalar]
    var _is_valid: Bool

    def __init__(
        out self,
        *,
        dtype: ArrowType,
        value: List[AnyScalar],
        is_valid: Bool,
    ):
        self._dtype = dtype.copy()
        self._value = value.copy()
        self._is_valid = is_valid

    @staticmethod
    def null(dtype: ArrowType) -> Self:
        return Self(dtype=dtype, value=List[AnyScalar](), is_valid=False)

    def type(self) -> ArrowType:
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
        StringScalar,
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

    # --- generic dispatch ---

    def _dispatch[
        R: Movable,
        //,
        func: def[T: Scalar](T) capturing[_] -> R,
    ](self) -> R:
        comptime for i in range(Variadic.size(Self.VariantType.Ts)):
            comptime T = downcast[Self.VariantType.Ts[i], Scalar]
            if self._v.isa[T]():
                return func(self._v[T])
        abort("unreachable: invalid scalar type for dispatch")

    # --- dispatch-based methods ---

    def type(self) -> ArrowType:
        @parameter
        def f[T: Scalar](t: T) -> ArrowType:
            return t.type()

        return self._dispatch[f]()

    def is_valid(self) -> Bool:
        @parameter
        def f[T: Scalar](t: T) -> Bool:
            return t.is_valid()

        return self._dispatch[f]()

    def is_null(self) -> Bool:
        return not self.is_valid()

    # --- typed downcasts ---

    def as_bool(self) -> BoolScalar:
        return self._v[BoolScalar].copy()

    def as_primitive[T: PrimitiveType](self) -> PrimitiveScalar[T]:
        return self._v[PrimitiveScalar[T]].copy()

    def as_string(self) -> StringScalar:
        return self._v[StringScalar].copy()

    def as_list(self) -> ListScalar:
        return self._v[ListScalar].copy()

    def as_fixed_size_list(self) -> ListScalar:
        return self._v[ListScalar].copy()

    def as_struct(self) -> StructScalar:
        return self._v[StructScalar].copy()

    def write_to[W: Writer](self, mut writer: W):
        @parameter
        def f[T: Scalar](t: T):
            t.write_to(writer)

        self._dispatch[f]()

    def write_repr_to[W: Writer](self, mut writer: W):
        @parameter
        def f[T: Scalar](t: T):
            t.write_repr_to(writer)

        self._dispatch[f]()

    def to_python_object(var self) raises -> PythonObject:
        """Convert to a Python Scalar wrapper object."""
        return PythonObject(alloc=self^)
