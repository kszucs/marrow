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

from std.python import PythonObject
from std.python.conversions import ConvertibleToPython

from .arrays import (
    PrimitiveArray,
    StringArray,
    ListArray,
    StructArray,
    AnyArray,
)
from .builders import PrimitiveBuilder, StringBuilder
from .buffers import Buffer, BufferBuilder
from .bitmap import Bitmap, BitmapBuilder
from .dtypes import DataType, Field, primitive_dtypes, numeric_dtypes

# Alias the built-in Scalar[DType] to avoid shadowing by the local Scalar trait.
from builtin.simd import Scalar as _Scalar


# ---------------------------------------------------------------------------
# Scalar trait
# ---------------------------------------------------------------------------


trait Scalar(Copyable, Movable, Writable):
    """Common interface for all typed Arrow scalars."""

    def dtype(self) -> DataType:
        ...

    def is_valid(self) -> Bool:
        ...

    def is_null(self) -> Bool:
        ...

    def as_any(self) -> AnyScalar:
        ...


# ---------------------------------------------------------------------------
# PrimitiveScalar[T]
# ---------------------------------------------------------------------------


struct PrimitiveScalar[T: DataType](Boolable, Copyable, Movable, Writable, Equatable):
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

    def dtype(self) -> DataType:
        return Self.T

    def is_valid(self) -> Bool:
        return self._is_valid

    def is_null(self) -> Bool:
        return not self._is_valid

    def value(self) -> _Scalar[Self.T.native]:
        """Get the underlying native value. Undefined if null."""
        return self._value

    def as_array(self) raises -> PrimitiveArray[Self.T]:
        """Build a length-1 PrimitiveArray from this scalar."""
        var b = PrimitiveBuilder[Self.T](1)
        if self._is_valid:
            b.append(self._value)
        else:
            b.append_null()
        return b.finish_typed()

    def as_any(self) -> AnyScalar:
        return self

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
# StringScalar
# ---------------------------------------------------------------------------


struct StringScalar(Copyable, Movable, Writable, Equatable):
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

    def dtype(self) -> DataType:
        from .dtypes import string
        return string

    def is_valid(self) -> Bool:
        return self._is_valid

    def is_null(self) -> Bool:
        return not self._is_valid

    def as_string(self) -> String:
        """Get the value as an owned String."""
        return self._value

    def as_array(self) raises -> StringArray:
        """Build a length-1 StringArray from this scalar."""
        var b = StringBuilder(1)
        if self._is_valid:
            b.append(self._value)
        else:
            b.append_null()
        return b.finish_typed()

    def as_any(self) raises -> AnyScalar:
        return AnyScalar(self)

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


struct ListScalar(Copyable, Movable, Writable):
    """A single list value: holds an AnyArray of child elements + validity flag."""

    var _value: AnyArray
    var _is_valid: Bool

    def __init__(out self, *, value: AnyArray, is_valid: Bool):
        self._value = value.copy()
        self._is_valid = is_valid

    def dtype(self) raises -> DataType:
        from .dtypes import list_
        return list_(self._value.dtype)

    def is_valid(self) -> Bool:
        return self._is_valid

    def is_null(self) -> Bool:
        return not self._is_valid

    def value(self) -> AnyArray:
        """Get the child elements array."""
        return self._value.copy()

    def as_any(self) raises -> AnyScalar:
        return AnyScalar(self)

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


struct StructScalar(Copyable, Movable, Writable):
    """A single struct value: holds one AnyScalar per field + validity flag."""

    var _dtype: DataType
    var _value: List[AnyScalar]
    var _is_valid: Bool

    def __init__(
        out self,
        *,
        dtype: DataType,
        value: List[AnyScalar],
        is_valid: Bool,
    ):
        self._dtype = dtype
        self._value = value.copy()
        self._is_valid = is_valid

    @staticmethod
    def null(dtype: DataType) -> Self:
        return Self(dtype=dtype, value=List[AnyScalar](), is_valid=False)

    def dtype(self) -> DataType:
        return self._dtype

    def is_valid(self) -> Bool:
        return self._is_valid

    def is_null(self) -> Bool:
        return not self._is_valid

    def num_fields(self) -> Int:
        return len(self._value)

    def field(self, index: Int) -> AnyScalar:
        """Return the i-th field as an AnyScalar."""
        return self._value[index].copy()

    def as_any(self) raises -> AnyScalar:
        return AnyScalar(self)

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
    """A single Arrow value backed by a length-1 AnyArray.

    Type-erased container. Converts implicitly from typed scalars by building
    a temporary length-1 array — an O(1) allocation, not O(n).
    """

    var _data: AnyArray

    @implicit
    def __init__[T: DataType](out self, typed: PrimitiveScalar[T]) raises:
        var b = PrimitiveBuilder[T](1)
        if typed._is_valid:
            b.append(typed._value)
        else:
            b.append_null()
        self._data = AnyArray(b.finish_typed())

    @implicit
    def __init__(out self, typed: StringScalar) raises:
        var b = StringBuilder(1)
        if typed._is_valid:
            b.append(typed._value)
        else:
            b.append_null()
        self._data = AnyArray(b.finish_typed())

    @implicit
    def __init__(out self, typed: ListScalar) raises:
        from .dtypes import list_
        var child = typed._value.copy()
        var obb = BufferBuilder.alloc[DType.int32](2)
        obb.unsafe_set[DType.int32](0, 0)
        obb.unsafe_set[DType.int32](1, Int32(child.length))
        var offsets = obb.finish()
        var bm: Optional[Bitmap] = None
        var nulls = 0
        if not typed._is_valid:
            var bmb = BitmapBuilder.alloc(1)
            bm = bmb.finish(1)
            nulls = 1
        self._data = AnyArray(
            dtype=list_(child.dtype),
            length=1,
            nulls=nulls,
            bitmap=bm^,
            buffers=[offsets],
            children=[child^],
            offset=0,
        )

    @implicit
    def __init__(out self, typed: StructScalar) raises:
        var bm: Optional[Bitmap] = None
        var nulls = 0
        if not typed._is_valid:
            var bmb = BitmapBuilder.alloc(1)
            bm = bmb.finish(1)
            nulls = 1
        var children = List[AnyArray]()
        for i in range(len(typed._value)):
            children.append(typed._value[i]._data.copy())
        self._data = AnyArray(
            dtype=typed._dtype,
            length=1,
            nulls=nulls,
            bitmap=bm^,
            buffers=[],
            children=children^,
            offset=0,
        )

    def __init__(out self, *, data: AnyArray):
        """Wrap an existing length-1 AnyArray directly."""
        self._data = data.copy()

    def is_valid(self) -> Bool:
        return self._data.is_valid(0)

    def is_null(self) -> Bool:
        return not self.is_valid()

    def dtype(self) -> DataType:
        return self._data.dtype

    def as_primitive[T: DataType](self) raises -> PrimitiveScalar[T]:
        var arr = self._data.as_primitive[T]()
        if not self._data.is_valid(0):
            return PrimitiveScalar[T].null()
        return PrimitiveScalar[T](arr.unsafe_get(0))

    def as_string(self) raises -> StringScalar:
        var arr = self._data.as_string()
        if not self._data.is_valid(0):
            return StringScalar.null()
        return StringScalar(String(arr.unsafe_get(UInt(0))))

    def as_list(self) raises -> ListScalar:
        var arr = self._data.as_list()
        return ListScalar(
            value=arr.unsafe_get(0), is_valid=self._data.is_valid(0)
        )

    def as_fixed_size_list(self) raises -> ListScalar:
        var arr = self._data.as_fixed_size_list()
        return ListScalar(
            value=arr.unsafe_get(0), is_valid=self._data.is_valid(0)
        )

    def as_struct(self) raises -> StructScalar:
        var arr = self._data.as_struct()
        if not self._data.is_valid(0):
            return StructScalar.null(arr.dtype)
        var fields = List[AnyScalar]()
        var offset = self._data.offset
        for i in range(len(arr.children)):
            fields.append(AnyScalar(data=arr.children[i].slice(offset, 1)))
        return StructScalar(dtype=arr.dtype, value=fields^, is_valid=True)

    def write_to[W: Writer](self, mut writer: W):
        if self.is_null():
            writer.write("null")
            return
        try:
            var dtype = self.dtype()
            comptime for T in primitive_dtypes:
                if dtype == T:
                    writer.write(self.as_primitive[T]().value())
                    return
            if dtype.is_string():
                writer.write(self.as_string().as_string())
            elif dtype.is_list():
                self.as_list().value().write_to(writer)
            elif dtype.is_fixed_size_list():
                self.as_fixed_size_list().value().write_to(writer)
            elif dtype.is_struct():
                self.as_struct().write_to(writer)
        except:
            writer.write("Scalar(?)")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def to_python_object(var self) raises -> PythonObject:
        """Convert to a Python Scalar wrapper object."""
        return PythonObject(alloc=self^)
