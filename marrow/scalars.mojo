"""Arrow scalar types — single-value containers wrapping length-1 arrays.

Following Arrow Rust's design: a Scalar is an AnyArray where ``length == 1``.
This reuses all existing infrastructure (buffers, bitmaps, dtypes, builders)
without duplicating it.

Type-erased container:
  ``AnyScalar`` — wraps any length-1 ``AnyArray``, converts implicitly to/from
  typed scalars.

Typed scalars:
  ``PrimitiveScalar[T]`` — wraps ``PrimitiveArray[T]`` (length 1)
  ``StringScalar``       — wraps ``StringArray`` (length 1)
  ``ListScalar``         — wraps ``ListArray`` (length 1)
  ``StructScalar``       — wraps ``StructArray`` (length 1)
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
from .dtypes import DataType, Field, primitive_dtypes, numeric_dtypes


# ---------------------------------------------------------------------------
# PrimitiveScalar[T]
# ---------------------------------------------------------------------------


struct PrimitiveScalar[T: DataType](Copyable, Movable, Writable):
    """A single primitive value, backed by a length-1 PrimitiveArray."""

    var _data: PrimitiveArray[Self.T]

    def __init__(out self, value: Scalar[Self.T.native]) raises:
        """Create a valid scalar from a Mojo Scalar value."""
        var b = PrimitiveBuilder[Self.T](1)
        b.append(value)
        self._data = b.finish_typed()

    def __init__(out self, *, data: PrimitiveArray[Self.T]):
        """Wrap an existing length-1 array."""
        self._data = PrimitiveArray[Self.T](copy=data)

    @staticmethod
    def null() raises -> Self:
        """Create a null scalar."""
        var b = PrimitiveBuilder[Self.T](1)
        b.append_null()
        return Self(data=b.finish_typed())

    def is_valid(self) -> Bool:
        return self._data.is_valid(0)

    def is_null(self) -> Bool:
        return not self.is_valid()

    def value(self) -> Scalar[Self.T.native]:
        """Get the underlying value. Undefined if null."""
        return self._data.unsafe_get(0)

    def as_array(self) -> PrimitiveArray[Self.T]:
        """View as a length-1 PrimitiveArray."""
        return PrimitiveArray[Self.T](copy=self._data)

    def write_to[W: Writer](self, mut writer: W):
        if self.is_valid():
            writer.write(self.value())
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# StringScalar
# ---------------------------------------------------------------------------


struct StringScalar(Copyable, Movable, Writable):
    """A single string value, backed by a length-1 StringArray."""

    var _data: StringArray

    def __init__(out self, value: String) raises:
        """Create a valid scalar from a String."""
        var b = StringBuilder(1)
        b.append(value)
        self._data = b.finish_typed()

    def __init__(out self, *, data: StringArray):
        """Wrap an existing length-1 array."""
        self._data = StringArray(copy=data)

    @staticmethod
    def null() raises -> Self:
        """Create a null scalar."""
        var b = StringBuilder(1)
        b.append_null()
        return Self(data=b.finish_typed())

    def is_valid(self) -> Bool:
        return self._data.is_valid(0)

    def is_null(self) -> Bool:
        return not self.is_valid()

    def as_string(self) -> String:
        """Get the value as an owned String."""
        return String(self._data.unsafe_get(0))

    def as_array(self) -> StringArray:
        """View as a length-1 StringArray."""
        return StringArray(copy=self._data)

    def write_to[W: Writer](self, mut writer: W):
        if self.is_valid():
            writer.write('"')
            writer.write(self.as_string())
            writer.write('"')
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# ListScalar
# ---------------------------------------------------------------------------


struct ListScalar(Copyable, Movable, Writable):
    """A single list value, backed by a length-1 ListArray."""

    var _data: ListArray

    def __init__(out self, *, data: ListArray):
        """Wrap an existing length-1 array."""
        self._data = ListArray(copy=data)

    def is_valid(self) -> Bool:
        return self._data.is_valid(0)

    def is_null(self) -> Bool:
        return not self.is_valid()

    def value(self) -> AnyArray:
        """Get the sub-array for this list element."""
        return self._data.unsafe_get(0)

    def as_array(self) -> ListArray:
        return ListArray(copy=self._data)

    def write_to[W: Writer](self, mut writer: W):
        if self.is_valid():
            self.value().write_to(writer)
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# StructScalar
# ---------------------------------------------------------------------------


struct StructScalar(Copyable, Movable, Writable):
    """A single struct value, backed by a length-1 StructArray."""

    var _data: StructArray

    def __init__(out self, *, data: StructArray):
        """Wrap an existing length-1 array."""
        self._data = StructArray(copy=data)

    def is_valid(self) -> Bool:
        return self._data.is_valid(0)

    def is_null(self) -> Bool:
        return not self.is_valid()

    def num_fields(self) -> Int:
        return len(self._data.children)

    def as_array(self) -> StructArray:
        return StructArray(copy=self._data)

    def write_to[W: Writer](self, mut writer: W):
        if self.is_valid():
            writer.write("{")
            for i in range(self.num_fields()):
                if i > 0:
                    writer.write(", ")
                self._data.children[i].slice(0, 1).write_to(writer)
            writer.write("}")
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# AnyScalar — type-erased scalar container
# ---------------------------------------------------------------------------


struct AnyScalar(ConvertibleToPython, Copyable, Movable, Writable):
    """A single Arrow value — a length-1 AnyArray.

    Type-erased container. Converts implicitly to/from typed scalars.
    """

    var _data: AnyArray

    @implicit
    def __init__[T: DataType](out self, typed: PrimitiveScalar[T]):
        self._data = AnyArray(typed._data)

    @implicit
    def __init__(out self, typed: StringScalar):
        self._data = AnyArray(typed._data)

    @implicit
    def __init__(out self, typed: ListScalar):
        self._data = AnyArray(typed._data)

    @implicit
    def __init__(out self, typed: StructScalar):
        self._data = AnyArray(typed._data)

    def __init__(out self, *, data: AnyArray):
        """Wrap an existing length-1 AnyArray."""
        self._data = data.copy()

    def is_valid(self) -> Bool:
        return self._data.is_valid(0)

    def is_null(self) -> Bool:
        return not self.is_valid()

    def dtype(self) -> DataType:
        return self._data.dtype

    def as_primitive[T: DataType](self) raises -> PrimitiveScalar[T]:
        return PrimitiveScalar[T](data=self._data.as_primitive[T]())

    def as_string(self) raises -> StringScalar:
        return StringScalar(data=self._data.as_string())

    def as_list(self) raises -> ListScalar:
        return ListScalar(data=self._data.as_list())

    def as_struct(self) raises -> StructScalar:
        return StructScalar(data=self._data.as_struct())

    def write_to[W: Writer](self, mut writer: W):
        if self.is_valid():
            self._data.slice(0, 1).write_to(writer)
        else:
            writer.write("null")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def to_python_object(var self) raises -> PythonObject:
        """Convert to a Python scalar via the underlying AnyArray."""
        if self.is_null():
            # TODO: we should have a null scalar to return with
            return PythonObject(None)
        return self._data.slice(0, 1).to_python_object()
