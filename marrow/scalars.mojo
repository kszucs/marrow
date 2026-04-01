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

from std.memory import ArcPointer
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
    AnyType,
    PrimitiveType,
    Field,
    primitive_types,
    numeric_types,
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

    def type(self) -> AnyType:
        ...

    def is_valid(self) -> Bool:
        ...

    def is_null(self) -> Bool:
        ...

    def to_any(deinit self) -> AnyScalar:
        ...


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

    def type(self) -> AnyType:
        return Self.T

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

    def type(self) -> AnyType:
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

    def type(self) -> AnyType:
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

    var _dtype: AnyType
    var _value: List[AnyScalar]
    var _is_valid: Bool

    def __init__(
        out self,
        *,
        dtype: AnyType,
        value: List[AnyScalar],
        is_valid: Bool,
    ):
        self._dtype = dtype
        self._value = value.copy()
        self._is_valid = is_valid

    @staticmethod
    def null(dtype: AnyType) -> Self:
        return Self(dtype=dtype, value=List[AnyScalar](), is_valid=False)

    def type(self) -> AnyType:
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
    """Type-erased scalar container backed by an ArcPointer.

    Wraps any typed scalar on the heap behind an `ArcPointer`.
    Copies are O(1) ref-count bumps + function-pointer copies.
    Runtime dispatch goes through function-pointer trampolines (vtable).
    """

    var _data: ArcPointer[NoneType]
    var _virt_type: def(ArcPointer[NoneType]) -> AnyType
    var _virt_is_valid: def(ArcPointer[NoneType]) -> Bool
    var _virt_drop: def(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    def _tramp_type[T: Scalar](ptr: ArcPointer[NoneType]) -> AnyType:
        return rebind[ArcPointer[T]](ptr)[].type()

    @staticmethod
    def _tramp_is_valid[T: Scalar](ptr: ArcPointer[NoneType]) -> Bool:
        return rebind[ArcPointer[T]](ptr)[].is_valid()

    @staticmethod
    def _tramp_drop[T: Scalar](var ptr: ArcPointer[NoneType]):
        _ = rebind[ArcPointer[T]](ptr^)

    # --- construction ---

    @implicit
    def __init__[T: Scalar](out self, var typed: T):
        var ptr = ArcPointer(typed^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_type = Self._tramp_type[T]
        self._virt_is_valid = Self._tramp_is_valid[T]
        self._virt_drop = Self._tramp_drop[T]

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._virt_type = copy._virt_type
        self._virt_is_valid = copy._virt_is_valid
        self._virt_drop = copy._virt_drop

    # --- vtable dispatch ---

    def type(self) -> AnyType:
        return self._virt_type(self._data)

    def is_valid(self) -> Bool:
        return self._virt_is_valid(self._data)

    def is_null(self) -> Bool:
        return not self.is_valid()

    # --- typed downcasts ---

    def as_primitive[
        T: PrimitiveType
    ](ref self) -> ref[self._data[]] PrimitiveScalar[T]:
        return rebind[ArcPointer[PrimitiveScalar[T]]](self._data)[]

    def as_string(ref self) -> ref[self._data[]] StringScalar:
        return rebind[ArcPointer[StringScalar]](self._data)[]

    def as_list(ref self) -> ref[self._data[]] ListScalar:
        return rebind[ArcPointer[ListScalar]](self._data)[]

    def as_fixed_size_list(ref self) -> ref[self._data[]] ListScalar:
        return rebind[ArcPointer[ListScalar]](self._data)[]

    def as_struct(ref self) -> ref[self._data[]] StructScalar:
        return rebind[ArcPointer[StructScalar]](self._data)[]

    def write_to[W: Writer](self, mut writer: W):
        if self.is_null():
            writer.write("null")
            return
        var dtype = self.type()
        comptime for T in primitive_types:
            if dtype == T():
                self.as_primitive[T]().write_to(writer)
                return
        if dtype.is_string():
            self.as_string().write_to(writer)
        elif dtype.is_list() or dtype.is_fixed_size_list():
            self.as_list().write_to(writer)
        elif dtype.is_struct():
            self.as_struct().write_to(writer)

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def to_python_object(var self) raises -> PythonObject:
        """Convert to a Python Scalar wrapper object."""
        return PythonObject(alloc=self^)

    def __del__(deinit self):
        self._virt_drop(self._data^)
