# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What a JSON value is (`JsonKind`) and what Arrow type a column is
(`JsonType`) — the vocabulary the reader and the writer share.

`JsonType` is resolved once per column, so the per-value loops switch on a
small integer rather than on a `DynType`.
"""

from ..dtypes import (
    DynType,
    bool_,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)


@fieldwise_init
struct JsonKind(Equatable, ImplicitlyCopyable, Movable, Writable):
    """The kind of a JSON value, written as Arrow's error messages word it:
    `Column(/a) changed from number to string`."""

    var code: Int

    comptime NULL = Self(0)
    comptime BOOL = Self(1)
    comptime NUMBER = Self(2)
    comptime STRING = Self(3)
    comptime ARRAY = Self(4)
    comptime OBJECT = Self(5)

    @staticmethod
    @always_inline
    def of(byte: Byte) -> Optional[Self]:
        """The kind of the value starting with `byte`, or `None` if no JSON
        value starts with it."""
        if byte == Byte(ord('"')):
            return Self.STRING
        elif byte == Byte(ord("{")):
            return Self.OBJECT
        elif byte == Byte(ord("[")):
            return Self.ARRAY
        elif byte == Byte(ord("t")) or byte == Byte(ord("f")):
            return Self.BOOL
        elif byte == Byte(ord("n")):
            return Self.NULL
        elif byte == Byte(ord("-")) or (
            byte >= Byte(ord("0")) and byte <= Byte(ord("9"))
        ):
            return Self.NUMBER
        return None

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def write_to[W: Writer](self, mut writer: W):
        if self == Self.NULL:
            writer.write("null")
        elif self == Self.BOOL:
            writer.write("boolean")
        elif self == Self.NUMBER:
            writer.write("number")
        elif self == Self.STRING:
            writer.write("string")
        elif self == Self.ARRAY:
            writer.write("array")
        else:
            writer.write("object")


@fieldwise_init
struct JsonType(Equatable, ImplicitlyCopyable, Movable):
    """The Arrow type of a column JSON is read into or written from: one
    code per type, a timestamp one code whatever its unit."""

    var code: Int

    comptime NULL = Self(0)
    comptime BOOL = Self(1)
    comptime INT8 = Self(2)
    comptime INT16 = Self(3)
    comptime INT32 = Self(4)
    comptime INT64 = Self(5)
    comptime UINT8 = Self(6)
    comptime UINT16 = Self(7)
    comptime UINT32 = Self(8)
    comptime UINT64 = Self(9)
    comptime FLOAT16 = Self(10)
    comptime FLOAT32 = Self(11)
    comptime FLOAT64 = Self(12)
    comptime STRING = Self(13)
    comptime LARGE_STRING = Self(14)
    comptime TIMESTAMP = Self(15)
    comptime DATE32 = Self(16)
    comptime DATE64 = Self(17)
    comptime LIST = Self(18)
    comptime LARGE_LIST = Self(19)
    comptime STRUCT = Self(20)

    @staticmethod
    def of(dtype: DynType) -> Optional[Self]:
        """The column type `dtype` is, or `None` if JSON has no mapping for
        it."""
        if dtype.is_null():
            return Self.NULL
        elif dtype == bool_:
            return Self.BOOL
        elif dtype == int8:
            return Self.INT8
        elif dtype == int16:
            return Self.INT16
        elif dtype == int32:
            return Self.INT32
        elif dtype == int64:
            return Self.INT64
        elif dtype == uint8:
            return Self.UINT8
        elif dtype == uint16:
            return Self.UINT16
        elif dtype == uint32:
            return Self.UINT32
        elif dtype == uint64:
            return Self.UINT64
        elif dtype == float16:
            return Self.FLOAT16
        elif dtype == float32:
            return Self.FLOAT32
        elif dtype == float64:
            return Self.FLOAT64
        elif dtype.is_string():
            return Self.STRING
        elif dtype.is_large_string():
            return Self.LARGE_STRING
        elif dtype.is_timestamp():
            return Self.TIMESTAMP
        elif dtype.is_date32():
            return Self.DATE32
        elif dtype.is_date64():
            return Self.DATE64
        elif dtype.is_list():
            return Self.LIST
        elif dtype.is_large_list():
            return Self.LARGE_LIST
        elif dtype.is_struct():
            return Self.STRUCT
        return None

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def is_readable(self) -> Bool:
        """Whether the reader builds this type. `date32` is left out on
        purpose: pyarrow reads it from a number, not a date string, and
        `float16`, `date64` and `large_list` are written only."""
        return not (
            self == Self.FLOAT16
            or self == Self.DATE32
            or self == Self.DATE64
            or self == Self.LARGE_LIST
        )

    @always_inline
    def kind(self) -> JsonKind:
        """The JSON value kind a column of this type holds."""
        if self == Self.NULL:
            return JsonKind.NULL
        elif self == Self.BOOL:
            return JsonKind.BOOL
        elif self.code >= Self.INT8.code and self.code <= Self.FLOAT64.code:
            return JsonKind.NUMBER
        elif self.code >= Self.STRING.code and self.code <= Self.DATE64.code:
            return JsonKind.STRING
        elif self == Self.LIST or self == Self.LARGE_LIST:
            return JsonKind.ARRAY
        return JsonKind.OBJECT
