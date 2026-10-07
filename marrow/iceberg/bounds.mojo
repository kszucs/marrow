# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Iceberg's binary single-value serialization (spec Appendix D).

What a manifest's `lower_bounds` and `upper_bounds` maps hold, one value per
column: fixed-width values little-endian at their own width (`date` as int32
days, `time` and `timestamp` as int64 micros, the `_ns` timestamps as int64
nanos), `string` as its UTF-8 bytes, `uuid`, `fixed` and `binary` as their raw
bytes, `boolean` as one byte, and a decimal as its unscaled value in the fewest
big-endian two's-complement bytes that hold it.

The dtypes are the ones `types.primitive_type` returns, plus the reverse
mapping's alternatives (`large_string`, `large_binary`, narrower decimals).
"""

from std.sys import size_of

from ..dtypes import DecimalType, DynType, PrimitiveType
from ..errors import CorruptError, DynError, InvalidError, TypeError
from ..scalars import (
    BinaryScalar,
    BoolScalar,
    DynScalar,
    FixedSizeBinaryScalar,
    LargeBinaryScalar,
    LargeStringScalar,
    PrimitiveScalar,
    StringScalar,
)
from ..codecs import BigEndian, LittleEndian
from .types import primitive_name


def min_big_endian[D: DType](value: Scalar[D]) -> List[UInt8]:
    """`value` as two's-complement big-endian bytes, with every leading byte
    that only repeats the sign dropped: Iceberg's decimal encoding, for bounds
    and for the bucket hash alike. Always at least one byte."""
    comptime assert D.is_signed(), "min_big_endian needs a signed integer"
    # Narrow while the bits from the next byte's sign bit up are all sign.
    var width = size_of[Scalar[D]]()
    while width > 1:
        var high = value >> Scalar[D](8 * (width - 1) - 1)
        if high != 0 and high != -1:
            break
        width -= 1
    var out = List[UInt8](capacity=width)
    BigEndian.put_signed(out, value, width)
    return out^


def from_big_endian[
    D: DType
](data: Span[UInt8, _]) raises CorruptError -> Scalar[D]:
    """The inverse of `min_big_endian`: sign-extend 1 to `size_of[D]` bytes of
    two's-complement big-endian to a `D`."""
    if len(data) == 0 or len(data) > size_of[Scalar[D]]():
        raise CorruptError(
            t"iceberg bound: a {D} decimal takes 1 to"
            t" {size_of[Scalar[D]]()} bytes, got {len(data)}"
        )
    return BigEndian.signed[D](data)


def _string_of(data: Span[UInt8, _]) -> String:
    """`data` in a `String` — how marrow's string *and* binary scalars hold
    their bytes; nothing here validates UTF-8."""
    return String(StringSlice(unsafe_from_utf8=data))


def _list_of(data: Span[UInt8, _]) -> List[UInt8]:
    var out = List[UInt8](capacity=len(data))
    out.extend(data)
    return out^


def _expect(
    dtype: DynType, data: Span[UInt8, _], width: Int
) raises CorruptError:
    if len(data) != width:
        raise CorruptError(
            t"iceberg bound: {dtype} takes {width} bytes, got {len(data)}"
        )


def decode_bound(
    dtype: DynType, data: Span[UInt8, _]
) raises DynError -> DynScalar:
    """The value of type `dtype` serialized as `data`.

    Raises `CorruptError` when `data` has the wrong length for `dtype`, and
    `TypeError` for a dtype Iceberg has no primitive type for.
    """
    _ = primitive_name(dtype)
    if dtype.is_bool():
        _expect(dtype, data, 1)
        return BoolScalar(data[0] != 0)
    elif dtype.is_decimal():

        def decimal[T: DecimalType](d: T) raises {imm} -> DynScalar:
            return PrimitiveScalar[T](from_big_endian[T.native](data), d)

        return dtype.dispatch_decimal(decimal)
    elif dtype.is_primitive():

        def fixed[T: PrimitiveType](d: T) raises {imm} -> DynScalar:
            _expect(dtype, data, size_of[Scalar[T.native]]())
            return PrimitiveScalar[T](LittleEndian.fixed[T.native](data, 0), d)

        return dtype.dispatch_primitive(fixed)
    elif dtype.is_string():
        return StringScalar(_string_of(data))
    elif dtype.is_large_string():
        return LargeStringScalar(_string_of(data))
    elif dtype.is_binary():
        return BinaryScalar(_string_of(data))
    elif dtype.is_large_binary():
        return LargeBinaryScalar(_string_of(data))
    elif dtype.is_fixed_size_binary():
        var width = dtype.as_fixed_size_binary().byte_width
        _expect(dtype, data, width)
        return FixedSizeBinaryScalar(_list_of(data), width)
    raise TypeError(t"iceberg bound: cannot decode {dtype}")


def encode_bound(value: DynScalar) raises -> List[UInt8]:
    """`value` in Iceberg's binary single-value serialization.

    Raises `InvalidError` for a null — a bound is never null — and `TypeError`
    for a dtype Iceberg has no primitive type for.
    """
    var dtype = value.type()
    _ = primitive_name(dtype)
    if value.is_null():
        raise InvalidError(t"iceberg bound: cannot encode a null {dtype}")
    var out = List[UInt8]()
    if dtype.is_bool():
        out.append(UInt8(1) if value.as_bool().value() else UInt8(0))
    elif dtype.is_decimal():

        def decimal[T: DecimalType](d: T) raises {imm} -> List[UInt8]:
            return min_big_endian(value.as_primitive[T]().value())

        out = dtype.dispatch_decimal(decimal)
    elif dtype.is_primitive():

        def fixed[T: PrimitiveType](d: T) raises {imm} -> List[UInt8]:
            # `as_bytes`, not `LittleEndian.append`: that one shifts, so it
            # takes integers only, and floats land here too.
            var le = (
                value.as_primitive[T]().value().as_bytes[big_endian=False]()
            )
            return _list_of(Span(le))

        out = dtype.dispatch_primitive(fixed)
    elif dtype.is_string():
        out = _list_of(value.as_string().value().as_bytes())
    elif dtype.is_large_string():
        out = _list_of(value.as_large_string().value().as_bytes())
    elif dtype.is_binary():
        out = _list_of(value.as_binary().value().as_bytes())
    elif dtype.is_large_binary():
        out = _list_of(value.as_large_binary().value().as_bytes())
    elif dtype.is_fixed_size_binary():
        out = value.as_fixed_size_binary().value().copy()
    else:
        raise TypeError(t"iceberg bound: cannot encode {dtype}")
    return out^
