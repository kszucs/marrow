# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Iceberg partition transforms: what a partition spec's `transform` names.

`Transform.parse` reads the spec's spelling — `identity`, `bucket[N]`,
`truncate[W]`, `year`, `month`, `day`, `hour`, `void` — and `apply` computes a
partition value per row, null for a null. `result_type` is the spec's
partition transforms table: `bucket`, `year`, `month` and `hour` give `int`,
`day` gives `date`, and the rest keep the source type.

`bucket` hashes with 32-bit Murmur3 (x86, seed 0) over exactly the bytes spec
Appendix B names: `int`, `long` and `date` as an 8-byte little-endian long,
`time` and the timestamps as a long of microseconds — nanoseconds floored to
microseconds first — a decimal as its unscaled value's minimal big-endian
two's-complement bytes, and strings, uuids, `fixed` and `binary` as their raw
bytes. The time transforms count from 1970-01-01 and floor, so a pre-epoch
value lands in the earlier period. A zoned timestamp holds UTC instants, and
the transforms read them as UTC, as the spec requires.
"""

from ..arrays import DynArray, PrimitiveArray, BinaryLikeArray
from ..builders import BinaryLikeBuilder, Int32Builder, PrimitiveBuilder, nulls
from ..dtypes import (
    BinaryLikeType,
    DecimalType,
    DynType,
    PrimitiveType,
    StringLikeType,
    date32,
    int32,
    microsecond,
    nanosecond,
)
from ..errors import InvalidError, TypeError
from ..kernels.temporal import ticks_per_second
from ..codecs import LittleEndian
from ..utils import CivilDate, Epoch, floor_div
from .bounds import min_big_endian
from .types import parse_int, primitive_name


# ---------------------------------------------------------------------------
# Murmur3, x86 32-bit
# ---------------------------------------------------------------------------

comptime _C1: UInt32 = 0xCC9E2D51
comptime _C2: UInt32 = 0x1B873593


@always_inline
def _rotl(x: UInt32, r: UInt32) -> UInt32:
    return (x << r) | (x >> (32 - r))


@always_inline
def _mix_k(var k: UInt32) -> UInt32:
    k *= _C1
    k = _rotl(k, 15)
    return k * _C2


@always_inline
def _mix_h(h: UInt32, k: UInt32) -> UInt32:
    return _rotl(h ^ _mix_k(k), 13) * 5 + 0xE6546B64


@always_inline
def _finish(var h: UInt32, length: Int) -> Int32:
    h ^= UInt32(length)
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return Int32(h.cast[DType.int32]())


def murmur3_x86_32(data: Span[UInt8, _]) -> Int32:
    """32-bit Murmur3, x86 variant, seed 0: the hash Iceberg buckets by.

    Signed, as Java's `int` is, so the spec's test values compare directly."""
    var h = UInt32(0)
    var blocks = len(data) // 4
    for i in range(blocks):
        h = _mix_h(h, LittleEndian.fixed[DType.uint32](data, i * 4))
    var tail = blocks * 4
    var rest = len(data) - tail
    if rest > 0:
        var k = UInt32(0)
        if rest == 3:
            k ^= UInt32(data[tail + 2]) << 16
        if rest >= 2:
            k ^= UInt32(data[tail + 1]) << 8
        k ^= UInt32(data[tail])
        h ^= _mix_k(k)
    return _finish(h, len(data))


@always_inline
def _hash_long(v: Int64) -> Int32:
    """`murmur3_x86_32` of `v`'s 8 little-endian bytes, without them."""
    var bits = UInt64(v)
    var h = _mix_h(UInt32(0), UInt32(bits & 0xFFFFFFFF))
    h = _mix_h(h, UInt32(bits >> 32))
    return _finish(h, 8)


@always_inline
def _bucket_of(hash: Int32, n: Int) -> Int32:
    return Int32((Int(hash) & 0x7FFFFFFF) % n)


# ---------------------------------------------------------------------------
# Per-family loops
# ---------------------------------------------------------------------------


def _micros(v: Int, per_second: Int) -> Int:
    """`v` ticks of `per_second` a second as microseconds, floored."""
    if per_second > 1_000_000:
        return floor_div(v, per_second // 1_000_000)
    return v * (1_000_000 // per_second)


def _bucket_longs[
    T: PrimitiveType
](arr: PrimitiveArray[T], n: Int, per_second: Int) raises -> DynArray:
    """The bucket of each value as a long; a time or timestamp of `per_second`
    ticks a second is hashed as microseconds, anything else with
    `per_second == 1_000_000` as it is."""
    var b = Int32Builder(len(arr))
    for i in range(len(arr)):
        if arr.is_valid(i):
            var key = _micros(Int(arr.unsafe_get(i)), per_second)
            b.unsafe_append(_bucket_of(_hash_long(Int64(key)), n))
        else:
            b.unsafe_append_null()
    return b.finish()


def _bucket_decimals[
    T: DecimalType
](arr: PrimitiveArray[T], n: Int) raises -> DynArray:
    var b = Int32Builder(len(arr))
    for i in range(len(arr)):
        if arr.is_valid(i):
            var bytes = min_big_endian(arr.unsafe_get(i))
            b.unsafe_append(_bucket_of(murmur3_x86_32(bytes), n))
        else:
            b.unsafe_append_null()
    return b.finish()


def _bucket_bytes[
    T: BinaryLikeType
](arr: BinaryLikeArray[T], n: Int) raises -> DynArray:
    var b = Int32Builder(len(arr))
    for i in range(len(arr)):
        if arr.is_valid(i):
            var hash = murmur3_x86_32(arr.unsafe_get(UInt(i)).as_bytes())
            b.unsafe_append(_bucket_of(hash, n))
        else:
            b.unsafe_append_null()
    return b.finish()


def _truncate_values[
    T: PrimitiveType
](arr: PrimitiveArray[T], width: Int) raises -> DynArray:
    """`v - (((v % W) + W) % W)` — floor to a multiple of `W` — on the
    native value: the integer itself, or a decimal's unscaled value."""
    var w = Scalar[T.native](width)
    var b = PrimitiveBuilder[T](arr.dtype, len(arr))
    for i in range(len(arr)):
        if arr.is_valid(i):
            var v = arr.unsafe_get(i)
            b.unsafe_append(v - (((v % w) + w) % w))
        else:
            b.unsafe_append_null()
    return b.finish()


def _truncate_bytes[
    T: BinaryLikeType
](arr: BinaryLikeArray[T], width: Int) raises -> DynArray:
    """The first `width` code points of a string, or bytes of a binary."""
    var b = BinaryLikeBuilder[T](len(arr))
    for i in range(len(arr)):
        if arr.is_valid(i):
            var bytes = arr.unsafe_get(UInt(i)).as_bytes()
            var end = min(width, len(bytes))
            comptime if conforms_to(T, StringLikeType):
                # Stop at the start byte of code point `width + 1`.
                var seen = 0
                end = len(bytes)
                for j in range(len(bytes)):
                    if (bytes[j] & 0xC0) != 0x80:
                        if seen == width:
                            end = j
                            break
                        seen += 1
            b.append(StringSlice(unsafe_from_utf8=bytes[:end]))
        else:
            b.append_null()
    return b.finish()


def _calendar_values[
    T: PrimitiveType, O: PrimitiveType
](
    arr: PrimitiveArray[T],
    out_dtype: O,
    kind: Int,
    ticks_per_day: Int,
    ticks_per_hour: Int,
) raises -> DynArray:
    """`year`, `month`, `day` or `hour` of a date (`ticks_per_day == 1`) or a
    timestamp, counted from 1970-01-01."""
    var b = PrimitiveBuilder[O](out_dtype, len(arr))
    for i in range(len(arr)):
        if arr.is_valid(i):
            var v = Int(arr.unsafe_get(i))
            var r: Int
            if kind == Transform.HOUR:
                r = floor_div(v, ticks_per_hour)
            else:
                var days = floor_div(v, ticks_per_day)
                if kind == Transform.DAY:
                    r = days
                else:
                    var civil = CivilDate.from_days(days)
                    if kind == Transform.YEAR:
                        r = civil.year - 1970
                    else:
                        r = (civil.year - 1970) * 12 + civil.month - 1
            b.unsafe_append(Scalar[O.native](r))
        else:
            b.unsafe_append_null()
    return b.finish()


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def _is_timestamp(dtype: DynType) -> Bool:
    if not dtype.is_timestamp():
        return False
    var unit = dtype.as_timestamp().unit
    return unit == microsecond or unit == nanosecond


@fieldwise_init
struct Transform(Copyable, Equatable, Movable, Writable):
    """A partition transform: a kind and, for `bucket` and `truncate`, its
    `N` or `W`."""

    comptime IDENTITY = 0
    comptime BUCKET = 1
    comptime TRUNCATE = 2
    comptime YEAR = 3
    comptime MONTH = 4
    comptime DAY = 5
    comptime HOUR = 6
    comptime VOID = 7

    var kind: Int
    var parameter: Int
    """`N` for `bucket[N]`, `W` for `truncate[W]`, 0 otherwise."""

    @staticmethod
    def parse(text: StringSlice) raises -> Self:
        """The transform the spec spells `text`; `InvalidError` otherwise."""
        var name = text.strip()
        if name == "identity":
            return Self(Self.IDENTITY, 0)
        elif name == "year":
            return Self(Self.YEAR, 0)
        elif name == "month":
            return Self(Self.MONTH, 0)
        elif name == "day":
            return Self(Self.DAY, 0)
        elif name == "hour":
            return Self(Self.HOUR, 0)
        elif name == "void":
            return Self(Self.VOID, 0)
        var kind: Int
        var rest: String
        if name.startswith("bucket[") and name.endswith("]"):
            kind = Self.BUCKET
            rest = String(name.removeprefix("bucket[").removesuffix("]"))
        elif name.startswith("truncate[") and name.endswith("]"):
            kind = Self.TRUNCATE
            rest = String(name.removeprefix("truncate[").removesuffix("]"))
        else:
            raise InvalidError(t"iceberg transform: unknown transform '{name}'")
        var value = parse_int(
            rest, String(t"iceberg transform: '{name}' parameter")
        )
        if value <= 0:
            raise InvalidError(
                t"iceberg transform: '{name}' needs a positive parameter"
            )
        return Self(kind, value)

    def __eq__(self, other: Self) -> Bool:
        return self.kind == other.kind and self.parameter == other.parameter

    def write_to[W: Writer](self, mut writer: W):
        if self.kind == Self.IDENTITY:
            writer.write("identity")
        elif self.kind == Self.BUCKET:
            writer.write("bucket[", self.parameter, "]")
        elif self.kind == Self.TRUNCATE:
            writer.write("truncate[", self.parameter, "]")
        elif self.kind == Self.YEAR:
            writer.write("year")
        elif self.kind == Self.MONTH:
            writer.write("month")
        elif self.kind == Self.DAY:
            writer.write("day")
        elif self.kind == Self.HOUR:
            writer.write("hour")
        else:
            writer.write("void")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def result_type(self, source: DynType) raises -> DynType:
        """The partition value's type for a `source` column, or `TypeError`
        when the transform does not take `source`."""
        if self.kind == Self.VOID:
            return source.copy()
        # Every transform but `void` takes Iceberg primitives only.
        _ = primitive_name(source)
        if self.kind == Self.IDENTITY:
            return source.copy()
        elif self.kind == Self.BUCKET:
            if not (
                source.is_bool() or source.is_float32() or source.is_float64()
            ):
                return int32
        elif self.kind == Self.TRUNCATE:
            if (
                source.is_int32()
                or source.is_int64()
                or source.is_decimal()
                or source.is_binary_like()
            ):
                return source.copy()
        elif self.kind == Self.HOUR:
            if _is_timestamp(source):
                return int32
        elif source.is_date32() or _is_timestamp(source):
            if self.kind == Self.DAY:
                return date32()
            return int32
        raise TypeError(t"iceberg transform: {self} does not take {source}")

    def apply(self, array: DynArray) raises -> DynArray:
        """The partition value of every row of `array`; null stays null."""
        var dtype = array.dtype()
        var result = self.result_type(dtype)
        if self.kind == Self.IDENTITY:
            return array.copy()
        elif self.kind == Self.VOID:
            return nulls(array.length(), result)
        elif self.kind == Self.BUCKET:
            return self._bucket(array, dtype)
        elif self.kind == Self.TRUNCATE:
            return self._truncate(array, dtype)
        return self._calendar(array, dtype)

    def _bucket(self, array: DynArray, dtype: DynType) raises -> DynArray:
        var n = self.parameter
        if dtype.is_int32():
            return _bucket_longs(array.as_int32(), n, 1_000_000)
        elif dtype.is_int64():
            return _bucket_longs(array.as_int64(), n, 1_000_000)
        elif dtype.is_date32():
            return _bucket_longs(array.as_date32(), n, 1_000_000)
        elif dtype.is_time64():
            return _bucket_longs(array.as_time64(), n, ticks_per_second(dtype))
        elif dtype.is_timestamp():
            return _bucket_longs(
                array.as_timestamp(), n, ticks_per_second(dtype)
            )
        elif dtype.is_decimal():

            def decimal[T: DecimalType](d: T) raises {imm} -> DynArray:
                return _bucket_decimals(array.as_primitive[T](), n)

            return dtype.dispatch_decimal(decimal)
        elif dtype.is_binary_like():

            def binarylike[T: BinaryLikeType](d: T) raises {imm} -> DynArray:
                return _bucket_bytes(array.as_binary_like[T](), n)

            return dtype.dispatch_binarylike(binarylike)
        # fixed and uuid: the raw bytes of each slot.
        ref arr = array.as_fixed_size_binary()
        var width = arr.byte_width
        var b = Int32Builder(len(arr))
        for i in range(len(arr)):
            if arr.is_valid(i):
                var start = (arr.offset + i) * width
                var bytes = arr.buffer.slice(start, width).as_span()
                b.unsafe_append(_bucket_of(murmur3_x86_32(bytes), n))
            else:
                b.unsafe_append_null()
        return b.finish()

    def _truncate(self, array: DynArray, dtype: DynType) raises -> DynArray:
        var w = self.parameter
        if dtype.is_int32():
            return _truncate_values(array.as_int32(), w)
        elif dtype.is_int64():
            return _truncate_values(array.as_int64(), w)
        elif dtype.is_decimal():

            def decimal[T: DecimalType](d: T) raises {imm} -> DynArray:
                return _truncate_values(array.as_primitive[T](), w)

            return dtype.dispatch_decimal(decimal)

        def binarylike[T: BinaryLikeType](d: T) raises {imm} -> DynArray:
            return _truncate_bytes(array.as_binary_like[T](), w)

        return dtype.dispatch_binarylike(binarylike)

    def _calendar(self, array: DynArray, dtype: DynType) raises -> DynArray:
        if dtype.is_date32():
            if self.kind == Self.DAY:
                return array.copy()
            return _calendar_values(array.as_date32(), int32, self.kind, 1, 0)
        var tps = ticks_per_second(dtype)
        var per_day = tps * Epoch.SECONDS_PER_DAY
        var per_hour = tps * 3600
        if self.kind == Self.DAY:
            return _calendar_values(
                array.as_timestamp(), date32(), self.kind, per_day, per_hour
            )
        return _calendar_values(
            array.as_timestamp(), int32, self.kind, per_day, per_hour
        )
