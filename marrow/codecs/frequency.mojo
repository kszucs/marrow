# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Frequency: a column dominated by one value, as that value and the
exceptions to it."""

from std.memory import bitcast

from ..errors import CorruptError
from .bits import Bits
from .byteorder import Leb128, LittleEndian
from .core import Append, Decoder, Emitter, Codec, Source, Values
from .plain import Plain


@fieldwise_init
struct Frequency(Codec):
    """The most frequent value, the exception count, each exception's
    distance from the previous one's row as a ULEB128, then the exceptions'
    values little-endian. Ties go to the value seen first; values compare by
    their bits."""

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        var block = src.collect[T]()
        var values = Span(block)
        var counts = Dict[UInt64, Int]()
        var top = Scalar[T](0)
        var top_count = 0
        for v in values:
            var c = counts.get(Bits.of(v), 0) + 1
            counts[Bits.of(v)] = c
            if c > top_count:
                top = v
                top_count = c
        var gaps = List[Int]()
        var exceptions = List[Scalar[T]]()
        var next = 0
        for i in range(len(values)):
            if Bits.of(values[i]) != Bits.of(top):
                gaps.append(i - next)
                exceptions.append(values[i])
                next = i + 1
        comptime U = Bits.unsigned[T]
        LittleEndian.append[U](out, bitcast[U](top))
        Leb128.write(out, UInt64(len(gaps)))
        for g in gaps:
            Leb128.write(out, UInt64(g))
        Plain.encode[T](Values(Span(exceptions)), out)

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        var top = src.fixed[T]()
        var m = src.length()
        if m > count:
            raise CorruptError(t"codecs: {m} exceptions for {count} rows")
        var rows = List[Int](capacity=m)
        var next = 0
        for _ in range(m):
            var row = next + src.length()
            if row >= count:
                raise CorruptError(t"codecs: exception row {row} of {count}")
            rows.append(row)
            next = row + 1
        var read = Append[T](List[Scalar[T]](capacity=m))
        read = Plain.decode[T](src, m, read^)
        var exceptions = read^.take()
        out.reserve(count)
        var j = 0
        for row in range(count):
            if j < m and rows[j] == row:
                out.emit(exceptions[j])
                j += 1
            else:
                out.emit(top)
        return out^
