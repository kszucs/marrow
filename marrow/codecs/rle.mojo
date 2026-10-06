# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Run-length encoding: a column as runs of one repeated value."""

from ..errors import CorruptError
from .bits import Bits
from .byteorder import Leb128
from .core import Append, Decoder, Emitter, Codec, Source, Values
from .plain import Plain


@fieldwise_init
struct Rle(Codec):
    """The run count, each run's value little-endian, then each run's length
    as a ULEB128. Values compare by their bits, so `-0.0` and `0.0` are
    different runs and a `NaN` survives."""

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        var runs = List[Scalar[T]]()
        var lengths = List[Int]()
        src.rewind()
        for k in range(len(src)):
            var v = src.next[T]()
            if k > 0 and Bits.of(v) == Bits.of(runs[len(runs) - 1]):
                lengths[len(lengths) - 1] += 1
            else:
                runs.append(v)
                lengths.append(1)
        Leb128.write(out, UInt64(len(runs)))
        Plain.encode[T](Values(Span(runs)), out)
        for n in lengths:
            Leb128.write(out, UInt64(n))

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        var nruns = src.length()
        if nruns > count:
            raise CorruptError(t"codecs: {nruns} runs for {count} values")
        var read = Append[T](List[Scalar[T]](capacity=nruns))
        read = Plain.decode[T](src, nruns, read^)
        var runs = read^.take()
        out.reserve(count)
        var total = 0
        for r in range(nruns):
            var n = src.length()
            if n > count - total:
                raise CorruptError(t"codecs: runs cover more than {count}")
            total += n
            for _ in range(n):
                out.emit(runs[r])
        if total != count:
            raise CorruptError(
                t"codecs: runs cover {total} values, not {count}"
            )
        return out^
