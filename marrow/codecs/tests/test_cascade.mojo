# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""A `Cascade` writes exactly the bytes its `DynCascade` writes, and reads
them back as the `DynCascade` does."""

from std.memory import bitcast
from std.testing import assert_raises, assert_true

from ...codecs import (
    BitPack,
    Bits,
    DynCascade,
    Codec,
    Delta,
    Cascade,
    Rle,
    Varint,
    Xor,
    Zigzag,
)
from ...utils.testing import Rng


def _same[T: DType](a: List[Scalar[T]], b: List[Scalar[T]]) -> Bool:
    """Equal bit for bit -- so a NaN equals itself and -0.0 is not 0.0."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[Bits.unsigned[T]](a[i]) != bitcast[Bits.unsigned[T]](b[i]):
            return False
    return True


def _columns[T: DType]() -> List[List[Scalar[T]]]:
    var cols = List[List[Scalar[T]]]()
    cols.append(List[Scalar[T]]())
    cols.append([Scalar[T](0)])
    cols.append([Scalar[T].MIN_FINITE, Scalar[T].MAX_FINITE])
    var sorted = List[Scalar[T]]()
    for i in range(37):
        sorted.append(Scalar[T](i * 3))
    cols.append(sorted^)
    var rng = Rng(7)
    var noise = List[Scalar[T]]()
    for _ in range(10_003):
        noise.append(bitcast[T](rng.next().cast[Bits.unsigned[T]]()))
    cols.append(noise^)
    return cols^


def _check[T: DType, *Ts: Codec]() raises:
    comptime F = Cascade[*Ts]
    var cascade: DynCascade = F()
    for ref values in _columns[T]():
        var fused = F.encode[T](values)
        var runtime = cascade.encode[T](values)
        assert_true(
            fused == runtime, String(cascade, "[", T, "]: bytes differ")
        )
        assert_true(_same(F.decode[T](fused), values), String(cascade))
        assert_true(_same(cascade.decode[T](fused), values), String(cascade))
        if len(values) < 100:
            for n in range(len(fused)):
                with assert_raises(contains="CorruptError"):
                    _ = F.decode[T](Span(fused)[:n])


def test_cascade_delta_zigzag_bitpack() raises:
    _check[DType.int64, Delta, Zigzag, BitPack]()
    _check[DType.int32, Delta, Zigzag, BitPack]()
    _check[DType.int8, Delta, Zigzag, BitPack]()


def test_cascade_double_delta() raises:
    _check[DType.int64, Delta, Delta, Zigzag, BitPack]()


def test_cascade_xor() raises:
    _check[DType.float64, Xor, BitPack]()
    _check[DType.float32, Xor, Rle]()


def test_cascade_after_bytes() raises:
    """A codec over the bytes another wrote, elementwise or not."""
    _check[DType.int64, BitPack, Rle]()
    _check[DType.int32, BitPack, Delta]()
    _check[DType.int64, Delta, Zigzag, BitPack, Rle]()


def test_cascade_elementwise_only() raises:
    _check[DType.int16, Delta, Zigzag]()
    _check[DType.int64, Delta, Zigzag, Varint]()
