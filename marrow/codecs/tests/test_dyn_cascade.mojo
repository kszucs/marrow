# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Dynamic cascades: built, printed and checked; streams decoded with the chain that
wrote them, and malformed ones refused."""

from std.testing import assert_equal, assert_raises, assert_true

from ...codecs import (
    BitPack,
    DynCascade,
    DynCascadeReader,
    DynCascadeWriter,
    Constant,
    Delta,
    DynCodec,
    CascadeReader,
    CascadeWriter,
    Plain,
    Rle,
    Varint,
    Xor,
    Zigzag,
)


def _timestamps() -> List[Int64]:
    var out = List[Int64]()
    var t = Int64(1_700_000_000_000_000)
    for i in range(1000):
        t += Int64(1_000 + (i * 7919) % 500)
        out.append(t)
    return out^


def test_dyn_cascade_chain_round_trips_and_shrinks() raises:
    var values = _timestamps()
    var chain = DynCascade(Delta(), Zigzag(), BitPack())
    var data = chain.encode[DType.int64](values)
    assert_true(chain.decode[DType.int64](data) == values)
    var plain = DynCascade(List[DynCodec]()).encode[DType.int64](values)
    assert_true(len(data) * 4 < len(plain))


def test_dyn_cascade_double_delta() raises:
    """Evenly spaced values are all-zero second differences."""
    var values = List[Int64]()
    for i in range(500):
        values.append(Int64(1_000_000 + i * 60))
    var chain = DynCascade(Delta(), Delta(), Zigzag(), BitPack())
    var data = chain.encode[DType.int64](values)
    assert_true(chain.decode[DType.int64](data) == values)
    assert_true(len(data) < 32)


def test_dyn_cascade_elementwise_only_and_empty() raises:
    """A chain that writes no bytes stores its values plain."""
    var values: List[Int32] = [5, -3, 7, 7, 0]
    for chain in [DynCascade(Delta(), Zigzag()), DynCascade(List[DynCodec]())]:
        var data = chain.encode[DType.int32](values)
        assert_true(chain.decode[DType.int32](data) == values)


def test_dyn_cascade_printing() raises:
    assert_equal(
        String(DynCascade(Delta(), Zigzag(), BitPack())),
        "Delta -> Zigzag -> BitPack",
    )
    assert_equal(String(DynCascade(Rle())), "Rle")
    assert_equal(String(DynCascade(List[DynCodec]())), "Plain")


def test_dyn_cascade_follows_a_codec_that_wrote_bytes() raises:
    """Any codec may follow one that wrote bytes, if it takes them."""
    var values = _timestamps()
    for chain in [DynCascade(BitPack(), Rle()), DynCascade(BitPack(), Delta())]:
        var data = chain.encode[DType.int64](values)
        assert_true(chain.decode[DType.int64](data) == values, String(chain))
    with assert_raises(contains="Zigzag does not take uint8"):
        _ = DynCascade(BitPack(), Zigzag()).encode[DType.int64](values)


def test_dyn_cascade_types_are_checked() raises:
    var unsigned: List[UInt32] = [1, 2, 3]
    with assert_raises(contains="Zigzag does not take uint32"):
        _ = DynCascade(Zigzag(), Varint()).encode[DType.uint32](unsigned)
    var floats: List[Float64] = [1.5, 2.5]
    with assert_raises(contains="Delta does not take float64"):
        _ = DynCascade(Delta(), BitPack()).encode[DType.float64](floats)
    var ints: List[Int64] = [1, 2, 3]
    var chain = DynCascade(Zigzag(), Varint())
    var data = chain.encode[DType.int64](ints)
    with assert_raises(contains="Zigzag does not take float64"):
        _ = chain.decode[DType.float64](data)


def test_dyn_cascade_corrupt_streams() raises:
    var values: List[Int64] = [1, 2, 3, 4]
    var chain = DynCascade(Delta(), BitPack())
    var data = chain.encode[DType.int64](values)
    var trailing = data.copy()
    trailing.append(0)
    with assert_raises(contains="bytes left after the column"):
        _ = chain.decode[DType.int64](trailing)
    for n in range(len(data)):
        with assert_raises(contains="CorruptError"):
            _ = chain.decode[DType.int64](Span(data)[:n])


def test_dyn_cascade_xor_floats() raises:
    var values = List[Float64]()
    for i in range(300):
        values.append(20.0 + Float64(i % 50) * 0.25)
    var chain = DynCascade(Xor(), BitPack())
    var data = chain.encode[DType.float64](values)
    assert_true(chain.decode[DType.float64](data) == values)


def test_dyn_cascade_constant() raises:
    var same = List[Float32](length=1000, fill=2.5)
    var chain = DynCascade(Constant())
    var data = chain.encode[DType.float32](same)
    # a block of 1000 values -- its count, its length, the value once -- and
    # the end
    assert_equal(len(data), 2 + 4 + 4 + 1)
    assert_true(chain.decode[DType.float32](data) == same)
    same[999] = 3.0
    with assert_raises(contains="Constant got a column holding"):
        _ = DynCascade(Constant()).encode[DType.float32](same)


def test_dyn_cascade_plain_is_what_a_chain_stores() raises:
    """`Plain` writes what a chain that writes no bytes stores."""
    var values: List[Int64] = [5, -3, 7, 7, 0]
    var data = DynCascade(Delta(), Plain()).encode[DType.int64](values)
    assert_true(data == DynCascade(Delta()).encode[DType.int64](values))
    assert_true(DynCascade(Delta()).decode[DType.int64](data) == values)
    assert_equal(String(DynCodec(Plain())), "Plain")


def test_streaming_writes_the_same_bytes_in_any_chunks() raises:
    var values = List[Int64]()
    var t = Int64(1_700_000_000_000_000)
    for i in range(10_007):
        t += Int64(1_000 + (i * 7919) % 500)
        values.append(t)
    var whole = DynCascade(Delta(), Zigzag(), BitPack()).encode[DType.int64](
        values, block=1000
    )
    for chunk in [1, 7, 999, 1000, 4096]:
        var runtime = DynCascadeWriter[DType.int64](
            DynCascade(Delta(), Zigzag(), BitPack()), 1000
        )
        var fused = CascadeWriter[DType.int64, Delta, Zigzag, BitPack](1000)
        var a = List[UInt8]()
        var b = List[UInt8]()
        var i = 0
        while i < len(values):
            var part = Span(values)[i : min(i + chunk, len(values))]
            runtime.write(part, a)
            fused.write(part, b)
            i += chunk
        runtime.finish(a)
        fused.finish(b)
        assert_true(a == whole, String("runtime, chunks of ", chunk))
        assert_true(b == whole, String("fused, chunks of ", chunk))


def test_streaming_reads_a_block_at_a_time() raises:
    var values = List[Int32]()
    for i in range(2_500):
        values.append(Int32(i * 3 - 1000))
    var data = DynCascade(Delta(), Zigzag(), BitPack()).encode[DType.int32](
        values, block=1000
    )
    var runtime = DynCascadeReader[DType.int32](
        DynCascade(Delta(), Zigzag(), BitPack()), Span(data)
    )
    var fused = CascadeReader[DType.int32, Delta, Zigzag, BitPack](Span(data))
    var all_runtime = List[Int32]()
    var all_fused = List[Int32]()
    var sizes = List[Int]()
    var block = List[Int32]()
    while runtime.read(block):
        sizes.append(len(block))
        all_runtime.extend(Span(block))
        block.clear()
    while fused.read(all_fused):
        pass
    var want: List[Int] = [1000, 1000, 500]
    assert_true(sizes == want)
    assert_true(all_runtime == values)
    assert_true(all_fused == values)
