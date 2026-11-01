# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Each codec over 64K values, and each byte-string packer over 64K strings:
encode, and decode what was encoded. The inputs are shaped for each codec.

Run with:
    pixi run -e dev pytest marrow/kernels/tests/bench_codecs.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep
from std.builtin.rebind import rebind_var
from std.sys import size_of

from ...utils.testing import Benchmark
from ...views import BufferView
from ...kernels.bitwise import (
    OrderedDecode,
    OrderedEncode,
    ZigzagDecode,
    ZigzagEncode,
)
from ...kernels.compose import (
    Auto,
    ChoicePack,
    Codec,
    Compose,
    Decoder,
    Encoder,
)
from ...kernels.core import PackKernel, UnpackKernel
from ...kernels.dictionary import (
    DictionaryBytesPack,
    DictionaryBytesPacked,
    DictionaryBytesUnpack,
    DictionaryPack,
    DictionaryUnpack,
)
from ...kernels.float import AlpPack, AlpRdPack, AlpRdUnpack, AlpUnpack
from ...kernels.packing import (
    BLOCK,
    BitPack,
    BitUnpack,
    ConstantPack,
    ConstantUnpack,
    FrequencyPack,
    FrequencyUnpack,
    FsstPack,
    FsstPacked,
    FsstUnpack,
    LengthsPack,
    LengthsPacked,
    LengthsUnpack,
    PlainPack,
    PlainUnpack,
    PforPack,
    PforUnpack,
    PrefixesPack,
    PrefixesPacked,
    PrefixesUnpack,
    ReferencePack,
    ReferenceUnpack,
    SplitPack,
    SplitUnpack,
)
from ...kernels.runs import RunsPack, RunsUnpack
from ...kernels.scan import DiffScan, SumScan

comptime N = 1 << 16


def _data[T: DType, shape: StaticString]() -> List[Scalar[T]]:
    var out = List[Scalar[T]](capacity=N)
    var t = Int64(1_700_000_000)
    for i in range(N):
        comptime if shape == "timestamps":
            t += Int64(1_000 + (i * 7919) % 500)
            out.append(Scalar[T](t))
        elif shape == "small":
            out.append(Scalar[T]((i * 7919) % 100))
        elif shape == "runs":
            out.append(Scalar[T]((i // 50) % 7))
        elif shape == "cardinality":
            out.append(Scalar[T]((i * 7919) % 1000) * 1_003)
        elif shape == "constant":
            out.append(Scalar[T](42))
        elif shape == "interval":
            # Timestamps every second, a jitter of a few milliseconds.
            t += Int64(1_000 + (i * 7919) % 7 - 3)
            out.append(Scalar[T](t))
        elif shape == "signed":
            out.append(Scalar[T]((i * 7919) % 201 - 100))
        elif shape == "sparse":
            out.append(Scalar[T](0 if i % 1000 < 990 else (i * 7919) % 100_000))
        elif shape == "outliers":
            # Small values, every hundredth far past them.
            out.append(
                Scalar[T]((1 << 40) + i if i % 100 == 0 else (i * 7919) % 100)
            )
        elif shape == "gauge":
            # A reading held for ten samples at a time.
            out.append(
                Scalar[T](
                    Float64(20 + (i // 100) % 10)
                    + Float64((i // 10) % 7) * 0.125
                )
            )
        elif shape == "real":
            # Doubles with more digits than a decimal integer holds.
            out.append(
                Scalar[T](
                    Float64(i) * 0.7071067811865476
                    + Float64((i * 7919) % 1000) * 1e-9
                )
            )
        else:
            out.append(Scalar[T](Float64(i % 1000) * 0.25 + 20.0))
    return out^


def _ratio(raw: Int, packed: Int) -> String:
    """Raw bytes over packed bytes, to two decimals."""
    return String(round(Float64(raw) / Float64(max(packed, 1)), 2))


def _bench[
    T: DType,
    shape: StaticString,
    decode: Bool,
    E: PackKernel,
    D: UnpackKernel,
](mut b: Benchmark, var codec: Codec[E, D]) raises:
    var values = _data[T, shape]()
    var packed = codec.encode(BufferView(Span(values)))
    var out = List[Scalar[T]](length=N, fill=0)
    codec.decode[T](packed.copy(), BufferView(Span(out)))
    for i in range(N):
        if out[i] != values[i]:
            raise Error("bench: the codec does not round-trip")
    b.throughput(BenchMetric.elements, N)
    b.extra_info("shape", shape)
    b.extra_info(
        "ratio",
        _ratio(N * size_of[Scalar[T]](), codec.encoder.nbytes[T](packed)),
    )

    @always_inline
    def call() raises {mut out, imm}:
        comptime if decode:
            codec.decode[T](packed.copy(), BufferView(Span(out)))
            keep(out[N - 1])
        else:
            var again = codec.encode(BufferView(Span(values)))
            keep(again)

    b.iter(call)
    keep(values)
    keep(out)
    keep(packed)
    keep(codec)


def _strings(mut offsets: List[Int32], mut data: List[UInt8]):
    offsets.append(0)
    for i in range(N):
        data.extend(String("row-", i // 10, "-", i * 7919 % 1000).as_bytes())
        offsets.append(Int32(len(data)))


# `_bench`, for byte strings: one per pair, as the pairs share no trait.


def _bench_bytes[
    decode: Bool, L: PackKernel, M: UnpackKernel
](
    mut b: Benchmark, var packer: LengthsPack[L], var unpacker: LengthsUnpack[M]
) raises:
    comptime assert L.Packed == M.Packed, "the unpacker takes what is packed"
    var offsets = List[Int32](capacity=N + 1)
    var data = List[UInt8]()
    _strings(offsets, data)
    var packed = packer.pack(Span(offsets), Span(data))
    var ends: List[Int32] = [0]
    var bytes = List[UInt8]()
    unpacker.unpack(
        rebind_var[LengthsPacked[M.Packed]](packed.copy()), N, ends, bytes
    )
    if not (ends == offsets and bytes == data):
        raise Error("bench: the packers do not round-trip")
    b.throughput(BenchMetric.elements, N)
    b.extra_info("shape", "strings")
    b.extra_info(
        "ratio", _ratio(len(data) + 4 * (N + 1), packer.nbytes(packed))
    )

    @always_inline
    def call() raises {imm}:
        comptime if decode:
            var ends: List[Int32] = [0]
            var bytes = List[UInt8]()
            unpacker.unpack(
                rebind_var[LengthsPacked[M.Packed]](packed.copy()),
                N,
                ends,
                bytes,
            )
            keep(len(bytes))
        else:
            var again = packer.pack(Span(offsets), Span(data))
            keep(again)

    b.iter(call)
    keep(offsets)
    keep(data)
    keep(packed)
    keep(packer)
    keep(unpacker)


def _bench_bytes[
    decode: Bool,
    P: PackKernel,
    L: PackKernel,
    Q: UnpackKernel,
    M: UnpackKernel,
](
    mut b: Benchmark,
    var packer: PrefixesPack[P, L],
    var unpacker: PrefixesUnpack[Q, M],
) raises:
    comptime assert (
        P.Packed == Q.Packed and L.Packed == M.Packed
    ), "the unpacker takes what is packed"
    var offsets = List[Int32](capacity=N + 1)
    var data = List[UInt8]()
    _strings(offsets, data)
    var packed = packer.pack(Span(offsets), Span(data))
    var ends: List[Int32] = [0]
    var bytes = List[UInt8]()
    unpacker.unpack(
        rebind_var[PrefixesPacked[Q.Packed, M.Packed]](packed.copy()),
        N,
        ends,
        bytes,
    )
    if not (ends == offsets and bytes == data):
        raise Error("bench: the packers do not round-trip")
    b.throughput(BenchMetric.elements, N)
    b.extra_info("shape", "strings")
    b.extra_info(
        "ratio", _ratio(len(data) + 4 * (N + 1), packer.nbytes(packed))
    )

    @always_inline
    def call() raises {imm}:
        comptime if decode:
            var ends: List[Int32] = [0]
            var bytes = List[UInt8]()
            unpacker.unpack(
                rebind_var[PrefixesPacked[Q.Packed, M.Packed]](packed.copy()),
                N,
                ends,
                bytes,
            )
            keep(len(bytes))
        else:
            var again = packer.pack(Span(offsets), Span(data))
            keep(again)

    b.iter(call)
    keep(offsets)
    keep(data)
    keep(packed)
    keep(packer)
    keep(unpacker)


def _bench_bytes[
    decode: Bool, C: PackKernel, D: UnpackKernel
](
    mut b: Benchmark,
    var packer: DictionaryBytesPack[C],
    var unpacker: DictionaryBytesUnpack[D],
) raises:
    comptime assert C.Packed == D.Packed, "the unpacker takes what is packed"
    var offsets = List[Int32](capacity=N + 1)
    var data = List[UInt8]()
    _strings(offsets, data)
    var packed = packer.pack(Span(offsets), Span(data))
    var ends: List[Int32] = [0]
    var bytes = List[UInt8]()
    unpacker.unpack(
        rebind_var[DictionaryBytesPacked[D.Packed]](packed.copy()),
        N,
        ends,
        bytes,
    )
    if not (ends == offsets and bytes == data):
        raise Error("bench: the packers do not round-trip")
    b.throughput(BenchMetric.elements, N)
    b.extra_info("shape", "strings")
    b.extra_info(
        "ratio", _ratio(len(data) + 4 * (N + 1), packer.nbytes(packed))
    )

    @always_inline
    def call() raises {imm}:
        comptime if decode:
            var ends: List[Int32] = [0]
            var bytes = List[UInt8]()
            unpacker.unpack(
                rebind_var[DictionaryBytesPacked[D.Packed]](packed.copy()),
                N,
                ends,
                bytes,
            )
            keep(len(bytes))
        else:
            var again = packer.pack(Span(offsets), Span(data))
            keep(again)

    b.iter(call)
    keep(offsets)
    keep(data)
    keep(packed)
    keep(packer)
    keep(unpacker)


def _bench_bytes[
    decode: Bool, L: PackKernel, M: UnpackKernel
](
    mut b: Benchmark, var packer: FsstPack[L], var unpacker: FsstUnpack[M]
) raises:
    comptime assert L.Packed == M.Packed, "the unpacker takes what is packed"
    var offsets = List[Int32](capacity=N + 1)
    var data = List[UInt8]()
    _strings(offsets, data)
    var packed = packer.pack(Span(offsets), Span(data))
    var ends: List[Int32] = [0]
    var bytes = List[UInt8]()
    unpacker.unpack(
        rebind_var[FsstPacked[M.Packed]](packed.copy()), N, ends, bytes
    )
    if not (ends == offsets and bytes == data):
        raise Error("bench: the packers do not round-trip")
    b.throughput(BenchMetric.elements, N)
    b.extra_info("shape", "strings")
    b.extra_info(
        "ratio", _ratio(len(data) + 4 * (N + 1), packer.nbytes(packed))
    )

    @always_inline
    def call() raises {imm}:
        comptime if decode:
            var ends: List[Int32] = [0]
            var bytes = List[UInt8]()
            unpacker.unpack(
                rebind_var[FsstPacked[M.Packed]](packed.copy()),
                N,
                ends,
                bytes,
            )
            keep(len(bytes))
        else:
            var again = packer.pack(Span(offsets), Span(data))
            keep(again)

    b.iter(call)
    keep(offsets)
    keep(data)
    keep(packed)
    keep(packer)
    keep(unpacker)


def _plain() -> Codec[PlainPack, PlainUnpack]:
    return Codec(PlainPack(), PlainUnpack())


def _bss() -> Codec[SplitPack, SplitUnpack]:
    return Codec(SplitPack(), SplitUnpack())


def _bitpack() -> Codec[BitPack, BitUnpack]:
    return Codec(BitPack(), BitUnpack())


def _reference() -> (
    Codec[ReferencePack[PlainPack], ReferenceUnpack[PlainUnpack]]
):
    return Codec(
        ReferencePack(BLOCK, PlainPack()), ReferenceUnpack(BLOCK, PlainUnpack())
    )


def _runs() -> (
    Codec[RunsPack[PlainPack, PlainPack], RunsUnpack[PlainUnpack, PlainUnpack]]
):
    return Codec(
        RunsPack(PlainPack(), PlainPack()),
        RunsUnpack(PlainUnpack(), PlainUnpack()),
    )


def _delta() -> (
    Codec[
        Encoder[Compose[DiffScan], PlainPack],
        Decoder[PlainUnpack, Compose[SumScan], DType.int64],
    ]
):
    return Codec(
        Encoder(Compose((DiffScan(),)), PlainPack()),
        Decoder[PlainUnpack, Compose[SumScan], DType.int64](
            PlainUnpack(), Compose((SumScan(),))
        ),
    )


def _ends_pack() -> Encoder[Compose[DiffScan], BitPack]:
    """Run ends packed as the differences between them, the runs' lengths."""
    return Encoder(Compose((DiffScan(),)), BitPack())


def _ends_unpack() -> Decoder[BitUnpack, Compose[SumScan], DType.uint64]:
    return Decoder[BitUnpack, Compose[SumScan], DType.uint64](
        BitUnpack(), Compose((SumScan(),))
    )


def _float_planes() -> (
    Codec[
        RunsPack[SplitPack, Encoder[Compose[DiffScan], BitPack]],
        RunsUnpack[
            SplitUnpack, Decoder[BitUnpack, Compose[SumScan], DType.uint64]
        ],
    ]
):
    return Codec(
        RunsPack(SplitPack(), _ends_pack()),
        RunsUnpack(SplitUnpack(), _ends_unpack()),
    )


def _dictionary() -> (
    Codec[DictionaryPack[PlainPack], DictionaryUnpack[PlainUnpack]]
):
    return Codec(DictionaryPack(PlainPack()), DictionaryUnpack(PlainUnpack()))


def _zigzag() -> (
    Codec[
        Encoder[Compose[ZigzagEncode], PlainPack],
        Decoder[PlainUnpack, Compose[ZigzagDecode], DType.uint64],
    ]
):
    return Codec(
        Encoder(Compose((ZigzagEncode(),)), PlainPack()),
        Decoder[PlainUnpack, Compose[ZigzagDecode], DType.uint64](
            PlainUnpack(), Compose((ZigzagDecode(),))
        ),
    )


def _ordered() -> (
    Codec[
        Encoder[Compose[OrderedEncode], PlainPack],
        Decoder[
            PlainUnpack, Compose[OrderedDecode[DType.float64]], DType.uint64
        ],
    ]
):
    return Codec(
        Encoder(Compose((OrderedEncode(),)), PlainPack()),
        Decoder[
            PlainUnpack, Compose[OrderedDecode[DType.float64]], DType.uint64
        ](PlainUnpack(), Compose((OrderedDecode[DType.float64](),))),
    )


def _delta_zigzag_pack() -> Encoder[Compose[DiffScan, ZigzagEncode], BitPack]:
    return Encoder(Compose((DiffScan(), ZigzagEncode())), BitPack())


def _delta_zigzag_unpack[
    U: DType
]() -> Decoder[BitUnpack, Compose[ZigzagDecode, SumScan], U]:
    return Decoder[BitUnpack, Compose[ZigzagDecode, SumScan], U](
        BitUnpack(), Compose((ZigzagDecode(), SumScan()))
    )


def _cascade_double_delta() -> (
    Codec[
        Encoder[Compose[DiffScan, DiffScan, ZigzagEncode], BitPack],
        Decoder[
            BitUnpack, Compose[ZigzagDecode, SumScan, SumScan], DType.uint64
        ],
    ]
):
    return Codec(
        Encoder(Compose((DiffScan(), DiffScan(), ZigzagEncode())), BitPack()),
        Decoder[
            BitUnpack, Compose[ZigzagDecode, SumScan, SumScan], DType.uint64
        ](BitUnpack(), Compose((ZigzagDecode(), SumScan(), SumScan()))),
    )


def _cascade_t64(
    size: Int, width: Int
) -> Codec[
    Encoder[Compose[OrderedEncode], ReferencePack[BitPack]],
    Decoder[
        ReferenceUnpack[BitUnpack],
        Compose[OrderedDecode[DType.int64]],
        DType.uint64,
    ],
]:
    return Codec(
        Encoder(
            Compose((OrderedEncode(),)), ReferencePack(size, BitPack(width))
        ),
        Decoder[
            ReferenceUnpack[BitUnpack],
            Compose[OrderedDecode[DType.int64]],
            DType.uint64,
        ](
            ReferenceUnpack(size, BitUnpack(width)),
            Compose((OrderedDecode[DType.int64](),)),
        ),
    )


def _cascade_delta_t64() -> (
    Codec[
        Encoder[Compose[DiffScan, OrderedEncode], ReferencePack[BitPack]],
        Decoder[
            ReferenceUnpack[BitUnpack],
            Compose[OrderedDecode[DType.int64], SumScan],
            DType.uint64,
        ],
    ]
):
    return Codec(
        Encoder(
            Compose((DiffScan(), OrderedEncode())),
            ReferencePack(128, BitPack(32)),
        ),
        Decoder[
            ReferenceUnpack[BitUnpack],
            Compose[OrderedDecode[DType.int64], SumScan],
            DType.uint64,
        ](
            ReferenceUnpack(128, BitUnpack(32)),
            Compose((OrderedDecode[DType.int64](), SumScan())),
        ),
    )


def _cascade_double_delta_t64() -> (
    Codec[
        Encoder[
            Compose[DiffScan, DiffScan, OrderedEncode], ReferencePack[BitPack]
        ],
        Decoder[
            ReferenceUnpack[BitUnpack],
            Compose[OrderedDecode[DType.int64], SumScan, SumScan],
            DType.uint64,
        ],
    ]
):
    return Codec(
        Encoder(
            Compose((DiffScan(), DiffScan(), OrderedEncode())),
            ReferencePack(128, BitPack(32)),
        ),
        Decoder[
            ReferenceUnpack[BitUnpack],
            Compose[OrderedDecode[DType.int64], SumScan, SumScan],
            DType.uint64,
        ](
            ReferenceUnpack(128, BitUnpack(32)),
            Compose((OrderedDecode[DType.int64](), SumScan(), SumScan())),
        ),
    )


def _cascade_zigzag() -> (
    Codec[
        Encoder[Compose[ZigzagEncode], BitPack],
        Decoder[BitUnpack, Compose[ZigzagDecode], DType.uint64],
    ]
):
    return Codec(
        Encoder(Compose((ZigzagEncode(),)), BitPack()),
        Decoder[BitUnpack, Compose[ZigzagDecode], DType.uint64](
            BitUnpack(), Compose((ZigzagDecode(),))
        ),
    )


def _cascade_low_cardinality_runs() -> (
    Codec[
        DictionaryPack[RunsPack[BitPack, Encoder[Compose[DiffScan], BitPack]]],
        DictionaryUnpack[
            RunsUnpack[
                BitUnpack, Decoder[BitUnpack, Compose[SumScan], DType.uint64]
            ]
        ],
    ]
):
    return Codec(
        DictionaryPack(RunsPack(BitPack(), _ends_pack())),
        DictionaryUnpack(RunsUnpack(BitUnpack(), _ends_unpack())),
    )


def _cascade_sparse() -> (
    Codec[
        RunsPack[
            Encoder[Compose[OrderedEncode], ReferencePack[BitPack]],
            Encoder[Compose[DiffScan], BitPack],
        ],
        RunsUnpack[
            Decoder[
                ReferenceUnpack[BitUnpack],
                Compose[OrderedDecode[DType.int64]],
                DType.uint64,
            ],
            Decoder[BitUnpack, Compose[SumScan], DType.uint64],
        ],
    ]
):
    return Codec(
        RunsPack(
            Encoder(
                Compose((OrderedEncode(),)), ReferencePack(BLOCK, BitPack())
            ),
            _ends_pack(),
        ),
        RunsUnpack(
            Decoder[
                ReferenceUnpack[BitUnpack],
                Compose[OrderedDecode[DType.int64]],
                DType.uint64,
            ](
                ReferenceUnpack(BLOCK, BitUnpack()),
                Compose((OrderedDecode[DType.int64](),)),
            ),
            _ends_unpack(),
        ),
    )


def _pfor() -> (
    Codec[
        Encoder[Compose[OrderedEncode], ReferencePack[PforPack]],
        Decoder[
            ReferenceUnpack[PforUnpack],
            Compose[OrderedDecode[DType.int64]],
            DType.uint64,
        ],
    ]
):
    """Patched frame of reference."""
    return Codec(
        Encoder(
            Compose((OrderedEncode(),)),
            ReferencePack(BLOCK, PforPack(128, BitPack(128))),
        ),
        Decoder[
            ReferenceUnpack[PforUnpack],
            Compose[OrderedDecode[DType.int64]],
            DType.uint64,
        ](
            ReferenceUnpack(BLOCK, PforUnpack(128, BitUnpack(128))),
            Compose((OrderedDecode[DType.int64](),)),
        ),
    )


def _frequency() -> Codec[FrequencyPack, FrequencyUnpack]:
    return Codec(
        FrequencyPack(BLOCK, ConstantPack()),
        FrequencyUnpack(BLOCK, ConstantUnpack()),
    )


def _alp() -> Codec[AlpPack, AlpUnpack[DType.float64]]:
    """Floats as the integers of their decimal digits, framed and packed."""
    return Codec(
        AlpPack(
            BLOCK,
            Encoder(
                Compose((OrderedEncode(),)), ReferencePack(BLOCK, BitPack())
            ),
        ),
        AlpUnpack[DType.float64](
            BLOCK,
            Decoder[
                ReferenceUnpack[BitUnpack],
                Compose[OrderedDecode[DType.int64]],
                DType.uint64,
            ](
                ReferenceUnpack(BLOCK, BitUnpack()),
                Compose((OrderedDecode[DType.int64](),)),
            ),
        ),
    )


def _alp_rd() -> (
    Codec[
        AlpRdPack[DictionaryPack[BitPack], BitPack],
        AlpRdUnpack[DictionaryUnpack[BitUnpack], BitUnpack],
    ]
):
    """Floats' bits cut in two, the left parts through a dictionary and the
    right bit-packed."""
    return Codec(
        AlpRdPack(BLOCK, DictionaryPack(BitPack()), BitPack()),
        AlpRdUnpack(BLOCK, DictionaryUnpack(BitUnpack()), BitUnpack()),
    )


def _bench_choice[
    T: DType,
    shape: StaticString,
    decode: Bool,
    D: UnpackKernel,
    *Ps: PackKernel,
](mut b: Benchmark, var codec: Codec[ChoicePack[*Ps], D]) raises:
    """`_bench`, recording too the most a segment packs past the bytes its
    sample predicted."""
    var values = _data[T, shape]()
    var packed = codec.encode(BufferView(Span(values)))
    var most = 0.0
    for s in range(len(packed.chosen)):
        var nbytes = codec.encoder.segment_nbytes[T](packed, s)
        most = max(most, Float64(nbytes) / Float64(packed.predicted[s]))
    b.extra_info("overshoot", String(round(most, 3)))
    _bench[T, shape, decode](b, codec^)


def _choice[T: DType, U: DType]() -> Codec[Auto.Int2, Auto.IntUnpack2[T, U]]:
    """Integers chosen for three levels deep."""
    return Codec(Auto.int2(), Auto.int_unpack2[T, U]())


def _float_choice() -> (
    Codec[
        Auto.Float2,
        Auto.FloatUnpack2[DType.int64, DType.uint64],
    ]
):
    """Floats chosen for three levels deep."""
    return Codec(
        Auto.float2(),
        Auto.float_unpack2[DType.int64, DType.uint64](),
    )


# --- each packer alone ---------------------------------------------------------


def bench_plain_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", False](b, _plain())


def bench_plain_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", True](b, _plain())


def bench_bss_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", False](b, _bss())


def bench_bss_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", True](b, _bss())


def bench_bitpack_encode(mut b: Benchmark) raises:
    _bench[DType.uint32, "small", False](b, _bitpack())


def bench_bitpack_decode(mut b: Benchmark) raises:
    _bench[DType.uint32, "small", True](b, _bitpack())


def bench_reference_encode(mut b: Benchmark) raises:
    _bench[DType.uint64, "cardinality", False](b, _reference())


def bench_reference_decode(mut b: Benchmark) raises:
    _bench[DType.uint64, "cardinality", True](b, _reference())


def bench_runs_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "runs", False](b, _runs())


def bench_runs_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "runs", True](b, _runs())


def bench_delta_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", False](b, _delta())


def bench_delta_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", True](b, _delta())


def bench_dictionary_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", False](b, _dictionary())


def bench_dictionary_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", True](b, _dictionary())


def bench_zigzag_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", False](b, _zigzag())


def bench_zigzag_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", True](b, _zigzag())


def bench_ordered_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", False](b, _ordered())


def bench_ordered_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", True](b, _ordered())


def bench_lengths_encode(mut b: Benchmark) raises:
    _bench_bytes[False](
        b, LengthsPack(PlainPack()), LengthsUnpack(PlainUnpack())
    )


def bench_lengths_decode(mut b: Benchmark) raises:
    _bench_bytes[True](
        b, LengthsPack(PlainPack()), LengthsUnpack(PlainUnpack())
    )


def bench_prefixes_encode(mut b: Benchmark) raises:
    _bench_bytes[False](
        b,
        PrefixesPack(PlainPack(), PlainPack()),
        PrefixesUnpack(PlainUnpack(), PlainUnpack()),
    )


def bench_prefixes_decode(mut b: Benchmark) raises:
    _bench_bytes[True](
        b,
        PrefixesPack(PlainPack(), PlainPack()),
        PrefixesUnpack(PlainUnpack(), PlainUnpack()),
    )


def bench_dictionary_bytes_encode(mut b: Benchmark) raises:
    _bench_bytes[False](
        b,
        DictionaryBytesPack(PlainPack()),
        DictionaryBytesUnpack(PlainUnpack()),
    )


def bench_dictionary_bytes_decode(mut b: Benchmark) raises:
    _bench_bytes[True](
        b,
        DictionaryBytesPack(PlainPack()),
        DictionaryBytesUnpack(PlainUnpack()),
    )


def bench_fsst_encode(mut b: Benchmark) raises:
    _bench_bytes[False](
        b,
        FsstPack(PlainPack()),
        FsstUnpack(PlainUnpack()),
    )


def bench_fsst_decode(mut b: Benchmark) raises:
    _bench_bytes[True](
        b,
        FsstPack(PlainPack()),
        FsstUnpack(PlainUnpack()),
    )


def bench_constant_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "constant", False](
        b, Codec(ConstantPack(), ConstantUnpack())
    )


def bench_constant_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "constant", True](
        b, Codec(ConstantPack(), ConstantUnpack())
    )


def bench_frequency_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "sparse", False](b, _frequency())


def bench_frequency_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "sparse", True](b, _frequency())


def bench_pfor_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "outliers", False](b, _pfor())


def bench_pfor_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "outliers", True](b, _pfor())


def bench_alp_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", False](b, _alp())


def bench_alp_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", True](b, _alp())


def bench_alp_rd_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "real", False](b, _alp_rd())


def bench_alp_rd_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "real", True](b, _alp_rd())


# --- packers chosen per segment ----------------------------------------------


def bench_choice_timestamps_encode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "timestamps", False](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_timestamps_decode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "timestamps", True](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_runs_encode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "runs", False](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_runs_decode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "runs", True](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_cardinality_encode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "cardinality", False](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_cardinality_decode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "cardinality", True](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_constant_encode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "constant", False](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_constant_decode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "constant", True](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_interval_encode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "interval", False](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_interval_decode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "interval", True](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_signed_encode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "signed", False](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_signed_decode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "signed", True](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_sparse_encode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "sparse", False](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_sparse_decode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "sparse", True](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_outliers_encode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "outliers", False](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_outliers_decode(mut b: Benchmark) raises:
    _bench_choice[DType.int64, "outliers", True](
        b, _choice[DType.int64, DType.uint64]()
    )


def bench_choice_small_encode(mut b: Benchmark) raises:
    _bench_choice[DType.uint32, "small", False](
        b, _choice[DType.uint32, DType.uint32]()
    )


def bench_choice_small_decode(mut b: Benchmark) raises:
    _bench_choice[DType.uint32, "small", True](
        b, _choice[DType.uint32, DType.uint32]()
    )


def bench_choice_gauge_encode(mut b: Benchmark) raises:
    _bench_choice[DType.float64, "gauge", False](b, _float_choice())


def bench_choice_gauge_decode(mut b: Benchmark) raises:
    _bench_choice[DType.float64, "gauge", True](b, _float_choice())


def bench_choice_floats_encode(mut b: Benchmark) raises:
    _bench_choice[DType.float64, "floats", False](b, _float_choice())


def bench_choice_floats_decode(mut b: Benchmark) raises:
    _bench_choice[DType.float64, "floats", True](b, _float_choice())


def bench_choice_real_encode(mut b: Benchmark) raises:
    _bench_choice[DType.float64, "real", False](b, _float_choice())


def bench_choice_real_decode(mut b: Benchmark) raises:
    _bench_choice[DType.float64, "real", True](b, _float_choice())


# --- cascades, after ClickHouse's codec chains ---------------------------------


def bench_cascade_float_planes_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "gauge", False](b, _float_planes())


def bench_cascade_float_planes_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "gauge", True](b, _float_planes())


def bench_cascade_delta_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", False](
        b, Codec(_delta_zigzag_pack(), _delta_zigzag_unpack[DType.uint64]())
    )


def bench_cascade_delta_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", True](
        b, Codec(_delta_zigzag_pack(), _delta_zigzag_unpack[DType.uint64]())
    )


def bench_cascade_double_delta_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "interval", False](b, _cascade_double_delta())


def bench_cascade_double_delta_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "interval", True](b, _cascade_double_delta())


def bench_cascade_t64_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", False](b, _cascade_t64(BLOCK, BLOCK))


def bench_cascade_t64_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", True](b, _cascade_t64(BLOCK, BLOCK))


def bench_cascade_t64_blocks_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", False](b, _cascade_t64(64, 64))


def bench_cascade_t64_blocks_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", True](b, _cascade_t64(64, 64))


def bench_cascade_delta_t64_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", False](b, _cascade_delta_t64())


def bench_cascade_delta_t64_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", True](b, _cascade_delta_t64())


def bench_cascade_double_delta_t64_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "interval", False](b, _cascade_double_delta_t64())


def bench_cascade_double_delta_t64_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "interval", True](b, _cascade_double_delta_t64())


def bench_cascade_zigzag_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "signed", False](b, _cascade_zigzag())


def bench_cascade_zigzag_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "signed", True](b, _cascade_zigzag())


def bench_cascade_low_cardinality_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", False](
        b, Codec(DictionaryPack(BitPack()), DictionaryUnpack(BitUnpack()))
    )


def bench_cascade_low_cardinality_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", True](
        b, Codec(DictionaryPack(BitPack()), DictionaryUnpack(BitUnpack()))
    )


def bench_cascade_low_cardinality_runs_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "runs", False](b, _cascade_low_cardinality_runs())


def bench_cascade_low_cardinality_runs_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "runs", True](b, _cascade_low_cardinality_runs())


def bench_cascade_sparse_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "sparse", False](b, _cascade_sparse())


def bench_cascade_sparse_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "sparse", True](b, _cascade_sparse())


def bench_cascade_lengths_encode(mut b: Benchmark) raises:
    _bench_bytes[False](
        b,
        LengthsPack(_delta_zigzag_pack()),
        LengthsUnpack(_delta_zigzag_unpack[DType.uint32]()),
    )


def bench_cascade_lengths_decode(mut b: Benchmark) raises:
    _bench_bytes[True](
        b,
        LengthsPack(_delta_zigzag_pack()),
        LengthsUnpack(_delta_zigzag_unpack[DType.uint32]()),
    )


def bench_cascade_prefixes_encode(mut b: Benchmark) raises:
    _bench_bytes[False](
        b,
        PrefixesPack(_delta_zigzag_pack(), PlainPack()),
        PrefixesUnpack(_delta_zigzag_unpack[DType.uint32](), PlainUnpack()),
    )


def bench_cascade_prefixes_decode(mut b: Benchmark) raises:
    _bench_bytes[True](
        b,
        PrefixesPack(_delta_zigzag_pack(), PlainPack()),
        PrefixesUnpack(_delta_zigzag_unpack[DType.uint32](), PlainUnpack()),
    )


def bench_cascade_fsst_encode(mut b: Benchmark) raises:
    _bench_bytes[False](
        b,
        FsstPack(_delta_zigzag_pack()),
        FsstUnpack(_delta_zigzag_unpack[DType.uint32]()),
    )


def bench_cascade_fsst_decode(mut b: Benchmark) raises:
    _bench_bytes[True](
        b,
        FsstPack(_delta_zigzag_pack()),
        FsstUnpack(_delta_zigzag_unpack[DType.uint32]()),
    )


def bench_cascade_strings_dictionary_encode(mut b: Benchmark) raises:
    _bench_bytes[False](
        b,
        DictionaryBytesPack(RunsPack(BitPack(), _ends_pack())),
        DictionaryBytesUnpack(RunsUnpack(BitUnpack(), _ends_unpack())),
    )


def bench_cascade_strings_dictionary_decode(mut b: Benchmark) raises:
    _bench_bytes[True](
        b,
        DictionaryBytesPack(RunsPack(BitPack(), _ends_pack())),
        DictionaryBytesUnpack(RunsUnpack(BitUnpack(), _ends_unpack())),
    )
