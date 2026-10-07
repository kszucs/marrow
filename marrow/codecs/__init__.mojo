# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Column codecs that chain.

A codec maps a block of values to a block of its output type and back, and
a chain runs each codec over what the one before handed on. Any codec may
follow any other whose output it takes, and a chain may end anywhere: its
last stream is stored as its values' bytes. A `Cascade` fixes the chain at
compile time and fuses it -- an `Elementwise` codec maps each value as the
next codec reads it, in one loop:

    comptime C = Cascade[Delta, Zigzag, BitPack]
    var data = C.encode[DType.int64](values)
    var back = C.decode[DType.int64](data)

A `DynCascade` is the same chain chosen at run time -- built from codecs, or
converted from a `Cascade` -- running one codec at a time over a block. It
writes the same bytes:

    var chain = DynCascade(Delta(), Zigzag(), BitPack())
    var same = chain.encode[DType.int64](values)

The stream does not name its codecs: it is read with the chain that wrote
it. Both stream: a column goes through in blocks of `BLOCK` values, each
encoded on its own, so a writer and a reader hold one block whatever the
column's length. `encode` and `decode` are the whole column through them:

    var writer = CascadeWriter[DType.int64, Delta, Zigzag, BitPack]()
    var out = List[UInt8]()
    for chunk in chunks:
        writer.write(chunk, out)    # appends each block as it fills
    writer.finish(out)

    var reader = CascadeReader[DType.int64, Delta, Zigzag, BitPack](out)
    var block = List[Int64]()
    while reader.read(block):       # a block at a time
        consume(block)
        block.clear()

`DynCascadeWriter` and `DynCascadeReader` do the same for a `DynCascade`;
any reader reads any writer's stream of the same chain.

| Elementwise | Takes | Each value becomes |
|---|---|---|
| `Delta` | integers | its difference from the one before |
| `Zigzag` | signed integers | `0, -1, 1, -2, ...` as the unsigned `0, 1, 2, 3, ...` |
| `Xor` | numbers | its bits XOR the previous value's |
| `OrderPreserving` | numbers | an unsigned integer that sorts as the value does |

| Writes bytes | Takes | Writes |
|---|---|---|
| `BitPack` | numbers | each value's distance from the smallest, at the fewest bits |
| `Varint` | integers | each value as a ULEB128 |
| `ByteStreamSplit` | numbers | the values' bytes in per-byte planes |
| `Constant` | numbers | the one value |
| `Rle` | numbers | runs of one value |
| `Dictionary` | numbers | the distinct values, and a code per row |
| `Frequency` | numbers | the most frequent value, and the exceptions |
| `Hybrid` | unsigned integers | Parquet's RLE / bit-packed hybrid |
| `DeltaBinaryPacked` | signed integers | Parquet's DELTA_BINARY_PACKED |
| `Plain` | numbers | each value's little-endian bytes |

Double-delta is two `Delta`s: `Cascade[Delta, Delta, Zigzag, BitPack]`, and
a codec may follow one that wrote bytes: `Cascade[BitPack, Rle]` run-length
encodes the packed bytes.

Byte strings take a `BinaryCodec` over the two buffers of an Arrow binary
column -- offsets and data: `PlainBinary`, `DeltaLengthByteArray` and
`DeltaByteArray`.

`BitPack`, `ByteStreamSplit`, `Hybrid` and `DeltaBinaryPacked` also carry
their kernels as static methods, for formats with their own framing; `Bits`
holds the register-only ones a GPU kernel can call too. `LittleEndian`,
`BigEndian` and `Leb128` are the byte order every format reads and writes
with.

A codec reads the values it encodes from a `Source` -- rewinding it for
another pass -- and hands the values it decodes to an `Emitter`. `Values`
reads a block as it is and `Append` appends to a list; `Mapped` and
`Unmapped` take an elementwise codec through both, so a `Cascade` never
stores what one hands on.

Codecs see present values only; nulls stay with the caller. A new codec
implements `Codec`, or `Elementwise` when it maps one value to one, and joins
`DynCodec`'s variant in `dyn_cascade.mojo`.

Explicit re-exports, never `import *`.
"""

from .binary import BinaryCodec
from .bitpack import BitPack
from .bits import Bits
from .byteorder import BigEndian, Leb128, LittleEndian
from .byte_stream_split import ByteStreamSplit
from .cascade import Cascade, CascadeReader, CascadeWriter
from .constant import Constant
from .core import (
    Append,
    BLOCK,
    Codec,
    Decoder,
    Elementwise,
    Emitter,
    Mapped,
    Source,
    Unmapped,
    Values,
)
from .delta import Delta
from .delta_binary_packed import DeltaBinaryPacked
from .delta_byte_array import DeltaByteArray
from .delta_length_byte_array import DeltaLengthByteArray
from .order_preserving import OrderPreserving
from .dyn_cascade import (
    DynCascade,
    DynCascadeReader,
    DynCascadeWriter,
    DynCodec,
)
from .dictionary import Dictionary
from .frequency import Frequency
from .hybrid import Hybrid
from .plain import Plain
from .plain_binary import PlainBinary
from .rle import Rle
from .varint import Varint
from .xor import Xor
from .zigzag import Zigzag
