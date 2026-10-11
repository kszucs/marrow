# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Column codecs, chained: values to fewer bytes, a register at a time.

| Module | What it holds |
|---|---|
| `core` | `Codec`, `Cascade`, `Encoded`: the trait, the chain -- itself a codec -- and the stored form; and what codecs share |
| `lanes` | the codecs a cascade passes values through |
| `stores` | the codecs that end a cascade and hold the bytes, `Compressed[C]` -- a `marrow.compress` compressor -- among them |
| `strings` | the codecs that take byte strings first |

Every codec of a `Cascade` but the last turns each value into one value;
the last stores them. What else a codec makes -- widths, the smallest
values, a dictionary -- is a slot, stored by a cascade of its own. A column
is stored as an `Encoded` tree, which a container frames as it likes.

    var cascade = Cascade[Delta, Zigzag, BitPack[]]()
    var encoded = cascade.encode(values)
    cascade.decode(encoded, out)

A codec configured at run time is given as a value, its slots as values
or type parameters:

    Cascade((Delta(first), Ordered(), FrameOfReference(128), BitPack(32)))
    Cascade[Alp[], Ordered, FrameOfReference[], BitPack[]]()

the first Parquet's DELTA_BINARY_PACKED after its first value, the second
DuckDB's ALP.
"""

from .core import (
    BLOCK,
    Cascade,
    Codec,
    Encoded,
)
from .lanes import (
    Alp,
    ByteSwap,
    Cast,
    Delta,
    Dictionary,
    FrameOfReference,
    Gcd,
    Ordered,
    Patch,
    Xor,
    Zigzag,
)
from .stores import (
    BitPack,
    Buckets,
    ByteStreamSplit,
    Compressed,
    Frequency,
    Gorilla,
    Plain,
    Runs,
)
from .strings import Lengths, Prefixes
