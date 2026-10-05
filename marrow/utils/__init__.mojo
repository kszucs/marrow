# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Format-agnostic primitives shared across marrow.

Each submodule is a self-contained block that depends on nothing in marrow — no
arrays, no dtypes, no execution context — so it can be read, tested and lifted
out on its own:

| module | what |
|---|---|
| `argparse` | `ArgumentParser` — argv into named values, flags and `--help` |
| `byteorder` | `LittleEndian`, `BigEndian`, `Zigzag` — byte, bit and LEB128-varint reads/writes |
| `datetime` | `CivilDate`, `Epoch`, `floor_div` — proleptic-Gregorian arithmetic |
| `checksum` | `Crc32` / `Crc32c` — the ISO-3309 / zlib / gzip CRC and CRC-32C, over one `Crc` |
| `hex` | `hex_digit` — one hexadecimal digit's value |
| `hashing` | `Hasher` plus `RapidHash64`, `XxHash64`, `AHash64`, `Fmix64`, `TruncatedHash64`, `KeyHash`; `XxHash32` |
| `compression` | `CompressionLibs` — `dlopen`ed snappy / zlib / brotli, and libzstd / liblz4 where chosen over the Mojo codecs |
| `snappy` | `Snappy` — Snappy raw-block compression, in Mojo |
| `lz4` | `Lz4` — LZ4 block, frame and Hadoop-framed compression, in Mojo |
| `zstd` | `Zstd` — Zstandard, writing libzstd's level-1 frames, in Mojo |
| `dylib` | `Dylib`, `LibSpec` — declaring and opening an optional C library |
| `uri` | `Uri`, `StorageOptions` — a location and the service config it implies |
| `testing` | `TestSuite` / `BenchSuite` / `Benchmark` — the harness pytest drives |

The names are re-exported here, so `from ..utils import LittleEndian` is the
import everywhere; the submodule split is about where the code *lives*, not
about making callers spell out a path.

**`testing` is the one exception and is deliberately not re-exported.** Every
module in the tree imports `marrow.utils`, and none of them should pull
`std.benchmark` in behind it — test and bench files import
`..utils.testing` explicitly.

This replaced a single 312-line `marrow/utils.mojo` that was four unrelated
things, and a second `marrow/parquet/utils.mojo` that held the codec bindings —
two modules named `utils`, neither describing its contents. Device capability
(`GPU_ENABLED`, `has_accelerator_support`) went the other way, to
`marrow.execution`, where `ExecContext` already owns every question about
whether there is a device.
"""

from .argparse import ArgSpec, ArgumentParser, ParsedArgs, parse_bool
from .byteorder import BigEndian, LittleEndian, Zigzag
from .checksum import Crc32, Crc32c
from .compression import CompressionLibs
from .datetime import CivilDate, Epoch, floor_div
from .dylib import Dylib, LibSpec
from .uri import StorageOptions, Uri
from .hex import hex_digit
from .hashing import (
    AHash64,
    Hasher,
    Fmix64,
    KeyHash,
    RapidHash64,
    RapidSecret,
    TruncatedHash64,
    XxHash32,
    XxHash64,
)
from .lz4 import Lz4
from .snappy import Snappy
from .zstd import Zstd
