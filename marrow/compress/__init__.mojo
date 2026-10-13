# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Byte compressors -- what Parquet's pages, Arrow IPC's buffers and Avro's
blocks are compressed with -- each a `Compressor`.

| Module | What it holds |
|---|---|
| `methods` | `Method`, a compression method by name, run by the codec a caller allows |
| `core` | `Compressor`: bytes to fewer bytes and back; `UnsizedDecompressor`; `compress_with`, `decompress_with`, `decompress_unsized_with`; `Codecs`, the system libraries |
| `zstd` | `Zstd`, and `LibZstd` through libzstd |
| `lz4` | `Lz4` and `Lz4Frame`, and `LibLz4` and `LibLz4Frame` through liblz4 |
| `snappy` | `Snappy`, and `LibSnappy` through libsnappy |
| `brotli` | `Brotli`, and `LibBrotli` through libbrotli; `brotli_dictionary`, the format's constants |
| `zlib` | `LibGzip` and `LibDeflate` through zlib: deflate has no codec in Mojo |
| `hadoop` | `Hadoop`, the framing of Parquet's deprecated LZ4 |
| `huffman`, `fse` | the entropy coders Zstandard is built from |
| `lz77` | the copies the LZ decoders share, and the match length their compressors do |

A reader or writer chooses marrow's own codec or the system library's by
type -- `Zstd` or `LibZstd` -- or at run time with `compress_with` and
`decompress_with`, and the tests hold the two to the same bytes. A format
names a `Method` and lets it pick.
"""

from .brotli import Brotli, LibBrotli
from .core import (
    Codecs,
    Compressor,
    UnsizedDecompressor,
    compress_with,
    decompress_unsized_with,
    decompress_with,
)
from .hadoop import Hadoop
from .lz4 import LibLz4, LibLz4Frame, Lz4, Lz4Frame
from .methods import Method
from .snappy import LibSnappy, Snappy
from .zlib import LibDeflate, LibGzip
from .zstd import LibZstd, Zstd
