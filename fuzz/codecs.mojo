# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Fuzz target: Parquet page decompression through `Compression`.

Covers every codec -- ZSTD and LZ4 (raw and the legacy Hadoop frame) both in
Mojo and through their libraries, GZIP, Brotli, Snappy -- and the uncompressed
copy. What is under
test is marrow's side of each call: the expected size it promises the library
and the buffer it hands over, which is what a page header controls.

Input layout:

    byte 0       low 7 bits: codec 0, 1, 2, 4, 5, 6, 7 by `bits % 7`;
                 high bit set: LZ4 and ZSTD through their libraries
    bytes 1..3   the page's declared uncompressed size, little endian,
                 masked to under 1 MiB
    bytes 4..    the compressed page
"""

from marrow.parquet import Compression
from marrow.utils.compression import CompressionLibs


def fuzz_one(data: Span[UInt8, _]) raises:
    if len(data) < 4:
        return
    var codecs: List[Int] = [0, 1, 2, 4, 5, 6, 7]
    var code = codecs[Int(data[0] & 0x7F) % 7]
    var native = (data[0] & 0x80) == 0
    var out_size = (
        Int(data[1]) | (Int(data[2]) << 8) | (Int(data[3]) << 16)
    ) & 0xFFFFF
    var libs = CompressionLibs(native=native)
    _ = Compression(code).decompress_owned(libs, data[4:], out_size)
