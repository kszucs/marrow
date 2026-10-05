# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Fuzz target: marrow's native Snappy decoder, checked against libsnappy.

The input is a raw Snappy block. Two oracles beyond "does not crash":

- **libsnappy agrees**: both decoders accept or both reject, and when both
  accept they produce the same bytes. The native decoder is a port, so any
  disagreement is a defect in one of them.
- **round trip**: whatever the native decoder produces, native compression
  followed by native decompression gives back.
"""

from std.os import abort

from marrow.utils.compression import CompressionLibs
from marrow.utils.snappy import Snappy

# A declared length past this is left to the native decoder alone: allocating
# it for libsnappy would make one input cost a gigabyte.
comptime MAX_DIFFERENTIAL = 1 << 22


def fuzz_one(data: Span[UInt8, _]) raises:
    var native = List[UInt8]()
    var native_ok = True
    try:
        Snappy.decompress(data, native)
    except:
        native_ok = False

    var declared = -1
    try:
        declared = Snappy.uncompressed_length(data)
    except:
        pass

    if declared >= 0 and declared <= MAX_DIFFERENTIAL:
        var libs = CompressionLibs()
        var reference = List[UInt8](length=declared, fill=0)
        var reference_ok = True
        try:
            libs.snappy_decompress(data, reference.unsafe_ptr(), declared)
        except:
            reference_ok = False
        if native_ok != reference_ok:
            abort(
                "snappy: native "
                + ("accepted" if native_ok else "rejected")
                + " a block libsnappy "
                + ("accepted" if reference_ok else "rejected")
            )
        if native_ok and native != reference:
            abort("snappy: native and libsnappy decoded different bytes")

    if native_ok:
        var packed = List[UInt8]()
        Snappy.compress(Span(native), packed)
        var unpacked = List[UInt8]()
        Snappy.decompress(Span(packed), unpacked)
        if unpacked != native:
            abort("snappy: compress then decompress changed the bytes")
