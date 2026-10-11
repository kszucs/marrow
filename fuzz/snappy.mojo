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

from marrow.compress import LibSnappy, Snappy

# A declared length past this is left to the native decoder alone: allocating
# it for libsnappy would make one input cost a gigabyte.
comptime MAX_DIFFERENTIAL = 1 << 22


def _decoded[
    native: Bool
](data: Span[UInt8, _], declared: Int) -> Optional[List[UInt8]]:
    """`data` decoded into the `declared` bytes it promises, natively or by
    libsnappy, or nothing if the decoder refuses it."""
    var out = List[UInt8](length=declared, fill=0)
    try:
        comptime if native:
            Snappy.decompress(data, Span(out))
        else:
            LibSnappy.decompress(data, Span(out))
    except:
        return None
    return out^


def fuzz_one(data: Span[UInt8, _]) raises:
    var declared = -1
    try:
        declared = Snappy.uncompressed_length(data)
    except:
        pass
    if declared < 0 or declared > Snappy.max_decompressed_length(len(data)):
        return
    var native = _decoded[True](data, declared)

    if declared <= MAX_DIFFERENTIAL:
        var reference = _decoded[False](data, declared)
        if Bool(native) != Bool(reference):
            abort(
                "snappy: native "
                + ("accepted" if native else "rejected")
                + " a block libsnappy "
                + ("accepted" if reference else "rejected")
            )
        if native and native.value() != reference.value():
            abort("snappy: native and libsnappy decoded different bytes")

    if native:
        var packed = List[UInt8]()
        Snappy.compress(Span(native.value()), packed)
        var unpacked = _decoded[True](Span(packed), len(native.value()))
        if not unpacked or unpacked.value() != native.value():
            abort("snappy: compress then decompress changed the bytes")
