# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Byte-string codecs, over the two buffers an Arrow binary column is: the
offsets, value `i` at `data[offsets[i] : offsets[i + 1]]`, and the data."""

from std.reflection import reflect


trait BinaryCodec(Copyable, Defaultable, Deinitable):
    """Byte strings to bytes, and back. Byte strings are not a stream of one
    type, so this is not a `Codec`; one that stores numbers -- lengths,
    prefix lengths -- stores them with a `Codec` of its choosing, written in
    place."""

    @staticmethod
    def name() -> StaticString:
        """The codec's type name."""
        comptime full = reflect[Self].name()
        return full[byte = full.rfind(".") + 1 :]

    @staticmethod
    def encode(
        offsets: Span[Int32, _], data: Span[UInt8, _], mut out: List[UInt8]
    ) raises:
        """Append the `len(offsets) - 1` values."""
        ...

    @staticmethod
    def decode(
        src: Span[UInt8, _],
        pos: Int,
        count: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises -> Int:
        """Append the `count` values at `src[pos]` to `data`, and where each
        ends to `offsets` -- which must already hold where `data` ends.
        Return the position after them. Raises `CorruptError` on input
        `encode` could not have written."""
        ...
