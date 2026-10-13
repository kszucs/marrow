# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Parquet's page encodings refuse corrupt bytes with `CorruptError`, before
any value is read past the page: each case is a page written by hand."""

from std.testing import assert_raises, assert_true

from ....utils.byteorder import Leb128
from ..encodings import (
    DeltaBinaryPackedEncoding,
    DeltaLengthEncoding,
    Encoding,
)


def _bytes(*values: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for v in values:
        out.append(UInt8(v))
    return out^


def _dbp_header(block: Int, miniblocks: Int, total: Int) -> List[UInt8]:
    var out = List[UInt8]()
    Leb128.write(out, UInt64(block))
    Leb128.write(out, UInt64(miniblocks))
    Leb128.write(out, UInt64(total))
    Leb128.write(out, 0)
    return out^


def test_dbp_miniblocks_must_divide_the_block() raises:
    var data = _dbp_header(17, 2, 3)
    var out = List[Int32]()
    with assert_raises(contains="miniblocks in a block"):
        _ = DeltaBinaryPackedEncoding.decode[DType.int32](Span(data), 3, out)


def test_dlba_count_is_the_streams() raises:
    var lengths: List[Int32] = [1, 1, 1]
    var page = List[UInt8]()
    DeltaBinaryPackedEncoding.encode(Span(lengths), page)
    var bytes = _bytes(1, 2, 3)
    page.extend(Span(bytes))
    var offsets: List[Int32] = [0]
    var data = List[UInt8]()
    with assert_raises(contains="from a stream of 3"):
        _ = DeltaLengthEncoding.decode(Span(page), 2, offsets, data)


def test_rle_booleans_longer_than_the_page() raises:
    var page = _bytes(0xFF, 0, 0, 0, 1)
    with assert_raises(contains="RLE booleans"):
        _ = Encoding.RLE.decode_bool(Span(page), 3)
