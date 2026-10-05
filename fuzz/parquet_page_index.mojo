# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Fuzz target: the Parquet page index (OffsetIndex and ColumnIndex).

Raising is the correct answer to a malformed file; an abort or a crash is a
bug.
"""

from marrow.io import BufferSource
from marrow.parquet import ParquetFile


def fuzz_one(data: Span[UInt8, _]) raises:
    var file = ParquetFile(BufferSource(data))
    var index = file.page_index()
    _ = len(index)
