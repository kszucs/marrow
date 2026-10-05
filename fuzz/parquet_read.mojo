# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Fuzz target: reading a whole Parquet file -- pages, codecs and encodings.

Decoded on the calling thread, so one input is one deterministic execution.
Raising is the correct answer to a malformed file; an abort or a crash is a
bug.
"""

from marrow.execution import ExecContext
from marrow.io import BufferSource
from marrow.parquet import ParquetFile

from common import consume


def fuzz_one(data: Span[UInt8, _]) raises:
    var file = ParquetFile(BufferSource(data))
    consume(file.read(ctx=ExecContext.serial()))
