# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Fuzz target: the Parquet footer -- Thrift metadata and the schema mapping.

The input is a Parquet file; only its footer is read, which is what
`read_metadata` does. Raising is the correct answer to a malformed one; an
abort or a crash is a bug.
"""

from marrow.io import BufferSource
from marrow.parquet import ParquetFile


def fuzz_one(data: Span[UInt8, _]) raises:
    var file = ParquetFile(BufferSource(data))
    ref meta = file.metadata()
    _ = String(file.schema())
    _ = len(meta.row_groups)
