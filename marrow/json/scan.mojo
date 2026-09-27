# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The reader `marrow.expr`'s `scan_json` runs, through `ExternalScan`."""

from ..expr.logical import BatchReader
from ..io import DynSource
from ..schema import Schema
from ..tabular import RecordBatch

from .options import ParseOptions, ReadOptions, UnexpectedFieldBehavior
from .reader import JsonReader, open_json


struct JsonBatchReader(BatchReader):
    """Newline-delimited JSON for `JsonScan`: a `JsonReader` whose schema is
    the scan's, one batch per block.

    Keys outside the schema are skipped, not errors, so a scan narrowed by
    `ColumnPruning` reads only the columns the plan needs out of rows that
    carry more.
    """

    var _reader: JsonReader[DynSource]

    def __init__(out self, var reader: JsonReader[DynSource]):
        self._reader = reader^

    @staticmethod
    def open(path: String, schema: Schema) raises -> Self:
        return Self(
            open_json(
                path,
                ReadOptions(),
                ParseOptions(schema, UnexpectedFieldBehavior.IGNORE),
            )
        )

    def read_next_batch(mut self) raises -> Optional[RecordBatch]:
        return self._reader.read_next_batch()

    @staticmethod
    def format_name() -> String:
        return "json"
