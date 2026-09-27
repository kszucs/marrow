# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""NDJSON as a plan source — `scan_json`, the JSON twin of `marrow.expr.scan`.

`marrow.expr` does not import this module (see `marrow/json/__init__.mojo`),
so the scan is an `ExternalScan` built from `JsonScan`: the plan carries
one function pointer into this module and `marrow.expr` never names it.
"""

from ..dtypes import StringType
from ..expr import BatchReader, DynRelation, ExternalScan, ScanPath
from ..expr import StringParam
from ..io import DynSource
from ..schema import Schema
from ..tabular import RecordBatch

from .options import ParseOptions, ReadOptions, UnexpectedFieldBehavior
from .reader import JsonReader, open_json


struct JsonScan(BatchReader):
    """The JSON scan: what an `ExternalScan` over NDJSON reads with, a
    `JsonReader` whose schema is the scan's, one batch per block.

    A `BatchReader` rather than a `DynRelation` member, so `marrow.expr` never
    imports it: a variant member of the plan IR is compiled into every program
    that plans a query, and this one would bring EmberJson with it.

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


def scan_json(var path: String, var schema: Schema) raises -> DynRelation:
    """A newline-delimited JSON file, as a plan.

    The schema is required, as it is for `scan`: a plan is a description, and
    building it must not open the file.
    """
    return ExternalScan.of[JsonScan](ScanPath(path^), schema^)


def scan_json(
    var path: StringParam[StringType], var schema: Schema
) raises -> DynRelation:
    """A newline-delimited JSON file named at run time, as a plan; `QueryCli`
    offers the parameter as an option."""
    return ExternalScan.of[JsonScan](ScanPath(path^), schema^)
