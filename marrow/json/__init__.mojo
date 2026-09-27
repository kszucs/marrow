# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Newline-delimited JSON — `read_json`, `open_json`, `write_json`.

Tokenized, escaped and float-formatted by EmberJson. As a plan source it is
`marrow.expr`'s `scan_json`, an `ExternalScan` over `JsonBatchReader`.

Explicit re-exports, never `import *`.
"""

from .options import ParseOptions, ReadOptions, UnexpectedFieldBehavior
from .reader import JsonReader, open_json, read_json
from .scan import JsonBatchReader
from .writer import JsonWriter, render_json, write_json
