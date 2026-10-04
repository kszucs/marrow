# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Newline-delimited JSON — `read_json`, `open_json`, `write_json`.

Tokenized, escaped and float-formatted by EmberJson.

Explicit re-exports, never `import *`.
"""

from .reader import (
    JsonReader,
    ParseOptions,
    ReadOptions,
    UnexpectedFieldBehavior,
    open_json,
    read_json,
)
from .writer import JsonWriter, write_json
