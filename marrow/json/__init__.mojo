# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Newline-delimited JSON — `read_json`, `open_json`, `scan_json` and
`write_json`.

The one part of `marrow` that needs a third-party Mojo package, EmberJson, for
its tokenizer. Nothing else in `marrow` imports this module — `marrow.expr`
runs a JSON scan through `ExternalScan` — so a program built from source needs
EmberJson only if it reads JSON. A *precompiled* `marrow` needs it regardless:
it needs every package any of its modules imports.

Explicit re-exports, never `import *`.
"""

from .options import ParseOptions, ReadOptions, UnexpectedFieldBehavior
from .reader import JsonReader, open_json, read_json
from .scan import JsonScanReader, scan_json
from .writer import JsonWriter, render_json, write_json
