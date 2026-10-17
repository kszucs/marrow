# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""File formats: the `BatchReader` / `BatchWriter` protocol every format's
reader and writer implements, `ChunkedReader` for reading a file in parallel,
and `Format`, which names a format's readers.

Explicit re-exports, never `import *`.
"""

from .core import BatchReader, BatchWriter, ChunkedReader, Format
