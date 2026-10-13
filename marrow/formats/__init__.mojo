# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""File formats: the `BatchReader` / `BatchWriter` protocol every format's
reader and writer implements, and `Format`, which names a format's reader.

Explicit re-exports, never `import *`.
"""

from .core import BatchReader, BatchWriter, Format
