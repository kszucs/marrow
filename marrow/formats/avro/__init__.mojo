# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Apache Avro object container files — `read_avro`, `write_avro`, and the
`AvroFile` / `AvroReader` / `AvroWriter` they are built on, and `AvroFormat`.

Explicit re-exports, never `import *`.
"""

from .binary import AvroBytes, AvroCursor
from .decoder import RecordDecoder
from .encoder import RecordEncoder
from .file import (
    AvroBlock,
    AvroChunks,
    AvroCodec,
    AvroFile,
    AvroFormat,
    AvroReader,
    AvroWriter,
    read_avro,
    write_avro,
)
from .mapping import from_arrow, to_arrow, writes_as
from .schema import AvroKind, AvroSchema
