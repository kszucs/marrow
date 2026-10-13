# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Arrow IPC files and streams, read and written a record batch at a time.

`read_ipc_file` / `write_ipc_file` and their stream twins move whole lists of
batches; `RecordBatchFileReader`, `RecordBatchStreamReader` and the two
writers move one batch at a time. Bodies compressed with LZ4_FRAME or ZSTD
(`BodyCompression`) are read, and written when a writer is given one.

The modules, bottom up: `format` (wire constants and small records),
`flatbuf` (the FlatBuffers codec), `metadata` (schemas, messages, footer),
`body` (record batch bodies), `compression`, then `reader` and `writer`.

Explicit re-exports, never `import *`.
"""

from .compression import BodyCompression
from .reader import (
    IpcFormat,
    RecordBatchFileReader,
    RecordBatchStreamReader,
    read_ipc_file,
    read_ipc_file_schema,
    read_ipc_stream,
    read_ipc_stream_schema,
)
from .writer import (
    RecordBatchFileWriter,
    RecordBatchStreamWriter,
    write_ipc_file,
    write_ipc_stream,
)
