# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Arrow IPC — the file and stream formats, by path.

Path-based only: the Mojo reader and writer take a filename, so there is no
buffer or file-object source to expose yet.
"""

from . import libmarrow as _ma
from ._wrapper import unwrap
from .tabular import RecordBatch

__all__ = [
    "read_ipc_file",
    "read_ipc_file_schema",
    "read_ipc_stream",
    "read_ipc_stream_schema",
    "write_ipc_file",
    "write_ipc_stream",
]


def write_ipc_file(
    path, batches=None, schema=None, compression=None, native_codecs=True
):
    """Write an IPC file; `compression` is `None`, `"lz4"` or `"zstd"`, as
    for pyarrow's `IpcWriteOptions`. The codec runs in Mojo, or with
    `native_codecs=False` through liblz4 and libzstd."""
    raw = [unwrap(b) for b in batches] if batches is not None else None
    return _ma.write_ipc_file(path, raw, unwrap(schema), compression, native_codecs)


def write_ipc_stream(
    path, batches=None, schema=None, compression=None, native_codecs=True
):
    """Write an IPC stream; `compression` and `native_codecs` as for
    `write_ipc_file`."""
    raw = [unwrap(b) for b in batches] if batches is not None else None
    return _ma.write_ipc_stream(path, raw, unwrap(schema), compression, native_codecs)


def read_ipc_file(path, native_codecs=True):
    """Read an IPC file; compressed bodies decode in Mojo, or with
    `native_codecs=False` through liblz4 and libzstd."""
    return [RecordBatch.wrap(b) for b in _ma.read_ipc_file(path, native_codecs)]


def read_ipc_stream(path, native_codecs=True):
    """Read an IPC stream; `native_codecs` as for `read_ipc_file`."""
    return [RecordBatch.wrap(b) for b in _ma.read_ipc_stream(path, native_codecs)]


def read_ipc_file_schema(path):
    return RecordBatch.wrap(_ma.read_ipc_file_schema(path))


def read_ipc_stream_schema(path):
    return RecordBatch.wrap(_ma.read_ipc_stream_schema(path))
