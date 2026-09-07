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


def write_ipc_file(path, batches=None, schema=None):
    raw = [unwrap(b) for b in batches] if batches is not None else None
    return _ma.write_ipc_file(path, raw, unwrap(schema))


def write_ipc_stream(path, batches=None, schema=None):
    raw = [unwrap(b) for b in batches] if batches is not None else None
    return _ma.write_ipc_stream(path, raw, unwrap(schema))


def read_ipc_file(path):
    return [RecordBatch.wrap(b) for b in _ma.read_ipc_file(path)]


def read_ipc_stream(path):
    return [RecordBatch.wrap(b) for b in _ma.read_ipc_stream(path)]


def read_ipc_file_schema(path):
    return RecordBatch.wrap(_ma.read_ipc_file_schema(path))


def read_ipc_stream_schema(path):
    return RecordBatch.wrap(_ma.read_ipc_stream_schema(path))
