# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Arrow IPC — the file and stream formats, by path.

Whole-file functions (`read_ipc_file`, `write_ipc_stream`, ...) and the
pyarrow-shaped `new_file`, `new_stream`, `open_file` and `open_stream`, which
move one batch at a time and carry custom metadata:

    with marrow.ipc.new_file("out.arrow", schema, metadata={"k": "v"}) as w:
        w.write_batch(batch, custom_metadata={"batch": "first"})
    reader = marrow.ipc.open_file("out.arrow")
    batch, custom_metadata = reader.get_batch_with_custom_metadata(0)

Path-based only: the Mojo reader and writer take a filename, so there is no
buffer or file-object source to expose yet.
"""

import os
from collections import namedtuple

from . import libmarrow as _ma
from ._wrapper import _Wrapper, unwrap
from .tabular import RecordBatch, Table, table as _table
from .types import Schema

__all__ = [
    "IpcWriteOptions",
    "RecordBatchWithMetadata",
    "new_file",
    "new_stream",
    "open_file",
    "open_stream",
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
    """Write an IPC file; `schema` is anything exporting an Arrow schema, and
    `compression` is `None`, `"lz4"` or `"zstd"`, as for pyarrow's
    `IpcWriteOptions`. The codec runs in Mojo, or with `native_codecs=False`
    through liblz4 and libzstd."""
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


RecordBatchWithMetadata = namedtuple(
    "RecordBatchWithMetadata", ["batch", "custom_metadata"]
)


class IpcWriteOptions:
    """How a writer encodes its batches; `compression` is `None`, `"lz4"` or
    `"zstd"`."""

    def __init__(self, compression=None):
        self.compression = compression


def _path(sink):
    if not isinstance(sink, (str, os.PathLike)):
        raise TypeError(f"sink must be a path, got {type(sink).__name__}")
    return os.fspath(sink)


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _metadata_in(metadata):
    """A metadata mapping with `str` or `bytes` keys and values, as `str`."""
    return {_text(k): _text(v) for k, v in (metadata or {}).items()}


def _metadata_out(metadata):
    """The metadata mapping as `bytes`, as pyarrow returns it."""
    return {k.encode(): v.encode() for k, v in metadata.items()}


def _compression(options):
    return options.compression if options is not None else None


class _IpcWriter(_Wrapper):
    __slots__ = ()

    def write_batch(self, batch, custom_metadata=None):
        """Write one record batch, with `custom_metadata` on its message."""
        self._binding.write(unwrap(batch), _metadata_in(custom_metadata))

    def write_table(self, table, max_chunksize=None):
        """Write every batch of a table."""
        if max_chunksize is not None:
            raise NotImplementedError("max_chunksize is not supported")
        if not isinstance(table, Table):
            table = _table(table)
        for batch in table.to_batches():
            self.write_batch(batch)

    def write(self, table_or_batch):
        """Write a record batch or a table, marrow's or any Arrow producer's."""
        obj = table_or_batch
        # A foreign batch is an Arrow array producer with a schema; a bare
        # array exports `__arrow_c_array__` too, but has none.
        if isinstance(obj, RecordBatch) or (
            hasattr(obj, "__arrow_c_array__") and hasattr(obj, "schema")
        ):
            self.write_batch(obj)
        elif isinstance(obj, Table) or hasattr(obj, "__arrow_c_stream__"):
            self.write_table(obj)
        else:
            raise TypeError(f"expected a record batch or a table, got {type(obj)}")

    def close(self):
        """Finish the output; further writes are an error."""
        self._binding.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RecordBatchFileWriter(_IpcWriter):
    """Writes the IPC file format; see :func:`new_file`."""

    __slots__ = ()


class RecordBatchStreamWriter(_IpcWriter):
    """Writes the IPC stream format; see :func:`new_stream`."""

    __slots__ = ()


def new_file(sink, schema, *, options=None, metadata=None, native_codecs=True):
    """A writer for the IPC file format at the path `sink`; `metadata` is
    stored in the file's footer."""
    return RecordBatchFileWriter.wrap(
        _ma.ipc_new_file(
            _path(sink),
            unwrap(schema),
            _compression(options),
            native_codecs,
            _metadata_in(metadata),
        )
    )


def new_stream(sink, schema, *, options=None, native_codecs=True):
    """A writer for the IPC stream format at the path `sink`."""
    return RecordBatchStreamWriter.wrap(
        _ma.ipc_new_stream(
            _path(sink), unwrap(schema), _compression(options), native_codecs
        )
    )


class _IpcReader(_Wrapper):
    __slots__ = ()

    @property
    def schema(self):
        return Schema.wrap(self._binding.schema())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class RecordBatchFileReader(_IpcReader):
    """Random access to the batches of an IPC file; see :func:`open_file`."""

    __slots__ = ()

    @property
    def num_record_batches(self):
        return self._binding.num_record_batches()

    @property
    def metadata(self):
        """The footer's custom metadata as a `dict` of `bytes`, or `None`."""
        metadata = self._binding.metadata()
        return _metadata_out(metadata) if metadata else None

    def get_batch_with_custom_metadata(self, i):
        """Batch `i` and the custom metadata of its message."""
        batch, custom_metadata = self._binding.get_batch(i)
        return RecordBatchWithMetadata(
            RecordBatch.wrap(batch), _metadata_out(custom_metadata) or None
        )

    def get_batch(self, i):
        """Batch `i`."""
        return self.get_batch_with_custom_metadata(i).batch

    def read_all(self):
        """Every batch, as a :class:`marrow.Table`."""
        batches = [self.get_batch(i) for i in range(self.num_record_batches)]
        return Table.from_batches(batches, schema=self.schema)


class RecordBatchStreamReader(_IpcReader):
    """The batches of an IPC stream in order; see :func:`open_stream`."""

    __slots__ = ()

    def read_next_batch_with_custom_metadata(self):
        """The next batch and its custom metadata; raises `StopIteration` at
        the end."""
        pair = self._binding.read_next_batch()
        if pair is None:
            raise StopIteration
        batch, custom_metadata = pair
        return RecordBatchWithMetadata(
            RecordBatch.wrap(batch), _metadata_out(custom_metadata) or None
        )

    def read_next_batch(self):
        """The next batch; raises `StopIteration` at the end."""
        return self.read_next_batch_with_custom_metadata().batch

    __next__ = read_next_batch

    def __iter__(self):
        return self

    def iter_batches_with_custom_metadata(self):
        """Every remaining batch with its custom metadata."""
        while True:
            try:
                yield self.read_next_batch_with_custom_metadata()
            except StopIteration:
                return

    def read_all(self):
        """Every remaining batch, as a :class:`marrow.Table`."""
        return Table.from_batches(list(self), schema=self.schema)


def open_file(source, *, native_codecs=True):
    """A reader over the IPC file at the path `source`."""
    return RecordBatchFileReader.wrap(_ma.ipc_open_file(_path(source), native_codecs))


def open_stream(source, *, native_codecs=True):
    """A reader over the IPC stream at the path `source`."""
    return RecordBatchStreamReader.wrap(
        _ma.ipc_open_stream(_path(source), native_codecs)
    )
