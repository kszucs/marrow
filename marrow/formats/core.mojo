# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What every file format provides: batches in, batches out.

A reader hands out one `RecordBatch` at a time and a writer takes one at a
time, so a file of any size passes through in the memory of a batch -- a row
group, a block, a message -- rather than the whole of it. `Format` names a
format's reader, so a scan can be written once for all of them.
"""

from ..schema import Schema
from ..tabular import RecordBatch


trait BatchReader(Deinitable, Movable):
    """Reads a file one `RecordBatch` at a time."""

    def schema(self) -> Schema:
        """The schema of every batch this reader returns."""
        ...

    def read_next_batch(mut self) raises -> Optional[RecordBatch]:
        """The next batch, or `None` once the file is exhausted."""
        ...


trait BatchWriter(Deinitable, Movable):
    """Writes a file one `RecordBatch` at a time."""

    def write(mut self, batch: RecordBatch) raises:
        """Append `batch` to the file."""
        ...

    def close(mut self) raises:
        """Finish the file and commit it to its sink. Closing twice is a
        no-op; a writer dropped without closing commits nothing."""
        ...


trait Format:
    """A file format, as a scan sees it: a name and a way to open a file of it
    as a `BatchReader`."""

    comptime Reader: BatchReader

    @staticmethod
    def name() -> StaticString:
        """The format's name, as a plan prints it: `Ipc`, `Json`, ..."""
        ...

    @staticmethod
    def open(path: String, schema: Schema) raises -> Self.Reader:
        """Open the file at `path` to read `schema`'s columns.

        `schema` is a projection a format may use to read less; its batches
        may still carry other columns, so a caller selects `schema`'s names
        from each.
        """
        ...
