# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What every file format provides: batches in, batches out.

A reader hands out one `RecordBatch` at a time and a writer takes one at a
time, so a file of any size passes through in the memory of a batch -- a row
group, a block, a message -- rather than the whole of it. `Format` names a
format's reader, so a scan can be written once for all of them.
"""

from ..io import StorageOptions
from ..schema import Schema
from ..tabular import RecordBatch, Table


trait BatchReader(Deinitable, Movable):
    """Reads a file one `RecordBatch` at a time."""

    def schema(self) -> Schema:
        """The schema of every batch this reader returns."""
        ...

    def read_next_batch(mut self) raises -> Optional[RecordBatch]:
        """The next batch, or `None` once the file is exhausted."""
        ...

    def read_all(mut self) raises -> Table:
        """Every remaining batch, as a table with one chunk per batch."""
        var batches = List[RecordBatch]()
        while True:
            var batch = self.read_next_batch()
            if not batch:
                break
            batches.append(batch.take())
        return Table.from_batches(self.schema(), batches)


trait BatchWriter(Deinitable, Movable):
    """Writes a file one `RecordBatch` at a time."""

    def write(mut self, batch: RecordBatch) raises:
        """Append `batch` to the file."""
        ...

    def write(mut self, table: Table) raises:
        """Append `table`'s batches to the file, in order."""
        for ref batch in table.to_batches():
            self.write(batch)

    def close(mut self) raises:
        """Finish the file and commit it to its sink. Closing twice is a
        no-op; a writer dropped without closing commits nothing."""
        ...


trait Format:
    """A file format, as a scan sees it: its name, the extensions its files
    carry, and a way to open one as a `BatchReader`."""

    comptime Reader: BatchReader

    @staticmethod
    def name() -> StaticString:
        """The format's name, as a plan prints it: `Ipc`, `Json`, ..."""
        ...

    @staticmethod
    def extensions() -> List[StaticString]:
        """The lower-case file extensions, dot included, that name this
        format."""
        ...

    @staticmethod
    def open(
        path: String, schema: Optional[Schema], options: StorageOptions
    ) raises -> Self.Reader:
        """Open the file at `path` -- a path or a URL, `options` configuring
        its storage -- to read `schema`'s columns, or every column of the
        file's own schema when `schema` is `None`.

        `schema` is a projection a format may use to read less; its batches
        may still carry other columns, so a caller selects `schema`'s names
        from each.
        """
        ...
