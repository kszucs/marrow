# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What every file format provides: batches in, batches out.

A reader hands out one `RecordBatch` at a time and a writer takes one at a
time, so a file of any size passes through in the memory of a batch -- a row
group, a block, a message -- rather than the whole of it. A chunked reader
splits a file into parts that decode independently, so several threads can
read one file at once. `Format` names a format's readers, so a scan can be
written once for all of them.
"""

from ..execution import ExecContext
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


trait ChunkedReader(Deinitable, Movable):
    """Reads a file as chunks that decode independently -- a record batch, a
    row group, a run of blocks, the lines of a byte range -- in any order and
    from several threads at once."""

    def num_chunks(self) -> Int:
        """How many chunks the file has."""
        ...

    def read_chunk(self, i: Int, ctx: ExecContext) raises -> List[RecordBatch]:
        """Chunk `i`'s rows, as one or more batches, read on `ctx`."""
        ...


trait Format:
    """A file format, as a scan sees it: its name, the extensions its files
    carry, and the ways to open one -- in order as a `BatchReader`, or in
    parallel as a `ChunkedReader`."""

    comptime Reader: BatchReader
    comptime Chunks: ChunkedReader

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

    @staticmethod
    def chunks(
        path: String, schema: Optional[Schema], options: StorageOptions
    ) raises -> Self.Chunks:
        """Open the file at `path` as `open` does, to read it chunk by chunk,
        reading only what it takes to know where its chunks are -- a footer, a
        header, block headers, a size."""
        ...
