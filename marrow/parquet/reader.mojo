"""Native Parquet reader → Arrow: `read_table` orchestrates the decode.

The file is memory-mapped; per row group each leaf column is decoded by a
`ColumnReader` (see `column.mojo`), and the results are folded back into the
Arrow type tree by `SchemaNode.assemble`. Milestone: flat columns + struct
nesting; primitives, string/binary; PLAIN and dictionary encodings; v1/v2 pages.
"""

from std.ffi import external_call
from std.io.file import FileHandle

from ..arrays import AnyArray
from ..schema import Schema
from ..tabular import Table, RecordBatch

from .compression import Codecs
from .schema import parquet_to_arrow
from .format import read_footer
from .column import ColumnReader


# ---------------------------------------------------------------------------
# Memory-mapped file — read zero-copy instead of copying the file in
# ---------------------------------------------------------------------------


struct MappedFile(Movable):
    """A read-only mmap of the whole file, unmapped when the value drops.
    Decoded values are copied into owned Arrow buffers, so the map only needs to
    outlive the decode."""

    var ptr: UnsafePointer[UInt8, ImmutUntrackedOrigin]
    var size: Int

    def __init__(out self, path: String) raises:
        # Mojo's file open (the libc variadic `open` cannot be external_call'd in
        # an archive build); mmap/lseek are plain syscalls and are fine.
        var f = FileHandle(path, "r")
        var size = Int(
            external_call["lseek", Int64](
                f.handle, Int64(0), Int(2)
            )  # SEEK_END
        )
        # PROT_READ=1, MAP_PRIVATE=2; the mapping outlives the fd.
        var ptr = external_call[
            "mmap", UnsafePointer[UInt8, ImmutUntrackedOrigin]
        ](UInt(0), size, Int32(1), Int32(2), Int32(f.handle), Int64(0))
        _ = f^  # close the fd; mmap stays valid
        if Int(ptr) == 0 or Int(ptr) == -1:
            raise Error("parquet: mmap failed for " + path)
        self.ptr = ptr
        self.size = size

    def span(self) -> Span[UInt8, ImmutUntrackedOrigin]:
        return Span[UInt8, ImmutUntrackedOrigin](ptr=self.ptr, length=self.size)

    def __del__(deinit self):
        _ = external_call["munmap", Int32](self.ptr, self.size)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def read_table(path: String) raises -> Table:
    """Read a Parquet file into a Marrow `Table`."""
    var mapped = MappedFile(path)
    var data = mapped.span()
    var meta = read_footer(data)
    var parsed = parquet_to_arrow(meta)

    var codecs = Codecs()
    var batches = List[RecordBatch]()
    for ref rg in meta.row_groups:
        var leaf_arrays = List[AnyArray]()
        for ci in range(len(parsed.leaves)):
            var reader = ColumnReader(
                data,
                rg.columns[ci].meta_data.copy(),
                parsed.leaves[ci].copy(),
                rg.num_rows,
            )
            leaf_arrays.append(reader.read(codecs))
        var cols = List[AnyArray]()
        for ref node in parsed.nodes:
            cols.append(node.assemble(leaf_arrays))
        batches.append(
            RecordBatch(schema=Schema(copy=parsed.schema), columns=cols^)
        )

    if len(batches) == 0:
        batches.append(RecordBatch.empty(parsed.schema))
    var result = Table.from_batches(parsed.schema, batches)
    # `data` is an untracked view into `mapped`; keep the map alive until every
    # value has been copied into owned Arrow buffers above, then unmap.
    _ = mapped^
    return result^
