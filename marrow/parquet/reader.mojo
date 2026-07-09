"""Native Parquet reader → Arrow: `read_table` orchestrates the decode.

The file is memory-mapped; per row group each leaf column is decoded by a
`ColumnReader` (see `column.mojo`), and the results are folded back into the
Arrow type tree by `SchemaNode.assemble`. Milestone: flat columns + struct
nesting; primitives, string/binary; PLAIN and dictionary encodings; v1/v2 pages.
"""

from std.ffi import external_call
from std.io.file import FileHandle
from std.algorithm.functional import sync_parallelize
from std.sys.info import num_physical_cores

from ..arrays import AnyArray
from ..dtypes import Field
from ..schema import Schema
from ..tabular import Table, RecordBatch

from .compression import Codecs
from .schema import ParsedSchema, SchemaNode
from .format import read_footer
from .column import ColumnReader
from .nested import DecodedLeaf, LeveledColumnReader


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


# Below this many rows a file is decoded on one thread — the thread-dispatch
# overhead is not worth it for small files.
comptime _PARALLEL_MIN_ROWS = 4096


def _project(
    parsed: ParsedSchema,
    columns: List[String],
    mut out_schema: Schema,
    mut nodes: List[SchemaNode],
    mut decode_order: List[Int],
) raises:
    """Select the requested top-level columns, filling `out_schema` (selected
    fields), `nodes` (assembly nodes remapped to a compact decoded list), and
    `decode_order` (original flat leaf/column-chunk indices to decode, in
    compact order)."""
    var fields = List[Field]()
    var mapping = List[Int](length=len(parsed.leaves), fill=-1)
    for ci in range(len(columns)):
        var found = -1
        for ni in range(len(parsed.nodes)):
            if parsed.nodes[ni].field.name == columns[ci]:
                found = ni
                break
        if found == -1:
            raise Error("parquet: column not found: " + columns[ci])
        ref node = parsed.nodes[found]
        var node_leaves = List[Int]()
        node.collect_leaf_indices(node_leaves)
        for orig in node_leaves:
            mapping[orig] = len(decode_order)
            decode_order.append(orig)
        nodes.append(node.remapped(mapping))
        fields.append(node.field.copy())
    out_schema = Schema(fields=fields^)


def read_table(
    path: String, columns: Optional[List[String]] = None
) raises -> Table:
    """Read a Parquet file into a Marrow `Table`.

    `columns` projects the read to the named top-level columns (in the given
    order); `None` reads all. Only the selected columns' chunks are touched.

    Every (row group, selected leaf) pair decodes independently — each reads a
    disjoint byte range of the shared read-only mmap and writes its own result
    slot — so the whole grid is decoded across `num_physical_cores()` workers.
    Each worker owns a `Codecs` (the lazy `dlopen` handles and reused size cell
    are not shareable across threads); the mmap and metadata are read-only.
    """
    var mapped = MappedFile(path)
    var data = mapped.span()
    var meta = read_footer(data)
    var parsed = ParsedSchema.from_metadata(meta)

    # compact leaf slot -> original column-chunk index
    var out_schema = Schema(copy=parsed.schema)  # full schema unless projecting
    var nodes = List[SchemaNode]()
    var decode_order = List[Int]()
    if columns:
        _project(parsed, columns.value(), out_schema, nodes, decode_order)
    else:
        nodes = parsed.nodes.copy()
        for i in range(len(parsed.leaves)):
            decode_order.append(i)

    var num_leaves = len(decode_order)
    var num_rg = len(meta.row_groups)
    var total = num_rg * num_leaves

    var nt = 1
    if total >= 2 and meta.num_rows >= _PARALLEL_MIN_ROWS:
        nt = min(num_physical_cores(), total)

    # One result slot per (row group, selected leaf), pre-sized so workers assign
    # by index without racing on list growth. (DecodedLeaf is move-only, so the
    # slots are filled by appending rather than the copy-based `fill=`.)
    var grid = List[Optional[DecodedLeaf]](capacity=total)
    for _ in range(total):
        grid.append(None)

    @parameter
    def worker(w: Int) raises:
        var codecs = Codecs()  # per-worker: lazy dlopen handles are not shared
        var t = w
        while t < total:
            var rg_idx = t // num_leaves
            var orig = decode_order[t % num_leaves]  # original column-chunk idx
            ref rg = meta.row_groups[rg_idx]
            if parsed.leaves[orig].max_rep >= 1:
                var reader = LeveledColumnReader(
                    data,
                    rg.columns[orig].meta_data.copy(),
                    parsed.leaves[orig].copy(),
                    rg.num_rows,
                )
                grid[t] = reader.read(codecs)
            else:
                var reader = ColumnReader(
                    data,
                    rg.columns[orig].meta_data.copy(),
                    parsed.leaves[orig].copy(),
                    rg.num_rows,
                )
                grid[t] = DecodedLeaf.flat(reader.read(codecs))
            t += nt

    sync_parallelize[worker](nt)

    # Fold each row group's decoded leaves back into the Arrow type tree.
    var batches = List[RecordBatch]()
    for rg_idx in range(num_rg):
        var decoded = List[DecodedLeaf]()
        for ci in range(num_leaves):
            decoded.append(grid[rg_idx * num_leaves + ci].take())
        var cols = List[AnyArray]()
        for ref node in nodes:
            cols.append(node.assemble(decoded))
        batches.append(
            RecordBatch(schema=Schema(copy=out_schema), columns=cols^)
        )

    if len(batches) == 0:
        batches.append(RecordBatch.empty(out_schema))
    var result = Table.from_batches(out_schema, batches)
    # `data` is an untracked view into `mapped`; keep the map alive until every
    # value has been copied into owned Arrow buffers above, then unmap.
    _ = mapped^
    return result^
