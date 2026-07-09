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
from ..schema import Schema
from ..tabular import Table, RecordBatch
from ..scalars import (
    AnyScalar,
    BoolScalar,
    StringScalar,
    Int8Scalar,
    Int16Scalar,
    Int32Scalar,
    Int64Scalar,
    UInt8Scalar,
    UInt16Scalar,
    UInt32Scalar,
    UInt64Scalar,
    Float32Scalar,
    Float64Scalar,
)
from .. import dtypes as dt

from .utils import CompressionLibs
from .codecs import LittleEndian
from .schema import SchemaMapping, Projection, DecodedLeaf
from .format import FileMetaData
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


# Below this many rows a file is decoded on one thread — the thread-dispatch
# overhead is not worth it for small files.
comptime _PARALLEL_MIN_ROWS = 4096


def read_table(
    path: String, columns: Optional[List[String]] = None
) raises -> Table:
    """Read a Parquet file into a Marrow `Table`.

    `columns` projects the read to the named top-level columns (in the given
    order); `None` reads all. Only the selected columns' chunks are touched.

    Every (row group, selected leaf) pair decodes independently — each reads a
    disjoint byte range of the shared read-only mmap and writes its own result
    slot — so the whole grid is decoded across `num_physical_cores()` workers.
    Each worker owns a `CompressionLibs` (the lazy `dlopen` handles and reused size cell
    are not shareable across threads); the mmap and metadata are read-only.
    """
    var mapped = MappedFile(path)
    var data = mapped.span()
    var meta = FileMetaData.read_footer(data)
    var mapping = SchemaMapping.from_parquet(meta)

    # the read plan: which column chunks to decode and how to reassemble them
    var plan = mapping.project(columns.value()) if columns else mapping.full()

    var num_leaves = len(plan.decode_order)
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
        var codecs = (
            CompressionLibs()
        )  # per-worker: lazy dlopen handles are not shared
        var t = w
        while t < total:
            var rg_idx = t // num_leaves
            # original column-chunk index for this compact slot
            var orig = plan.decode_order[t % num_leaves]
            ref rg = meta.row_groups[rg_idx]
            # ColumnReader.decode picks the flat vs leveled path from the leaf's
            # max repetition, so the same call serves every column shape.
            var reader = ColumnReader(
                data,
                rg.columns[orig].meta_data.copy(),
                mapping.leaves[orig].copy(),
                rg.num_rows,
            )
            grid[t] = reader.decode(codecs)
            t += nt

    sync_parallelize[worker](nt)

    # Fold each row group's decoded leaves back into the Arrow type tree.
    var batches = List[RecordBatch]()
    for rg_idx in range(num_rg):
        var decoded = List[DecodedLeaf]()
        for ci in range(num_leaves):
            decoded.append(grid[rg_idx * num_leaves + ci].take())
        var cols = List[AnyArray]()
        for ref node in plan.nodes:
            cols.append(node.assemble(decoded))
        batches.append(
            RecordBatch(schema=Schema(copy=plan.schema), columns=cols^)
        )

    if len(batches) == 0:
        batches.append(RecordBatch.empty(Schema(copy=plan.schema)))
    var result = Table.from_batches(Schema(copy=plan.schema), batches)
    # `data` is an untracked view into `mapped`; keep the map alive until every
    # value has been copied into owned Arrow buffers above, then unmap.
    _ = mapped^
    return result^


# ---------------------------------------------------------------------------
# Metadata / statistics — read the footer without decoding any column data.
# Mirrors PyArrow's `read_metadata` / `ColumnChunkMetaData.statistics`.
# ---------------------------------------------------------------------------


def read_metadata(path: String) raises -> FileMetaData:
    """Read only the file footer: schema, row groups, and per-column-chunk
    metadata (offsets, sizes, codec, null_count, and the raw min/max statistic
    bytes). No column data is decoded. Mirrors `pyarrow.parquet.read_metadata`.
    """
    var mapped = MappedFile(path)
    var meta = FileMetaData.read_footer(mapped.span())
    _ = mapped^  # read_footer copies every field into owned storage
    return meta^


struct ColumnStatistics(Copyable, Movable):
    """Decoded per-column-chunk statistics: `null_count` (-1 if absent) plus the
    `min`/`max` bounds as typed scalars, valid only when `has_min_max`."""

    var has_min_max: Bool
    var null_count: Int
    var min: AnyScalar
    var max: AnyScalar

    def __init__(out self):
        self.has_min_max = False
        self.null_count = -1
        self.min = BoolScalar(is_valid=False)
        self.max = BoolScalar(is_valid=False)


def _decode_stat(
    dtype: dt.AnyDataType, b: List[UInt8]
) raises -> Optional[AnyScalar]:
    """Decode one PLAIN-encoded min/max value to a typed scalar, mirroring the
    writer's encoding (`LittleEndian.fixed` reads `size_of[dt]` LE bytes and
    reinterprets — the inverse of the writer's byte emission). Returns None for
    types this reader does not yet decode (raw bytes stay in `read_metadata`).
    """
    var s = Span(b)
    if dtype == dt.int8:
        return AnyScalar(
            Int8Scalar(LittleEndian.fixed[DType.int32](s, 0).cast[DType.int8]())
        )
    elif dtype == dt.int16:
        return AnyScalar(
            Int16Scalar(
                LittleEndian.fixed[DType.int32](s, 0).cast[DType.int16]()
            )
        )
    elif dtype == dt.int32:
        return AnyScalar(Int32Scalar(LittleEndian.fixed[DType.int32](s, 0)))
    elif dtype == dt.uint8:
        return AnyScalar(
            UInt8Scalar(
                LittleEndian.fixed[DType.uint32](s, 0).cast[DType.uint8]()
            )
        )
    elif dtype == dt.uint16:
        return AnyScalar(
            UInt16Scalar(
                LittleEndian.fixed[DType.uint32](s, 0).cast[DType.uint16]()
            )
        )
    elif dtype == dt.uint32:
        return AnyScalar(UInt32Scalar(LittleEndian.fixed[DType.uint32](s, 0)))
    elif dtype == dt.int64:
        return AnyScalar(Int64Scalar(LittleEndian.fixed[DType.int64](s, 0)))
    elif dtype == dt.uint64:
        return AnyScalar(UInt64Scalar(LittleEndian.fixed[DType.uint64](s, 0)))
    elif dtype == dt.float32:
        return AnyScalar(Float32Scalar(LittleEndian.fixed[DType.float32](s, 0)))
    elif dtype == dt.float64:
        return AnyScalar(Float64Scalar(LittleEndian.fixed[DType.float64](s, 0)))
    elif dtype == dt.bool_:
        return AnyScalar(BoolScalar(len(b) > 0 and b[0] != 0))
    elif dtype.is_string():
        return AnyScalar(
            StringScalar(String(StringSlice(unsafe_from_utf8=Span(b))))
        )
    else:
        return None


def read_statistics(path: String) raises -> List[List[ColumnStatistics]]:
    """Per-(row group, leaf column) decoded statistics, indexed
    `result[row_group][leaf]`. `min`/`max` are typed scalars matching each
    column's Arrow type; `has_min_max` is false when the file stored no bounds
    or the type's bounds are not yet decoded."""
    var meta = read_metadata(path)
    var mapping = SchemaMapping.from_parquet(meta)
    var out = List[List[ColumnStatistics]]()
    for ref rg in meta.row_groups:
        var row = List[ColumnStatistics]()
        for ci in range(len(rg.columns)):
            ref cm = rg.columns[ci].meta_data
            var cs = ColumnStatistics()
            cs.null_count = cm.null_count
            if cm.has_min_max:
                var mn = _decode_stat(mapping.leaves[ci].dtype, cm.min_value)
                var mx = _decode_stat(mapping.leaves[ci].dtype, cm.max_value)
                if Bool(mn) and Bool(mx):
                    cs.min = mn.value().copy()
                    cs.max = mx.value().copy()
                    cs.has_min_max = True
            row.append(cs^)
        out.append(row^)
    return out^
