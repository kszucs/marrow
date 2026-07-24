"""The byte-source seam for the Parquet reader.

The decoder addresses column data by *absolute file offset* (from
`ColumnMetaData` / `PageLocation`) and reads it as one contiguous byte span.
`ByteSource` abstracts *where* those bytes come from so the decode path no
longer hard-codes a local `mmap`: today the only implementation is
`MappedFile` (a whole-file memory map), but the same seam admits a streaming
reader or a remote (OpenDAL) object store later without touching the decoder.

`read_at` returns a *borrowed* view over the source's own storage — no copy —
so the local path stays zero-copy. The view is only valid while the source is
alive; `ParquetFile` owns its source for its whole lifetime, so every span it
hands to the decoder is backed for the duration of the read.
"""

from std.ffi import external_call
from std.io.file import FileHandle


trait ByteSource(ImplicitlyDeletable, Movable):
    """A random-access source of the Parquet file's bytes.

    The reader consumes the file through this seam: `size()` bounds the footer
    read and `read_at(offset, length)` returns the bytes of a metadata or page
    region. Implementations must return a *borrowed, non-owning* view (no copy)
    tied to the source's storage — the caller keeps the source alive for as long
    as it uses the returned span.
    """

    def size(self) -> Int:
        """The total number of bytes in the file."""
        ...

    def read_at(
        self, offset: Int, length: Int
    ) -> Span[UInt8, ImmUntrackedOrigin]:
        """A borrowed view of `length` bytes starting at absolute `offset`.

        No copy — the span points into the source's own storage and is valid
        only while the source is alive.
        """
        ...


# ---------------------------------------------------------------------------
# Memory-mapped file — the local ByteSource: read zero-copy instead of copying
# the file in. This is the one place the mmap/lseek syscalls live.
# ---------------------------------------------------------------------------


struct MappedFile(ByteSource):
    """A read-only mmap of the whole file, unmapped when the value drops.
    Decoded values are copied into owned Arrow buffers, so the map only needs to
    outlive the decode."""

    var ptr: UnsafePointer[UInt8, ImmUntrackedOrigin]
    var _size: Int

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
            "mmap", UnsafePointer[UInt8, ImmUntrackedOrigin]
        ](UInt(0), size, Int32(1), Int32(2), Int32(f.handle), Int64(0))
        _ = f^  # close the fd; mmap stays valid
        if Int(ptr) == 0 or Int(ptr) == -1:
            raise Error("parquet: mmap failed for " + path)
        self.ptr = ptr
        self._size = size

    def size(self) -> Int:
        return self._size

    def read_at(
        self, offset: Int, length: Int
    ) -> Span[UInt8, ImmUntrackedOrigin]:
        # A sub-span of the whole-file map — zero copy, same untracked origin as
        # the backing mmap (kept alive by this MappedFile).
        return Span[UInt8, ImmUntrackedOrigin](ptr=self.ptr, length=self._size)[
            offset : offset + length
        ]

    def __del__(deinit self):
        _ = external_call["munmap", Int32](self.ptr, self._size)
