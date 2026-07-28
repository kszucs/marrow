"""The byte-source seam for the Parquet reader.

The decoder addresses column data by *absolute file offset* (from
`ColumnMetaData` / `PageLocation`) and reads it as one contiguous byte span.
`ByteSource` abstracts *where* those bytes come from so the decode path no
longer hard-codes a local `mmap`: today the only implementation is
`MappedFile` (a whole-file memory map), but the same seam admits a streaming
reader or a remote (OpenDAL) object store later without touching the decoder.

`read_at` returns a span **tied to the source's origin** — borrowed, but
tracked, so the compiler knows a page body outlives nothing it shouldn't. The
storage itself is owned by a `Buffer`: `MappedFile` holds the mapping as a
MAPPED-kind allocation, which unmaps when the last reference drops rather than
when this struct happens to go out of scope.
"""

from ..buffers import Buffer


trait ByteSource(ImplicitlyDeletable, Movable):
    """A random-access source of the Parquet file's bytes.

    The reader consumes the file through this seam: `size()` bounds the footer
    read and `read_at(offset, length)` returns the bytes of a metadata or page
    region — a borrowed view over storage the source owns, so no copy.
    """

    def size(self) -> Int:
        """The total number of bytes in the file."""
        ...

    def read_at(
        ref self, offset: Int, length: Int
    ) -> Span[UInt8, origin_of(self)]:
        """A view of `length` bytes starting at absolute `offset`.

        No copy. The origin ties the result to this source, so the borrow is
        checked rather than asserted in a comment.
        """
        ...


# ---------------------------------------------------------------------------
# Memory-mapped file — the local ByteSource: read zero-copy instead of copying
# the file in.
# ---------------------------------------------------------------------------


struct MappedFile(ByteSource):
    """A read-only whole-file memory map.

    The mapping is owned by a `Buffer` (MAPPED-kind allocation), so it is
    unmapped when the last reference drops. Previously this struct did its own
    `mmap`/`munmap` and handed out `ImmUntrackedOrigin` spans, which meant the
    only thing keeping a page body valid was the reader outliving it by
    convention.
    """

    var _buf: Buffer[mut=False]
    var _size: Int

    def __init__(out self, path: String) raises:
        self._buf = Buffer.mmap_file(path)
        # The mapping's true extent, not the Buffer's 64-byte-padded logical
        # size: Parquet's footer offsets are *file* offsets.
        self._size = self._buf.mapped_size()

    def __init__(out self, *, copy: Self):
        self._buf = copy._buf
        self._size = copy._size

    def size(self) -> Int:
        return self._size

    def read_at(
        ref self, offset: Int, length: Int
    ) -> Span[UInt8, origin_of(self)]:
        # The buffer is a field of `self`, so widening its origin to the
        # source's is sound — and it is what the trait promises callers.
        return rebind[Span[UInt8, origin_of(self)]](
            self._buf.view[DType.uint8](offset, length).as_span()
        )
