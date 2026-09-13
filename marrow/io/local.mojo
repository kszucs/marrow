"""The local provider: bytes already on this machine.

A memory map, bytes the caller already holds, a file on disk, a buffer in
memory. One file because they are one backend — nothing here reaches a network
and nothing here needs `libopendal_c`, which is what lets marrow read a local
Parquet file with that library absent.

`BufferSource` covers *both* ways bytes become resident, and that is why there
is no separate `FileSource`: reading a local file **is** mapping it into
memory. After construction a mapping and a heap buffer are the same thing — a
`Buffer` plus a logical length, read through the same bounds check — and the
distinction that matters, unmap versus free on last release, is the `Buffer`'s
allocation kind rather than this module's business.
"""

from std.ffi import c_int, external_call, get_errno
from std.io.file import FileHandle
from std.os import remove

from ..buffers import Buffer
from .core import ByteSink, ByteSource, Fetched, require_range


def _getpid() -> Int:
    """The process id, to uniquify a temp path. `std.os` does not expose it."""
    return Int(external_call["getpid", c_int]())


def _rename(var src: String, var dst: String) raises:
    """`rename(2)`. Not in `std.os`, which has `remove`/`unlink` but no rename;
    this is the same `external_call` + `errno` shape `os.remove` uses.

    Both paths are taken by value because `as_c_string_span` mutates the
    `String` (it appends the terminator) and so needs an lvalue."""
    var err = external_call["rename", c_int](
        src.as_c_string_span(), dst.as_c_string_span()
    )
    if err != 0:
        raise Error("rename '", src, "' -> '", dst, "': ", String(get_errno()))


# ---------------------------------------------------------------------------
# Resident bytes — the local source, however they got here.
# ---------------------------------------------------------------------------


struct BufferSource(ByteSource):
    """An object already resident in memory, read without copying.

    One type for the two ways bytes get here, because after construction there
    is no difference: a whole-file memory map and a heap buffer are both a
    `Buffer` plus a logical length, and every read is the same bounds check and
    the same view. The distinction that matters -- unmap on last release versus
    free on last release -- is the `Buffer`'s allocation kind, which is
    `Buffer`'s job and not this one's.

    A mapping is owned by a MAPPED-kind `Buffer`, so it is unmapped when the
    last reference drops rather than when this struct goes out of scope.
    """

    var _buf: Buffer[mut=False]
    var _size: Int
    """The object's logical length. **Not** `len(_buf)`, which a `Buffer`
    rounds up to a multiple of 64 -- a file offset must be bounded by the file.
    """

    def __init__(out self, path: String) raises:
        """Memory-map a local file."""
        self._buf = Buffer.mmap_file(path)
        self._size = self._buf.mapped_size()

    def __init__(out self, data: Span[UInt8, _]):
        """Copy bytes the caller already has -- a Python `bytes`, an IPC
        payload off a socket, a fixture in a test."""
        # Element-wise rather than a `memcpy`: this module is deliberately not
        # on CLAUDE.md's `unsafe_ptr()` allowlist, and `Buffer` exposes no bulk
        # copy from a `Span`. Adopting bytes is not a hot path -- the sources
        # read in a loop are a mapping, which copies nothing, and
        # `OpenDalSource`, whose module *is* on the allowlist and does `memcpy`.
        var b = Buffer.alloc_uninit[DType.uint8](len(data))
        for i in range(len(data)):
            b.unsafe_set[DType.uint8](i, data[i])
        self._size = len(data)
        self._buf = b^.to_immutable()

    def __init__(out self, var buf: Buffer[mut=False], size: Int):
        """Adopt a buffer someone else allocated."""
        self._buf = buf^
        self._size = size

    def __init__(out self, *, copy: Self):
        self._buf = copy._buf
        self._size = copy._size

    def size(self) -> Int:
        return self._size

    def read_at(
        ref self, offset: Int, length: Int
    ) raises -> Span[UInt8, origin_of(self)]:
        require_range(offset, length, self._size, "BufferSource.read_at")
        # The buffer is a field of `self`, so widening its origin to the
        # source's is sound -- and it is what the trait promises callers.
        return rebind[Span[UInt8, origin_of(self)]](
            self._buf.view[DType.uint8](offset, length).as_span()
        )

    def read_ranges(ref self, ranges: List[Tuple[Int, Int]]) raises -> Fetched:
        # One allocation, N handles: a `Buffer` copy is a ref-count bump, so
        # this moves no bytes. Batching buys resident data nothing -- it costs
        # it nothing either, which is what lets one call shape serve both ends
        # of the seam.
        var out = Fetched(capacity=len(ranges))
        for ref r in ranges:
            require_range(r[0], r[1], self._size, "BufferSource.read_ranges")
            out.append(self._buf, r[0], r[1])
        return out^


# ---------------------------------------------------------------------------
# Local file — write to a sibling temp path, rename on commit.
# ---------------------------------------------------------------------------


struct FileSink(ByteSink):
    """A local file, committed by `rename(2)`.

    The temp-and-rename is what *preserves* the behaviour incremental writing
    would otherwise destroy. `Path.write_bytes` was all-or-nothing for free:
    either the whole file appeared or none of it did. Streaming to the final
    path directly would leave a truncated `.parquet` behind on a mid-write
    failure — a file that looks readable and is not. Renaming within a
    directory is atomic, so a reader sees the old file or the new one.

    It also makes commit-on-close a uniform property of the seam rather than an
    OpenDAL quirk, which is what lets a writer be written once against
    `ByteSink` and mean the same thing locally and remotely.
    """

    var _path: String
    var _tmp: String
    var _file: Optional[FileHandle]
    """Open until `close` takes it. Emptiness *is* the closed flag: a separate
    `Bool` would be a second copy of the same bit, and the two disagreeing would
    abort inside `Optional.value()` rather than raise."""

    def __init__(out self, path: String) raises:
        self._path = path
        # A sibling of the destination, so the rename stays within one
        # filesystem — across devices it is not atomic and not even a rename.
        self._tmp = String(path, ".marrow-tmp-", _getpid())
        self._file = open(self._tmp, "w")

    def __deinit__(deinit self):
        # An abandoned sink leaves no output, but it must not leave litter
        # either. Unlinking a path whose descriptor is still open is fine on
        # POSIX -- the name goes away now, the bytes when the handle drops just
        # below -- so the temp file does not have to be closed first. Best
        # effort: a failure here has nowhere to go.
        if self._file:
            try:
                remove(self._tmp)
            except:
                pass

    def write[o: Origin[mut=False]](mut self, data: Span[UInt8, o]) raises:
        if not self._file:
            raise Error("FileSink.write: '", self._path, "' is closed")
        if len(data) > 0:
            self._file.value().write_bytes(data)

    def close(mut self) raises:
        if not self._file:
            return
        # Drop the handle before renaming: the bytes have to be flushed to the
        # temp path before it becomes the destination.
        self._file = None
        _rename(self._tmp, self._path)


# ---------------------------------------------------------------------------
# In-memory sink — bytes that never reach a filesystem.
# ---------------------------------------------------------------------------


struct MemorySink(ByteSink):
    """Accumulates into a `List[UInt8]`.

    What the Python lane writes through when it wants bytes rather than a
    file, and what a test asserts on without touching the filesystem. Not
    reachable through a URI, for the reason `BufferSource` gives.
    """

    var _out: List[UInt8]
    var _closed: Bool

    def __init__(out self):
        self._out = List[UInt8]()
        self._closed = False

    def write[o: Origin[mut=False]](mut self, data: Span[UInt8, o]) raises:
        if self._closed:
            raise Error("MemorySink.write: closed")
        self._out.extend(data)

    def close(mut self) raises:
        self._closed = True

    def bytes(ref self) -> ref[self._out] List[UInt8]:
        """The accumulated bytes. Readable before `close`, unlike a file — an
        in-memory sink has nothing to commit."""
        return self._out
