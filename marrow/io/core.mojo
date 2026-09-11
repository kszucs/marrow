"""What a storage backend has to provide.

Two traits, and the pieces both sides of them need. The backends live one file
per provider — `local.mojo`, `opendal.mojo` — so adding a service means adding
a file rather than editing four.

`ByteSource` is the read seam and `ByteSink` the write seam. Parquet, Arrow IPC,
and the CSV/JSON readers that come later address bytes through these and
nothing else, which is what makes a backend written once work for every format.
"""

from ..buffers import Buffer


comptime FOOTER_READ_SIZE = 64 * 1024
"""How much of an object's tail to fetch speculatively when opening it.

Big enough that one read almost always covers a footer, small enough that it is
not a download. A property of *sources* rather than of either format, which is
why it lives here: Parquet and IPC both play this trick, and a window tuned for
one of them that silently applied to only that one is how the two drift.
"""


def require_range(offset: Int, length: Int, size: Int, what: String) raises:
    """Reject a range that is not wholly inside an object of `size` bytes.

    Shared by every source: a memory map would fault, a heap buffer would read
    the 64-byte padding a `Buffer` rounds up to, and an object store would
    answer a short read -- three different wrong answers to the same question,
    so it is asked in one place. `what` names the caller for the message.
    """
    if offset < 0 or length < 0 or offset + length > size:
        raise Error(
            what,
            ": [",
            offset,
            ", ",
            offset + length,
            ") outside an object of ",
            size,
            " bytes",
        )


struct Fetched(Movable, Sized):
    """The bytes of one batch of ranges, owned by whoever asked for them.

    The spans it vends are tied to **its** origin, not the source's. That is the
    whole point: a source that must allocate keeps the storage here rather than
    in an arena shared with the decode workers, so the batch is built once on
    the calling thread and borrowed immutably afterwards — no interior
    mutability, no lock, no atomics, and no "release invalidates every span you
    were handed" rule for a caller to get wrong. Residency is bounded by the
    batch's own scope.

    `BufferSource` fills `_bufs` with N copies of one mapping handle: N ref-count
    bumps, zero bytes moved, so the local path stays zero-copy. A fetching
    source puts one freshly allocated `Buffer` per range in instead.
    """

    var _bufs: List[Buffer[mut=False]]
    var _at: List[Int]
    """Byte offset of range `i` *inside* `_bufs[i]` — absolute for a whole-file
    map, zero for a per-range allocation."""
    var _len: List[Int]

    def __init__(out self, capacity: Int = 0):
        self._bufs = List[Buffer[mut=False]](capacity=capacity)
        self._at = List[Int](capacity=capacity)
        self._len = List[Int](capacity=capacity)

    def append(mut self, var buf: Buffer[mut=False], at: Int, length: Int):
        self._bufs.append(buf^)
        self._at.append(at)
        self._len.append(length)

    def __len__(self) -> Int:
        return len(self._bufs)

    def span(ref self, i: Int) raises -> Span[UInt8, origin_of(self)]:
        """Range `i`'s bytes. No copy — the storage is a field of `self`, so
        widening its origin to the batch's is sound, and it is what every
        caller relies on."""
        if i < 0 or i >= len(self._bufs):
            raise Error("Fetched.span: index ", i, " out of range")
        return rebind[Span[UInt8, origin_of(self)]](
            self._bufs[i].view[DType.uint8](self._at[i], self._len[i]).as_span()
        )


trait ByteSource(Deinitable, Movable):
    """A random-access source of one object's bytes."""

    def size(self) -> Int:
        """The object's total length in bytes.

        Non-raising: a remote source resolves this once, with `stat`, when it is
        constructed. A backend that cannot answer it — an HTTP endpoint with no
        `Content-Length` — cannot back a format that seeks, and must fail at
        construction rather than report zero here, which would surface as a
        corrupt-footer error blaming the file."""
        ...

    def read_at(
        ref self, offset: Int, length: Int
    ) raises -> Span[UInt8, origin_of(self)]:
        """One region, borrowed from storage the source owns.

        **Call this on one thread only.** A source that has to fetch must
        retain what it hands back, for as long as `self` lives, because the
        span is borrowed from the source rather than owned by the caller --
        `OpenDalSource` keeps an arena to do it, and mutates that arena without
        a lock, which marrow has none of. Single-range callers are all
        single-threaded: the footer, the page index and the bloom filters are
        read before any fan-out. **A fan-out must use `read_ranges`**, which is
        what `ParquetFile.read` does.

        `raises` because an object store can fail where a memory map cannot — a
        503 needs somewhere to go other than `abort()`."""
        ...

    def read_ranges(ref self, ranges: List[Tuple[Int, Int]]) raises -> Fetched:
        """Every `(offset, length)` at once, in one caller-owned batch.

        Declared abstract rather than defaulted: a default would have to produce
        an owning `Buffer` out of `read_at`'s borrowed `Span`, which it cannot,
        and defaulting `read_at` off a shared primitive instead makes the
        default's return value outlive a local. Five lines per conformer is the
        price."""
        ...


trait ByteSink(Deinitable, Movable):
    """A sequential, append-only destination for one object's bytes."""

    def write[o: Origin[mut=False]](mut self, data: Span[UInt8, o]) raises:
        """Append `data`. May buffer; only `close` promises durability."""
        ...

    def close(mut self) raises:
        """Commit.

        **Dropping a sink without closing it discards the output.** That is the
        same all-or-nothing contract `Path.write_bytes` gave implicitly — a
        failed write left no file rather than a truncated one — made explicit,
        and the same one an object store enforces anyway, since a multipart
        upload that is never completed does not exist.

        Idempotent, so a `close()` in a `finally` and a `close()` on the happy
        path do not fight."""
        ...


# ---------------------------------------------------------------------------
# The adapter every buffering writer uses.
# ---------------------------------------------------------------------------


struct BufferedSink[S: ByteSink](Movable):
    """A staging `List[UInt8]` in front of a sink, plus the absolute position.

    This is what lets the writers keep encoding into a `List[UInt8]` — which is
    what `FileMetaData.write_footer`, `ColumnWriter.write`,
    `OffsetIndex.append_to` and `LittleEndian.append` all take — while the bytes
    leave memory at a boundary the writer chooses. Parquet flushes per row
    group, IPC per message; peak residency drops from the whole file to one row
    group. Because the encoders keep their `List[UInt8]` vocabulary, none of
    `codecs.mojo` or `format.mojo` changes and nothing extra is monomorphized.

    **`tell()` is absolute, and that is the load-bearing part.** Every recorded
    offset in both formats is a *file* offset, and `len(buffer())` stopped being
    one the moment a flush happened.
    """

    var _sink: Self.S
    var _buf: List[UInt8]
    var _flushed: Int
    """Bytes already handed to the sink. `tell` is this plus the staging
    buffer's length."""

    def __init__(out self, var sink: Self.S):
        self._sink = sink^
        self._buf = List[UInt8]()
        self._flushed = 0

    def buffer(mut self) -> ref[self._buf] List[UInt8]:
        """The staging buffer, for encoders that take `mut out: List[UInt8]`."""
        return self._buf

    def tell(self) -> Int:
        """The absolute position in the output — what a recorded offset means.
        """
        return self._flushed + len(self._buf)

    def write[o: Origin[mut=False]](mut self, data: Span[UInt8, o]) raises:
        self._buf.extend(data)

    def pad_to(mut self, alignment: Int) raises:
        """Zero-pad to an `alignment` boundary **of the output**, not of the
        staging buffer — the two stop agreeing after the first flush."""
        var rem = self.tell() % alignment
        if rem != 0:
            for _ in range(alignment - rem):
                self._buf.append(0)

    def flush(mut self) raises:
        """Hand the staging buffer to the sink and start a new one."""
        if len(self._buf) == 0:
            return
        self._sink.write(Span(self._buf))
        self._flushed += len(self._buf)
        self._buf.clear()

    def close(mut self) raises:
        """Flush, then commit the sink."""
        self.flush()
        self._sink.close()

    def sink(ref self) -> ref[self._sink] Self.S:
        """The underlying sink — how a caller reaches `MemorySink.bytes()`."""
        return self._sink
