"""`ByteSink` and the buffering adapter the format writers use.

Two properties carry most of the weight here, because both are things the old
`Path.write_bytes` gave for free and an incremental writer has to earn back:

- **nothing appears at the destination until `close`** — a mid-write failure
  must leave no file rather than a truncated one;
- **`BufferedSink.tell()` stays absolute across a flush** — every recorded
  offset in Parquet and in the IPC file format is a *file* offset, and
  `len(buffer())` stops being one the moment a flush happens.
"""

from std.os import listdir, remove
from std.os.path import exists
from std.testing import assert_equal, assert_false, assert_true

from ...buffers import Buffer
from ...io import BufferedSink, FileSink, MemorySink


def _has_scratch(path: String) raises -> Bool:
    """Whether a `FileSink` scratch file for `path` is still on disk.

    `FileSink` writes to `path + ".marrow-tmp-" + getpid()` and renames on
    close. The pid is not knowable here without duplicating the shim in
    `io/local.mojo`, so this matches the prefix instead -- which is also what
    a human would grep for.
    """
    var d = String("/tmp")
    var prefix = String(path[byte = d.byte_length() + 1 :], ".marrow-tmp-")
    for name in listdir(d):
        if String(name).startswith(prefix):
            return True
    return False


def _pattern(n: Int, seed: Int = 0) -> List[UInt8]:
    var out = List[UInt8](capacity=n)
    for i in range(n):
        out.append(UInt8((i * 7 + seed * 31 + 3) % 251))
    return out^


def _read_file(path: String) raises -> List[UInt8]:
    var buf = Buffer.mmap_file(path)
    var n = buf.mapped_size()
    var view = buf.view[DType.uint8](0, n).as_span()
    return List[UInt8](view)


def _assert_bytes_eq(got: Span[UInt8, _], want: Span[UInt8, _]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


# --- MemorySink -------------------------------------------------------------


def test_io_sink_memory_round_trip() raises:
    var a = _pattern(64, 1)
    var b = _pattern(32, 2)
    var sink = MemorySink()
    sink.write(Span(a))
    sink.write(Span(b))
    sink.close()

    ref got = sink.bytes()
    assert_equal(len(got), 96)
    _assert_bytes_eq(Span(got)[:64], Span(a))
    _assert_bytes_eq(Span(got)[64:], Span(b))
    _ = a^
    _ = b^


def test_io_sink_memory_write_after_close_raises() raises:
    var sink = MemorySink()
    sink.close()
    var raised = False
    try:
        var d = _pattern(4)
        sink.write(Span(d))
        _ = d^
    except:
        raised = True
    assert_true(raised)


def test_io_sink_close_is_idempotent() raises:
    var sink = MemorySink()
    var d = _pattern(8)
    sink.write(Span(d))
    sink.close()
    sink.close()
    assert_equal(len(sink.bytes()), 8)
    _ = d^


def test_io_sink_empty_write_is_allowed() raises:
    """A format writer flushing an empty staging buffer must not be a failure —
    an object store treats a zero-length body as a special case, so the seam
    has to be explicit that nothing happens."""
    var sink = MemorySink()
    var empty = List[UInt8]()
    sink.write(Span(empty))
    sink.close()
    assert_equal(len(sink.bytes()), 0)
    _ = empty^


# --- FileSink ---------------------------------------------------------------


def test_io_sink_file_commits_on_close() raises:
    var path = String("/tmp/marrow_io_sink_commit.bin")
    if exists(path):
        remove(path)
    var a = _pattern(100, 3)
    var b = _pattern(50, 4)

    var sink = FileSink(path)
    sink.write(Span(a))
    sink.write(Span(b))
    # The destination must not exist yet: an incremental writer that streamed
    # straight to `path` would leave a truncated file behind on a mid-write
    # failure, which `Path.write_bytes` never could.
    assert_false(exists(path), "FileSink published before close")
    sink.close()

    assert_true(exists(path))
    var got = _read_file(path)
    assert_equal(len(got), 150)
    _assert_bytes_eq(Span(got)[:100], Span(a))
    _assert_bytes_eq(Span(got)[100:], Span(b))
    remove(path)
    _ = a^
    _ = b^


def test_io_sink_file_abandoned_leaves_nothing() raises:
    """Dropping a sink without closing publishes nothing and leaves no litter.
    """
    var path = String("/tmp/marrow_io_sink_abandoned.bin")
    var tmp_before: Bool

    if exists(path):
        remove(path)
    var d = _pattern(64, 5)
    var sink = FileSink(path)
    sink.write(Span(d))
    tmp_before = exists(path)
    _ = sink^

    assert_false(tmp_before, "FileSink published before close")
    assert_false(exists(path), "an abandoned FileSink published its output")
    # The litter half. `FileSink.__deinit__` exists only to remove the scratch
    # file, so asserting on the destination alone would pass with that
    # destructor deleted -- which is how this test read before.
    assert_false(
        _has_scratch(path),
        "an abandoned FileSink left its .marrow-tmp-<pid> file behind",
    )
    _ = d^


def test_io_sink_file_overwrites_existing() raises:
    """The rename replaces whatever was there, which is what `write_bytes` did.
    """
    var path = String("/tmp/marrow_io_sink_overwrite.bin")
    var old = _pattern(200, 6)
    with open(path, "w") as f:
        f.write_bytes(Span(old))

    var new = _pattern(10, 7)
    var sink = FileSink(path)
    sink.write(Span(new))
    sink.close()

    var got = _read_file(path)
    assert_equal(len(got), 10)
    _assert_bytes_eq(Span(got), Span(new))
    remove(path)
    _ = old^
    _ = new^


# --- BufferedSink -----------------------------------------------------------


def test_io_sink_buffered_tell_survives_a_flush() raises:
    """The one thing a `len(buffer())` implementation gets wrong, and the one
    that would silently corrupt a Parquet footer: after a flush the staging
    buffer restarts at zero while the file position does not."""
    var b = BufferedSink(MemorySink())
    assert_equal(b.tell(), 0)

    var a = _pattern(100, 8)
    b.write(Span(a))
    assert_equal(b.tell(), 100)

    b.flush()
    assert_equal(len(b.buffer()), 0)
    assert_equal(b.tell(), 100, "tell() reset with the staging buffer")

    var c = _pattern(30, 9)
    b.write(Span(c))
    assert_equal(b.tell(), 130)

    b.close()
    assert_equal(len(b.sink().bytes()), 130)
    _ = a^
    _ = c^


def test_io_sink_buffered_encoders_write_through_the_buffer() raises:
    """`ColumnWriter` and friends take `mut out: List[UInt8]`, so the adapter
    has to hand out a real, appendable list — that is what keeps `format.mojo`
    and `codecs.mojo` unchanged."""
    var b = BufferedSink(MemorySink())
    b.buffer().append(1)
    b.buffer().append(2)
    b.flush()
    b.buffer().append(3)
    b.close()

    ref got = b.sink().bytes()
    assert_equal(len(got), 3)
    assert_equal(got[0], UInt8(1))
    assert_equal(got[1], UInt8(2))
    assert_equal(got[2], UInt8(3))


def test_io_sink_buffered_pad_to_pads_the_output() raises:
    """Padding is to a boundary of the *file*, not of the staging buffer. With
    100 bytes flushed, a further 4 written, an 8-alignment needs 4 more — a
    buffer-relative implementation would add 4 for a different reason and then
    disagree on the next call."""
    var b = BufferedSink(MemorySink())
    var a = _pattern(100, 10)
    b.write(Span(a))
    b.flush()

    var c = _pattern(4, 11)
    b.write(Span(c))
    assert_equal(b.tell(), 104)
    b.pad_to(8)
    assert_equal(b.tell(), 104, "104 is already 8-aligned")

    var d = _pattern(3, 12)
    b.write(Span(d))
    assert_equal(b.tell(), 107)
    b.pad_to(8)
    assert_equal(b.tell(), 112)

    b.close()
    ref got = b.sink().bytes()
    assert_equal(len(got), 112)
    for i in range(107, 112):
        assert_equal(got[i], UInt8(0), "padding must be zero")
    _ = a^
    _ = c^
    _ = d^


def test_io_sink_buffered_flush_of_empty_buffer_is_a_noop() raises:
    var b = BufferedSink(MemorySink())
    b.flush()
    b.flush()
    assert_equal(b.tell(), 0)
    b.close()
    assert_equal(len(b.sink().bytes()), 0)


def test_io_sink_buffered_over_a_file_matches_memory() raises:
    """The adapter is backend-agnostic: the same writes produce the same bytes
    through a file as through memory. This is the property that lets a format
    writer be written once against `ByteSink`."""
    var path = String("/tmp/marrow_io_sink_buffered_file.bin")
    if exists(path):
        remove(path)

    var chunks: List[List[UInt8]] = [
        _pattern(37, 13),
        _pattern(1, 14),
        _pattern(4096, 15),
        _pattern(11, 16),
    ]

    var mem = BufferedSink(MemorySink())
    var fil = BufferedSink(FileSink(path))
    for ref ch in chunks:
        mem.write(Span(ch))
        fil.write(Span(ch))
        # Flush at a different cadence than the writes, so `tell` is exercised
        # with a non-empty staging buffer on one side.
        if len(ch) > 100:
            mem.flush()
            fil.flush()
    assert_equal(mem.tell(), fil.tell())
    mem.close()
    fil.close()

    var from_file = _read_file(path)
    _assert_bytes_eq(Span(from_file), Span(mem.sink().bytes()))
    remove(path)
    _ = chunks^
