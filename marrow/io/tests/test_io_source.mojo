"""`ByteSource` and its resident backends.

The batch entry point is what the decode fan-out reads through, so its edges —
empty, overlapping, out-of-order, out-of-bounds — are the ones worth pinning:
`read_at` is exercised incidentally by every Parquet and IPC test in the tree,
`read_ranges` by none of them yet.
"""

from std.testing import assert_equal, assert_true

from ...buffers import Buffer
from ...io import Fetched, BufferSource


def _pattern(n: Int) -> List[UInt8]:
    """`n` bytes with no run of repeats, so a misaligned read cannot pass."""
    var out = List[UInt8](capacity=n)
    for i in range(n):
        out.append(UInt8((i * 7 + 3) % 251))
    return out^


def _write(path: String, data: Span[UInt8, _]) raises:
    with open(path, "w") as f:
        f.write_bytes(data)


def _assert_span_is(got: Span[UInt8, _], want: Span[UInt8, _]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


# --- BufferSource -----------------------------------------------------------


def test_io_source_memory_read_at() raises:
    var data = _pattern(256)
    var src = BufferSource(Span(data))
    assert_equal(src.size(), 256)
    _assert_span_is(src.read_at(64, 32), Span(data)[64:96])
    _assert_span_is(src.read_at(0, 256), Span(data))
    _assert_span_is(src.read_at(255, 1), Span(data)[255:])
    assert_equal(len(src.read_at(10, 0)), 0)
    _ = data^


def test_io_source_memory_read_at_out_of_bounds_raises() raises:
    """A remote source answers a bad range with an error; a map would fault.
    The seam has to behave the same either way."""
    var data = _pattern(64)
    var src = BufferSource(Span(data))
    for bad in [(0, 65), (64, 1), (-1, 4), (0, -1), (63, 2)]:
        var raised = False
        try:
            _ = src.read_at(bad[0], bad[1])
        except:
            raised = True
        assert_true(raised, "expected a raise for that range")
    _ = data^


def test_io_source_memory_read_ranges() raises:
    var data = _pattern(512)
    var src = BufferSource(Span(data))
    # Out of order, overlapping, adjacent, and a zero-length range: the batch
    # must preserve the order it was asked in and not merge anything, because
    # the caller indexes the result positionally.
    var ranges: List[Tuple[Int, Int]] = [
        (400, 16),
        (0, 8),
        (8, 8),
        (4, 8),
        (100, 0),
        (496, 16),
    ]
    var got = src.read_ranges(ranges)
    assert_equal(len(got), 6)
    for i in range(len(ranges)):
        _assert_span_is(
            got.span(i), Span(data)[ranges[i][0] : ranges[i][0] + ranges[i][1]]
        )
    _ = data^


def test_io_source_read_ranges_empty_batch() raises:
    var data = _pattern(32)
    var src = BufferSource(Span(data))
    var got = src.read_ranges(List[Tuple[Int, Int]]())
    assert_equal(len(got), 0)
    _ = data^


def test_io_source_read_ranges_out_of_bounds_raises() raises:
    var data = _pattern(32)
    var src = BufferSource(Span(data))
    var raised = False
    try:
        _ = src.read_ranges([(0, 8), (24, 16)])
    except:
        raised = True
    assert_true(raised)
    _ = data^


def test_io_source_fetched_span_index_out_of_range_raises() raises:
    var data = _pattern(32)
    var src = BufferSource(Span(data))
    var got = src.read_ranges([(0, 8)])
    var raised = False
    try:
        _ = got.span(1)
    except:
        raised = True
    assert_true(raised)
    _ = data^


def test_io_source_fetched_outlives_its_source() raises:
    """The batch owns its bytes, which is what lets the decode workers borrow it
    after the source is gone. If `Fetched` merely borrowed, this would read
    freed memory — under ASAN, loudly."""
    var data = _pattern(128)
    var got: Fetched

    var src = BufferSource(Span(data))
    got = src.read_ranges([(0, 64), (64, 64)])
    _ = src^

    _assert_span_is(got.span(0), Span(data)[:64])
    _assert_span_is(got.span(1), Span(data)[64:])
    _ = data^


# --- BufferSource -------------------------------------------------------------


def test_io_source_mapped_file_read_at() raises:
    var path = "/tmp/marrow_io_source_mapped.bin"
    var data = _pattern(1024)
    _write(path, Span(data))

    var src = BufferSource(path)
    assert_equal(src.size(), 1024)
    _assert_span_is(src.read_at(0, 1024), Span(data))
    _assert_span_is(src.read_at(512, 64), Span(data)[512:576])
    # The last byte: `Buffer`'s size is padded up to a multiple of 64, but the
    # *mapping's* extent is what bounds a file offset.
    _assert_span_is(src.read_at(1023, 1), Span(data)[1023:])
    _ = data^


def test_io_source_mapped_file_read_at_past_end_raises() raises:
    var path = "/tmp/marrow_io_source_mapped_oob.bin"
    var data = _pattern(100)
    _write(path, Span(data))

    var src = BufferSource(path)
    assert_equal(src.size(), 100)
    var raised = False
    try:
        # Within the 64-byte-padded `Buffer`, past the end of the file. Reading
        # it would return real memory holding garbage rather than failing, which
        # is why the bound is the mapped size and not `len(_buf)`.
        _ = src.read_at(96, 32)
    except:
        raised = True
    assert_true(raised)
    _ = data^


def test_io_source_mapped_file_read_ranges_matches_read_at() raises:
    """The differential: a batch says exactly what N single reads say."""
    var path = "/tmp/marrow_io_source_mapped_ranges.bin"
    var data = _pattern(4096)
    _write(path, Span(data))

    var src = BufferSource(path)
    var ranges: List[Tuple[Int, Int]] = [
        (0, 64),
        (4032, 64),
        (1000, 1),
        (2048, 512),
        (7, 13),
    ]
    var got = src.read_ranges(ranges)
    assert_equal(len(got), len(ranges))
    for i in range(len(ranges)):
        _assert_span_is(got.span(i), src.read_at(ranges[i][0], ranges[i][1]))
    _ = data^


def test_io_source_mapped_file_batch_shares_one_mapping() raises:
    """A batch over a memory map must stay zero-copy: N ref-count bumps on one
    allocation, not N copies. `unsafe_ptr` is confined to the buffer layer, so
    this cannot compare addresses — what it can show is that a 1,000-range batch
    over a 4 MiB file is not paying per byte. `bench_io.mojo` measures it."""
    var path = "/tmp/marrow_io_source_mapped_share.bin"
    var data = _pattern(65536)
    _write(path, Span(data))

    var src = BufferSource(path)
    var ranges = List[Tuple[Int, Int]](capacity=1024)
    for i in range(1024):
        ranges.append((i * 64, 64))
    var got = src.read_ranges(ranges)
    assert_equal(len(got), 1024)
    _assert_span_is(got.span(0), Span(data)[:64])
    _assert_span_is(got.span(1023), Span(data)[65472:])
    _ = data^


def test_io_source_memory_source_matches_mapped_file() raises:
    """The two resident backends are interchangeable, which is what makes
    `BufferSource` usable as the in-memory stand-in for a file in tests."""
    var path = "/tmp/marrow_io_source_equivalence.bin"
    var data = _pattern(777)
    _write(path, Span(data))

    var mapped = BufferSource(path)
    var mem = BufferSource(Span(data))
    assert_equal(mapped.size(), mem.size())
    for r in [(0, 777), (100, 200), (776, 1), (0, 0)]:
        _assert_span_is(mapped.read_at(r[0], r[1]), mem.read_at(r[0], r[1]))
    _ = data^
