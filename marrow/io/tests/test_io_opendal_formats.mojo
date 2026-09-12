"""Parquet and IPC over an object store.

The point of the whole `marrow/io` seam, exercised end to end: the format code
is unchanged and unaware, and only the backend differs. Runs against OpenDAL's
`fs` service, so it needs no network and no credentials -- `fs` is a different
code path from a memory map all the way down, which is what makes it a real
test rather than a rehearsal.
"""

from std.python import Python
from std.os import getenv, remove
from std.os.path import exists
from std.testing import assert_equal, assert_true

from std.memory import ArcPointer

from ...io import ByteSource, Fetched, FileSink, BufferSource
from ...parquet import ParquetFile, read_table, write_table
from ...parquet.codecs import Compression
from ...parquet.writer import FileWriter
from ...ipc import (
    RecordBatchFileReader,
    RecordBatchFileWriter,
    read_ipc_file,
    write_ipc_file,
)
from ...tabular import Table
from ...c_data import CArrowArrayStream
from ..opendal import OpenDalStore, OpenDalWriter, OpenDalSource


def _fs() raises -> OpenDalStore:
    """A fresh `fs` store. `OpenDalStore` owns its operator and so is move-only;
    separate operators over the same directory are equivalent, and building one
    is a local call."""
    return OpenDalStore("fs", {"root": "/tmp"})


def _store() raises -> Optional[OpenDalStore]:
    """An `fs` store rooted at /tmp, or None when this build cannot make one."""
    try:
        return OpenDalStore("fs", {"root": "/tmp"})
    except:
        if getenv("MARROW_REQUIRE_OPENDAL").byte_length() != 0:
            raise Error("MARROW_REQUIRE_OPENDAL is set but 'fs' is missing")
        return None


struct _Counting(ByteSource):
    """An `OpenDalSource` that records how many bytes were asked for.

    Wrapping rather than instrumenting: `read_at`'s span is tied to
    `origin_of(self)`, so a forwarding source has to rebind the origin, which
    is the same one-line widening every real source already does. The counter
    lives behind an `ArcPointer` so the caller keeps a handle to it after the
    source is moved into `ParquetFile`.
    """

    var _inner: OpenDalSource
    var _read: ArcPointer[Int]

    def __init__(
        out self, var inner: OpenDalSource, var counter: ArcPointer[Int]
    ):
        self._inner = inner^
        self._read = counter^

    def size(self) -> Int:
        return self._inner.size()

    def read_at(
        ref self, offset: Int, length: Int
    ) raises -> Span[UInt8, origin_of(self)]:
        self._read[] += length
        return rebind[Span[UInt8, origin_of(self)]](
            self._inner.read_at(offset, length)
        )

    def read_ranges(ref self, ranges: List[Tuple[Int, Int]]) raises -> Fetched:
        for ref r in ranges:
            self._read[] += r[1]
        return self._inner.read_ranges(ranges)


def _sample() raises -> Table:
    var pa = Python.import_module("pyarrow")
    return CArrowArrayStream.from_pycapsule(
        pa.table(
            {
                "i": pa.array([1, 2, 3, 4, 5, 6, 7, 8], type=pa.int64()),
                "s": pa.array(["a", "bb", "ccc", "d", "ee", "f", "gg", "h"]),
                "f": pa.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
            }
        ).__arrow_c_stream__(Python.none())
    ).to_table()


def test_opendal_parquet_read_matches_mmap() raises:
    """A Parquet file decoded through an object store is the same table as one
    decoded through a memory map -- and the decoder cannot tell the difference,
    because it only ever asks for byte ranges."""
    var maybe = _store()
    if not maybe:
        return

    var name = String("marrow_opendal_pq.parquet")
    var path = String("/tmp/", name)
    write_table(_sample(), path, compression=Compression.SNAPPY)

    var via_mmap = read_table(path)
    var pf = ParquetFile(OpenDalSource(_fs(), name))
    var via_store = pf.read()

    assert_equal(via_store.num_rows(), via_mmap.num_rows())
    assert_equal(via_store.num_columns(), via_mmap.num_columns())
    assert_true(via_store == via_mmap)
    remove(path)


def test_opendal_parquet_reads_only_the_ranges_it_needs() raises:
    """A projection must fetch less than reading every column.

    Stated as a *ratio between two reads of the same file*, not as "less than
    the file size", because the absolute number includes a fixed 64 KiB
    speculative footer tail -- on a small object that alone can exceed the
    file, and the first version of this test failed for exactly that reason
    rather than for a real regression. Comparing projection against
    read-everything cancels the footer overhead out.

    `drop` holds distinct strings on purpose: 20,000 copies of one value
    dictionary-encodes to almost nothing, so skipping it would save nothing
    and the assertion would measure noise.
    """
    var maybe = _store()
    if not maybe:
        return

    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var wide = CArrowArrayStream.from_pycapsule(
        pa.table(
            {
                "keep": pa.array(np.arange(20000), type=pa.int64()),
                "drop": pa.array(
                    np.char.add("v", np.arange(20000).astype("U16"))
                ),
            }
        ).__arrow_c_stream__(Python.none())
    ).to_table()

    var name = String("marrow_opendal_pq_wide.parquet")
    var path = String("/tmp/", name)
    write_table(wide, path, compression=Compression.UNCOMPRESSED)
    var total = BufferSource(path).size()

    var projected = ArcPointer(0)
    var pf = ParquetFile(_Counting(OpenDalSource(_fs(), name), projected))
    var cols: List[String] = ["keep"]
    var got = pf.read(columns=cols^)
    assert_equal(got.num_rows(), 20000)
    assert_equal(got.num_columns(), 1)

    var everything = ArcPointer(0)
    var pf_all = ParquetFile(_Counting(OpenDalSource(_fs(), name), everything))
    var all_cols = pf_all.read()
    assert_equal(all_cols.num_columns(), 2)
    remove(path)

    # The assertion this test is named for. Both reads pay the same footer
    # tail and the same page index, so the saving is bounded by the data and
    # cannot approach 100%: measured here it is ~46% (263 KB against 489 KB),
    # with the fixed overhead about 100 KB of each. Three quarters leaves room
    # for that overhead to grow without turning this red, while still failing
    # loudly if a projection ever starts fetching the column it skipped.
    assert_true(
        projected[] * 4 < everything[] * 3,
        String(
            "a one-column projection read ",
            projected[],
            " bytes where reading both read ",
            everything[],
            "; the skipped chunk should not have been fetched",
        ),
    )
    _ = total


def test_opendal_parquet_write_then_read_back() raises:
    """Write a Parquet file *to* an object store and read it back from one.

    `FileWriter` is unchanged and unaware -- it stages bytes and commits them
    to whatever `ByteSink` it was handed.
    """
    var maybe = _store()
    if not maybe:
        return

    var name = String("marrow_opendal_pq_out.parquet")
    var path = String("/tmp/", name)
    if exists(path):
        remove(path)

    var t = _sample()
    var w = FileWriter(_fs().writer(name), Compression.SNAPPY)
    w.write(t, row_group_size=3)
    assert_true(exists(path), "the sink did not commit")

    # Byte-identical to the same table written through a local sink. That is
    # the strong form: it pins the sink as a pure destination change, where
    # comparing decoded tables would also pass if the writer had quietly
    # reshaped something.
    var local = String("/tmp/marrow_opendal_pq_local.parquet")
    var lw = FileWriter(FileSink(local), Compression.SNAPPY)
    lw.write(t, row_group_size=3)

    var a = BufferSource(path)
    var b = BufferSource(local)
    assert_equal(a.size(), b.size())
    var sa = a.read_at(0, a.size())
    var sb = b.read_at(0, b.size())
    for i in range(a.size()):
        assert_equal(sa[i], sb[i])

    # And it reads back, through both backends.
    var back = read_table(path)
    assert_equal(back.num_rows(), 8)
    assert_true(back.combine_chunks() == t.combine_chunks())
    var pf = ParquetFile(OpenDalSource(_fs(), name))
    assert_true(pf.read().combine_chunks() == t.combine_chunks())

    remove(path)
    remove(local)


def test_opendal_ipc_round_trip() raises:
    """The same for Arrow IPC, which shares the seam and nothing else."""
    var maybe = _store()
    if not maybe:
        return

    var name = String("marrow_opendal_ipc.arrow")
    var path = String("/tmp/", name)
    if exists(path):
        remove(path)

    var t = _sample()
    var batch = t.combine_chunks()
    var w = RecordBatchFileWriter(_fs().writer(name), batch.schema)
    w.write_batch(batch)
    w.close()
    assert_true(exists(path), "the sink did not commit")

    var r = RecordBatchFileReader(OpenDalSource(_fs(), name))
    assert_equal(r.num_record_batches(), 1)
    assert_true(r.read_batch(0) == batch)
    remove(path)


def test_opendal_uri_dispatch_round_trip() raises:
    """The whole point, through the public surface: `write_table` /
    `read_table` given a URL rather than a path.

    Exercised over `fs://` because it needs no credentials, but the dispatch
    is the same one `s3://` takes -- `Uri.parse` picks the service, and neither
    the reader nor the writer learns anything about it.
    """
    if not _store():
        return

    var name = String("marrow_opendal_uri_dispatch.parquet")
    var path = String("/tmp/", name)
    if exists(path):
        remove(path)

    # `fs:///tmp/x` -- an authority-less URI whose root defaults to `/`, so it
    # needs no options at all.
    var uri = String("fs://", path)

    var t = _sample()
    write_table(t, uri)
    assert_true(exists(path), "write_table did not publish through the URI")

    var back = read_table(uri)
    assert_true(back.combine_chunks() == t.combine_chunks())

    # A bare path names the same bytes, and takes the local mmap route.
    assert_true(read_table(path).combine_chunks() == t.combine_chunks())
    remove(path)


def test_opendal_uri_dispatch_ipc_round_trip() raises:
    """Same surface, IPC."""
    if not _store():
        return

    var name = String("marrow_opendal_uri_dispatch.arrow")
    var path = String("/tmp/", name)
    if exists(path):
        remove(path)

    var uri = String("fs://", path)

    var batch = _sample().combine_chunks()
    write_ipc_file(uri, [batch.copy()])
    assert_true(exists(path), "write_ipc_file did not publish through the URI")

    var got = read_ipc_file(uri)
    assert_equal(len(got), 1)
    assert_true(got[0] == batch)
    remove(path)


def test_opendal_parquet_parallel_decode_is_sound() raises:
    """A remote read big enough to fan out across threads.

    The regression this exists for: `ParquetFile.read` used to call
    `ByteSource.read_at` from inside its `sync_parallelize` workers.
    `OpenDalSource.read_at` has to retain what it hands back — the trait
    returns a span borrowed from the source — so it appended to an arena
    behind an `ArcPointer`, and N workers appending to one unsynchronised
    `List` is a realloc race with a use-after-free hanging off it. The reader
    now fetches every range through `read_ranges` *before* dispatching, so the
    workers borrow a batch nobody is mutating.

    The fan-out needs `total >= 2` and `>= 4096` rows (`_PARALLEL_MIN_ROWS`),
    which is why the other cases here never reached it: they project one
    column of eight rows and stay single-threaded. This one is four columns
    over three row groups of 5,000 rows.
    """
    var maybe = _store()
    if not maybe:
        return

    var name = String("marrow_opendal_parallel.parquet")
    var path = String("/tmp/", name)
    if exists(path):
        remove(path)

    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var n = 15000
    var wide = CArrowArrayStream.from_pycapsule(
        pa.table(
            {
                "a": pa.array(np.arange(n), type=pa.int64()),
                "b": pa.array(np.arange(n) * 2, type=pa.int64()),
                "c": pa.array(np.full(n, "xyzzy")),
                "d": pa.array(np.arange(n), type=pa.float64()),
            }
        ).__arrow_c_stream__(Python.none())
    ).to_table()

    var w = FileWriter(FileSink(path), Compression.SNAPPY)
    w.write(wide, row_group_size=5000)

    # 3 row groups x 4 leaves = 12 slots, 15,000 rows -> genuinely parallel.
    var pf = ParquetFile(OpenDalSource(_fs(), name))
    var got = pf.read()
    assert_equal(got.num_rows(), n)
    assert_equal(got.num_columns(), 4)

    # And the bytes are right, not merely the shape -- a torn arena would have
    # produced plausible-looking garbage.
    assert_true(got.combine_chunks() == read_table(path).combine_chunks())
    remove(path)
