"""The OpenDAL C ABI binding, exercised against services that need no
credentials: `memory` and `fs`.

Skipped as a whole when `libopendal_c` is absent -- it is not a package marrow
can depend on -- so these cases are guarded by `_available()` rather than by a
pytest marker. The one case that must run either way lives in
`test_io_dispatch.mojo`: a marrow without OpenDAL must still read a local file.
"""

from std.os import getenv
from std.os.path import exists
from std.testing import assert_equal, assert_true

from ..opendal import OpenDalStore


def _available() -> Bool:
    try:
        _ = OpenDalStore("memory")
        return True
    except:
        return False


def test_opendal_library_is_present_when_required() raises:
    """Fail loudly when the suite was *supposed* to exercise OpenDAL.

    Every other case here returns early when `libopendal_c` is absent, which is
    right for a developer without it and rots into permanent green everywhere
    else -- a suite that silently tests nothing looks identical to one that
    passes. `MARROW_REQUIRE_OPENDAL=1` is what CI sets, and this is the case
    that makes it mean something.
    """
    if getenv("MARROW_REQUIRE_OPENDAL").byte_length() == 0:
        return
    assert_true(
        _available(),
        "MARROW_REQUIRE_OPENDAL is set but libopendal_c did not load",
    )
    # And the `fs` service specifically, since that is what the file-backed
    # cases need and it is a cargo feature a stock build omits.
    var has_fs = True
    try:
        _ = OpenDalStore("fs", {"root": "/tmp"})
    except:
        has_fs = False
    assert_true(has_fs, "libopendal_c was built without opendal/services-fs")


def _pattern(n: Int) -> List[UInt8]:
    """`n` bytes with no repeated run, so a misaligned range cannot pass."""
    var out = List[UInt8](capacity=n)
    for i in range(n):
        out.append(UInt8((i * 7 + 3) % 251))
    return out^


def _assert_eq(got: Span[UInt8, _], want: Span[UInt8, _]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_opendal_memory_round_trip() raises:
    if not _available():
        return
    var store = OpenDalStore("memory")
    var data = _pattern(1024)
    store.write("/a.bin", Span(data))
    var got = store.read("/a.bin")
    _assert_eq(Span(got), Span(data))
    assert_equal(store.content_length("/a.bin"), 1024)
    _ = data^


def test_opendal_empty_write_and_read() raises:
    """A zero-length `opendal_bytes` must carry a NULL pointer; the C ABI
    rejects an empty buffer pointing anywhere else, so this is the whole test.
    """
    if not _available():
        return
    var store = OpenDalStore("memory")
    var empty = List[UInt8]()
    store.write("/empty.bin", Span(empty))
    assert_equal(store.content_length("/empty.bin"), 0)
    assert_equal(len(store.read("/empty.bin")), 0)
    _ = empty^


def test_opendal_read_range_middle() raises:
    """The single most important case: a ranged read is what lets a Parquet
    reader live on an object store."""
    if not _available():
        return
    var store = OpenDalStore("memory")
    var data = _pattern(256)
    store.write("/r.bin", Span(data))
    _assert_eq(Span(store.read_range("/r.bin", 64, 32)), Span(data)[64:96])
    _assert_eq(Span(store.read_range("/r.bin", 248, 8)), Span(data)[248:])
    _assert_eq(Span(store.read_range("/r.bin", 1, 1)), Span(data)[1:2])
    _ = data^


def test_opendal_read_range_from_zero() raises:
    """Pins one half of the `offset > 0 || has_length` branch on the Rust side:
    a hand-built options struct would return the whole object here, and only
    here. Reaching the options through `read_options_new` is what makes that
    unrepresentable, and this is the case that proves it."""
    if not _available():
        return
    var store = OpenDalStore("memory")
    var data = _pattern(256)
    store.write("/z.bin", Span(data))
    var got = store.read_range("/z.bin", 0, 16)
    assert_equal(len(got), 16, "a zero-offset range returned the whole object")
    _assert_eq(Span(got), Span(data)[:16])
    _ = data^


def test_opendal_read_range_zero_length() raises:
    """The other half of the same branch."""
    if not _available():
        return
    var store = OpenDalStore("memory")
    var data = _pattern(64)
    store.write("/zl.bin", Span(data))
    assert_equal(len(store.read_range("/zl.bin", 0, 0)), 0)
    _ = data^


def test_opendal_read_range_into_a_caller_buffer() raises:
    """What the source uses: fill a buffer the caller already aligned, rather
    than allocate one and copy again."""
    if not _available():
        return
    var store = OpenDalStore("memory")
    var data = _pattern(512)
    store.write("/into.bin", Span(data))

    var dst = List[UInt8](length=64, fill=0)
    var n = store.read_range_into("/into.bin", 128, 64, Span(dst))
    assert_equal(n, 64)
    _assert_eq(Span(dst)[:n], Span(data)[128:192])
    _ = data^


def test_opendal_missing_path_raises_not_found() raises:
    """Error translation end to end: the message is copied out of a
    non-NUL-terminated `opendal_bytes` by length, the error is freed, and the
    code becomes a name."""
    if not _available():
        return
    var store = OpenDalStore("memory")
    var msg = String()
    try:
        _ = store.read("/nope.bin")
    except e:
        msg = String(e)
    assert_true(
        msg.startswith("opendal: NotFound: "),
        String("unexpected message: ", msg),
    )


def test_opendal_unknown_scheme_raises() raises:
    """An unsupported service must raise, not abort the process."""
    if not _available():
        return
    var raised = False
    try:
        _ = OpenDalStore("nosuchservice")
    except:
        raised = True
    assert_true(raised)


def test_opendal_path_with_nul_raises_before_reaching_c() raises:
    """A NUL would truncate the path inside `CStr::from_ptr` and address a
    different object, silently. Rejected on this side instead."""
    if not _available():
        return
    var store = OpenDalStore("memory")
    var raised = False
    try:
        _ = store.content_length(String("/a\0b.bin"))
    except:
        raised = True
    assert_true(raised)


def test_opendal_delete_then_read_raises() raises:
    if not _available():
        return
    var store = OpenDalStore("memory")
    var data = _pattern(8)
    store.write("/d.bin", Span(data))
    store.delete("/d.bin")
    var raised = False
    try:
        _ = store.read("/d.bin")
    except:
        raised = True
    assert_true(raised)
    # Deleting a path that is not there succeeds.
    store.delete("/d.bin")
    _ = data^


def test_opendal_streaming_writer_commits_on_close() raises:
    if not _available():
        return
    var store = OpenDalStore("memory")
    var a = _pattern(100)
    var b = _pattern(50)
    var w = store.writer("/w.bin")
    w.write(Span(a))
    w.write(Span(b))
    w.close()
    w.close()  # idempotent

    var got = store.read("/w.bin")
    assert_equal(len(got), 150)
    _assert_eq(Span(got)[:100], Span(a))
    _assert_eq(Span(got)[100:], Span(b))
    _ = a^
    _ = b^


def test_opendal_fs_reads_a_file_written_out_of_band() raises:
    """The `fs` service over bytes another writer produced -- proof the binding
    reads real files, not only ones it wrote itself."""
    var store: OpenDalStore
    try:
        store = OpenDalStore("fs", {"root": "/tmp"})
    except:
        # `fs` is a cargo feature; a stock libopendal_c has only `memory`.
        return

    var path = String("/tmp/marrow_opendal_fs_probe.bin")
    var data = _pattern(4096)
    with open(path, "w") as f:
        f.write_bytes(Span(data))

    assert_equal(store.content_length("marrow_opendal_fs_probe.bin"), 4096)
    _assert_eq(Span(store.read("marrow_opendal_fs_probe.bin")), Span(data))
    # The differential that catches an off-by-one no single assertion would.
    for r in [(0, 4096), (1, 1), (100, 200), (4095, 1), (2048, 1024)]:
        _assert_eq(
            Span(store.read_range("marrow_opendal_fs_probe.bin", r[0], r[1])),
            Span(data)[r[0] : r[0] + r[1]],
        )
    store.delete("marrow_opendal_fs_probe.bin")
    assert_true(not exists(path))
    _ = data^
