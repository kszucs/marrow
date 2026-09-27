# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The OpenDAL C ABI binding, exercised against services that need no
credentials: `memory` and `fs`.

Skipped as a whole when `libopendal_c` is absent -- it is not a package marrow
can depend on -- so these cases are guarded by `_available()` rather than by a
pytest marker. The one case that must run either way lives in
`test_io_dispatch.mojo`: a marrow without OpenDAL must still read a local file.
"""

from std.os import getenv
from std.os.path import exists, join
from std.testing import assert_equal, assert_true

from ...execution import ExecContext
from ...utils.testing import ScratchDir
from ..opendal import OpenDalSource, OpenDalStore


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
    """`n` pseudo-random bytes (xorshift32), so a range read from the wrong
    offset matches the right one only by chance: 2^-8L for `L` bytes.

    Not an arithmetic sequence: `(i * 7 + 3) % 251` repeats every 251 bytes, so
    a read displaced by any multiple of 251 returned the expected bytes.
    """
    var out = List[UInt8](capacity=n)
    var state = UInt32(0x9E3779B9)
    for _ in range(n):
        state ^= state << 13
        state ^= state >> 17
        state ^= state << 5
        out.append(UInt8(state >> 24))
    return out^


def _fs_store(root: String) -> Optional[OpenDalStore]:
    """An `fs` store rooted at `root`, or `None` when the library lacks the
    service: `fs` is a cargo feature, and a stock libopendal_c has only
    `memory`."""
    try:
        return OpenDalStore("fs", {"root": root})
    except:
        return None


def _write(dir: String, name: String, data: List[UInt8]) raises:
    with open(join(dir, name), "w") as f:
        f.write_bytes(Span(data))


def _scattered(count: Int) -> List[Tuple[Int, Int]]:
    """`count` ranges over a 64 KiB object, scattered and with a length that
    does not grow with the index, so neither position nor size can stand in
    for the slot number. `37` is invertible mod `244`, so for `count <= 244`
    every length differs -- and every length is at least 8 bytes, which is
    what makes a displaced read a 2^-64 coincidence rather than a likely one.
    """
    var ranges = List[Tuple[Int, Int]](capacity=count)
    for i in range(count):
        ranges.append((i * 613 + 7, (i * 37) % 244 + 8))
    return ranges^


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
    with ScratchDir() as dir:
        var found = _fs_store(dir)
        if not found:
            return
        ref store = found.value()
        var name = String("probe.bin")
        var data = _pattern(4096)
        _write(dir, name, data)

        assert_equal(store.content_length(name), 4096)
        _assert_eq(Span(store.read(name)), Span(data))
        # The differential that catches an off-by-one no single assertion would.
        for r in [(0, 4096), (1, 1), (100, 200), (4095, 1), (2048, 1024)]:
            _assert_eq(
                Span(store.read_range(name, r[0], r[1])),
                Span(data)[r[0] : r[0] + r[1]],
            )
        store.delete(name)
        assert_true(not exists(join(dir, name)))
        _ = data^


def test_opendal_source_read_ranges_many_scattered() raises:
    """`read_ranges` fans its fetches out across workers, so what has to be
    pinned is that range `i` still lands in slot `i`.

    More ranges than any plausible core count, so every worker wraps its stride
    several times -- a bug that filled the slots in completion order rather
    than by index would be invisible with four ranges and four cores.
    """
    with ScratchDir() as dir:
        var found = _fs_store(dir)
        if not found:
            return
        var name = String("ranges.bin")
        var data = _pattern(64 * 1024)
        _write(dir, name, data)

        var src = OpenDalSource(found.take(), name)
        assert_equal(src.size(), 64 * 1024)

        # `parallel(8)` forces a real fan-out on any machine; `serial()` is the
        # one-thread path a plan run under that context takes.
        var ranges = _scattered(97)
        for ctx in [ExecContext.parallel(8), ExecContext.serial()]:
            var got = src.read_ranges(ranges, ctx)
            assert_equal(len(got), len(ranges))
            for i in range(len(ranges)):
                var off, length = ranges[i]
                _assert_eq(got.span(i), Span(data)[off : off + length])
        _ = data^


def test_opendal_source_read_ranges_edge_counts() raises:
    """No ranges, one range, and a zero-length range: the counts at which a
    fan-out has nothing to split, and the length at which a fetch has nothing
    to ask for."""
    with ScratchDir() as dir:
        var found = _fs_store(dir)
        if not found:
            return
        var name = String("edges.bin")
        var data = _pattern(256)
        _write(dir, name, data)
        var src = OpenDalSource(found.take(), name)

        assert_equal(
            len(
                src.read_ranges(
                    List[Tuple[Int, Int]](), ExecContext.parallel(8)
                )
            ),
            0,
        )

        var one = src.read_ranges([(17, 100)], ExecContext.parallel(8))
        assert_equal(len(one), 1)
        _assert_eq(one.span(0), Span(data)[17:117])

        var mixed = src.read_ranges(
            [(0, 0), (256, 0), (255, 1)], ExecContext.parallel(8)
        )
        assert_equal(len(mixed), 3)
        assert_equal(len(mixed.span(0)), 0)
        assert_equal(len(mixed.span(1)), 0)
        _assert_eq(mixed.span(2), Span(data)[255:256])
        _ = data^


def test_opendal_source_read_ranges_rejects_a_bad_range() raises:
    """One range past the end fails the whole batch, on the calling thread: the
    bounds check runs before any worker starts, so a caller bug costs no
    requests and the message is the one check's."""
    with ScratchDir() as dir:
        var found = _fs_store(dir)
        if not found:
            return
        var name = String("bad_range.bin")
        var data = _pattern(256)
        _write(dir, name, data)

        var src = OpenDalSource(found.take(), name)
        var msg = String()
        try:
            _ = src.read_ranges(
                [(0, 16), (128, 16), (250, 16)], ExecContext.parallel(8)
            )
        except e:
            msg = String(e)
        assert_true(
            msg.startswith("OpenDalSource.read_ranges: [250, 266)"),
            String("unexpected message: ", msg),
        )
        _ = data^


def test_opendal_source_read_ranges_raises_the_first_failed_fetch() raises:
    """A fetch that fails *inside* a worker reaches the caller as an error, and
    it is the error of the lowest failing range -- the one a serial loop would
    have stopped at -- not whichever worker happened to report.

    The bounds check cannot produce this: it runs before dispatch. So the
    object shrinks after the source has recorded its size, and every range
    past the new end passes the check and then comes back short. Ranges from
    the first failing one onward are spread round-robin over every worker, so
    several fail at once.
    """
    with ScratchDir() as dir:
        var found = _fs_store(dir)
        if not found:
            return
        var name = String("shrinks.bin")
        var data = _pattern(64 * 1024)
        _write(dir, name, data)
        var src = OpenDalSource(found.take(), name)

        var half = 32 * 1024
        var head = List[UInt8](capacity=half)
        for i in range(half):
            head.append(data[i])
        _write(dir, name, head)

        var ranges = _scattered(97)
        var first = -1
        var failing = 0
        for i in range(len(ranges)):
            if ranges[i][0] + ranges[i][1] > half:
                failing += 1
                if first < 0:
                    first = i
        # The case only discriminates if more than one range fails.
        assert_true(failing > 1)

        # What the first failing range raises on its own, with no fan-out.
        var want = String()
        try:
            _ = src.read_ranges([ranges[first]], ExecContext.serial())
        except e:
            want = String(e)
        assert_true(want.byte_length() > 0, "a short read must raise")
        # And the case only discriminates if another range's message differs.
        var last = String()
        try:
            _ = src.read_ranges([ranges[len(ranges) - 1]], ExecContext.serial())
        except e:
            last = String(e)
        assert_true(last != want, String("indistinct messages: ", last))

        for ctx in [ExecContext.parallel(8), ExecContext.serial()]:
            var got = String()
            try:
                _ = src.read_ranges(ranges, ctx)
            except e:
                got = String(e)
            assert_equal(got, want)
        _ = data^
        _ = head^
