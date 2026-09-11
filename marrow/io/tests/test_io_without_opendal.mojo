"""Marrow must work with no OpenDAL library present.

`libopendal_c` is `publish = false` upstream with no conda build, so it can
never be a dependency -- it is opened at runtime if it happens to be there.
That makes "reading a local file still works when it is not" a guarantee, and
guarantees need a test. This one runs in the **default** environment, where the
library is normally absent; the `test_io_opendal*` cases beside this one are the
other half and skip themselves without it.

**`setenv` here is nearly always inert, and that is why nothing below depends
on it.** `MARROW_OPENDAL_LIBRARY` is read exactly once per process, inside the
`_Global` initialiser in `io/opendal.mojo`; one pytest selection is one driver
and one process, so if any earlier case already touched OpenDAL the variable
has been read and poisoning it afterwards changes nothing. Cases that need to
know whether a library is present therefore *ask* -- `_opendal_present()` --
and assert something in either answer, rather than assuming absence and
returning early when the assumption fails. A case that returns early is a PASS
to this harness, so "assume and bail" is indistinguishable from a real pass.
"""

from std.os import getenv, remove, setenv
from std.testing import assert_equal, assert_true

from ...io import BufferSource, DynSink, DynSource
from ...io.opendal import OpenDalStore


def _opendal_present() -> Bool:
    """Whether this process can open `libopendal_c` at all.

    Asked rather than assumed, and asked with `memory` -- the one service in
    every build -- so the probe itself cannot reach the network.
    """
    try:
        _ = OpenDalStore("memory")
        return True
    except:
        return False


def test_io_local_reads_work_without_opendal() raises:
    """A local read must not touch `dlopen` at all.

    Not a proof that nothing loaded -- that would need to observe the loader --
    but the property that matters is behavioural: with the library pointed
    somewhere that cannot exist, local sources still answer.
    """
    _ = setenv(
        "MARROW_OPENDAL_LIBRARY", "/nonexistent/libopendal_c.dylib", True
    )

    var path = String("/tmp/marrow_io_without_opendal.bin")
    var data = List[UInt8](capacity=256)
    for i in range(256):
        data.append(UInt8(i))
    with open(path, "w") as f:
        f.write_bytes(Span(data))

    var mapped = BufferSource(path)
    assert_equal(mapped.size(), 256)
    var got = mapped.read_at(64, 32)
    for i in range(32):
        assert_equal(got[i], data[64 + i])

    var mem = BufferSource(Span(data))
    assert_equal(mem.size(), 256)
    assert_equal(len(mem.read_ranges([(0, 16), (240, 16)])), 2)

    remove(path)
    _ = data^


def test_io_missing_opendal_raises_naming_the_library() raises:
    """A caller who does ask for a remote object gets a failure that says what
    is missing, rather than a mysterious later error.

    Asserts in both directions: with no library the error must name it; with
    one, the same call must succeed. Previously this returned early when a
    library turned out to be present, which made it a pass that had checked
    nothing.
    """
    if _opendal_present():
        # A library is loadable, so the absence path is unreachable in this
        # process. Assert the presence half instead of asserting nothing.
        var op = OpenDalStore("memory")
        op.write("marrow_probe", "ok".as_bytes())
        assert_equal(len(op.read("marrow_probe")), 2)
    else:
        var msg = String()
        try:
            _ = OpenDalStore("memory")
        except e:
            msg = String(e)
        assert_true(
            "opendal_c" in msg,
            String("the error should name the library, got: ", msg),
        )


def test_io_dispatch_local_uri_needs_no_opendal() raises:
    """`DynSource.open`/`DynSink.open` on a local URI must not reach for the library.

    This is the dispatch-level form of the guarantee above, and the reason
    `Uri.is_local()` exists as a branch rather than letting OpenDAL's `fs`
    service handle everything: routing local reads through the binding would
    make them depend on a library that is not a dependency.
    """
    _ = setenv(
        "MARROW_OPENDAL_LIBRARY", "/nonexistent/libopendal_c.dylib", True
    )
    var path = String("/tmp/marrow_io_dispatch_local.bin")

    var payload = List[UInt8](capacity=64)
    for i in range(64):
        payload.append(UInt8(i))

    var sink = DynSink.open(path)
    sink.write(Span(payload))
    sink.close()

    var src = DynSource.open(path)
    assert_equal(src.size(), 64)
    var got = src.read_at(16, 8)
    for i in range(8):
        assert_equal(got[i], payload[16 + i])

    # And a `file://` URI is the same bytes by the same route.
    var via_file = DynSource.open(String("file://", path))
    assert_equal(via_file.size(), 64)

    remove(path)
    _ = payload^


# `test_io_dispatch_remote_uri_without_opendal_raises` used to live here and
# was deleted rather than kept: with a library present it could assert nothing
# and returned early -- a PASS, by this harness -- and the `opendal` CI
# environment always has one, so it never checked anything where it ran.
# Without a library it duplicated the case above. The routing half it claimed
# to cover is in `marrow/utils/tests/test_utils_uri.mojo`, which needs no
# library and no network.
