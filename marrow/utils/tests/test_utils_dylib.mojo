"""`LibSpec`, `Dylib.open_spec` and `LibSet` -- declaring optional C libraries.

Exercised against the real `Codecs` set from `compression.mojo` rather than a
copy of its specs: the sonames already exist in two places (there, and
`python/marrow/compile.py` for the wheel), and a third would be a third thing
to keep in step. `libzstd` is a hard dependency of the dev environment, so a
failure here is a real regression rather than a missing optional library.

One finding from building this is worth keeping even though nothing depends on
it: a resolved symbol **can** be stored in a struct field, contrary to what
`io/opendal.mojo` once claimed -- `OwnedDLHandle.get_function` returns a
callable borrowing the handle, but `_DLHandle.get_function[result_type]`
returns a raw C-ABI pointer. Caching symbols that way was implemented,
measured and reverted; `backlog.md` has the numbers.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true

from ..compression import Codecs
from ..dylib import Dylib, LibSpec


def test_utils_dylib_set_opens_its_members() raises:
    """The whole open ritual from one declaration, and a name checked at
    compile time -- `handle["nope"]` is a build error, not a run-time miss."""
    var h = Codecs.handle["zstd"]()
    assert_true(h.call["ZSTD_versionNumber", Int32]() > 0)


def test_utils_dylib_candidates_put_the_env_override_first() raises:
    """An exact path from the spec's `env` beats every soname."""
    comptime SPEC = LibSpec("probe", ["libprobe.dylib"], ["MARROW_PROBE_LIB"])
    _ = setenv("MARROW_PROBE_LIB", "/tmp/explicit-libprobe.dylib", True)
    var paths = Dylib.candidates(materialize[SPEC]())
    assert_true(len(paths) > 1, "expected sonames after the override")
    assert_equal(String(paths[0]), "/tmp/explicit-libprobe.dylib")
    _ = setenv("MARROW_PROBE_LIB", "", True)


def test_utils_dylib_candidates_end_with_the_bare_sonames() raises:
    """The loader's own search path is the last resort, not the first."""
    comptime ZSTD = LibSpec("zstd", ["libzstd.dylib", "libzstd.so.1"], [])
    var paths = Dylib.candidates(materialize[ZSTD]())
    assert_equal(String(paths[len(paths) - 1]), "libzstd.so.1")


def test_utils_dylib_missing_library_raises_naming_it() raises:
    """A library that is not there is a catchable error, not a crash.

    This is the whole reason these are `dlopen`ed rather than linked: a linked
    library is resolved by the loader before `main`, so a missing one could
    not be reported this way at all.
    """
    comptime NOPE = LibSpec(
        "definitely_not_a_real_library",
        ["libdefinitely_not_a_real_library.dylib"],
        [],
    )
    var msg = String()
    try:
        _ = Dylib.open_spec(materialize[NOPE]()).get()
    except e:
        msg = String(e)
    assert_true(
        "definitely_not_a_real_library" in msg,
        String("the error should name the library, got: ", msg),
    )
