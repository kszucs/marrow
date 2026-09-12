"""Talking to a C library: finding it, opening it, and marshalling across.

Shared by every module in the tree that binds one -- the Parquet page codecs,
the OpenDAL object-store binding, and the Arrow C Data Interface. It lives here
rather than in any of them because the pieces are the same pieces: where to
look for a library, how to hand C a string, and how to copy bytes back out.
Each had grown its own copy, and the long caveat on `_exe_dir` below is exactly
the kind of thing that goes out of sync when it does.

What is deliberately *not* here is any one library's conventions -- OpenDAL's
`opendal_code` table and its `opendal_error_free` handshake stay in
`marrow/io/opendal.mojo`, because "NULL means success" is that ABI's idea and
not a general fact about C.
"""

from std.ffi import OwnedDLHandle, _DLHandle, _Global, _try_find_dylib, c_char
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.os import getenv
from std.os.path import dirname
from std.pathlib import Path
from std.sys import argv


def _exe_dir() -> String:
    """Best-effort directory containing the running executable, derived from
    ``argv()[0]``.

    A bare soname (`dlopen("libsnappy.dylib")`) is resolved by the dynamic
    loader's default search paths, never `@loader_path` — so a `marrow
    compile --bundle` directory that ships the codec dylibs next to the
    binary still fails to `dlopen` them unless something tells the loader to
    look there. `argv()[0]` carries that "there": when the process is
    launched as `./qp` or `/abs/path/qp` (as a bundle is documented to be
    run), it has a directory component, and a candidate built from it plus a
    slash resolves as a path, not a bare name, bypassing the search order
    entirely.

    Returns the empty string when `argv()[0]` has no `/` (an unqualified
    `PATH` lookup, e.g. a bare `qp` after `install`) — callers then fall back
    to the original bare-soname candidates below, exactly as before this
    existed.

    **`argv()[0]` is caller-supplied, not the true executable path.** It is
    whatever the launching process put in `argv[0]`, so it can be spoofed and
    it is not the same thing as `/proc/self/exe` or `_NSGetExecutablePath`. In
    particular a `./qp` launch yields `.`, so the codec dylib candidates
    become `./libsnappy.dylib` — i.e. resolved out of the **current working
    directory**, which is not necessarily the directory the binary lives in.
    That is a load-from-cwd surface for anyone who can write to the cwd a
    marrow binary is run from. Recorded rather than fixed: the real fix is to
    ask the OS for the executable path, and this module is on the restricted
    `unsafe_ptr` list (see `CLAUDE.md`), so it is not the place to grow a new
    platform-conditional syscall wrapper.
    """
    var args = argv()
    if len(args) == 0:
        return String()
    return dirname(args[0])


comptime DYLIB_DIR_ENV = "MARROW_DYLIB_DIR"
"""Directory to search before the loader's own paths, set by the caller.

`python/marrow/__init__.py` sets this to the installed package directory,
which is the only way a copy shipped *inside a wheel* is ever found:
`_exe_dir` reads `argv()[0]`, and for a wheel that names the interpreter, not
`site-packages/marrow/`. A bundled AOT binary needs no such help -- there
`argv()[0]` is the binary itself.
"""


def _search_dirs() -> List[String]:
    """Directories to look in before falling back to a bare soname, most
    specific first: `$MARROW_DYLIB_DIR`, then the running executable's own
    directory.

    Both answer the same question -- where has *marrow* put a copy the dynamic
    loader has no reason to know about -- and they are separate only because
    neither can be derived from the other.
    """
    var out = List[String]()
    var configured = getenv(DYLIB_DIR_ENV)
    if configured != "":
        out.append(configured)
    var exe = _exe_dir()
    if exe != "":
        out.append(exe)
    return out^


@fieldwise_init
struct Dylib(Movable):
    """One `dlopen`ed library: the handle, or the error that opening it
    produced.

    **Failure is recorded, not raised.** Every user of this opens inside a
    process global whose initializer cannot raise, and every one of them has a
    library that is genuinely optional — a box missing `libbrotlienc` must
    still get working zstd, and a box missing `libopendal_c` must still read a
    local Parquet file. The error is kept and re-raised from `get()`, at the
    point a caller actually needs that library, with the same text
    `_try_find_dylib` would have raised.

    **Nothing here is ever `dlclose`d.** For the codecs that is speed:
    closing dropped the last reference, dyld unmapped the image, and the next
    read re-mapped and re-bound it at roughly 0.9 ms a cycle -- a 1,000-row
    Snappy `read_table` spent 897 us of its 921 us there. For `libopendal_c`
    it is correctness: `opendal_operator_new` spawns tokio threads inside the
    image, and unloading an image with live threads is a crash. Callers hold
    their handles in a process-global and let them outlive everything.
    """

    var _handle: Optional[OwnedDLHandle]
    var _error: String

    @staticmethod
    def candidates(spec: LibSpec) -> List[Path]:
        """Where to look for a library, most specific first.

        The whole rule in one place: an absolute path from any of the spec's
        `env` names wins outright, then each directory `_search_dirs` names,
        so a copy shipped beside marrow beats whatever the bare soname would
        resolve to on the host, then the bare sonames themselves.
        """
        var out = List[Path]()

        for name in spec.env:
            var override = getenv(name)
            if override.byte_length() > 0:
                out.append(Path(override))
        for dir in _search_dirs():
            for s in spec.sonames:
                out.append(Path(dir + "/" + String(s)))
        for s in spec.sonames:
            out.append(Path(String(s)))
        return out^

    @staticmethod
    def open_spec(spec: LibSpec) -> Dylib:
        """Open the library a spec describes, recording failure.

        Takes the spec by **value**, so a set of libraries shares one
        instantiation of this instead of one each -- which is most of what
        made a global per library cost `query_cli` ~16 KB. The name is folded
        into the error text here because `_try_find_dylib`'s `name` parameter
        is exactly that and nothing more.
        """
        var out = Dylib(None, String())
        try:
            out._handle = _try_find_dylib[""](Dylib.candidates(spec))
        except e:
            out._error = String("failed to open ", spec.name, ": ", e)
        return out^

    def get(self) raises -> _DLHandle:
        """A non-owning borrow of the handle, or the `dlopen` failure this
        library was opened with.

        Borrowing is sound precisely because the owner is a process-wide
        global: it outlives every caller, so the returned handle can never
        dangle.
        """
        if not self._handle:
            raise Error(self._error)
        return self._handle.value().borrow()


# ---------------------------------------------------------------------------
# Declaring a library
# ---------------------------------------------------------------------------


@fieldwise_init
struct LibSpec(Copyable, Movable):
    """Everything marrow knows about one `dlopen`ed library, as data.

    A comptime value, so a `LibSet` can name its members at compile time. Holding `List`s does not disqualify it —
    `marrow/utils/tests/test_utils_dylib.mojo` pins that, and
    `marrow/expr/comptime/tests/test_schema_handle.mojo` pinned the general
    property first.

    `env` names are exact-path overrides and are tried before anything else,
    which is what lets a developer point marrow at a build of their own.
    """

    var name: StaticString
    """Short name, for `_try_find_dylib`'s error text.

    Not an identity on its own -- `Library.KEY` derives its global key from
    this *and* a soname, because a bare name is not unique enough to key a
    registry shared with the stdlib and MAX.
    """

    var sonames: List[StaticString]
    """Candidate filenames, most specific first; both platforms' spellings."""

    var env: List[StaticString]
    """Environment variables holding an absolute path, highest priority."""


struct LibSet[key: StaticString, specs: List[LibSpec]](Movable):
    """A set of optional C libraries behind **one** process-global.

    One declaration per module: the libraries, and the `_Global` key they share.
    Callers then use `Codecs.handle["zstd"]()`, with the name checked against the
    set at compile time. What that replaces is a struct of handles, a named
    `thin` opener, a `_Global` and an accessor, written out per module. A single
    library is the one-element set rather than a second shape for it.

    `key` is spelled by the caller rather than derived, because `_Global` keys
    against a registry shared with the stdlib and MAX: it has to be unique across
    the process, and that is a fact about the program, not about these specs.

        One global for the set rather than one per library, and one instantiation
        of the open path rather than one per spec -- `_open` loops over the specs
        at run time. Six separate globals cost `query_cli` ~16 KB of `__text`, and
        the AOT lane is size-gated.

        Never `dlclose`d; see `Dylib` for why that is speed for the codecs and
        correctness for `libopendal_c`.
    """

    @staticmethod
    def _index_of(members: List[LibSpec], name: StaticString) -> Int:
        """Which slot `name` occupies, or -1.

        Takes the members as an argument rather than reading `Self.specs`: a
        comptime `List[LibSpec]` cannot be materialised into a runtime value,
        and this has to stay non-raising to run at comptime at all -- which is
        what makes a typo a build error rather than a run-time miss.
        """
        for i in range(len(members)):
            if members[i].name == name:
                return i
        return -1

    @staticmethod
    def _open() -> List[Dylib]:
        var out = List[Dylib]()
        for s in materialize[Self.specs]():
            out.append(Dylib.open_spec(s))
        return out^

    comptime _GLOBAL = _Global[Self.key, Self._open]

    @staticmethod
    def handle[name: StaticString]() raises -> _DLHandle:
        """One member's handle, or the `dlopen` failure it recorded.

        `name` is checked against the set at compile time, so a typo is a
        build error naming the set rather than a miss at run time.
        """
        comptime index = Self._index_of(Self.specs, name)
        comptime assert index >= 0, "not in this library set: " + String(name)
        return Self._GLOBAL.get_or_create_ptr()[][index].get()

    @staticmethod
    def handle() raises -> _DLHandle:
        """The handle, for a set with exactly one member.

        Saves naming a library twice when there is only one of it, which is
        otherwise why a module grows a one-line accessor of its own.
        """
        comptime assert (
            len(Self.specs) == 1
        ), "handle() needs a name unless the set has exactly one member"
        return Self._GLOBAL.get_or_create_ptr()[][0].get()

    @staticmethod
    def preload() raises:
        """Open every member now, on the calling thread.

        `_Global` vends its pointer without locking and says nothing about
        racing *creation*, so the first touch must not be several workers at
        once.
        """
        _ = Self._GLOBAL.get_or_create_ptr()


# ---------------------------------------------------------------------------
# Marshalling across the boundary
# ---------------------------------------------------------------------------


comptime CStr = Pointer[c_char, MutUntrackedOrigin]
"""A NUL-terminated `const char *` handed to a C ABI."""


struct CString(Movable):
    """Owns a NUL-terminated copy of a string for the duration of a C call.

    For the common case: the callee reads the string and is done with it before
    returning, so the buffer only has to outlive the call -- but it **must**.
    `ptr()` hands back an untracked-origin pointer the compiler cannot tie to
    this value, so it would otherwise be destroyed at the `ptr()` call itself
    rather than after the FFI call. Every borrowing scope therefore ends with
    `_ = name^`.

    Use `alloc_c_string` instead when the callee *keeps* the pointer.

    An embedded NUL is rejected rather than passed on: C would stop at it, so
    the callee would silently act on a shorter string than the caller wrote --
    a wrong path, a wrong key -- and succeed.
    """

    var _buf: List[UInt8]

    def __init__(out self, s: StringSlice) raises:
        var b = s.as_bytes()
        for i in range(len(b)):
            if b[i] == 0:
                raise Error(
                    (
                        "ffi: string contains a NUL byte, which C would treat"
                        " as its end: '"
                    ),
                    s,
                    "'",
                )
        self._buf = List[UInt8](b)
        self._buf.append(0)

    def ptr(mut self) -> CStr:
        return (
            self._buf.unsafe_ptr()
            .unsafe_bitcast[c_char]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )


def alloc_c_string(s: String) -> CStr:
    """A heap-allocated NUL-terminated copy. **The caller owns it** and must
    free it.

    The counterpart to `CString` for the case where the callee keeps the
    pointer -- an Arrow `CArrowSchema.format`, say, which lives until its
    release callback runs. `String.unsafe_ptr()` cannot be handed over
    directly: it is not guaranteed NUL-terminated, because small-string
    storage leaves the bytes past `len(s)` uninitialised.
    """
    var n = s.byte_length()
    var buf = unsafe_alloc[UInt8](n + 1)
    unsafe_memcpy(dest=buf, src=s.unsafe_ptr(), count=n)
    buf[unsafe_offset=n] = 0
    return CStr(unsafe_from_address=Int(buf))


def c_bytes(ptr: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> List[UInt8]:
    """Copy `n` bytes out of a C buffer into an owned `List`.

    One `memcpy`. Not `List(Span(...))`: that goes through `List`'s `Iterable`
    constructor, which is an `append` per element -- the same per-byte call
    this exists to avoid, just one level down. On a megabyte column chunk that
    is the whole cost.
    """
    if n <= 0:
        return List[UInt8]()
    var out = List[UInt8](unsafe_uninit_length=n)
    unsafe_memcpy(dest=out.unsafe_ptr(), src=ptr, count=n)
    return out^


def c_string(ptr: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> String:
    """Copy `n` bytes out of a C buffer into an owned `String`.

    Non-validating: a C library's error message is not guaranteed UTF-8, and
    translating an error must not itself raise.
    """
    if n <= 0:
        return String()
    return String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=n)))
