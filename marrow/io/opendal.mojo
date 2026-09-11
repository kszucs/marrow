"""Object stores, through OpenDAL.

Everything OpenDAL-shaped lives here: the vendored C ABI, the `OpenDalStore`
handle built on it, and the `ByteSource` / `ByteSink` the formats actually use.
One file because the three are a single unit -- the binding exists only to back
the seam, and splitting them meant three modules that could only ever be
imported together.

[Apache OpenDAL](https://opendal.apache.org) speaks ~90 storage services behind
one blocking C ABI. This module is the whole of marrow's binding to it: the
mirror structs, the symbol table, and a small Mojo-facing API
(`OpenDalStore` / `OpenDalWriter`) that `marrow.io.opendal.source` and
`.sink` build a `ByteSource` and a `ByteSink` on.

## The C ABI

**Transcribed from `bindings/c/include/opendal.h` at OpenDAL v0.59.0**, the
latest released tag -- every struct mirrored below is byte-identical between it
and the unreleased 0.59.1 the working tree calls itself. That version is a pin,
not a note. The C ABI is not stable in the way a mirrored
struct needs: `opendal_read_options` has gained fields and `opendal_code` has
gained variants, and either change corrupts a mirror silently rather than
failing to compile. The `comptime assert size_of[...]` guards below are the
second line of defence; the first is building against the pinned tag.

**`libopendal_c` is not a package.** The crate is `publish = false` and there
is no conda build of it, so marrow cannot depend on it: it is opened at runtime
if present, and a box without it must still read local Parquet. That is why the
open failure is *recorded* rather than raised -- nothing fails until a caller
actually asks for a remote object. `pixi run -e opendal build_opendal`
builds one; `$MARROW_OPENDAL_LIBRARY` points at it.

Two ABI contracts are easy to get wrong and neither shows up in a test:

- **`opendal_bytes` is directional.** Coming *out* of a read it is a Rust `Vec`
  that must be released with `opendal_bytes_free` and never `free()`. Going
  *in* to a write it is borrowed -- OpenDAL copies it -- and calling
  `opendal_bytes_free` on one is heap corruption. Nothing in the struct
  distinguishes the two.
- **A Mojo value whose address crosses the boundary must be kept alive past the
  call.** `CString.ptr()` hands back an untracked-origin pointer the compiler
  cannot tie to its owner, so it destroys the buffer at the last *tracked* use
  -- which is the `ptr()` call, not the FFI call. Every borrowing scope here
  ends with `_ = name^`, and a caller that builds a `List[UInt8]` and passes
  `Span(list)` to `write` in the same function needs one too.

A NULL or malformed `path` **panics inside Rust**, and a panic across
`extern "C"` aborts the process -- no Mojo `Error`, nothing catchable. `CString`
therefore has no null state and rejects an embedded NUL.
"""

from std.ffi import _DLHandle, _Global, c_char
from std.memory import ArcPointer, unsafe_memcpy
from std.os import getenv
from std.pathlib import Path
from std.sys import size_of

from ..buffers import Buffer
from ..utils.dylib import CStr, CString, Dylib, c_bytes, c_string
from .core import ByteSink, ByteSource, Fetched, require_range


# ---------------------------------------------------------------------------
# Raw declarations — mirrors of `bindings/c/include/opendal.h`
# ---------------------------------------------------------------------------

comptime Opaque = OpaquePointer[MutUntrackedOrigin]
"""An opaque handle owned by the Rust side (`opendal_operator *` and friends).
"""

comptime ErrorPtr = Optional[Pointer[CError, MutUntrackedOrigin]]
"""`opendal_error *`, where `None` models the NULL that signals success."""

comptime BytePtr = Optional[Pointer[UInt8, MutUntrackedOrigin]]
"""`uint8_t *`, where `None` models NULL. A zero-length `opendal_bytes` must
carry a NULL `data`; the C ABI rejects an empty buffer pointing anywhere else.
"""


@fieldwise_init
struct CBytes(RegisterPassable):
    """Mirrors `opendal_bytes`."""

    var data: BytePtr
    var len: UInt
    var capacity: UInt


@fieldwise_init
struct CError(RegisterPassable):
    """Mirrors `opendal_error`. `message` is **not** NUL-terminated."""

    var code: Int32
    var message: CBytes

    @staticmethod
    def _name(code: Int32) -> String:
        """The `opendal_code` variant name, or `Code(n)` for one this build does
        not know -- which is what a newer library adding a variant looks like.
        """
        var names: List[StaticString] = [
            "Unexpected",
            "Unsupported",
            "ConfigInvalid",
            "NotFound",
            "PermissionDenied",
            "IsADirectory",
            "NotADirectory",
            "AlreadyExists",
            "RateLimited",
            "IsSameFile",
            "ConditionNotMatch",
            "RangeNotSatisfied",
            "Conflict",
        ]
        var i = Int(code)
        if i >= 0 and i < len(names):
            return String(names[i])
        return String("Code(", i, ")")

    @staticmethod
    def raise_if(lib: _DLHandle, err: ErrorPtr) raises:
        """Raise a C error as a Mojo `Error`, freeing it first.

        The message is a non-NUL-terminated `opendal_bytes` living inside the error
        struct, so it is copied *by length* before `opendal_error_free` reclaims
        it. Copy, free, then raise -- in that order, or the error leaks on exactly
        the path that matters.
        """
        if err is None:
            return
        var e = err.value()
        var msg = String()
        var data = e[].message.data
        if data is not None:
            msg = c_string(data.value(), Int(e[].message.len))
        var name = Self._name(e[].code)
        lib.call["opendal_error_free"](e)
        raise Error("opendal: ", name, ": ", msg)


@fieldwise_init
struct CResultHandle(RegisterPassable):
    """Mirrors the `{ handle, error }` results: `opendal_result_operator_new`,
    `opendal_result_stat` and `opendal_result_operator_writer`.

    One mirror for the three because they *are* one layout — an owned pointer
    and a nullable error. Three names for the same two fields would be three
    things to keep in agreement with the header, and three layout asserts
    checking the same 16 bytes.
    """

    var handle: Opaque
    var error: ErrorPtr


@fieldwise_init
struct CResultRead(RegisterPassable):
    """Mirrors `opendal_result_read`."""

    var data: CBytes
    var error: ErrorPtr


@fieldwise_init
struct CResultWriterWrite(RegisterPassable):
    """Mirrors `opendal_result_writer_write`."""

    var size: UInt
    var error: ErrorPtr


# `opendal_read_options` is deliberately **not** mirrored. It is a transparent
# 20-field struct mixing `u64`, `bool`, `const char *` and `uintptr_t` -- the
# most layout-fragile thing in the ABI, and one that has already gained fields.
# Treating it as opaque also sidesteps a live trap: the Rust side reads a range
# only `if offset > 0 || has_length`, so a hand-built, zeroed struct silently
# requests the *whole object* instead of a range. Reaching it only through
# `opendal_read_options_new` + `set_range` makes that unrepresentable.


def _assert_abi_layout():
    """Layout guards for the mirror structs above.

    `ErrorPtr` is an `Optional[Pointer]` inside a `RegisterPassable` struct,
    which assumes `Optional` is niche-optimised to pointer size. If that ever
    stops holding, every struct here silently becomes the wrong size and the
    ABI returns garbage rather than failing -- so it is asserted rather than
    assumed. Called from `_OpenDal.__init__`, because a `comptime assert` has to
    live in a function body.

    **These check size, not field order.** Reordering `opendal_bytes`'s three
    members upstream keeps it 24 bytes and breaks every read; widening
    `opendal_code` to 64 bits keeps `opendal_error` at 32. Size is what
    `size_of` can see, so the pinned tag in `pixi.toml` is the real defence and
    this is the backstop.
    """
    comptime assert size_of[ErrorPtr]() == size_of[Opaque](), (
        "Optional[Pointer] lost its pointer niche; every mirror struct is now"
        " the wrong size"
    )
    comptime assert size_of[CBytes]() == 24, "opendal_bytes layout drifted"
    comptime assert size_of[CError]() == 32, "opendal_error layout drifted"
    comptime assert (
        size_of[CResultRead]() == 32
    ), "opendal_result_read layout drifted"
    comptime assert (
        size_of[CResultHandle]() == 16
    ), "an opendal `{handle, error}` result layout drifted"
    comptime assert (
        size_of[CResultWriterWrite]() == 16
    ), "opendal_result_writer_write layout drifted"


# ---------------------------------------------------------------------------
# Locating and opening the library — once per process, failure recorded
# ---------------------------------------------------------------------------

comptime _OPENDAL_PATHS: List[Path] = [
    "libopendal_c.dylib",
    "libopendal_c.so",
]


def _open_opendal() -> Dylib:
    """Open `libopendal_c` once per process, recording any failure.

    `Dylib` is shared with the Parquet codec loader: both open a genuinely
    optional library from inside a `_Global` initializer that cannot raise, and
    both need the error to surface later at the point of use. The layout guards
    run here because a `comptime assert` needs a function body and this is the
    one place that runs before any C call.
    """
    _assert_abi_layout()
    return Dylib.open["opendal_c"](
        Dylib.candidates["MARROW_OPENDAL_LIBRARY", "OPENDAL_C_LIBRARY"](
            materialize[_OPENDAL_PATHS]()
        )
    )


comptime _OPENDAL = _Global["MARROW_OPENDAL", _open_opendal]
"""Process-lifetime handle. Never `dlclose`d, and here that is correctness
rather than speed: `opendal_operator_new` forces a `LazyLock<tokio::Runtime>`
that spawns worker threads inside this image, and unloading an image with live
threads is a crash."""


def _lib() raises -> _DLHandle:
    """The loaded library, or the recorded open failure.

    Symbols are resolved per call rather than cached in a table:
    `get_function` returns a callable carrying an immutable borrow of the
    handle, which cannot be a struct field, and a `dlsym` is not measurable
    against the round trip it precedes. This is the shape
    `marrow/utils/compression.mojo` already uses.
    """
    return _OPENDAL.get_or_create_ptr()[].get()


# ---------------------------------------------------------------------------
# Marshalling and error translation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The Mojo-facing API
# ---------------------------------------------------------------------------


struct _Handle(Movable):
    """Sole owner of one `opendal_operator *`, freed exactly once."""

    var _op: Opaque

    def __init__(out self, op: Opaque):
        self._op = op

    def __deinit__(deinit self):
        try:
            _lib().call["opendal_operator_free"](self._op)
        except:
            # The library was found once to build this handle, so it is still
            # loaded; there is nowhere for a failure here to go regardless.
            pass


struct OpenDalStore(Copyable, Movable):
    """A storage service reached through OpenDAL. O(1) to copy.

    `ArcPointer`-backed because a dataset is many objects behind one service
    and they must share one operator -- one connection pool, one credential
    chain, one entry into the tokio runtime. It also removes, by construction,
    any chance of freeing the operator while a read is in flight.

    Concurrent reads on one store are sound. Every C entry point takes the
    operator by shared reference, and the chain underneath is
    `blocking::Operator{Handle, AsyncOperator}` where every `AsyncOperator`
    field is an `Arc<dyn T>` with `T: Send + Sync` -- so `blocking::Operator`
    is `Sync`. `OpenDalWriter` is **not**: `opendal_writer_write` takes
    `&mut self`, which is why that type is `Movable` only and must never be
    captured by a `sync_parallelize` closure.

    ```mojo
    var store = OpenDalStore("s3", {"bucket": "example", "region": "us-east-1"})
    var head = store.read_range("data.parquet", 0, 4)
    ```
    """

    var _inner: ArcPointer[_Handle]

    def __init__(
        out self, scheme: StringSlice, options: Dict[String, String] = {}
    ) raises:
        """Build an operator for `scheme`, configured by `options`.

        `scheme` names the service (`memory`, `fs`, `s3`, ...) and must be one
        the loaded `libopendal_c` was built with -- services are cargo
        features, so a stock build has only `memory`. `options` carries
        service-specific keys such as `root` or `bucket`.
        """
        var lib = _lib()
        # Every `CString` is built *before* `options_new`, because `CString`
        # raises on an embedded NUL and a raise between the new and the free
        # leaks the Rust-side options map. Same ordering rule as
        # `_read_range_raw`, for the same reason.
        var cscheme = CString(scheme)
        var keys = List[CString]()
        var values = List[CString]()
        for entry in options.items():
            keys.append(CString(entry.key))
            values.append(CString(entry.value))

        var opts = lib.call["opendal_operator_options_new", Opaque]()
        for i in range(len(keys)):
            lib.call["opendal_operator_options_set"](
                opts, keys[i].ptr(), values[i].ptr()
            )
        var res = lib.call["opendal_operator_new", CResultHandle](
            cscheme.ptr(), opts
        )
        _ = keys^
        _ = values^
        _ = cscheme^
        lib.call["opendal_operator_options_free"](opts)
        # Before the raise: the options are ours on both paths.
        CError.raise_if(lib, res.error)
        self._inner = ArcPointer(_Handle(res.handle))

    def _call_path[
        name: StaticString, R: RegisterPassable
    ](self, lib: _DLHandle, path: StringSlice) raises -> R:
        """Call an `(operator, path)` entry point with the path kept alive.

        The `_ = cpath^` is the whole reason this exists. `CString.ptr()` hands
        back an untracked-origin pointer the compiler cannot tie to its owner,
        so without it the buffer is destroyed at the `ptr()` call rather than
        after the FFI call -- and OpenDAL would read a freed path, or a
        truncated one, and quietly address the wrong object. Written once so it
        cannot be forgotten at one of five call sites.
        """
        var cpath = CString(path)
        var res = lib.call[name, R](self._inner[]._op, cpath.ptr())
        _ = cpath^
        return res^

    def content_length(self, path: StringSlice) raises -> Int:
        """The object's size in bytes.

        `stat` rather than an `exists` + `read`: it answers "is it there" and
        "how big" in one round trip, where asking separately is both an extra
        request and a TOCTOU on a store where the object can vanish between
        them.
        """
        var lib = _lib()
        var res = self._call_path["opendal_operator_stat", CResultHandle](
            lib, path
        )
        CError.raise_if(lib, res.error)
        var n = lib.call["opendal_metadata_content_length", UInt64](res.handle)
        lib.call["opendal_metadata_free"](res.handle)
        return Int(n)

    def read(self, path: StringSlice) raises -> List[UInt8]:
        """The whole object."""
        var lib = _lib()
        var res = self._call_path["opendal_operator_read", CResultRead](
            lib, path
        )
        CError.raise_if(lib, res.error)
        return self._take(lib, res^)

    def _read_range_raw(
        self, lib: _DLHandle, path: StringSlice, offset: Int, length: Int
    ) raises -> CResultRead:
        """Issue one ranged read and hand back the raw result, still owned by
        OpenDAL. **The caller must release it** with `opendal_bytes_free`.

        Split out so `read_range` and `read_range_into` share the call rather
        than one being written in terms of the other -- writing the buffer-
        filling form on top of the allocating form is what made it copy twice.
        """
        var cpath = CString(path)
        var opts = lib.call["opendal_read_options_new", Opaque]()
        lib.call["opendal_read_options_set_range"](
            opts, UInt64(offset), UInt64(length)
        )
        var res = lib.call["opendal_operator_read_with", CResultRead](
            self._inner[]._op, cpath.ptr(), opts
        )
        _ = cpath^
        # Before the raise: the options are ours on both paths.
        lib.call["opendal_read_options_free"](opts)
        CError.raise_if(lib, res.error)
        return res^

    def read_range(
        self, path: StringSlice, offset: Int, length: Int
    ) raises -> List[UInt8]:
        """`length` bytes of `path` starting at `offset`.

        One request, not a whole-object fetch and slice -- this is the whole
        reason a Parquet reader can live on an object store.
        """
        var lib = _lib()
        return self._take(lib, self._read_range_raw(lib, path, offset, length))

    def read_range_into(
        self,
        path: StringSlice,
        offset: Int,
        length: Int,
        dst: Span[mut=True, UInt8, _],
    ) raises -> Int:
        """Read a range straight into `dst`, returning the byte count.

        The form the hot path wants: **one** copy, out of OpenDAL's buffer and
        into memory the caller already owns and aligned. `read_range` would
        cost a second, since the source then has to move that `List` into a
        64-byte-aligned `Buffer`.

        The remaining copy cannot be removed. `Buffer.from_foreign` would have
        to wrap OpenDAL's own allocation, and `Buffer.__init__` asserts a
        64-byte-aligned pointer *and* size, which a Rust `Bytes` satisfies only
        by luck -- an abort under `ASSERT=all` on some inputs and a pass on
        others. The C ABI's ranged read allocates; there is no read-into-caller-
        buffer entry point except the stateful reader, which cannot be shared
        across threads.

        A short return means the object ended early; it is not an error.
        """
        var lib = _lib()
        var res = self._read_range_raw(lib, path, offset, length)
        var n = 0
        var over = False
        if res.data.data is not None:
            n = Int(res.data.len)
            over = n > len(dst)
            if over:
                n = len(dst)
            if n > 0:
                unsafe_memcpy(
                    dest=dst.unsafe_ptr(), src=res.data.data.value(), count=n
                )
        # Before the raise: the bytes are ours either way.
        lib.call["opendal_bytes_free"](Pointer(to=res.data))
        if over:
            # More than was asked for means the service ignored the range --
            # an HTTP endpoint answering 200 with the whole body, say.
            # Truncating would hand back the object's *first* `len(dst)` bytes
            # under the name of some interior range, which decodes to
            # plausible nonsense. Only a short read is legitimate.
            raise Error(
                "opendal: reading '",
                path,
                "' at ",
                offset,
                " returned more than the ",
                len(dst),
                " bytes requested; the service ignored the range",
            )
        return n

    def write(self, path: StringSlice, data: Span[UInt8, _]) raises:
        """Replace `path` with `data`.

        The bytes are borrowed for the call -- OpenDAL copies them -- so this
        must not free them, and the caller must keep them alive across it.
        """
        var lib = _lib()
        var cpath = CString(path)
        var n = len(data)
        # An empty write must carry a NULL pointer; the C ABI rejects a
        # zero-length buffer that points anywhere else.
        var cb = CBytes(None, 0, 0)
        if n > 0:
            cb = CBytes(
                rebind[Pointer[UInt8, MutUntrackedOrigin]](data.unsafe_ptr()),
                UInt(n),
                UInt(n),
            )
        var err = lib.call["opendal_operator_write", ErrorPtr](
            self._inner[]._op, cpath.ptr(), Pointer(to=cb)
        )
        _ = cpath^
        CError.raise_if(lib, err)

    def delete(self, path: StringSlice) raises:
        """Delete `path`. Deleting one that is not there succeeds."""
        var lib = _lib()
        CError.raise_if(
            lib, self._call_path["opendal_operator_delete", ErrorPtr](lib, path)
        )

    def writer(self, path: StringSlice) raises -> OpenDalWriter:
        """Open `path` for streaming writes. Nothing is committed until
        `OpenDalWriter.close()` succeeds."""
        var lib = _lib()
        var res = self._call_path["opendal_operator_writer", CResultHandle](
            lib, path
        )
        CError.raise_if(lib, res.error)
        return OpenDalWriter(res.handle)

    def _take(self, lib: _DLHandle, var res: CResultRead) raises -> List[UInt8]:
        """Copy an `opendal_bytes` out and release it.

        One bulk copy, not a loop: on a megabyte column chunk a per-element
        append is a function call per byte. `opendal_bytes_free` is
        NULL-tolerant, so an empty result needs no special case.
        """
        var out = List[UInt8]()
        if res.data.data is not None:
            out = c_bytes(res.data.data.value(), Int(res.data.len))
        lib.call["opendal_bytes_free"](Pointer(to=res.data))
        return out^


struct OpenDalWriter(ByteSink):
    """A streaming write to a storage service — the object store's `ByteSink`.

    Commit-on-close is not a convention here, it is how object stores work: a
    multipart upload that is never completed does not exist. `FileSink` goes to
    the trouble of a temp path and a rename to give the local backend the same
    contract, so a format writer means one thing wherever it writes.

    `Movable` only, never `Copyable`, and never shared across threads:
    `opendal_writer_write` takes `&mut self`.
    """

    var _writer: Opaque
    var _closed: Bool

    def __init__(out self, writer: Opaque):
        self._writer = writer
        self._closed = False

    def __deinit__(deinit self):
        # Always, closed or not: `opendal_writer_close` commits the object but
        # does **not** release the handle.
        try:
            _lib().call["opendal_writer_free"](self._writer)
        except:
            pass

    def __enter__(var self) -> Self:
        return self^

    def write[o: Origin[mut=False]](mut self, data: Span[UInt8, o]) raises:
        if self._closed:
            raise Error("opendal: writer is closed")
        if len(data) == 0:
            return
        var lib = _lib()
        var cb = CBytes(
            rebind[Pointer[UInt8, MutUntrackedOrigin]](data.unsafe_ptr()),
            UInt(len(data)),
            UInt(len(data)),
        )
        var res = lib.call["opendal_writer_write", CResultWriterWrite](
            self._writer, Pointer(to=cb)
        )
        CError.raise_if(lib, res.error)

    def close(mut self) raises:
        """Commit. Idempotent."""
        if self._closed:
            return
        self._closed = True
        var lib = _lib()
        CError.raise_if(
            lib, lib.call["opendal_writer_close", ErrorPtr](self._writer)
        )


# ---------------------------------------------------------------------------
# The seam: an object store as a ByteSource and a ByteSink
# ---------------------------------------------------------------------------


struct OpenDalSource(ByteSource):
    """One object on a storage service, read by byte range.

    `size()` is resolved once, with `stat`, when the source is constructed --
    the trait requires it non-raising, and a format that seeks needs it before
    it can read anything. A backend that cannot answer it (an HTTP endpoint
    with no `Content-Length`) fails here rather than reporting zero, which
    would surface later as a corrupt-footer error blaming the file.

    **`read_at` accumulates.** The trait hands back a span borrowed from
    storage the source owns, so a source that must fetch has to keep every
    range it returned alive for as long as `self` -- hence an arena behind an
    `ArcPointer`, since `read_at` takes `ref self` rather than `mut self`.
    **The arena only ever grows**, for the life of the source, and `read_at` is
    therefore only safe on one thread -- it mutates that arena without a lock,
    and marrow has none. Both are why `ParquetFile.read` fetches through
    `read_ranges` before it fans out rather than calling `read_at` from its
    workers: a `Fetched` is scoped by its caller and needs no arena at all.

    **So a bulk read must not come through here**, and the only thing keeping
    that true is that every remaining caller reads metadata: a Parquet footer,
    page index or bloom filter, and an IPC frame prefix, each bounded by the
    file's metadata rather than by its data. `_read_message` reads its message
    bodies through `read_ranges` for exactly this reason -- on `read_at` a
    5 GB IPC stream ended up holding 5 GB. A new `read_at` caller whose length
    scales with the data reintroduces that, silently.
    """

    var _store: OpenDalStore
    var _path: String
    var _size: Int
    var _arena: ArcPointer[List[Buffer[mut=False]]]

    def __init__(out self, var store: OpenDalStore, path: String) raises:
        self._size = store.content_length(path)
        self._store = store^
        self._path = path
        self._arena = ArcPointer(List[Buffer[mut=False]]())

    def __init__(out self, *, copy: Self):
        self._store = copy._store.copy()
        self._path = copy._path
        self._size = copy._size
        self._arena = copy._arena

    def size(self) -> Int:
        return self._size

    def _fetch(self, offset: Int, length: Int) raises -> Buffer[mut=False]:
        """One range, in a 64-byte-aligned buffer of marrow's own.

        The alignment is why this cannot wrap OpenDAL's allocation instead:
        `Buffer.__init__` asserts a 64-byte-aligned pointer *and* size, and a
        Rust `Bytes` satisfies neither except by luck -- under `ASSERT=all`
        that is an abort on some inputs and a pass on others. So the bytes are
        read straight into an aligned buffer, which is one copy, not two.
        """
        require_range(offset, length, self._size, "OpenDalSource.read")
        var buf = Buffer.alloc_uninit[DType.uint8](max(length, 1))
        if length > 0:
            var got = self._store.read_range_into(
                self._path,
                offset,
                length,
                # `view()` with no length spans the *padded* allocation, which
                # is up to 63 bytes longer than the request.
                buf.view[DType.uint8](0, length).as_span(),
            )
            if got != length:
                raise Error(
                    "OpenDalSource.read: asked '",
                    self._path,
                    "' for ",
                    length,
                    " bytes at ",
                    offset,
                    ", got ",
                    got,
                )
        return buf^.to_immutable()

    def read_at(
        ref self, offset: Int, length: Int
    ) raises -> Span[UInt8, origin_of(self)]:
        var buf = self._fetch(offset, length)
        self._arena[].append(buf^)
        return rebind[Span[UInt8, origin_of(self)]](
            self._arena[][len(self._arena[]) - 1]
            .view[DType.uint8](0, length)
            .as_span()
        )

    def read_ranges(ref self, ranges: List[Tuple[Int, Int]]) raises -> Fetched:
        """Every range at once, in a batch the caller owns.

        Nothing lands in the arena: the point of the batch is that its storage
        belongs to whoever asked for it, so a decode fan-out can borrow it
        without anyone mutating anything. Requests still go out one at a time;
        issuing them concurrently is a worthwhile change and not this one.
        """
        var out = Fetched(capacity=len(ranges))
        for ref r in ranges:
            out.append(self._fetch(r[0], r[1]), 0, r[1])
        return out^
