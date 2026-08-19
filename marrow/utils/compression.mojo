"""System-library loading for the block compression codecs.

The codecs are not reimplemented; the standard C libraries (`libzstd`,
`libsnappy`, `liblz4`, `libz`, `libbrotli`) are `dlopen`-ed at runtime and their
block APIs called directly — the same approach arrow-rs and duckdb take, just
without a link-time dependency. `CompressionLibs` is the primitive block calls
plus the per-call scratch they need; the `dlopen` handles themselves live in one
process-wide `_CodecHandles`.

**Nothing here is Parquet-specific**, which is why it lives in `marrow.utils`
rather than in `marrow.parquet` where it started (as a second module named
`utils`). The format-specific half — the Parquet `CompressionCodec` codes, the
legacy Hadoop LZ4 frame tolerance, the scratch slack the bit-unpackers need — is
`Compression` in `marrow.parquet.codecs`, which dispatches onto this.

The other consumer is Arrow IPC, which currently *refuses* compressed bodies
(`ipc.mojo`, "reading compressed IPC bodies (LZ4_FRAME / ZSTD) is not
supported"). These bindings are what that needs.
"""

from std.ffi import OwnedDLHandle, _DLHandle, _Global, _try_find_dylib
from std.pathlib import Path
from std.memory import unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.sys import argv

comptime _ZSTD_PATHS: List[Path] = [
    "libzstd.dylib",
    "libzstd.1.dylib",
    "libzstd.so",
    "libzstd.so.1",
]
comptime _SNAPPY_PATHS: List[Path] = [
    "libsnappy.dylib",
    "libsnappy.so",
    "libsnappy.so.1",
]
comptime _LZ4_PATHS: List[Path] = [
    "liblz4.dylib",
    "liblz4.so",
    "liblz4.so.1",
]
comptime _ZLIB_PATHS: List[Path] = [
    "libz.dylib",
    "libz.1.dylib",
    "libz.so",
    "libz.so.1",
]
comptime _BROTLI_ENC_PATHS: List[Path] = [
    "libbrotlienc.dylib",
    "libbrotlienc.so",
    "libbrotlienc.so.1",
]
comptime _BROTLI_DEC_PATHS: List[Path] = [
    "libbrotlidec.dylib",
    "libbrotlidec.so",
    "libbrotlidec.so.1",
]


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
    """
    var args = argv()
    if len(args) == 0:
        return String()
    var exe = String(args[0])
    var parts = exe.split("/")
    if len(parts) <= 1:
        return String()
    var out = String()
    for i in range(len(parts) - 1):
        if i != 0:
            out += "/"
        out += parts[i]
    return out


def _with_exe_dir(dir: String, paths: List[Path]) -> List[Path]:
    """`paths`, prefixed with an executable-relative candidate (under `dir`,
    the result of `_exe_dir()`) for each entry when `dir` is non-empty.

    Tried first, so a bundle's own copy of a codec library wins over
    whatever the bare soname would otherwise resolve to on the host.
    """
    var out = List[Path]()
    if dir.byte_length() != 0:
        for p in paths:
            out.append(Path(dir + "/" + String(p)))
    for p in paths:
        out.append(p)
    return out^


@fieldwise_init
struct _Library(Movable):
    """One compression library: the `dlopen` handle, or the error that opening
    it produced.

    Failure is *recorded*, not raised. The whole set is opened together inside
    a process global whose initializer cannot raise, so a box that is missing
    `libbrotlienc` must still get working zstd — the error is kept and re-raised
    from `get()`, at the point a page actually needs that codec, with the same
    text `_try_find_dylib` would have raised."""

    var _handle: Optional[OwnedDLHandle]
    var _error: String

    @staticmethod
    def open[name: StaticString](paths: List[Path]) -> _Library:
        var out = _Library(None, String())
        try:
            out._handle = _try_find_dylib[name](paths)
        except e:
            out._error = String(e)
        return out^

    def get(self) raises -> _DLHandle:
        """A non-owning borrow of the handle, or the `dlopen` failure this
        library was opened with.

        Borrowing is sound precisely because the owner is the process-wide
        `_CodecHandles`: it outlives every caller, so the returned handle can
        never dangle."""
        if not self._handle:
            raise Error(self._error)
        return self._handle.value().borrow()


struct _CodecHandles(Movable):
    """Every compression library the codecs can use, opened once per process.

    Written exactly once — inside the `_Global` initializer below — and only
    read afterwards, which is what makes it safe to share across the workers a
    Parquet read dispatches. `OwnedDLHandle.call` takes `self` immutably and
    resolves the symbol with `dlsym`, itself thread-safe, so a concurrent
    decompress is a concurrent *read* of this struct and nothing more."""

    var zstd: _Library
    var snappy: _Library
    var lz4: _Library
    var zlib: _Library
    var brotli_enc: _Library
    var brotli_dec: _Library

    def __init__(out self):
        var exe_dir = _exe_dir()  # resolved once, shared by every candidate list
        self.zstd = _Library.open["zstd"](
            _with_exe_dir(exe_dir, materialize[_ZSTD_PATHS]())
        )
        self.snappy = _Library.open["snappy"](
            _with_exe_dir(exe_dir, materialize[_SNAPPY_PATHS]())
        )
        self.lz4 = _Library.open["lz4"](
            _with_exe_dir(exe_dir, materialize[_LZ4_PATHS]())
        )
        self.zlib = _Library.open["z"](
            _with_exe_dir(exe_dir, materialize[_ZLIB_PATHS]())
        )
        self.brotli_enc = _Library.open["brotlienc"](
            _with_exe_dir(exe_dir, materialize[_BROTLI_ENC_PATHS]())
        )
        self.brotli_dec = _Library.open["brotlidec"](
            _with_exe_dir(exe_dir, materialize[_BROTLI_DEC_PATHS]())
        )


def _open_codec_handles() -> _CodecHandles:
    return _CodecHandles()


comptime _CODEC_HANDLES = _Global["MARROW_CODEC_HANDLES", _open_codec_handles]
"""The process-wide handle set: allocated by the runtime on first
`get_or_create_ptr()`, and never `dlclose`d before process teardown.

This used to be per read/write, and it dominated any workload that opens the
same file repeatedly. Nothing else in a marrow-only process holds these
libraries, so the matching `dlclose` dropped the last reference and dyld really
unmapped the image; the next read re-mapped, re-bound and re-initialised it, at
roughly 0.9 ms a cycle. A 1,000-row Snappy `read_table` spent 897 us of its
921 us there. Process-lifetime handles are what every library that `dlopen`s its
optional dependencies does, and there is nothing to reclaim: six images that
outlive every reader and writer by construction."""


struct CompressionLibs(Movable):
    """The primitive block calls each codec needs, plus the per-call scratch
    they write through. `Compression` dispatches into these; each fills exactly
    `out_size` bytes at `dst` (decompress) or returns the codec's output
    (compress).

    The `dlopen` handles are **not** here — they are the process-wide
    `_CodecHandles` above, shared by every instance. What an instance owns is
    the reused size out-param snappy needs, which is not safe to share, so a
    Parquet read still holds one of these per worker."""

    var _handles: Optional[Pointer[_CodecHandles, MutUntrackedOrigin]]
    var _sz: List[UInt]  # reusable size out-param for snappy

    def __init__(out self):
        self._handles = None
        self._sz = [UInt(0)]

    def _libs(mut self) raises -> Pointer[_CodecHandles, MutUntrackedOrigin]:
        """The process-wide handles, resolved on first codec use and cached.

        Resolved here rather than in `__init__` because every Parquet read
        constructs a `CompressionLibs` whether or not its pages are compressed:
        a program that only ever reads uncompressed data must still `dlopen`
        nothing."""
        if not self._handles:
            self._handles = _CODEC_HANDLES.get_or_create_ptr()
        return self._handles.value()

    @staticmethod
    def preload() raises:
        """Open the handle set now, on the calling thread.

        `_Global` vends its pointer without locking and says nothing about
        racing *creation*, so the first touch must not be several workers at
        once. `ParquetFile.read` calls this before dispatching whenever the
        chunks it is about to decode are compressed; after it returns, every
        worker's `_libs()` is a pure read."""
        _ = _CODEC_HANDLES.get_or_create_ptr()

    # --- decompress: write exactly `out_size` bytes to `dst` ---

    def zstd_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: Pointer[UInt8, _],
        out_size: Int,
    ) raises:
        var libs = self._libs()
        var n = (
            libs[]
            .zstd.get()
            .call["ZSTD_decompress", Int](
                dst, out_size, src.unsafe_ptr(), len(src)
            )
        )
        if n != out_size:
            raise Error("zstd: decompressed size mismatch")

    def snappy_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: Pointer[UInt8, _],
        out_size: Int,
    ) raises:
        var libs = self._libs()
        self._sz[0] = UInt(out_size)
        var status = (
            libs[]
            .snappy.get()
            .call["snappy_uncompress", Int32](
                src.unsafe_ptr(), len(src), dst, self._sz.unsafe_ptr()
            )
        )
        if status != 0 or Int(self._sz[0]) != out_size:
            raise Error("snappy: decompress failed")

    def lz4_raw_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: Pointer[UInt8, _],
        out_size: Int,
    ) raises:
        var libs = self._libs()
        var n = (
            libs[]
            .lz4.get()
            .call["LZ4_decompress_safe", Int32](
                src.unsafe_ptr(), dst, Int32(len(src)), Int32(out_size)
            )
        )
        if Int(n) != out_size:
            raise Error("lz4: decompressed size mismatch")

    def gzip_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: Pointer[UInt8, _],
        out_size: Int,
    ) raises:
        var libs = self._libs()
        var z = libs[].zlib.get()
        # z_stream is 112 bytes on LP64; drive it directly. Fields we set:
        # next_in @0, avail_in @8, next_out @24, avail_out @32; total_out @40.
        var strm = unsafe_alloc[UInt64](16)
        var sp = strm.unsafe_bitcast[UInt8]()
        unsafe_memset_zero(sp, 128)
        strm[unsafe_offset=0] = UInt64(Int(src.unsafe_ptr()))
        (sp.unsafe_offset(8)).unsafe_bitcast[UInt32]()[
            unsafe_offset=0
        ] = UInt32(len(src))
        strm[unsafe_offset=3] = UInt64(Int(dst))
        (sp.unsafe_offset(32)).unsafe_bitcast[UInt32]()[
            unsafe_offset=0
        ] = UInt32(out_size)

        var version = z.call[
            "zlibVersion", Pointer[UInt8, MutUntrackedOrigin]
        ]()
        # windowBits 31 = 15 | 16 → gzip; stream_size = sizeof(z_stream) = 112
        var rc = z.call["inflateInit2_", Int32](
            sp, Int32(31), version, Int32(112)
        )
        if Int(rc) != 0:
            strm.unsafe_free()
            raise Error("gzip: inflateInit2 failed")
        var st = z.call["inflate", Int32](sp, Int32(4))  # Z_FINISH
        var produced = Int(strm[unsafe_offset=5])  # total_out
        _ = z.call["inflateEnd", Int32](sp)
        strm.unsafe_free()
        if Int(st) != 1:  # Z_STREAM_END
            raise Error("gzip: inflate failed")
        if produced != out_size:
            raise Error("gzip: decompressed size mismatch")

    def brotli_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: Pointer[UInt8, _],
        out_size: Int,
    ) raises:
        var libs = self._libs()
        var sz = unsafe_alloc[UInt](1)
        sz[unsafe_offset=0] = UInt(out_size)
        # BrotliDecoderResult BrotliDecoderDecompress(size_t encoded_size,
        #   const uint8_t* encoded, size_t* decoded_size, uint8_t* decoded);
        # returns BROTLI_DECODER_RESULT_SUCCESS == 1.
        var rc = (
            libs[]
            .brotli_dec.get()
            .call["BrotliDecoderDecompress", Int32](
                len(src), src.unsafe_ptr(), sz, dst
            )
        )
        var produced = Int(sz[unsafe_offset=0])
        sz.unsafe_free()
        if Int(rc) != 1 or produced != out_size:
            raise Error("brotli: decompress failed")

    # --- compress: return the codec's output bytes ---

    @staticmethod
    def _take(dst: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> List[UInt8]:
        """Copy `n` bytes out of a freshly-`alloc`'d compression scratch buffer,
        free the buffer, and return an owned List — the shared tail of every
        `*_compress` method."""
        var out = List[UInt8](Span(unsafe_ptr=dst, length=n))
        dst.unsafe_free()
        return out^

    def zstd_compress(mut self, src: Span[UInt8, _]) raises -> List[UInt8]:
        var libs = self._libs()
        var z = libs[].zstd.get()
        var bound = z.call["ZSTD_compressBound", Int](len(src))
        var dst = unsafe_alloc[UInt8](bound)
        var n = z.call["ZSTD_compress", Int](
            dst, bound, src.unsafe_ptr(), len(src), Int32(1)
        )
        return Self._take(dst, n)

    def snappy_compress(mut self, src: Span[UInt8, _]) raises -> List[UInt8]:
        var libs = self._libs()
        var s = libs[].snappy.get()
        var bound = s.call["snappy_max_compressed_length", Int](len(src))
        var dst = unsafe_alloc[UInt8](bound)
        var sz = unsafe_alloc[UInt](1)
        sz[unsafe_offset=0] = UInt(bound)
        _ = s.call["snappy_compress", Int32](
            src.unsafe_ptr(), len(src), dst, sz
        )
        var produced = Int(sz[unsafe_offset=0])
        sz.unsafe_free()
        return Self._take(dst, produced)

    def lz4_compress(mut self, src: Span[UInt8, _]) raises -> List[UInt8]:
        var libs = self._libs()
        var l = libs[].lz4.get()
        var bound = Int(l.call["LZ4_compressBound", Int32](Int32(len(src))))
        var dst = unsafe_alloc[UInt8](bound)
        var n = l.call["LZ4_compress_default", Int32](
            src.unsafe_ptr(), dst, Int32(len(src)), Int32(bound)
        )
        if n == 0:
            dst.unsafe_free()
            raise Error("lz4: compression failed")
        return Self._take(dst, Int(n))

    def gzip_compress(mut self, src: Span[UInt8, _]) raises -> List[UInt8]:
        var libs = self._libs()
        var z = libs[].zlib.get()
        # gzip worst-case: deflate expansion (~len/1000 + 12) plus the 18-byte
        # gzip header/trailer; pad generously.
        var bound = len(src) + len(src) // 1000 + 128
        var dst = unsafe_alloc[UInt8](bound)
        # z_stream is 112 bytes on LP64; same field layout as gzip_decompress.
        var strm = unsafe_alloc[UInt64](16)
        var sp = strm.unsafe_bitcast[UInt8]()
        unsafe_memset_zero(sp, 128)
        strm[unsafe_offset=0] = UInt64(Int(src.unsafe_ptr()))  # next_in @0
        (sp.unsafe_offset(8)).unsafe_bitcast[UInt32]()[
            unsafe_offset=0
        ] = UInt32(
            len(src)
        )  # avail_in @8
        strm[unsafe_offset=3] = UInt64(Int(dst))  # next_out @24
        (sp.unsafe_offset(32)).unsafe_bitcast[UInt32]()[
            unsafe_offset=0
        ] = UInt32(
            bound
        )  # avail_out @32

        var version = z.call[
            "zlibVersion", Pointer[UInt8, MutUntrackedOrigin]
        ]()
        # level 6, method Z_DEFLATED(8), windowBits 31 = gzip, memLevel 8,
        # strategy Z_DEFAULT_STRATEGY(0), stream_size = sizeof(z_stream) = 112.
        var rc = z.call["deflateInit2_", Int32](
            sp,
            Int32(6),
            Int32(8),
            Int32(31),
            Int32(8),
            Int32(0),
            version,
            Int32(112),
        )
        if Int(rc) != 0:
            strm.unsafe_free()
            dst.unsafe_free()
            raise Error("gzip: deflateInit2 failed")
        var st = z.call["deflate", Int32](sp, Int32(4))  # Z_FINISH
        var produced = Int(strm[unsafe_offset=5])  # total_out @40
        _ = z.call["deflateEnd", Int32](sp)
        strm.unsafe_free()
        if Int(st) != 1:  # Z_STREAM_END
            dst.unsafe_free()
            raise Error("gzip: deflate failed")
        return Self._take(dst, produced)

    def brotli_compress(mut self, src: Span[UInt8, _]) raises -> List[UInt8]:
        var libs = self._libs()
        var e = libs[].brotli_enc.get()
        var bound = Int(
            e.call["BrotliEncoderMaxCompressedSize", UInt](UInt(len(src)))
        )
        if bound == 0:
            bound = len(src) + len(src) // 2 + 512
        var dst = unsafe_alloc[UInt8](bound)
        var sz = unsafe_alloc[UInt](1)
        sz[unsafe_offset=0] = UInt(bound)
        # BROTLI_BOOL BrotliEncoderCompress(int quality, int lgwin,
        #   BrotliEncoderMode mode, size_t input_size, const uint8_t* input,
        #   size_t* encoded_size, uint8_t* encoded); quality 11, lgwin 22,
        #   mode 0 (GENERIC); returns BROTLI_TRUE == 1.
        var rc = e.call["BrotliEncoderCompress", Int32](
            Int32(11),
            Int32(22),
            Int32(0),
            UInt(len(src)),
            src.unsafe_ptr(),
            sz,
            dst,
        )
        var produced = Int(sz[unsafe_offset=0])
        sz.unsafe_free()
        if Int(rc) != 1:
            dst.unsafe_free()
            raise Error("brotli: compression failed")
        return Self._take(dst, produced)
