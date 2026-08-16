"""System-library loading for the block compression codecs.

The codecs are not reimplemented; the standard C libraries (`libzstd`,
`libsnappy`, `liblz4`, `libz`, `libbrotli`) are `dlopen`-ed at runtime and their
block APIs called directly — the same approach arrow-rs and duckdb take, just
without a link-time dependency. `CompressionLibs` is the lazily-opened,
per-read/write handle pool plus the primitive block calls.

**Nothing here is Parquet-specific**, which is why it lives in `marrow.utils`
rather than in `marrow.parquet` where it started (as a second module named
`utils`). The format-specific half — the Parquet `CompressionCodec` codes, the
legacy Hadoop LZ4 frame tolerance, the scratch slack the bit-unpackers need — is
`Compression` in `marrow.parquet.codecs`, which dispatches onto this.

The other consumer is Arrow IPC, which currently *refuses* compressed bodies
(`ipc.mojo`, "reading compressed IPC bodies (LZ4_FRAME / ZSTD) is not
supported"). These bindings are what that needs.
"""

from std.ffi import OwnedDLHandle, _try_find_dylib
from std.pathlib import Path
from std.memory import unsafe_memset_zero
from std.memory.alloc import unsafe_alloc

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


struct CompressionLibs(Movable):
    """Lazily-opened handles to the compression libraries, plus the primitive
    block calls each codec needs, cached per read/write. Held by one worker at a
    time — the `dlopen` handles and the reused size cell are not thread-safe to
    share. `Compression` dispatches into these; each fills exactly `out_size`
    bytes at `dst` (decompress) or returns the codec's output (compress).
    """

    var _zstd: Optional[OwnedDLHandle]
    var _snappy: Optional[OwnedDLHandle]
    var _lz4: Optional[OwnedDLHandle]
    var _zlib: Optional[OwnedDLHandle]
    var _brotli_enc: Optional[OwnedDLHandle]
    var _brotli_dec: Optional[OwnedDLHandle]
    var _sz: List[UInt]  # reusable size out-param for snappy

    def __init__(out self):
        self._zstd = None
        self._snappy = None
        self._lz4 = None
        self._zlib = None
        self._brotli_enc = None
        self._brotli_dec = None
        self._sz = [UInt(0)]

    def _ensure_zstd(mut self) raises:
        if not self._zstd:
            self._zstd = _try_find_dylib["zstd"](materialize[_ZSTD_PATHS]())

    def _ensure_snappy(mut self) raises:
        if not self._snappy:
            self._snappy = _try_find_dylib["snappy"](
                materialize[_SNAPPY_PATHS]()
            )

    def _ensure_lz4(mut self) raises:
        if not self._lz4:
            self._lz4 = _try_find_dylib["lz4"](materialize[_LZ4_PATHS]())

    def _ensure_zlib(mut self) raises:
        if not self._zlib:
            self._zlib = _try_find_dylib["z"](materialize[_ZLIB_PATHS]())

    def _ensure_brotli_enc(mut self) raises:
        if not self._brotli_enc:
            self._brotli_enc = _try_find_dylib["brotlienc"](
                materialize[_BROTLI_ENC_PATHS]()
            )

    def _ensure_brotli_dec(mut self) raises:
        if not self._brotli_dec:
            self._brotli_dec = _try_find_dylib["brotlidec"](
                materialize[_BROTLI_DEC_PATHS]()
            )

    # --- decompress: write exactly `out_size` bytes to `dst` ---

    def zstd_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: Pointer[UInt8, _],
        out_size: Int,
    ) raises:
        self._ensure_zstd()
        var n = self._zstd.value().call["ZSTD_decompress", Int](
            dst, out_size, src.unsafe_ptr(), len(src)
        )
        if n != out_size:
            raise Error("zstd: decompressed size mismatch")

    def snappy_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: Pointer[UInt8, _],
        out_size: Int,
    ) raises:
        self._ensure_snappy()
        self._sz[0] = UInt(out_size)
        var status = self._snappy.value().call["snappy_uncompress", Int32](
            src.unsafe_ptr(), len(src), dst, self._sz.unsafe_ptr()
        )
        if status != 0 or Int(self._sz[0]) != out_size:
            raise Error("snappy: decompress failed")

    def lz4_raw_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: Pointer[UInt8, _],
        out_size: Int,
    ) raises:
        self._ensure_lz4()
        var n = self._lz4.value().call["LZ4_decompress_safe", Int32](
            src.unsafe_ptr(), dst, Int32(len(src)), Int32(out_size)
        )
        if Int(n) != out_size:
            raise Error("lz4: decompressed size mismatch")

    def gzip_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: Pointer[UInt8, _],
        out_size: Int,
    ) raises:
        self._ensure_zlib()
        ref z = self._zlib.value()
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
        self._ensure_brotli_dec()
        var sz = unsafe_alloc[UInt](1)
        sz[unsafe_offset=0] = UInt(out_size)
        # BrotliDecoderResult BrotliDecoderDecompress(size_t encoded_size,
        #   const uint8_t* encoded, size_t* decoded_size, uint8_t* decoded);
        # returns BROTLI_DECODER_RESULT_SUCCESS == 1.
        var rc = self._brotli_dec.value().call[
            "BrotliDecoderDecompress", Int32
        ](len(src), src.unsafe_ptr(), sz, dst)
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
        self._ensure_zstd()
        var bound = self._zstd.value().call["ZSTD_compressBound", Int](len(src))
        var dst = unsafe_alloc[UInt8](bound)
        var n = self._zstd.value().call["ZSTD_compress", Int](
            dst, bound, src.unsafe_ptr(), len(src), Int32(1)
        )
        return Self._take(dst, n)

    def snappy_compress(mut self, src: Span[UInt8, _]) raises -> List[UInt8]:
        self._ensure_snappy()
        var bound = self._snappy.value().call[
            "snappy_max_compressed_length", Int
        ](len(src))
        var dst = unsafe_alloc[UInt8](bound)
        var sz = unsafe_alloc[UInt](1)
        sz[unsafe_offset=0] = UInt(bound)
        _ = self._snappy.value().call["snappy_compress", Int32](
            src.unsafe_ptr(), len(src), dst, sz
        )
        var produced = Int(sz[unsafe_offset=0])
        sz.unsafe_free()
        return Self._take(dst, produced)

    def lz4_compress(mut self, src: Span[UInt8, _]) raises -> List[UInt8]:
        self._ensure_lz4()
        var bound = Int(
            self._lz4.value().call["LZ4_compressBound", Int32](Int32(len(src)))
        )
        var dst = unsafe_alloc[UInt8](bound)
        var n = self._lz4.value().call["LZ4_compress_default", Int32](
            src.unsafe_ptr(), dst, Int32(len(src)), Int32(bound)
        )
        if n == 0:
            dst.unsafe_free()
            raise Error("lz4: compression failed")
        return Self._take(dst, Int(n))

    def gzip_compress(mut self, src: Span[UInt8, _]) raises -> List[UInt8]:
        self._ensure_zlib()
        ref z = self._zlib.value()
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
        self._ensure_brotli_enc()
        ref e = self._brotli_enc.value()
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
