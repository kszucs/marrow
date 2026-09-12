"""System-library loading for the block compression codecs.

The codecs are not reimplemented; the standard C libraries (`libzstd`,
`libsnappy`, `liblz4`, `libz`, `libbrotli`) are `dlopen`-ed at runtime and their
block APIs called directly — the same approach arrow-rs and duckdb take, just
without a link-time dependency. `CompressionLibs` is the primitive block calls
plus the per-call scratch they need; the handles themselves live in the
`Codecs` set below, one process-global for all six.

**Nothing here is Parquet-specific**, which is why it lives in `marrow.utils`
rather than in `marrow.parquet` where it started (as a second module named
`utils`). The format-specific half — the Parquet `CompressionCodec` codes, the
legacy Hadoop LZ4 frame tolerance, the scratch slack the bit-unpackers need — is
`Compression` in `marrow.parquet.codecs`, which dispatches onto this.

The other consumer is Arrow IPC, which currently *refuses* compressed bodies
(`ipc.mojo`, "reading compressed IPC bodies (LZ4_FRAME / ZSTD) is not
supported"). These bindings are what that needs.
"""

from .dylib import LibSet, LibSpec, c_bytes
from std.memory import unsafe_memset_zero
from std.memory.alloc import unsafe_alloc

comptime Codecs = LibSet[
    "MARROW_CODECS",
    [
        LibSpec(
            "zstd",
            ["libzstd.dylib", "libzstd.1.dylib", "libzstd.so", "libzstd.so.1"],
            [],
        ),
        LibSpec(
            "snappy", ["libsnappy.dylib", "libsnappy.so", "libsnappy.so.1"], []
        ),
        LibSpec("lz4", ["liblz4.dylib", "liblz4.so", "liblz4.so.1"], []),
        LibSpec(
            "z", ["libz.dylib", "libz.1.dylib", "libz.so", "libz.so.1"], []
        ),
        LibSpec(
            "brotlienc",
            ["libbrotlienc.dylib", "libbrotlienc.so", "libbrotlienc.so.1"],
            [],
        ),
        LibSpec(
            "brotlidec",
            ["libbrotlidec.dylib", "libbrotlidec.so", "libbrotlidec.so.1"],
            [],
        ),
    ],
]
"""Every codec library: what it is called, where to look for it, and one
process-global holding all six.

One global for the set, not one each -- six cost `query_cli` ~16 KB of
`__text` and the AOT lane is size-gated. A program that reads only
uncompressed Parquet opens nothing, because nothing touches this until a codec
method runs.
"""


struct CompressionLibs(Movable):
    """The primitive block calls each codec needs, plus the per-call scratch
    they write through. `Compression` dispatches into these; each fills exactly
    `out_size` bytes at `dst` (decompress) or returns the codec's output
    (compress).

    The `dlopen` handles are **not** here — they are the `Codecs` set above,
    shared by every instance. What an instance owns is the reused size
    out-param snappy needs, which is not safe to share, so a Parquet read
    still holds one of these per worker."""

    var _sz: List[UInt]  # reusable size out-param for snappy

    def __init__(out self):
        self._sz = [UInt(0)]

    @staticmethod
    def preload() raises:
        """Open every codec now, on the calling thread.

        `_Global` vends its pointer without locking and says nothing about
        racing *creation*, so the first touch must not be several workers at
        once. `ParquetFile.read` calls this before dispatching whenever the
        chunks it is about to decode are compressed; after it returns, every
        worker's `Codecs.handle[...]()` is a pure read.

        All six open together, because the caller knows only that *something*
        ahead is compressed, not which codec. A missing one is not an error
        here -- each `Dylib` records its own failure and re-raises it at the
        call that needs it.
        """
        Codecs.preload()

    # --- decompress: write exactly `out_size` bytes to `dst` ---

    def zstd_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: Pointer[UInt8, _],
        out_size: Int,
    ) raises:
        var n = Codecs.handle["zstd"]().call["ZSTD_decompress", Int](
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
        self._sz[0] = UInt(out_size)
        var status = Codecs.handle["snappy"]().call["snappy_uncompress", Int32](
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
        var n = Codecs.handle["lz4"]().call["LZ4_decompress_safe", Int32](
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
        var z = Codecs.handle["z"]()
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
        var sz = unsafe_alloc[UInt](1)
        sz[unsafe_offset=0] = UInt(out_size)
        # BrotliDecoderResult BrotliDecoderDecompress(size_t encoded_size,
        #   const uint8_t* encoded, size_t* decoded_size, uint8_t* decoded);
        # returns BROTLI_DECODER_RESULT_SUCCESS == 1.
        var rc = Codecs.handle["brotlidec"]().call[
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
        var out = c_bytes(dst, n)
        dst.unsafe_free()
        return out^

    def zstd_compress(mut self, src: Span[UInt8, _]) raises -> List[UInt8]:
        var z = Codecs.handle["zstd"]()
        var bound = z.call["ZSTD_compressBound", Int](len(src))
        var dst = unsafe_alloc[UInt8](bound)
        var n = z.call["ZSTD_compress", Int](
            dst, bound, src.unsafe_ptr(), len(src), Int32(1)
        )
        return Self._take(dst, n)

    def snappy_compress(mut self, src: Span[UInt8, _]) raises -> List[UInt8]:
        var s = Codecs.handle["snappy"]()
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
        var l = Codecs.handle["lz4"]()
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
        var z = Codecs.handle["z"]()
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
        var e = Codecs.handle["brotlienc"]()
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
